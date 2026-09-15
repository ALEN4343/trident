"""Inference wrapper around the trained oil segmenter.

Tiles an arbitrary-sized scene into the 256 px chips the model was trained on,
runs them in batches, and averages the overlaps back together. Overlap matters:
a slick crossing a tile seam gets predicted twice from different contexts, and
averaging removes the seam that hard tiling would leave through the middle of a
detection.

Every result carries whether its checkpoint came from a real training run. A
smoke-test checkpoint still produces a confident-looking probability map, so
that flag travels with the numbers all the way to the UI rather than being
checked once and forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

TILE_PX = 256
TILE_OVERLAP_PX = 32

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_CHECKPOINT = _WEIGHTS_DIR / "oil_seg.pt"


@dataclass(frozen=True)
class SARResult:
    prob: np.ndarray
    mask: np.ndarray
    oil_fraction: float
    threshold: float
    representative: bool
    epoch: int | None
    oil_iou: float | None
    checkpoint: str

    @property
    def caveat(self) -> str | None:
        if self.representative:
            return None
        return (
            "This checkpoint is from a reduced or CPU smoke-test run. The mask "
            "shows that the pipeline executes; it is not a measurement."
        )


def checkpoint_status(path: Path | str = DEFAULT_CHECKPOINT) -> dict:
    """What weights are available, without loading the network."""
    p = Path(path)
    if not p.exists():
        return {"available": False, "path": str(p), "reason": "no checkpoint on disk"}
    try:
        import torch
    except ImportError:
        return {"available": False, "path": str(p), "reason": "torch not installed"}

    try:
        blob = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(p), "reason": f"unreadable: {exc}"}

    meta = {k: v for k, v in blob.items() if k != "model"} if isinstance(blob, dict) else {}
    return {
        "available": True,
        "path": str(p),
        "representative": bool(meta.get("representative", False)),
        "epoch": meta.get("epoch"),
        "oil_iou": meta.get("oil_iou"),
        "input_size": meta.get("input_size", TILE_PX),
        "size_mb": round(p.stat().st_size / 1e6, 1),
    }


def _tiles(shape: tuple[int, int], tile: int, overlap: int):
    h, w = shape
    step = max(1, tile - overlap)
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            yield max(0, y), min(h, y + tile), max(0, x), min(w, x + tile)


def _to_grey(image: np.ndarray) -> np.ndarray:
    # Recorded before the channel mean, which promotes uint8 to float and would
    # otherwise send an already-byte-range image down the rescaling path. The
    # training chips were 8-bit PNGs used as-is, so stretching them here would
    # feed the model a contrast it never saw.
    was_byte = image.dtype == np.uint8

    if image.ndim == 3:
        # SOS stores greyscale replicated across RGB, so any single channel is
        # the signal; the mean is equivalent and tolerates a genuine colour
        # render of the same scene.
        image = image.mean(axis=2)
    image = image.astype(np.float32)

    if was_byte:
        return image

    # Float input is dB backscatter or reflectance, on no fixed scale.
    lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi - lo < 1e-6:
        # A flat scene carries no signal. Mid-grey rather than zero, which the
        # model would read as total radar shadow and call oil everywhere.
        return np.full_like(image, 127.5)
    return (image - lo) / (hi - lo) * 255.0


def _prepare(chip: np.ndarray) -> np.ndarray:
    x = chip / 255.0
    x = np.stack([x, x, x], axis=-1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1).astype(np.float32)


class SARSegmenter:
    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT, device: str | None = None):
        import torch

        from .model import DeepLabV3Plus

        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Train one with scripts/train_oil.py, or leave "
                f"the SAR branch disabled -- the optical branch does not need it."
            )

        # weights_only: a checkpoint may arrive from Colab or a shared drive, and
        # torch.load otherwise executes whatever was pickled into it.
        blob = torch.load(path, map_location="cpu", weights_only=True)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DeepLabV3Plus(num_classes=2, pretrained=False)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.meta = {k: v for k, v in blob.items() if k != "model"} if isinstance(blob, dict) else {}
        self.checkpoint = str(path)

    def predict(
        self, image: np.ndarray, *, threshold: float = 0.5, batch: int = 8
    ) -> SARResult:
        import torch

        grey = _to_grey(image)
        h, w = grey.shape

        # Pad rather than resize: resizing changes the texture scale the model
        # keys on, and SAR speckle statistics do not survive interpolation.
        pad_h, pad_w = max(0, TILE_PX - h), max(0, TILE_PX - w)
        if pad_h or pad_w:
            grey = np.pad(grey, ((0, pad_h), (0, pad_w)), mode="reflect")

        windows = list(_tiles(grey.shape, TILE_PX, TILE_OVERLAP_PX))
        total = np.zeros(grey.shape, np.float32)
        count = np.zeros(grey.shape, np.float32)

        with torch.no_grad():
            for i in range(0, len(windows), batch):
                chunk = windows[i : i + batch]
                arr = np.stack([_prepare(grey[y0:y1, x0:x1]) for y0, y1, x0, x1 in chunk])
                logits = self.model(torch.from_numpy(arr).to(self.device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for (y0, y1, x0, x1), p in zip(chunk, probs):
                    total[y0:y1, x0:x1] += p[: y1 - y0, : x1 - x0]
                    count[y0:y1, x0:x1] += 1.0

        prob = (total / np.maximum(count, 1e-6))[:h, :w]
        mask = prob >= threshold

        return SARResult(
            prob=prob,
            mask=mask,
            oil_fraction=float(mask.mean()),
            threshold=threshold,
            representative=bool(self.meta.get("representative", False)),
            epoch=self.meta.get("epoch"),
            oil_iou=self.meta.get("oil_iou"),
            checkpoint=self.checkpoint,
        )


def polygons(
    mask: np.ndarray, *, min_area_px: int = 96, max_shapes: int = 24, max_points: int = 48
) -> list[list[tuple[int, int]]]:
    """Outlines of every separate slick in the mask.

    Returns many shapes rather than one: a spill commonly breaks into a main
    body with detached streamers, and merging them into a single hull would
    overstate the affected area by the clean water between them.
    """
    import cv2

    found, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shapes = []
    for contour in sorted(found, key=cv2.contourArea, reverse=True)[:max_shapes]:
        if cv2.contourArea(contour) < min_area_px:
            continue
        epsilon = 0.006 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        if len(approx) > max_points:
            approx = approx[:: len(approx) // max_points + 1]
        shapes.append([(int(x), int(y)) for x, y in approx])
    return shapes


@lru_cache(maxsize=1)
def _cached(checkpoint: str) -> SARSegmenter:
    return SARSegmenter(checkpoint)


def segment_array(
    image: np.ndarray,
    *,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    threshold: float = 0.5,
) -> SARResult:
    return _cached(str(checkpoint)).predict(image, threshold=threshold)
