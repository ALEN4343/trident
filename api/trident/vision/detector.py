"""Instance detection branch.

Runs a YOLO segmentation model and maps its vocabulary onto the TRIDENT
taxonomy. The shipped default is a COCO-pretrained checkpoint, which already
covers the highest-frequency floating litter classes -- bottles, cups, bowls
and glassware -- and needs no training run to be useful.

COCO also gives two things the mission planner needs for free: an animal class
that arms the wildlife interlock, and a boat class that marks a vessel in
frame. Both are safety inputs, not pollution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
_DEFAULT_NAME = "yolo11n-seg.pt"
_TRASH_NAME = "trash_yolo11n.pt"

# Prefer the checked-out copy so a demo never depends on reaching a CDN.
DEFAULT_WEIGHTS = str(
    _WEIGHTS_DIR / _DEFAULT_NAME
    if (_WEIGHTS_DIR / _DEFAULT_NAME).exists()
    else _DEFAULT_NAME
)
TRASH_WEIGHTS = str(
    _WEIGHTS_DIR / _TRASH_NAME
    if (_WEIGHTS_DIR / _TRASH_NAME).exists()
    else _TRASH_NAME
)

# COCO label -> taxonomy class. Anything absent is ignored rather than guessed.
COCO_TO_TAXONOMY: dict[str, str] = {
    "bottle": "plastic_bottle",
    "cup": "food_container",
    "bowl": "food_container",
    "wine glass": "glass_bottle",
    "vase": "glass_bottle",
    "book": "paper_cardboard",
    "handbag": "textile_cloth",
    "backpack": "textile_cloth",
    "suitcase": "textile_cloth",
    "tie": "textile_cloth",
    "umbrella": "textile_cloth",
    "frisbee": "plastic_fragment",
    "sports ball": "plastic_fragment",
    "kite": "plastic_film_wrapper",
}

TRASH_TO_TAXONOMY: dict[str, str] = {
    "plastic": "plastic_bottle",
    "paper": "paper_cardboard",
    "glass": "glass_bottle",
    "trash": "plastic_fragment",
}

# Not pollution. These arm the autonomy interlocks instead.
WILDLIFE_LABELS = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
)
VESSEL_LABELS = frozenset({"boat"})
PERSON_LABELS = frozenset({"person"})


@dataclass(frozen=True)
class RawDetection:
    class_id: str
    source_label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None


@dataclass(frozen=True)
class DetectorOutput:
    detections: tuple[RawDetection, ...]
    wildlife: tuple[str, ...]
    vessels: int
    people: int
    model_name: str
    available: bool
    note: str | None = None


@lru_cache(maxsize=2)
def _load(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def _box_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def detect(
    rgb: np.ndarray,
    *,
    weights: str = DEFAULT_WEIGHTS,
    confidence: float = 0.20,
    imgsz: int = 640,
) -> DetectorOutput:
    """Run instance segmentation. Never raises -- a missing model degrades to
    the physics branch alone rather than taking the whole analysis down."""
    try:
        model = _load(weights)
    except Exception as exc:  # noqa: BLE001 - any failure here is non-fatal
        log.warning("detector unavailable (%s); physics branch only", exc)
        return DetectorOutput((), (), 0, 0, weights, False, str(exc))

    result = model.predict(rgb, conf=confidence, imgsz=imgsz, verbose=False)[0]
    names = result.names

    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    detections: list[RawDetection] = []
    wildlife: list[str] = []
    vessels = people = 0

    for i, box in enumerate(result.boxes):
        label = names[int(box.cls)]
        conf = float(box.conf)
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())

        if label in WILDLIFE_LABELS:
            wildlife.append(label)
            continue
        if label in VESSEL_LABELS:
            vessels += 1
            continue
        if label in PERSON_LABELS:
            people += 1
            continue

        class_id = COCO_TO_TAXONOMY.get(label)
        if class_id is None:
            continue

        mask = None
        if masks is not None and i < len(masks):
            mask = _resize_mask(masks[i], rgb.shape[:2])

        detections.append(RawDetection(class_id, label, conf, xyxy, mask))

    # Second pass: specialized marine litter / trash model
    if Path(TRASH_WEIGHTS).exists():
        try:
            trash_model = _load(TRASH_WEIGHTS)
            trash_result = trash_model.predict(rgb, conf=confidence, imgsz=imgsz, verbose=False)[0]
            for box in trash_result.boxes:
                t_label = trash_result.names[int(box.cls)]
                t_conf = float(box.conf)
                t_xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                t_class = TRASH_TO_TAXONOMY.get(t_label)
                if not t_class:
                    continue
                # Avoid duplicate overlapping detections
                if any(_box_iou(t_xyxy, d.bbox) > 0.4 for d in detections):
                    continue
                detections.append(RawDetection(t_class, t_label, t_conf, t_xyxy, None))
        except Exception as exc:  # noqa: BLE001
            log.warning("trash detector failed (%s)", exc)

    return DetectorOutput(
        detections=tuple(detections),
        wildlife=tuple(sorted(set(wildlife))),
        vessels=vessels,
        people=people,
        model_name=weights,
        available=True,
    )


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.float32), (shape[1], shape[0]))
    return mask > 0.5
