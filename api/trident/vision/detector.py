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

# Prefer the checked-out copy so a demo never depends on reaching a CDN.
DEFAULT_WEIGHTS = str(
    _WEIGHTS_DIR / _DEFAULT_NAME
    if (_WEIGHTS_DIR / _DEFAULT_NAME).exists()
    else _DEFAULT_NAME
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


def detect(
    rgb: np.ndarray,
    *,
    weights: str = DEFAULT_WEIGHTS,
    confidence: float = 0.25,
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
