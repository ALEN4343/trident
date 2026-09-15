"""End to end scene analysis.

Fuses the instance branch, the physics branch and the georectifier into one
result, then scores it. The ordering matters: candidate regions are named only
after competing hypotheses have been compared, so a rejected candidate is
reported as rejected rather than quietly dropped.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geo import CameraIntrinsics, CameraPose, PoseUncertainty, image_footprint, localise_pixel
from ..severity import DetectedItem, MPSI, RegionCoverage, Scene, compute as compute_mpsi
from ..taxonomy import load_taxonomy
from . import indices
from .detector import DetectorOutput, detect

# A candidate below this is indistinguishable from clean water and is not
# reported at all, as either a detection or a rejection.
_MIN_HYPOTHESIS_SCORE = 0.30

_CONFUSER_IDS = frozenset({"sea_foam", "sun_glint", "shallow_seabed"})


@dataclass(frozen=True)
class ItemDetection:
    id: str
    class_id: str
    label: str
    group: str
    confidence: float
    bbox: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    area_px: float
    hazard: float
    mass_kg: float
    auto_collect: bool
    lat: float | None = None
    lon: float | None = None
    uncertainty_m: float | None = None
    size_m: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_id": self.class_id,
            "label": self.label,
            "group": self.group,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox],
            "centroid_px": [round(v, 1) for v in self.centroid_px],
            "hazard": round(self.hazard, 3),
            "mass_kg": self.mass_kg,
            "auto_collect": self.auto_collect,
            "lat": self.lat,
            "lon": self.lon,
            "uncertainty_m": round(self.uncertainty_m, 1) if self.uncertainty_m else None,
            "size_m": round(self.size_m, 3) if self.size_m else None,
        }


@dataclass(frozen=True)
class RegionDetection:
    id: str
    class_id: str
    label: str
    confidence: float
    percent_of_water: float
    polygon_px: list[tuple[int, int]]
    evidence: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "percent_of_water": round(self.percent_of_water, 2),
            "polygon_px": self.polygon_px,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Rejection:
    """A candidate that was considered and thrown out.

    Reported rather than hidden: on open water, natural features are mistaken
    for pollution more often than pollution is missed, so a severity score is
    only credible alongside what it declined to count.
    """

    id: str
    rejected_as: str
    label: str
    would_have_been: str
    margin: float
    discriminator: str
    polygon_px: list[tuple[int, int]]
    evidence: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rejected_as": self.rejected_as,
            "label": self.label,
            "would_have_been": self.would_have_been,
            "margin": round(self.margin, 3),
            "discriminator": self.discriminator,
            "polygon_px": self.polygon_px,
            "evidence": self.evidence,
        }


@dataclass
class Analysis:
    width: int
    height: int
    items: list[ItemDetection]
    regions: list[RegionDetection]
    rejections: list[Rejection]
    mpsi: MPSI
    water_fraction: float
    water_area_m2: float | None
    footprint: list[tuple[float, float]] | None
    detector: DetectorOutput
    wildlife_present: tuple[str, ...] = ()
    vessels_in_frame: int = 0
    people_in_frame: int = 0
    georeferenced: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": {"width": self.width, "height": self.height},
            "georeferenced": self.georeferenced,
            "water_fraction": round(self.water_fraction, 3),
            "water_area_m2": round(self.water_area_m2, 1) if self.water_area_m2 else None,
            "footprint": self.footprint,
            "items": [d.to_dict() for d in self.items],
            "regions": [r.to_dict() for r in self.regions],
            "rejections": [r.to_dict() for r in self.rejections],
            "severity": {
                "score": self.mpsi.score,
                "band": self.mpsi.band.label,
                "colour": self.mpsi.band.colour,
                "lower": self.mpsi.lower,
                "upper": self.mpsi.upper,
                "components": [
                    {
                        "key": c.key,
                        "value": round(c.value, 3),
                        "weight": c.weight,
                        "contribution": round(c.contribution, 2),
                        "description": c.description,
                    }
                    for c in self.mpsi.components
                ],
                "item_count": self.mpsi.item_count,
                "total_mass_kg": self.mpsi.total_mass_kg,
                "recoverable_value_inr": self.mpsi.recoverable_value_inr,
                "dominant_hazard": self.mpsi.dominant_hazard,
                "notes": list(self.mpsi.notes),
            },
            "safety": {
                "wildlife": list(self.wildlife_present),
                "vessels": self.vessels_in_frame,
                "people": self.people_in_frame,
            },
            "detector": {
                "model": self.detector.model_name,
                "available": self.detector.available,
                "note": self.detector.note,
            },
            "notes": self.notes,
        }


def _polygon(mask: np.ndarray, max_points: int = 40) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    epsilon = 0.008 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True).reshape(-1, 2)
    if len(approx) > max_points:
        step = len(approx) // max_points + 1
        approx = approx[::step]
    return [(int(x), int(y)) for x, y in approx]


def _floating(
    water: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    margin: float = 0.7,
    threshold: float = 0.35,
) -> bool:
    """Is this object surrounded by water?

    Tests the ring around the object rather than the object itself. A floating
    bottle is not water-coloured -- that is precisely why the detector found
    it -- so sampling its own pixels would reject every genuine detection. What
    separates litter in the sea from litter on the beach is what lies around
    it.
    """
    height, width = water.shape
    x1, y1, x2, y2 = bbox
    pad_x = max(4.0, (x2 - x1) * margin)
    pad_y = max(4.0, (y2 - y1) * margin)

    ox1, oy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
    ox2, oy2 = int(min(width, x2 + pad_x)), int(min(height, y2 + pad_y))
    ix1, iy1 = int(max(0, x1)), int(max(0, y1))
    ix2, iy2 = int(min(width, x2)), int(min(height, y2))
    if ox2 <= ox1 or oy2 <= oy1:
        return False

    outer = water[oy1:oy2, ox1:ox2]
    ring_total = outer.size - max(0, (ix2 - ix1) * (iy2 - iy1))
    if ring_total <= 0:
        return bool(outer.mean() >= threshold)

    ring_water = int(outer.sum()) - int(water[iy1:iy2, ix1:ix2].sum())
    return ring_water / ring_total >= threshold


def _classify_candidates(
    stack: indices.IndexStack, water_px: int
) -> tuple[list[RegionDetection], list[Rejection]]:
    taxonomy = load_taxonomy()
    accepted: list[RegionDetection] = []
    rejected: list[Rejection] = []

    for mask in indices.candidate_masks(stack):
        scores = indices.score_region(stack, mask)
        if not scores:
            continue

        pollution = {k: v for k, v in scores.items() if k not in _CONFUSER_IDS}
        confusers = {k: v for k, v in scores.items() if k in _CONFUSER_IDS}

        best_pollution = max(pollution, key=pollution.get)
        best_confuser = max(confusers, key=confusers.get)
        if max(scores.values()) < _MIN_HYPOTHESIS_SCORE:
            continue

        polygon = _polygon(mask)
        evidence = indices.region_evidence(stack, mask)

        if confusers[best_confuser] >= pollution[best_pollution]:
            confuser = taxonomy.confusers[best_confuser]
            rejected.append(
                Rejection(
                    id=uuid.uuid4().hex[:8],
                    rejected_as=best_confuser,
                    label=confuser.label,
                    would_have_been=taxonomy[best_pollution].label,
                    margin=confusers[best_confuser] - pollution[best_pollution],
                    discriminator=confuser.discriminator,
                    polygon_px=polygon,
                    evidence=evidence,
                )
            )
            continue

        percent = 100.0 * float(mask.sum()) / max(water_px, 1)
        accepted.append(
            RegionDetection(
                id=uuid.uuid4().hex[:8],
                class_id=best_pollution,
                label=taxonomy[best_pollution].label,
                confidence=pollution[best_pollution],
                percent_of_water=min(percent, 100.0),
                polygon_px=polygon,
                evidence=evidence,
            )
        )

    return accepted, rejected


def analyse(
    rgb: np.ndarray,
    *,
    pose: CameraPose | None = None,
    intrinsics: CameraIntrinsics | None = None,
    uncertainty: PoseUncertainty | None = None,
    ecological_sensitivity: float = 0.0,
    weights: str | None = None,
) -> Analysis:
    height, width = rgb.shape[:2]
    notes: list[str] = []

    stack = indices.compute(rgb)
    water = stack.water > 0.5
    water_px = int(water.sum())
    water_fraction = water_px / float(water.size)
    if water_fraction < 0.25:
        notes.append(
            "Less than a quarter of the frame reads as water; "
            "density and coverage are unreliable."
        )

    output = detect(rgb, **({"weights": weights} if weights else {}))
    if not output.available:
        notes.append(
            "Instance detector unavailable, so only amorphous contamination "
            "was assessed. Object counts are not a measurement."
        )

    regions, rejections = _classify_candidates(stack, water_px)

    footprint = None
    water_area_m2 = None
    if pose is not None and intrinsics is not None:
        fp = image_footprint(pose, intrinsics)
        if fp is not None:
            footprint = fp.polygon
            water_area_m2 = fp.area_m2 * water_fraction
            if fp.horizon_clipped:
                notes.append(
                    "Frame includes the horizon; footprint was clipped and the "
                    "imaged area is a lower bound."
                )

    taxonomy = load_taxonomy()
    items: list[ItemDetection] = []
    for raw in output.detections:
        x1, y1, x2, y2 = raw.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if not _floating(water, raw.bbox):
            continue

        cls = taxonomy[raw.class_id]
        area_px = float(raw.mask.sum()) if raw.mask is not None else (x2 - x1) * (y2 - y1)

        lat = lon = unc = size_m = None
        if pose is not None and intrinsics is not None:
            fix = localise_pixel(cx, cy, pose, intrinsics, uncertainty=uncertainty, samples=96)
            if fix is not None:
                lat, lon, unc = fix.lat, fix.lon, fix.uncertainty_radius_m
                if np.isfinite(fix.gsd_m):
                    size_m = max(x2 - x1, y2 - y1) * fix.gsd_m

        items.append(
            ItemDetection(
                id=uuid.uuid4().hex[:8],
                class_id=cls.id,
                label=cls.label,
                group=cls.group,
                confidence=raw.confidence,
                bbox=(x1, y1, x2, y2),
                centroid_px=(cx, cy),
                area_px=area_px,
                hazard=cls.hazard,
                mass_kg=cls.typical_mass_kg or 0.0,
                auto_collect=cls.auto_collect,
                lat=lat,
                lon=lon,
                uncertainty_m=unc,
                size_m=size_m,
            )
        )

    scene = Scene(
        items=tuple(DetectedItem(d.class_id, d.confidence) for d in items),
        regions=tuple(
            RegionCoverage(r.class_id, r.percent_of_water, r.confidence) for r in regions
        ),
        water_area_m2=water_area_m2 or 0.0,
        ecological_sensitivity=ecological_sensitivity,
    )

    if output.wildlife:
        notes.append(
            f"Wildlife in frame ({', '.join(output.wildlife)}); "
            "autonomous collection is inhibited for this scene."
        )

    return Analysis(
        width=width,
        height=height,
        items=items,
        regions=regions,
        rejections=rejections,
        mpsi=compute_mpsi(scene),
        water_fraction=water_fraction,
        water_area_m2=water_area_m2,
        footprint=footprint,
        detector=output,
        wildlife_present=output.wildlife,
        vessels_in_frame=output.vessels,
        people_in_frame=output.people,
        georeferenced=water_area_m2 is not None,
        notes=notes,
    )
