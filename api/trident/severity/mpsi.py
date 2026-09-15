"""Marine Pollution Severity Index.

A 0-100 score built from five components that an operator can inspect and
argue with, rather than a threshold on a raw object count. Every weight and
saturation point lives in the taxonomy, so the number moves when the physics
or the policy moves, not when someone edits a magic constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..taxonomy import SeverityBand, load_taxonomy

# Keeps a single extreme item from being averaged away by a crowd of harmless
# ones: a ghost net in a field of paper cups still drives the score.
_PEAK_HAZARD_SHARE = 0.35


@dataclass(frozen=True)
class DetectedItem:
    class_id: str
    confidence: float = 1.0
    count: int = 1


@dataclass(frozen=True)
class RegionCoverage:
    class_id: str
    percent_of_water: float
    confidence: float = 1.0


@dataclass(frozen=True)
class Scene:
    items: tuple[DetectedItem, ...] = ()
    regions: tuple[RegionCoverage, ...] = ()
    water_area_m2: float = 0.0
    # 0 = open water far from sensitive habitat, 1 = inside a protected area,
    # reef, mangrove or active fishery.
    ecological_sensitivity: float = 0.0


@dataclass(frozen=True)
class Component:
    key: str
    value: float
    weight: float
    description: str

    @property
    def contribution(self) -> float:
        return self.value * self.weight * 100.0


@dataclass(frozen=True)
class MPSI:
    score: float
    band: SeverityBand
    components: tuple[Component, ...]
    lower: float
    upper: float
    item_count: int
    total_mass_kg: float
    recoverable_value_inr: float
    dominant_hazard: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def breakdown(self) -> dict[str, float]:
        return {c.key: round(c.contribution, 2) for c in self.components}


def _mass_weighted(pairs: list[tuple[float, float]]) -> float:
    """pairs of (weight, value) -> weighted mean, 0 when empty."""
    total = sum(w for w, _ in pairs)
    if total <= 0:
        return 0.0
    return sum(w * v for w, v in pairs) / total


def _evaluate(scene: Scene, min_confidence: float) -> tuple[float, dict]:
    taxonomy = load_taxonomy()
    cfg = taxonomy.severity

    items = [i for i in scene.items if i.confidence >= min_confidence]
    regions = [r for r in scene.regions if r.confidence >= min_confidence]

    mass_pairs: list[tuple[float, float]] = []
    persistence_pairs: list[tuple[float, float]] = []
    total_mass = 0.0
    value_inr = 0.0
    peak_hazard = 0.0
    dominant: str | None = None
    count = 0

    for item in items:
        cls = taxonomy[item.class_id]
        mass = (cls.typical_mass_kg or 0.0) * item.count
        total_mass += mass
        count += item.count
        value_inr += mass * cls.value_inr_per_kg if cls.recyclable else 0.0
        mass_pairs.append((max(mass, 1e-6), cls.hazard))
        persistence_pairs.append((max(mass, 1e-6), cls.persistence_years))
        if cls.hazard > peak_hazard:
            peak_hazard, dominant = cls.hazard, cls.id

    density = 0.0
    if scene.water_area_m2 > 0 and count:
        per_100 = count / (scene.water_area_m2 / 100.0)
        density = min(1.0, per_100 / cfg.density_saturation)

    hazard_load = 0.0
    if mass_pairs:
        hazard_load = min(
            1.0,
            (1.0 - _PEAK_HAZARD_SHARE) * _mass_weighted(mass_pairs)
            + _PEAK_HAZARD_SHARE * peak_hazard,
        )

    coverage_harm = 0.0
    for region in regions:
        cls = taxonomy[region.class_id]
        coverage_harm += region.percent_of_water * (cls.harm_per_pct_coverage or 0.0)
        if cls.hazard > peak_hazard:
            peak_hazard, dominant = cls.hazard, cls.id
    coverage = min(1.0, coverage_harm / 100.0)

    persistence = 0.0
    if persistence_pairs:
        persistence = min(
            1.0, _mass_weighted(persistence_pairs) / cfg.persistence_saturation
        )

    values = {
        "density": density,
        "hazard_load": hazard_load,
        "coverage": coverage,
        "persistence": persistence,
        "ecological": min(1.0, max(0.0, scene.ecological_sensitivity)),
    }
    score = sum(values[k] * w for k, w in cfg.component_weights.items()) * 100.0

    return score, {
        "values": values,
        "count": count,
        "total_mass": total_mass,
        "value_inr": value_inr,
        "dominant": dominant,
    }


def compute(scene: Scene, *, confidence_floor: float = 0.25) -> MPSI:
    """Score a scene, with a confidence band rather than a single number.

    The band is the score computed over every detection above the floor versus
    only the confident ones. A wide band is a signal that the imagery, not the
    water, is the problem.
    """
    taxonomy = load_taxonomy()
    cfg = taxonomy.severity

    score, detail = _evaluate(scene, confidence_floor)
    strict, _ = _evaluate(scene, 0.6)

    notes: list[str] = []
    if scene.water_area_m2 <= 0:
        notes.append(
            "No georeferenced footprint: density excluded, score is a lower bound."
        )
    if not scene.items and not scene.regions:
        notes.append("No pollution detected above the confidence floor.")

    components = tuple(
        Component(
            key=key,
            value=detail["values"][key],
            weight=weight,
            description=cfg.component_descriptions[key],
        )
        for key, weight in cfg.component_weights.items()
    )

    return MPSI(
        score=round(score, 1),
        band=cfg.band_for(score),
        components=components,
        lower=round(min(score, strict), 1),
        upper=round(max(score, strict), 1),
        item_count=detail["count"],
        total_mass_kg=round(detail["total_mass"], 3),
        recoverable_value_inr=round(detail["value_inr"], 2),
        dominant_hazard=detail["dominant"],
        notes=tuple(notes),
    )
