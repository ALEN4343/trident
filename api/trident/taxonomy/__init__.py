"""Loads classes.yaml and exposes it as typed objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

_TAXONOMY_PATH = Path(__file__).parent / "classes.yaml"


@dataclass(frozen=True)
class Group:
    id: str
    label: str
    colour: str


@dataclass(frozen=True)
class PollutionClass:
    id: str
    label: str
    group: str
    modality: str
    persistence_years: float
    windage: float
    entanglement_risk: float
    ingestion_risk: float
    toxicity: float
    recoverability: float
    auto_collect: bool
    hazard: float
    typical_mass_kg: float | None = None
    buoyancy: str | None = None
    recyclable: bool = False
    value_inr_per_kg: float = 0.0
    harm_per_pct_coverage: float | None = None

    @property
    def is_instance(self) -> bool:
        return self.modality == "instance"

    @property
    def is_region(self) -> bool:
        return self.modality == "region"


@dataclass(frozen=True)
class Confuser:
    id: str
    label: str
    group: str
    mimics: tuple[str, ...]
    discriminator: str


@dataclass(frozen=True)
class SeverityBand:
    max: float
    label: str
    colour: str


@dataclass(frozen=True)
class SeverityConfig:
    component_weights: dict[str, float]
    component_descriptions: dict[str, str]
    harm_dimensions: dict[str, float]
    density_saturation: float
    persistence_saturation: float
    bands: tuple[SeverityBand, ...]

    def band_for(self, score: float) -> SeverityBand:
        for band in self.bands:
            if score <= band.max:
                return band
        return self.bands[-1]


@dataclass(frozen=True)
class Taxonomy:
    version: int
    groups: dict[str, Group]
    classes: dict[str, PollutionClass]
    confusers: dict[str, Confuser]
    severity: SeverityConfig
    _by_mimic: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __getitem__(self, class_id: str) -> PollutionClass:
        try:
            return self.classes[class_id]
        except KeyError:
            raise KeyError(
                f"unknown pollution class {class_id!r}; "
                f"known: {sorted(self.classes)}"
            ) from None

    def instances(self) -> list[PollutionClass]:
        return [c for c in self.classes.values() if c.is_instance]

    def regions(self) -> list[PollutionClass]:
        return [c for c in self.classes.values() if c.is_region]

    def requires_human_signoff(self, class_id: str) -> bool:
        return not self[class_id].auto_collect

    def confusers_for(self, class_id: str) -> tuple[Confuser, ...]:
        """Confusers that can be mistaken for this class."""
        return tuple(self.confusers[cid] for cid in self._by_mimic.get(class_id, ()))


def _hazard(entry: dict, dims: dict[str, float]) -> float:
    total = sum(dims.values())
    return sum(entry[dim] * weight for dim, weight in dims.items()) / total


@lru_cache(maxsize=1)
def load_taxonomy(path: Path | None = None) -> Taxonomy:
    raw = yaml.safe_load((path or _TAXONOMY_PATH).read_text(encoding="utf-8"))

    sev_raw = raw["severity"]
    dims = sev_raw["harm_dimensions"]
    severity = SeverityConfig(
        component_weights={k: v["weight"] for k, v in sev_raw["components"].items()},
        component_descriptions={
            k: v["description"] for k, v in sev_raw["components"].items()
        },
        harm_dimensions=dims,
        density_saturation=sev_raw["density_saturation"],
        persistence_saturation=sev_raw["persistence_saturation"],
        bands=tuple(SeverityBand(**b) for b in sev_raw["bands"]),
    )

    classes = {}
    for entry in raw["classes"]:
        classes[entry["id"]] = PollutionClass(
            **entry, hazard=_hazard(entry, dims)
        )

    confusers = {}
    by_mimic: dict[str, list[str]] = {}
    for entry in raw["confusers"]:
        confusers[entry["id"]] = Confuser(
            id=entry["id"],
            label=entry["label"],
            group=entry["group"],
            mimics=tuple(entry["mimics"]),
            discriminator=entry["discriminator"],
        )
        for target in entry["mimics"]:
            by_mimic.setdefault(target, []).append(entry["id"])

    return Taxonomy(
        version=raw["version"],
        groups={k: Group(id=k, **v) for k, v in raw["groups"].items()},
        classes=classes,
        confusers=confusers,
        severity=severity,
        _by_mimic={k: tuple(v) for k, v in by_mimic.items()},
    )


__all__ = [
    "Confuser",
    "Group",
    "PollutionClass",
    "SeverityBand",
    "SeverityConfig",
    "Taxonomy",
    "load_taxonomy",
]
