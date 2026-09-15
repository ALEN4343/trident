"""Lagrangian particle advection for marine debris.

Integrates dx/dt = u_current + windage * u_wind with RK4 and a stochastic
walk for unresolved turbulence, following the approach OpenDrift implements at
research scale. Running the same integrator with a negative timestep turns a
forecast into a hindcast: forward tells a cleanup crew where to intercept the
debris, backward tells a regulator where it entered the water.

Windage is read from the taxonomy, so a styrofoam block and a waterlogged net
released from the same pixel separate over the forecast because they ride the
water differently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..taxonomy import load_taxonomy
from .forcing import Forcing, utc_now

EARTH_RADIUS_M = 6_378_137.0
_M_PER_DEG_LAT = math.pi * EARTH_RADIUS_M / 180.0


@dataclass(frozen=True)
class DriftConfig:
    horizon_hours: float = 6.0
    dt_seconds: float = 300.0
    particles: int = 300
    # Horizontal eddy diffusivity. 1-10 m^2/s is the usual coastal range; the
    # lower end here keeps the ensemble tight enough to be actionable.
    diffusivity_m2s: float = 2.0
    # Initial spread of the ensemble, normally the georeferencing uncertainty.
    release_sigma_m: float = 5.0
    seed: int = 0


@dataclass
class DriftResult:
    class_id: str
    windage: float
    origin: tuple[float, float]
    times: list[datetime]
    positions: np.ndarray  # (particles, steps, 2) as lat, lon
    reverse: bool
    synthetic: bool
    config: DriftConfig = field(default_factory=DriftConfig)

    @property
    def centroid_track(self) -> list[tuple[float, float]]:
        mean = self.positions.mean(axis=0)
        return [(float(lat), float(lon)) for lat, lon in mean]

    def spread_m(self, step: int = -1) -> float:
        """Radius containing 95% of the ensemble at a step, in metres."""
        cloud = self.positions[:, step, :]
        centre = cloud.mean(axis=0)
        dlat = (cloud[:, 0] - centre[0]) * _M_PER_DEG_LAT
        dlon = (
            (cloud[:, 1] - centre[1])
            * _M_PER_DEG_LAT
            * math.cos(math.radians(float(centre[0])))
        )
        return float(np.percentile(np.hypot(dlat, dlon), 95))

    def displacement_m(self) -> float:
        start = np.array(self.origin)
        end = self.positions[:, -1, :].mean(axis=0)
        dlat = (end[0] - start[0]) * _M_PER_DEG_LAT
        dlon = (
            (end[1] - start[1])
            * _M_PER_DEG_LAT
            * math.cos(math.radians(float(start[0])))
        )
        return float(math.hypot(dlat, dlon))

    def at_hour(self, hours: float) -> np.ndarray:
        """Ensemble cloud nearest the requested offset, as (particles, 2)."""
        step = min(
            len(self.times) - 1,
            max(0, round(hours * 3600.0 / self.config.dt_seconds)),
        )
        return self.positions[:, step, :]

    def to_geojson(self) -> dict:
        track = self.centroid_track
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in track],
                },
                "properties": {
                    "kind": "reverse_track" if self.reverse else "forecast_track",
                    "class_id": self.class_id,
                    "windage": self.windage,
                    "displacement_m": round(self.displacement_m(), 1),
                    "spread_m": round(self.spread_m(), 1),
                    "synthetic_forcing": self.synthetic,
                },
            }
        ]
        for hours in (1, 3, 6, 12, 24):
            if hours * 3600.0 > self.config.horizon_hours * 3600.0:
                break
            cloud = self.at_hour(hours)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiPoint",
                        "coordinates": [[float(lon), float(lat)] for lat, lon in cloud],
                    },
                    "properties": {"kind": "ensemble", "hours": hours},
                }
            )
        return {"type": "FeatureCollection", "features": features}


def _velocity_deg(
    forcing: Forcing, windage: float, lat: float
) -> tuple[float, float]:
    east, north = forcing.velocity(windage)
    dlat = north / _M_PER_DEG_LAT
    dlon = east / (_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return dlat, dlon


def advect(
    lat: float,
    lon: float,
    class_id: str,
    forcing_field,
    *,
    start: datetime | None = None,
    config: DriftConfig | None = None,
    reverse: bool = False,
) -> DriftResult:
    cfg = config or DriftConfig()
    taxonomy = load_taxonomy()
    windage = taxonomy[class_id].windage

    start = start or utc_now()
    steps = max(1, int(round(cfg.horizon_hours * 3600.0 / cfg.dt_seconds)))
    dt = cfg.dt_seconds * (-1.0 if reverse else 1.0)

    rng = np.random.default_rng(cfg.seed)
    cloud = np.empty((cfg.particles, 2))
    jitter = rng.normal(0.0, cfg.release_sigma_m, size=(cfg.particles, 2))
    cloud[:, 0] = lat + jitter[:, 0] / _M_PER_DEG_LAT
    cloud[:, 1] = lon + jitter[:, 1] / (
        _M_PER_DEG_LAT * math.cos(math.radians(lat))
    )

    positions = np.empty((cfg.particles, steps + 1, 2))
    positions[:, 0, :] = cloud
    times = [start]

    # Forcing is evaluated at the ensemble centroid rather than per particle.
    # The ensemble stays within a few hundred metres over these horizons while
    # a forcing cell is several kilometres across, so a per-particle lookup
    # returns the same value; the spread comes from the diffusion term below.
    sigma = math.sqrt(2.0 * cfg.diffusivity_m2s * abs(cfg.dt_seconds))

    now = start
    for step in range(steps):
        centre_lat, centre_lon = cloud.mean(axis=0)

        def derivative(offset_s: float, plat: float, plon: float):
            sample = forcing_field.at(
                plat, plon, now + timedelta(seconds=offset_s)
            )
            return _velocity_deg(sample, windage, plat)

        k1 = derivative(0.0, centre_lat, centre_lon)
        k2 = derivative(
            dt / 2, centre_lat + k1[0] * dt / 2, centre_lon + k1[1] * dt / 2
        )
        k3 = derivative(
            dt / 2, centre_lat + k2[0] * dt / 2, centre_lon + k2[1] * dt / 2
        )
        k4 = derivative(dt, centre_lat + k3[0] * dt, centre_lon + k3[1] * dt)

        dlat = (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6.0
        dlon = (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6.0

        walk = rng.normal(0.0, sigma, size=(cfg.particles, 2))
        cloud[:, 0] += dlat + walk[:, 0] / _M_PER_DEG_LAT
        cloud[:, 1] += dlon + walk[:, 1] / (
            _M_PER_DEG_LAT * math.cos(math.radians(centre_lat))
        )

        positions[:, step + 1, :] = cloud
        now = now + timedelta(seconds=dt)
        times.append(now)

    return DriftResult(
        class_id=class_id,
        windage=windage,
        origin=(lat, lon),
        times=times,
        positions=positions,
        reverse=reverse,
        synthetic=getattr(forcing_field, "is_synthetic", False),
        config=cfg,
    )


def forecast(lat: float, lon: float, class_id: str, forcing_field, **kwargs) -> DriftResult:
    """Where this item will be. Drives interception routing."""
    return advect(lat, lon, class_id, forcing_field, reverse=False, **kwargs)


def attribute_source(
    lat: float, lon: float, class_id: str, forcing_field, *, lookback_hours: float = 48.0, **kwargs
) -> DriftResult:
    """Where this item came from. Drives polluter attribution."""
    cfg = kwargs.pop("config", DriftConfig())
    cfg = DriftConfig(
        horizon_hours=lookback_hours,
        dt_seconds=cfg.dt_seconds,
        particles=cfg.particles,
        diffusivity_m2s=cfg.diffusivity_m2s,
        release_sigma_m=cfg.release_sigma_m,
        seed=cfg.seed,
    )
    return advect(lat, lon, class_id, forcing_field, reverse=True, config=cfg, **kwargs)
