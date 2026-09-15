"""Ocean current and wind forcing, fetched from Open-Meteo.

Open-Meteo serves both the marine and the atmospheric forecast without an API
key, which is the reason drift modelling is reachable for this project at all.

Direction conventions differ between the two feeds and mixing them up silently
reverses the forecast, so both are normalised to east/north vector components
here and nowhere else:

  * wind_direction_10m follows the meteorological convention and reports the
    direction the wind blows FROM.
  * ocean_current_direction follows the oceanographic convention and reports
    the direction the water flows TOWARD.
"""

from __future__ import annotations

import math
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import httpx
import numpy as np

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext | bool:
    """Verify against the OS trust store rather than certifi's bundle.

    Campus and corporate networks terminate TLS at an inspecting proxy whose
    CA is installed in the system store but absent from certifi, which fails
    verification for every outbound call. Reading the OS store keeps
    verification on instead of reaching for verify=False.
    """
    try:
        import truststore
    except ImportError:
        return True
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

# Forcing is fetched per cell rather than per particle; at this size the ocean
# state is effectively uniform across the cell for a coastal-scale forecast.
_CELL_DEG = 0.05


@dataclass(frozen=True)
class Forcing:
    """Depth-averaged surface forcing at a point, in m/s, east/north."""

    current_e: float
    current_n: float
    wind_e: float
    wind_n: float

    def velocity(self, windage: float) -> tuple[float, float]:
        """Total drift velocity for an object with the given windage."""
        return (
            self.current_e + windage * self.wind_e,
            self.current_n + windage * self.wind_n,
        )


def _from_to_components(speed: np.ndarray, bearing_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bearing the flow travels TOWARD -> east/north components."""
    theta = np.radians(bearing_deg)
    return speed * np.sin(theta), speed * np.cos(theta)


class ForcingField:
    """Hourly forcing time series, cached per cell and interpolated in time."""

    def __init__(self, *, timeout: float = 10.0, forecast_days: int = 3):
        self._timeout = timeout
        self._forecast_days = forecast_days
        self._cache: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return round(lat / _CELL_DEG), round(lon / _CELL_DEG)

    def _fetch(self, lat: float, lon: float) -> tuple[np.ndarray, ...]:
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "forecast_days": self._forecast_days,
            "past_days": 2,
            "timeformat": "unixtime",
        }
        with httpx.Client(timeout=self._timeout, verify=_ssl_context()) as client:
            marine = client.get(
                MARINE_URL,
                params={
                    **params,
                    "hourly": "ocean_current_velocity,ocean_current_direction",
                },
            )
            marine.raise_for_status()
            weather = client.get(
                WEATHER_URL,
                params={
                    **params,
                    "hourly": "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "ms",
                },
            )
            weather.raise_for_status()

        mh = marine.json()["hourly"]
        wh = weather.json()["hourly"]

        times = np.asarray(mh["time"], dtype=float)
        # The marine feed reports current velocity in km/h regardless of the
        # wind_speed_unit parameter, which only applies to the weather feed.
        speed = np.nan_to_num(np.asarray(mh["ocean_current_velocity"], dtype=float)) / 3.6
        bearing = np.nan_to_num(np.asarray(mh["ocean_current_direction"], dtype=float))
        cur_e, cur_n = _from_to_components(speed, bearing)

        wind_speed = np.nan_to_num(np.asarray(wh["wind_speed_10m"], dtype=float))
        wind_from = np.nan_to_num(np.asarray(wh["wind_direction_10m"], dtype=float))
        wind_e, wind_n = _from_to_components(wind_speed, wind_from + 180.0)

        n = min(len(times), len(wind_e))
        return times[:n], cur_e[:n], cur_n[:n], wind_e[:n], wind_n[:n]

    def at(self, lat: float, lon: float, when: datetime) -> Forcing:
        key = self._key(lat, lon)
        if key not in self._cache:
            self._cache[key] = self._fetch(lat, lon)
        times, cur_e, cur_n, wind_e, wind_n = self._cache[key]

        stamp = when.timestamp()
        return Forcing(
            current_e=float(np.interp(stamp, times, cur_e)),
            current_n=float(np.interp(stamp, times, cur_n)),
            wind_e=float(np.interp(stamp, times, wind_e)),
            wind_n=float(np.interp(stamp, times, wind_n)),
        )

    def prefetch(self, lat: float, lon: float) -> None:
        self._cache.setdefault(self._key(lat, lon), self._fetch(lat, lon))


class SyntheticForcing:
    """Deterministic stand-in used when the network is unavailable.

    A hackathon venue is the worst network environment a demo will ever face.
    This keeps the drift model runnable and reproducible offline; it is not a
    physical model and anything computed from it is labelled as synthetic in
    the report rather than passed off as a forecast.
    """

    is_synthetic = True

    def __init__(self, current_speed: float = 0.35, current_bearing: float = 115.0,
                 wind_speed: float = 5.0, wind_from: float = 225.0):
        self._cur = (
            current_speed * math.sin(math.radians(current_bearing)),
            current_speed * math.cos(math.radians(current_bearing)),
        )
        toward = math.radians(wind_from + 180.0)
        self._wind = (wind_speed * math.sin(toward), wind_speed * math.cos(toward))

    def at(self, lat: float, lon: float, when: datetime) -> Forcing:
        # A slow rotation so successive forecast hours differ, as a real tidal
        # signal would, without pretending to model an actual tide.
        phase = math.radians((when.timestamp() / 3600.0) * 12.0)
        swing = 1.0 + 0.25 * math.sin(phase)
        return Forcing(
            current_e=self._cur[0] * swing,
            current_n=self._cur[1] * swing,
            wind_e=self._wind[0],
            wind_n=self._wind[1],
        )

    def prefetch(self, lat: float, lon: float) -> None:
        return None


def resolve_forcing(lat: float, lon: float, *, allow_network: bool = True):
    """Live forcing when reachable, synthetic otherwise."""
    if allow_network:
        field = ForcingField()
        try:
            field.prefetch(lat, lon)
        except Exception:
            return SyntheticForcing()
        return field
    return SyntheticForcing()


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def hours_from(start: datetime, count: int, step_h: int = 1) -> list[datetime]:
    return [start + timedelta(hours=i * step_h) for i in range(count + 1)]
