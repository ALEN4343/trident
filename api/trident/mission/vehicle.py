"""Vehicle abstraction and a simulator that satisfies it.

The mission executor talks only to VehicleAdapter. The simulator below and a
MAVLink-backed implementation are interchangeable behind it, which is what
makes this a mission planner that currently drives a simulation rather than a
simulation pretending to be a mission planner.

MAVLinkVehicle is deliberately left as an unimplemented stub rather than a
half-working one: the seam is real, the hardware is not connected, and
claiming otherwise in a demo would be dishonest.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Protocol

from .planner import Mission, VehicleSpec, haversine


@dataclass
class Telemetry:
    t: float
    lat: float
    lon: float
    heading_deg: float
    speed_ms: float
    battery_pct: float
    payload_kg: float
    state: str
    seq: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lat"] = round(self.lat, 6)
        d["lon"] = round(self.lon, 6)
        d["heading_deg"] = round(self.heading_deg, 1)
        d["battery_pct"] = round(self.battery_pct, 1)
        d["payload_kg"] = round(self.payload_kg, 2)
        d["t"] = round(self.t, 1)
        return d


class VehicleAdapter(Protocol):
    spec: VehicleSpec

    async def arm(self) -> None: ...

    async def goto(self, lat: float, lon: float) -> AsyncIterator[Telemetry]: ...

    async def collect(self, mass_kg: float, seconds: float) -> AsyncIterator[Telemetry]: ...

    async def return_to_base(self) -> AsyncIterator[Telemetry]: ...


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


class SimulatedVehicle:
    """Kinematic surface vessel.

    Integrates position along each leg at the configured cruise speed and
    draws down a battery that depends on payload, so the numbers on screen
    move for reasons rather than on a timer.
    """

    def __init__(
        self,
        base: tuple[float, float],
        spec: VehicleSpec | None = None,
        *,
        time_scale: float = 60.0,
        step_s: float = 5.0,
    ):
        self.spec = spec or VehicleSpec()
        self.base = base
        self.lat, self.lon = base
        self.heading = 0.0
        self.payload_kg = 0.0
        self.battery_pct = 100.0
        self.clock = 0.0
        self.state = "idle"
        self.seq = 0
        # Simulated seconds advanced per wall-clock second, so a 40 minute
        # mission plays back in about 40 seconds.
        self.time_scale = time_scale
        self.step_s = step_s

    def _telemetry(self, message: str = "") -> Telemetry:
        return Telemetry(
            t=self.clock,
            lat=self.lat,
            lon=self.lon,
            heading_deg=self.heading,
            speed_ms=self.spec.cruise_speed_ms if self.state == "transit" else 0.0,
            battery_pct=self.battery_pct,
            payload_kg=self.payload_kg,
            state=self.state,
            seq=self.seq,
            message=message,
        )

    def _drain(self, seconds: float, *, working: bool = False) -> None:
        load_factor = 1.0 + 0.4 * (self.payload_kg / self.spec.payload_capacity_kg)
        rate = 100.0 / self.spec.endurance_s * load_factor
        self.battery_pct = max(0.0, self.battery_pct - rate * seconds * (1.3 if working else 1.0))

    async def arm(self) -> None:
        self.state = "armed"
        await asyncio.sleep(0.2)

    async def goto(self, lat: float, lon: float) -> AsyncIterator[Telemetry]:
        self.state = "transit"
        total = haversine(self.lat, self.lon, lat, lon)
        if total < 0.5:
            yield self._telemetry("already on station")
            return

        self.heading = _bearing(self.lat, self.lon, lat, lon)
        start_lat, start_lon = self.lat, self.lon
        duration = total / self.spec.cruise_speed_ms
        travelled = 0.0

        while travelled < duration:
            travelled = min(duration, travelled + self.step_s)
            f = travelled / duration
            self.lat = start_lat + (lat - start_lat) * f
            self.lon = start_lon + (lon - start_lon) * f
            self.clock += self.step_s
            self._drain(self.step_s)
            yield self._telemetry()
            await asyncio.sleep(self.step_s / self.time_scale)

        self.lat, self.lon = lat, lon
        self.state = "on_station"
        yield self._telemetry("arrived")

    async def collect(self, mass_kg: float, seconds: float) -> AsyncIterator[Telemetry]:
        self.state = "collecting"
        elapsed = 0.0
        while elapsed < seconds:
            elapsed = min(seconds, elapsed + self.step_s)
            self.clock += self.step_s
            self._drain(self.step_s, working=True)
            yield self._telemetry()
            await asyncio.sleep(self.step_s / self.time_scale)

        self.payload_kg += mass_kg
        yield self._telemetry(f"collected {mass_kg:.2f} kg")

    async def return_to_base(self) -> AsyncIterator[Telemetry]:
        async for t in self.goto(*self.base):
            yield t
        self.state = "docked"
        yield self._telemetry("docked")


class MAVLinkVehicle:
    """Seam for a real airframe or hull speaking MAVLink.

    Implementing this against pymavlink and an ArduPilot target is the only
    change required to fly the same mission on hardware: the planner, the
    approval gates and the executor are all vehicle agnostic.
    """

    def __init__(self, connection_string: str, spec: VehicleSpec | None = None):
        self.connection_string = connection_string
        self.spec = spec or VehicleSpec()

    async def arm(self) -> None:
        raise NotImplementedError(
            "MAVLink backend is not wired up. Use SimulatedVehicle, or implement "
            "this against pymavlink to fly the identical mission on hardware."
        )

    goto = collect = return_to_base = arm


async def execute(
    mission: Mission, vehicle: VehicleAdapter, *, skip: set[str] | None = None
) -> AsyncIterator[Telemetry]:
    """Fly an approved mission, honouring per-pickup human decisions."""
    if not mission.approved:
        raise PermissionError("mission has unresolved blocking approval gates")

    skip = skip or set()
    await vehicle.arm()

    for wp in mission.waypoints:
        vehicle.seq = wp.seq
        async for t in vehicle.goto(wp.lat, wp.lon):
            yield t

        if wp.hotspot_id in skip:
            yield Telemetry(
                t=vehicle.clock,
                lat=vehicle.lat,
                lon=vehicle.lon,
                heading_deg=vehicle.heading,
                speed_ms=0.0,
                battery_pct=vehicle.battery_pct,
                payload_kg=vehicle.payload_kg,
                state="skipped",
                seq=wp.seq,
                message="operator skipped this pickup",
            )
            continue

        mass = wp.payload_after_kg - vehicle.payload_kg
        async for t in vehicle.collect(
            max(mass, 0.0), mission.vehicle.collection_time_s * wp.item_count
        ):
            yield t

        if vehicle.battery_pct <= mission.vehicle.reserve_fraction * 100.0:
            yield Telemetry(
                t=vehicle.clock,
                lat=vehicle.lat,
                lon=vehicle.lon,
                heading_deg=vehicle.heading,
                speed_ms=0.0,
                battery_pct=vehicle.battery_pct,
                payload_kg=vehicle.payload_kg,
                state="rtl",
                seq=wp.seq,
                message="battery reserve reached, returning to base",
            )
            break

    async for t in vehicle.return_to_base():
        yield t
