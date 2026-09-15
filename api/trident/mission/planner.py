"""Cleanup mission planning with a human in the loop.

Three things separate this from a plain travelling-salesman route:

  * It plans against where the debris WILL be. A vessel reaching a waypoint an
    hour from now meets water that has moved; routing to the position observed
    in the photograph sends it to empty water. Targets are advected to their
    own estimated time of arrival and the route is re-solved.
  * Travel cost accounts for the current. Making way against a half-knot set
    costs real endurance, so the same geometry is not the same mission
    depending on which way the water is going.
  * Autonomy is gated, not assumed. Items the taxonomy marks as unsafe for
    unattended collection, low-confidence detections, and any scene with
    wildlife in frame raise an approval gate that a human has to clear.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

EARTH_RADIUS_M = 6_378_137.0

# Detections below this are collected only after a human confirms the call.
_CONFIDENCE_GATE = 0.55


@dataclass(frozen=True)
class VehicleSpec:
    name: str = "SeaCAT-class USV"
    cruise_speed_ms: float = 1.6
    payload_capacity_kg: float = 180.0
    endurance_s: float = 4.0 * 3600.0
    collection_time_s: float = 45.0
    # Reserve held back for the return leg; the planner refuses to spend it.
    reserve_fraction: float = 0.20


@dataclass(frozen=True)
class Target:
    id: str
    lat: float
    lon: float
    class_id: str
    label: str
    mass_kg: float
    hazard: float
    confidence: float
    auto_collect: bool


@dataclass
class Hotspot:
    id: str
    lat: float
    lon: float
    targets: list[Target]

    @property
    def mass_kg(self) -> float:
        return sum(t.mass_kg for t in self.targets)

    @property
    def peak_hazard(self) -> float:
        return max((t.hazard for t in self.targets), default=0.0)

    @property
    def needs_signoff(self) -> bool:
        return any(
            not t.auto_collect or t.confidence < _CONFIDENCE_GATE for t in self.targets
        )


@dataclass
class ApprovalGate:
    kind: str  # mission | pickup
    reason: str
    detail: str
    blocking: bool
    hotspot_id: str | None = None
    target_id: str | None = None
    resolved: bool = False
    decision: str | None = None  # approve | skip | flag

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "detail": self.detail,
            "blocking": self.blocking,
            "hotspot_id": self.hotspot_id,
            "target_id": self.target_id,
            "resolved": self.resolved,
            "decision": self.decision,
        }


@dataclass
class Waypoint:
    seq: int
    hotspot_id: str
    lat: float
    lon: float
    observed_lat: float
    observed_lon: float
    leg_distance_m: float
    eta_s: float
    payload_after_kg: float
    energy_used_pct: float
    item_count: int
    needs_signoff: bool

    @property
    def drift_offset_m(self) -> float:
        return haversine(self.observed_lat, self.observed_lon, self.lat, self.lon)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "hotspot_id": self.hotspot_id,
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "observed_lat": round(self.observed_lat, 6),
            "observed_lon": round(self.observed_lon, 6),
            "drift_offset_m": round(self.drift_offset_m, 1),
            "leg_distance_m": round(self.leg_distance_m, 1),
            "eta_s": round(self.eta_s),
            "payload_after_kg": round(self.payload_after_kg, 2),
            "energy_used_pct": round(self.energy_used_pct, 1),
            "item_count": self.item_count,
            "needs_signoff": self.needs_signoff,
        }


@dataclass
class Mission:
    id: str
    vehicle: VehicleSpec
    base: tuple[float, float]
    waypoints: list[Waypoint]
    gates: list[ApprovalGate]
    deferred: list[Hotspot] = field(default_factory=list)
    drift_aware: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_distance_m(self) -> float:
        return sum(w.leg_distance_m for w in self.waypoints)

    @property
    def duration_s(self) -> float:
        return self.waypoints[-1].eta_s if self.waypoints else 0.0

    @property
    def payload_kg(self) -> float:
        return self.waypoints[-1].payload_after_kg if self.waypoints else 0.0

    @property
    def item_count(self) -> int:
        return sum(w.item_count for w in self.waypoints)

    @property
    def approved(self) -> bool:
        return all(g.resolved for g in self.gates if g.blocking)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vehicle": {
                "name": self.vehicle.name,
                "cruise_speed_ms": self.vehicle.cruise_speed_ms,
                "payload_capacity_kg": self.vehicle.payload_capacity_kg,
                "endurance_s": self.vehicle.endurance_s,
            },
            "base": {"lat": self.base[0], "lon": self.base[1]},
            "drift_aware": self.drift_aware,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "gates": [g.to_dict() for g in self.gates],
            "deferred": [
                {"id": h.id, "lat": h.lat, "lon": h.lon, "items": len(h.targets)}
                for h in self.deferred
            ],
            "summary": {
                "waypoints": len(self.waypoints),
                "items": self.item_count,
                "distance_m": round(self.total_distance_m, 1),
                "duration_s": round(self.duration_s),
                "payload_kg": round(self.payload_kg, 2),
                "energy_used_pct": round(
                    self.waypoints[-1].energy_used_pct if self.waypoints else 0.0, 1
                ),
                "approved": self.approved,
                "blocking_gates": sum(
                    1 for g in self.gates if g.blocking and not g.resolved
                ),
            },
            "notes": self.notes,
        }


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def cluster(targets: list[Target], radius_m: float = 25.0) -> list[Hotspot]:
    """Group targets a vehicle can collect without repositioning."""
    remaining = list(targets)
    hotspots: list[Hotspot] = []

    while remaining:
        # Seed on the most hazardous item left so clusters form around the
        # things worth collecting first.
        seed = max(remaining, key=lambda t: t.hazard)
        remaining.remove(seed)
        members = [seed]

        for other in list(remaining):
            if haversine(seed.lat, seed.lon, other.lat, other.lon) <= radius_m:
                members.append(other)
                remaining.remove(other)

        hotspots.append(
            Hotspot(
                id=uuid.uuid4().hex[:8],
                lat=sum(t.lat for t in members) / len(members),
                lon=sum(t.lon for t in members) / len(members),
                targets=members,
            )
        )
    return hotspots


def _leg_seconds(
    lat1: float, lon1: float, lat2: float, lon2: float, vehicle: VehicleSpec, forcing
) -> tuple[float, float]:
    """Travel time for a leg, accounting for the along-track current."""
    distance = haversine(lat1, lon1, lat2, lon2)
    speed = vehicle.cruise_speed_ms

    if forcing is not None and distance > 1.0:
        from ..drift.forcing import utc_now

        sample = forcing.at(lat1, lon1, utc_now())
        east = (lon2 - lon1) * math.cos(math.radians(lat1))
        north = lat2 - lat1
        norm = math.hypot(east, north) or 1.0
        along = (sample.current_e * east + sample.current_n * north) / norm
        # Never let an adverse set drive effective speed to zero.
        speed = max(0.3 * vehicle.cruise_speed_ms, vehicle.cruise_speed_ms + along)

    return distance, distance / speed


def _order(
    base: tuple[float, float], hotspots: list[Hotspot], vehicle: VehicleSpec, forcing
) -> list[Hotspot]:
    """Nearest-neighbour seed refined by 2-opt.

    At the tens-of-waypoints scale a hackathon mission actually has, 2-opt
    lands on or near the optimum in microseconds, which is not worth an
    external solver dependency.
    """
    if len(hotspots) <= 2:
        return hotspots

    unvisited = list(hotspots)
    route: list[Hotspot] = []
    cur = base
    while unvisited:
        nxt = min(unvisited, key=lambda h: haversine(cur[0], cur[1], h.lat, h.lon))
        unvisited.remove(nxt)
        route.append(nxt)
        cur = (nxt.lat, nxt.lon)

    def cost(order: list[Hotspot]) -> float:
        total, prev = 0.0, base
        for h in order:
            total += _leg_seconds(prev[0], prev[1], h.lat, h.lon, vehicle, forcing)[1]
            prev = (h.lat, h.lon)
        return total

    best = cost(route)
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                candidate = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]
                c = cost(candidate)
                if c < best - 1e-6:
                    route, best, improved = candidate, c, True
    return route


def plan(
    targets: list[Target],
    base: tuple[float, float],
    *,
    vehicle: VehicleSpec | None = None,
    forcing=None,
    drift_fn=None,
    wildlife: tuple[str, ...] = (),
    cluster_radius_m: float = 25.0,
) -> Mission:
    """Build an approvable mission.

    drift_fn, when supplied, maps (lat, lon, class_id, seconds) to the position
    that item is expected to occupy after that delay. Passing it turns the
    route from a snapshot into an interception.
    """
    vehicle = vehicle or VehicleSpec()
    gates: list[ApprovalGate] = []
    notes: list[str] = []

    if not targets:
        return Mission(uuid.uuid4().hex[:8], vehicle, base, [], [], notes=["No collectable targets."])

    hotspots = cluster(targets, cluster_radius_m)
    ordered = _order(base, hotspots, vehicle, forcing)

    # First pass fixes the schedule, second pass re-aims each waypoint at where
    # its debris will have drifted by the arrival time from that schedule.
    waypoints = _walk(ordered, base, vehicle, forcing, None)
    if drift_fn is not None:
        eta_by_hotspot = {w.hotspot_id: w.eta_s for w in waypoints}
        waypoints = _walk(ordered, base, vehicle, forcing, (drift_fn, eta_by_hotspot))

    scheduled_ids = {w.hotspot_id for w in waypoints}
    deferred = [h for h in hotspots if h.id not in scheduled_ids]
    if deferred:
        notes.append(
            f"{len(deferred)} hotspot(s) deferred: endurance or payload reserve reached."
        )

    gates.append(
        ApprovalGate(
            kind="mission",
            reason="Mission authorisation",
            detail=(
                f"{len(waypoints)} waypoints, {sum(w.item_count for w in waypoints)} items, "
                f"{sum(w.leg_distance_m for w in waypoints) / 1000:.2f} km, "
                f"{(waypoints[-1].eta_s if waypoints else 0) / 60:.0f} min"
            ),
            blocking=True,
        )
    )

    if wildlife:
        gates.append(
            ApprovalGate(
                kind="mission",
                reason="Wildlife in survey area",
                detail=(
                    f"Detected: {', '.join(wildlife)}. Autonomous collection is "
                    "inhibited until an operator clears the area."
                ),
                blocking=True,
            )
        )

    by_id = {h.id: h for h in hotspots}
    for wp in waypoints:
        hotspot = by_id[wp.hotspot_id]
        for target in hotspot.targets:
            if not target.auto_collect:
                gates.append(
                    ApprovalGate(
                        kind="pickup",
                        reason=f"{target.label} requires sign-off",
                        detail=(
                            "Taxonomy marks this class as unsafe for unattended "
                            "collection (entanglement, sharps or unknown contents)."
                        ),
                        blocking=True,
                        hotspot_id=hotspot.id,
                        target_id=target.id,
                    )
                )
            elif target.confidence < _CONFIDENCE_GATE:
                gates.append(
                    ApprovalGate(
                        kind="pickup",
                        reason=f"Low confidence: {target.label}",
                        detail=(
                            f"Classified at {target.confidence:.0%}, below the "
                            f"{_CONFIDENCE_GATE:.0%} autonomy threshold."
                        ),
                        blocking=False,
                        hotspot_id=hotspot.id,
                        target_id=target.id,
                    )
                )

    return Mission(
        id=uuid.uuid4().hex[:8],
        vehicle=vehicle,
        base=base,
        waypoints=waypoints,
        gates=gates,
        deferred=deferred,
        drift_aware=drift_fn is not None,
        notes=notes,
    )


def _walk(
    ordered: list[Hotspot],
    base: tuple[float, float],
    vehicle: VehicleSpec,
    forcing,
    drift,
) -> list[Waypoint]:
    usable_endurance = vehicle.endurance_s * (1.0 - vehicle.reserve_fraction)
    waypoints: list[Waypoint] = []
    cur = base
    elapsed = payload = 0.0

    for hotspot in ordered:
        lat, lon = hotspot.lat, hotspot.lon
        if drift is not None:
            drift_fn, etas = drift
            lead = etas.get(hotspot.id, elapsed)
            lat, lon = drift_fn(
                hotspot.lat, hotspot.lon, hotspot.targets[0].class_id, lead
            )

        distance, travel = _leg_seconds(cur[0], cur[1], lat, lon, vehicle, forcing)
        service = vehicle.collection_time_s * len(hotspot.targets)
        arrival = elapsed + travel + service

        # Must still be able to get home on the reserve.
        _, home = _leg_seconds(lat, lon, base[0], base[1], vehicle, forcing)
        if arrival + home > usable_endurance:
            break
        if payload + hotspot.mass_kg > vehicle.payload_capacity_kg:
            break

        elapsed = arrival
        payload += hotspot.mass_kg
        cur = (lat, lon)

        waypoints.append(
            Waypoint(
                seq=len(waypoints) + 1,
                hotspot_id=hotspot.id,
                lat=lat,
                lon=lon,
                observed_lat=hotspot.lat,
                observed_lon=hotspot.lon,
                leg_distance_m=distance,
                eta_s=elapsed,
                payload_after_kg=payload,
                energy_used_pct=100.0 * elapsed / vehicle.endurance_s,
                item_count=len(hotspot.targets),
                needs_signoff=hotspot.needs_signoff,
            )
        )

    return waypoints
