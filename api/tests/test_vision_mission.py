import cv2
import numpy as np
import pytest

from trident.mission import Target, VehicleSpec, cluster, haversine, plan
from trident.vision import indices
from trident.vision.pipeline import analyse


def _water(h=360, w=540, seed=3):
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0], base[..., 1], base[..., 2] = 0.13, 0.34, 0.42
    wave = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    wave = (wave - wave.min()) / (np.ptp(wave) + 1e-6)
    return base + (wave[..., None] - 0.5) * 0.22


def _add_oil(img, centre=(150, 180)):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, centre, (85, 55), 20, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 8)
    swirl = np.sin(np.linspace(0, 9, w))[None, :] * np.cos(np.linspace(0, 7, h))[:, None]
    for c, amp in enumerate((0.05, -0.03, 0.06)):
        img[..., c] = img[..., c] * (1 - mask * 0.85) + mask * (0.30 + amp * swirl + 0.02)
    return img


def _add_foam(img, centre=(410, 220), seed=5):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, centre, (70, 45), -15, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    speck = np.clip(
        cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.0), -1, 1
    )
    return img * (1 - mask[..., None] * 0.9) + mask[..., None] * (
        0.88 + speck[..., None] * 0.16
    )


def _u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Physics branch
# --------------------------------------------------------------------------

def test_oil_is_accepted_and_foam_is_rejected():
    """The whole credibility claim: confusers are named, not silently counted."""
    rgb = _u8(_add_foam(_add_oil(_water())))
    result = analyse(rgb)

    assert any(r.class_id == "oil_slick" for r in result.regions)
    assert any(r.rejected_as == "sea_foam" for r in result.rejections)

    rejection = next(r for r in result.rejections if r.rejected_as == "sea_foam")
    assert rejection.would_have_been
    assert rejection.margin > 0
    assert "capillary" in rejection.discriminator or rejection.discriminator


def test_clean_water_yields_no_pollution():
    result = analyse(_u8(_water(seed=11)))
    assert result.regions == []
    assert result.mpsi.score < 5.0


def test_oil_requires_iridescence_not_just_smoothness():
    """A smooth patch with no interference film must not read as oil."""
    img = _water()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (150, 180), (85, 55), 20, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 8)
    flat = img * (1 - mask[..., None] * 0.8) + mask[..., None] * np.array(
        [0.13, 0.34, 0.42], np.float32
    )

    stack = indices.compute(_u8(flat))
    candidates = indices.candidate_masks(stack)
    for candidate in candidates:
        scores = indices.score_region(stack, candidate)
        assert scores["oil_slick"] < 0.45


def test_water_mask_excludes_bright_sky():
    img = _water()
    img[:100, :, :] = 0.92
    stack = indices.compute(_u8(img))
    assert stack.water[:80].mean() < 0.35
    assert stack.water[200:].mean() > 0.85


def test_analysis_serialises_for_the_api():
    payload = analyse(_u8(_add_oil(_water()))).to_dict()
    assert {"items", "regions", "rejections", "severity", "safety"} <= payload.keys()
    assert "components" in payload["severity"]


# --------------------------------------------------------------------------
# Mission planning
# --------------------------------------------------------------------------

BASE = (12.9700, 77.5900)


def _target(i, lat, lon, class_id="plastic_bottle", conf=0.9):
    from trident.taxonomy import load_taxonomy

    cls = load_taxonomy()[class_id]
    return Target(
        id=f"t{i}",
        lat=lat,
        lon=lon,
        class_id=class_id,
        label=cls.label,
        mass_kg=cls.typical_mass_kg or 0.0,
        hazard=cls.hazard,
        confidence=conf,
        auto_collect=cls.auto_collect,
    )


def test_nearby_targets_collapse_into_one_hotspot():
    targets = [
        _target(0, 12.9716, 77.5946),
        _target(1, 12.97161, 77.59461),
        _target(2, 12.9760, 77.5990),
    ]
    hotspots = cluster(targets, radius_m=25.0)
    assert len(hotspots) == 2


def test_unsafe_classes_raise_a_blocking_pickup_gate():
    mission = plan([_target(0, 12.9716, 77.5946, "fishing_net")], BASE)
    gates = [g for g in mission.gates if g.kind == "pickup"]
    assert gates and all(g.blocking for g in gates)
    assert not mission.approved


def test_low_confidence_raises_a_non_blocking_gate():
    mission = plan([_target(0, 12.9716, 77.5946, "plastic_bottle", conf=0.4)], BASE)
    gate = next(g for g in mission.gates if g.kind == "pickup")
    assert not gate.blocking
    assert "Low confidence" in gate.reason


def test_wildlife_in_frame_blocks_the_mission():
    mission = plan(
        [_target(0, 12.9716, 77.5946)], BASE, wildlife=("bird",)
    )
    assert any("Wildlife" in g.reason and g.blocking for g in mission.gates)


def test_mission_always_requires_explicit_authorisation():
    mission = plan([_target(0, 12.9716, 77.5946)], BASE)
    assert any(g.kind == "mission" and g.blocking for g in mission.gates)
    assert not mission.approved

    for gate in mission.gates:
        gate.resolved = True
    assert mission.approved


def test_route_visits_every_reachable_hotspot_once():
    targets = [
        _target(i, 12.9716 + 0.002 * i, 77.5946 + 0.002 * (i % 3)) for i in range(6)
    ]
    mission = plan(targets, BASE)
    seen = [w.hotspot_id for w in mission.waypoints]
    assert len(seen) == len(set(seen))
    assert [w.seq for w in mission.waypoints] == list(range(1, len(seen) + 1))


def test_endurance_limit_defers_rather_than_overcommits():
    tiny = VehicleSpec(cruise_speed_ms=0.5, endurance_s=600.0)
    targets = [_target(i, 12.9716 + 0.02 * i, 77.5946) for i in range(6)]
    mission = plan(targets, BASE, vehicle=tiny)
    assert mission.deferred
    assert any("deferred" in n for n in mission.notes)


def test_payload_capacity_is_respected():
    small = VehicleSpec(payload_capacity_kg=20.0)
    targets = [
        _target(i, 12.9716 + 0.003 * i, 77.5946, "fishing_net") for i in range(4)
    ]
    mission = plan(targets, BASE, vehicle=small)
    assert mission.payload_kg <= small.payload_capacity_kg


def test_drift_aware_routing_aims_ahead_of_the_observation():
    """Waypoints target predicted positions, not the ones in the photograph."""
    targets = [_target(i, 12.9716 + 0.002 * i, 77.5946) for i in range(3)]

    def drift_fn(lat, lon, class_id, seconds):
        return lat, lon + (seconds / 3600.0) * 0.004  # eastward set

    plain = plan(targets, BASE)
    aimed = plan(targets, BASE, drift_fn=drift_fn)

    assert not plain.drift_aware and aimed.drift_aware
    assert all(w.drift_offset_m == pytest.approx(0.0, abs=0.1) for w in plain.waypoints)
    assert any(w.drift_offset_m > 1.0 for w in aimed.waypoints)


def test_mission_serialises_for_the_api():
    payload = plan([_target(0, 12.9716, 77.5946)], BASE).to_dict()
    assert payload["summary"]["blocking_gates"] >= 1
    assert payload["waypoints"][0]["seq"] == 1


def test_haversine_matches_a_known_separation():
    # One degree of latitude is close to 111 km anywhere on the globe.
    assert haversine(0.0, 0.0, 1.0, 0.0) == pytest.approx(111_195, rel=0.01)
