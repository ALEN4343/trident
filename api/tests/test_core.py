import math

import numpy as np
import pytest

from trident.drift import DriftConfig, SyntheticForcing, attribute_source, forecast
from trident.geo import (
    CameraIntrinsics,
    CameraPose,
    PoseUncertainty,
    image_footprint,
    localise_pixel,
)
from trident.severity import DetectedItem, RegionCoverage, Scene, compute
from trident.taxonomy import load_taxonomy

NADIR = CameraPose(lat=12.9716, lon=77.5946, altitude_m=100.0, pitch_deg=-90.0)
CAM = CameraIntrinsics(width=1920, height=1080, hfov_deg=84.0)


# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------

def test_taxonomy_loads_and_ranks_hazard():
    tax = load_taxonomy()
    assert tax["fishing_net"].hazard > tax["plastic_bottle"].hazard
    assert tax["plastic_bottle"].hazard > tax["paper_cardboard"].hazard


def test_hazardous_classes_require_human_signoff():
    tax = load_taxonomy()
    for class_id in ("fishing_net", "drum_canister", "glass_bottle", "tyre"):
        assert tax.requires_human_signoff(class_id), class_id
    assert not tax.requires_human_signoff("plastic_bottle")


def test_windage_ordering_matches_buoyancy():
    tax = load_taxonomy()
    assert tax["styrofoam"].windage > tax["plastic_bottle"].windage
    assert tax["plastic_bottle"].windage > tax["fishing_net"].windage


def test_confusers_are_indexed_by_what_they_mimic():
    tax = load_taxonomy()
    mimics = {c.id for c in tax.confusers_for("oil_slick")}
    assert {"sun_glint", "sea_foam", "cloud_shadow"} <= mimics


def test_unknown_class_raises_with_helpful_message():
    with pytest.raises(KeyError, match="unknown pollution class"):
        load_taxonomy()["banana"]


# --------------------------------------------------------------------------
# Georectification
# --------------------------------------------------------------------------

def test_nadir_centre_pixel_lands_under_the_camera():
    fix = localise_pixel(960, 540, NADIR, CAM)
    assert fix is not None
    assert abs(fix.east_m) < 0.2
    assert abs(fix.north_m) < 0.2
    assert fix.slant_range_m == pytest.approx(100.0, abs=0.5)


def test_nadir_image_right_is_east_and_image_down_is_south():
    right = localise_pixel(1900, 540, NADIR, CAM)
    below = localise_pixel(960, 1000, NADIR, CAM)
    assert right.east_m > 50 and abs(right.north_m) < 0.5
    assert below.north_m < -30 and abs(below.east_m) < 0.5


def test_yaw_rotates_the_footprint():
    """Facing east, the right of a nadir frame points south."""
    east_facing = CameraPose(
        lat=NADIR.lat, lon=NADIR.lon, altitude_m=100.0, yaw_deg=90.0, pitch_deg=-90.0
    )
    fix = localise_pixel(1900, 540, east_facing, CAM)
    assert fix.north_m < -50 and abs(fix.east_m) < 0.5


def test_ground_sample_distance_matches_the_pinhole_model():
    fix = localise_pixel(960, 540, NADIR, CAM)
    expected = NADIR.altitude_m / CAM.fx  # metres per pixel at nadir
    assert fix.gsd_m == pytest.approx(expected, rel=0.02)


def test_horizon_rays_are_rejected_not_extrapolated():
    level = CameraPose(lat=NADIR.lat, lon=NADIR.lon, altitude_m=100.0, pitch_deg=0.0)
    assert localise_pixel(960, 540, level, CAM) is None


def test_nadir_footprint_area_matches_closed_form():
    fp = image_footprint(NADIR, CAM)
    width = 2 * NADIR.altitude_m * math.tan(math.radians(CAM.hfov_deg / 2))
    height = 2 * NADIR.altitude_m * math.tan(math.radians(CAM.vfov_deg / 2))
    assert fp.area_m2 == pytest.approx(width * height, rel=0.02)
    assert not fp.horizon_clipped


def test_oblique_footprint_is_flagged_as_clipped():
    oblique = CameraPose(
        lat=NADIR.lat, lon=NADIR.lon, altitude_m=100.0, pitch_deg=-8.0
    )
    fp = image_footprint(oblique, CAM, max_range_m=2000.0)
    assert fp.horizon_clipped
    assert fp.area_m2 > image_footprint(NADIR, CAM).area_m2


def test_uncertainty_grows_at_shallow_look_angles():
    unc = PoseUncertainty()
    steep = localise_pixel(960, 540, NADIR, CAM, uncertainty=unc, samples=200)
    shallow_pose = CameraPose(
        lat=NADIR.lat, lon=NADIR.lon, altitude_m=100.0, pitch_deg=-15.0
    )
    shallow = localise_pixel(960, 540, shallow_pose, CAM, uncertainty=unc, samples=200)
    assert steep.uncertainty_radius_m < shallow.uncertainty_radius_m


def test_projection_produces_plausible_coordinates():
    fix = localise_pixel(1900, 200, NADIR, CAM)
    assert abs(fix.lat - NADIR.lat) < 0.01
    assert abs(fix.lon - NADIR.lon) < 0.01
    assert fix.geojson["coordinates"] == [fix.lon, fix.lat]


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

FAST_DRIFT = DriftConfig(horizon_hours=6.0, dt_seconds=600.0, particles=120, seed=7)


def test_pure_eastward_current_moves_debris_east():
    field = SyntheticForcing(current_speed=1.0, current_bearing=90.0, wind_speed=0.0)
    result = forecast(12.9716, 77.5946, "plastic_bottle", field, config=FAST_DRIFT)
    end = result.centroid_track[-1]
    assert end[1] > 77.5946
    assert abs(end[0] - 12.9716) < 0.002


def test_windage_separates_a_foam_block_from_a_waterlogged_net():
    """Same release point, different buoyancy, measurably different endpoint."""
    field = SyntheticForcing(current_speed=0.0, wind_speed=10.0, wind_from=180.0)
    foam = forecast(12.9716, 77.5946, "styrofoam", field, config=FAST_DRIFT)
    net = forecast(12.9716, 77.5946, "fishing_net", field, config=FAST_DRIFT)
    assert foam.displacement_m() > 5 * net.displacement_m()


def test_reverse_drift_retraces_the_forward_track():
    """Hindcasting from the endpoint recovers the release point.

    The hindcast must start at the observation time, not at 'now' -- the
    forcing is time varying, so a hindcast anchored to the wrong hour
    integrates a different current field and lands somewhere else.
    """
    field = SyntheticForcing()
    fwd = forecast(12.9716, 77.5946, "plastic_bottle", field, config=FAST_DRIFT)
    end_lat, end_lon = fwd.centroid_track[-1]

    back = attribute_source(
        end_lat,
        end_lon,
        "plastic_bottle",
        field,
        lookback_hours=6.0,
        start=fwd.times[-1],
        config=FAST_DRIFT,
    )
    origin_lat, origin_lon = back.centroid_track[-1]

    assert origin_lat == pytest.approx(12.9716, abs=3e-3)
    assert origin_lon == pytest.approx(77.5946, abs=3e-3)


def test_ensemble_spreads_over_time():
    field = SyntheticForcing()
    result = forecast(12.9716, 77.5946, "plastic_bottle", field, config=FAST_DRIFT)
    assert result.spread_m(0) < result.spread_m(-1)


def test_drift_geojson_is_well_formed():
    field = SyntheticForcing()
    result = forecast(12.9716, 77.5946, "plastic_bottle", field, config=FAST_DRIFT)
    gj = result.to_geojson()
    assert gj["type"] == "FeatureCollection"
    track = gj["features"][0]
    assert track["geometry"]["type"] == "LineString"
    assert track["properties"]["synthetic_forcing"] is True
    assert len(track["geometry"]["coordinates"]) == len(result.times)


# --------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------

def test_one_ghost_net_outscores_twenty_cardboard_boxes():
    net = compute(
        Scene(items=(DetectedItem("fishing_net"),), water_area_m2=1000.0)
    )
    paper = compute(
        Scene(items=(DetectedItem("paper_cardboard", count=20),), water_area_m2=1000.0)
    )
    assert net.score > paper.score
    assert net.dominant_hazard == "fishing_net"


def test_empty_scene_scores_zero_and_says_so():
    result = compute(Scene(water_area_m2=500.0))
    assert result.score == 0.0
    assert any("No pollution detected" in n for n in result.notes)


def test_missing_footprint_is_flagged_as_a_lower_bound():
    result = compute(Scene(items=(DetectedItem("plastic_bottle", count=5),)))
    assert any("lower bound" in n for n in result.notes)
    assert result.breakdown()["density"] == 0.0


def test_oil_coverage_drives_score_without_any_countable_item():
    result = compute(
        Scene(
            regions=(RegionCoverage("oil_slick", percent_of_water=30.0),),
            water_area_m2=5000.0,
        )
    )
    assert result.breakdown()["coverage"] > 0
    assert result.score > 15
    assert result.dominant_hazard == "oil_slick"


def test_low_confidence_detections_widen_the_band():
    scene = Scene(
        items=(
            DetectedItem("fishing_net", confidence=0.95),
            DetectedItem("tyre", confidence=0.3),
        ),
        water_area_m2=800.0,
    )
    result = compute(scene)
    assert result.upper > result.lower


def test_recoverable_value_uses_scrap_rates():
    result = compute(
        Scene(items=(DetectedItem("metal_can", count=100),), water_area_m2=500.0)
    )
    assert result.total_mass_kg == pytest.approx(1.5, rel=1e-6)
    assert result.recoverable_value_inr == pytest.approx(165.0, rel=1e-6)


def test_component_weights_sum_to_one():
    weights = load_taxonomy().severity.component_weights
    assert sum(weights.values()) == pytest.approx(1.0)


def test_score_never_exceeds_one_hundred():
    worst = Scene(
        items=tuple(
            DetectedItem(c.id, count=500) for c in load_taxonomy().instances()
        ),
        regions=tuple(
            RegionCoverage(c.id, percent_of_water=100.0)
            for c in load_taxonomy().regions()
        ),
        water_area_m2=10.0,
        ecological_sensitivity=1.0,
    )
    assert compute(worst).score <= 100.0
