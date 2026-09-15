import numpy as np
import pytest

from trident.sar import checkpoint_status, polygons
from trident.sar.segment import TILE_OVERLAP_PX, TILE_PX, _tiles, _to_grey


def test_status_reports_absent_checkpoint_without_raising():
    status = checkpoint_status("weights/definitely-not-here.pt")
    assert status["available"] is False
    assert "no checkpoint" in status["reason"]


def test_tiles_cover_every_pixel():
    """Gaps in tiling become blind stripes across the scene."""
    shape = (700, 900)
    covered = np.zeros(shape, bool)
    for y0, y1, x0, x1 in _tiles(shape, TILE_PX, TILE_OVERLAP_PX):
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_tiles_overlap_so_seams_get_averaged():
    shape = (700, 900)
    count = np.zeros(shape, int)
    for y0, y1, x0, x1 in _tiles(shape, TILE_PX, TILE_OVERLAP_PX):
        count[y0:y1, x0:x1] += 1
    assert count.max() > 1


def test_tiles_handle_a_scene_smaller_than_one_tile():
    windows = list(_tiles((120, 90), TILE_PX, TILE_OVERLAP_PX))
    assert len(windows) == 1


def test_greyscale_conversion_collapses_colour_and_rescales():
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 1] = 100
    rgb[..., 2] = 0
    grey = _to_grey(rgb)
    assert grey.shape == (16, 16)
    assert grey.mean() == pytest.approx(100.0, abs=1.0)


def test_float_input_is_rescaled_to_byte_range():
    """SAR arrays arrive in dB or float reflectance, not 0-255."""
    arr = np.linspace(-35.0, 0.0, 64 * 64).reshape(64, 64)
    grey = _to_grey(arr)
    assert grey.min() == pytest.approx(0.0, abs=1e-3)
    assert grey.max() == pytest.approx(255.0, abs=1e-3)


def test_polygons_finds_separate_slicks_not_one_hull():
    mask = np.zeros((200, 300), bool)
    mask[20:80, 20:90] = True
    mask[120:180, 200:280] = True
    shapes = polygons(mask)
    assert len(shapes) == 2


def test_polygons_discards_speckle():
    mask = np.zeros((200, 200), bool)
    mask[10:90, 10:90] = True
    mask[150, 150] = True  # single pixel
    assert len(polygons(mask, min_area_px=96)) == 1


def test_polygons_on_empty_mask_returns_nothing():
    assert polygons(np.zeros((64, 64), bool)) == []


def test_segmenter_refuses_a_missing_checkpoint_with_guidance():
    from trident.sar.segment import SARSegmenter

    with pytest.raises(FileNotFoundError, match="train_oil.py"):
        SARSegmenter("weights/nope.pt")


def test_uint8_input_is_not_rescaled():
    """Training chips were 8-bit PNGs used as-is; stretching changes contrast."""
    arr = np.zeros((32, 32), np.uint8)
    arr[:16] = 40
    arr[16:] = 60
    grey = _to_grey(arr)
    assert grey.min() == pytest.approx(40.0)
    assert grey.max() == pytest.approx(60.0)


def test_flat_float_scene_maps_to_mid_grey_not_zero():
    grey = _to_grey(np.full((32, 32), -12.0))
    assert grey.mean() == pytest.approx(127.5)
