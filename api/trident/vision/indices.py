"""Physics-motivated image indices for amorphous surface contamination.

Oil sheen and algal bloom have no edges, so an object detector cannot find
them. Satellite work uses spectral indices for exactly this reason -- the
Floating Debris Index and NDVI separate floating material from water far more
reliably than colour alone. Those need a near-infrared band, which consumer
RGB imagery does not carry, so this module computes RGB-domain analogues that
keep the underlying physics:

  * Oil damps capillary waves. A slick is therefore a LOW-texture anomaly
    sitting inside an otherwise wave-textured field. This is the optical
    equivalent of the dark-spot segmentation step used on SAR.
  * Oil films are iridescent: locally high hue variance at low saturation.
  * Algae raise the green fraction, which is what the excess-green and green
    chromatic coordinate indices measure.
  * Foam and sun glint are the dominant false positives, so they get their own
    indices rather than being left to contaminate the oil score.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class IndexStack:
    """Per-pixel evidence maps, each normalised to roughly [0, 1]."""

    water: np.ndarray
    texture: np.ndarray
    smoothness: np.ndarray
    iridescence: np.ndarray
    excess_green: np.ndarray
    brightness: np.ndarray
    brightness_excess: np.ndarray
    saturation: np.ndarray
    turbidity: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.water.shape


def _normalise(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi - lo < _EPS:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _local_std(gray: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Windowed standard deviation: the wave-texture proxy."""
    mean = cv2.blur(gray, (ksize, ksize))
    sq = cv2.blur(gray * gray, (ksize, ksize))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def _circular_std(hue_deg: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Hue is an angle, so its spread needs a circular statistic.

    Treating hue as a scalar makes red (0 deg) and red (359 deg) look maximally
    different, which would flag every red object as iridescent.
    """
    theta = np.radians(hue_deg * 2.0)
    mean_cos = cv2.blur(np.cos(theta), (ksize, ksize))
    mean_sin = cv2.blur(np.sin(theta), (ksize, ksize))
    resultant = np.clip(np.sqrt(mean_cos**2 + mean_sin**2), _EPS, 1.0)
    return np.sqrt(-2.0 * np.log(resultant)).astype(np.float32)


def water_mask(rgb: np.ndarray) -> np.ndarray:
    """Heuristic water/not-water split.

    Deliberately a heuristic rather than a learned segmenter: it costs nothing
    to run and its failure mode is visible in the overlay. It exists to supply
    the density denominator and to stop detections being counted on sky, land
    or the deck of a boat.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    h, w = hue.shape

    # Sky: bright, weakly saturated, and near the top of the frame.
    rows = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    sky = (val > 0.78) & (sat < 0.30) & (rows < 0.45)

    # Vegetation and bare land sit in the green/yellow hues at real saturation.
    land = (hue > 20) & (hue < 45) & (sat > 0.35) & (val > 0.25)

    mask = ~(sky | land)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def compute(rgb: np.ndarray) -> IndexStack:
    """Build the full evidence stack for one RGB frame."""
    rgb = np.ascontiguousarray(rgb)
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    texture = _normalise(_local_std(gray))

    # Oil suppresses surface roughness, so the useful signal is the ABSENCE of
    # texture in a frame that otherwise has some. In flat calm there is no
    # texture anywhere and this index correctly carries no information.
    contrast = float(np.std(gray))
    smoothness = (1.0 - texture) * np.clip(contrast / 0.08, 0.0, 1.0)

    # Iridescence needs low saturation: a bright saturated object is paint or
    # plastic, not a thin interference film.
    iridescence = _normalise(_circular_std(hue)) * np.clip(1.0 - sat * 1.8, 0.0, 1.0)

    total = r + g + b + _EPS
    excess_green = _normalise(np.clip(2.0 * g - r - b, 0.0, None) * (g / total))
    turbidity = _normalise(np.clip(r - b, 0.0, None) * val)

    water = water_mask(rgb)

    # Brightness relative to the water around it, not absolute. Foam and glint
    # are defined by standing out from their own background, so a dim overcast
    # scene and a bright noon scene have to be judged on the same scale.
    reference = float(np.median(val[water])) if water.any() else float(np.median(val))
    brightness_excess = _normalise(np.clip(val - reference, 0.0, None))

    return IndexStack(
        water=water.astype(np.float32),
        texture=texture,
        smoothness=smoothness.astype(np.float32),
        iridescence=iridescence.astype(np.float32),
        excess_green=excess_green,
        brightness=val.astype(np.float32),
        brightness_excess=brightness_excess,
        saturation=sat.astype(np.float32),
        turbidity=turbidity,
    )


# Each hypothesis names the evidence it REQUIRES and the evidence that rules
# it out. Scoring takes the geometric mean of the required terms rather than a
# weighted sum, because a sum lets one strong irrelevant signal carry a
# hypothesis: a smooth patch alone would read as oil even with no iridescence
# at all. Requiring every term forces oil to be both smooth and iridescent,
# and lets shallow seabed -- also smooth -- be separated by the absence of an
# interference film.
_HYPOTHESES: dict[str, dict[str, tuple[str, ...]]] = {
    "oil_slick": {
        "require": ("smoothness", "iridescence"),
        "suppress": ("brightness_excess", "texture"),
    },
    "algal_bloom": {
        "require": ("excess_green",),
        "suppress": ("brightness_excess", "iridescence"),
    },
    "sewage_plume": {
        "require": ("turbidity",),
        "suppress": ("excess_green", "iridescence"),
    },
    "sea_foam": {
        "require": ("brightness_excess", "texture"),
        "suppress": ("saturation",),
    },
    "sun_glint": {
        "require": ("brightness_excess",),
        "suppress": ("texture", "saturation"),
    },
    "shallow_seabed": {
        "require": ("smoothness", "turbidity"),
        "suppress": ("iridescence", "brightness_excess"),
    },
}

_EVIDENCE_KEYS = (
    "smoothness",
    "iridescence",
    "excess_green",
    "turbidity",
    "brightness_excess",
    "saturation",
    "texture",
)

# How hard a disqualifying feature pushes a hypothesis down.
_SUPPRESSION = 0.85


def region_evidence(stack: IndexStack, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {}
    return {
        key: round(float(getattr(stack, key)[mask].mean()), 3) for key in _EVIDENCE_KEYS
    }


def score_region(stack: IndexStack, mask: np.ndarray) -> dict[str, float]:
    """Mean evidence inside a region, scored against every hypothesis."""
    evidence = region_evidence(stack, mask)
    if not evidence:
        return {}

    scores = {}
    for name, spec in _HYPOTHESES.items():
        required = [max(evidence[k], _EPS) for k in spec["require"]]
        support = float(np.exp(np.mean(np.log(required))))
        for key in spec["suppress"]:
            support *= 1.0 - _SUPPRESSION * evidence[key]
        scores[name] = float(np.clip(support, 0.0, 1.0))
    return scores


def candidate_masks(
    stack: IndexStack, *, min_area_frac: float = 0.004, max_candidates: int = 8
) -> list[np.ndarray]:
    """Propose amorphous regions worth classifying.

    Proposals come from anywhere the frame departs from plain water, without
    deciding yet what the departure means. Naming happens in the discriminator
    so that rejections are explicit and reportable.
    """
    water = stack.water > 0.5
    # brightness_excess is in here so that foam and glint get PROPOSED. A
    # confuser that is never proposed can never be reported as rejected, which
    # would leave the score looking cleaner than the evidence supports.
    anomaly = np.maximum.reduce(
        [
            stack.smoothness,
            stack.iridescence,
            stack.excess_green,
            stack.turbidity,
            stack.brightness_excess,
        ]
    )
    anomaly = cv2.GaussianBlur(anomaly, (0, 0), 3.0)
    anomaly[~water] = 0.0

    inside = anomaly[water]
    if inside.size == 0:
        return []
    threshold = max(float(np.percentile(inside, 88)), 0.35)

    binary = cv2.morphologyEx(
        (anomaly >= threshold).astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((7, 7), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = max(64.0, min_area_frac * stack.water.size)
    areas = [
        (stats[i, cv2.CC_STAT_AREA], i)
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    areas.sort(reverse=True)
    return [labels == i for _, i in areas[:max_candidates]]
