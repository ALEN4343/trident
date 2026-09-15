"""Projects image pixels onto the water plane to recover real-world coordinates.

Given a camera pose (position, altitude above the surface, and gimbal attitude)
this solves the pinhole ray / plane intersection for any pixel, which gives us
three things a bounding box alone cannot:

  * a latitude and longitude for every individual detection
  * the ground sample distance, and therefore true object size in metres
  * the imaged water area, which is the denominator for a density metric

Frames
------
Camera: x right, y down, z along the optical axis. Matches OpenCV.
World:  local ENU tangent plane -- x East, y North, z Up -- with the origin at
        the camera's ground track and the water surface at z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_378_137.0

# Rays flatter than this are treated as pointing at the horizon. Beyond it the
# intersection is numerically unstable and the position error exceeds any
# useful tolerance.
_MIN_DEPRESSION_RAD = math.radians(1.0)


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is and where it is looking.

    yaw is the heading of the optical axis in degrees clockwise from true
    north. pitch is the depression of the optical axis: 0 is horizontal and
    -90 is straight down (nadir). roll is rotation about the optical axis.
    """

    lat: float
    lon: float
    altitude_m: float
    yaw_deg: float = 0.0
    pitch_deg: float = -90.0
    roll_deg: float = 0.0


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    hfov_deg: float

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    @property
    def principal_point(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fx))


@dataclass(frozen=True)
class PoseUncertainty:
    """1-sigma error on the pose, used to size the confidence ellipse.

    Defaults are representative of a consumer drone: a gimbal attitude good to
    about a degree, a barometric altitude good to a couple of metres, and an
    uncorrected GNSS fix good to a couple of metres horizontally.
    """

    attitude_deg: float = 1.0
    altitude_m: float = 2.0
    position_m: float = 2.5


@dataclass(frozen=True)
class GroundFix:
    lat: float
    lon: float
    east_m: float
    north_m: float
    slant_range_m: float
    gsd_m: float
    uncertainty_radius_m: float | None = None

    @property
    def geojson(self) -> dict:
        return {"type": "Point", "coordinates": [self.lon, self.lat]}


def _rotation_world_from_camera(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))

    # Camera at rest: looking due north and level, so x->East, y->Down, z->North.
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    c, s = math.cos(pitch), math.sin(pitch)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    # Negated because yaw is clockwise from north while a right-handed rotation
    # about Up is counter-clockwise.
    c, s = math.cos(-yaw), math.sin(-yaw)
    r_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    c, s = math.cos(roll), math.sin(roll)
    r_roll = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    return r_yaw @ r_pitch @ base @ r_roll


def _ray_directions(
    pixels: np.ndarray, intrinsics: CameraIntrinsics, rotation: np.ndarray
) -> np.ndarray:
    cx, cy = intrinsics.principal_point
    rays = np.stack(
        [
            (pixels[:, 0] - cx) / intrinsics.fx,
            (pixels[:, 1] - cy) / intrinsics.fy,
            np.ones(len(pixels)),
        ],
        axis=1,
    )
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays @ rotation.T


def _enu_to_latlon(
    lat0: float, lon0: float, east: np.ndarray, north: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(north / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return lat, lon


def project_pixels(
    pixels: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect pixel rays with the water plane.

    Returns an (N, 3) array of local ENU coordinates and an (N,) boolean mask
    marking which pixels produced a usable intersection. Rays at or above the
    horizon are masked out rather than silently projected to a huge distance.
    """
    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    rotation = _rotation_world_from_camera(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    directions = _ray_directions(pixels, intrinsics, rotation)

    down = -directions[:, 2]
    valid = down > math.sin(_MIN_DEPRESSION_RAD)

    scale = np.where(valid, pose.altitude_m / np.where(valid, down, 1.0), np.nan)
    points = directions * scale[:, None]
    points[:, 2] = 0.0
    return points, valid


def localise_pixel(
    u: float,
    v: float,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    uncertainty: PoseUncertainty | None = None,
    samples: int = 256,
    rng: np.random.Generator | None = None,
) -> GroundFix | None:
    """Georeference one pixel, with a Monte Carlo confidence radius.

    The radius is the 95th percentile displacement over poses drawn from the
    supplied error model. It grows sharply at shallow look angles, which is the
    honest answer -- an oblique shot of the horizon cannot be localised well and
    the report should say so instead of printing six decimal places.
    """
    points, valid = project_pixels(np.array([[u, v]]), pose, intrinsics)
    if not valid[0]:
        return None

    east, north = float(points[0, 0]), float(points[0, 1])
    slant = float(math.hypot(math.hypot(east, north), pose.altitude_m))

    neighbour, neighbour_valid = project_pixels(
        np.array([[u + 1.0, v], [u, v + 1.0]]), pose, intrinsics
    )
    if neighbour_valid.all():
        gsd = float(
            np.mean(np.linalg.norm(neighbour[:, :2] - np.array([east, north]), axis=1))
        )
    else:
        gsd = float("nan")

    radius = None
    if uncertainty is not None:
        generator = rng or np.random.default_rng(0)
        offsets = np.empty((samples, 2))
        for i in range(samples):
            perturbed = CameraPose(
                lat=pose.lat,
                lon=pose.lon,
                altitude_m=max(
                    0.1,
                    pose.altitude_m
                    + generator.normal(0.0, uncertainty.altitude_m),
                ),
                yaw_deg=pose.yaw_deg + generator.normal(0.0, uncertainty.attitude_deg),
                pitch_deg=pose.pitch_deg
                + generator.normal(0.0, uncertainty.attitude_deg),
                roll_deg=pose.roll_deg + generator.normal(0.0, uncertainty.attitude_deg),
            )
            sample_points, sample_valid = project_pixels(
                np.array([[u, v]]), perturbed, intrinsics
            )
            offsets[i] = (
                sample_points[0, :2] if sample_valid[0] else (np.nan, np.nan)
            )

        jitter = generator.normal(0.0, uncertainty.position_m, size=(samples, 2))
        displacement = np.linalg.norm(
            offsets + jitter - np.array([east, north]), axis=1
        )
        finite = displacement[np.isfinite(displacement)]
        radius = float(np.percentile(finite, 95)) if finite.size else None

    lat, lon = _enu_to_latlon(
        pose.lat, pose.lon, np.array([east]), np.array([north])
    )
    return GroundFix(
        lat=float(lat[0]),
        lon=float(lon[0]),
        east_m=east,
        north_m=north,
        slant_range_m=slant,
        gsd_m=gsd,
        uncertainty_radius_m=radius,
    )


@dataclass(frozen=True)
class Footprint:
    polygon: list[tuple[float, float]]
    area_m2: float
    horizon_clipped: bool

    @property
    def geojson(self) -> dict:
        ring = [[lon, lat] for lat, lon in self.polygon]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return {"type": "Polygon", "coordinates": [ring]}


def image_footprint(
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    max_range_m: float = 2000.0,
    edge_samples: int = 16,
) -> Footprint | None:
    """The patch of water the frame actually covers.

    This is the denominator for any density figure. Counting twelve bottles
    means nothing until you know whether the frame covers 50 m^2 or 5000 m^2,
    and an oblique frame covers wildly more area at the top than the bottom.
    """
    w, h = intrinsics.width - 1, intrinsics.height - 1
    t = np.linspace(0.0, 1.0, edge_samples)
    border = np.concatenate(
        [
            np.stack([t * w, np.zeros_like(t)], axis=1),
            np.stack([np.full_like(t, w), t * h], axis=1),
            np.stack([(1 - t) * w, np.full_like(t, h)], axis=1),
            np.stack([np.zeros_like(t), (1 - t) * h], axis=1),
        ]
    )

    points, valid = project_pixels(border, pose, intrinsics)
    if not valid.any():
        return None

    kept = points[valid][:, :2]
    ranges = np.linalg.norm(kept, axis=1)
    clipped = bool((~valid).any() or (ranges > max_range_m).any())
    if clipped:
        scale = np.minimum(1.0, max_range_m / np.maximum(ranges, 1e-9))
        kept = kept * scale[:, None]

    east, north = kept[:, 0], kept[:, 1]
    area = 0.5 * abs(
        np.dot(east, np.roll(north, -1)) - np.dot(north, np.roll(east, -1))
    )

    lat, lon = _enu_to_latlon(pose.lat, pose.lon, east, north)
    return Footprint(
        polygon=[(float(a), float(o)) for a, o in zip(lat, lon)],
        area_m2=float(area),
        horizon_clipped=clipped,
    )
