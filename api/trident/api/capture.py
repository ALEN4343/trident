"""Recovering a camera pose from an uploaded image.

Three sources, in descending order of trust:

  1. Full EXIF plus the DJI XMP block, which carries gimbal attitude and
     height above the take-off point. A real drone frame georeferences with
     no user input at all.
  2. Plain EXIF GPS from a phone. Position is known, attitude is not, so the
     caller has to supply it.
  3. Nothing, which is the case for every public dataset image. The operator
     places the frame on a map and states the capture geometry.

Case 3 is labelled `declared` throughout so a synthesised pose is never
presented as measured telemetry.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from PIL import Image, ExifTags

from ..geo import CameraIntrinsics, CameraPose

_GPS_IFD = next(k for k, v in ExifTags.TAGS.items() if v == "GPSInfo")

# DJI and several other vendors write flight attitude into the XMP packet.
_XMP_FIELDS = {
    "gimbal_pitch": r'GimbalPitchDegree="?([-+0-9.]+)',
    "gimbal_yaw": r'GimbalYawDegree="?([-+0-9.]+)',
    "gimbal_roll": r'GimbalRollDegree="?([-+0-9.]+)',
    "relative_altitude": r'RelativeAltitude="?([-+0-9.]+)',
}


@dataclass(frozen=True)
class Capture:
    pose: CameraPose | None
    intrinsics: CameraIntrinsics
    source: str  # exif_full | exif_gps | declared | none
    has_attitude: bool
    warnings: tuple[str, ...] = ()

    @property
    def georeferenced(self) -> bool:
        return self.pose is not None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "georeferenced": self.georeferenced,
            "has_attitude": self.has_attitude,
            "pose": None
            if self.pose is None
            else {
                "lat": self.pose.lat,
                "lon": self.pose.lon,
                "altitude_m": self.pose.altitude_m,
                "yaw_deg": self.pose.yaw_deg,
                "pitch_deg": self.pose.pitch_deg,
                "roll_deg": self.pose.roll_deg,
            },
            "intrinsics": {
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "hfov_deg": self.intrinsics.hfov_deg,
            },
            "warnings": list(self.warnings),
        }


def _dms_to_degrees(dms, ref: str) -> float:
    degrees, minutes, seconds = (float(v) for v in dms)
    value = degrees + minutes / 60.0 + seconds / 3600.0
    return -value if ref in ("S", "W") else value


def _exif_gps(image: Image.Image) -> tuple[float, float, float | None] | None:
    try:
        exif = image.getexif()
        gps = exif.get_ifd(_GPS_IFD)
    except Exception:
        return None
    if not gps:
        return None

    tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
    if "GPSLatitude" not in tags or "GPSLongitude" not in tags:
        return None

    try:
        lat = _dms_to_degrees(tags["GPSLatitude"], tags.get("GPSLatitudeRef", "N"))
        lon = _dms_to_degrees(tags["GPSLongitude"], tags.get("GPSLongitudeRef", "E"))
    except Exception:
        return None

    altitude = None
    if "GPSAltitude" in tags:
        try:
            altitude = float(tags["GPSAltitude"])
        except Exception:
            altitude = None
    return lat, lon, altitude


def _xmp_attitude(raw: bytes) -> dict[str, float]:
    try:
        head = raw[:262_144].decode("latin-1", errors="ignore")
    except Exception:
        return {}
    found = {}
    for key, pattern in _XMP_FIELDS.items():
        match = re.search(pattern, head)
        if match:
            try:
                found[key] = float(match.group(1))
            except ValueError:
                pass
    return found


def read(
    raw: bytes,
    *,
    declared: dict | None = None,
    default_hfov_deg: float = 84.0,
) -> tuple[Image.Image, Capture]:
    """Parse an upload into an image plus the best pose we can justify."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    warnings: list[str] = []

    declared = declared or {}
    hfov = float(declared.get("hfov_deg") or default_hfov_deg)
    intrinsics = CameraIntrinsics(width=width, height=height, hfov_deg=hfov)

    gps = _exif_gps(image)
    attitude = _xmp_attitude(raw)

    if gps and attitude.get("gimbal_pitch") is not None:
        lat, lon, altitude = gps
        height_m = attitude.get("relative_altitude") or altitude or 0.0
        if height_m <= 0:
            warnings.append("Altitude missing from metadata; using declared value.")
            height_m = float(declared.get("altitude_m") or 60.0)
        return image, Capture(
            pose=CameraPose(
                lat=lat,
                lon=lon,
                altitude_m=height_m,
                yaw_deg=attitude.get("gimbal_yaw", 0.0),
                pitch_deg=attitude.get("gimbal_pitch", -90.0),
                roll_deg=attitude.get("gimbal_roll", 0.0),
            ),
            intrinsics=intrinsics,
            source="exif_full",
            has_attitude=True,
            warnings=tuple(warnings),
        )

    if gps:
        lat, lon, altitude = gps
        warnings.append(
            "EXIF has position but no gimbal attitude. Capture geometry below "
            "is declared, so per-object coordinates carry that assumption."
        )
        return image, Capture(
            pose=CameraPose(
                lat=lat,
                lon=lon,
                altitude_m=float(declared.get("altitude_m") or altitude or 60.0),
                yaw_deg=float(declared.get("yaw_deg") or 0.0),
                pitch_deg=float(declared.get("pitch_deg") or -90.0),
                roll_deg=0.0,
            ),
            intrinsics=intrinsics,
            source="exif_gps",
            has_attitude=False,
            warnings=tuple(warnings),
        )

    if {"lat", "lon"} <= declared.keys():
        warnings.append(
            "No positional metadata in this image. Capture pose is declared by "
            "the operator; coordinates are as accurate as that declaration."
        )
        return image, Capture(
            pose=CameraPose(
                lat=float(declared["lat"]),
                lon=float(declared["lon"]),
                altitude_m=float(declared.get("altitude_m") or 60.0),
                yaw_deg=float(declared.get("yaw_deg") or 0.0),
                pitch_deg=float(declared.get("pitch_deg") or -90.0),
                roll_deg=0.0,
            ),
            intrinsics=intrinsics,
            source="declared",
            has_attitude=False,
            warnings=tuple(warnings),
        )

    warnings.append(
        "No capture position available, so detections cannot be placed on a "
        "map and density cannot be computed."
    )
    return image, Capture(
        pose=None,
        intrinsics=intrinsics,
        source="none",
        has_attitude=False,
        warnings=tuple(warnings),
    )
