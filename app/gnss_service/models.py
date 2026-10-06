import math
from typing import Mapping

from .constants import SECONDS_PER_DAY
from .timeutils import parse_time


class ValidationError(ValueError):
    """Raised when an API or domain object fails validation."""


def require_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite")
    return value


def validate_degree(value: float, name: str, low: float, high: float) -> float:
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be in [{low}, {high}] degrees")
    return value


def _mask_pairs(profile) -> list[tuple[float, float]]:
    if profile is None:
        return []
    if not isinstance(profile, list):
        raise ValidationError("mask profile must be a list of [azimuth, elevation] pairs")
    pairs: list[tuple[float, float]] = []
    for index, point in enumerate(profile):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValidationError(f"mask point {index} must be [azimuth, elevation]")
        azimuth = require_number(point[0], f"mask azimuth {index}")
        elevation = require_number(point[1], f"mask elevation {index}")
        validate_degree(azimuth, f"mask azimuth {index}", 0, 360)
        validate_degree(elevation, f"mask elevation {index}", -90, 90)
        # 0 and 360 are the same azimuth. Keeping both would be a duplicate.
        if math.isclose(azimuth, 360.0, abs_tol=1e-12):
            azimuth = 0.0
        pairs.append((azimuth, elevation))
    if any(
        math.isclose(left[0], right[0], abs_tol=1e-12)
        for i, left in enumerate(pairs)
        for right in pairs[i + 1 :]
    ):
        raise ValidationError("mask profile contains duplicate azimuths")
    return sorted(pairs, key=lambda item: item[0])


def validate_satellite(raw: Mapping, index: int) -> dict:
    if not isinstance(raw, Mapping):
        raise ValidationError(f"satellite {index} must be an object")
    sat_id = str(raw.get("id", f"S{index + 1:02d}"))

    eccentricity = require_number(raw.get("eccentricity"), f"satellite {index} eccentricity")
    if not (0.0 <= eccentricity < 1.0):
        raise ValidationError(f"satellite {index} eccentricity must be in [0, 1)")

    has_a = raw.get("semi_major_axis") is not None
    has_sqrt_a = raw.get("sqrt_semi_major_axis") is not None
    if has_a == has_sqrt_a:
        raise ValidationError(
            f"satellite {index} must provide exactly one of semi_major_axis or sqrt_semi_major_axis"
        )
    if has_a:
        semi_major_axis = require_number(raw.get("semi_major_axis"), f"satellite {index} semi-major axis")
    else:
        sqrt_a = require_number(raw.get("sqrt_semi_major_axis"), f"satellite {index} sqrt semi-major axis")
        if sqrt_a <= 0:
            raise ValidationError(f"satellite {index} semi-major axis must be positive")
        semi_major_axis = sqrt_a * sqrt_a
    if semi_major_axis <= 0:
        raise ValidationError(f"satellite {index} semi-major axis must be positive")

    inclination = require_number(raw.get("inclination"), f"satellite {index} inclination")
    raan = require_number(raw.get("raan"), f"satellite {index} RAAN")
    raan_rate = require_number(raw.get("raan_rate", 0.0), f"satellite {index} RAAN rate")
    argument_of_perigee = require_number(
        raw.get("argument_of_perigee"), f"satellite {index} argument of perigee"
    )
    mean_anomaly = require_number(raw.get("mean_anomaly"), f"satellite {index} mean anomaly")
    reference_time = raw.get("reference_time")
    if not isinstance(reference_time, str) or not reference_time:
        raise ValidationError(f"satellite {index} reference_time must be an ISO timestamp")
    try:
        parse_time(reference_time)
    except (TypeError, ValueError):
        raise ValidationError(f"satellite {index} reference_time must be an ISO timestamp")

    return {
        "id": sat_id,
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "inclination": math.radians(inclination),
        "raan": math.radians(raan),
        "raan_rate": math.radians(raan_rate) / SECONDS_PER_DAY,
        "argument_of_perigee": math.radians(argument_of_perigee),
        "mean_anomaly": math.radians(mean_anomaly),
        "reference_time": reference_time,
    }


def validate_point_payload(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ValidationError("point payload must be an object")
    name = str(payload.get("name", payload.get("id", "point")))
    latitude = require_number(payload.get("latitude"), "latitude")
    longitude = require_number(payload.get("longitude"), "longitude")
    height = require_number(payload.get("height", 0.0), "height")
    validate_degree(latitude, "latitude", -90, 90)
    validate_degree(longitude, "longitude", -180, 180)
    default_mask = require_number(payload.get("default_elevation_mask", 5.0), "default elevation mask")
    validate_degree(default_mask, "default elevation mask", -90, 90)
    profile = _mask_pairs(payload.get("mask_profile", []))
    return {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "height": height,
        "default_elevation_mask": default_mask,
        "mask_profile": profile,
    }


def validate_plan_params(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ValidationError("planning parameters must be an object")
    threshold = require_number(payload.get("gdop_threshold"), "GDOP threshold")
    if threshold <= 0:
        raise ValidationError("GDOP threshold must be positive")
    min_duration = require_number(payload.get("min_duration_seconds"), "minimum duration")
    if min_duration < 0:
        raise ValidationError("minimum duration must not be negative")
    step = payload.get("step_seconds")
    if step is None:
        step = 10.0
    step = require_number(step, "step seconds")
    if not (0 < step <= 3600):
        raise ValidationError("step seconds must be in (0, 3600]")
    if step > min_duration:
        # The search guarantee is that an undetected interval is shorter than step.
        raise ValidationError("step_seconds must not exceed min_duration_seconds")
    return {"gdop_threshold": threshold, "min_duration_seconds": min_duration, "step_seconds": step}


def normalize_almanac(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ValidationError("almanac payload must be an object")
    satellites_raw = payload.get("satellites")
    if not isinstance(satellites_raw, list) or not satellites_raw:
        raise ValidationError("almanac must contain at least one satellite")
    satellites = [validate_satellite(item, i) for i, item in enumerate(satellites_raw)]
    ids = [sat["id"] for sat in satellites]
    if len(ids) != len(set(ids)):
        raise ValidationError("satellite ids must be unique")
    return {"name": str(payload.get("name", "almanac")), "satellites": satellites}
