import pytest

from gnss_service.models import (
    ValidationError,
    normalize_almanac,
    validate_plan_params,
    validate_point_payload,
)
from gnss_service.timeutils import validate_time_range


BASE = "2026-10-06T00:00:00Z"


def satellite(**overrides):
    data = {
        "id": "S1",
        "semi_major_axis": 26559800.0,
        "eccentricity": 0.01,
        "inclination": 55,
        "raan": 10,
        "raan_rate": 0,
        "argument_of_perigee": 20,
        "mean_anomaly": 30,
        "reference_time": BASE,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.0, 1.2],
)
def test_reject_bad_eccentricity(value):
    with pytest.raises(ValidationError):
        normalize_almanac({"satellites": [satellite(eccentricity=value)]})


def test_reject_non_positive_semi_major_axis():
    with pytest.raises(ValidationError):
        normalize_almanac({"satellites": [satellite(semi_major_axis=0)]})
    with pytest.raises(ValidationError):
        normalize_almanac({"satellites": [satellite(semi_major_axis=-1)]})


def test_reject_latitude_out_of_bounds():
    with pytest.raises(ValidationError):
        validate_point_payload({"latitude": 90.1, "longitude": 0})
    with pytest.raises(ValidationError):
        validate_point_payload({"latitude": -90.1, "longitude": 0})


def test_reject_invalid_or_duplicate_mask_azimuths():
    with pytest.raises(ValidationError):
        validate_point_payload({"latitude": 0, "longitude": 0, "mask_profile": [[-1, 5]]})
    with pytest.raises(ValidationError):
        validate_point_payload({"latitude": 0, "longitude": 0, "mask_profile": [[361, 5]]})
    with pytest.raises(ValidationError):
        validate_point_payload({"latitude": 0, "longitude": 0, "mask_profile": [[10, 5], [10, 8]]})
    with pytest.raises(ValidationError):
        validate_point_payload({"latitude": 0, "longitude": 0, "mask_profile": [[0, 5], [360, 8]]})


def test_reject_bad_plan_parameters():
    with pytest.raises(ValidationError):
        validate_plan_params({"gdop_threshold": 0, "min_duration_seconds": 1})
    with pytest.raises(ValidationError):
        validate_plan_params({"gdop_threshold": -1, "min_duration_seconds": 1})
    with pytest.raises(ValidationError):
        validate_plan_params({"gdop_threshold": 3, "min_duration_seconds": -1})


def test_reject_inverted_time_range():
    with pytest.raises(ValueError):
        validate_time_range("2026-10-07T00:00:00Z", "2026-10-06T00:00:00Z")
