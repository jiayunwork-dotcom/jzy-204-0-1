from datetime import timedelta

import pytest

from conftest import at
from gnss_service.search import find_windows
from gnss_service.visibility import MaskProfile


def test_raising_mask_never_extends_windows():
    profile = [(0, 0), (90, 0), (180, 0), (270, 0)]
    lower = MaskProfile(0.0, tuple((a, float(e)) for a, e in profile))
    higher = lower.raised(15.0)
    # A satellite moves in azimuth and briefly clears the low mask. Raising
    # every obstruction by 15 degrees removes it completely.
    def lower_ok(t):
        second = int((t - at(0)).total_seconds())
        azimuth = (second % 360)
        elevation = 10.0
        return elevation >= lower.elevation_at(azimuth)

    def higher_ok(t):
        second = int((t - at(0)).total_seconds())
        azimuth = (second % 360)
        elevation = 10.0
        return elevation >= higher.elevation_at(azimuth)

    low_windows = find_windows(at(0), at(360), lower_ok, lambda t: 1, 3, 0, step_seconds=1)
    high_windows = find_windows(at(0), at(360), higher_ok, lambda t: 1, 3, 0, step_seconds=1)
    assert low_windows
    assert not high_windows
