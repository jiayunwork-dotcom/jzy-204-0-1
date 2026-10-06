"""平方和关系与加星单调性。"""
import math

import numpy as np
import pytest

from app.dop import dop_from_az_el


def test_sum_of_squares_relations():
    # GDOP² = PDOP² + TDOP²；PDOP² = HDOP² + VDOP²
    rng = np.random.default_rng(42)
    for n in [4, 6, 8, 12]:
        az = rng.uniform(0, 2 * np.pi, size=n)
        el = np.deg2rad(rng.uniform(8, 88, size=n))
        d = dop_from_az_el(az, el)
        assert d.gdop ** 2 == pytest.approx(d.pdop ** 2 + d.tdop ** 2,
                                            rel=1e-10)
        assert d.pdop ** 2 == pytest.approx(d.hdop ** 2 + d.vdop ** 2,
                                            rel=1e-10)


def test_adding_satellite_never_increases_gdop():
    # 同一时刻额外增加一颗可见卫星，GDOP 不会变大
    rng = np.random.default_rng(7)
    for trial in range(50):
        n = rng.integers(4, 10)
        az = rng.uniform(0, 2 * np.pi, size=n)
        el = np.deg2rad(rng.uniform(10, 85, size=n))
        d_before = dop_from_az_el(az, el)
        extra_az = rng.uniform(0, 2 * np.pi)
        extra_el = np.deg2rad(rng.uniform(10, 85))
        d_after = dop_from_az_el(np.append(az, extra_az),
                                 np.append(el, extra_el))
        assert np.isfinite(d_before.gdop)
        assert d_after.gdop <= d_before.gdop + 1e-12
        # 其余 DOP 同样单调
        assert d_after.pdop <= d_before.pdop + 1e-12
        assert d_after.tdop <= d_before.tdop + 1e-12


def test_textbook_geometry_sum_relation():
    az = np.deg2rad([0.0, 120.0, 240.0, 0.0])
    el = np.deg2rad([30.0, 30.0, 30.0, 90.0])
    d = dop_from_az_el(az, el)
    # GDOP² = 85/9（与 3.073 的核对值一致）
    assert d.gdop ** 2 == pytest.approx(85.0 / 9.0, abs=1e-9)
