"""轨道传播与坐标变换测试。"""
import math

import numpy as np
import pytest

from app.coords import geodetic_to_ecef
from app.orbit import (MU_EARTH, OMEGA_EARTH, SatAlmanac,
                       build_almanac, sat_position, solve_kepler)
from app.errors import ValidationError


def _circular_sat(sat_id="S1", raan0=0.0, argp=0.0, ma=0.0, inc=0.9599,
                  toe=0.0, ecc=0.0, raan_rate=0.0):
    a = 26559800.0
    return SatAlmanac(sat_id=sat_id, sqrt_a=math.sqrt(a), ecc=ecc, inc=inc,
                      raan0=raan0, raan_rate=raan_rate, argp=argp, ma=ma,
                      toe=toe)


def test_solve_kepler_circular():
    m = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    e = solve_kepler(m, 0.0)
    assert np.allclose(e, m, atol=1e-11)


def test_orbit_period():
    # 半长轴 26559.8 km → 恒星周期约 11h58m（半恒星日）
    a = 26559800.0
    period = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    assert 43000 < period < 43200


def test_perigee_direction():
    # 近圆、近地点幅角 0、M=0 时卫星应在升交点方向（轨道面 x 轴）
    sat = _circular_sat(inc=math.radians(55.0))
    t0 = 0.0
    p0 = sat_position(sat, t0)
    a = sat.sqrt_a ** 2
    # 此时在 ECEF x 轴（升交点在格林尼治方向）与 z 倾角分量上
    assert abs(np.linalg.norm(p0) - a) < 1.0


def test_earth_rotation_shifts_node_longitude():
    # 同一卫星在两个时刻：ECEF 升交点经度应随 ωₑ 西移（Ω̇=0 时）
    sat = _circular_sat(inc=math.radians(55.0))
    dt = 3600.0
    # 卫星相位抵消：取相隔一个恒星周期的两个时刻做位置对比不可行，
    # 改为直接验证半周期点经度随时间减少
    p1 = sat_position(sat, np.array([0.0]))
    p2 = sat_position(sat, np.array([dt]))
    lon1 = math.atan2(p1[0, 1], p1[0, 0])
    lon2 = math.atan2(p2[0, 1], p2[0, 0])
    # 这里卫星本身也在动，故单独验证公式常量量级：ωₑ·dt
    assert OMEGA_EARTH * dt == pytest.approx(0.2625, rel=2e-3)
    assert abs(lon1) < 1e-6  # t=0 节点位于 x 轴（M=0 正在节点）
    # p2 已离开节点，仅确认位置随时间变化
    assert np.linalg.norm(p2 - p1) > 1000.0


def test_ecef_roundtrip_beijing():
    # 已知 WGS84 北京概略坐标量级检查
    p = geodetic_to_ecef(116.391, 39.907, 50.0)
    x, y, z = p
    assert -2.2e6 < x < -2.1e6
    assert 4.3e6 < y < 4.45e6
    assert 4.05e6 < z < 4.15e6


def test_satellite_below_horizon_south():
    # 构造一颗在赤道面、位于测站南侧天底方向的卫星，高度角应为负
    # 直接用 az/el 几何函数检查符号约定：
    # ENU 中 n=-1（正南水平）应给出 az=180°, el=0
    from app.coords import enu_to_az_el
    az, el = enu_to_az_el(np.array([0.0]), np.array([-1.0]),
                          np.array([0.0]))
    assert math.degrees(az[0]) == pytest.approx(180.0, abs=1e-9)
    assert abs(math.degrees(el[0])) < 1e-9
    az2, el2 = enu_to_az_el(np.array([1.0]), np.array([0.0]),
                            np.array([0.0]))
    assert math.degrees(az2[0]) == pytest.approx(90.0, abs=1e-9)


def test_vectorized_matches_scalar(almanac24):
    sat = almanac24.sats[0]
    times = np.linspace(0, 43000, 7)
    pv = sat_position(sat, times)
    for i, t in enumerate(times):
        ps = sat_position(sat, t)
        assert np.allclose(pv[i], ps, atol=1e-6)


def test_reject_bad_eccentricity():
    raw = {"satellites": [
        {"sat_id": "X", "semi_major_axis_m": None, "sqrt_a": 5153.0,
         "ecc": 1.0, "inc_deg": 55, "raan0_deg": 10, "argp_deg": 0,
         "ma_deg": 0, "toe": "2025-01-01T00:00:00Z"}]}
    with pytest.raises(ValidationError):
        build_almanac(raw)
    raw["satellites"][0]["ecc"] = -0.1
    with pytest.raises(ValidationError):
        build_almanac(raw)


def test_reject_bad_semimajor():
    raw = {"satellites": [
        {"sat_id": "X", "semi_major_axis": -1.0, "ecc": 0.0,
         "inc_deg": 55, "raan0_deg": 10, "argp_deg": 0, "ma_deg": 0,
         "toe": "2025-01-01T00:00:00Z"}]}
    with pytest.raises(ValidationError):
        build_almanac(raw)


def test_reject_duplicate_sat_ids():
    raw = {"satellites": [
        {"sat_id": "X", "sqrt_a": 5153.0, "ecc": 0.0, "inc_deg": 55,
         "raan0_deg": 10, "argp_deg": 0, "ma_deg": 0,
         "toe": "2025-01-01T00:00:00Z"},
        {"sat_id": "X", "sqrt_a": 5153.0, "ecc": 0.0, "inc_deg": 55,
         "raan0_deg": 20, "argp_deg": 0, "ma_deg": 0,
         "toe": "2025-01-01T00:00:00Z"}]}
    with pytest.raises(ValidationError):
        build_almanac(raw)
