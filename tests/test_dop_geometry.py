"""几何核对例子（不经轨道计算，直接给方向）。"""
import math

import numpy as np
import pytest

from app.dop import dop_from_az_el


def _geometry_matrix(az, el, zenith=True):
    rows = []
    for a, e in zip(az, el):
        u = np.array([math.cos(e) * math.sin(a),
                      math.cos(e) * math.cos(a),
                      math.sin(e)])
        rows.append([-u[0], -u[1], -u[2], 1.0])
    if zenith:
        rows.append([0.0, 0.0, -1.0, 1.0])
    return np.array(rows)


def test_textbook_geometry_gdop_3073():
    # 一颗在天顶，三颗高度角 30°，方位角 0°/120°/240°
    az = np.deg2rad([0.0, 120.0, 240.0])
    el = np.deg2rad([30.0, 30.0, 30.0])
    # dop_from_az_el 直接接收全部 4 颗（第 4 颗在天顶）
    az4 = np.append(az, 0.0)
    el4 = np.append(el, math.pi / 2.0)   # 天顶：高度角 90°（弧度）
    d = dop_from_az_el(az4, el4)
    # 题目给的核对值：GDOP ≈ 3.073
    assert d.n_sat == 4
    assert d.gdop == pytest.approx(3.073, abs=2e-3)

    # 独立用矩阵求逆核对
    g = _geometry_matrix(az, el, zenith=True)
    q = np.linalg.inv(g.T @ g)
    assert math.sqrt(np.trace(q)) == pytest.approx(d.gdop, rel=1e-12)


def test_exact_manual_values():
    # 精确值（Q 矩阵直接验证）：
    # HDOP²=16/9, VDOP²=16/3, TDOP²=7/3, PDOP=8/3, GDOP=√85/3
    az = np.deg2rad([0.0, 120.0, 240.0, 0.0])
    el = np.deg2rad([30.0, 30.0, 30.0, 90.0])
    d = dop_from_az_el(az, el)
    assert d.hdop == pytest.approx(4.0 / 3.0, abs=1e-9)
    assert d.vdop == pytest.approx(4.0 / math.sqrt(3.0), abs=1e-9)
    assert d.pdop == pytest.approx(8.0 / 3.0, abs=1e-9)
    assert d.tdop == pytest.approx(math.sqrt(7.0 / 3.0), abs=1e-9)
    assert d.gdop == pytest.approx(math.sqrt(85.0 / 9.0), abs=1e-9)


def test_fewer_than_four_unavailable():
    az = np.deg2rad([0.0, 120.0, 240.0])
    el = np.deg2rad([30.0, 30.0, 30.0])
    d = dop_from_az_el(az, el)
    assert d.n_sat == 3
    assert not np.isfinite(d.gdop)

    d0 = dop_from_az_el(np.array([]), np.array([]))
    assert d0.n_sat == 0
    assert not np.isfinite(d0.gdop)
