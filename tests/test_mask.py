"""遮挡轮廓、点位校验、可见性判定测试。"""
import numpy as np
import pytest

from app.errors import ValidationError
from app.mask import MaskProfile, Point, build_point


def test_mask_wraparound_interpolation():
    m = MaskProfile.create(points=[(0, 10.0), (90, 30.0), (180, 20.0)],
                           cutoff_deg=5.0)
    # 控制点处取值
    assert np.rad2deg(m.elevation_at(np.deg2rad([0.0]))[0]) == \
        pytest.approx(10.0)
    assert np.rad2deg(m.elevation_at(np.deg2rad([90.0]))[0]) == \
        pytest.approx(30.0)
    # 45° 线性插值
    assert np.rad2deg(m.elevation_at(np.deg2rad([45.0]))[0]) == \
        pytest.approx(20.0)
    # 环绕段（270° 位于 180° 与 360°/0° 之间）：(20+10)/2=15
    assert np.rad2deg(m.elevation_at(np.deg2rad([270.0]))[0]) == \
        pytest.approx(15.0)


def test_mask_cutoff_is_floor():
    # 控制点给出低于统一截止角的遮挡 -> 取截止角
    m = MaskProfile.create(points=[(0, 2.0), (180, 1.0)], cutoff_deg=10.0)
    vals = np.rad2deg(m.elevation_at(np.deg2rad([0.0, 90.0, 180.0])))
    assert np.all(vals >= 10.0 - 1e-9)


def test_empty_mask_uses_cutoff():
    m = MaskProfile.create(cutoff_deg=7.0)
    vals = m.elevation_at(np.deg2rad(np.linspace(0, 359, 36)))
    assert np.allclose(np.rad2deg(vals), 7.0)


def test_reject_azimuth_out_of_range():
    with pytest.raises(ValidationError):
        MaskProfile.create(points=[(361.0, 10.0)])
    with pytest.raises(ValidationError):
        MaskProfile.create(points=[(-1.0, 10.0)])


def test_reject_duplicate_azimuth():
    with pytest.raises(ValidationError):
        MaskProfile.create(points=[(45.0, 10.0), (45.0, 20.0)])
    # 0 与 360 等价，也算重复
    with pytest.raises(ValidationError):
        MaskProfile.create(points=[(0.0, 10.0), (360.0, 20.0)])


def test_reject_latitude_out_of_range():
    with pytest.raises(ValidationError):
        build_point({"point_id": "P", "lon_deg": 116.0, "lat_deg": 90.5})
    with pytest.raises(ValidationError):
        build_point({"point_id": "P", "lon_deg": 116.0, "lat_deg": -91.0})


def test_mask_monotone_raises_elevation():
    # 整体抬高轮廓：每个方位的遮挡高度角都不小于原轮廓
    low = MaskProfile.create(points=[(0, 5), (120, 8), (240, 6)],
                             cutoff_deg=5.0)
    high = MaskProfile.create(points=[(0, 20), (120, 22), (240, 21)],
                              cutoff_deg=15.0)
    az = np.deg2rad(np.linspace(0, 359.9, 720))
    assert np.all(high.elevation_at(az) >= low.elevation_at(az) - 1e-12)


def test_point_serialization_roundtrip():
    p = build_point({"point_id": "P9", "lon_deg": 10.0, "lat_deg": 20.0,
                     "height_m": 3.0,
                     "mask": {"cutoff_deg": 8.0,
                              "points": [[10, 12], [200, 18]]}})
    q = Point.from_dict(p.to_dict())
    assert q.point_id == "P9"
    assert np.allclose(q.mask.az, p.mask.az)
    assert np.allclose(q.mask.el, p.mask.el)
