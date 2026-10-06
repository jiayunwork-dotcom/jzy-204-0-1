"""WGS84 坐标变换：大地坐标 ↔ ECEF，ECEF → ENU/高度角方位角。"""
from __future__ import annotations

import numpy as np

# WGS84 椭球参数
WGS84_A = 6378137.0                # 长半轴 [m]
WGS84_F = 1.0 / 298.257223563      # 扁率
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lon_deg, lat_deg, h):
    """大地经纬度（度）+ 大地高（米）→ ECEF（米）。支持数组。"""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    # 卯酉圈曲率半径
    rn = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (rn + h) * cos_lat * np.cos(lon)
    y = (rn + h) * cos_lat * np.sin(lon)
    z = (rn * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack([np.asarray(x, dtype=float),
                     np.asarray(y, dtype=float),
                     np.asarray(z, dtype=float)], axis=-1)


def _enu_axes(lon_deg, lat_deg):
    """返回 ENU 基向量（以地心系原点视角）：east, north, up，形状各 (...,3)。"""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    up = np.stack([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)], axis=-1)
    east = np.stack([-np.sin(lon), np.cos(lon),
                     np.zeros_like(np.asarray(lon))], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon),
                      -np.sin(lat) * np.sin(lon),
                      np.cos(lat)], axis=-1)
    return east, north, up


def ecef_to_enu(point_ecef, point_lon_deg, point_lat_deg, sat_ecef):
    """卫星相对点位的视线转成 ENU 分量。

    point_ecef/sat_ecef 最后一维为 3；时间批量时形状 (T,3)。
    基向量显式扩展为可广播形状，避免 (3,) 与 (T,3) 在 T=3 时的
    维度歧义。
    """
    east, north, up = _enu_axes(point_lon_deg, point_lat_deg)
    los = np.asarray(sat_ecef) - np.asarray(point_ecef)
    # 在分量轴前补轴，使 east/north/up 与 los 的前导维正确广播
    east = np.expand_dims(east, axis=tuple(range(los.ndim - 1)))
    north = np.expand_dims(north, axis=tuple(range(los.ndim - 1)))
    up = np.expand_dims(up, axis=tuple(range(los.ndim - 1)))
    e = np.sum(los * east, axis=-1)
    n = np.sum(los * north, axis=-1)
    u = np.sum(los * up, axis=-1)
    return e, n, u


def enu_to_az_el(e, n, u):
    """ENU 分量 → 方位角（自正北顺时针，0~2π）、高度角（弧度）。"""
    horiz = np.hypot(e, n)
    el = np.arctan2(u, horiz)
    az = np.arctan2(e, n)          # 北偏东
    az = az % (2.0 * np.pi)
    return az, el


def sat_az_el(point_lon_deg, point_lat_deg, point_h, sat_ecef):
    """一步完成：点位大地坐标 + 卫星 ECEF → (方位角弧度, 高度角弧度, 距离米)。"""
    p = geodetic_to_ecef(point_lon_deg, point_lat_deg, point_h)
    e, n, u = ecef_to_enu(p, point_lon_deg, point_lat_deg, sat_ecef)
    az, el = enu_to_az_el(e, n, u)
    rng = np.hypot(horiz := np.hypot(e, n), u)
    return az, el, rng
