"""可见性、遮挡判定与单时刻/批量精度因子评估。

VisibilityEngine 绑定一版历书与一个点位版本：
  * 单颗卫星在整数秒上的方位角/高度角经 SharedSkyCache 共享，
    同一点位被多份规划引用时不会重复计算同一时刻；
  * states_at 批量返回每个时刻的可见性与各项 DOP，供时段搜索使用。
可见判据严格按需求：卫星高度角 **严格高于** 该方向遮挡高度角才可见。
"""
from __future__ import annotations

import numpy as np

from .coords import geodetic_to_ecef, ecef_to_enu, enu_to_az_el
from .dop import Dop, dop_from_unit_vectors
from .orbit import Almanac, sat_position
from .mask import Point
from .cache import SharedSkyCache


class VisibilityEngine:
    def __init__(self, almanac: Almanac, point: Point,
                 cache: SharedSkyCache | None = None):
        self.almanac = almanac
        self.point = point
        # 不传缓存时给一个可用的默认 LRU；容量 0 会导致写入即被淘汰。
        self.cache = cache if cache is not None else SharedSkyCache()
        self._p_ecef = geodetic_to_ecef(point.lon_deg, point.lat_deg,
                                        point.height_m)

    # ---------------------------------------------------------------- 几何
    def _az_el_raw(self, sat, times) -> tuple[np.ndarray, np.ndarray]:
        """不查缓存，直接批量算某颗卫星在 times 上的 az/el（弧度）。

        始终返回一维列向量 (T,)；标量输入返回 shape=(1,)。
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        sat_ecef = sat_position(sat, t)               # (T,3)
        # 测站坐标必须显式广播到 (T,3)：直接用 (3,) 与 (T,3) 相减会被
        # NumPy 沿最后一维错误对齐（仅当 T=3 时侥幸成立）。
        p = np.broadcast_to(np.reshape(self._p_ecef, (1, 3)),
                            sat_ecef.shape)
        e, n, u = ecef_to_enu(p, self.point.lon_deg, self.point.lat_deg,
                              sat_ecef)
        az, el = enu_to_az_el(np.atleast_1d(e), np.atleast_1d(n),
                              np.atleast_1d(u))
        return np.asarray(az, dtype=float).reshape(-1), \
            np.asarray(el, dtype=float).reshape(-1)

    def sat_az_el(self, sat_id: str, times) -> tuple[np.ndarray, np.ndarray]:
        """整数秒走共享缓存（在途键合并，绝不重复计算）；含非整数秒时直算。"""
        sat = self.almanac.get(sat_id)
        t = np.asarray(times, dtype=float)
        scalar = t.ndim == 0
        t1 = np.atleast_1d(t)
        int_like = np.all(np.abs(t1 - np.round(t1)) < 1e-9)
        seconds = np.round(t1).astype(np.int64)
        if int_like:
            def _computer(secs, _sat=sat):
                a_z, e_l = self._az_el_raw(_sat, secs.astype(float))
                # 必须 column_stack：np.asarray((az,el)) 会得到 (2,T)
                # 并在 reshape(-1,2) 后把 az/el 错位
                return np.column_stack([a_z, e_l])
            az_el = self.cache.get_or_compute(
                self.almanac.version, self.point.version, sat_id, seconds,
                computer=_computer)
            az, el = az_el[:, 0], az_el[:, 1]
        else:
            az, el = self._az_el_raw(sat, t1)
        if scalar:
            return az[0], el[0]
        return az, el

    # ------------------------------------------------------------- 批量评估
    def all_az_el(self, times) -> tuple[np.ndarray, np.ndarray]:
        """返回所有卫星在 times 上的方位角/高度角，shape=(T,K)。"""
        t = np.asarray(times, dtype=float)
        scalar = t.ndim == 0
        t1 = np.atleast_1d(t)
        k = len(self.almanac.sats)
        az = np.empty((t1.shape[0], k), dtype=float)
        el = np.empty_like(az)
        for j, sat in enumerate(self.almanac.sats):
            a, e = self.sat_az_el(sat.sat_id, t1)
            az[:, j] = a
            el[:, j] = e
        if scalar:
            return az[0], el[0]
        return az, el

    def states_at(self, times):
        """批量评估。

        返回 (visible, dops, n_visible)：
            visible   bool 数组 (T,K)
            dops      ndarray (T,5)，列依次 gdop/pdop/hdop/vdop/tdop；
                      不可用时为 inf
            n_visible (T,) 可见卫星数
        """
        t = np.asarray(times, dtype=float)
        scalar = t.ndim == 0
        t1 = np.atleast_1d(t)
        az, el = self.all_az_el(t1)
        mask_el = self.point.mask.elevation_at(az)     # (T,K)
        visible = el > mask_el
        # 单位视线（ENU）
        units = np.zeros((t1.shape[0], visible.shape[1], 3), dtype=float)
        cos_el = np.cos(el)
        units[..., 0] = cos_el * np.sin(az)
        units[..., 1] = cos_el * np.cos(az)
        units[..., 2] = np.sin(el)
        units[~visible] = 0.0
        dops = np.full((t1.shape[0], 5), np.inf)
        n_vis = visible.sum(axis=1)
        for i in range(t1.shape[0]):
            idx = np.where(visible[i])[0]
            if idx.size >= 4:
                d = dop_from_unit_vectors(units[i, idx, :])
                dops[i] = (d.gdop, d.pdop, d.hdop, d.vdop, d.tdop)
        if scalar:
            return visible[0], dops[0], int(n_vis[0])
        return visible, dops, n_vis

    def state_at(self, t: float):
        """单时刻评估，返回 (可用: bool, gdop: float, n_visible, Dop, 明细)。"""
        az, el = self.all_az_el(np.asarray(float(t)))
        mask_el = self.point.mask.elevation_at(az)
        visible = el > mask_el
        idx = np.where(visible)[0]
        details = []
        for j, sat in enumerate(self.almanac.sats):
            details.append({
                "sat_id": sat.sat_id,
                "azimuth_deg": float(np.rad2deg(az[j])),
                "elevation_deg": float(np.rad2deg(el[j])),
                "mask_elevation_deg": float(np.rad2deg(mask_el[j])),
                "visible": bool(visible[j]),
            })
        if idx.size >= 4:
            u = np.column_stack([
                np.cos(el[idx]) * np.sin(az[idx]),
                np.cos(el[idx]) * np.cos(az[idx]),
                np.sin(el[idx]),
            ])
            d = dop_from_unit_vectors(u)
            ok = bool(np.isfinite(d.gdop))
            return ok, (d.gdop if ok else np.inf), int(idx.size), d, details
        d = Dop(np.inf, np.inf, np.inf, np.inf, np.inf, int(idx.size))
        return False, np.inf, int(idx.size), d, details
