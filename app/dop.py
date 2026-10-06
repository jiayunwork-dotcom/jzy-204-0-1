"""精度因子（DOP）计算。

给定可见卫星相对点位的单位视线（ENU），几何矩阵
        G = [ -e_x -e_y -e_z  1 ]  每颗可见卫星一行
视线分量取自 ENU 局部系（天顶方向 u=1）。
        Q = (GᵀG)⁻¹
        GDOP² = trace(Q)
        PDOP² = Q00 + Q11 + Q22
        TDOP² = Q33
        HDOP² = Q00 + Q11   （东、北两水平分量）
        VDOP² = Q22         （垂直分量）
可见卫星少于 4 颗，或矩阵奇异时，结果不可用（gdop=+inf）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dop:
    gdop: float
    pdop: float
    hdop: float
    vdop: float
    tdop: float
    n_sat: int

    def to_dict(self) -> dict:
        finite = bool(np.isfinite(self.gdop))
        return {
            "available": finite,
            "n_sat": int(self.n_sat),
            "gdop": float(self.gdop) if finite else None,
            "pdop": float(self.pdop) if finite else None,
            "hdop": float(self.hdop) if finite else None,
            "vdop": float(self.vdop) if finite else None,
            "tdop": float(self.tdop) if finite else None,
        }


_UNAVAILABLE_CACHE = {}


def unavailable(n_sat: int) -> Dop:
    d = _UNAVAILABLE_CACHE.get(n_sat)
    if d is None:
        d = Dop(np.inf, np.inf, np.inf, np.inf, np.inf, n_sat)
        _UNAVAILABLE_CACHE[n_sat] = d
    return d


def dop_from_unit_vectors(units, rcond: float = 1e-12) -> Dop:
    """由单位视线数组 (...,3)（ENU 分量）计算 DOP。

    支持批量：输入 shape=(T,K,3)，返回 Dop 数组属性 shape=(T,)；
    标量场景输入 shape=(K,3)。为简单与可核对，逐时刻用小矩阵求逆。
    """
    arr = np.asarray(units, dtype=float)
    batched = arr.ndim == 3
    frames = arr if batched else arr[None, :, :]
    t_count = frames.shape[0]
    out = np.full((t_count, 5), np.inf)
    n_count = np.zeros(t_count, dtype=int)
    for ti in range(t_count):
        frame = frames[ti]
        k = frame.shape[0]
        n_count[ti] = k
        if k < 4:
            continue
        g = np.empty((k, 4), dtype=float)
        g[:, 0] = -frame[:, 0]
        g[:, 1] = -frame[:, 1]
        g[:, 2] = -frame[:, 2]
        g[:, 3] = 1.0
        gt_g = g.T @ g
        # 条件数保护：正常几何下最小特征值远大于此
        try:
            q = np.linalg.inv(gt_g)
        except np.linalg.LinAlgError:
            continue
        diag = np.diag(q)
        if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
            # 数值上不是合法协方差（几何退化）
            continue
        pdop2 = diag[0] + diag[1] + diag[2]
        hdop2 = diag[0] + diag[1]
        vdop2 = diag[2]
        tdop2 = diag[3]
        gdop2 = float(np.sum(diag))
        if gdop2 <= 0.0:
            continue
        out[ti] = (np.sqrt(gdop2), np.sqrt(pdop2), np.sqrt(hdop2),
                   np.sqrt(vdop2), np.sqrt(tdop2))
    if not batched:
        idx = 0
        return Dop(*(float(v) for v in out[idx]), int(n_count[idx]))
    return out, n_count


def dop_from_az_el(az_rad, el_rad) -> Dop:
    """核对用几何入口：直接给方位角/高度角（弧度），无需轨道计算。"""
    az = np.asarray(az_rad, dtype=float)
    el = np.asarray(el_rad, dtype=float)
    u = np.stack([np.cos(el) * np.sin(az),   # 东
                  np.cos(el) * np.cos(az),   # 北
                  np.sin(el)], axis=-1)      # 天顶
    return dop_from_unit_vectors(u)
