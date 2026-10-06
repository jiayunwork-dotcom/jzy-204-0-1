"""点位与遮挡轮廓（按方位角给出的遮挡高度角曲线）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ValidationError

DEFAULT_CUTOFF_DEG = 5.0


@dataclass(frozen=True)
class MaskProfile:
    """遮挡轮廓。

    az_deg/el_deg 为一组有序控制点（方位角、该方向遮挡高度角，度）。
    方位角必须落在 [0,360) 且不重复；构造时自动排序。
    未给出控制点的方向用统一截止高度角 cutoff_deg。

    采用环形线性插值；当某点两侧存在控制点时，若插值结果低于统一截止角，
    取两者较高者（截止角是任何方向的最低遮挡门限）。
    """

    az: np.ndarray   # 弧度，[0,2π)，已排序
    el: np.ndarray   # 弧度
    cutoff: float    # 弧度

    @staticmethod
    def create(points=None, cutoff_deg=DEFAULT_CUTOFF_DEG) -> "MaskProfile":
        if not np.isfinite(cutoff_deg):
            raise ValidationError("截止高度角必须为有限值", field="cutoff_deg")
        points = list(points or [])
        azs, els = [], []
        seen = set()
        for item in points:
            az = float(item[0])
            el = float(item[1])
            if not np.isfinite(az) or not (0.0 <= az <= 360.0):
                raise ValidationError(
                    f"遮挡轮廓方位角必须在 [0,360] 度内，收到 {az}",
                    field="mask")
            # 360 与 0 等价，视为重复
            key = 0.0 if az == 360.0 else az
            if key in seen:
                raise ValidationError(
                    f"遮挡轮廓方位角重复: {az}", field="mask")
            seen.add(key)
            if not np.isfinite(el):
                raise ValidationError("遮挡高度角必须为有限值", field="mask")
            azs.append(np.deg2rad(key))
            els.append(np.deg2rad(el))
        if azs:
            order = np.argsort(azs)
            az_arr = np.asarray(azs, dtype=float)[order]
            el_arr = np.asarray(els, dtype=float)[order]
        else:
            az_arr = np.array([], dtype=float)
            el_arr = np.array([], dtype=float)
        return MaskProfile(az=az_arr, el=el_arr,
                           cutoff=np.deg2rad(float(cutoff_deg)))

    def elevation_at(self, az_rad):
        """查询若干方位角上的遮挡高度角（弧度），环形分段线性插值。"""
        az = np.asarray(az_rad, dtype=float)
        if self.az.size == 0:
            return np.full_like(az, self.cutoff, dtype=float)
        # np.interp 在 [x[-1]-2π, x[0]] 之间需要环绕处理：
        # 把方位角映射到以首控制点为起点的 [a0, a0+2π) 区间
        a0 = self.az[0]
        shift = (az - a0) % (2.0 * np.pi)
        xp = np.r_[self.az - a0, self.az[0] - a0 + 2.0 * np.pi]
        fp = np.r_[self.el, self.el[0]]
        value = np.interp(shift, xp, fp)
        # 统一截止角是任何方向上的最低门限
        return np.maximum(value, self.cutoff)

    def to_dict(self) -> dict:
        return {
            "cutoff_deg": float(np.rad2deg(self.cutoff)),
            "points": [[float(np.rad2deg(a)), float(np.rad2deg(e))]
                       for a, e in zip(self.az, self.el)],
        }

    @staticmethod
    def from_dict(d: dict | None) -> "MaskProfile":
        if d is None:
            return MaskProfile.create()
        return MaskProfile.create(points=d.get("points", []),
                                  cutoff_deg=float(
                                      d.get("cutoff_deg", DEFAULT_CUTOFF_DEG)))


@dataclass(frozen=True)
class Point:
    """测量点位：经纬度（度）、大地高（米）与遮挡轮廓。"""

    point_id: str
    lon_deg: float
    lat_deg: float
    height_m: float
    mask: MaskProfile
    version: int = 0

    def to_dict(self) -> dict:
        return {
            "point_id": self.point_id,
            "lon_deg": self.lon_deg,
            "lat_deg": self.lat_deg,
            "height_m": self.height_m,
            "mask": self.mask.to_dict(),
            "version": self.version,
        }

    @staticmethod
    def from_dict(d: dict) -> "Point":
        mask = MaskProfile.from_dict(d.get("mask"))
        return Point(
            point_id=str(d["point_id"]),
            lon_deg=float(d["lon_deg"]),
            lat_deg=float(d["lat_deg"]),
            height_m=float(d.get("height_m", 0.0)),
            mask=mask,
            version=int(d.get("version", 0)),
        )


def build_point(raw: dict, version: int = 0) -> Point:
    """从接口负载构造 Point 并做拒收校验。"""
    pid = str(raw.get("point_id", "")).strip()
    if not pid:
        raise ValidationError("点位缺少 point_id", field="point_id")
    lat = float(raw.get("lat_deg", raw.get("latitude", np.nan)))
    lon = float(raw.get("lon_deg", raw.get("longitude", np.nan)))
    if not np.isfinite(lat) or not (-90.0 <= lat <= 90.0):
        raise ValidationError(
            f"纬度越界（需在 [-90,90] 度），收到 {lat}", field="lat_deg")
    if not np.isfinite(lon) or not (-180.0 <= lon <= 180.0):
        raise ValidationError(
            f"经度越界（需在 [-180,180] 度），收到 {lon}", field="lon_deg")
    h = float(raw.get("height_m", 0.0))
    if not np.isfinite(h):
        raise ValidationError("大地高必须为有限值", field="height_m")
    mask = MaskProfile.from_dict(raw.get("mask"))
    return Point(point_id=pid, lon_deg=lon, lat_deg=lat, height_m=h,
                 mask=mask, version=version)
