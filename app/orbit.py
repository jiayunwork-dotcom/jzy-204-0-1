"""历书（开普勒轨道根数）与卫星 ECEF 位置计算。

历书模型（与 GPS/Galileo 广播历书同构，不依赖任何卫星导航库）：
    每颗卫星给出
        sqrt_a   半长轴平方根 [sqrt(m)]，或直接给 semi_major_axis [m]
        ecc      偏心率 ∈ [0, 1)
        inc      轨道倾角 i [rad]
        raan0    参考时刻升交点赤经 Ω₀ [rad]
        raan_rate 升交点赤经变化率 Ω̇ [rad/s]
        argp     近地点幅角 ω [rad]
        ma       参考时刻平近点角 M₀ [rad]
        toe      参考时刻（历元）t_oe [Unix 秒]
    可选
        mean_motion_correction Δn [rad/s]，修正平均运动
        muid     若整份历书不共用 GM，可按星指定

轨道平面内用二体开普勒模型；地球自转通过 ECI→ECEF 旋转计入。
约定：t = t_oe 时惯性系 x 轴与格林尼治子午面重合，因此升交点在
ECEF 中的经度为 Ω = Ω₀ + (Ω̇ − ωₑ)(t − t_oe)。这正是广播历书
常用形式（Ω̇ 通常给负值，Ω₀ 是周历元时刻的 ECEF 升交点经度）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ValidationError

# WGS84 基本常数
MU_EARTH = 3.986005e14           # 地球引力常数 GM [m^3/s^2]
OMEGA_EARTH = 7.2921151467e-5    # 地球自转角速度 [rad/s]


@dataclass(frozen=True)
class SatAlmanac:
    """单颗卫星的开普勒根数。角度均为弧度，时间为 Unix 秒。"""

    sat_id: str
    sqrt_a: float
    ecc: float
    inc: float
    raan0: float
    raan_rate: float
    argp: float
    ma: float
    toe: float
    d_n: float = 0.0
    mu: float = MU_EARTH

    def to_dict(self) -> dict:
        return {
            "sat_id": self.sat_id,
            "sqrt_a": self.sqrt_a,
            "ecc": self.ecc,
            "inc": self.inc,
            "raan0": self.raan0,
            "raan_rate": self.raan_rate,
            "argp": self.argp,
            "ma": self.ma,
            "toe": self.toe,
            "d_n": self.d_n,
            "mu": self.mu,
        }

    @staticmethod
    def from_dict(d: dict) -> "SatAlmanac":
        return SatAlmanac(
            sat_id=str(d["sat_id"]),
            sqrt_a=float(d["sqrt_a"]),
            ecc=float(d["ecc"]),
            inc=float(d["inc"]),
            raan0=float(d["raan0"]),
            raan_rate=float(d.get("raan_rate", 0.0)),
            argp=float(d["argp"]),
            ma=float(d["ma"]),
            toe=float(d["toe"]),
            d_n=float(d.get("d_n", 0.0)),
            mu=float(d.get("mu", MU_EARTH)),
        )


@dataclass(frozen=True)
class Almanac:
    """一版历书：一组卫星根数。version 由 repository 分配。"""

    sats: tuple[SatAlmanac, ...]
    version: int = 0
    source: str | None = None

    @property
    def sat_ids(self) -> tuple[str, ...]:
        return tuple(s.sat_id for s in self.sats)

    def get(self, sat_id: str) -> SatAlmanac:
        for s in self.sats:
            if s.sat_id == sat_id:
                return s
        raise KeyError(sat_id)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "satellites": [s.to_dict() for s in self.sats],
        }

    @staticmethod
    def from_dict(d: dict) -> "Almanac":
        sats = tuple(SatAlmanac.from_dict(s) for s in d["satellites"])
        return Almanac(sats=sats, version=int(d.get("version", 0)),
                       source=d.get("source"))


def build_sat(raw: dict, default_toe: float | None = None) -> SatAlmanac:
    """从接口/导入字典构造 SatAlmanac，并做拒收校验。

    角度字段接受弧度（rad）或度（字段名加 ``_deg``），二选一。
    """
    sat_id = str(raw.get("sat_id", "")).strip()
    if not sat_id:
        raise ValidationError("卫星缺少 sat_id", field="sat_id")

    # 半长轴：sqrt_a [sqrt(m)] 或 semi_major_axis [m]
    if "sqrt_a" in raw and raw["sqrt_a"] is not None:
        sqrt_a = float(raw["sqrt_a"])
    elif "semi_major_axis" in raw and raw["semi_major_axis"] is not None:
        a = float(raw["semi_major_axis"])
        if not np.isfinite(a) or a <= 0.0:
            raise ValidationError(f"卫星 {sat_id} 半长轴必须为正",
                                  field="semi_major_axis")
        sqrt_a = float(np.sqrt(a))
    else:
        raise ValidationError(f"卫星 {sat_id} 缺少半长轴", field="sqrt_a")
    if not np.isfinite(sqrt_a) or sqrt_a <= 0.0:
        raise ValidationError(f"卫星 {sat_id} 半长轴必须为正",
                              field="sqrt_a")

    ecc = float(raw.get("ecc", 0.0))
    if not np.isfinite(ecc) or not (0.0 <= ecc < 1.0):
        raise ValidationError(
            f"卫星 {sat_id} 偏心率必须在 [0,1)，收到 {ecc}", field="ecc")

    toe = raw.get("toe", default_toe)
    if toe is None:
        raise ValidationError(f"卫星 {sat_id} 缺少参考时刻 toe", field="toe")
    from .timeutils import parse_time
    toe = parse_time(toe)

    def angle(name: str, required: bool = True):
        rad_name = name
        deg_name = name + "_deg"
        if rad_name in raw and raw[rad_name] is not None:
            value = float(raw[rad_name])
        elif deg_name in raw and raw[deg_name] is not None:
            value = np.deg2rad(float(raw[deg_name]))
        elif required:
            raise ValidationError(f"卫星 {sat_id} 缺少根数 {name}", field=name)
        else:
            value = 0.0
        if not np.isfinite(value):
            raise ValidationError(f"卫星 {sat_id} 根数 {name} 非有限值",
                                  field=name)
        return float(value)

    return SatAlmanac(
        sat_id=sat_id,
        sqrt_a=sqrt_a,
        ecc=ecc,
        inc=angle("inc"),
        raan0=angle("raan0"),
        raan_rate=angle("raan_rate", required=False),
        argp=angle("argp"),
        ma=angle("ma"),
        toe=toe,
        d_n=angle("d_n", required=False),
        mu=float(raw.get("mu", MU_EARTH)),
    )


def build_almanac(raw: dict, version: int = 0) -> Almanac:
    """从导入负载构造一版历书。

    格式：{"source": ..., "default_toe": ...,
           "satellites": [{...根数，角度可给 _deg...}]}
    """
    sat_raws = raw.get("satellites")
    if not isinstance(sat_raws, list) or not sat_raws:
        raise ValidationError("历书必须至少包含一颗卫星",
                              field="satellites")
    default_toe = raw.get("default_toe")
    if default_toe is not None:
        from .timeutils import parse_time
        default_toe = parse_time(default_toe)
    sats = tuple(build_sat(s, default_toe) for s in sat_raws)
    ids = [s.sat_id for s in sats]
    if len(set(ids)) != len(ids):
        raise ValidationError("历书中 sat_id 不允许重复", field="satellites")
    return Almanac(sats=sats, version=version, source=raw.get("source"))


def solve_kepler(ma: np.ndarray, ecc: float, tol: float = 1e-12,
                 max_iter: int = 30) -> np.ndarray:
    """牛顿迭代解 M = E − e·sinE，返回偏近点角 E。"""
    m = np.asarray(ma, dtype=float)
    e = ecc
    # 初值
    ea = m + e * np.sin(m)
    for _ in range(max_iter):
        f = ea - e * np.sin(ea) - m
        fp = 1.0 - e * np.cos(ea)
        step = f / fp
        ea = ea - step
        if np.all(np.abs(step) < tol):
            break
    return ea


def sat_position(sat: SatAlmanac, times) -> np.ndarray:
    """计算卫星在若干时刻的 ECEF 位置。

    参数
        times: 标量或数组，Unix 秒
    返回
        shape=(..., 3) 的 ndarray（米）。输入为标量时形状为 (3,)。
    """
    scalar = np.isscalar(times)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    dt = t - sat.toe

    a = sat.sqrt_a * sat.sqrt_a
    # 平均运动（含可选修正），开普勒第三定律 n=sqrt(GM/a^3)
    n0 = np.sqrt(sat.mu / (a ** 3))
    n = n0 + sat.d_n
    mk = sat.ma + n * dt

    ea = solve_kepler(mk, sat.ecc)
    ecc = sat.ecc
    cos_e, sin_e = np.cos(ea), np.sin(ea)
    # 真近点角
    nu = np.arctan2(np.sqrt(1.0 - ecc * ecc) * sin_e, cos_e - ecc)
    phi = nu + sat.argp                     # 升交距角（近地点起算）
    # 矢径长度
    r = a * (1.0 - ecc * cos_e)

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x_p = r * cos_phi                       # 轨道面坐标（x 指向升交点）
    y_p = r * sin_phi

    # ECEF 升交点经度：Ω₀ + (Ω̇ − ωₑ)Δt，地球自转在此计入
    omegak = sat.raan0 + (sat.raan_rate - OMEGA_EARTH) * dt
    cos_o, sin_o = np.cos(omegak), np.sin(omegak)
    ci, si = np.cos(sat.inc), np.sin(sat.inc)

    x = x_p * cos_o - y_p * ci * sin_o
    y = x_p * sin_o + y_p * ci * cos_o
    z = y_p * si

    pos = np.stack([x, y, z], axis=-1)
    if scalar:
        return pos[0]
    return pos


def almanac_positions(alm: Almanac, times) -> dict[str, np.ndarray]:
    """批量计算整份历书所有卫星的位置。

    返回 {sat_id: shape=(T,3)}（times 为数组时）。
    """
    scalar = np.isscalar(times)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    out = {}
    for s in alm.sats:
        p = sat_position(s, t)
        out[s.sat_id] = p[0] if scalar else p
    return out
