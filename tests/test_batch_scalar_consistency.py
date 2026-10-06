"""批量与单点可见性评估必须严格一致的回归测试。

历史上曾出现两个形状陷阱：
  1. 测站坐标 (3,) 与批量视线 (T,3) 相减时的广播错位；
  2. 缓存 computer 返回 (2,T) 被 reshape 成 (T,2)，az/el 两列对调。
这两个 bug 只在批量长度 != 3 且走缓存时暴露，因此这里强制用
"网格批 + 加密批 + 单帧" 的真实搜索调用序列来防回归。
"""
import math

import numpy as np

from app.mask import MaskProfile, Point
from app.search import EngineEvaluator, find_windows
from app.visibility import VisibilityEngine


def _point(mask_cutoff=5.0, pid="P"):
    return Point(point_id=pid, lon_deg=116.0, lat_deg=40.0, height_m=0.0,
                 mask=MaskProfile.create(cutoff_deg=mask_cutoff), version=1)


def test_batch_matches_scalar_after_grid_then_refine(almanac24):
    point = _point()
    eng = VisibilityEngine(almanac24, point)

    start, end, step = 1_000_000, 1_028_800, 120
    grid = np.arange(start, end + 1, step, dtype=float)
    if grid[-1] != end:
        grid = np.append(grid, end)
    vis_g, dops_g, n_g = eng.states_at(grid)

    # 抽查多个跨越/普通单元的逐秒加密批
    rng = np.random.default_rng(123)
    probe_cells = rng.choice(len(grid) - 1, size=20, replace=False)
    for ci in probe_cells:
        a, b = int(grid[ci]), int(grid[ci + 1])
        if b - a <= 1:
            continue
        inner = np.arange(a + 1, min(b, a + 120), dtype=float)
        vis_b, dops_b, n_b = eng.states_at(inner)
        for i, t in enumerate(inner):
            vis_s, dops_s, n_s = eng.states_at(np.array([t]))
            assert np.array_equal(vis_b[i], vis_s[0]), t
            assert n_b[i] == n_s[0], t
            assert np.array_equal(
                np.nan_to_num(dops_b[i]), np.nan_to_num(dops_s[0])), t


def test_window_boundaries_truly_flip(almanac24):
    point = _point()
    eng = VisibilityEngine(almanac24, point)
    ws = find_windows(EngineEvaluator(eng, 6.0), 1_000_000, 1_028_800,
                      gdop_threshold=6.0, scan_step=120, min_duration=0)
    assert ws

    def good(t):
        _, d, n = eng.states_at(np.array([t]))
        return bool(n[0] >= 4 and np.isfinite(d[0, 0])
                    and d[0, 0] <= 6.0)

    for w in ws:
        if w.start > 1_000_000:
            assert not good(w.start - 1)
        assert good(w.start)
        assert good(w.end - 1)
        if w.end < 1_028_800:
            assert not good(w.end)


def test_az_el_columns_not_swapped(almanac24):
    # az 与 el 数值范围不同；若两列被对调，高度角会出现大量 >pi/2 的值
    point = _point()
    eng = VisibilityEngine(almanac24, point)
    times = np.arange(1_000_000, 1_086_400, 137.0)  # 刻意避开 3 的倍数长度
    az, el = eng.all_az_el(times)
    assert az.shape == el.shape == (times.shape[0], len(almanac24.sats))
    # 高度角必须在 [-pi/2, pi/2]；方位角在 [0,2pi)
    assert np.all(el >= -math.pi / 2 - 1e-12)
    assert np.all(el <= math.pi / 2 + 1e-12)
    assert np.all(az >= 0.0)
    assert np.all(az < 2 * math.pi)
