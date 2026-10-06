"""时段搜索测试：短时尖峰、边界精度、最小值、遮挡抬高、取消。"""
import math

import numpy as np
import pytest

from app.errors import ValidationError
from app.mask import MaskProfile, Point
from app.search import EngineEvaluator, SyntheticEvaluator, find_windows
from app.visibility import VisibilityEngine


# --------------------------------------------------------------- 人造场景
def test_long_good_spike_is_found():
    # 长度 = 2 个步长（120s），且跨在网格中间，必须被发现
    step = 60
    spike_start, spike_end = 500_030, 500_150   # 半开，120s

    def fn(t):
        good = (t >= spike_start) & (t < spike_end)
        gdop = np.where(good, 2.0, np.inf)
        return good, gdop

    ev = SyntheticEvaluator(fn)
    ws = find_windows(ev, 500_000, 500_600, gdop_threshold=4.0,
                      scan_step=step, min_duration=0)
    assert len(ws) == 1
    w = ws[0]
    assert (w.start, w.end) == (spike_start, spike_end)
    # 边界精确到 1 秒
    assert w.start == spike_start
    assert w.end == spike_end


def test_short_spike_missed_within_bound():
    # 长度严格小于步长（15s < 60s），整体落在同一单元内部，按算法允许漏检；
    # 这正是漏检最短时长上界 scan_step 的含义。
    step = 60
    spike_start, spike_end = 500_010, 500_025   # 15s，完全在 [000,060] 内

    def fn(t):
        good = (t >= spike_start) & (t < spike_end)
        return good, np.where(good, 2.0, np.inf)

    ws = find_windows(SyntheticEvaluator(fn), 500_000, 500_600,
                      gdop_threshold=4.0, scan_step=step)
    # 被漏掉（两端网格点都是 False，同态单元不细分）
    assert ws == []


def test_spike_near_grid_point_found():
    # 即使很短，但只要尖峰覆盖到某个整秒网格点，也会被逐秒加密捕获
    step = 60
    spike_start, spike_end = 500_000, 500_010   # 10s，覆盖网格点 000

    def fn(t):
        good = (t >= spike_start) & (t < spike_end)
        return good, np.where(good, 2.0, np.inf)

    ws = find_windows(SyntheticEvaluator(fn), 500_000, 500_600,
                      gdop_threshold=4.0, scan_step=step)
    assert len(ws) == 1
    assert ws[0].duration == 10


def test_boundary_precision_continuous_crossing():
    # GDOP 线性穿越阈值（好→坏），边界由逐秒加密给出，误差 0 秒
    start, end = 100_000, 100_600
    cut = 100_137

    def fn(t):
        good = t < cut
        # 平滑变化：阈值 3，穿越发生在 cut
        gdop = 1.0 + (t - start) / (cut - start) * 3.0
        gdop = np.where(good, gdop, 5.0)
        return good, gdop

    ws = find_windows(SyntheticEvaluator(fn), start, end,
                      gdop_threshold=3.0, scan_step=60)
    assert len(ws) == 1
    w = ws[0]
    assert w.start == start
    assert w.end == cut
    assert abs(w.end - cut) <= 1


def test_boundary_precision_jump_rise_set(almanac24, simple_point):
    # 真实升/落造成的跳变：以 1° 截止角人为制造极扁可见区间，
    # 检查所有窗口边界都在某颗卫星穿越遮挡高度的 ±1 秒内
    point = Point(point_id="P001", lon_deg=simple_point.lon_deg,
                  lat_deg=simple_point.lat_deg, height_m=0.0,
                  mask=MaskProfile.create(cutoff_deg=5.0), version=1)
    engine = VisibilityEngine(almanac24, point)
    start, length = 1_000_000, 8 * 3600
    ev = EngineEvaluator(engine, gdop_threshold=6.0)
    ws = find_windows(ev, start, start + length, gdop_threshold=6.0,
                      scan_step=120, min_duration=0)
    assert ws, "24 星在 8 小时内应当存在可用时段"

    def good_at(t):
        vis, dops, n = engine.states_at(np.array([t]))
        return n[0] >= 4 and np.isfinite(dops[0, 0]) and \
            dops[0, 0] <= 6.0

    for w in ws:
        # 半开区间：起点前一秒坏、起点秒好；终点前一秒好、终点秒坏
        if w.start > start:
            assert not good_at(w.start - 1)
        assert good_at(w.start)
        assert good_at(w.end - 1)
        if w.end < start + length:
            assert not good_at(w.end)


def test_min_gdop_location_and_value():
    # V 形 GDOP，理论最小值在 t=100_300，值 1.5
    start, end = 100_000, 100_600

    def fn(t):
        gd = 1.5 + np.abs(t - 100_300) / 1000.0
        good = gd <= 3.0
        return good, gd

    ws = find_windows(SyntheticEvaluator(fn), start, end,
                      gdop_threshold=3.0, scan_step=60)
    assert len(ws) == 1
    w = ws[0]
    assert abs(w.gdop_min_time - 100_300) <= 1
    assert w.gdop_min == pytest.approx(1.5, abs=1e-3)


def test_min_duration_filter():
    def fn(t):
        good = (t >= 1000) & (t < 1045)     # 45s
        return good, np.where(good, 1.0, np.inf)

    ev = SyntheticEvaluator(fn)
    assert find_windows(ev, 0, 2000, scan_step=60, min_duration=60) == []
    ws = find_windows(SyntheticEvaluator(fn), 0, 2000, scan_step=60,
                      min_duration=30)
    assert len(ws) == 1 and ws[0].duration == 45


def test_reject_negative_min_duration():
    ev = SyntheticEvaluator(lambda t: (np.zeros_like(t, bool),
                                       np.full_like(t, np.inf, float)))
    with pytest.raises(ValidationError):
        find_windows(ev, 0, 100, min_duration=-1)


def test_reject_bad_range():
    ev = SyntheticEvaluator(lambda t: (np.zeros_like(t, bool),
                                       np.full_like(t, np.inf, float)))
    with pytest.raises(ValidationError):
        find_windows(ev, 100, 50)


def test_cancellation_propagates():
    from app.errors import JobAborted
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        return np.zeros_like(t, bool), np.full_like(t, np.inf, float)

    with pytest.raises(JobAborted):
        find_windows(SyntheticEvaluator(fn), 0, 100_000, scan_step=60,
                     is_cancelled=lambda: True)
