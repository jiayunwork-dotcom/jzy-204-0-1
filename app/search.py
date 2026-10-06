"""时间轴时段搜索。

策略（固定粗步长 + 边界细化，兼顾速度与可证明的不漏检性质）
==================================================================
1. 粗扫描：以 ``scan_step``（默认 60 秒）为间隔在 [range_start, range_end)
   的整数秒网格上**批量**评估每个时刻的可用状态
   （可见卫星 ≥ 4 且 GDOP ≤ 阈值）。
2. 边界细化：相邻两个网格点状态不同（"跨越单元"）时，把该单元内部
   所有整数秒一次性批量评估，逐秒定位跳变（既可能是 GDOP 连续穿越
   阈值，也可能是某颗卫星在遮挡轮廓处升起/落下造成的**跳变**——
   逐秒加密对两者都适用，不假设曲线连续或单峰）。
3. 相邻网格点状态相同的单元内部不再细分，直接连成游程（run-length）。
4. 每个可用游程内先在已采样点找 GDOP 最小处，再在其附近 ±2 个步长内
   做 1 秒局部加密，给出最小值及其出现时刻（整秒）。
5. 按最短连续时长过滤，输出半开区间 [start, end)，边界精确到 1 秒。

漏检分析
--------
由于跨越单元被加密到每一秒，任何状态变化要漏检，只能是：某可用片段
（或不可用缺口）连同它的两个边界**一起落在同一个单元内部**，即整个
事件比 ``scan_step`` 还短。因此：

  **任何持续时长 ≥ scan_step 的连续可用时段必定被发现；**
  被漏掉的可用尖峰（若存在）持续时长严格小于 scan_step 秒。

即漏检最短时长上界 = scan_step（默认 60 秒）。边界时刻本身由逐秒
加密给出，误差为 0 秒（整秒网格意义下）。测试中以一个人造的短时
尖峰验证：长度 ≥ 步长者被发现，长度 < 步长者漏检并符合上界结论。

注意：遮挡轮廓抬高只会让同一时刻的可见卫星集合变小（见 mask 模块），
DOP 矩阵是 GᵀG 随卫星行增加的 Schur 和，GDOP 单调不增——因此本
搜索得到的时段在轮廓抬高时只会缩短或消失，不会产生新时段。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import JobAborted, ValidationError


@dataclass(frozen=True)
class Window:
    start: int                 # Unix 秒，半开区间起点（含）
    end: int                   # Unix 秒，半开区间终点（不含）
    gdop_min: float
    gdop_min_time: int         # 最小值出现时刻（Unix 秒）

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        from .timeutils import to_iso
        return {
            "start": to_iso(self.start),
            "end": to_iso(self.end),
            "start_epoch": self.start,
            "end_epoch": self.end,
            "duration_s": self.duration,
            "gdop_min": float(self.gdop_min),
            "gdop_min_time": to_iso(self.gdop_min_time),
            "gdop_min_time_epoch": self.gdop_min_time,
        }


class Evaluator:
    """评估器协议：states(times) -> (good bool(T,), gdop float(T,))。"""

    def states(self, times: np.ndarray):
        raise NotImplementedError


class EngineEvaluator(Evaluator):
    """把 VisibilityEngine 适配成时段搜索所需的布尔/最小值接口。"""

    def __init__(self, engine, gdop_threshold: float):
        self.engine = engine
        self.threshold = gdop_threshold

    def states(self, times: np.ndarray):
        visible, dops, n_vis = self.engine.states_at(times)
        gdop = dops[:, 0]
        good = (n_vis >= 4) & np.isfinite(gdop) & (gdop <= self.threshold)
        return good, gdop


class SyntheticEvaluator(Evaluator):
    """测试用人造评估器：fn(整数秒数组) -> (good, gdop)。"""

    def __init__(self, fn):
        self._fn = fn

    def states(self, times: np.ndarray):
        times = np.asarray(times, dtype=np.int64)
        good, gdop = self._fn(times)
        return np.asarray(good, dtype=bool), np.asarray(gdop, dtype=float)


def _check_cancel(is_cancelled):
    if is_cancelled is not None and is_cancelled():
        raise JobAborted()


def find_windows(evaluator: Evaluator, range_start: int, range_end: int,
                 gdop_threshold: float | None = None,
                 min_duration: int = 0, scan_step: int = 60,
                 on_progress=None, is_cancelled=None,
                 refine_min: bool = True) -> list[Window]:
    """主入口。range 为半开 [range_start, range_end)，全部取整秒。"""
    range_start = int(range_start)
    range_end = int(range_end)
    if range_end < range_start:
        raise ValidationError("日期范围结束早于开始")
    if min_duration < 0:
        raise ValidationError("最短时长不能为负")
    if scan_step < 1:
        raise ValidationError("扫描步长必须为正")

    if range_end <= range_start:
        return []

    # ------------------------------------------------------------ 粗扫描
    grid = np.arange(range_start, range_end + 1, scan_step, dtype=np.int64)
    if grid[-1] != range_end:
        # 保证范围终点也被采样，最后一段通常短于一个步长
        grid = np.append(grid, range_end)
    _check_cancel(is_cancelled)
    good_g, gdop_g = evaluator.states(grid)

    # state[t] 为逐秒加密后"已知确定"的时刻状态；同态单元内部不采样。
    state: dict[int, bool] = {}
    gdop: dict[int, float] = {}
    for t, go, gd in zip(grid.tolist(), good_g.tolist(), gdop_g.tolist()):
        state[int(t)] = bool(go)
        gdop[int(t)] = float(gd)

    # 真实翻转事件（整秒对）：(sec, new_state) 表示 sec 起状态为 new_state。
    # 跨越单元逐秒加密，单元内可能出现多次翻转（短时尖峰），逐一收集。
    events: list[tuple[int, bool]] = []
    total_cells = max(1, len(grid) - 1)
    for ci in range(len(grid) - 1):
        a, b = int(grid[ci]), int(grid[ci + 1])
        sa, sb = bool(good_g[ci]), bool(good_g[ci + 1])
        if sa != sb and b - a > 1:
            inner = np.arange(a + 1, b, dtype=np.int64)
            go_i, gd_i = evaluator.states(inner)
            prev_t, prev_go = a, sa
            for t, go, gv in zip(inner.tolist(), go_i.tolist(),
                                 gd_i.tolist()):
                t = int(t)
                state[t] = bool(go)
                gdop[t] = float(gv)
                if bool(go) != prev_go:
                    events.append((t, bool(go)))
                    prev_go = bool(go)
                prev_t = t
            if prev_go != sb:
                events.append((b, sb))
        if on_progress is not None:
            on_progress((ci + 1) / total_cells)
        _check_cancel(is_cancelled)

    # -------------------------------------- 由翻转事件构造精确整秒时段
    # 已知范围起点状态为 state[range_start]；翻转时刻把时间轴切成
    # 状态恒定（就已知信息而言）的区间。
    events.sort(key=lambda x: x[0])
    intervals: list[tuple[int, int, bool]] = []
    cur_start = range_start
    cur_state = state[range_start]
    for sec, new_state in events:
        if sec <= cur_start:
            cur_state = new_state
            continue
        if sec >= range_end:
            intervals.append((cur_start, range_end, cur_state))
            cur_start = range_end
            cur_state = new_state
            break
        # [cur_start, sec) 为 cur_state；sec 起翻转为 new_state
        intervals.append((cur_start, sec, cur_state))
        cur_start, cur_state = sec, new_state
    if cur_start < range_end:
        intervals.append((cur_start, range_end, cur_state))

    good_intervals = [(a, b) for a, b, go in intervals if go]

    # ----------------------------- 各可用时段 GDOP 最小值细化与时长过滤
    result: list[Window] = []
    for w_start, w_end in good_intervals:
        if w_end - w_start < min_duration:
            continue
        # 已采样点里的最小值
        known_ts = [t for t in state if w_start <= t < w_end]
        best_t = min(known_ts, key=lambda t: gdop[t]) if known_ts else \
            w_start
        best_v = gdop.get(best_t, np.inf)
        if refine_min and w_end - w_start > 1:
            lo = max(w_start, best_t - 2 * scan_step)
            hi = min(w_end - 1, best_t + 2 * scan_step)
            dense = np.arange(lo, hi + 1, dtype=np.int64)
            fresh = np.array([t for t in dense.tolist()
                              if t not in state], dtype=np.int64)
            if fresh.size:
                _check_cancel(is_cancelled)
                _, gd_f = evaluator.states(fresh)
                kk = int(np.argmin(gd_f))
                if float(gd_f[kk]) < best_v:
                    best_v = float(gd_f[kk])
                    best_t = int(fresh[kk])
        result.append(Window(start=w_start, end=w_end,
                             gdop_min=float(best_v), gdop_min_time=best_t))
    result.sort(key=lambda w: w.start)
    return result
