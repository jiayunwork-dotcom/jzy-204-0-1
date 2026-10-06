"""两版规划结果之间的差异对比。

逐点位比较两组时段窗口（窗口均以半开整秒区间表示）：
  * 消失 disappeared：旧版有、新版无匹配；
  * 缩短 shortened：匹配上，但时长明显变短；
  * 平移 shifted：匹配上，起止时刻有移动（时长未明显缩短；增长也归此类，
    并标注方向）；
  * 新增 added：新版有、旧版无匹配。

匹配采用以"时间接近 + 重叠"为代价的贪心配对：优先配对重叠时长最长、
中心时刻最近的窗口。所有容差（秒）可配。
"""
from __future__ import annotations

import numpy as np


def _w(window: dict) -> tuple[int, int]:
    return int(window["start_epoch"]), int(window["end_epoch"])


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def diff_point_windows(old_windows: list[dict], new_windows: list[dict],
                       shift_tol_s: int = 1,
                       shrink_tol_s: int = 1,
                       match_gap_s: int = 120) -> dict:
    """比较单点位两组窗口。返回四类变化。

    match_gap_s：新旧窗口即使不重叠，只要间隔不超过该值仍视为同一条
    （用于跨过一次重规划后边界小幅平移的配对）；间隔更大则分别归入
    消失/新增。
    """
    old = [dict(w) for w in old_windows]
    new = [dict(w) for w in new_windows]
    old_idx = list(range(len(old)))
    new_idx = list(range(len(new)))

    # 所有可能配对按代价排序（重叠越大、中心越近越优先）
    pairs = []
    for i in old_idx:
        a = _w(old[i])
        for k in new_idx:
            b = _w(new[k])
            ov = _overlap(a, b)
            center_dist = abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2)
            gap = 0 if ov > 0 else (max(a[0], b[0]) - min(a[1], b[1]))
            cost = (0 if ov > 0 else 1, gap, -ov, center_dist)
            pairs.append((cost, i, k, ov, gap))
    pairs.sort(key=lambda x: x[0])

    matches: dict[int, int] = {}
    used_new: set[int] = set()
    for _, i, k, ov, gap in pairs:
        if i in matches or k in used_new:
            continue
        # 无重叠且间隔过大的不配对（视为不同窗口）
        if ov == 0 and gap > match_gap_s:
            continue
        matches[i] = k
        used_new.add(k)

    disappeared, shortened, shifted, added = [], [], [], []
    for i, w in enumerate(old):
        a = _w(w)
        if i not in matches:
            disappeared.append(w)
            continue
        w2 = new[matches[i]]
        b = _w(w2)
        dur_a, dur_b = a[1] - a[0], b[1] - b[0]
        start_delta = b[0] - a[0]
        end_delta = b[1] - a[1]
        moved = abs(start_delta) > shift_tol_s or abs(end_delta) > shift_tol_s
        if dur_b + shrink_tol_s < dur_a:
            shortened.append({
                "old": w, "new": w2,
                "duration_change_s": dur_b - dur_a,
                "start_shift_s": start_delta,
                "end_shift_s": end_delta,
            })
        elif moved:
            shifted.append({
                "old": w, "new": w2,
                "duration_change_s": dur_b - dur_a,
                "start_shift_s": start_delta,
                "end_shift_s": end_delta,
            })
    for k, w in enumerate(new):
        if k not in used_new:
            added.append(w)

    return {
        "n_old": len(old),
        "n_new": len(new),
        "unchanged": len(matches) - len(shortened) - len(shifted),
        "disappeared": disappeared,
        "shortened": shortened,
        "shifted": shifted,
        "added": added,
    }


def diff_results(old_results: list[dict], new_results: list[dict],
                 shift_tol_s: int = 1) -> dict:
    """比较两份完整规划结果（逐点结果列表）。"""
    old_by_id = {r["point_id"]: r for r in old_results}
    new_by_id = {r["point_id"]: r for r in new_results}
    points = sorted(set(old_by_id) | set(new_by_id))
    per_point = {}
    totals = {"disappeared": 0, "shortened": 0, "shifted": 0,
              "added": 0, "unchanged": 0}
    for pid in points:
        o = old_by_id.get(pid)
        n = new_by_id.get(pid)
        if o is None:
            d = diff_point_windows([], n["windows"], shift_tol_s)
        elif n is None:
            d = diff_point_windows(o["windows"], [], shift_tol_s)
        else:
            d = diff_point_windows(o["windows"], n["windows"], shift_tol_s)
        per_point[pid] = d
        for key in totals:
            totals[key] += len(d[key]) if isinstance(d[key], list) else d[key]
    return {
        "points": per_point,
        "totals": totals,
        "has_changes": any(v > 0 for k, v in totals.items()
                           if k != "unchanged"),
        "old_almanac_version": _alm(old_results),
        "new_almanac_version": _alm(new_results),
    }


def _alm(results):
    versions = {int(r["almanac_version"]) for r in results}
    return max(versions) if versions else None
