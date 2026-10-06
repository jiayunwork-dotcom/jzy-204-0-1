"""业务主流程测试。

覆盖题目要求的：
  * 历书更新后的差异；
  * 连续两次历书更新的作业取代；
  * 作业取消（取消后不写入部分结果）；
  * 修改遮挡轮廓触发重算；
  * 共享计算与单独计算一致；
  * 遮挡抬高时段不增。
"""
import math
import threading
import time
from datetime import datetime, timezone

import mongomock
import numpy as np
import pytest

from app.cache import SharedSkyCache
from app.diff import diff_results
from app.mask import MaskProfile, Point
from app.orbit import Almanac
from app.repository import Repository
from app.scheduler import Scheduler
from app.visibility import VisibilityEngine

from .conftest import epoch_iso


# --------------------------------------------------------------- 辅助
def _make_stack(repo_version_counters=True):
    client = mongomock.MongoClient()
    repo = Repository(client["gnss_flow"])
    repo.setup_indexes()
    return repo


def _seed(repo, almanac, point):
    repo.counters.update_one({"_id": "almanac_version"},
                             {"$set": {"seq": almanac.version}}, upsert=True)
    repo.insert_almanac(almanac)
    point = Point(point_id=point.point_id, lon_deg=point.lon_deg,
                  lat_deg=point.lat_deg, height_m=point.height_m,
                  mask=point.mask, version=1)
    repo.counters.update_one({"_id": "point_version:P01"},
                             {"$set": {"seq": 1}}, upsert=True)
    repo.upsert_point(point)
    return point


def _create_plan(repo, scheduler, almanac_version, point_ids, start, end,
                 threshold=6.0, min_duration=300, scan_step=60):
    plan_id = f"plan-{repo.next_id('plan'):06d}"
    doc = {"plan_id": plan_id, "point_ids": list(point_ids),
           "almanac_version": almanac_version,
           "params": {"gdop_threshold": threshold,
                      "min_duration_s": min_duration,
                      "date_start": float(start),
                      "date_end": float(end),
                      "scan_step_s": scan_step},
           "status": "active"}
    repo.insert_plan(doc)
    return doc, scheduler.submit_plan_job(doc, triggered_by="manual")


def _wait(scheduler, job_ids, timeout=60):
    deadline = time.time() + timeout
    pending = list(job_ids)
    out = {}
    while time.time() < deadline and pending:
        still = []
        for jid in pending:
            doc = scheduler.get_job(jid)
            if doc["status"] in ("done", "failed", "canceled",
                                 "interrupted"):
                out[jid] = doc
            else:
                still.append(jid)
        if still:
            time.sleep(0.01)
        pending = still
    assert not pending, f"作业超时: {pending}"
    return out


def _shifted_almanac(alm: Almanac, version: int, phase: float) -> Almanac:
    """生成一版"更新后的历书"：所有卫星平近点角整体平移。"""
    sats = []
    for s in alm.sats:
        sats.append(type(s)(
            sat_id=s.sat_id, sqrt_a=s.sqrt_a, ecc=s.ecc, inc=s.inc,
            raan0=s.raan0, raan_rate=s.raan_rate, argp=s.argp,
            ma=s.ma + phase, toe=s.toe, d_n=s.d_n, mu=s.mu))
    return Almanac(sats=tuple(sats), version=version)


# --------------------------------------------------------------- 差异
def test_almanac_update_produces_diff(almanac24, simple_point,
                                      future_window):
    repo = _make_stack()
    cache = SharedSkyCache()
    sched = Scheduler(repo, cache=cache, workers=2)
    sched.startup()
    start, end = future_window
    point = _seed(repo, almanac24, simple_point)
    doc, j1 = _create_plan(repo, sched, 1, ["P001"], start, end)
    _wait(sched, [j1])

    before = repo.get_result(doc["plan_id"])
    assert before["almanac_version"] == 1
    old_windows = [r["windows"] for r in before["results"]][0]
    assert len(old_windows) > 0   # 24 星 2 天内必有可用时段

    # 导入新历书（相位大平移，使时段安排明显不同）
    alm2 = _shifted_almanac(almanac24, 2, phase=math.pi)
    repo.counters.update_one({"_id": "almanac_version"},
                             {"$set": {"seq": 2}}, upsert=True)
    repo.insert_almanac(alm2)
    jobs = sched.trigger_almanac_replan(2, now_epoch_value=start - 1)
    assert len(jobs) == 1
    _wait(sched, jobs)

    after = repo.get_result(doc["plan_id"])
    assert after["almanac_version"] == 2
    # 历史保留一份（v1）
    assert len(after["history"]) == 1
    assert after["history"][0]["almanac_version"] == 1

    d = diff_results(before["results"], after["results"])
    n_changes = sum(d["totals"][k] for k in
                    ("disappeared", "shortened", "shifted", "added"))
    assert n_changes > 0

    # 通过差异接口（历史基线）也可得到同样结论
    from app.diff import diff_point_windows
    dd = diff_point_windows(old_windows,
                            after["results"][0]["windows"])
    total_dd = sum(len(dd[k]) for k in
                   ("disappeared", "shortened", "shifted", "added"))
    assert total_dd == n_changes
    sched.shutdown()


# --------------------------------------------------------- 连续更新取代
class Gate:
    """每个点位开始时阻塞，测试借此精确控制作业处于 running。"""

    def __init__(self):
        self.events: dict[str, threading.Event] = {}
        self.reached = []
        self._lock = threading.Lock()

    def hook(self, point_id):
        with self._lock:
            self.reached.append(point_id)
            ev = self.events.setdefault(point_id, threading.Event())
        ev.wait(timeout=30)

    def release(self, point_id):
        self.events.setdefault(point_id, threading.Event()).set()


def test_consecutive_almanac_updates_supersede(almanac24, simple_point,
                                               future_window):
    repo = _make_stack()
    gate = Gate()
    sched = Scheduler(repo, cache=SharedSkyCache(), workers=1,
                      per_point_hook=gate.hook)
    sched.startup()
    start, end = future_window
    point = _seed(repo, almanac24, simple_point)
    doc, j1 = _create_plan(repo, sched, 1, ["P001"], start, end)

    # 等 j1 在点位开始处被挡住
    deadline = time.time() + 10
    while time.time() < deadline and not gate.reached:
        time.sleep(0.01)
    assert gate.reached == ["P001"]
    assert sched.get_job(j1)["status"] == "running"

    # 第一次历书更新 -> j2 取代 j1（j1 被取消）
    alm2 = _shifted_almanac(almanac24, 2, phase=0.3)
    repo.counters.update_one({"_id": "almanac_version"},
                             {"$set": {"seq": 2}}, upsert=True)
    repo.insert_almanac(alm2)
    j2 = sched.trigger_almanac_replan(2, now_epoch_value=start - 1)
    assert len(j2) == 1 and j2[0] != j1
    gate.release("P001")
    # j1 被取代取消；j2 的点位钩子事件已置位，可一路跑完。
    done = _wait(sched, [j1, j2[0]])
    assert done[j1]["status"] == "canceled"
    assert done[j2[0]]["status"] == "done"
    # 此时当前结果为 v2
    assert repo.get_result(doc["plan_id"])["almanac_version"] == 2

    # 第二次历书更新 -> j3 取代；计划引用的历书版本变为 3
    alm3 = _shifted_almanac(almanac24, 3, phase=0.6)
    repo.counters.update_one({"_id": "almanac_version"},
                             {"$set": {"seq": 3}}, upsert=True)
    repo.insert_almanac(alm3)
    j3 = sched.trigger_almanac_replan(3, now_epoch_value=start - 1)
    assert len(j3) == 1 and j3[0] not in (j1, j2[0])
    done3 = _wait(sched, j3, timeout=60)
    assert done3[j3[0]]["status"] == "done"

    result = repo.get_result(doc["plan_id"])
    # 最终当前结果基于最新历书 v3，历史只保留一份（v2）
    assert result["almanac_version"] == 3
    assert len(result["history"]) == 1
    assert result["history"][0]["almanac_version"] == 2
    sched.shutdown()


# --------------------------------------------------------------- 取消
def test_cancel_writes_no_partial_result(almanac24, simple_point,
                                         future_window):
    repo = _make_stack()
    gate = Gate()
    sched = Scheduler(repo, cache=SharedSkyCache(), workers=1,
                      per_point_hook=gate.hook)
    sched.startup()
    start, end = future_window
    point = _seed(repo, almanac24, simple_point)
    # 两个点位，作业在第一个点位处被取消
    p2 = Point(point_id="P002", lon_deg=117.0, lat_deg=41.0, height_m=0.0,
               mask=MaskProfile.create(cutoff_deg=10.0), version=1)
    repo.counters.update_one({"_id": "point_version:P002"},
                             {"$set": {"seq": 1}}, upsert=True)
    repo.upsert_point(p2)

    doc, j1 = _create_plan(repo, sched, 1, ["P001", "P002"], start, end)
    deadline = time.time() + 10
    while time.time() < deadline and not gate.reached:
        time.sleep(0.01)
    assert sched.get_job(j1)["status"] == "running"

    sched.cancel(j1)
    gate.release("P001")
    _wait(sched, [j1])
    assert sched.get_job(j1)["status"] == "canceled"

    # 取消后不写入任何结果（get_result 应查不到）
    assert repo.results.find_one({"plan_id": doc["plan_id"]}) is None
    sched.shutdown()


# ----------------------------------------------------------- 遮挡触发
def test_point_mask_update_triggers_replan(almanac24, simple_point,
                                           future_window):
    repo = _make_stack()
    sched = Scheduler(repo, cache=SharedSkyCache(), workers=2)
    sched.startup()
    start, end = future_window
    point = _seed(repo, almanac24, simple_point)
    doc, j1 = _create_plan(repo, sched, 1, ["P001"], start, end)
    _wait(sched, [j1])
    before = repo.get_result(doc["plan_id"])
    n_before = sum(len(r["windows"]) for r in before["results"])

    # 更新点位（遮挡整体抬高），版本自增 -> 自动重算
    raised = Point(point_id="P001", lon_deg=point.lon_deg,
                   lat_deg=point.lat_deg, height_m=point.height_m,
                   mask=MaskProfile.create(cutoff_deg=35.0), version=2)
    repo.counters.update_one({"_id": "point_version:P001"},
                             {"$set": {"seq": 2}}, upsert=True)
    repo.upsert_point(raised)
    jobs = sched.trigger_point_replan("P001", 2,
                                      now_epoch_value=start - 1)
    assert len(jobs) == 1
    _wait(sched, jobs)

    after = repo.get_result(doc["plan_id"])
    n_after = sum(len(r["windows"]) for r in after["results"])
    assert n_after <= n_before
    sched.shutdown()


# ------------------------------------------------- 遮挡抬高：时段不增
def test_raised_mask_windows_are_subset(almanac24, future_window):
    start, end = future_window
    low = Point(point_id="LOW", lon_deg=116.39, lat_deg=39.9,
                height_m=50.0, mask=MaskProfile.create(cutoff_deg=5.0),
                version=1)
    high = Point(point_id="HIGH", lon_deg=116.39, lat_deg=39.9,
                 height_m=50.0,
                 mask=MaskProfile.create(
                     points=[(a, 25.0) for a in range(0, 360, 30)],
                     cutoff_deg=25.0), version=1)
    from app.search import EngineEvaluator, find_windows
    e_low = VisibilityEngine(almanac24, low)
    e_high = VisibilityEngine(almanac24, high)
    ws_low = find_windows(EngineEvaluator(e_low, 6.0), start, end,
                          scan_step=120, min_duration=300)
    ws_high = find_windows(EngineEvaluator(e_high, 6.0), start, end,
                           scan_step=120, min_duration=300)

    assert ws_low, "低遮挡下 24 星应存在可用时段"

    def covered(windows, t):
        return any(w.start <= t < w.end for w in windows)

    # 逐秒抽查高轮廓的可用时刻必须全部落在低轮廓时段内
    for t in range(start, end, 300):
        if covered(ws_high, t):
            assert covered(ws_low, t), \
                f"t={t} 在高轮廓可用却在低轮廓不可用，单调性被破坏"
    # 总时长单调不增
    total_low = sum(w.duration for w in ws_low)
    total_high = sum(w.duration for w in ws_high)
    assert total_high <= total_low


# ----------------------------------------------- 共享计算与单独计算一致
def test_shared_cache_matches_standalone(almanac24, beijing_point,
                                         future_window):
    start, _ = future_window
    times = np.arange(start, start + 3600, 60)

    shared = SharedSkyCache()
    # 规划 A 与规划 B 引用同一点位（同一版本）——先后计算
    eng_a = VisibilityEngine(almanac24, beijing_point, cache=shared)
    az_a, el_a = eng_a.all_az_el(times)
    calls_after_a = shared.compute_calls
    assert calls_after_a > 0

    eng_b = VisibilityEngine(almanac24, beijing_point, cache=shared)
    az_b, el_b = eng_b.all_az_el(times)
    calls_after_b = shared.compute_calls
    # 第二次全部命中缓存：没有任何新的轨道计算
    assert calls_after_b == calls_after_a
    assert shared.hits > 0

    # 与"单独计算"（无缓存的独立引擎）逐位一致
    eng_alone = VisibilityEngine(almanac24, beijing_point,
                                 cache=SharedSkyCache())
    az_s, el_s = eng_alone.all_az_el(times)
    assert np.array_equal(az_a, az_s)
    assert np.array_equal(el_a, el_s)
    assert np.array_equal(az_b, az_s)
    assert np.array_equal(el_b, el_s)

    # DOP/可见性也一致
    vis1, dops1, n1 = eng_a.states_at(times)
    vis2, dops2, n2 = eng_alone.states_at(times)
    assert np.array_equal(vis1, vis2)
    assert np.array_equal(n1, n2)
    assert np.array_equal(np.nan_to_num(dops1), np.nan_to_num(dops2))
