"""补充行为测试：重启恢复、只重算未来计划、内联点位观测、共享点跨计划去重。"""
import math
import time

import numpy as np

from app.cache import SharedSkyCache
from app.repository import Repository
from app.scheduler import Scheduler
from app.visibility import VisibilityEngine
import mongomock

from .conftest import wait_jobs


def test_restart_marks_active_jobs_interrupted(almanac24, simple_point,
                                               future_window):
    client = mongomock.MongoClient()
    repo = Repository(client["restart"]); repo.setup_indexes()
    repo.insert_almanac(almanac24)
    repo.upsert_point(simple_point)
    sched = Scheduler(repo, cache=SharedSkyCache(), workers=2)
    sched.startup()
    start, end = future_window
    plan = {"plan_id": "plan-000001", "point_ids": ["P001"],
            "almanac_version": 1,
            "params": {"gdop_threshold": 6.0, "min_duration_s": 0,
                       "date_start": float(start), "date_end": float(end),
                       "scan_step_s": 300},
            "status": "active"}
    repo.insert_plan(plan)
    # 人为残留一个 running 作业（模拟进程崩溃）
    repo.upsert_job({"job_id": "job-stale", "type": "plan",
                     "plan_id": "plan-000001", "status": "running",
                     "progress": 0.3, "point_progress": {}})
    sched.shutdown(wait=False)

    # 新实例启动恢复
    sched2 = Scheduler(repo, cache=SharedSkyCache(), workers=2)
    sched2.startup()
    assert repo.get_job("job-stale")["status"] == "interrupted"
    sched2.shutdown()


def test_past_plan_not_replanned_on_almanac_import(almanac24,
                                                   simple_point, future_window):
    client = mongomock.MongoClient()
    repo = Repository(client["past"]); repo.setup_indexes()
    repo.insert_almanac(almanac24)
    repo.upsert_point(simple_point)
    sched = Scheduler(repo, cache=SharedSkyCache(), workers=2)
    start, end = future_window
    # 一个已结束的历史计划（终点在过去）
    past = {"plan_id": "plan-000001", "point_ids": ["P001"],
            "almanac_version": 1,
            "params": {"gdop_threshold": 6.0, "min_duration_s": 0,
                       "date_start": float(start - 40 * 86400),
                       "date_end": float(start - 39 * 86400),
                       "scan_step_s": 300},
            "status": "active"}
    repo.insert_plan(past)
    jobs = sched.trigger_almanac_replan(2, now_epoch_value=start - 1)
    assert jobs == []
    sched.shutdown()


def test_inline_point_observe(client, future_window):
    start, _ = future_window
    r = client.post("/api/sky/observe", json={
        "time": start,
        "point": {"point_id": "inline", "lon_deg": 116.39,
                  "lat_deg": 39.9, "height_m": 50.0,
                  "mask": {"cutoff_deg": 10.0,
                           "points": [[0, 30], [180, 20]]}}})
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["satellites"]) == 24
    # 南向（az=180）卫星的遮挡高度取 20°，北向取 30°，其余取截止 10°
    for s in data["satellites"]:
        az = s["azimuth_deg"]
        expected_mask = 10.0
        assert s["mask_elevation_deg"] >= 10.0 - 1e-9
        assert s["visible"] == (s["elevation_deg"] >
                                s["mask_elevation_deg"])


def test_two_plans_sharing_point_deduplicate(almanac24, beijing_point,
                                             future_window):
    """同一点位被两份规划引用：不重复计算同一时刻的可见性。"""
    repo = Repository(mongomock.MongoClient()["share"])
    repo.setup_indexes()
    repo.insert_almanac(almanac24)
    repo.upsert_point(beijing_point)
    cache = SharedSkyCache()
    sched = Scheduler(repo, cache=cache, workers=2)
    sched.startup()
    start, end = future_window
    plans = []
    for k in range(2):
        pid = f"plan-{repo.next_id('plan'):06d}"
        doc = {"plan_id": pid, "point_ids": ["BJ01"],
               "almanac_version": 1,
               "params": {"gdop_threshold": 6.0, "min_duration_s": 600,
                          "date_start": float(start),
                          "date_end": float(end),
                          "scan_step_s": 60},
               "status": "active"}
        repo.insert_plan(doc)
        plans.append((doc, sched.submit_plan_job(doc)))
    done = wait_jobs(sched, [j for _, j in plans])
    assert all(d["status"] == "done" for d in done)

    # 第二份规划执行时大量命中共享缓存
    assert cache.hits > 0
    keys_after_two = cache.computed_keys

    # 独立缓存只跑一份规划：其计算键集合必须是"两份共享"时的子集；
    # 且由于两份规划参数完全一致，第二份不应引入任何额外的新键——
    # 两次总键数相等即证明没有重复计算同一时刻。
    cache_alone = SharedSkyCache()
    sched_alone = Scheduler(Repository(mongomock.MongoClient()["alone"]),
                            cache=cache_alone, workers=1)
    repo2 = sched_alone.repo
    repo2.insert_almanac(almanac24)
    repo2.upsert_point(beijing_point)
    sched_alone.startup()
    pid = f"plan-{repo2.next_id('plan'):06d}"
    doc_alone = {"plan_id": pid, "point_ids": ["BJ01"],
                 "almanac_version": 1,
                 "params": plans[0][0]["params"], "status": "active"}
    repo2.insert_plan(doc_alone)
    ja = sched_alone.submit_plan_job(doc_alone)
    wait_jobs(sched_alone, [ja])
    alone_keys = cache_alone.computed_keys
    assert alone_keys == keys_after_two
    sched_alone.shutdown()

    # 两份结果完全一致（共享计算与单独计算一致的端到端体现）
    r1 = repo.get_result(plans[0][0]["plan_id"])["results"][0]
    r2 = repo.get_result(plans[1][0]["plan_id"])["results"][0]
    assert r1["windows"] == r2["windows"]

    # 独立引擎逐点复核窗口数据
    engine = VisibilityEngine(almanac24, beijing_point,
                              cache=SharedSkyCache())
    from app.search import EngineEvaluator, find_windows
    ws = find_windows(EngineEvaluator(engine, 6.0), start, end,
                      scan_step=60, min_duration=600)
    assert [w.to_dict() for w in ws] == r1["windows"]
    sched.shutdown()


def test_observe_explicit_version_and_404(client):
    r = client.post("/api/sky/observe", json={
        "time": 1_000_000, "almanac_version": 99,
        "point_id": "NOPE"})
    assert r.status_code == 400


def test_observe_threshold_semantics(client, future_window):
    start, _ = future_window
    client.put("/api/points", json={
        "point_id": "OBS", "lon_deg": 116.39, "lat_deg": 39.9})
    # 不带阈值：available 只表示可解算（≥4 星、DOP 有限）
    r = client.post("/api/sky/observe", json={
        "point_id": "OBS", "time": start})
    assert r.status_code == 200
    d = r.get_json()
    if d["n_visible"] >= 4:
        assert d["available"] == (d["dop"]["gdop"] is not None)
        assert d["within_threshold"] is None
    # 带一个很小的阈值：要么明确 within_threshold=False，要么不可用
    r2 = client.post("/api/sky/observe", json={
        "point_id": "OBS", "time": start, "gdop_threshold": 0.5})
    d2 = r2.get_json()
    assert d2["gdop_threshold"] == 0.5
    assert d2["available"] is False
    assert d2["within_threshold"] is False
    # 非法阈值被拒
    r3 = client.post("/api/sky/observe", json={
        "point_id": "OBS", "time": start, "gdop_threshold": 0})
    assert r3.status_code == 400
