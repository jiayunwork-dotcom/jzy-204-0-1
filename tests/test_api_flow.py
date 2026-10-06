"""完整 HTTP 端到端流程测试（mongomock 持久化 + 真实调度器）。"""
import math
import time
from datetime import datetime, timezone

from .conftest import epoch_iso, wait_job_done


def _iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def test_full_flow_over_http(client, repo, scheduler, almanac24,
                             future_window):
    start, end = future_window

    # 健康检查
    assert client.get("/api/health").get_json()["status"] == "ok"

    # 历书版本列表（夹具已导入 v1）
    r = client.get("/api/almanacs")
    versions = [a["version"] for a in r.get_json()["almanacs"]]
    assert versions == [1]
    r = client.get("/api/almanacs/1")
    assert len(r.get_json()["satellites"]) == 24

    # 建点
    r = client.put("/api/points", json={
        "point_id": "HTTP1", "lon_deg": 116.39, "lat_deg": 39.9,
        "height_m": 44.0,
        "mask": {"cutoff_deg": 8.0,
                 "points": [[0, 20], [90, 15], [180, 25], [270, 10]]}})
    assert r.status_code == 200
    assert r.get_json()["point_version"] == 1

    # 单时刻可见性查询
    r = client.post("/api/sky/observe", json={
        "point_id": "HTTP1", "time": _iso(start)})
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["satellites"]) == 24
    assert data["almanac_version"] == 1
    assert data["point_version"] == 1
    vis = [s for s in data["satellites"] if s["visible"]]
    assert len(vis) == data["n_visible"]
    if data["available"]:
        assert data["n_visible"] >= 4
        assert data["dop"]["gdop"] is not None
    else:
        assert data["dop"]["gdop"] is None

    # 提交规划
    r = client.post("/api/plans", json={
        "point_ids": ["HTTP1"],
        "date_start": _iso(start), "date_end": _iso(end),
        "gdop_threshold": 6.0, "min_duration_s": 300,
        "scan_step_s": 120})
    assert r.status_code == 202
    plan_id = r.get_json()["plan_id"]
    job_id = r.get_json()["job_id"]

    final = wait_job_done(client, job_id, timeout=60)
    assert final["status"] == "done", final

    # 结果中绑定历书/点位/参数版本
    r = client.get(f"/api/plans/{plan_id}/result")
    assert r.status_code == 200
    result = r.get_json()
    assert result["almanac_version"] == 1
    point_res = result["results"][0]
    assert point_res["almanac_version"] == 1
    assert point_res["point_version"] == 1
    assert point_res["params"]["gdop_threshold"] == 6.0
    assert result["n_windows"] >= 1
    for w in point_res["windows"]:
        assert w["end_epoch"] > w["start_epoch"]
        assert w["duration_s"] == w["end_epoch"] - w["start_epoch"]
        assert w["gdop_min"] <= 6.0 + 1e-9
        assert w["start_epoch"] <= w["gdop_min_time_epoch"] < \
            w["end_epoch"]

    # 历史接口：尚无历史
    r = client.get(f"/api/plans/{plan_id}/history")
    assert r.get_json()["history"] == []

    # 导入新版历书（相位平移）-> 自动重规划
    shifted = []
    for s in almanac24.sats:
        d = s.to_dict()
        d["ma"] = s.ma + math.pi
        shifted.append(d)
    r = client.post("/api/almanacs", json={
        "source": "shifted", "satellites": shifted})
    assert r.status_code == 201
    assert r.get_json()["almanac_version"] == 2
    replan_jobs = r.get_json()["replan_jobs"]
    assert len(replan_jobs) == 1
    final2 = wait_job_done(client, replan_jobs[0], timeout=60)
    assert final2["status"] == "done"

    # 差异接口（默认与上一版历史比）
    r = client.post(f"/api/plans/{plan_id}/diff", json={})
    assert r.status_code == 200
    d = r.get_json()
    assert d["old_almanac_version"] == 1
    assert d["new_almanac_version"] == 2
    assert "HTTP1" in d["points"]
    assert d["totals"]["unchanged"] + d["totals"]["disappeared"] + \
        d["totals"]["added"] + d["totals"]["shortened"] + \
        d["totals"]["shifted"] >= 1

    # 历史此时为 v1
    r = client.get(f"/api/plans/{plan_id}/history")
    hist = r.get_json()["history"]
    assert len(hist) == 1
    assert hist[0]["almanac_version"] == 1


def test_job_cancel_over_http(client, repo, scheduler, almanac24,
                              future_window, monkeypatch):
    start, end = future_window
    client.put("/api/points", json={
        "point_id": "C1", "lon_deg": 116.39, "lat_deg": 39.9})
    client.put("/api/points", json={
        "point_id": "C2", "lon_deg": 117.0, "lat_deg": 40.0})

    import threading
    gate = threading.Event()
    reached = threading.Event()

    def hook(pid):
        if pid == "C1":
            reached.set()
            gate.wait(timeout=10)

    scheduler.per_point_hook = hook
    r = client.post("/api/plans", json={
        "point_ids": ["C1", "C2"],
        "date_start": _iso(start), "date_end": _iso(end),
        "gdop_threshold": 6.0, "min_duration_s": 60,
        "scan_step_s": 300})
    plan_id = r.get_json()["plan_id"]
    job_id = r.get_json()["job_id"]
    assert reached.wait(5)
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    gate.set()
    final = wait_job_done(client, job_id)
    assert final["status"] == "canceled"
    # 取消后不写入任何结果
    assert repo.results.find_one({"plan_id": plan_id}) is None
