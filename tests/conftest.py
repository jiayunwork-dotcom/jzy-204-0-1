"""pytest 公共夹具：mongomock + Flask 测试客户端 + 仿真 24 星历书。"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import mongomock
import numpy as np
import pytest

from app.cache import SharedSkyCache
from app.mask import MaskProfile, Point
from app.orbit import MU_EARTH, Almanac, SatAlmanac
from app.repository import Repository
from app.scheduler import Scheduler


GPS_A = 26559800.0                 # 半长轴 [m]
GPS_SQRT_A = math.sqrt(GPS_A)
GPS_TOE_OFFSET = 0.0


def epoch_iso(dt) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def future_window():
    """相对今天较远的未来 2 天窗口，保证自动重规划被触发。"""
    base = int(time.time()) + 30 * 86400
    start = base - base % 86400       # 对齐到 UTC 零点
    return start, start + 2 * 86400


@pytest.fixture
def almanac24():
    """构造一份 24 颗 GPS 型卫星的历书：6 轨道面 × 4 星。"""
    sats = []
    inc = math.radians(55.0)
    n = math.sqrt(MU_EARTH / GPS_A ** 3)
    period = 2 * math.pi / n
    toe = 0.0
    plane_spacing = 2 * math.pi / 6
    for plane in range(6):
        raan = plane * plane_spacing
        for slot in range(4):
            ma = slot * math.pi / 2 + plane * 0.05
            sats.append(SatAlmanac(
                sat_id=f"G{plane+1:02d}{slot+1}",
                sqrt_a=GPS_SQRT_A,
                ecc=0.005 + 0.001 * ((plane + slot) % 3),
                inc=inc + math.radians(0.2) * (slot - 1.5),
                raan0=raan,
                raan_rate=-8e-10,
                argp=math.radians((30 * plane + 45 * slot) % 360),
                ma=ma,
                toe=toe,
            ))
    return Almanac(sats=tuple(sats), version=1)


@pytest.fixture
def beijing_point():
    # 北京附近一处控制点：轻微的非均匀遮挡轮廓
    mask = MaskProfile.create(
        points=[(0, 20), (45, 15), (90, 10), (135, 25), (180, 30),
                (225, 20), (270, 12), (315, 18)],
        cutoff_deg=7.0)
    return Point(point_id="BJ01", lon_deg=116.391, lat_deg=39.907,
                 height_m=50.0, mask=mask, version=1)


@pytest.fixture
def simple_point():
    return Point(point_id="P001", lon_deg=116.0, lat_deg=40.0,
                 height_m=0.0, mask=MaskProfile.create(cutoff_deg=10.0),
                 version=1)


@pytest.fixture
def repo(almanac24):
    client = mongomock.MongoClient()
    r = Repository(client["gnss_test"])
    r.setup_indexes()
    # 预置一版历书
    alm = almanac24
    r.insert_almanac(alm)
    # counters 校准：下一号为 2
    r.counters.update_one({"_id": "almanac_version"},
                          {"$set": {"seq": 1}}, upsert=True)
    return r


@pytest.fixture
def scheduler(repo):
    sched = Scheduler(repo, cache=SharedSkyCache(maxsize=100_000), workers=2)
    sched.startup()
    yield sched
    sched.shutdown(wait=True)


@pytest.fixture
def client(repo, scheduler):
    from app.api import create_app
    app = create_app(repo=repo, scheduler=scheduler)
    app.config["TESTING"] = True
    return app.test_client()


def wait_job_done(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.post(f"/api/jobs/{job_id}/wait?timeout=5")
        data = resp.get_json()
        if data["status"] in ("done", "failed", "canceled", "interrupted"):
            return data
    raise AssertionError(f"作业 {job_id} 等待超时")


def wait_jobs(scheduler, job_ids, timeout=30.0):
    """不经过 HTTP 直接等待调度器作业完成（返回最终作业文档列表）。"""
    deadline = time.time() + timeout
    out = []
    pending = list(job_ids)
    while time.time() < deadline and pending:
        still = []
        for jid in pending:
            doc = scheduler.get_job(jid)
            if doc["status"] in ("done", "failed", "canceled",
                                 "interrupted"):
                out.append(doc)
            else:
                still.append(jid)
        pending = still
        if pending:
            time.sleep(0.01)
    if pending:
        raise AssertionError(f"作业超时未结束: {pending}")
    return out
