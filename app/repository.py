"""MongoDB 持久化层与版本号分配。

集合：
    counters           原子自增版本号（almanac 全局、point 按 point_id）
    almanacs           历书版本（每次导入新版本，旧版保留）
    points             点位当前版本（含遮挡轮廓）
    point_versions     点位历史版本
    plans              规划定义（点位集合 + 参数快照）
    plan_results       每份规划的"当前结果"（最新历书）与"历史结果"列表
    jobs               异步作业
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from .errors import ValidationError
from .mask import Point
from .orbit import Almanac


def _utcnow():
    return datetime.now(tz=timezone.utc)


class _LockedCollection:
    """对底层集合操作加进程内锁的薄封装。

    mongomock 不是线程安全的（作业线程高频 replace_one 进度与请求线程
    轮询并发时会出现文档瞬时丢失）；真 pymongo 连接本身线程安全，
    这把锁对其没有正确性影响，仅把同进程内的访问串行化。
    """

    __slots__ = ("_coll", "_lock")

    def __init__(self, coll, lock):
        object.__setattr__(self, "_coll", coll)
        object.__setattr__(self, "_lock", lock)

    def _run(self, name, args, kwargs):
        with self._lock:
            result = getattr(self._coll, name)(*args, **kwargs)
        # 游标惰性求值会在锁外访问存储，这里立即物化为列表，
        # 再用一个轻量包装返回以兼容 sort(...) 等链式调用。
        if hasattr(result, "next") and hasattr(result, "__iter__") and \
                not isinstance(result, (list, dict)):
            try:
                materialized = list(result)
            except TypeError:
                return result
            return materialized
        return result

    def __getattr__(self, name):
        coll = object.__getattribute__(self, "_coll")
        attr = getattr(coll, name)
        if not callable(attr):
            # 属性（不太用得到）直接在锁内读取
            with object.__getattribute__(self, "_lock"):
                return getattr(coll, name)

        def method(*args, **kwargs):
            return self._run(name, args, kwargs)
        return method


class Repository:
    def __init__(self, db):
        self.db = db
        self._lock = threading.RLock()
        self.counters = _LockedCollection(db["counters"], self._lock)
        self.almanacs = _LockedCollection(db["almanacs"], self._lock)
        self.points = _LockedCollection(db["points"], self._lock)
        self.point_versions = _LockedCollection(
            db["point_versions"], self._lock)
        self.plans = _LockedCollection(db["plans"], self._lock)
        self.results = _LockedCollection(db["plan_results"], self._lock)
        self.jobs = _LockedCollection(db["jobs"], self._lock)

    def setup_indexes(self):
        # mongomock 也支持 create_index；重复创建无副作用
        self.almanacs.create_index("version", unique=True)
        self.points.create_index("point_id", unique=True)
        self.point_versions.create_index(
            [("point_id", 1), ("version", 1)], unique=True)
        self.plans.create_index("plan_id", unique=True)
        self.results.create_index("plan_id", unique=True)
        self.jobs.create_index("job_id", unique=True)

    # ----------------------------------------------------------- 版本号
    def next_almanac_version(self) -> int:
        doc = self.counters.find_one_and_update(
            {"_id": "almanac_version"},
            {"$inc": {"seq": 1}},
            upsert=True, return_document=True)
        return int(doc["seq"])

    def next_point_version(self, point_id: str) -> int:
        doc = self.counters.find_one_and_update(
            {"_id": f"point_version:{point_id}"},
            {"$inc": {"seq": 1}}, upsert=True, return_document=True)
        return int(doc["seq"])

    def next_id(self, name: str) -> int:
        doc = self.counters.find_one_and_update(
            {"_id": f"id:{name}"}, {"$inc": {"seq": 1}}, upsert=True,
            return_document=True)
        return int(doc["seq"])

    # ------------------------------------------------------------- 历书
    def insert_almanac(self, almanac: Almanac) -> dict:
        doc = almanac.to_dict()
        doc["created_at"] = _utcnow()
        self.almanacs.insert_one(doc)
        return doc

    def get_almanac(self, version: int) -> Almanac:
        doc = self.almanacs.find_one({"version": int(version)})
        if doc is None:
            raise ValidationError(f"历书版本 {version} 不存在",
                                  field="almanac_version")
        return Almanac.from_dict(doc)

    def latest_almanac_version(self) -> int | None:
        doc = self.almanacs.find_one(sort=[("version", -1)])
        return None if doc is None else int(doc["version"])

    def get_latest_almanac(self) -> Almanac | None:
        v = self.latest_almanac_version()
        return None if v is None else self.get_almanac(v)

    def list_almanacs(self) -> list[dict]:
        return [{
            "version": int(d["version"]),
            "source": d.get("source"),
            "n_sat": len(d.get("satellites", [])),
            "created_at": d.get("created_at"),
        } for d in self.almanacs.find(sort=[("version", -1)])]

    # ------------------------------------------------------------- 点位
    def upsert_point(self, point: Point) -> Point:
        doc = point.to_dict()
        doc["updated_at"] = _utcnow()
        # 当前表
        self.points.replace_one({"point_id": point.point_id}, doc,
                                upsert=True)
        # 历史表
        self.point_versions.replace_one(
            {"point_id": point.point_id, "version": point.version},
            dict(doc), upsert=True)
        return point

    def get_point(self, point_id: str, version: int | None = None) -> Point:
        if version is not None:
            doc = self.point_versions.find_one(
                {"point_id": point_id, "version": int(version)})
        else:
            doc = self.points.find_one({"point_id": point_id})
        if doc is None:
            raise ValidationError(f"点位 {point_id} 不存在",
                                  field="point_id")
        return Point.from_dict(doc)

    def list_points(self) -> list[dict]:
        return [{
            "point_id": d["point_id"],
            "version": int(d["version"]),
            "lon_deg": d["lon_deg"],
            "lat_deg": d["lat_deg"],
            "height_m": d["height_m"],
            "mask": d["mask"],
        } for d in self.points.find(sort=[("point_id", 1)])]

    # ------------------------------------------------------------- 规划
    def insert_plan(self, plan_doc: dict) -> dict:
        plan_doc["created_at"] = _utcnow()
        self.plans.replace_one({"plan_id": plan_doc["plan_id"]}, plan_doc,
                               upsert=True)
        return plan_doc

    def upsert_plan(self, plan_doc: dict) -> dict:
        plan_doc["updated_at"] = _utcnow()
        self.plans.replace_one({"plan_id": plan_doc["plan_id"]}, plan_doc,
                               upsert=True)
        return plan_doc

    def get_plan(self, plan_id: str) -> dict:
        doc = self.plans.find_one({"plan_id": plan_id})
        if doc is None:
            raise ValidationError(f"规划 {plan_id} 不存在", field="plan_id")
        return doc

    def list_future_plans(self, now_epoch: float) -> list[dict]:
        """日期范围终点仍在未来的规划（自动重规划候选）。"""
        return list(self.plans.find(
            {"params.date_end": {"$gt": float(now_epoch)}}))

    def plans_using_point(self, point_id: str) -> list[dict]:
        return list(self.plans.find({"point_ids": point_id}))

    # -------------------------------------------------------- 规划结果
    def save_result(self, plan_id: str, current: dict, job_id: str,
                    triggered_by: str = "manual") -> dict | None:
        """写入一次规划结果。

        current 为本次逐点结果列表（每项含 point_id、almanac_version、
        point_version、windows）。整份文档级替换；若该计划已有相同或更新
        历书版本的当前结果（例如旧作业与新作业竞争），拒绝写入旧版。
        返回写入后的文档，若因版本过旧被拒则返回 None。
        """
        alm_versions = {int(p["almanac_version"]) for p in current}
        new_alm_version = max(alm_versions)
        now = _utcnow()
        doc = self.results.find_one({"plan_id": plan_id})
        if doc is not None:
            cur_ver = int(doc.get("almanac_version", -1))
            if new_alm_version < cur_ver:
                return None
            history = list(doc.get("history", []))
            # 历史只保留上一份结果（即"一份历史"）
            history_entry = {
                "almanac_version": cur_ver,
                "results": doc.get("results", []),
                "job_id": doc.get("job_id"),
                "created_at": doc.get("created_at"),
                "triggered_by": doc.get("triggered_by", "manual"),
            }
            history = [history_entry]
        else:
            history = []
        new_doc = {
            "plan_id": plan_id,
            "almanac_version": new_alm_version,
            "results": current,
            "history": history,
            "job_id": job_id,
            "triggered_by": triggered_by,
            "created_at": now,
            "updated_at": now,
        }
        self.results.replace_one({"plan_id": plan_id}, new_doc, upsert=True)
        return new_doc

    def get_result(self, plan_id: str) -> dict:
        doc = self.results.find_one({"plan_id": plan_id})
        if doc is None:
            raise ValidationError(f"规划 {plan_id} 还没有结果",
                                  field="plan_id")
        return doc

    def list_history(self, plan_id: str) -> list[dict]:
        doc = self.results.find_one({"plan_id": plan_id})
        if doc is None:
            return []
        return list(doc.get("history", []))

    # --------------------------------------------------------------- 作业
    def upsert_job(self, job: dict) -> dict:
        self.jobs.replace_one({"job_id": job["job_id"]}, job, upsert=True)
        return job

    def get_job(self, job_id: str) -> dict | None:
        return self.jobs.find_one({"job_id": job_id})

    def list_active_jobs(self) -> list[dict]:
        return list(self.jobs.find({"status": {"$in": ["queued", "running"]}}))

    def mark_stale_jobs_interrupted(self) -> int:
        """服务重启后，把仍在排队/运行的作业标记为 interrupted。"""
        res = self.jobs.update_many(
            {"status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "interrupted", "updated_at": _utcnow()}})
        return res.modified_count
