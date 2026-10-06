"""异步作业调度：提交、进度查询、取消、历书/点位更新后的自动重规划与取代。

并发模型：进程内 ThreadPoolExecutor。作业状态持久化在 MongoDB，
服务重启后未完成作业统一标记为 interrupted（不保留半成品结果）。

自动重规划与取代
----------------
* 导入新历书：对所有日期范围仍在未来的规划，用最新历书发起重规划
  （trigger=almanac）。新作业入队前，取消该计划上尚未完成的同类作业；
  因而连续两次历书更新时，旧的重规划作业被取消让位，最终每份规划只
  保留基于最新历书的当前结果和一份历史。
* 修改点位遮挡轮廓：对引用该点位且仍在未来的规划同样发起重规划。
* 已取消的作业在写入结果前抛出 JobAborted，绝不写入部分结果。
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .cache import SharedSkyCache
from .diff import diff_results
from .errors import JobAborted, JobError, ValidationError
from .planning import run_planning
from .timeutils import now_epoch


def _utcnow():
    return datetime.now(tz=timezone.utc)


class Scheduler:
    def __init__(self, repo, cache: SharedSkyCache | None = None,
                 workers: int = 4, per_point_hook=None):
        self.repo = repo
        self.cache = cache or SharedSkyCache()
        self.per_point_hook = per_point_hook   # 测试注入（如延迟）
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers),
            thread_name_prefix="plan")
        self._cancel = threading.local()
        self._cancel_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._shutdown = False

    # ----------------------------------------------------------- 生命周期
    def startup(self):
        """重启恢复：把上次未完成的作业标记为 interrupted。"""
        self.repo.mark_stale_jobs_interrupted()

    def shutdown(self, wait: bool = True):
        self._shutdown = True
        with self._events_lock:
            for ev in self._cancel_events.values():
                ev.set()
        self._executor.shutdown(wait=wait)

    # --------------------------------------------------------------- 提交
    def submit_plan_job(self, plan_doc: dict, triggered_by: str = "manual",
                        replacing: bool = True) -> str:
        if self._shutdown:
            raise JobError("调度器已关闭")
        job_id = f"job-{uuid.uuid4().hex}"
        # 取代：取消同一计划上尚未完成的自动/手动重规划作业
        if replacing:
            self._supersede_for_plan(plan_doc["plan_id"])
        event = threading.Event()
        with self._events_lock:
            self._cancel_events[job_id] = event
        job = {
            "job_id": job_id,
            "type": "plan",
            "plan_id": plan_doc["plan_id"],
            "status": "queued",
            "triggered_by": triggered_by,
            "almanac_version": plan_doc.get("almanac_version"),
            "progress": 0.0,
            "point_progress": {},
            "error": None,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
        }
        self.repo.upsert_job(job)
        self._executor.submit(self._run_job, job_id, plan_doc, event)
        return job_id

    def _supersede_for_plan(self, plan_id: str) -> list[str]:
        """取消某计划所有未完成作业（被新历书/新点位版本取代）。"""
        canceled = []
        for j in self.repo.list_active_jobs():
            if j.get("plan_id") == plan_id and j.get("type") == "plan":
                self._mark_canceled(j["job_id"], reason="superseded")
                canceled.append(j["job_id"])
        return canceled

    # --------------------------------------------------------------- 取消
    def cancel(self, job_id: str, reason: str = "user_canceled") -> dict:
        job = self.repo.get_job(job_id)
        if job is None:
            raise ValidationError(f"作业 {job_id} 不存在", field="job_id")
        if job["status"] in ("done", "canceled", "failed", "interrupted"):
            raise JobError(f"作业已结束（{job['status']}），不能取消")
        return self._mark_canceled(job_id, reason)

    def _mark_canceled(self, job_id: str, reason: str) -> dict:
        with self._events_lock:
            ev = self._cancel_events.get(job_id)
            if ev is not None:
                ev.set()
        doc = self.repo.get_job(job_id)
        if doc is not None and doc["status"] in ("queued", "running"):
            doc["status"] = "canceled"
            doc["cancel_reason"] = reason
            doc["updated_at"] = _utcnow()
            self.repo.upsert_job(doc)
        return doc

    # ------------------------------------------------------------- 查询
    def get_job(self, job_id: str) -> dict:
        doc = self.repo.get_job(job_id)
        if doc is None:
            raise ValidationError(f"作业 {job_id} 不存在", field="job_id")
        return self._public_job(doc)

    def _public_job(self, doc: dict) -> dict:
        return {k: v for k, v in doc.items() if k != "_id"}

    # ----------------------------------------------------------- 作业主体
    def _run_job(self, job_id: str, plan_doc: dict,
                 cancel_event: threading.Event):
        job = self.repo.get_job(job_id)
        if job is None or job["status"] == "canceled":
            return  # 入队前/运行前已被取代
        job["status"] = "running"
        job["started_at"] = _utcnow()
        job["updated_at"] = _utcnow()
        self.repo.upsert_job(job)

        def is_cancelled():
            return cancel_event.is_set()

        point_ids = list(plan_doc["point_ids"])
        total = max(1, len(point_ids))
        per_point = {}
        results = []
        try:
            alm_version = int(plan_doc["almanac_version"])
            almanac = self.repo.get_almanac(alm_version)
            params = plan_doc["params"]
            # 点位版本快照：作业开始时取每个点位的当前版本
            point_versions = {}
            points = []
            for pid in point_ids:
                p = self.repo.get_point(pid)
                point_versions[pid] = p.version
                points.append(p)

            for idx, point in enumerate(points):
                if is_cancelled():
                    raise JobAborted()

                def on_progress(frac, _idx=idx, _pid=point.point_id):
                    per_point[_pid] = frac
                    cur = self.repo.get_job(job_id)
                    if cur is not None and cur["status"] == "running":
                        cur["progress"] = (_idx + float(frac)) / total
                        cur["point_progress"] = dict(per_point)
                        cur["updated_at"] = _utcnow()
                        self.repo.upsert_job(cur)

                res = run_planning(
                    almanac, point, params, self.cache,
                    on_progress=on_progress,
                    is_cancelled=is_cancelled,
                    per_point_hook=self.per_point_hook)
                results.append(res)
                per_point[point.point_id] = 1.0

            if is_cancelled():
                raise JobAborted()

            # 原子语义的最后保障：只接受不旧于当前结果历书版本的写入
            saved = self.repo.save_result(
                plan_doc["plan_id"], results, job_id,
                triggered_by=job.get("triggered_by", "manual"))
            if saved is None:
                # 已被更新历书版本的作业抢先提交：本作业让位为 canceled
                self._mark_canceled(job_id, reason="superseded_by_newer")
                return

            cur = self.repo.get_job(job_id)
            cur["status"] = "done"
            cur["progress"] = 1.0
            cur["point_progress"] = {pid: 1.0 for pid in point_ids}
            cur["n_windows"] = sum(len(r["windows"]) for r in results)
            cur["finished_at"] = _utcnow()
            cur["updated_at"] = _utcnow()
            # 与上一版的差异（便于作业完成后直接告知变化）
            history = saved.get("history", [])
            if history:
                cur["diff_summary"] = diff_results(
                    history[-1]["results"], results)["totals"]
            else:
                cur["diff_summary"] = None
            self.repo.upsert_job(cur)
        except JobAborted:
            # 取消：不写任何结果
            doc = self.repo.get_job(job_id)
            if doc is not None and doc["status"] not in ("done", "canceled"):
                doc["status"] = "canceled"
                doc.setdefault("cancel_reason", "canceled")
                doc["updated_at"] = _utcnow()
                self.repo.upsert_job(doc)
        except Exception as exc:  # noqa: BLE001 - 作业内任何异常都落库
            doc = self.repo.get_job(job_id)
            if doc is not None and doc["status"] not in ("canceled",):
                doc["status"] = "failed"
                doc["error"] = f"{type(exc).__name__}: {exc}"
                doc["updated_at"] = _utcnow()
                self.repo.upsert_job(doc)
        finally:
            with self._events_lock:
                self._cancel_events.pop(job_id, None)

    # --------------------------------------------------- 历书/点位触发
    def trigger_almanac_replan(self, new_almanac_version: int,
                               now_epoch_value: float | None = None,
                               params_override: dict | None = None) -> list[str]:
        """新历书导入后，对仍在未来的规划自动发起重规划。返回作业号列表。"""
        ref = now_epoch_value if now_epoch_value is not None else now_epoch()
        job_ids = []
        for plan in self.repo.list_future_plans(ref):
            # 同历书版本无需重算
            if int(plan.get("almanac_version", -1)) == int(
                    new_almanac_version):
                continue
            plan["almanac_version"] = int(new_almanac_version)
            self.repo.upsert_plan(plan)
            job_ids.append(self.submit_plan_job(
                plan, triggered_by="almanac"))
        return job_ids

    def trigger_point_replan(self, point_id: str,
                             new_point_version: int,
                             now_epoch_value: float | None = None) -> list[str]:
        """点位（含遮挡轮廓）更新后，重算引用它且仍在未来的规划。"""
        ref = now_epoch_value if now_epoch_value is not None else now_epoch()
        latest_alm = self.repo.latest_almanac_version()
        job_ids = []
        for plan in self.repo.plans_using_point(point_id):
            if float(plan["params"]["date_end"]) <= ref:
                continue
            if latest_alm is not None:
                plan["almanac_version"] = int(latest_alm)
            self.repo.upsert_plan(plan)
            job_ids.append(self.submit_plan_job(
                plan, triggered_by=f"point:{point_id}"))
        return job_ids
