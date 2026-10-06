from concurrent.futures import Future, ThreadPoolExecutor

from .timeutils import iso, utc_now
from .uuid import new_id


class JobScheduler:
    """In-process asynchronous job queue with cooperative cancellation."""

    def __init__(self, repository, max_workers: int = 2):
        self.repo = repository
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="planning")
        self._futures: dict[str, Future] = {}

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def is_finished(job: dict) -> bool:
        return job["status"] in {"completed", "failed", "cancelled", "superseded"}

    def is_cancelled(self, job_id: str) -> bool:
        job = self.repo.get("jobs", job_id)
        return job is None or job["status"] in {"cancelled", "superseded"}

    def create_and_submit(self, plan: dict, kind: str, calculator) -> dict:
        job = {
            "id": new_id("job"),
            "plan_id": plan["id"],
            "point_ids": list(plan["point_ids"]),
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "almanac_version_id": plan["almanac_version_id"],
            "point_version_ids": dict(plan["point_version_ids"]),
            "params": dict(plan["params"]),
            "created_at": iso(utc_now()),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        self.repo.insert("jobs", job)
        self.repo.set_fields("plans", plan["id"], {"active_job_id": job["id"]})
        self._futures[job["id"]] = self.executor.submit(self._run, job["id"], calculator)
        return job

    def supersede_active(self, plan: dict, reason: str) -> None:
        active_id = plan.get("active_job_id")
        if not active_id:
            return
        active = self.repo.get("jobs", active_id)
        if active and not self.is_finished(active):
            self.repo.set_fields(
                "jobs",
                active_id,
                {"status": "superseded", "completed_at": iso(utc_now()), "error": reason},
            )
            future = self._futures.get(active_id)
            if future is not None:
                future.cancel()
        self.repo.set_fields("plans", plan["id"], {"active_job_id": None})

    def cancel(self, job_id: str) -> dict:
        job = self.repo.get("jobs", job_id)
        if job is None:
            raise KeyError("job not found")
        if not self.is_finished(job):
            self.repo.set_fields(
                "jobs",
                job_id,
                {"status": "cancelled", "completed_at": iso(utc_now())},
            )
            future = self._futures.get(job_id)
            if future is not None:
                future.cancel()
            plan_id = job.get("plan_id")
            if plan_id:
                plan = self.repo.get("plans", plan_id)
                if plan and plan.get("active_job_id") == job_id:
                    self.repo.set_fields("plans", plan_id, {"active_job_id": None})
        return self.repo.get("jobs", job_id)

    def _run(self, job_id: str, calculator) -> None:
        job = self.repo.get("jobs", job_id)
        if job is None or self.is_finished(job):
            return
        plan = self.repo.get("plans", job["plan_id"])
        if plan is None or plan.get("active_job_id") != job_id:
            return
        self.repo.set_fields("jobs", job_id, {"status": "running", "started_at": iso(utc_now())})
        try:
            result = calculator(job_id, lambda: self.is_cancelled(job_id))
            plan_id = job["plan_id"]
            old_plan = self.repo.get("plans", plan_id)
            history = old_plan.get("latest_result")
            from .diff import diff_results

            result["diff"] = diff_results(history, result)
            committed = self.repo.complete_plan_job(
                job_id,
                plan_id,
                {"status": "completed", "progress": 1.0, "completed_at": iso(utc_now())},
                result,
                history,
            )
            if not committed and self.repo.get("jobs", job_id)["status"] == "running":
                self.repo.set_fields(
                    "jobs",
                    job_id,
                    {"status": "superseded", "completed_at": iso(utc_now())},
                )
        except InterruptedError:
            self.repo.set_fields("jobs", job_id, {"status": "cancelled", "completed_at": iso(utc_now())})
        except Exception as exc:
            self.repo.set_fields(
                "jobs",
                job_id,
                {"status": "failed", "completed_at": iso(utc_now()), "error": str(exc)},
            )
