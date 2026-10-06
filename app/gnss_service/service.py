import copy

from .computation import ObservationEngine
from .diff import diff_results, window_to_dict
from .models import validate_plan_params
from .repositories import Repository
from .scheduler import JobScheduler
from .search import find_windows
from .timeutils import iso, parse_time, utc_now, validate_time_range
from .uuid import new_id
from .versions import build_almanac_document, build_point_version


class PlanningService:
    def __init__(self, repository: Repository, engine: ObservationEngine | None = None, max_workers: int = 2):
        self.repo = repository
        self.engine = engine or ObservationEngine()
        self.scheduler = JobScheduler(repository, max_workers=max_workers)

    def shutdown(self) -> None:
        self.scheduler.shutdown()

    def import_almanac(self, payload: dict, trigger: bool = True) -> dict:
        document = build_almanac_document(payload)
        self.repo.insert("almanacs", document)
        if trigger:
            self._replan_for_almanac(document["id"])
        return self.get_almanac(document["id"], include_satellites=True)

    def list_almanacs(self) -> list[dict]:
        documents = self.repo.find("almanacs")
        return [self.get_almanac(item["id"], include_satellites=False) for item in documents]

    def get_almanac(self, almanac_id: str, include_satellites: bool = True) -> dict:
        document = self.repo.get("almanacs", almanac_id)
        if document is None:
            raise KeyError("almanac not found")
        result = {"id": document["id"], "name": document["name"], "created_at": document["created_at"]}
        if include_satellites:
            result["satellites"] = copy.deepcopy(document["satellites"])
        return result

    def latest_almanac(self) -> dict:
        documents = self.repo.find("almanacs")
        if not documents:
            raise KeyError("no almanac has been imported")
        return max(documents, key=lambda item: item["created_at"])

    def upsert_point(self, point_id: str, payload: dict, trigger: bool = True) -> dict:
        existing = self.repo.get("points", point_id)
        if existing is None:
            point = {
                "id": point_id,
                "current_version_id": None,
                "created_at": iso(utc_now()),
                "updated_at": iso(utc_now()),
            }
            self.repo.insert("points", point)
            version_number = 1
        else:
            version = self.repo.get("point_versions", existing["current_version_id"])
            version_number = version["version"] + 1
        version = build_point_version(point_id, version_number, payload)
        version_id = version["id"]
        self.repo.insert("point_versions", version)
        self.repo.set_fields(
            "points",
            point_id,
            {"current_version_id": version_id, "updated_at": iso(utc_now())},
        )
        if trigger and existing is not None:
            self._replan_for_point(point_id, version_id)
        return self.get_point_version(version_id)

    def get_point_version(self, version_id: str) -> dict:
        version = self.repo.get("point_versions", version_id)
        if version is None:
            raise KeyError("point version not found")
        return {"id": version["id"], "point_id": version["point_id"], "version": version["version"], **version["data"], "created_at": version["created_at"]}

    def get_point(self, point_id: str) -> dict:
        point = self.repo.get("points", point_id)
        if point is None:
            raise KeyError("point not found")
        result = self.get_point_version(point["current_version_id"])
        result["updated_at"] = point["updated_at"]
        return result

    def list_points(self) -> list[dict]:
        return [self.get_point(item["id"]) for item in self.repo.find("points")]

    def _point_version_map(self, point_ids: list[str]) -> dict[str, dict]:
        result = {}
        for point_id in point_ids:
            point = self.repo.get("points", point_id)
            if point is None:
                raise KeyError(f"point {point_id} not found")
            version = self.repo.get("point_versions", point["current_version_id"])
            result[point_id] = version
        return result

    def query_instant(self, almanac_id: str, point_id: str, at: str) -> dict:
        almanac = self.repo.get("almanacs", almanac_id)
        if almanac is None:
            raise KeyError("almanac not found")
        point = self.repo.get("points", point_id)
        if point is None:
            raise KeyError("point not found")
        point_version = self.repo.get("point_versions", point["current_version_id"])
        observations = self.engine.observe(
            almanac, point_version["data"], at, almanac_id, point_version["id"]
        )
        dop = self.engine.evaluate(almanac, point_version["data"], at, almanac_id, point_version["id"])
        return {
            "time": iso(parse_time(at)),
            "almanac_version_id": almanac_id,
            "point_id": point_id,
            "point_version_id": point_version["id"],
            "visible_satellites": [
                {"id": item.satellite_id, "azimuth": item.azimuth, "elevation": item.elevation}
                for item in observations
            ],
            "satellite_count": dop.satellite_count,
            "available": dop.visible,
            "gdop": dop.gdop,
            "pdop": dop.pdop,
            "hdop": dop.hdop,
            "vdop": dop.vdop,
            "tdop": dop.tdop,
        }

    def create_plan(self, payload: dict) -> dict:
        point_ids = payload.get("point_ids")
        if not isinstance(point_ids, list) or not point_ids:
            raise ValueError("point_ids must be a non-empty list")
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point_ids must be unique")
        point_versions = self._point_version_map(point_ids)
        almanac_id = payload.get("almanac_version_id")
        if almanac_id is None:
            almanac_id = self.latest_almanac()["id"]
        if self.repo.get("almanacs", almanac_id) is None:
            raise KeyError("almanac not found")
        start, end = validate_time_range(payload["start"], payload["end"])
        params = validate_plan_params(payload)
        point_version_ids = {point_id: version["id"] for point_id, version in point_versions.items()}
        plan_id = new_id("plan")
        plan = {
            "id": plan_id,
            "point_ids": point_ids,
            "start": iso(start),
            "end": iso(end),
            "params": params,
            "almanac_version_id": almanac_id,
            "point_version_ids": point_version_ids,
            "active_job_id": None,
            "latest_result": None,
            "history_result": None,
            "created_at": iso(utc_now()),
        }
        self.repo.insert("plans", plan)
        job = self._start_job(plan, "initial")
        plan["active_job_id"] = job["id"]
        plan["job_id"] = job["id"]
        return copy.deepcopy(plan)

    def get_plan(self, plan_id: str) -> dict:
        plan = self.repo.get("plans", plan_id)
        if plan is None:
            raise KeyError("plan not found")
        return copy.deepcopy(plan)

    def list_plans(self) -> list[dict]:
        return self.repo.find("plans")

    def get_job(self, job_id: str) -> dict:
        job = self.repo.get("jobs", job_id)
        if job is None:
            raise KeyError("job not found")
        return copy.deepcopy(job)

    def cancel_job(self, job_id: str) -> dict:
        return self.scheduler.cancel(job_id)

    def _start_job(self, plan: dict, kind: str) -> dict:
        return self.scheduler.create_and_submit(plan, kind, self._calculate)

    def _replan_for_almanac(self, almanac_id: str) -> list[str]:
        now = utc_now()
        jobs = []
        for plan in self.repo.find("plans"):
            if parse_time(plan["end"]) < now:
                continue
            self.scheduler.supersede_active(plan, "superseded by newer almanac")
            plan = self.repo.get("plans", plan["id"])
            self.repo.set_fields("plans", plan["id"], {"almanac_version_id": almanac_id})
            plan = self.repo.get("plans", plan["id"])
            jobs.append(self._start_job(plan, "almanac_replan")["id"])
        return jobs

    def _replan_for_point(self, point_id: str, point_version_id: str) -> list[str]:
        now = utc_now()
        jobs = []
        for plan in self.repo.find("plans"):
            if point_id not in plan["point_ids"] or parse_time(plan["end"]) < now:
                continue
            self.scheduler.supersede_active(plan, "superseded by point mask revision")
            self.repo.set_fields(
                "plans",
                plan["id"],
                {f"point_version_ids.{point_id}": point_version_id},
            )
            plan = self.repo.get("plans", plan["id"])
            jobs.append(self._start_job(plan, "point_replan")["id"])
        return jobs

    def _calculate(self, job_id: str, should_cancel) -> dict:
        job = self.repo.get("jobs", job_id)
        plan = self.repo.get("plans", job["plan_id"])
        almanac = self.repo.get("almanacs", job["almanac_version_id"])
        start = parse_time(plan["start"])
        end = parse_time(plan["end"])
        params = job["params"]
        point_ids = job["point_ids"]
        windows_by_point = {}

        for point_index, point_id in enumerate(point_ids):
            version_id = job["point_version_ids"][point_id]
            point_version = self.repo.get("point_versions", version_id)
            point_data = point_version["data"]

            def available(at, _almanac=almanac, _point=point_data, _vid=version_id):
                result = self.engine.evaluate(
                    _almanac, _point, at, job["almanac_version_id"], _vid
                )
                return result.visible and result.gdop is not None and result.gdop <= params["gdop_threshold"]

            def gdop_value(at, _almanac=almanac, _point=point_data, _vid=version_id):
                return self.engine.evaluate(
                    _almanac, _point, at, job["almanac_version_id"], _vid
                ).gdop

            def progress(value, base=point_index, total=len(point_ids)):
                self.repo.set_fields("jobs", job_id, {"progress": (base + value) / total})

            windows = find_windows(
                start,
                end,
                available,
                gdop_value,
                params["gdop_threshold"],
                params["min_duration_seconds"],
                params.get("step_seconds"),
                progress=progress,
                should_cancel=should_cancel,
            )
            windows_by_point[point_id] = [window_to_dict(item) for item in windows]

        return {
            "plan_id": plan["id"],
            "almanac_version_id": job["almanac_version_id"],
            "point_version_ids": dict(job["point_version_ids"]),
            "params": dict(params),
            "start": iso(start),
            "end": iso(end),
            "windows_by_point": windows_by_point,
            "generated_at": iso(utc_now()),
        }

    def get_result(self, plan_id: str, historical: bool = False) -> dict | None:
        plan = self.get_plan(plan_id)
        key = "history_result" if historical else "latest_result"
        result = plan.get(key)
        return copy.deepcopy(result)

    def compare_results(self, old_result: dict | None, new_result: dict | None) -> dict:
        return diff_results(old_result, new_result)
