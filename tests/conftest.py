import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from gnss_service.dop import DopResult, dop_from_directions
from gnss_service.repositories import MemoryRepository
from gnss_service.search import find_windows
from gnss_service.service import PlanningService


def direction(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (
        math.cos(el) * math.sin(az),
        math.cos(el) * math.cos(az),
        math.sin(el),
    )


@pytest.fixture
def repo():
    return MemoryRepository()


def wait_job(service: PlanningService, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = service.get_job(job_id)
        if job["status"] in {"completed", "failed", "cancelled", "superseded"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {service.get_job(job_id)}")


def wait_plan(service: PlanningService, plan_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        plan = service.get_plan(plan_id)
        if plan.get("latest_result") is not None:
            return plan
        if plan.get("active_job_id"):
            job = wait_job(service, plan["active_job_id"], timeout=max(0.01, deadline - time.time()))
            if job["status"] != "completed":
                raise AssertionError(job)
        time.sleep(0.01)
    raise AssertionError("plan result was not produced")
