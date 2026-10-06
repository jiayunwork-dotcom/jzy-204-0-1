from datetime import datetime, timedelta, timezone
import threading
import time

from conftest import wait_job, wait_plan
from gnss_service.computation import ObservationEngine
from gnss_service.dop import DopResult
from gnss_service.repositories import MemoryRepository
from gnss_service.service import PlanningService


BASE = datetime(2026, 10, 6, tzinfo=timezone.utc)


def second(n):
    return BASE + timedelta(seconds=n)


def almanac_payload():
    reference = (BASE - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    return {
        "name": "test",
        "satellites": [
            {
                "id": f"S{i}",
                "semi_major_axis": 26559800.0 + i * 1000,
                "eccentricity": 0.001 + i * 0.0001,
                "inclination": 55.0 + i,
                "raan": i * 60,
                "raan_rate": -0.03,
                "argument_of_perigee": 30 + i * 20,
                "mean_anomaly": i * 45,
                "reference_time": reference,
            }
            for i in range(8)
        ],
    }


def point_payload():
    return {
        "name": "control point",
        "latitude": 30.0,
        "longitude": 120.0,
        "height": 10.0,
        "default_elevation_mask": 5.0,
        "mask_profile": [[0, 10], [90, 15], [180, 8], [270, 20]],
    }


class FakeEngine:
    def __init__(self, schedules=None, blockers=None):
        self.schedules = schedules or {}
        self.blockers = blockers or {}
        self.observations = []

    def evaluate(self, almanac, point, at, almanac_version="none", point_version="none"):
        event = self.blockers.get(almanac_version)
        if event is not None and not event.is_set():
            event.wait(2)
        t = int((at if isinstance(at, datetime) else datetime.fromtimestamp(at, tz=timezone.utc)) - BASE).total_seconds()
        state = self.schedules.get(almanac_version, lambda p, s: False)(point["name"], t)
        if state:
            return DopResult(True, 8, 2.0, 1.7, 1.0, 1.3, 1.0, tuple(f"S{i}" for i in range(8)))
        return DopResult(False, 0, visible_satellites=())

    def observe(self, *args, **kwargs):
        return self.observations


def service_with(engine):
    repository = MemoryRepository()
    return PlanningService(repository, engine, max_workers=2), repository


def prepare_plan(engine, schedule, min_duration=1):
    service, repo = service_with(engine)
    almanac = service.import_almanac(almanac_payload(), trigger=False)
    point = service.upsert_point("P1", point_payload(), trigger=False)
    engine.schedules[almanac["id"]] = schedule
    plan = service.create_plan(
        {
            "point_ids": ["P1"],
            "start": second(0).isoformat().replace("+00:00", "Z"),
            "end": second(800).isoformat().replace("+00:00", "Z"),
            "gdop_threshold": 3.0,
            "min_duration_seconds": min_duration,
            "step_seconds": 1,
            "almanac_version_id": almanac["id"],
        }
    )
    wait_plan(service, plan["id"])
    return service, plan


def interval(name, t):
    return (100 <= t <= 200) or (400 <= t <= 500) if name == "control point" else False


def test_almanac_update_produces_window_difference():
    engine = FakeEngine({})
    service, plan = prepare_plan(engine, interval)

    def changed(name, t):
        if name != "control point":
            return False
        return (120 <= t <= 180) or (600 <= t <= 700)

    almanac2 = service.import_almanac({**almanac_payload(), "name": "second"}, trigger=False)
    engine.schedules[almanac2["id"]] = changed
    service._replan_for_almanac(almanac2["id"])
    # Wait for active replacement plan.
    deadline = time.time() + 5
    while time.time() < deadline:
        current = service.get_plan(plan["id"])
        result = current["latest_result"]
        if result and result["almanac_version_id"] == almanac2["id"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("replanning did not finish")

    result = service.get_plan(plan["id"])["latest_result"]
    p1 = result["diff"]["points"]["P1"]
    assert len(p1["disappeared"]) == 1
    assert p1["disappeared"][0]["start"] == second(400).isoformat().replace("+00:00", "Z")
    assert len(p1["added"]) == 1
    assert p1["added"][0]["start"] == second(600).isoformat().replace("+00:00", "Z")
    assert len(p1["changed"]) == 1
    assert "shortened" in p1["changed"][0]["tags"]
    assert p1["changed"][0]["start_shift_seconds"] == 20


def test_point_mask_revision_replans_only_affected_plan():
    engine = FakeEngine({})
    service, plan = prepare_plan(engine, interval)
    initial_version = service.get_plan(plan["id"])["point_version_ids"]["P1"]

    current_alm = service.get_plan(plan["id"])["almanac_version_id"]
    engine.schedules[current_alm] = lambda name, t: 600 <= t <= 700
    service.upsert_point("P1", {**point_payload(), "default_elevation_mask": 20.0}, trigger=True)

    deadline = time.time() + 5
    while time.time() < deadline:
        current = service.get_plan(plan["id"])
        if current["point_version_ids"]["P1"] != initial_version and current["latest_result"]:
            if current["latest_result"]["point_version_ids"]["P1"] == current["point_version_ids"]["P1"]:
                break
        time.sleep(0.01)
    else:
        raise AssertionError("mask revision replan did not finish")

    result = service.get_plan(plan["id"])["latest_result"]
    assert result["windows_by_point"]["P1"][0]["start"] == second(600).isoformat().replace("+00:00", "Z")
    assert result["diff"]["old_almanac_version_id"] == result["almanac_version_id"]


def test_two_consecutive_almanac_updates_replace_old_replan():
    engine = FakeEngine({})
    service, plan = prepare_plan(engine, interval)
    alm2 = service.import_almanac({**almanac_payload(), "name": "v2"}, trigger=False)
    alm3 = service.import_almanac({**almanac_payload(), "name": "v3"}, trigger=False)
    old_event = threading.Event()
    engine.blockers[alm2["id"]] = old_event
    engine.schedules[alm2["id"]] = lambda name, t: False
    engine.schedules[alm3["id"]] = lambda name, t: 600 <= t <= 700

    service._replan_for_almanac(alm2["id"])
    plan_state = service.get_plan(plan["id"])
    old_job_id = plan_state["active_job_id"]
    time.sleep(0.05)
    service._replan_for_almanac(alm3["id"])
    old_event.set()

    deadline = time.time() + 5
    while time.time() < deadline:
        result = service.get_plan(plan["id"])["latest_result"]
        if result and result["almanac_version_id"] == alm3["id"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("newest almanac result missing")

    old_job = service.get_job(old_job_id)
    assert old_job["status"] in {"cancelled", "superseded"}
    final = service.get_plan(plan["id"])
    assert final["latest_result"]["almanac_version_id"] == alm3["id"]
    assert final["history_result"]["almanac_version_id"] != alm2["id"]


def test_job_cancellation_writes_no_partial_result():
    engine = FakeEngine({})
    service, repo = service_with(engine)
    alm = service.import_almanac(almanac_payload(), trigger=False)
    service.upsert_point("P1", point_payload(), trigger=False)
    event = threading.Event()
    engine.blockers[alm["id"]] = event
    engine.schedules[alm["id"]] = lambda name, t: False
    plan = service.create_plan(
        {
            "point_ids": ["P1"],
            "start": second(0).isoformat().replace("+00:00", "Z"),
            "end": second(100).isoformat().replace("+00:00", "Z"),
            "gdop_threshold": 3.0,
            "min_duration_seconds": 1,
            "step_seconds": 1,
            "almanac_version_id": alm["id"],
        }
    )
    job_id = service.get_plan(plan["id"])["active_job_id"]
    time.sleep(0.05)
    cancelled = service.cancel_job(job_id)
    event.set()
    final_job = wait_job(service, job_id)
    assert cancelled["status"] == "cancelled"
    assert final_job["status"] in {"cancelled", "superseded"}
    assert service.get_plan(plan["id"])["latest_result"] is None


def test_shared_computation_equals_standalone_and_reuses_work():
    def make_shared():
        repository = MemoryRepository()
        engine = ObservationEngine()
        service = PlanningService(repository, engine, max_workers=2)
        alm = service.import_almanac(almanac_payload(), trigger=False)
        service.upsert_point("P1", point_payload(), trigger=False)
        return service, alm

    shared, alm = make_shared()
    payload = {
        "point_ids": ["P1"],
        "start": second(0).isoformat().replace("+00:00", "Z"),
        "end": second(300).isoformat().replace("+00:00", "Z"),
        "gdop_threshold": 20.0,
        "min_duration_seconds": 1,
        "step_seconds": 1,
        "almanac_version_id": alm["id"],
    }
    plan1 = shared.create_plan(payload)
    wait_plan(shared, plan1["id"])
    position_calls = shared.engine.position_calls
    plan2 = shared.create_plan(payload)
    wait_plan(shared, plan2["id"])
    assert shared.engine.position_calls == position_calls
    first = shared.get_result(plan1["id"])["windows_by_point"]["P1"]
    second = shared.get_result(plan2["id"])["windows_by_point"]["P1"]
    assert first == second

    standalone, standalone_alm = make_shared()
    standalone_payload = dict(payload, almanac_version_id=standalone_alm["id"])
    plan3 = standalone.create_plan(standalone_payload)
    wait_plan(standalone, plan3["id"])
    third = standalone.get_result(plan3["id"])["windows_by_point"]["P1"]
    assert third == first
