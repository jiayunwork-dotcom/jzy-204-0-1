from gnss_service.api import create_app
from gnss_service.repositories import MemoryRepository
from gnss_service.service import PlanningService


BASE = "2026-10-06T00:00:00Z"


def make_client():
    service = PlanningService(MemoryRepository(), max_workers=1)
    app = create_app(service)
    app.config.update(TESTING=True)
    return app.test_client(), service


def almanac():
    return {
        "name": "api",
        "satellites": [
            {
                "id": f"S{i}",
                "semi_major_axis": 26559800.0 + i * 1000,
                "eccentricity": 0.01,
                "inclination": 55 + i,
                "raan": i * 45,
                "argument_of_perigee": i * 30,
                "mean_anomaly": i * 40,
                "reference_time": BASE,
            }
            for i in range(8)
        ],
    }


def point():
    return {"name": "P", "latitude": 30, "longitude": 120, "height": 0, "default_elevation_mask": 5}


def test_http_workflow_and_validation_error():
    client, service = make_client()
    response = client.post("/api/almanacs", json=almanac())
    assert response.status_code == 201
    alm = response.get_json()
    response = client.put("/api/points/P1", json=point())
    assert response.status_code == 201
    response = client.get(
        f"/api/query?almanac_version_id={alm['id']}&point_id=P1&time=2026-10-06T01:00:00Z"
    )
    assert response.status_code == 200
    assert "gdop" in response.get_json()

    bad = {**almanac(), "satellites": [{**almanac()["satellites"][0], "eccentricity": 1}]}
    response = client.post("/api/almanacs", json=bad)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_create_plan_and_query_job():
    client, service = make_client()
    alm = client.post("/api/almanacs", json=almanac()).get_json()
    client.put("/api/points/P1", json=point())
    response = client.post(
        "/api/plans",
        json={
            "point_ids": ["P1"],
            "start": BASE,
            "end": "2026-10-06T00:10:00Z",
            "gdop_threshold": 30,
            "min_duration_seconds": 1,
            "step_seconds": 1,
            "almanac_version_id": alm["id"],
        },
    )
    assert response.status_code == 202
    plan = response.get_json()
    job_id = plan["active_job_id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] in {"queued", "running", "completed"}
