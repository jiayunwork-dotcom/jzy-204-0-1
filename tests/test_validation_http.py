"""接口层拒收校验（HTTP 400）。"""
import pytest


def test_reject_bad_eccentricity_http(client):
    payload = {"satellites": [
        {"sat_id": "X", "semi_major_axis": 26_600_000.0, "ecc": 1.0,
         "inc_deg": 55, "raan0_deg": 0, "argp_deg": 0, "ma_deg": 0,
         "toe": "2026-01-01T00:00:00Z"}]}
    r = client.post("/api/almanacs", json=payload)
    assert r.status_code == 400
    assert "偏心率" in r.get_json()["message"]


def test_reject_nonpositive_semimajor_http(client):
    payload = {"satellites": [
        {"sat_id": "X", "semi_major_axis": 0.0, "ecc": 0.0,
         "inc_deg": 55, "raan0_deg": 0, "argp_deg": 0, "ma_deg": 0,
         "toe": "2026-01-01T00:00:00Z"}]}
    r = client.post("/api/almanacs", json=payload)
    assert r.status_code == 400


def test_reject_latitude_http(client):
    r = client.put("/api/points", json={
        "point_id": "BAD", "lon_deg": 116.0, "lat_deg": 91.0})
    assert r.status_code == 400
    assert "纬度" in r.get_json()["message"]


def test_reject_mask_azimuth_http(client):
    r = client.put("/api/points", json={
        "point_id": "BAD", "lon_deg": 116.0, "lat_deg": 40.0,
        "mask": {"points": [[400, 10]]}})
    assert r.status_code == 400
    r = client.put("/api/points", json={
        "point_id": "BAD", "lon_deg": 116.0, "lat_deg": 40.0,
        "mask": {"points": [[45, 10], [45, 12]]}})
    assert r.status_code == 400


def test_reject_nonpositive_threshold_http(client, future_window):
    start, end = future_window
    client.put("/api/points", json={
        "point_id": "P1", "lon_deg": 116.0, "lat_deg": 40.0})
    r = client.post("/api/plans", json={
        "point_ids": ["P1"],
        "date_start": _iso(start), "date_end": _iso(end),
        "gdop_threshold": 0.0, "min_duration_s": 0})
    assert r.status_code == 400
    assert "阈值" in r.get_json()["message"]


def test_reject_negative_min_duration_http(client, future_window):
    start, end = future_window
    r = client.post("/api/plans", json={
        "point_ids": ["P1"],
        "date_start": _iso(start), "date_end": _iso(end),
        "gdop_threshold": 5.0, "min_duration_s": -1})
    assert r.status_code == 400
    assert "最短时长" in r.get_json()["message"]


def test_reject_inverted_range_http(client):
    r = client.post("/api/plans", json={
        "point_ids": ["P1"],
        "date_start": "2026-03-02T00:00:00Z",
        "date_end": "2026-03-01T00:00:00Z",
        "gdop_threshold": 5.0, "min_duration_s": 0})
    assert r.status_code == 400


def test_reject_plan_with_missing_point(client, future_window):
    start, end = future_window
    r = client.post("/api/plans", json={
        "point_ids": ["GHOST"],
        "date_start": _iso(start), "date_end": _iso(end),
        "gdop_threshold": 5.0})
    assert r.status_code == 400


def _iso(epoch):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
