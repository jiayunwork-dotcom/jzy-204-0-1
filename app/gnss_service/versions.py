from .models import normalize_almanac, validate_point_payload
from .timeutils import iso, utc_now
from .uuid import new_id


def build_almanac_document(payload: dict) -> dict:
    normalized = normalize_almanac(payload)
    return {
        "id": new_id("alm"),
        "name": normalized["name"],
        "satellites": normalized["satellites"],
        "created_at": iso(utc_now()),
    }


def build_point_version(point_id: str, version_number: int, payload: dict) -> dict:
    data = validate_point_payload(payload)
    return {
        "id": new_id("ptv"),
        "point_id": point_id,
        "version": version_number,
        "data": data,
        "created_at": iso(utc_now()),
    }
