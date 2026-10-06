from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError("timestamp must be an ISO-8601 string or datetime")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return parse_time(value).isoformat().replace("+00:00", "Z")


def validate_time_range(start, end) -> tuple[datetime, datetime]:
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if end_dt < start_dt:
        raise ValueError("end time must not be earlier than start time")
    return start_dt, end_dt
