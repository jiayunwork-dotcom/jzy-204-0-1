from .timeutils import iso, parse_time


def window_to_dict(window) -> dict:
    return {
        "start": iso(window.start),
        "end": iso(window.end),
        "duration_seconds": int((window.end - window.start).total_seconds()),
        "minimum_gdop": window.minimum_gdop,
        "minimum_at": iso(window.minimum_at),
    }


def window_from_dict(payload: dict) -> dict:
    return dict(payload)


def _window_overlap(left: dict, right: dict) -> float:
    start = max(parse_time(left["start"]).timestamp(), parse_time(right["start"]).timestamp())
    end = min(parse_time(left["end"]).timestamp(), parse_time(right["end"]).timestamp())
    return max(0.0, end - start + 1.0)


def diff_windows(old_windows: list[dict], new_windows: list[dict]) -> dict:
    """Compare non-overlapping intervals by maximum one-to-one overlap."""
    candidates = []
    for oi, old in enumerate(old_windows):
        for ni, new in enumerate(new_windows):
            overlap = _window_overlap(old, new)
            if overlap > 0:
                candidates.append((overlap, oi, ni))
    used_old: set[int] = set()
    used_new: set[int] = set()
    pairs: dict[int, int] = {}
    for _, oi, ni in sorted(candidates, reverse=True):
        if oi not in used_old and ni not in used_new:
            used_old.add(oi)
            used_new.add(ni)
            pairs[oi] = ni

    disappeared = [window_from_dict(old_windows[i]) for i in range(len(old_windows)) if i not in used_old]
    added = [window_from_dict(new_windows[i]) for i in range(len(new_windows)) if i not in used_new]
    changed = []
    unchanged = []
    for oi, ni in sorted(pairs.items()):
        old = old_windows[oi]
        new = new_windows[ni]
        start_shift = int((parse_time(new["start"]) - parse_time(old["start"])).total_seconds())
        end_shift = int((parse_time(new["end"]) - parse_time(old["end"])).total_seconds())
        old_duration = int(old["duration_seconds"])
        new_duration = int(new["duration_seconds"])
        tags = []
        if new_duration < old_duration:
            tags.append("shortened")
        elif new_duration > old_duration:
            tags.append("lengthened")
        if start_shift != 0 or end_shift != 0:
            tags.append("shifted")
        item = {
            "old": window_from_dict(old),
            "new": window_from_dict(new),
            "start_shift_seconds": start_shift,
            "end_shift_seconds": end_shift,
            "tags": tags,
        }
        if tags:
            changed.append(item)
        else:
            unchanged.append(item)
    return {
        "disappeared": disappeared,
        "added": added,
        "changed": changed,
        "unchanged": unchanged,
    }


def diff_results(old_result: dict | None, new_result: dict | None) -> dict:
    old_points = (old_result or {}).get("windows_by_point", {})
    new_points = (new_result or {}).get("windows_by_point", {})
    point_ids = sorted(set(old_points) | set(new_points))
    return {
        "old_almanac_version_id": (old_result or {}).get("almanac_version_id"),
        "new_almanac_version_id": (new_result or {}).get("almanac_version_id"),
        "points": {
            point_id: diff_windows(old_points.get(point_id, []), new_points.get(point_id, []))
            for point_id in point_ids
        },
    }
