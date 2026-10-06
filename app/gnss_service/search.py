from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Callable

from .timeutils import parse_time


@dataclass(frozen=True)
class AvailableWindow:
    start: datetime
    end: datetime
    minimum_gdop: float
    minimum_at: datetime


def effective_step(min_duration_seconds: float, requested_step: float | None = None) -> float:
    """Grid spacing used by the availability search.

    The spacing is at most half of the shortest acceptable duration. The one-
    second floor matches the required output resolution; the 10s cap keeps
    ordinary day-long jobs small.
    """
    cap = requested_step if requested_step is not None else 10.0
    return min(float(cap), max(1.0, float(min_duration_seconds) / 2.0))


def _dt(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def find_windows(
    start,
    end,
    available: Callable[[datetime], bool],
    gdop: Callable[[datetime], float | None],
    threshold: float,
    min_duration_seconds: float,
    step_seconds: float | None = None,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[AvailableWindow]:
    """Find availability intervals using a bounded grid and integer refinement.

    Grid samples bracket the time axis. Every run of true samples is expanded
    outward with integer-second bisection, locating threshold crossings and
    post-rise/post-set jump states to one-second resolution. The minimum is
    then read at each integer second in the proposed window.

    Guarantee: with spacing h, any true interval longer than h contains an
    interior grid sample. Since the planner uses h <= min_duration / 2, every
    interval eligible by duration is detected. The dual statement also defines
    the error: an isolated crossing or over-threshold spike shorter than or
    equal to h can fall between samples; the shortest possible miss length is
    arbitrarily close to h from below. Internal sub-h false gaps can merge two
    neighbouring intervals, which is the same bounded sampling limitation.
    """
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    if end_ts < start_ts:
        raise ValueError("end time must not be earlier than start time")

    h = effective_step(min_duration_seconds, step_seconds)
    sample_count = int(math.floor((end_ts - start_ts) / h))
    sample_times = [start_ts + index * h for index in range(sample_count + 1)]
    if sample_times[-1] < end_ts:
        sample_times.append(end_ts)
    else:
        sample_times[-1] = end_ts

    values: list[bool] = []
    for index, timestamp in enumerate(sample_times):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("planning job cancelled")
        value = bool(available(_dt(timestamp)))
        values.append(value)
        if progress is not None and index % 32 == 0:
            progress(min(0.95, index / max(1, len(sample_times))))

    windows: list[AvailableWindow] = []
    index = 0
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        run_start = index
        while index < len(values) and values[index]:
            index += 1
        run_end = index - 1

        if run_start == 0:
            left_offset = int(math.floor(round(sample_times[0] - start_ts, 9)))
        else:
            false_ts = sample_times[run_start - 1]
            true_ts = sample_times[run_start]
            low = int(math.floor(round(false_ts - start_ts, 9))) + 1
            high = int(math.floor(round(true_ts - start_ts, 9)))
            while low < high:
                middle = (low + high) // 2
                if available(_dt(start_ts + middle)):
                    high = middle
                else:
                    low = middle + 1
            left_offset = low

        if run_end == len(values) - 1:
            right_offset = int(math.ceil(round(sample_times[-1] - start_ts, 9)))
        else:
            true_ts = sample_times[run_end]
            false_ts = sample_times[run_end + 1]
            low = int(math.floor(round(true_ts - start_ts, 9)))
            high = int(math.ceil(round(false_ts - start_ts, 9)))
            while low < high:
                middle = (low + high + 1) // 2
                if available(_dt(start_ts + middle)):
                    low = middle
                else:
                    high = middle - 1
            right_offset = low

        duration = right_offset - left_offset
        if duration < min_duration_seconds:
            continue

        window_start_ts = start_ts + left_offset
        window_end_ts = start_ts + right_offset
        minimum_value = math.inf
        minimum_ts = window_start_ts
        # Integer scan also inspects every one-second jump caused by a
        # satellite mask rise/set inside a GDOP-acceptable run.
        for offset in range(left_offset, right_offset + 1):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("planning job cancelled")
            timestamp = start_ts + offset
            value = gdop(_dt(timestamp))
            if value is not None and value <= threshold and value < minimum_value:
                minimum_value = value
                minimum_ts = timestamp
        if math.isinf(minimum_value):
            minimum_value = threshold
            minimum_ts = window_start_ts

        windows.append(
            AvailableWindow(
                _dt(window_start_ts),
                _dt(window_end_ts),
                float(minimum_value),
                _dt(minimum_ts),
            )
        )

    if progress is not None:
        progress(1.0)
    return windows
