from datetime import datetime, timedelta, timezone

import pytest

from gnss_service.search import effective_step, find_windows


def at(second: int) -> datetime:
    return datetime(2026, 10, 6, tzinfo=timezone.utc) + timedelta(seconds=second)


def test_constructed_short_spike_is_found():
    # 20 s is longer than effective h=min(10, 15/2=7.5) and satisfies the
    # requested 15 s minimum. A naive 60 s scan would miss this short spike.
    values = set(range(210, 230))

    def available(t):
        return int((t - at(0)).total_seconds()) in values

    def gdop(t):
        return 2.0 if available(t) else 100.0

    windows = find_windows(at(0), at(600), available, gdop, 3.0, 15)
    assert [(w.start, w.end) for w in windows] == [(at(210), at(229))]
    assert windows[0].minimum_gdop == 2.0


def test_constructed_over_threshold_spike_splits_window():
    # A 20 s GDOP excursion (>7.5 s grid) must be found and split the window.
    def available(t):
        second_value = int((t - at(0)).total_seconds())
        return not (210 <= second_value <= 229)

    def gdop(t):
        return 10.0 if not available(t) else 2.0

    windows = find_windows(at(0), at(600), available, gdop, 3.0, 15)
    assert [(w.start, w.end) for w in windows] == [(at(0), at(209)), (at(230), at(600))]


def test_sub_step_spike_miss_bound_is_documented_by_step_size():
    # A five-second spike lies strictly in a ten-second grid gap. The method
    # only promises not to miss intervals longer than h; it may miss this one.
    def available(t):
        return 15 <= int((t - at(0)).total_seconds()) <= 19

    windows = find_windows(at(0), at(60), available, lambda t: 1, 3, 10, step_seconds=10)
    assert windows == []


def test_boundaries_refined_to_one_second_and_minimum_recorded():
    def available(t):
        s = int((t - at(0)).total_seconds())
        return 123 <= s <= 456

    def gdop(t):
        s = int((t - at(0)).total_seconds())
        if not available(t):
            return None
        return 1.5 + abs(s - 300) / 1000

    windows = find_windows(at(0), at(1000), available, gdop, 3, 60, step_seconds=60)
    assert len(windows) == 1
    assert windows[0].start == at(123)
    assert windows[0].end == at(456)
    assert windows[0].minimum_at == at(300)
    assert windows[0].minimum_gdop == pytest.approx(1.5)


def test_effective_step_at_most_half_minimum_duration():
    assert effective_step(30) == 7.5
    assert effective_step(0) == 1
    assert effective_step(180) == 10
