"""结果差异对比单元测试。"""
from app.diff import diff_results


def w(start, end, gdop=2.0):
    return {"start_epoch": start, "end_epoch": end,
            "start": str(start), "end": str(end),
            "duration_s": end - start, "gdop_min": gdop,
            "gdop_min_time": start}


def _res(pid, almanac_version, windows):
    return {"point_id": pid, "almanac_version": almanac_version,
            "point_version": 1, "windows": windows}


def test_added_and_disappeared():
    old = [_res("P", 1, [w(0, 100), w(1000, 1100)])]
    new = [_res("P", 2, [w(1000, 1100), w(2000, 2100)])]
    d = diff_results(old, new)
    p = d["points"]["P"]
    assert len(p["disappeared"]) == 1
    assert p["disappeared"][0]["start_epoch"] == 0
    assert len(p["added"]) == 1
    assert p["added"][0]["start_epoch"] == 2000
    assert d["totals"]["disappeared"] == 1
    assert d["totals"]["added"] == 1


def test_shortened_classification():
    old = [_res("P", 1, [w(0, 600)])]
    new = [_res("P", 2, [w(0, 400)])]
    d = diff_results(old, new)
    p = d["points"]["P"]
    assert len(p["shortened"]) == 1
    assert p["shortened"][0]["duration_change_s"] == -200


def test_shifted_classification():
    old = [_res("P", 1, [w(1000, 1600)])]
    new = [_res("P", 2, [w(1100, 1700)])]
    d = diff_results(old, new)
    p = d["points"]["P"]
    assert len(p["shifted"]) == 1
    assert p["shifted"][0]["start_shift_s"] == 100
    assert p["shifted"][0]["end_shift_s"] == 100


def test_identical_windows_unchanged():
    old = [_res("P", 1, [w(0, 100), w(1000, 1100)])]
    new = [_res("P", 2, [w(0, 100), w(1000, 1100)])]
    d = diff_results(old, new)
    p = d["points"]["P"]
    assert p["unchanged"] == 2
    assert not d["has_changes"]


def test_point_appears_in_new_almanac():
    old = [_res("A", 1, [w(0, 100)])]
    new = [_res("A", 2, [w(0, 100)]), _res("B", 2, [w(0, 100)])]
    d = diff_results(old, new)
    assert "B" in d["points"]
    assert len(d["points"]["B"]["added"]) == 1
