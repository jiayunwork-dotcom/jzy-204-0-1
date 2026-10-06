"""单点规划编排：把历书、点位、参数组合成时段搜索。"""
from __future__ import annotations

from .cache import SharedSkyCache
from .search import EngineEvaluator, find_windows
from .visibility import VisibilityEngine


def validate_params(params: dict) -> dict:
    """校验并规整一份规划参数（角度度、时间 ISO/秒）。"""
    from .errors import ValidationError
    from .timeutils import parse_time

    gdop_threshold = params.get("gdop_threshold")
    if gdop_threshold is None or not isinstance(
            gdop_threshold, (int, float)) or isinstance(gdop_threshold, bool):
        raise ValidationError("缺少 GDOP 阈值", field="gdop_threshold")
    gdop_threshold = float(gdop_threshold)
    if not (gdop_threshold > 0.0):
        raise ValidationError("GDOP 阈值必须为正", field="gdop_threshold")

    min_duration = int(params.get("min_duration_s", 0))
    if min_duration < 0:
        raise ValidationError("最短时长不能为负", field="min_duration_s")

    start = parse_time(params.get("date_start"))
    end = parse_time(params.get("date_end"))
    if end < start:
        raise ValidationError("日期范围结束早于开始", field="date_end")
    scan_step = int(params.get("scan_step_s", 60))
    if scan_step < 1:
        raise ValidationError("扫描步长必须为正", field="scan_step_s")

    return {
        "gdop_threshold": gdop_threshold,
        "min_duration_s": min_duration,
        "date_start": float(start),
        "date_end": float(end),
        "scan_step_s": scan_step,
    }


def run_planning(almanac, point, params: dict, cache: SharedSkyCache,
                 on_progress=None, is_cancelled=None, per_point_hook=None):
    """对单个点位执行一次规划，返回可直接入库的结果体（不含版本包装）。"""
    if per_point_hook is not None:
        per_point_hook(point.point_id)
    if is_cancelled is not None and is_cancelled():
        from .errors import JobAborted
        raise JobAborted()
    engine = VisibilityEngine(almanac, point, cache=cache)
    evaluator = EngineEvaluator(engine, params["gdop_threshold"])
    windows = find_windows(
        evaluator,
        range_start=int(params["date_start"]),
        range_end=int(params["date_end"]),
        gdop_threshold=params["gdop_threshold"],
        min_duration=params["min_duration_s"],
        scan_step=params["scan_step_s"],
        on_progress=on_progress,
        is_cancelled=is_cancelled,
    )
    return {
        "point_id": point.point_id,
        "point_version": point.version,
        "almanac_version": almanac.version,
        "params": dict(params),
        "windows": [w.to_dict() for w in windows],
    }
