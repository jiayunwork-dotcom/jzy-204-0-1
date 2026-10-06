"""时间工具：接口用 ISO8601（UTC）字符串，数值计算统一用 Unix 秒（float/int）。"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .errors import ValidationError


def now_epoch() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def parse_time(value) -> float:
    """把 ISO8601 字符串 / 数字解析为 Unix 秒。

    不带时区的 ISO 字符串按 UTC 处理。
    """
    if value is None:
        raise ValidationError("缺少时间字段")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        raise ValidationError(f"无法解析的时间格式: {value!r}")
    text = value.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"无法解析的时间格式: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def to_iso(epoch_seconds: float) -> str:
    dt = datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def wrap_angle(x):
    """把角度（弧度）规整到 [-pi, pi)。"""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi
