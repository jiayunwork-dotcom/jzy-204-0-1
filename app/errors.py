"""统一的异常类型。"""
from __future__ import annotations



class ValidationError(ValueError):
    """输入数据不合法，接口层映射为 HTTP 400。"""

    def __init__(self, message, field=None):
        super().__init__(message)
        self.message = message
        self.field = field


class JobAborted(Exception):
    """作业被取消，工作线程据此立即退出。"""


class JobError(RuntimeError):
    """作业状态相关的非法操作（例如重复取消已完成作业）。"""
