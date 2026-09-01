"""兼容层：蓝图访问核心引导壳模块的全局状态。"""
from __future__ import annotations

from types import ModuleType

from flask import current_app


def legacy() -> ModuleType:
    """返回核心引导壳 carla_relay_core 模块（运行时状态宿主）。

    引导壳在 Flask 应用创建后执行 ``app.extensions["legacy"] = sys.modules[__name__]``。
    蓝图路由在请求期调用本函数，动态读取 world / 帧缓存 / stream 目标等全局，
    保证与引导壳内实验代码操作同一份状态。
    """
    return current_app.extensions["legacy"]
