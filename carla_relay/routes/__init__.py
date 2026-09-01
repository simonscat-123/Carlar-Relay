"""routes 蓝图包。

访问模式：全局运行时状态（world / 帧缓存 / stream 目标等）宿主在核心
引导壳 carla_relay_core.py 模块中，蓝图经 ``app.extensions["legacy"]``
在请求期动态访问（_compat.legacy()）。

注册入口：``register_all(app)``（由引导壳在 Flask 应用创建后调用）。
"""
from __future__ import annotations

from flask import Flask


def _blueprints():
    """惰性导入各蓝图，避免包加载期触发 flask 之外的依赖。"""
    from .health import bp as health_bp
    from .sync import bp as sync_bp
    from .vehicle import bp as vehicle_bp
    from .sensors import bp as sensors_bp
    from .stream import bp as stream_bp
    from .misc import bp as misc_bp
    return [health_bp, sync_bp, vehicle_bp, sensors_bp, stream_bp, misc_bp]


def register_all(app: Flask) -> None:
    """注册全部路由蓝图。"""
    for bp in _blueprints():
        app.register_blueprint(bp)
