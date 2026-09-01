"""同步模式路由。

_old_settings 为重新绑定型全局：蓝图通过 ``lg._old_settings = ...``
写模块属性，引导壳内其它读点动态可见（等价于 ``global`` 声明）。
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ._compat import legacy

bp = Blueprint("sync", __name__)


@bp.route("/sync/enable", methods=["POST"])
def sync_enable():
    lg = legacy()
    data = request.get_json(silent=True) or {}
    fixed_delta = data.get("fixed_delta", 0.05)
    settings = lg.world.get_settings()
    lg._old_settings = {
        "synchronous_mode": settings.synchronous_mode,
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "no_rendering_mode": settings.no_rendering_mode,
    }
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    lg.world.apply_settings(settings)
    lg.traffic_manager.set_synchronous_mode(True)
    return jsonify({"status": "ok", "fixed_delta": fixed_delta})


@bp.route("/sync/tick", methods=["POST"])
def sync_tick():
    lg = legacy()
    if lg.world.get_settings().synchronous_mode:
        lg.world.tick()
        return jsonify({"status": "ok", "frame": lg.world.get_snapshot().frame})
    return jsonify({"status": "error", "message": "not in sync mode"}), 400


@bp.route("/sync/disable", methods=["POST"])
def sync_disable():
    lg = legacy()
    if lg._old_settings:
        settings = lg.world.get_settings()
        settings.synchronous_mode = lg._old_settings["synchronous_mode"]
        settings.fixed_delta_seconds = lg._old_settings["fixed_delta_seconds"]
        settings.no_rendering_mode = lg._old_settings["no_rendering_mode"]
        lg.world.apply_settings(settings)
        lg.traffic_manager.set_synchronous_mode(False)
        lg._old_settings = None
    return jsonify({"status": "ok"})
