"""健康检查与地图信息路由。"""
from __future__ import annotations

from flask import Blueprint, jsonify

from ._compat import legacy

bp = Blueprint("health", __name__)


@bp.route("/health")
def health():
    lg = legacy()
    try:
        ver = lg.client.get_server_version()
        return jsonify({"status": "ok", "version": ver, "map": lg.world.get_map().name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@bp.route("/map/info")
def map_info():
    lg = legacy()
    m = lg.world.get_map()
    spawn_pts = m.get_spawn_points()
    return jsonify({
        "name": m.name,
        "spawn_points_count": len(spawn_pts),
        "sample_spawn_points": [
            {"x": round(t.location.x, 1), "y": round(t.location.y, 1), "z": round(t.location.z, 1),
             "yaw": round(t.rotation.yaw, 1)}
            for t in spawn_pts[:5]
        ],
    })
