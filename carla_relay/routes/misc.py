"""诊断与批量管理路由。"""
from __future__ import annotations

from flask import Blueprint, jsonify

from ._compat import legacy

bp = Blueprint("misc", __name__)


@bp.route("/debug/bbox")
def debug_bbox():
    """诊断端点：暴露 bbox 相关全局状态，排查包围框相机黑屏"""
    lg = legacy()
    inst_sid = lg._stream_bbox
    info = {
        "stream_bbox": inst_sid,
        "instance_raw_keys": sorted(lg._instance_raw.keys()),
        "semantic_raw_keys": sorted(lg._semantic_raw.keys()),
        "sensor_frames_keys": sorted(lg._sensor_frames.keys()),
        "sensor_dtype": {str(k): v for k, v in lg._sensor_dtype.items()},
    }
    if inst_sid is not None:
        info["instance_raw_len"] = len(lg._instance_raw.get(inst_sid, b""))
        try:
            a = lg.world.get_actor(inst_sid)
            info["inst_alive"] = bool(a and a.is_alive)
            info["inst_listening"] = bool(a and a.is_listening) if a else None
        except Exception as exc:
            info["inst_alive"] = f"error: {exc}"
    return jsonify(info)


@bp.route("/actors")
def list_actors():
    lg = legacy()
    with lg._lock:
        ids = list(lg._managed_actors)
    result = []
    for aid in ids:
        actor = lg.world.get_actor(aid)
        if actor is None or not actor.is_alive:
            continue
        result.append({"id": aid, "type": actor.type_id, "is_alive": actor.is_alive})
    return jsonify({"actors": result})


@bp.route("/cleanup", methods=["DELETE"])
def cleanup():
    lg = legacy()
    # 实验运行中拒绝清理：防止前端「停止实验后延时触发的 cleanup」误杀
    # 刚快速重启的新实验的 actor（各实验线程的 finally 已自行销毁自己的 actor）
    if lg._any_experiment_running():
        return jsonify({"status": "ok", "message": "实验运行中，跳过清理"})
    with lg._lock:
        ids = list(lg._managed_actors)
    for aid in ids:
        actor = lg.world.get_actor(aid)
        if actor is not None and actor.is_alive:
            actor.destroy()
    with lg._lock:
        lg._managed_actors.clear()
    lg._sensor_frames.clear()
    lg._semantic_raw.clear()
    lg._instance_raw.clear()
    lg._sensor_dtype.clear()
    lg._stream_vehicle = None
    lg._stream_camera = None
    lg._stream_camera_left = None
    lg._stream_camera_right = None
    lg._stream_semantic = None
    lg._stream_bird = None
    lg._stream_bbox = None
    lg._camera_actor_ref = None
    lg._sensor_refs.clear()
    return jsonify({"status": "ok", "cleaned": len(ids)})
