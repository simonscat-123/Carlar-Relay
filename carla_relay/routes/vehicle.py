"""车辆管理与控制路由。"""
from __future__ import annotations

import math

import carla
import numpy as np
from flask import Blueprint, jsonify, request

from ._compat import legacy

bp = Blueprint("vehicle", __name__)


@bp.route("/vehicle/spawn", methods=["POST"])
def vehicle_spawn():
    """生成车辆，可选参数: blueprint(默认随机vehicle.*), autopilot(默认true),
       spawn_index(默认随机), role_name(默认hero)"""
    lg = legacy()
    data = request.get_json(silent=True) or {}
    bp_filter = data.get("blueprint", "vehicle.*")
    autopilot = data.get("autopilot", True)
    spawn_index = data.get("spawn_index", None)
    role_name = data.get("role_name", "hero")

    bps = lg.world.get_blueprint_library().filter(bp_filter)
    if not bps:
        return jsonify({"status": "error", "message": f"no blueprints matching '{bp_filter}'"}), 400
    bp = np.random.choice(bps)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    if bp.has_attribute("color"):
        bp.set_attribute("color", np.random.choice(bp.get_attribute("color").recommended_values))

    spawn_points = lg.world.get_map().get_spawn_points()
    if spawn_index is not None:
        if spawn_index < 0 or spawn_index >= len(spawn_points):
            return jsonify({"status": "error", "message": f"spawn_index out of range [0, {len(spawn_points)-1}]"}), 400
        points = [spawn_points[spawn_index]]
    else:
        indices = list(range(len(spawn_points)))
        np.random.shuffle(indices)
        points = [spawn_points[i] for i in indices]

    for t in points:
        vehicle = lg.world.try_spawn_actor(bp, t)
        if vehicle is not None:
            vehicle.set_autopilot(autopilot)
            with lg._lock:
                lg._managed_actors.add(vehicle.id)
            return jsonify({
                "status": "ok",
                "id": vehicle.id,
                "type": vehicle.type_id,
                "location": {"x": round(t.location.x, 2), "y": round(t.location.y, 2), "z": round(t.location.z, 2)},
                "autopilot": autopilot,
            })
    return jsonify({"status": "error", "message": "no free spawn point"}), 409


@bp.route("/vehicle/<int:vid>", methods=["GET", "DELETE"])
def vehicle_handle(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if request.method == "DELETE":
        if actor is not None and actor.is_alive:
            actor.destroy()
        with lg._lock:
            lg._managed_actors.discard(vid)
        return jsonify({"status": "ok", "destroyed": vid})
    # GET
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    v = actor
    vel = v.get_velocity()
    loc = v.get_location()
    ctrl = v.get_control() if isinstance(v, carla.Vehicle) else None
    return jsonify({
        "id": v.id,
        "type": v.type_id,
        "location": {"x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2)},
        "velocity": round(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2), 2),
        "is_alive": v.is_alive,
        "control": {
            "throttle": round(ctrl.throttle, 3),
            "steer": round(ctrl.steer, 3),
            "brake": round(ctrl.brake, 3),
        } if ctrl else None,
    })


@bp.route("/vehicle/<int:vid>/control", methods=["POST"])
def vehicle_control(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    throttle = data.get("throttle", 0.0)
    steer = data.get("steer", 0.0)
    brake = data.get("brake", 0.0)
    reverse = data.get("reverse", False)
    hand_brake = data.get("hand_brake", False)

    ctrl = carla.VehicleControl(
        throttle=float(throttle),
        steer=float(steer),
        brake=float(brake),
        reverse=bool(reverse),
        hand_brake=bool(hand_brake),
    )
    actor.apply_control(ctrl)
    return jsonify({"status": "ok", "control": data})


@bp.route("/vehicle/<int:vid>/autopilot", methods=["POST"])
def vehicle_autopilot(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    enable = data.get("enable", True)
    actor.set_autopilot(enable)
    lg._autopilot_state[vid] = enable
    return jsonify({"status": "ok", "autopilot": enable})


@bp.route("/vehicle/<int:vid>/autopilot", methods=["GET"])
def vehicle_autopilot_get(vid: int):
    lg = legacy()
    return jsonify({"autopilot": lg._autopilot_state.get(vid, True)})


@bp.route("/vehicle/<int:vid>/spectator", methods=["POST"])
def vehicle_spectator(vid: int):
    """将观察者镜头移到车辆后方"""
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    spectator = lg.world.get_spectator()
    transform = actor.get_transform()
    # 镜头移到车后上方
    back_offset = transform.get_forward_vector() * -6.0
    transform.location += back_offset
    transform.location.z += 3.0
    transform.rotation.pitch = -14.0
    spectator.set_transform(transform)
    return jsonify({"status": "ok", "location": {"x": round(transform.location.x,1), "y": round(transform.location.y,1), "z": round(transform.location.z,1)}})
