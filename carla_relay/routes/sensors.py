"""传感器挂载与管理路由。

序列化逻辑已抽取至 carla_relay.core.sensors，此处仅保留
挂载 / 帧查询 / 删除的 HTTP 层。
"""
from __future__ import annotations

import base64

import carla
from flask import Blueprint, jsonify, request

from ._compat import legacy

bp = Blueprint("sensors", __name__)


@bp.route("/vehicle/<int:vid>/sensor/camera", methods=["POST"])
def attach_camera(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    camera_type = data.get("type", "rgb")  # rgb | semantic | depth
    width = data.get("width", 1280)
    height = data.get("height", 720)
    fov = data.get("fov", 90.0)
    x = data.get("x", 1.6)
    y = data.get("y", 0.0)
    z = data.get("z", 1.7)
    pitch = data.get("pitch", -5.0)
    yaw = data.get("yaw", 0.0)
    roll = data.get("roll", 0.0)

    bp_id = f"sensor.camera.{camera_type}"
    bp = lg.world.get_blueprint_library().find(bp_id)
    if bp is None:
        return jsonify({"status": "error", "message": f"unknown camera type: {camera_type}"}), 400
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    if bp.has_attribute("fov"):
        bp.set_attribute("fov", str(fov))

    transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))
    sensor = lg.world.spawn_actor(bp, transform, attach_to=actor)

    sid = sensor.id
    with lg._lock:
        lg._sensor_refs[sid] = sensor  # 保持引用，防止 GC 导致 listen() 回调失效
    sensor.listen(lambda data, sid=sid: lg._sensor_callback(sid, "camera", data))

    with lg._lock:
        lg._managed_actors.add(sid)
    return jsonify({
        "status": "ok",
        "sensor_id": sid,
        "type": camera_type,
        "resolution": f"{width}x{height}",
        "vehicle_id": vid,
    })


@bp.route("/vehicle/<int:vid>/sensor/lidar", methods=["POST"])
def attach_lidar(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    bp = lg.world.get_blueprint_library().find("sensor.lidar.ray_cast")
    bp.set_attribute("range", str(data.get("range", 100.0)))
    bp.set_attribute("channels", str(data.get("channels", 32)))
    bp.set_attribute("points_per_second", str(data.get("points_per_second", 100000)))
    bp.set_attribute("rotation_frequency", str(data.get("rotation_frequency", 10.0)))

    z = data.get("z", 2.4)
    transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=z))
    sensor = lg.world.spawn_actor(bp, transform, attach_to=actor)

    sid = sensor.id
    with lg._lock:
        lg._sensor_refs[sid] = sensor
    sensor.listen(lambda data, sid=sid: lg._sensor_callback(sid, "lidar", data))

    with lg._lock:
        lg._managed_actors.add(sid)
    return jsonify({"status": "ok", "sensor_id": sid, "type": "lidar", "vehicle_id": vid})


@bp.route("/vehicle/<int:vid>/sensor/gnss", methods=["POST"])
def attach_gnss(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    bp = lg.world.get_blueprint_library().find("sensor.other.gnss")
    sensor = lg.world.spawn_actor(bp, carla.Transform(), attach_to=actor)

    sid = sensor.id
    with lg._lock:
        lg._sensor_refs[sid] = sensor
    sensor.listen(lambda data, sid=sid: lg._sensor_callback(sid, "gnss", data))

    with lg._lock:
        lg._managed_actors.add(sid)
    return jsonify({"status": "ok", "sensor_id": sid, "type": "gnss", "vehicle_id": vid})


@bp.route("/vehicle/<int:vid>/sensor/imu", methods=["POST"])
def attach_imu(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    bp = lg.world.get_blueprint_library().find("sensor.other.imu")
    sensor = lg.world.spawn_actor(bp, carla.Transform(), attach_to=actor)

    sid = sensor.id
    with lg._lock:
        lg._sensor_refs[sid] = sensor
    sensor.listen(lambda data, sid=sid: lg._sensor_callback(sid, "imu", data))

    with lg._lock:
        lg._managed_actors.add(sid)
    return jsonify({"status": "ok", "sensor_id": sid, "type": "imu", "vehicle_id": vid})


@bp.route("/vehicle/<int:vid>/sensor/depth", methods=["POST"])
def attach_depth(vid: int):
    """挂载深度相机（sensor.camera.depth）"""
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    width = data.get("width", 640)
    height = data.get("height", 360)
    fov = data.get("fov", 90.0)
    bp = lg.world.get_blueprint_library().find("sensor.camera.depth")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    if bp.has_attribute("fov"):
        bp.set_attribute("fov", str(fov))
    x = data.get("x", 1.6); y = data.get("y", 0.0); z = data.get("z", 1.7)
    pitch = data.get("pitch", -5.0)
    transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch))
    sensor = lg.world.spawn_actor(bp, transform, attach_to=actor)
    sid = sensor.id
    with lg._lock:
        lg._sensor_refs[sid] = sensor
    sensor.listen(lambda data, sid=sid: lg._sensor_callback(sid, "depth", data))
    with lg._lock:
        lg._managed_actors.add(sid)
    return jsonify({"status": "ok", "sensor_id": sid, "type": "depth", "vehicle_id": vid})


@bp.route("/vehicle/<int:vid>/sensor/radar", methods=["POST"])
def attach_radar(vid: int):
    lg = legacy()
    actor = lg.world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    bp = lg.world.get_blueprint_library().find("sensor.other.radar")
    if bp.has_attribute("range"):
        bp.set_attribute("range", str(data.get("range", 100.0)))
    if bp.has_attribute("horizontal_fov"):
        bp.set_attribute("horizontal_fov", str(data.get("horizontal_fov", 60.0)))
    if bp.has_attribute("vertical_fov"):
        bp.set_attribute("vertical_fov", str(data.get("vertical_fov", 20.0)))
    if bp.has_attribute("points_per_second"):
        bp.set_attribute("points_per_second", str(data.get("points_per_second", 1500)))
    z = data.get("z", 1.5)
    transform = carla.Transform(carla.Location(x=0.8, y=0.0, z=z))
    sensor = lg.world.spawn_actor(bp, transform, attach_to=actor)
    sid = sensor.id
    with lg._lock:
        lg._sensor_refs[sid] = sensor
    sensor.listen(lambda data, sid=sid: lg._sensor_callback(sid, "radar", data))
    with lg._lock:
        lg._managed_actors.add(sid)
    return jsonify({"status": "ok", "sensor_id": sid, "type": "radar", "vehicle_id": vid})


@bp.route("/sensor/<int:sid>/frame")
def sensor_frame(sid: int):
    lg = legacy()
    if sid not in lg._sensor_frames:
        return jsonify({"status": "error", "message": "no frame yet or sensor not found"}), 404
    dtype = lg._sensor_dtype.get(sid, "unknown")
    if dtype in ("camera", "instance"):
        # 返回 base64 JPEG（instance 为 bbox 渲染后的 JPEG 帧）
        return jsonify({
            "sensor_id": sid,
            "type": "camera",
            "format": "jpeg",
            "base64": base64.b64encode(lg._sensor_frames[sid]).decode(),
        })
    else:
        # 返回 JSON 字符串
        return jsonify({
            "sensor_id": sid,
            "type": dtype,
            "data": lg._sensor_frames[sid].decode(),
        })


@bp.route("/sensor/<int:sid>", methods=["DELETE"])
def sensor_delete(sid: int):
    lg = legacy()
    actor = lg.world.get_actor(sid)
    if actor is not None and actor.is_alive:
        actor.destroy()
    lg._sensor_frames.pop(sid, None)
    lg._semantic_raw.pop(sid, None)
    lg._sensor_dtype.pop(sid, None)
    lg._sensor_refs.pop(sid, None)
    with lg._lock:
        lg._managed_actors.discard(sid)
    return jsonify({"status": "ok", "destroyed": sid})
