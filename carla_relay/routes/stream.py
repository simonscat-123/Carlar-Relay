"""SSE 数据流与画面预览路由。

stream 目标（_stream_vehicle / _stream_camera 等）为重新绑定型全局：
蓝图通过 ``lg._stream_camera = ...`` 写模块属性，
引导壳内 _sse_stream_thread 的读点动态可见。
"""
from __future__ import annotations

import json
import queue
import time

import carla
import numpy as np
from flask import Blueprint, Response, jsonify, request

from carla_relay.core.sse import hub as _sse_hub

from ._compat import legacy

bp = Blueprint("stream", __name__)


@bp.route("/stream")
def stream():
    """SSE 端点：实时推送车辆状态 + 相机画面。
    前端用 new EventSource('/stream?vehicle_id=10&camera_sid=5') 连接。
    或者先调 /stream/setup 设置目标，再连 /stream。
    """
    lg = legacy()
    vehicle_id = request.args.get("vehicle_id", type=int)
    camera_sid = request.args.get("camera_sid", type=int)

    # 如果传入参数，覆盖全局设置
    if vehicle_id is not None:
        lg._stream_vehicle = vehicle_id
    if camera_sid is not None:
        lg._stream_camera = camera_sid

    lg._start_stream_thread()

    q: queue.Queue = queue.Queue(maxsize=10)
    _sse_hub.subscribe(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    if msg is None:
                        # 被踢出（队列积压满）：主动断开连接，浏览器 EventSource 会自动重连拿到新队列
                        break
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'ts': time.time()})}\n\n"
        except GeneratorExit:
            pass
        finally:
            _sse_hub.unsubscribe(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.route("/stream/setup", methods=["POST"])
def stream_setup():
    """配置数据流目标：设置要监控的车辆ID和相机传感器ID。
    之后连 /stream 就会自动推送这两个的数据。
    {"vehicle_id": 10, "camera_sid": 5}
    """
    lg = legacy()
    data = request.get_json(silent=True) or {}
    if "vehicle_id" in data:
        lg._stream_vehicle = data["vehicle_id"]
    if "camera_sid" in data:
        lg._stream_camera = data["camera_sid"]
    lg._start_stream_thread()
    return jsonify({"status": "ok", "vehicle_id": lg._stream_vehicle, "camera_sid": lg._stream_camera})


@bp.route("/stream/start", methods=["POST"])
def stream_start():
    """一键启动：生成车辆 + 挂相机 + 开自动驾驶 + 开启数据流。
    返回 {"vehicle_id": ..., "camera_sid": ...}
    """
    lg = legacy()

    # 1. 生成车辆
    bps = lg.world.get_blueprint_library().filter("vehicle.*")
    bp = np.random.choice(bps)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    if bp.has_attribute("color"):
        bp.set_attribute("color", np.random.choice(bp.get_attribute("color").recommended_values))

    spawn_pts = lg.world.get_map().get_spawn_points()
    vehicle = None
    for t in spawn_pts:
        vehicle = lg.world.try_spawn_actor(bp, t)
        if vehicle is not None:
            break
    if vehicle is None:
        return jsonify({"status": "error", "message": "no free spawn point"}), 409

    vehicle.set_autopilot(True)
    lg._stream_vehicle = vehicle.id
    lg._autopilot_state[vehicle.id] = True
    with lg._lock:
        lg._managed_actors.add(vehicle.id)

    # 2. 镜头跟随
    try:
        spectator = lg.world.get_spectator()
        pt = vehicle.get_transform()
        back = pt.get_forward_vector() * -6.0
        pt.location += back
        pt.location.z += 3.0
        pt.rotation.pitch = -14.0
        spectator.set_transform(pt)
    except Exception:
        pass

    # 3. 挂载 RGB 相机
    cam_bp = lg.world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "90")
    cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0))
    camera = lg.world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    lg._stream_camera = camera.id
    lg._camera_actor_ref = camera  # 保持引用，防止 GC 导致 listen 回调失效
    lg._sensor_refs[camera.id] = camera
    camera.listen(lambda data, sid=camera.id: lg._sensor_callback(sid, "camera", data))
    with lg._lock:
        lg._managed_actors.add(camera.id)

    time.sleep(1.0)  # 等第一帧

    lg._start_stream_thread()

    return jsonify({
        "status": "ok",
        "vehicle_id": vehicle.id,
        "vehicle_type": vehicle.type_id,
        "camera_sid": camera.id,
        "stream_url": "/stream",
    })


@bp.route("/preview/start", methods=["POST"])
def preview_start():
    """启动画面预览：在世界中放置一个静态俯瞰相机并推流。
    不依赖车辆，适用于离线实验或仅需观察 CARLA 场景时使用。"""
    lg = legacy()

    # 实验运行中不启动预览：避免抢占/销毁实验相机，SSE 沿用实验画面
    # （也消除前端「停止后延时 preview/start」与快速重启的新实验之间的竞态）
    if lg._any_experiment_running():
        return jsonify({"status": "ok", "message": "实验运行中，沿用实验画面，跳过预览相机"})

    # 先销毁上一次的预览相机：CARLA 每帧渲染所有存活相机，累积相机越用越卡
    old_sid = lg._stream_camera
    if old_sid is not None:
        try:
            old = lg.world.get_actor(old_sid)
            if old is not None and old.is_alive:
                if getattr(old, "is_listening", False):
                    old.stop()
                old.destroy()
        except Exception:
            pass
        with lg._lock:
            lg._managed_actors.discard(old_sid)
        lg._sensor_frames.pop(old_sid, None)
        lg._sensor_dtype.pop(old_sid, None)
        lg._sensor_refs.pop(old_sid, None)
        lg._stream_camera = None
        lg._camera_actor_ref = None

    data = request.get_json(silent=True) or {}
    width = int(data.get("width", 1280))
    height = int(data.get("height", 720))
    fov = float(data.get("fov", 90))

    # 以 spectator 当前位置为基础，拉高形成俯瞰视角
    spectator = lg.world.get_spectator()
    spec_transform = spectator.get_transform()
    cam_loc = carla.Location(
        x=spec_transform.location.x,
        y=spec_transform.location.y,
        z=spec_transform.location.z + 10.0,
    )
    cam_rot = carla.Rotation(pitch=-60.0, yaw=0.0, roll=0.0)
    cam_transform = carla.Transform(cam_loc, cam_rot)

    cam_bp = lg.world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))

    camera = lg.world.spawn_actor(cam_bp, cam_transform)
    lg._stream_camera = camera.id
    lg._camera_actor_ref = camera
    lg._sensor_refs[camera.id] = camera
    camera.listen(lambda data, sid=camera.id: lg._sensor_callback(sid, "camera", data))

    with lg._lock:
        lg._managed_actors.add(camera.id)

    # 不设置 _stream_vehicle，SSE 线程仅推送 camera 帧
    lg._start_stream_thread()
    time.sleep(0.5)

    return jsonify({
        "status": "ok",
        "camera_sid": camera.id,
        "location": {"x": round(cam_loc.x, 1), "y": round(cam_loc.y, 1), "z": round(cam_loc.z, 1)},
    })


@bp.route("/preview/stop", methods=["POST"])
def preview_stop():
    """停止预览相机"""
    lg = legacy()
    if lg._stream_camera is not None:
        actor = lg.world.get_actor(lg._stream_camera)
        if actor is not None and actor.is_alive:
            actor.destroy()
        with lg._lock:
            lg._managed_actors.discard(lg._stream_camera)
        lg._sensor_frames.pop(lg._stream_camera, None)
        lg._sensor_dtype.pop(lg._stream_camera, None)
        lg._sensor_refs.pop(lg._stream_camera, None)
        lg._stream_camera = None
        lg._camera_actor_ref = None
    return jsonify({"status": "ok"})
