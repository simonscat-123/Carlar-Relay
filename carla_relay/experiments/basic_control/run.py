"""CARLA 基本控制（API 实验ID 1）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# =============================================================================
# 实验 1: CARLA 基本控制
# =============================================================================

_EXP01_RUNNING = False
_EXP01_ABORT = False


def _run_exp01(args):
    global _EXP01_RUNNING, _EXP01_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle
    _EXP_CURRENT_ID = 1
    _EXP01_RUNNING = True
    _EXP01_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验1 (CARLA 基本控制) 启动")

    duration = float(args.get("duration", 20.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    seed = int(args.get("seed", 7))
    log_every = int(args.get("log_every", 10))

    actors = []
    try:
        settings = world.get_settings()
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=fixed_delta))
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        bp = world.get_blueprint_library().filter("vehicle.*")[0]
        spawn_pts = world.get_map().get_spawn_points()
        rng = random.Random(seed)
        vehicle = world.spawn_actor(bp, rng.choice(spawn_pts))
        actors.append(vehicle)
        v_id = vehicle.id

        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.8)), attach_to=vehicle)
        actors.append(cam)
        _sensor_refs[cam.id] = cam
        cam.listen(lambda d, sid=cam.id: _sensor_callback(sid, "camera", d))
        _stream_camera = cam.id
        _stream_vehicle = v_id
        _exp_log(f"车辆已生成 (id={v_id})")

        for a in actors:
            with _lock:
                _managed_actors.add(a.id)

        settle_ticks = int(1.0 / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP01_ABORT:
                raise RuntimeError("已中止")
            world.tick()

        vehicle.set_autopilot(True, tm.get_port())
        _exp_log("自动驾驶已启用，开始采集")

        total = int(duration / fixed_delta)
        rows = []
        # 相对时间基准：elapsed_seconds 是会话累计世界时钟，须以首采集帧为 0 起点
        t0 = None
        for i in range(total):
            if _EXP01_ABORT:
                break
            world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds
            if t0 is None:
                t0 = t
            t = t - t0
            tform = vehicle.get_transform()
            rows.append({"frame": i + 1, "time": t, "x": tform.location.x, "y": tform.location.y, "yaw": tform.rotation.yaw})
            if i % 4 == 0:
                _push_to_sse({"experiment": {"id": 1, "trajectory": {"frame": i + 1, "t": round(t, 3), "x": round(tform.location.x, 2), "y": round(tform.location.y, 2), "yaw": round(tform.rotation.yaw, 1), "progress": round((i + 1) / total * 100, 1)}}})
            if (i + 1) % log_every == 0:
                _exp_log(f"frame={i+1} x={tform.location.x:.2f} y={tform.location.y:.2f} yaw={tform.rotation.yaw:.1f}")

        _push_to_sse({"experiment": {"id": 1, "result": {"elapsed": round(t if rows else 0, 1), "rows": len(rows)}}})
        _exp_log(f"实验1 完成 — {len(rows)} 行")
    except Exception as e:
        _exp_log(f"实验1 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        try:
            _sensor_frames.pop(cam.id, None)
            _sensor_refs.pop(cam.id, None)
        except Exception:
            pass
        _EXP01_RUNNING = False
        if _EXP01_ABORT:
            _push_to_sse({"experiment": {"id": 1, "status": "stopped"}})


@app.route("/experiment/1/start", methods=["POST"])
def experiment_1_start():
    global _EXP01_RUNNING
    if _EXP01_RUNNING:
        return jsonify({"status": "error", "message": "实验1 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp01, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 1, "message": "实验1 已启动"})


@app.route("/experiment/1/stop", methods=["POST"])
def experiment_1_stop():
    global _EXP01_ABORT
    _EXP01_ABORT = True
    return jsonify({"status": "ok", "message": "实验1 停止请求已发送"})

