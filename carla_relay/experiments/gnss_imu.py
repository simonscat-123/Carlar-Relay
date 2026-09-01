"""GNSS/IMU 数据采集（API 实验ID 2，已并入定位分析）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 2: GNSS/IMU 数据采集
# =============================================================================

_EXP02_RUNNING = False
_EXP02_ABORT = False


def _run_exp02(args):
    global _EXP02_RUNNING, _EXP02_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle
    _EXP_CURRENT_ID = 2
    _EXP02_RUNNING = True
    _EXP02_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验2 (GNSS/IMU 数据采集) 启动")

    duration = float(args.get("duration", 20.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    settle_s = float(args.get("settle_seconds", 1.5))
    launch_s = float(args.get("launch_seconds", 1.5))
    launch_throttle = float(args.get("launch_throttle", 0.7))
    seed = int(args.get("seed", 7))

    actors = []
    try:
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

        # GNSS + IMU
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", str(fixed_delta))
        gnss = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
        actors.append(gnss)
        _sensor_refs[gnss.id] = gnss
        gnss.listen(lambda d, sid=gnss.id: _sensor_callback(sid, "gnss", d))

        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(fixed_delta))
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        actors.append(imu)
        _sensor_refs[imu.id] = imu
        imu.listen(lambda d, sid=imu.id: _sensor_callback(sid, "imu", d))

        for a in actors:
            with _lock:
                _managed_actors.add(a.id)
        _exp_log(f"车辆+GNSS+IMU 已挂载 (id={v_id})")

        latest_imu = {"accel": (0, 0, 0), "gyro": (0, 0, 0), "compass": 0}

        def _read_imu():
            sid = imu.id
            if sid in _sensor_frames:
                d = json.loads(_sensor_frames[sid].decode())
                latest_imu["accel"] = (d.get("accelerometer", [0, 0, 0]) or [0, 0, 0])
                latest_imu["gyro"] = (d.get("gyroscope", [0, 0, 0]) or [0, 0, 0])
                latest_imu["compass"] = d.get("compass", 0) or 0

        # 稳定期
        settle_ticks = int(settle_s / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP02_ABORT:
                raise RuntimeError("已中止")
            vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))
            world.tick()

        # 启动期
        launch_ticks = int(launch_s / fixed_delta)
        for _ in range(launch_ticks):
            if _EXP02_ABORT:
                raise RuntimeError("已中止")
            vehicle.apply_control(carla.VehicleControl(throttle=launch_throttle))
            world.tick()

        vehicle.set_autopilot(True, tm.get_port())
        _exp_log("自动驾驶已启用，开始采集")

        total = int(duration / fixed_delta)
        rows = []
        for i in range(total):
            if _EXP02_ABORT:
                break
            world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds
            tform = vehicle.get_transform()
            vel = vehicle.get_velocity()

            # GNSS
            gnss_data = {}
            if gnss.id in _sensor_frames:
                gnss_data = json.loads(_sensor_frames[gnss.id].decode())
            lat = gnss_data.get("latitude", 0) or 0
            lon = gnss_data.get("longitude", 0) or 0
            alt = gnss_data.get("altitude", 0) or 0

            # IMU
            _read_imu()
            ax, ay, az = latest_imu["accel"]
            gx, gy, gz = latest_imu["gyro"]
            compass = latest_imu["compass"]

            rows.append({"frame": i + 1, "time": t, "gt_x": tform.location.x, "gt_y": tform.location.y, "gt_z": tform.location.z,
                         "gt_yaw": tform.rotation.yaw, "gt_speed": math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2),
                         "lat": lat, "lon": lon, "alt": alt, "accel_x": ax, "accel_y": ay, "accel_z": az,
                         "gyro_x": gx, "gyro_y": gy, "gyro_z": gz, "compass": compass})

            if i % 4 == 0:
                _push_to_sse({"experiment": {"id": 2, "trajectory": {
                    "frame": i + 1, "t": round(t, 3), "lat": round(lat, 6), "lon": round(lon, 6),
                    "alt": round(alt, 2), "gt_speed": round(rows[-1]["gt_speed"], 2),
                    "progress": round((i + 1) / total * 100, 1)}}})

        _push_to_sse({"experiment": {"id": 2, "result": {"elapsed": round(t if rows else 0, 1), "rows": len(rows)}}})
        _exp_log(f"实验2 完成 — {len(rows)} 行")
    except Exception as e:
        _exp_log(f"实验2 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        for sid in [cam.id, gnss.id, imu.id]:
            try:
                _sensor_frames.pop(sid, None)
                _sensor_refs.pop(sid, None)
            except Exception:
                pass
        _EXP02_RUNNING = False
        if _EXP02_ABORT:
            _push_to_sse({"experiment": {"id": 2, "status": "stopped"}})


@app.route("/experiment/2/start", methods=["POST"])
def experiment_2_start():
    global _EXP02_RUNNING
    if _EXP02_RUNNING:
        return jsonify({"status": "error", "message": "实验2 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp02, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 2, "message": "实验2 已启动"})


@app.route("/experiment/2/stop", methods=["POST"])
def experiment_2_stop():
    global _EXP02_ABORT
    _EXP02_ABORT = True
    return jsonify({"status": "ok", "message": "实验2 停止请求已发送"})

