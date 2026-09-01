"""LiDAR-Camera 投影（API 实验ID 6）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 6: LiDAR-Camera 投影
# =============================================================================

_EXP06_RUNNING = False
_EXP06_ABORT = False


def _run_exp06(args):
    global _EXP06_RUNNING, _EXP06_ABORT, _EXP_CURRENT_ID
    _EXP_CURRENT_ID = 6
    _EXP06_RUNNING = True
    _EXP06_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验6 (LiDAR-Camera 投影) 启动")

    fixed_delta = float(args.get("fixed_delta", 0.05))
    seed = int(args.get("seed", 7))
    fov = float(args.get("fov", 90.0))
    width = int(args.get("width", 1920))
    height = int(args.get("height", 1080))

    actors = []
    try:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=fixed_delta))
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        bp = world.get_blueprint_library().filter("vehicle.*")[0]
        rng = random.Random(seed)
        vehicle = world.spawn_actor(bp, rng.choice(world.get_map().get_spawn_points()))
        actors.append(vehicle)

        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.8)), attach_to=vehicle)
        actors.append(cam)
        _sensor_refs[cam.id] = cam
        cam.listen(lambda d, sid=cam.id: _sensor_callback(sid, "camera", d))

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / fixed_delta))
        lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.4)), attach_to=vehicle)
        actors.append(lidar)
        _sensor_refs[lidar.id] = lidar
        latest_lidar_pts = None

        def _lidar_cb(data):
            nonlocal latest_lidar_pts
            latest_lidar_pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))[:, :3]

        lidar.listen(_lidar_cb)

        for a in actors:
            with _lock:
                _managed_actors.add(a.id)
        _exp_log(f"车辆+Camera+LiDAR 已挂载 (id={vehicle.id})")

        settle_ticks = int(1.0 / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP06_ABORT:
                raise RuntimeError("已中止")
            world.tick()

        vehicle.set_autopilot(True, tm.get_port())
        _exp_log("等待 Camera+LiDAR 同步帧...")

        # 等待同步帧
        max_wait = int(3.0 / fixed_delta)
        for i in range(max_wait):
            if _EXP06_ABORT:
                raise RuntimeError("已中止")
            world.tick()
            if cam.id in _sensor_frames and latest_lidar_pts is not None:
                break

        if cam.id not in _sensor_frames or latest_lidar_pts is None:
            raise RuntimeError("未能获取同步帧")

        # 投影计算
        cam_tform = cam.get_transform()
        lidar_tform = lidar.get_transform()

        # 相机内参
        fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        fy = height / (2.0 * math.tan(math.radians(fov) / 2.0))
        cx, cy = width / 2.0, height / 2.0

        # LiDAR → 世界 → 相机
        lidar_to_world = np.array(lidar_tform.get_matrix())
        world_to_camera = np.array(cam_tform.get_inverse_matrix())
        lidar_to_camera = world_to_camera @ lidar_to_world

        pts_homo = np.hstack([latest_lidar_pts, np.ones((latest_lidar_pts.shape[0], 1))])
        cam_pts = (lidar_to_camera @ pts_homo.T).T[:, :3]

        # CARLA 坐标系: x→前, y→右, z→上 → 相机坐标系: z→前, x→右, y→下
        cam_pts = np.column_stack([cam_pts[:, 1], -cam_pts[:, 2], cam_pts[:, 0]])
        front_mask = cam_pts[:, 2] > 0
        valid = cam_pts[front_mask]
        u = (valid[:, 0] * fx / valid[:, 2] + cx).astype(int)
        v = (valid[:, 1] * fy / valid[:, 2] + cy).astype(int)
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        sample_pts = valid[in_image]

        sample_count = min(30, len(sample_pts))
        sample_rows = [{"index": i, "lidar_x": sample_pts[i, 2], "lidar_y": sample_pts[i, 0],
                        "lidar_z": -sample_pts[i, 1], "camera_x": valid[in_image][i, 0],
                        "camera_y": valid[in_image][i, 1], "camera_z": valid[in_image][i, 2],
                        "pixel_u": u[in_image][i], "pixel_v": v[in_image][i],
                        "depth": valid[in_image][i, 2]}
                       for i in range(sample_count)]

        _exp_log(f"投影完成: {len(sample_pts)} 个点投影到图像内 (共 {len(latest_lidar_pts)} 点)")
        _push_to_sse({"experiment": {"id": 6, "result": {"elapsed": 0, "projected_points": len(sample_pts),
                    "total_points": len(latest_lidar_pts), "samples": sample_rows}}})
        _exp_log("实验6 完成")
    except Exception as e:
        _exp_log(f"实验6 错误: {e}")
    finally:
        for sid in [cam.id, lidar.id]:
            try:
                _sensor_frames.pop(sid, None)
                _sensor_refs.pop(sid, None)
            except Exception:
                pass
        _EXP06_RUNNING = False
        if _EXP06_ABORT:
            _push_to_sse({"experiment": {"id": 6, "status": "stopped"}})


@app.route("/experiment/6/start", methods=["POST"])
def experiment_6_start():
    global _EXP06_RUNNING
    if _EXP06_RUNNING:
        return jsonify({"status": "error", "message": "实验6 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp06, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 6, "message": "实验6 已启动"})


@app.route("/experiment/6/stop", methods=["POST"])
def experiment_6_stop():
    global _EXP06_ABORT
    _EXP06_ABORT = True
    return jsonify({"status": "ok", "message": "实验6 停止请求已发送"})

