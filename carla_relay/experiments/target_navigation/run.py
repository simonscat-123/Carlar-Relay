"""多源定位与目标导航（API 实验ID 9）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 9: 多源定位与目标导航
# =============================================================================

_EXP09_RUNNING = False
_EXP09_ABORT = False


def _run_exp09(args):
    global _EXP09_RUNNING, _EXP09_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle
    _EXP_CURRENT_ID = 9
    _EXP09_RUNNING = True
    _EXP09_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验9 (多源定位与目标导航) 启动")

    duration = float(args.get("duration", 60.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    seed = int(args.get("seed", 7))
    target_x = float(args.get("target_x", 100))
    target_y = float(args.get("target_y", 0))
    target_speed = float(args.get("target_speed", 10.0))
    lookahead = float(args.get("lookahead", 5.0))
    safe_distance = float(args.get("safe_distance", 10.0))
    gnss_noise = float(args.get("gnss_noise", 0.5))
    ins_noise = float(args.get("ins_noise", 0.1))
    alpha = float(args.get("alpha", 0.9))
    traffic_vehicles = int(args.get("traffic_vehicles", 10))
    arrival_distance = float(args.get("arrival_distance", 3.0))

    actors = []
    traffic_actors = []
    try:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=fixed_delta))
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)

        carla_map = world.get_map()
        spawn_pts = carla_map.get_spawn_points()
        rng = random.Random(seed)

        start_pt = rng.choice(spawn_pts)
        target_loc = carla.Location(x=start_pt.location.x + target_x, y=start_pt.location.y + target_y,
                                     z=start_pt.location.z)

        # 全局路径
        from carla_relay.vendor.grp_loader import make_route_planner
        grp = make_route_planner(carla_map, 2.0)
        route = grp.trace_route(start_pt.location, target_loc)
        if not route or len(route) < 10:
            raise RuntimeError("路径规划失败")
        route_wps = [(wp.transform.location.x, wp.transform.location.y) for wp, _ in route]

        # 车辆
        bp = world.get_blueprint_library().filter("vehicle.*")[0]
        vehicle = world.spawn_actor(bp, start_pt)
        actors.append(vehicle)
        v_id = vehicle.id

        # 相机 + GNSS + IMU + LiDAR
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-5.5, z=2.8),
                                carla.Rotation(pitch=-15)), attach_to=vehicle)
        actors.append(cam)
        _sensor_refs[cam.id] = cam
        cam.listen(lambda d, sid=cam.id: _sensor_callback(sid, "camera", d))
        _stream_camera = cam.id
        _stream_vehicle = v_id

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

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "40")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "80000")
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / fixed_delta))
        lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.4)), attach_to=vehicle)
        actors.append(lidar)
        _sensor_refs[lidar.id] = lidar
        latest_obstacle = float("inf")

        def _lidar_cb(data):
            nonlocal latest_obstacle
            pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
            front = pts[(pts[:, 0] > 0) & (np.abs(pts[:, 1]) < 2.5) & (pts[:, 2] > -1.7)]
            latest_obstacle = float(np.min(np.linalg.norm(front[:, :2], axis=1))) if len(front) else float("inf")

        lidar.listen(_lidar_cb)

        # 交通车辆
        for _ in range(traffic_vehicles):
            try:
                tbp = rng.choice(world.get_blueprint_library().filter("vehicle.*"))
                tsp = rng.choice(spawn_pts)
                if tsp.location.distance(vehicle.get_location()) < 20:
                    continue
                tv = world.spawn_actor(tbp, tsp)
                tv.set_autopilot(True, tm.get_port())
                traffic_actors.append(tv)
            except Exception:
                continue
        _exp_log(f"生成了 {len(traffic_actors)} 辆交通车辆")

        for a_actor in actors + traffic_actors:
            with _lock:
                _managed_actors.add(a_actor.id)
        _exp_log(f"主车+传感器+交通车辆已就位 (id={v_id})")

        settle_ticks = int(1.0 / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP09_ABORT:
                raise RuntimeError("已中止")
            world.tick()

        _exp_log(f"目标: ({target_loc.x:.1f}, {target_loc.y:.1f}), 路径 {len(route_wps)} 航点")

        # 初始化定位
        gt0 = vehicle.get_transform()
        lat0, lon0 = 0.0, 0.0
        if gnss.id in _sensor_frames:
            gd = json.loads(_sensor_frames[gnss.id].decode())
            lat0 = gd.get("latitude", 0) or 0
            lon0 = gd.get("longitude", 0) or 0
        RAD = math.pi / 180.0
        lat0r = lat0 * RAD
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat0r) + 1.175 * math.cos(4 * lat0r)
        m_per_deg_lon = 111412.84 * math.cos(lat0r) - 93.5 * math.cos(3 * lat0r)
        fused_x, fused_y = gt0.location.x, gt0.location.y
        ins_vx, ins_vy = 0.0, 0.0
        prev_yaw = 0.0

        total = int(duration / fixed_delta)
        rows = []
        wp_idx = 0
        pid_err, pid_int = 0.0, 0.0

        for i in range(total):
            if _EXP09_ABORT:
                break
            world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds
            tform = vehicle.get_transform()
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

            # GNSS + INS 互补滤波
            gnss_data = {}
            if gnss.id in _sensor_frames:
                gnss_data = json.loads(_sensor_frames[gnss.id].decode())
            lat = gnss_data.get("latitude", lat0) or lat0
            lon = gnss_data.get("longitude", lon0) or lon0
            gnss_x = gt0.location.x + (lon - lon0) * m_per_deg_lon + rng.gauss(0, gnss_noise)
            gnss_y = gt0.location.y + (lat - lat0) * m_per_deg_lat + rng.gauss(0, gnss_noise)

            imu_data = {}
            if imu.id in _sensor_frames:
                imu_data = json.loads(_sensor_frames[imu.id].decode())
            ax = (imu_data.get("accelerometer", [0, 0, 0]) or [0, 0, 0])[0] + rng.gauss(0, ins_noise)
            ay = (imu_data.get("accelerometer", [0, 0, 0]) or [0, 0, 0])[1] + rng.gauss(0, ins_noise)
            yaw = math.radians(tform.rotation.yaw)

            if i > 0:
                ins_ax = ax * math.cos(yaw) - ay * math.sin(yaw)
                ins_ay = ax * math.sin(yaw) + ay * math.cos(yaw)
                ins_vx += ins_ax * fixed_delta
                ins_vy += ins_ay * fixed_delta
                fused_x += ins_vx * fixed_delta
                fused_y += ins_vy * fixed_delta
            fused_x = (1 - alpha) * fused_x + alpha * gnss_x
            fused_y = (1 - alpha) * fused_y + alpha * gnss_y

            # 控制
            fused_tform = carla.Transform(carla.Location(x=fused_x, y=fused_y, z=tform.location.z),
                                           tform.rotation)
            steer, wp_idx, cte = _pure_pursuit_steer(fused_tform, route_wps, lookahead, wp_idx)

            # LiDAR 安全
            desired_speed = target_speed
            if latest_obstacle < safe_distance:
                desired_speed = max(1.0, target_speed * (latest_obstacle / safe_distance))
            if latest_obstacle < safe_distance * 0.6:
                desired_speed = 0

            throttle, brake, pid_err, pid_int = _pid_speed_control(speed, desired_speed,
                                                                     pid_err, pid_int, fixed_delta)
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            dist_to_target = math.hypot(tform.location.x - target_loc.x, tform.location.y - target_loc.y)
            rows.append({
                "time": t, "target_x": target_loc.x, "target_y": target_loc.y,
                "gt_x": tform.location.x, "gt_y": tform.location.y,
                "gnss_x": gnss_x, "gnss_y": gnss_y,
                "fused_x": fused_x, "fused_y": fused_y,
                "speed": speed, "desired_speed": desired_speed,
                "steer": steer, "throttle": throttle, "brake": brake,
                "cte": cte, "front_obstacle": latest_obstacle,
                "dist_to_target": dist_to_target,
            })

            if i % 4 == 0:
                _push_to_sse({"experiment": {"id": 9, "trajectory": {
                    "frame": i + 1, "t": round(t, 3), "speed": round(speed, 2),
                    "steer": round(steer, 3), "cte": round(cte, 2),
                    "fused_x": round(fused_x, 2), "fused_y": round(fused_y, 2),
                    "front_obstacle": round(latest_obstacle, 2) if latest_obstacle != float("inf") else None,
                    "dist_to_target": round(dist_to_target, 2),
                    "progress": round((i + 1) / total * 100, 1)}}})

            if dist_to_target < arrival_distance and speed < 0.3:
                _exp_log(f"已到达目标 (距离 {dist_to_target:.2f} m)")
                break
            prev_yaw = yaw

        arrived = dist_to_target < arrival_distance
        _push_to_sse({"experiment": {"id": 9, "result": {
            "elapsed": round(t if rows else 0, 1), "rows": len(rows),
            "arrived": arrived,
            "final_distance": round(dist_to_target, 2) if rows else None}}})
        _exp_log(f"实验9 完成 — {len(rows)} 行, 已到达={arrived}")
    except Exception as e:
        _exp_log(f"实验9 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        for sid in [cam.id, gnss.id, imu.id, lidar.id]:
            try:
                _sensor_frames.pop(sid, None)
                _sensor_refs.pop(sid, None)
            except Exception:
                pass
        for tv in traffic_actors:
            try:
                with _lock:
                    _managed_actors.discard(tv.id)
                tv.destroy()
            except Exception:
                pass
        _EXP09_RUNNING = False
        if _EXP09_ABORT:
            _push_to_sse({"experiment": {"id": 9, "status": "stopped"}})


@app.route("/experiment/9/start", methods=["POST"])
def experiment_9_start():
    global _EXP09_RUNNING
    if _EXP09_RUNNING:
        return jsonify({"status": "error", "message": "实验9 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp09, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 9, "message": "实验9 已启动"})


@app.route("/experiment/9/stop", methods=["POST"])
def experiment_9_stop():
    global _EXP09_ABORT
    _EXP09_ABORT = True
    return jsonify({"status": "ok", "message": "实验9 停止请求已发送"})

