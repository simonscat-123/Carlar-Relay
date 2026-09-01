"""路径跟踪：Pure Pursuit + PID（API 实验ID 8）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 8: 车辆路径跟踪控制
# =============================================================================

_EXP08_RUNNING = False
_EXP08_ABORT = False


def _run_exp08(args):
    global _EXP08_RUNNING, _EXP08_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle
    _EXP_CURRENT_ID = 8
    _EXP08_RUNNING = True
    _EXP08_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验8 (路径跟踪控制) 启动")

    duration = float(args.get("duration", 40.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    target_speed = float(args.get("target_speed", 8.0))
    lookahead = float(args.get("lookahead", 5.0))
    seed = int(args.get("seed", 7))

    actors = []
    try:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=fixed_delta))
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        carla_map = world.get_map()
        spawn_pts = carla_map.get_spawn_points()
        rng = random.Random(seed)

        # 规划路径
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(carla_map, 2.0)
        for _ in range(100):
            a = rng.choice(spawn_pts)
            b = rng.choice(spawn_pts)
            if a.location.distance(b.location) >= 120:
                route = grp.trace_route(a.location, b.location)
                if route and len(route) >= 50:
                    break
        else:
            raise RuntimeError("无法规划 ≥120m 路径")

        route_wps = [(wp.transform.location.x, wp.transform.location.y) for wp, _ in route]
        _exp_log(f"路径规划: {len(route_wps)} 航点, 长度 ~{sum(math.hypot(route_wps[i][0]-route_wps[i-1][0], route_wps[i][1]-route_wps[i-1][1]) for i in range(1, len(route_wps))):.0f} m")

        # 生成车辆
        bp = world.get_blueprint_library().filter("vehicle.*")[0]
        vehicle = world.spawn_actor(bp, a)
        actors.append(vehicle)
        v_id = vehicle.id

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

        for a_actor in actors:
            with _lock:
                _managed_actors.add(a_actor.id)
        _exp_log(f"车辆已生成 (id={v_id})")

        settle_ticks = int(1.0 / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP08_ABORT:
                raise RuntimeError("已中止")
            world.tick()

        _exp_log(f"开始 Pure Pursuit + PID 控制 (target={target_speed} m/s)")

        total = int(duration / fixed_delta)
        rows = []
        wp_idx = 0
        pid_err, pid_int = 0.0, 0.0

        for i in range(total):
            if _EXP08_ABORT:
                break
            world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds
            tform = vehicle.get_transform()
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

            steer, wp_idx, cte = _pure_pursuit_steer(tform, route_wps, lookahead, wp_idx)
            throttle, brake, pid_err, pid_int = _pid_speed_control(speed, target_speed,
                                                                     pid_err, pid_int, fixed_delta)

            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            rows.append({"time": t, "x": tform.location.x, "y": tform.location.y,
                         "speed": speed, "steer": steer, "throttle": throttle,
                         "brake": brake, "cte": cte,
                         "target_x": route_wps[min(wp_idx, len(route_wps) - 1)][0],
                         "target_y": route_wps[min(wp_idx, len(route_wps) - 1)][1]})

            if i % 4 == 0:
                _push_to_sse({"experiment": {"id": 8, "trajectory": {
                    "frame": i + 1, "t": round(t, 3), "speed": round(speed, 2),
                    "steer": round(steer, 3), "cte": round(cte, 2),
                    "x": round(tform.location.x, 2), "y": round(tform.location.y, 2),
                    "progress": round((i + 1) / total * 100, 1)}}})

            if wp_idx >= len(route_wps) - 5:
                _exp_log("已到达终点附近")
                break

        cte_rmse = math.sqrt(sum(r["cte"] ** 2 for r in rows) / len(rows)) if rows else 0
        _push_to_sse({"experiment": {"id": 8, "result": {
            "elapsed": round(t if rows else 0, 1), "rows": len(rows),
            "cte_rmse": round(cte_rmse, 3)}}})
        _exp_log(f"实验8 完成 — {len(rows)} 行, CTE RMSE={cte_rmse:.3f} m")
    except Exception as e:
        _exp_log(f"实验8 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        try:
            _sensor_frames.pop(cam.id, None)
            _sensor_refs.pop(cam.id, None)
        except Exception:
            pass
        _EXP08_RUNNING = False
        if _EXP08_ABORT:
            _push_to_sse({"experiment": {"id": 8, "status": "stopped"}})


@app.route("/experiment/8/start", methods=["POST"])
def experiment_8_start():
    global _EXP08_RUNNING
    if _EXP08_RUNNING:
        return jsonify({"status": "error", "message": "实验8 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp08, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 8, "message": "实验8 已启动"})


@app.route("/experiment/8/stop", methods=["POST"])
def experiment_8_stop():
    global _EXP08_ABORT
    _EXP08_ABORT = True
    return jsonify({"status": "ok", "message": "实验8 停止请求已发送"})

