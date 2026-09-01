"""导航地图与路径规划（API 实验ID 7）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 7: 导航地图与路径规划
# =============================================================================

_EXP07_RUNNING = False
_EXP07_ABORT = False


def _run_exp07(args):
    global _EXP07_RUNNING, _EXP07_ABORT, _EXP_CURRENT_ID
    _EXP_CURRENT_ID = 7
    _EXP07_RUNNING = True
    _EXP07_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验7 (路径规划与地图匹配) 启动")

    seed = int(args.get("seed", 7))
    sampling_res = float(args.get("sampling_resolution", 2.0))
    noise_std = float(args.get("noise_std", 1.0))
    map_wp_dist = float(args.get("map_waypoint_distance", 1.0))
    rng = random.Random(seed)

    try:
        carla_map = world.get_map()
        spawn_pts = carla_map.get_spawn_points()
        if len(spawn_pts) < 2:
            raise RuntimeError("地图中 spawn 点不足")

        # 随机起点/终点（距离 > 100m）
        for _ in range(100):
            a, b = rng.sample(spawn_pts, 2)
            if a.location.distance(b.location) >= 100:
                break
        else:
            a, b = spawn_pts[0], spawn_pts[-1]

        _exp_log(f"路径: ({a.location.x:.1f}, {a.location.y:.1f}) → ({b.location.x:.1f}, {b.location.y:.1f})")
        _exp_log(f"直线距离: {a.location.distance(b.location):.1f} m")

        # 全局路径规划（A* 简化版：沿 waypoint 拓扑推演）
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(carla_map, sampling_res)
        route = grp.trace_route(a.location, b.location)
        if not route:
            raise RuntimeError("路径规划失败：无可行路径")

        route_wps = [(wp.transform.location.x, wp.transform.location.y) for wp, _ in route]
        _exp_log(f"路径规划完成: {len(route_wps)} 个航点, 路径长度 {sum(math.hypot(route_wps[i][0] - route_wps[i-1][0], route_wps[i][1] - route_wps[i-1][1]) for i in range(1, len(route_wps))):.1f} m")

        # 地图匹配
        rows = []
        step = max(1, len(route_wps) // 200)
        for i in range(0, len(route_wps), step):
            rx, ry = route_wps[i]
            nx = rx + rng.gauss(0, noise_std)
            ny = ry + rng.gauss(0, noise_std)
            matched = carla_map.get_waypoint(carla.Location(x=nx, y=ny, z=0), project_to_road=True, lane_type=carla.LaneType.Driving)
            if matched:
                mx, my = matched.transform.location.x, matched.transform.location.y
                err = math.hypot(nx - mx, ny - my)
                rows.append({"index": i, "route_x": rx, "route_y": ry, "noisy_x": nx, "noisy_y": ny,
                             "matched_x": mx, "matched_y": my, "match_error_m": err,
                             "road_id": matched.road_id, "lane_id": matched.lane_id})

            if i % 10 == 0:
                _push_to_sse({"experiment": {"id": 7, "trajectory": {
                    "index": i, "total": len(route_wps), "route_x": rx, "route_y": ry,
                    "progress": round((i + 1) / len(route_wps) * 100, 1)}}})

        mean_err = sum(r["match_error_m"] for r in rows) / len(rows) if rows else 0
        _exp_log(f"地图匹配完成: {len(rows)} 个点, 平均误差 {mean_err:.3f} m")
        _push_to_sse({"experiment": {"id": 7, "result": {
            "elapsed": 0, "route_length": len(route_wps),
            "matched_points": len(rows), "mean_error_m": round(mean_err, 3)}}})
        _exp_log("实验7 完成")
    except Exception as e:
        _exp_log(f"实验7 错误: {e}")
    finally:
        _EXP07_RUNNING = False
        if _EXP07_ABORT:
            _push_to_sse({"experiment": {"id": 7, "status": "stopped"}})


@app.route("/experiment/7/start", methods=["POST"])
def experiment_7_start():
    global _EXP07_RUNNING
    if _EXP07_RUNNING:
        return jsonify({"status": "error", "message": "实验7 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp07, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 7, "message": "实验7 已启动"})


@app.route("/experiment/7/stop", methods=["POST"])
def experiment_7_stop():
    global _EXP07_ABORT
    _EXP07_ABORT = True
    return jsonify({"status": "ok", "message": "实验7 停止请求已发送"})

