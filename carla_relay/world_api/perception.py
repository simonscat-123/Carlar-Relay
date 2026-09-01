"""感知查询 API：障碍物 / 信号灯 / 路径障碍采样与规划。

本文件由 carla_relay.world_api.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# API: 闭环自动驾驶（实验10）— 感知/定位/规划/控制 四合一
# =============================================================================

# --- 感知：障碍物列表 ---

@app.route("/perception/obstacles")
def perception_obstacles():
    """扫描自车附近的车辆/行人/骑行者，返回障碍物列表。
    Query: ?vehicle_id=10&radius=50"""
    vid = request.args.get("vehicle_id", type=int)
    radius = request.args.get("radius", 50.0, type=float)
    if vid is None:
        return jsonify({"status": "error", "message": "vehicle_id required"}), 400
    ego = world.get_actor(vid)
    if ego is None or not ego.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    ego_loc = ego.get_location()
    ego_fwd = ego.get_transform().get_forward_vector()
    ego_yaw = math.radians(ego.get_transform().rotation.yaw)
    obstacles = []
    idx = 0
    actor_list = world.get_actors()
    for actor in actor_list:
        if actor.id == vid:
            continue
        tid = actor.type_id
        is_vehicle = tid.startswith("vehicle.")
        is_walker = tid.startswith("walker.")
        if not (is_vehicle or is_walker):
            continue
        loc = actor.get_location()
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > radius:
            continue
        # 方位角（相对自车朝向）
        bearing = math.degrees(math.atan2(dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw),
                                          -dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)))
        vel = actor.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
        # 大小估算
        bb = actor.bounding_box.extent
        size = max(bb.x, bb.y) * 2
        category = "pedestrian" if is_walker else _dynamic_class(tid)
        idx += 1
        obstacles.append({
            "idx": idx, "category": category,
            "dist": round(dist, 1), "bearing": round(bearing, 0),
            "size": round(size, 1), "vel": round(speed, 1),
            "actor_id": actor.id,
        })
    obstacles.sort(key=lambda o: o["dist"])
    return jsonify({"obstacles": obstacles, "count": len(obstacles)})


# --- 感知：红绿灯状态 ---

_TL_STATE_MAP = {
    carla.TrafficLightState.Green: "green",
    carla.TrafficLightState.Yellow: "yellow",
    carla.TrafficLightState.Red: "red",
    carla.TrafficLightState.Off: "off",
}


@app.route("/perception/traffic_lights")
def perception_traffic_lights():
    """扫描附近红绿灯状态。Query: ?vehicle_id=10&radius=80"""
    vid = request.args.get("vehicle_id", type=int)
    radius = request.args.get("radius", 80.0, type=float)
    ego_loc = None
    if vid is not None:
        ego = world.get_actor(vid)
        if ego is not None and ego.is_alive:
            ego_loc = ego.get_location()
    tls = world.get_actors().filter("traffic.traffic_light*")
    result = []
    for tl in tls:
        loc = tl.get_location()
        if ego_loc is not None:
            dist = loc.distance(ego_loc)
            if dist > radius:
                continue
        else:
            dist = 0.0
        state_str = _TL_STATE_MAP.get(tl.state, "unknown")
        result.append({
            "id": tl.id,
            "state": state_str,
            "location": {"x": round(loc.x, 1), "y": round(loc.y, 1), "z": round(loc.z, 1)},
            "distance": round(dist, 1),
        })
    result.sort(key=lambda t: t["distance"])
    return jsonify({"traffic_lights": result, "count": len(result)})


# --- 规划：全局路线 ---

def _sample_route_obstacles(waypoints, n=3, min_gap=10.0):
    """在导航路线（路点 dict 列表）上随机采样 n 个障碍物位置，任意两个至少相隔 min_gap 米。
    偏移取路线中部（15%~85%），避免与起终点/自车重叠。返回 [{"x","y","z","yaw"},...]；
    路线过短放不下返回尽量多的点。"""
    if len(waypoints) < 2:
        return []
    cum = [0.0]
    for i in range(1, len(waypoints)):
        cum.append(cum[-1] + math.hypot(waypoints[i]["x"] - waypoints[i - 1]["x"],
                                        waypoints[i]["y"] - waypoints[i - 1]["y"]))
    total = cum[-1]
    if total < 30:
        return []
    lo, hi = 0.15 * total, 0.85 * total
    # 多次随机放置，找到满足最小间距的组合；否则退化为等分
    best = []
    for _ in range(40):
        cand = sorted(random.uniform(lo, hi) for _ in range(n))
        if all(cand[i + 1] - cand[i] >= min_gap for i in range(len(cand) - 1)):
            best = cand
            break
    if not best and hi - lo > 0:
        best = [lo + (hi - lo) * (i + 1) / (n + 1) for i in range(n)]

    def interp(s):
        p = 0
        while p < len(waypoints) - 2 and cum[p + 1] < s:
            p += 1
        seg = cum[p + 1] - cum[p]
        t = 0.0 if seg <= 0 else (s - cum[p]) / seg
        x = waypoints[p]["x"] + (waypoints[p + 1]["x"] - waypoints[p]["x"]) * t
        y = waypoints[p]["y"] + (waypoints[p + 1]["y"] - waypoints[p]["y"]) * t
        z = waypoints[p]["z"] + (waypoints[p + 1]["z"] - waypoints[p]["z"]) * t
        yaw = math.degrees(math.atan2(waypoints[p + 1]["y"] - waypoints[p]["y"],
                                      waypoints[p + 1]["x"] - waypoints[p]["x"]))
        return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "yaw": round(yaw, 1)}

    return [interp(s) for s in best]


@app.route("/route/plan", methods=["POST"])
def route_plan():
    """用 GlobalRoutePlanner 规划全局路线。
    Body: {"start": {"x","y","yaw"}, "end": {"x","y","yaw"}, "sampling_resolution": 2.0}
    返回 {"route": [{"x","y","z","yaw"}], "length": m}"""
    data = request.get_json(silent=True) or {}
    s = data.get("start", {})
    e = data.get("end", {})
    sampling = float(data.get("sampling_resolution", 2.0))

    start_loc = carla.Location(x=float(s.get("x", 0)), y=float(s.get("y", 0)), z=float(s.get("z", 0)))
    end_loc = carla.Location(x=float(e.get("x", 0)), y=float(e.get("y", 0)), z=float(e.get("z", 0)))

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(world.get_map(), sampling)
        path = grp.trace_route(start_loc, end_loc)
        waypoints = []
        total_len = 0.0
        prev = start_loc
        for wp, _ in path:
            loc = wp.transform.location
            waypoints.append({
                "x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2),
                "yaw": round(wp.transform.rotation.yaw, 1),
            })
            total_len += prev.distance(loc)
            prev = loc
        obstacles = _sample_route_obstacles(waypoints)
        global _EXP10_PLANNED_OBSTACLES, _EXP10_PLANNED_CHANGED, _EXP10_LAST_PLAN
        # 障碍物与路线绑定：起终点均与上次规划一致（容差内）视为「同一路线重复规划」，
        # 沿用已规划的障碍物位置，不置重新规划标志（下次运行实验不清理、不重新生成）
        same_route = (
            _EXP10_LAST_PLAN is not None
            and math.hypot(start_loc.x - _EXP10_LAST_PLAN["start"][0],
                          start_loc.y - _EXP10_LAST_PLAN["start"][1]) <= _EXP10_ROUTE_SAME_TOL
            and math.hypot(end_loc.x - _EXP10_LAST_PLAN["end"][0],
                           end_loc.y - _EXP10_LAST_PLAN["end"][1]) <= _EXP10_ROUTE_SAME_TOL
        )
        if same_route:
            obstacles = _EXP10_PLANNED_OBSTACLES
        else:
            _EXP10_PLANNED_OBSTACLES = obstacles
            _EXP10_PLANNED_CHANGED = True  # 规划了新路线：下次运行实验清空旧障碍并重新生成
            _EXP10_LAST_PLAN = {
                "start": (start_loc.x, start_loc.y),
                "end": (end_loc.x, end_loc.y),
            }
        return jsonify({
            "status": "ok", "route": waypoints, "obstacles": obstacles,
            "length": round(total_len, 1), "count": len(waypoints),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

