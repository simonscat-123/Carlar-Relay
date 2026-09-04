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

# ── 路线障碍物采样参数（调参入口）─────────────────────────────────────────
OBSTACLE_COUNT = 4              # 每条路线规划的障碍物数量（个）
OBSTACLE_MIN_GAP = 7.0         # 障碍物之间的最小间距（m）
OBSTACLE_RANGE = (0.1, 0.9)   # 采样范围（占路线全长比例，避开起终点/自车）
OBSTACLE_MIN_ROUTE_LEN = 30.0   # 路线最短长度（m）：短于此不放置障碍
OBSTACLE_JUNCTION_CLEAR = 5.0  # 障碍物距路口的最小距离（m，避开减速/停车带）

def _sample_route_obstacles(waypoints, n=OBSTACLE_COUNT, min_gap=OBSTACLE_MIN_GAP):
    """在导航路线（路点 dict 列表）上随机采样 n 个障碍物位置，任意两个至少相隔 min_gap 米。
    偏移取路线中部（OBSTACLE_RANGE 比例区间），避免与起终点/自车重叠。返回 [{"x","y","z","yaw"},...]；
    路线过短放不下返回尽量多的点。
    单车道路段（行驶方向无同向相邻车道，自车无法换道避让）不放障碍，避免堵死自车。"""
    if len(waypoints) < 2:
        return []
    cum = [0.0]
    for i in range(1, len(waypoints)):
        cum.append(cum[-1] + math.hypot(waypoints[i]["x"] - waypoints[i - 1]["x"],
                                        waypoints[i]["y"] - waypoints[i - 1]["y"]))
    total = cum[-1]
    if total < OBSTACLE_MIN_ROUTE_LEN:
        return []
    lo, hi = OBSTACLE_RANGE[0] * total, OBSTACLE_RANGE[1] * total

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

    # 单车道判定：该位置所在车道若「无同向相邻 Driving 车道」（get_left/right_lane
    # 要么为空、要么为对向/交叉车道），自车无法换道避让，放置障碍会直接堵死自车
    # → 跳过不生成（避免阻碍车辆运行）。判定方式与实验10规划的可行驶域扩展一致
    # （前向点积 >0 才算同向可绕邻道）。
    def _has_escape(s):
        pos = interp(s)
        try:
            wp = world.get_map().get_waypoint(
                carla.Location(x=pos["x"], y=pos["y"], z=0.0),
                project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                return False
            cf = wp.transform.get_forward_vector()
            for nb in (wp.get_left_lane(), wp.get_right_lane()):
                if nb is not None and nb.lane_type == carla.LaneType.Driving:
                    nf = nb.transform.get_forward_vector()
                    if nf.x * cf.x + nf.y * cf.y > 0.0:
                        return True
            return False
        except Exception:
            return False

    # 路口弧长集合（采样时保持 OBSTACLE_JUNCTION_CLEAR 远离，避免把障碍放到路口及其减速/停车带）
    junction_s = [cum[i] for i, wp in enumerate(waypoints) if wp.get("is_junction")]

    def _far_from_junction(s):
        return all(abs(s - js) >= OBSTACLE_JUNCTION_CLEAR for js in junction_s)

    # 多次随机放置，找到「间距达标、非单车道、且离路口达标」的组合；
    # 否则退化为等分后逐点剔除不合格点
    best = []
    for _ in range(60):
        cand = sorted(random.uniform(lo, hi) for _ in range(n))
        if all(cand[i + 1] - cand[i] >= min_gap for i in range(len(cand) - 1)) \
                and all(_has_escape(s) and _far_from_junction(s) for s in cand):
            best = cand
            break
    if not best and hi - lo > 0:
        best = [lo + (hi - lo) * (i + 1) / (n + 1) for i in range(n)]
    return [interp(s) for s in best if _has_escape(s) and _far_from_junction(s)]


@app.route("/route/plan", methods=["POST"])
def route_plan():
    """用 GlobalRoutePlanner 规划全局路线。
    Body: {"start": {"x","y","yaw"}, "end": {"x","y","yaw"}, "sampling_resolution": 2.0}
    返回 {"route": [{"x","y","z","yaw"}], "length": m}"""
    data = request.get_json(silent=True) or {}
    s = data.get("start", {})
    e = data.get("end", {})
    sampling = float(data.get("sampling_resolution", 2.0))
    # 可配置的规划算法与三类成本惩罚（缺省关闭惩罚，保持原始 A* 行为）
    algorithm = str(data.get("algorithm", "astar")).lower()
    if algorithm not in ("astar", "dijkstra", "bfs"):
        algorithm = "astar"
    lane_change_cost = float(data.get("lane_change_cost", 0.0))
    intersection_cost = float(data.get("intersection_cost", 0.0))
    curvature_gain = float(data.get("curvature_gain", 0.0))

    start_loc = carla.Location(x=float(s.get("x", 0)), y=float(s.get("y", 0)), z=float(s.get("z", 0)))
    end_loc = carla.Location(x=float(e.get("x", 0)), y=float(e.get("y", 0)), z=float(e.get("z", 0)))

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        # 自定义成本权重：仅当任一惩罚 >0 时启用，否则保持默认按边长度寻路
        weight_fn = None
        if lane_change_cost > 0 or intersection_cost > 0 or curvature_gain > 0:
            import numpy as _np
            from agents.navigation.local_planner import RoadOption
            def _route_cost(_u, _v, edge):
                # 变道边的 add_edge 未携带 entry_vector/exit_vector（仅 exit_vector=None），
                # 需用 .get 防御性读取，否则换道时 KeyError
                c = edge.get('length', 0)
                if edge.get('type') in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                    c += lane_change_cost          # 抑制变道
                if edge.get('intersection'):
                    c += intersection_cost          # 抑制穿过路口
                ev, xv = edge.get('entry_vector'), edge.get('exit_vector')
                if ev is not None and xv is not None:
                    cosn = _np.clip(_np.dot(ev, xv) / (_np.linalg.norm(ev) * _np.linalg.norm(xv)), -1, 1)
                    c += curvature_gain * _np.arccos(cosn)   # 弯越急代价越高
                return c
            weight_fn = _route_cost
        grp = GlobalRoutePlanner(world.get_map(), sampling, algorithm=algorithm, weight_fn=weight_fn)
        path = grp.trace_route(start_loc, end_loc)
        waypoints = []
        total_len = 0.0
        prev = start_loc
        for wp, _ in path:
            loc = wp.transform.location
            waypoints.append({
                "x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2),
                "yaw": round(wp.transform.rotation.yaw, 1),
                "is_junction": bool(wp.is_junction),
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

