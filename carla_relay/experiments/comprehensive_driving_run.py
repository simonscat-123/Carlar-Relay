"""综合驾驶（下）：闭环自动驾驶主循环 / 障碍物管理 / 路由 · 前端关卡 comprehensive-driving（API 实验ID 10）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
def _run_exp10(args):
    """闭环自动驾驶实验线程：
    生成车辆 → 挂载全套传感器 → 全局路线规划 → Pure Pursuit + PID 控制循环 → SSE 推送
    """
    global _EXP10_RUNNING, _EXP10_ABORT, _EXP_CURRENT_ID, _EXP10_VEHICLE_ID, _EXP10_ROUTE
    global _EXP10_SPAWN_REQ
    global _stream_vehicle, _stream_camera, _stream_semantic, _stream_bird, _stream_bbox
    global _EXP10_PLANNED_CHANGED
    global _EXP10_OBSTACLE_ACTORS

    with _EXP10_LOCK:
        _EXP10_RUNNING = True
        _EXP10_ABORT = False
    _EXP_CURRENT_ID = 10
    _EXP_LOG.clear()
    _exp_log("实验10 (闭环自动驾驶) 启动")
    _sweep_stale_actors()
    _exp_log("正在确定起终点并配置仿真环境…")

    duration = float(args.get("duration", 60.0))
    target_speed = float(args.get("target_speed", 8.0))
    lookahead = float(args.get("lookahead", 8.0))
    kp_steer = float(args.get("kp_steer", 1.4))
    steer_delay = float(args.get("steer_delay", 0.0))
    brake_force = float(args.get("brake_force", 1.0))
    safe_dist = float(args.get("safe_distance", 12.0))
    gnss_noise = float(args.get("gnss_noise", 1.0))
    ins_noise = float(args.get("ins_noise", 0.1))
    alpha = float(args.get("alpha", 0.08))
    _exp_log(f"本次参数: gnssσ={gnss_noise:.2f} insσ={ins_noise:.2f} alpha={alpha:.3f} "
             f"target={target_speed:.1f}m/s lookahead={lookahead:.1f}m kp={kp_steer:.2f}")
    # 感知闭环：开启后障碍物距离由 bbox 相机（实例+语义分割）单目估计，
    # 不再查询世界真值；关闭则沿用真值扫描（两种模式可运行中实时切换对比）
    perception_mode = bool(args.get("perception", False))
    _exp_log(f"感知闭环: {'开（bbox 相机单目测距）' if perception_mode else '关（世界真值）'}")
    # 初始化运行中可实时调整的控制参数（前端滑杆可在运行中覆盖）
    with _EXP10_CTRL_LOCK:
        _EXP10_CTRL.clear()
        _EXP10_CTRL.update({
            "kp_steer": kp_steer,
            "lookahead": lookahead,
            "target_speed": target_speed,
            "steer_delay": steer_delay,
            "brake_force": brake_force,
            "perception": 1.0 if perception_mode else 0.0,
        })
    gps_failure = bool(args.get("gps_failure", False))
    spawn_pedestrian = bool(args.get("spawn_pedestrian", False))
    sampling_res = float(args.get("sampling_resolution", 2.0))

    start_idx = int(args.get("start_spawn_idx", 0))
    end_idx = int(args.get("end_spawn_idx", 1))
    # 前端点选的起终点坐标（世界坐标），优先使用；未提供时回退到 spawn 索引
    start_coord = args.get("start")
    end_coord = args.get("end")

    def _snap(loc):
        """把坐标吸附到最近的可行驶车道航点，避免生成在路外"""
        try:
            wp = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            return wp.transform
        except Exception:
            return carla.Transform(loc)

    created_sids = []
    vehicle = None
    _plan_log_f = None    # 规划调试日志文件句柄（主循环前打开，finally 关闭）
    _exp10_old_settings = None

    try:
        spawn_pts = world.get_map().get_spawn_points()
        # 确定生成变换与终点位置
        if start_coord:
            start_loc = carla.Location(x=float(start_coord.get("x", 0)), y=float(start_coord.get("y", 0)), z=0.0)
            spawn_tf = _snap(start_loc)
        else:
            if start_idx >= len(spawn_pts):
                _exp_log(f"生成点索引越界: start={start_idx}")
                _push_to_sse({"experiment": {"id": 10, "status": "error", "message": "spawn idx out of range"}})
                return
            spawn_tf = spawn_pts[start_idx]
        if end_coord:
            end_loc = carla.Location(x=float(end_coord.get("x", 0)), y=float(end_coord.get("y", 0)), z=0.0)
        else:
            if end_idx >= len(spawn_pts):
                _exp_log(f"生成点索引越界: end={end_idx}")
                _push_to_sse({"experiment": {"id": 10, "status": "error", "message": "spawn idx out of range"}})
                return
            end_loc = spawn_pts[end_idx].location

        _exp_log(f"起终点: ({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f}) → ({end_loc.x:.1f}, {end_loc.y:.1f}), 直线 {spawn_tf.location.distance(end_loc):.1f} m")

        # 同步模式 + 固定时间步，保证控制循环内车辆真正推进会；结束时恢复原设置
        _exp10_old_settings = world.get_settings()
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=0.05))
        try:
            _tmp_tm = client.get_trafficmanager(8000)
            _tmp_tm.set_synchronous_mode(True)
        except Exception:
            pass

        # 1. 生成车辆
        _exp_log("正在生成主车…")
        bps = world.get_blueprint_library().filter("vehicle.*")
        bp = bps[0]
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")
        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle is None:
            _exp_log("车辆生成失败")
            return
        _stream_vehicle = vehicle.id
        with _lock:
            _managed_actors.add(vehicle.id)
        _EXP10_VEHICLE_ID = vehicle.id

        # 2. 挂载传感器
        _exp_log("正在挂载传感器（相机 ×4 / LiDAR / GNSS / IMU）…")
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0)), attach_to=vehicle)
        _stream_camera = cam.id
        _sensor_refs[cam.id] = cam
        # 前相机原始 BGRA 帧缓存：包围框渲染需要未编码帧（_sensor_callback 只存 JPEG）
        rgb_raw = {"raw": None, "w": 1280, "h": 720}

        def _on_front_rgb(d, sid=cam.id):
            rgb_raw["raw"] = d.raw_data
            rgb_raw["w"] = d.width
            rgb_raw["h"] = d.height
            _sensor_callback(sid, "camera", d)

        cam.listen(_on_front_rgb)
        created_sids.append(cam.id)
        with _lock:
            _managed_actors.add(cam.id)

        # 实例分割相机（包围框检测，参照官方 bounding_boxes.py；与前相机同位姿，
        # 半分辨率降低掩码计算量，渲染时坐标自动缩放到 RGB 帧）
        inst_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        inst_bp.set_attribute("image_size_x", "640")
        inst_bp.set_attribute("image_size_y", "360")
        inst = world.spawn_actor(inst_bp, carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0)), attach_to=vehicle)
        _stream_bbox = inst.id
        _sensor_refs[inst.id] = inst
        inst.listen(lambda d, sid=inst.id: _sensor_callback(sid, "instance", d))
        created_sids.append(inst.id)
        with _lock:
            _managed_actors.add(inst.id)

        # 语义分割相机
        sem_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", "640")
        sem_bp.set_attribute("image_size_y", "360")
        sem = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0)), attach_to=vehicle)
        _stream_semantic = sem.id
        _sensor_refs[sem.id] = sem
        sem.listen(lambda d, sid=sem.id: _sensor_callback(sid, "semantic", d))
        created_sids.append(sem.id)
        with _lock:
            _managed_actors.add(sem.id)

        # 高空俯视相机（跟随车辆，真实渲染俯瞰画面）
        bird_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bird_bp.set_attribute("image_size_x", "960")
        bird_bp.set_attribute("image_size_y", "960")
        bird_bp.set_attribute("fov", "90")
        bird = world.spawn_actor(
            bird_bp,
            carla.Transform(carla.Location(z=45), carla.Rotation(pitch=-90, yaw=0, roll=0)),
            attach_to=vehicle,
        )
        _stream_bird = bird.id
        _sensor_refs[bird.id] = bird
        bird.listen(lambda d, sid=bird.id: _sensor_callback(sid, "camera", d))
        created_sids.append(bird.id)
        with _lock:
            _managed_actors.add(bird.id)

        # LiDAR
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "50000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.4)), attach_to=vehicle)
        _sensor_refs[lidar.id] = lidar
        lidar.listen(lambda d, sid=lidar.id: _sensor_callback(sid, "lidar", d))
        created_sids.append(lidar.id)
        with _lock:
            _managed_actors.add(lidar.id)

        # GNSS
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
        _sensor_refs[gnss.id] = gnss
        gnss.listen(lambda d, sid=gnss.id: _sensor_callback(sid, "gnss", d))
        created_sids.append(gnss.id)
        with _lock:
            _managed_actors.add(gnss.id)

        # IMU
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        _sensor_refs[imu.id] = imu
        imu.listen(lambda d, sid=imu.id: _sensor_callback(sid, "imu", d))
        created_sids.append(imu.id)
        with _lock:
            _managed_actors.add(imu.id)

        # 碰撞传感器（评分报告用）：碰撞事件 → 事故记录 + SSE 实时告警。
        # 回调在传感器线程触发，只做轻量追加；1 秒内的连续接触合并为一次事故。
        col_events = []       # 原始事件 [{wall, cls, impulse}]
        col_incidents = []     # 合并后的事故 [{wall, cls, impulse, contacts}]
        col_lock = threading.Lock()

        def _on_collision(event):
            try:
                imp = event.normal_impulse
                mag = math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)
                other = event.other_actor
                tid = other.type_id if other is not None else ""
                if tid.startswith("walker."):
                    cls = "行人"
                elif tid.startswith("vehicle."):
                    cls = "车辆"
                else:
                    cls = "静态物体"
                now = time.time()
                with col_lock:
                    col_events.append({"wall": now, "cls": cls, "impulse": round(mag, 1)})
                    if col_incidents and now - col_incidents[-1]["wall"] <= 1.0:
                        inc = col_incidents[-1]
                        inc["impulse"] = max(inc["impulse"], round(mag, 1))
                        inc["contacts"] += 1
                        is_new = False
                    else:
                        col_incidents.append({"wall": now, "cls": cls,
                                              "impulse": round(mag, 1), "contacts": 1})
                        is_new = True
                if is_new:
                    _push_to_sse({"experiment": {"id": 10, "collision": {
                        "cls": cls, "impulse": round(mag, 1), "count": len(col_incidents),
                    }}})
            except Exception:
                pass

        col_bp = world.get_blueprint_library().find("sensor.other.collision")
        col = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        _sensor_refs[col.id] = col
        col.listen(_on_collision)
        created_sids.append(col.id)
        with _lock:
            _managed_actors.add(col.id)

        _exp_log(f"车辆+传感器就绪: vid={vehicle.id} cam={cam.id} inst={inst.id} lidar={lidar.id} gnss={gnss.id} imu={imu.id} collision={col.id}")

        # 3.5 只有「重新规划后运行」才清空世界中遗留的车辆/行人（含上次实验残留，重启后仍有效），
        #     并保留本车 ego；反之（停止后调参再启动、未重新规划）则沿用世界已有障碍，不清空。
        need_fresh = _EXP10_PLANNED_CHANGED
        _EXP10_PLANNED_CHANGED = False  # 消费标志
        if need_fresh:
            _exp_log("正在清理上次实验残留…")
            try:
                _clear_exp10_obstacles(world)
            except Exception as exc:
                _exp_log(f"清除上次障碍物失败: {exc}")

        # 3. 全局路线规划（起点终点沿用上面确定的 start_loc/spawn_tf 与 end_loc）
        start_loc = spawn_tf.location if start_coord else spawn_pts[start_idx].location
        _exp_log("正在规划全局路线（构建路网拓扑，可能需要数秒）…")
        carla_map = world.get_map()
        route_lane_ids = []       # 与 route_wp 平行：各路点所在车道 id（车道级避障判据用）
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            grp = GlobalRoutePlanner(carla_map, sampling_res)
            path = grp.trace_route(start_loc, end_loc)
            route_wp = [wp.transform.location for wp, _ in path]
            route_lane_ids = [(wp.road_id, wp.lane_id) for wp, _ in path]
            route_wps = [wp for wp, _ in path]   # 保留 waypoint 对象（车道/邻道查询用）
            _exp_log(f"路线规划完成: {len(route_wp)} waypoints")
        except Exception as exc:
            _exp_log(f"路线规划失败({exc})，使用直线插值")
            route_wp = [start_loc, end_loc]
            route_wps = []
        route_lane_ids += [None] * (len(route_wp) - len(route_lane_ids))
        route_wps += [None] * (len(route_wp) - len(route_wps))
        _EXP10_ROUTE = list(route_wp)

        # ── 参考线层：弧长表 + Frenet 变换 + 可行驶域（时空联合规划的基础设施）──
        # s_tab: 与 route_wp 平行的累计弧长；参考线本身即车道中心线（GRP 沿车道中心采样）
        s_tab = [0.0]
        for _j in range(1, len(route_wp)):
            s_tab.append(s_tab[-1] + route_wp[_j].distance(route_wp[_j - 1]))

        def _frenet(px, py, j0=0, j1=None):
            """世界坐标 → Frenet (s, l, 切向tx, 切向ty)。
            在 [j0, j1) 路点窗口内找最近路点，s = 弧长 + 切向投影，
            l = 相对切线的横向偏移（左正，与控制层约定一致）。"""
            if j1 is None:
                j1 = len(route_wp)
            best_j, best_d2 = j0, float("inf")
            for j in range(j0, j1):
                d2 = (px - route_wp[j].x) ** 2 + (py - route_wp[j].y) ** 2
                if d2 < best_d2:
                    best_d2, best_j = d2, j
            a = route_wp[max(0, best_j - 1)]
            b = route_wp[min(len(route_wp) - 1, best_j + 1)]
            tx, ty = b.x - a.x, b.y - a.y
            tl = math.hypot(tx, ty)
            if tl < 1e-6:
                tx, ty = 1.0, 0.0
            else:
                tx, ty = tx / tl, ty / tl
            rx, ry = px - route_wp[best_j].x, py - route_wp[best_j].y
            return (s_tab[best_j] + tx * rx + ty * ry,
                    tx * ry - ty * rx, tx, ty)

        def _world(s, l):
            """Frenet (s, l) → 世界坐标 (x, y)。s 超出路线范围时钳到端点。"""
            s = max(0.0, min(s, s_tab[-1]))
            lo, hi = 0, len(s_tab) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s_tab[mid] <= s:
                    lo = mid
                else:
                    hi = mid
            nxt = min(lo + 1, len(route_wp) - 1)
            a, b = route_wp[lo], route_wp[nxt]
            seg = s_tab[nxt] - s_tab[lo]
            u = 0.0 if seg < 1e-6 else max(0.0, min(1.0, (s - s_tab[lo]) / seg))
            px, py = a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u
            tx, ty = b.x - a.x, b.y - a.y
            tl = math.hypot(tx, ty)
            if tl < 1e-6:
                tx, ty = 1.0, 0.0
            else:
                tx, ty = tx / tl, ty / tl
            return px - ty * l, py + tx * l   # 左法向 (-ty, tx) × l

        # 可行驶域：从路线当前车道向两侧扩展「同向 Driving」车道，得到相对参考线的
        # 横向边界 (l_min, l_max)——对向车道/路缘即边界，逆行轨迹从此无法通过硬约束。
        _bounds_cache = {}

        def _drivable_bounds(j):
            """route_wp[j] 所在车道的可行驶横向边界（相对该处参考线）。失败返回 None。"""
            key = route_lane_ids[j] if j < len(route_lane_ids) else None
            if key is None:
                return None
            if key in _bounds_cache:
                return _bounds_cache[key]
            wp = route_wps[j]
            l_min = l_max = None
            try:
                if wp is not None:
                    w0 = wp.lane_width
                    edge_l, edge_r = -w0 / 2.0, w0 / 2.0
                    cur = wp
                    # 向左扩展：仅接受 Driving 且前向同向的车道
                    for _ in range(3):
                        nb = cur.get_left_lane()
                        if nb is None or nb.lane_type != carla.LaneType.Driving:
                            break
                        nf = nb.transform.get_forward_vector()
                        cf = cur.transform.get_forward_vector()
                        if nf.x * cf.x + nf.y * cf.y <= 0.0:
                            break   # 对向车道，禁入
                        edge_l -= nb.lane_width
                        cur = nb
                    cur = wp
                    for _ in range(3):
                        nb = cur.get_right_lane()
                        if nb is None or nb.lane_type != carla.LaneType.Driving:
                            break
                        nf = nb.transform.get_forward_vector()
                        cf = cur.transform.get_forward_vector()
                        if nf.x * cf.x + nf.y * cf.y <= 0.0:
                            break
                        edge_r += nb.lane_width
                        cur = nb
                    l_min, l_max = edge_l, edge_r
            except Exception:
                pass
            _bounds_cache[key] = (l_min, l_max)
            return l_min, l_max

        def _bounds_at_s(s):
            """弧长 s 处的可行驶域边界（换道轨迹跨多车道段时逐点取当地边界，
            而非只用车头处的边界——前方车道收窄/对向开始处才不会误入）。"""
            lo, hi = 0, len(s_tab) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s_tab[mid] <= s:
                    lo = mid
                else:
                    hi = mid
            b = _drivable_bounds(lo)
            if b is not None and b[0] is not None:
                return b
            return None

        def _lat_lane_ok(l_t, s_probe):
            """横向偏移 l_t 在弧长 s_probe 处是否落在「同向 Driving 车道」。
            直接取该偏移点的地图车道做前向点积校验——不依赖边界缓存的推断，
            对向车道（含斜向/路口交叉车道，点积≤0.3）一律判不可行。"""
            try:
                px, py = _world(s_probe, l_t)
                wp2 = carla_map.get_waypoint(
                    carla.Location(x=px, y=py, z=0.0),
                    project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp2 is None:
                    return False
                lo, hi = 0, len(s_tab) - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if s_tab[mid] <= s_probe:
                        lo = mid
                    else:
                        hi = mid
                a = route_wp[max(0, lo - 1)]
                b = route_wp[min(len(route_wp) - 1, lo + 1)]
                tx, ty = b.x - a.x, b.y - a.y
                f = wp2.transform.get_forward_vector()
                return (f.x * tx + f.y * ty) > 0.3
            except Exception:
                return False

        # ── 信号灯绑定（一次性）：停止线 → 参考线弧长 s_stop。
        # 只绑定「停止线所在车道属于本路线」的灯——管着本路线的灯才有效；
        # 运行时用 s 差判定：车越过停止线后该灯自动退出考虑（committed 语义），
        # 路口内/出口不再被交叉方向的红灯误刹。
        tl_bindings = []
        try:
            _route_lane_set = set(l for l in route_lane_ids if l is not None)
            for _tl in world.get_actors().filter("traffic.traffic_light*"):
                try:
                    for _swp in _tl.get_stop_waypoints():
                        if (_swp.road_id, _swp.lane_id) in _route_lane_set:
                            _s_stop = _frenet(_swp.transform.location.x,
                                              _swp.transform.location.y)[0]
                            tl_bindings.append({"tl": _tl, "s_stop": _s_stop})
                            break
                except Exception:
                    pass
            if tl_bindings:
                tl_bindings.sort(key=lambda b: b["s_stop"])
                _exp_log(f"信号灯绑定: {len(tl_bindings)} 处停止线已关联到路线")
        except Exception as exc:
            _exp_log(f"信号灯绑定失败: {exc}")

        # 4. 横穿行人
        if spawn_pedestrian:
            try:
                ego_fwd = vehicle.get_transform().get_forward_vector()
                ego_tf = vehicle.get_transform()
                right = carla.Vector3D(x=-ego_fwd.y, y=ego_fwd.x, z=0)
                ped_loc = carla.Location(
                    x=ego_tf.location.x + ego_fwd.x * 15 + right.x * 6,
                    y=ego_tf.location.y + ego_fwd.y * 15 + right.y * 6,
                    z=ego_tf.location.z,
                )
                walker_bp = world.get_blueprint_library().find("walker.pedestrian.0001")
                walker = world.try_spawn_actor(walker_bp, carla.Transform(ped_loc))
                if walker:
                    ctl_bp = world.get_blueprint_library().find("controller.ai.walker")
                    ctl = world.spawn_actor(ctl_bp, carla.Transform(ped_loc), attach_to=walker)
                    cross = carla.Location(x=ego_tf.location.x + ego_fwd.x * 10 - right.x * 6,
                                           y=ego_tf.location.y + ego_fwd.y * 10 - right.y * 6)
                    ctl.start(); ctl.go_to_location(cross); ctl.set_max_speed(1.5)
                    _EXP10_PEDI.extend([walker, ctl])
                    with _lock:
                        _managed_actors.update([walker.id, ctl.id])
                    _exp_log("横穿行人生成成功")
            except Exception as exc:
                _exp_log(f"行人生成失败: {exc}")

        # 4.5 障碍物与路线绑定：
        #     - 规划了新路线 → 清掉旧障碍并按新规划生成；
        #     - 同一路线再次运行 → 沿用世界中已有障碍，不清理不重生；
        #       若其间被其它实验/清理销毁，则按原规划位置补齐缺失的障碍。
        _exp10_obstacles = []
        if need_fresh:
            try:
                _exp10_obstacles = _exp10_spawn_route_obstacles(world, _EXP10_PLANNED_OBSTACLES)
                n_ok = sum(1 for r in _exp10_obstacles if r.get("ok"))
                if _exp10_obstacles:
                    _exp_log(f"已自动生成 {n_ok}/{len(_exp10_obstacles)} 个路线障碍物")
            except Exception as exc:
                _exp_log(f"路线障碍物生成失败: {exc}")
            _EXP10_OBSTACLE_ACTORS = [
                {"id": r["id"], "pos": p}
                for r, p in zip(_exp10_obstacles, _EXP10_PLANNED_OBSTACLES)
                if r.get("ok")
            ]
        else:
            alive = []
            missing = []
            for ent in _EXP10_OBSTACLE_ACTORS:
                try:
                    a = world.get_actor(ent["id"])
                except Exception:
                    a = None
                if a is not None and a.is_alive:
                    # 被撞离原位（上次运行避障剐蹭/物理推移）的障碍视为缺失：
                    # 不在路线上的障碍没有避障意义，销毁后按原规划位置重生成
                    disp = math.hypot(a.get_location().x - ent["pos"]["x"],
                                      a.get_location().y - ent["pos"]["y"])
                    if disp <= 3.0:
                        alive.append(ent)
                    else:
                        _exp_log(f"检测到障碍物被撞离原位 {disp:.1f}m，按原位置重生成")
                        try:
                            a.destroy()
                        except Exception:
                            pass
                        missing.append(ent["pos"])
                else:
                    missing.append(ent["pos"])
            # 路线未变但障碍从未生成过（如全部生成失败），按规划全量补齐
            if not _EXP10_OBSTACLE_ACTORS and _EXP10_PLANNED_OBSTACLES:
                missing = list(_EXP10_PLANNED_OBSTACLES)
            if missing:
                try:
                    respawned = _exp10_spawn_route_obstacles(world, missing)
                    n_ok = sum(1 for r in respawned if r.get("ok"))
                    if n_ok:
                        _exp_log(f"检测到 {len(missing)} 个路线障碍物缺失，已按原位置补齐 {n_ok} 个")
                    alive.extend(
                        {"id": r["id"], "pos": p}
                        for r, p in zip(respawned, missing)
                        if r.get("ok")
                    )
                except Exception as exc:
                    _exp_log(f"补齐路线障碍物失败: {exc}")
            _EXP10_OBSTACLE_ACTORS = alive

        # 5. 统一设置全图信号灯时长，压缩等待、避免超长红灯卡停车辆（简化逻辑：不区分是否贴近路线）
        try:
            _tl_set = 0
            for tl in world.get_actors().filter("traffic.traffic_light*"):
                try:
                    tl.set_frozen(False)
                    tl.set_green_time(15.0)
                    tl.set_yellow_time(2.0)
                    tl.set_red_time(5.0)
                    _tl_set += 1
                except Exception:
                    pass
            if _tl_set:
                _exp_log(f"已设置全图 {_tl_set} 处信号灯：红 5s / 黄 2s / 绿 15s")
        except Exception as exc:
            _exp_log(f"设置信号灯失败: {exc}")

        _start_stream_thread()
        # 同步模式下沉稳车辆 + 让 GNSS/IMU 帧先到达
        _exp_log("稳定预热中…")
        for _ in range(20):
            if _EXP10_ABORT:
                break
            world.tick()

        # 6. 控制循环
        _push_to_sse({"experiment": {"id": 10, "status": "running", "route_len": len(route_wp)}})

        t0 = time.time()
        wp_idx = 0
        arrived = False
        fused_loc = vehicle.get_location()
        ins_vel = carla.Vector3D()
        fused_yaw_deg = 0.0  # 融合航向（度），GNSS 观测航向 + INS 航向递推的互补滤波结果
        prev_gt_yaw_deg = None  # 上一帧真值航向，用于差分出航向角速度（可靠源）
        steer_history = []
        _bbox_diag = {"ok": 0, "skip": 0, "err": 0}  # bbox 帧渲染诊断计数
        _perceive_diag = {"err": 0}   # 相机感知异常诊断计数
        prev_gt = None                # 上一帧真值位置（里程统计用）
        traveled = 0.0                # 累计行驶里程（m）
        perception_used = False       # 本次运行是否启用过感知闭环（报告用）

        import collections
        loc_err_history = collections.deque(maxlen=240)
        speed_history = collections.deque(maxlen=240)
        cte_history = collections.deque(maxlen=240)
        _dbg_frame = 0  # 诊断日志计数器（每 20 帧≈1s 输出一次）

        # ── 决策规划状态（Frenet 采样式时空联合规划，参照 Werling/PythonRobotics）──
        # 每帧重估：行为 FSM 输出意图标签（日志/可视化/教学用），
        # 6 条候选轨迹（横向 {保持,左换,右换} × 纵向 {巡航,停驻}）经硬约束筛选后取代价最小者。
        _ego_bb = vehicle.bounding_box.extent
        ego_half_w = float(_ego_bb.y)    # 自车半宽（碰撞检查/走廊判据用）
        ego_half_len = float(_ego_bb.x)  # 自车半长（碰撞检查含障碍长度，修"量到中心"缺陷）
        # 规划参数
        T_HORIZON = 4.0        # 轨迹展开时长（s）
        DT_PLAN = 0.25         # 展开步长（s）
        DEC_WIN = 60.0         # 决策窗口：邻道占用/本道被占的检查范围（m）
        RED_MARGIN = 0.5       # 红灯停止线前的停止余量（沿 s，停在线前 0.5m）
        STOP_MARGIN = 6.0      # 障碍后缘前的刹停余量（沿 s；前端 safe_distance 滑杆可调）
        COLL_S = 0.5           # 纵向碰撞余量（m）
        COLL_L = 0.3           # 横向碰撞余量（m）
        COMFORT_A = 2.5        # 舒适减速度（STOP 剖面用，m/s²）
        MAX_DECEL = 4.0        # 最大可用减速度（仲裁用，brake≈1）
        # 感知模式下无真值尺寸，按类别取典型半长（保守补偿，修"只算障碍中心"缺陷）
        _PERC_HALF_LEN = {"walker": 0.3, "vehicle": 2.4}
        # SSE 兼容字段（由规划器驱动，前端零改动）
        avoiding = False
        avoid_side = 0
        avoid_lat_target = 0.0
        avoid_dest_lane = None
        fsm_state = "CRUISE"   # 行为状态标签：CRUISE/APPROACH_RED/FOLLOW/LANE_CHANGE
        _avoid_prev = False    # 上一帧是否换道中（转换日志用）
        plan_traj = []         # 规划轨迹（世界坐标点列，Pure Pursuit 前视 + 鸟瞰参考线）
        plan_end_l = 0.0       # 规划轨迹末端的横向偏移（鸟瞰参考线续接用）

        # ── 规划调试日志系统：逐帧决策细节写入文件（排障用）──
        # SSE 实验日志只推关键事件（模式切换/异常/结果），高频细节全部落盘：
        # 每帧一行（自车状态/信号灯/障碍/候选与代价/拒绝统计/控制输出），
        # 外加事件行（最优解切换/信号灯变化/回退兜底）。文件路径启动时推给前端。
        import os as _os
        import tempfile as _tempfile
        try:
            _plan_log_dir = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "logs")
        except NameError:
            _plan_log_dir = _os.path.join(_tempfile.gettempdir(), "carla_exp10_logs")
        _os.makedirs(_plan_log_dir, exist_ok=True)
        _plan_log_path = _os.path.join(
            _plan_log_dir, f"exp10_plan_{time.strftime('%Y%m%d_%H%M%S')}.log")
        _plan_log_f = open(_plan_log_path, "w", encoding="utf-8", buffering=1)

        def _plan_log(msg):
            try:
                _plan_log_f.write(
                    f"[{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d}] {msg}\n")
            except Exception:
                pass

        _plan_log(f"=== 实验10 规划调试日志 start ===")
        _plan_log(f"参数: target={target_speed} lookahead={lookahead} kp={kp_steer} "
                  f"safe_dist={safe_dist} brake_force={brake_force} "
                  f"perception={'on' if perception_mode else 'off'} "
                  f"route={len(route_wp)}wp 总弧长={s_tab[-1]:.0f}m "
                  f"tl_bindings={len(tl_bindings)}")
        _exp_log(f"规划调试日志: {_plan_log_path}")
        _best_prev_key = None    # 上一帧最优解 (mode, l_t)——切换事件检测
        _tl_state_prev = None    # 上一帧信号灯状态——变化事件检测

        while not _EXP10_ABORT:
            world.tick()  # 同步模式：推进一帧，车辆据此移动

            # 处理前端"生成障碍物"请求：必须在主循环线程内操作 CARLA 客户端，
            # 否则与 world.tick() 跨线程并发调用会导致协议错乱、CARLA 崩溃
            with _EXP10_SPAWN_LOCK:
                _sreq = _EXP10_SPAWN_REQ
            if _sreq is not None and not _sreq["done"].is_set():
                try:
                    _sreq["result"] = _exp10_spawn_obstacle(world, vehicle, route_wp, _sreq["distance"])
                except Exception as exc:
                    _sreq["result"] = {"ok": False, "message": f"生成异常: {exc}"}
                finally:
                    _sreq["done"].set()

            t = time.time() - t0
            # 不再按时长结束实验：跑到终点或手动停止才退出，
            # 避免车辆在红灯/拥堵等待时因时间到而判定"未到达"。

            # 读取前端运行中实时调整的控制参数（滑杆即时生效，无需重启实验）
            with _EXP10_CTRL_LOCK:
                kp_steer = float(_EXP10_CTRL.get("kp_steer", kp_steer))
                lookahead = float(_EXP10_CTRL.get("lookahead", lookahead))
                target_speed = float(_EXP10_CTRL.get("target_speed", target_speed))
                steer_delay = float(_EXP10_CTRL.get("steer_delay", steer_delay))
                brake_force = float(_EXP10_CTRL.get("brake_force", brake_force))
                perception_on = float(_EXP10_CTRL.get("perception", 0.0)) > 0.5

            # 真值
            ego_tf = vehicle.get_transform()
            gt_loc = ego_tf.location
            gt_yaw = ego_tf.rotation.yaw
            if prev_gt is not None:
                traveled += prev_gt.distance(gt_loc)
            prev_gt = carla.Location(x=gt_loc.x, y=gt_loc.y, z=gt_loc.z)
            perception_used = perception_used or perception_on

            # GNSS + 互补滤波
            gnss_data = None
            if gnss.id in _sensor_frames:
                try:
                    gnss_data = json.loads(_sensor_frames[gnss.id].decode())
                except Exception:
                    pass

            if gnss_data and not (gps_failure and random.random() < 0.3):
                noise = gnss_noise * (20.0 if gps_failure else 4.0)
                ngx = gt_loc.x + random.gauss(0, noise)
                ngy = gt_loc.y + random.gauss(0, noise)
            else:
                ngx = gt_loc.x + random.gauss(0, gnss_noise * 10)
                ngy = gt_loc.y + random.gauss(0, gnss_noise * 10)

            # INS 推算：以真值车速为基准叠加有界测量噪声（等效里程计/INS 误差）。
            # 速度加噪后限幅，防止个别异常帧把推算值拉飞
            true_vel = vehicle.get_velocity()
            ins_vel.x = max(-60.0, min(60.0, true_vel.x + random.gauss(0, ins_noise * 8.0)))
            ins_vel.y = max(-60.0, min(60.0, true_vel.y + random.gauss(0, ins_noise * 8.0)))
            ins_x = fused_loc.x + ins_vel.x * 0.05
            ins_y = fused_loc.y + ins_vel.y * 0.05
            a = alpha * (0.1 if gps_failure else 1.0)
            fused_loc = carla.Location(
                x=a * ngx + (1 - a) * ins_x,
                y=a * ngy + (1 - a) * ins_y,
                z=gt_loc.z,
            )
            # 防御性限幅：融合定位不应超出地图范围（避免 NaN/异常漂移污染误差统计）
            fused_loc.x = max(-10000.0, min(10000.0, fused_loc.x))
            fused_loc.y = max(-10000.0, min(10000.0, fused_loc.y))
            loc_err = fused_loc.distance(gt_loc)

            # 航向融合（互补滤波，与位置同构）：GNSS 观测航向 + INS 航向递推。
            # INS 角速度源用「真值航向差分」（CARLA 的 get_angular_velocity 在此环境返回
            # 不可靠的巨幅值，0 噪声下也会被积分成航向漂移），再叠加陀螺噪声；GNSS 给出带噪观测。
            if prev_gt_yaw_deg is None:
                fused_yaw_deg = gt_yaw  # 首帧：直接以当前真值航向初始化，避免初始 90° 大误差
                prev_gt_yaw_deg = gt_yaw
                gyro_w = 0.0
                ins_yaw_deg = fused_yaw_deg
                gnss_yaw_deg = gt_yaw
                yaw_a = alpha * (0.1 if gps_failure else 1.0)
            else:
                delta_deg = ((gt_yaw - prev_gt_yaw_deg + 180.0) % 360.0) - 180.0  # 本帧航向变化(度)
                prev_gt_yaw_deg = gt_yaw
                gyro_w = delta_deg / 0.05 + random.gauss(0, ins_noise * 30.0)      # °/s
                ins_yaw_deg = fused_yaw_deg + gyro_w * 0.05
                gnss_yaw_deg = gt_yaw + random.gauss(0, gnss_noise * 4.0)          # 度
                yaw_a = alpha * (0.1 if gps_failure else 1.0)
                # 用「卷绕后的相位差(innovation)」做互补滤波：若直接对绝对角度求加权，
                # 在 ±180° 边界处会把 179.9° 与 -180.1° 平均成 -0.1°（实则同向），
                # 导致融合航向瞬间跳变，纯跟踪 cte 剧增、车辆跑偏（含 0 噪声场景）。
                innov_deg = gnss_yaw_deg - ins_yaw_deg
                innov_deg = (innov_deg + 180.0) % 360.0 - 180.0   # 卷绕到 [-180,180]
                fused_yaw_deg = ins_yaw_deg + yaw_a * innov_deg
                fused_yaw_deg = (fused_yaw_deg + 180.0) % 360.0 - 180.0  # 归一到 [-180,180]
            if not math.isfinite(loc_err):
                loc_err = 0.0
            loc_err_history.append(round(loc_err, 3))

            # ── 障碍物统一扫描：全量保留（不预筛选本道），输出 Frenet 障碍列表 ──
            # 筛选交给规划器的碰撞检查——旁道/远端障碍天然进入时空联合检查（修问题5）。
            # 每个障碍含半长/半宽（修"距离只算到障碍中心"的问题2）。
            front_obstacle = float("inf")
            front_obs_src = "none"
            dst_actor2 = 999.0
            obs_list = []
            obstacles = []            # 统一障碍列表：[{s,l,half_len,half_w,v_s,lane,cls,id}]
            bbox_cands = []           # 包围框相机候选：50m 内的车辆/行人 (actor, 距离)
            perceived = []            # 感知闭环：bbox 相机单目感知到的障碍物
            fwd = ego_tf.get_forward_vector()
            left = carla.Vector3D(x=fwd.y, y=-fwd.x, z=0)  # 车体系左向（lat 左正约定；(-fwd.y,fwd.x) 是右向，曾致感知坐标镜像）
            _scan_j0 = max(0, wp_idx - 5)
            _scan_j1 = min(len(route_wp), wp_idx + int(70.0 / max(0.5, sampling_res)) + 10)

            if perception_on:
                # ── 感知闭环：不查询世界真值，障碍物只来自 bbox 相机（实例+语义）──
                # 视野限制即真实限制：FOV ±45°、约 50m 内的目标才能被"看见"
                try:
                    if inst.id in _instance_raw and sem.id in _semantic_raw:
                        ih = int(inst.attributes["image_size_y"])
                        iw = int(inst.attributes["image_size_x"])
                        inst_arr = np.frombuffer(_instance_raw[inst.id], dtype=np.uint8).reshape((ih, iw, 4))
                        sh = int(sem.attributes["image_size_y"])
                        sw = int(sem.attributes["image_size_x"])
                        sem_arr = np.frombuffer(_semantic_raw[sem.id], dtype=np.uint8).reshape((sh, sw, 4))
                        perceived = _exp10_camera_perceive(inst_arr, sem_arr, exclude_ids=(vehicle.id,))
                except Exception as exc:
                    _perceive_diag["err"] += 1
                    if _perceive_diag["err"] <= 3:
                        _exp_log(f"相机感知异常#{_perceive_diag['err']}: {exc!r}")
                for p in perceived:
                    fwd_dist, lat = p["fwd"], p["lat"]
                    # 由感知结果反算世界坐标（基准=融合定位；再用 HD Map 做车道归属），
                    # 与真实系统一致：相机检测目标 → 地图匹配 → 车道级行为决策
                    wpx = fused_loc.x + fwd.x * fwd_dist + left.x * lat
                    wpy = fused_loc.y + fwd.y * fwd_dist + left.y * lat
                    s_o, l_o, _tx, _ty = _frenet(wpx, wpy, _scan_j0, _scan_j1)
                    lane_id = None
                    try:
                        owp = carla_map.get_waypoint(carla.Location(x=wpx, y=wpy, z=0.0),
                                                     project_to_road=True, lane_type=carla.LaneType.Driving)
                        if owp is not None:
                            lane_id = (owp.road_id, owp.lane_id)
                    except Exception:
                        pass
                    obstacles.append({
                        "s": s_o, "l": l_o,
                        "half_len": _PERC_HALF_LEN.get(p["cls"], 2.0),  # 类别典型半长（保守）
                        "half_w": max(0.3, p["width_m"] / 2.0),
                        "v_s": 0.0,   # 单帧感知无速度估计（多帧跟踪是进阶内容）
                        "lane": lane_id, "cls": p["cls"], "id": None,
                    })
                    obs_list.append({
                        "category": _EXP10_PERC_STYLE.get(p["cls"], ("目标",))[0],
                        "dist": round(p["dist"], 1), "size": round(p["width_m"], 1),
                        "x": round(wpx, 1), "y": round(wpy, 1),  # 感知估计的世界坐标（鸟瞰图标记）
                        "vel": None,
                    })
            else:
                actor_list = world.get_actors()
                for actor in actor_list:
                    if actor.id == vehicle.id:
                        continue
                    tid = actor.type_id
                    if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                        continue
                    dist = actor.get_location().distance(fused_loc)
                    if dist > 50:
                        continue
                    dst_actor2 = min(dst_actor2, dist)
                    vel = actor.get_velocity()
                    speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
                    bb = actor.bounding_box.extent
                    aloc = actor.get_location()
                    s_o, l_o, tx_o, ty_o = _frenet(aloc.x, aloc.y, _scan_j0, _scan_j1)
                    lane_id = None
                    try:
                        owp = carla_map.get_waypoint(aloc, project_to_road=True, lane_type=carla.LaneType.Driving)
                        if owp is not None:
                            lane_id = (owp.road_id, owp.lane_id)
                    except Exception:
                        pass
                    obstacles.append({
                        "s": s_o, "l": l_o,
                        "half_len": float(bb.x),   # 真值模式：包围盒半长（extent 为半尺寸）
                        "half_w": float(bb.y),
                        "v_s": vel.x * tx_o + vel.y * ty_o,  # 沿参考线的纵向速度
                        "lane": lane_id, "cls": tid, "id": actor.id,
                    })
                    bbox_cands.append((actor, dist))
                    obs_list.append({
                        "category": _dynamic_class(tid),
                        "dist": round(dist, 1), "size": round(max(bb.x, bb.y) * 2, 1),
                        "x": round(aloc.x, 1), "y": round(aloc.y, 1),
                        "vel": round(speed, 1),
                    })

            # bbox 渲染：扫描之后立即用本 tick 检测结果 + 本 tick 相机帧渲染。位置从原尾部
            # （控制+语义渲染之后）前移到此处（扫描之后、控制之前），避开尾部延迟让 bbox 赶上
            # SSE 采样；用本 tick 数据保证检测框与画面同步（不再跨 tick 快照，避免时序错位丢框）。
            if rgb_raw["raw"] is not None and inst.id in _instance_raw:
                try:
                    _rgb_arr = np.frombuffer(rgb_raw["raw"], dtype=np.uint8).reshape((rgb_raw["h"], rgb_raw["w"], 4))
                    _ih = int(inst.attributes["image_size_y"])
                    _iw = int(inst.attributes["image_size_x"])
                    if perception_on:
                        _sensor_frames[inst.id] = _render_perceived_frame(_rgb_arr, perceived, _iw, _ih)
                    else:
                        _inst_arr = np.frombuffer(_instance_raw[inst.id], dtype=np.uint8).reshape((_ih, _iw, 4))
                        _sensor_frames[inst.id] = _render_bbox_frame(_rgb_arr, _inst_arr, bbox_cands)
                    _sensor_frame_num[inst.id] = _sensor_frame_num.get(inst.id, 0) + 1
                    _bbox_diag["ok"] += 1
                except Exception as exc:
                    _bbox_diag["err"] += 1
                    if _bbox_diag["err"] <= 3:
                        _exp_log(f"bbox渲染异常#{_bbox_diag['err']}: {exc!r}")
            else:
                _bbox_diag["skip"] += 1

            # 车速（先于红绿灯判定与决策规划，供同帧使用）
            vel = vehicle.get_velocity()
            spd = math.sqrt(vel.x ** 2 + vel.y ** 2)

            # ── 红绿灯（停止线绑定 + s 判定，修问题1）──
            # 有效灯 = 绑定到本路线、且停止线仍在前方(0.5~80m)的最近一个；
            # 车越过停止线后 s 差变负，该灯自动退出考虑——路口内/出口不再被
            # 交叉方向的红灯误刹（committed 语义由 s 比较天然实现）。
            tl_state = "green"
            tl_dist = 999.0
            red_stop_s = None
            ego_s, ego_l, ego_tx, ego_ty = _frenet(fused_loc.x, fused_loc.y, _scan_j0, _scan_j1)
            v_long = max(0.0, vel.x * ego_tx + vel.y * ego_ty)   # 纵向车速（沿参考线）
            for b in tl_bindings:
                d = b["s_stop"] - ego_s
                if 0.5 < d < 80.0 and d < tl_dist:
                    try:
                        st = _TL_STATE_MAP.get(b["tl"].state, "green")
                    except Exception:
                        continue
                    tl_dist, tl_state = d, st
                    red_stop_s = (b["s_stop"] - RED_MARGIN) if st == "red" else None
            if tl_state != _tl_state_prev:
                _plan_log(f"EVENT 信号灯: {_tl_state_prev} → {tl_state} "
                          f"@前方{tl_dist:.0f}m (red_stop_s="
                          f"{f'{red_stop_s:.1f}' if red_stop_s is not None else 'None'})")
                _tl_state_prev = tl_state

            # ── 决策 + 规划：Frenet 采样式时空联合（每帧重估，修问题2/3/4/5）──
            # 候选 = 横向{保持,左换,右换} × 纵向{巡航,停驻}，五次多项式横向剖面；
            # 硬约束（碰撞/可行驶域/红灯）逐时刻检查全量障碍（含长度、含旁道、含动态外推）；
            # 代价最小者胜出；全部被拒 → 本道刹停兜底（横向不够纵向补）。
            STOP_MARGIN = max(3.0, safe_dist * 0.5)   # 刹停余量（safe_distance 滑杆联动，默认12→6m）
            throttle = 0.0
            brake = 0.0
            a_need = 0.0
            desired = target_speed

            # 可行驶域 + 车道宽度（无车道信息时退化为「只保持车道」的安全模式）
            bnds = _drivable_bounds(wp_idx) if wp_idx < len(route_wp) else None
            w_lane = 3.5
            try:
                _wp_cur = route_wps[wp_idx] if wp_idx < len(route_wps) else None
                if _wp_cur is not None and _wp_cur.lane_width:
                    w_lane = float(_wp_cur.lane_width)
            except Exception:
                pass
            l_min = l_max = None
            if bnds is not None and bnds[0] is not None:
                l_min, l_max = bnds

            # 邻道候选（clamp 进可行驶域：对向侧边界自动收缩，无可行邻道则该侧消失）。
            # 每侧独立判定：该侧窄/对向 → 跳过该侧但不中断另一侧（原 break 会连
            # 另一侧一起跳过）；目标偏移点再做一次地图级同向校验（双保险防逆行）
            _neighbors = []
            if l_min is not None:
                lo_b = l_min + ego_half_w + COLL_L
                hi_b = l_max - ego_half_w - COLL_L
                for l_t in (w_lane, -w_lane):
                    if lo_b >= hi_b - 0.2:
                        break      # 域过窄（如仅一条车道），只保持
                    l_c = max(lo_b, min(hi_b, l_t))
                    if abs(l_c) < 0.5:
                        continue   # clamp 后已贴回本道，不算可行邻道
                    # 前方 12m 处该横向偏移落点必须是同向 Driving 车道（防对向）
                    if not _lat_lane_ok(l_c, ego_s + 12.0):
                        _plan_log(f"EVENT 邻道 {l_c:+.1f}m 被方向校验否决（对向/交叉车道）")
                        continue
                    _neighbors.append(l_c)

            def _obs_stop(l_t):
                """走廊内最近障碍停驻点（后缘-余量，动态障碍 1s 前瞻）；inf = 走廊无障碍"""
                s_stop = float("inf")
                for o in obstacles:
                    if abs(o["l"] - l_t) < ego_half_w + o["half_w"] + 0.25:
                        rear = o["s"] + max(0.0, o["v_s"]) * 1.0 - o["half_len"]
                        if ego_s + 0.5 < rear < ego_s + DEC_WIN or ego_s + 0.5 < o["s"] + o["half_len"] < ego_s + DEC_WIN:
                            s_stop = min(s_stop, rear - STOP_MARGIN)
                return s_stop

            def _corridor_stop(l_t):
                """走廊内最近停驻点 = min(障碍停驻点, 红灯停止线)"""
                s_obs = _obs_stop(l_t)
                if red_stop_s is not None:
                    return red_stop_s if s_obs == float("inf") else min(red_stop_s, s_obs)
                return s_obs

            # ── 行为决策（FSM 意图，修问题3/5 的"逐个处理"与"不查旁道"）──
            # 本道走廊被占 → 选「走廊无障碍且在可行驶域内」的邻道换道（先左后右，
            # 超车靠左惯例）；两侧皆不可行 → 保持+跟停（安全兜底，不冒险切道）。
            # 换道途中每帧重估：邻道新出现障碍/本道清空都会即时改变意图。
            _blocked = _obs_stop(0.0) < float("inf")
            intent_l = 0.0
            if _blocked:
                for l_t in _neighbors:
                    if _obs_stop(l_t) == float("inf"):
                        intent_l = l_t
                        break
            lat_targets = [0.0] if intent_l == 0.0 else [intent_l, 0.0]

            # 候选生成与展开
            cands = []
            _rej_bounds = _rej_red = _rej_coll = _rej_dir = 0   # 拒绝统计（日志用）
            _n_steps = int(T_HORIZON / DT_PLAN)
            T_lat = max(2.0, min(4.0, 1.2 * max(2.0, v_long)))
            for l_t in lat_targets:
                s_stop = _corridor_stop(l_t)
                for mode in ("CRUISE", "STOP"):
                    if mode == "STOP" and s_stop == float("inf"):
                        continue   # 无停驻点则无需 STOP 候选
                    if mode == "CRUISE":
                        a_long = max(-2.0, min(1.5, (target_speed - v_long) / 2.0))
                    else:
                        d = s_stop - ego_s
                        a_long = 0.0 if d <= 0.5 else max(-MAX_DECEL, -(v_long * v_long) / (2.0 * d))
                    # 逐时刻展开：横向五次多项式（最小急动度）+ 纵向逐步积分。
                    # CRUISE 按停驻点（红灯/障碍）生成舒适制动剖面
                    # v ≤ √(2·COMFORT_A·(s_stop−s))，到停止线恰好停住。
                    # 顺序要点（修期望速度横跳）：① 物理减速度下限先施加；
                    # ② 剖面钳制最后施加且允许超过舒适值——若剖面放在下限之前，
                    # 贴线归零会被下限顶回 v>0.3，整条 CRUISE 被红灯硬约束拒掉，
                    # 与 STOP 候选逐帧轮替胜出 → desired 在最大/最小间跳变。
                    samples = []
                    ok = True
                    dl = l_t - ego_l
                    s_prev, v_prev = ego_s, v_long
                    for k in range(1, _n_steps + 1):
                        tk = k * DT_PLAN
                        tau = min(1.0, tk / T_lat)
                        l_k = ego_l + dl * tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)
                        v_k = max(0.0, v_prev + a_long * DT_PLAN)
                        v_k = max(v_k, v_prev - MAX_DECEL * DT_PLAN)   # 物理极限内
                        if mode == "CRUISE":
                            if a_long > 0.0:
                                v_k = min(v_k, target_speed)
                            if s_stop < float("inf"):
                                v_k = min(v_k, math.sqrt(
                                    2.0 * COMFORT_A * max(0.0, s_stop - s_prev)))
                                # 贴线归零：本步内将抵达停止线就停（判据=剩余距离
                                # 小于本步行程，而非固定 0.2m——低速步长 0.5m 会被
                                # 漏判带速越线触发整条拒绝）
                                if s_stop - s_prev <= max(0.25, v_prev * DT_PLAN):
                                    v_k = 0.0
                        s_k = s_prev + 0.5 * (v_prev + v_k) * DT_PLAN
                        s_prev, v_prev = s_k, v_k
                        # 硬约束①：可行驶域（越界即逆行/出路缘，整条拒绝）。
                        # 逐点取「当地」边界——轨迹展开 30m+，前方路段可能收窄/
                        # 变两车道，只用车头处边界会放行前方的对向车道（蓝线逆行）
                        if l_min is not None:
                            b_k = _bounds_at_s(s_k)
                            if b_k is None:
                                b_k = (l_min, l_max)
                            if (l_k < min(b_k[0] + ego_half_w + COLL_L, ego_l - 0.05)
                                    or l_k > max(b_k[1] - ego_half_w - COLL_L, ego_l + 0.05)):
                                _rej_bounds += 1
                                ok = False
                                break
                        # 硬约束②：红灯（CRUISE 不得带速越过停止线）
                        if (mode == "CRUISE" and red_stop_s is not None
                                and s_k > red_stop_s and v_k > 0.3):
                            _rej_red += 1
                            ok = False
                            break
                        # 硬约束③：碰撞——对全量障碍（含长度、含动态外推、含旁道）
                        for o in obstacles:
                            if o["s"] < ego_s - 1.0 and o["v_s"] > v_long:
                                continue   # 后方更快的超车车辆：后车责任，不因此误刹
                            s_o = o["s"] + o["v_s"] * tk
                            if (abs(s_k - s_o) < o["half_len"] + ego_half_len + COLL_S
                                    and abs(l_k - o["l"]) < o["half_w"] + ego_half_w + COLL_L):
                                _rej_coll += 1
                                ok = False
                                break
                        if not ok:
                            break
                        samples.append((tk, s_k, l_k, v_k))
                    if not ok or not samples:
                        continue
                    # 换道候选：中段与末端落点必须是同向 Driving 车道。
                    # 边界缓存按「邻接链+点积」推断，路口/车道斜接处可能漏判对向
                    # （如左换道落在交叉来车道上）——地图级查询兜底，逆行零容忍
                    if abs(dl) > 0.5:
                        if not (_lat_lane_ok(l_t, samples[len(samples) // 2][1])
                                and _lat_lane_ok(l_t, samples[-1][1])):
                            _rej_dir += 1
                            continue
                    # 代价：意图对齐（决策层意图优先，非意图轨迹仅作兜底）+
                    # 偏离参考线 + 横摆平顺 + 舒适性 + 效率 + 换道机动惩罚
                    l_arr = [ego_l] + [sm[2] for sm in samples]
                    j_lat = sum(x * x for x in l_arr) / len(l_arr)
                    j_dl = sum(((l_arr[i + 1] - l_arr[i]) / DT_PLAN) ** 2
                               for i in range(len(l_arr) - 1)) / max(1, len(l_arr) - 1)
                    v_end = samples[-1][3]
                    cost = (0.5 * j_lat + 0.3 * j_dl + 0.2 * a_long * a_long
                            + 0.4 * max(0.0, target_speed - v_end)
                            + (0.6 if abs(l_t) > 0.5 else 0.0)
                            + (0.0 if l_t == intent_l else 5.0))
                    cands.append({"cost": cost, "l_t": l_t, "mode": mode,
                                  "s_stop": s_stop, "a_long": a_long, "samples": samples})

            best = min(cands, key=lambda c: c["cost"]) if cands else None
            if best is not None:
                # 由最优候选导出控制量与可视化状态
                if best["mode"] == "STOP":
                    # 逼近速度 = 最高速一半（远端），贴近停驻点按舒适制动剖面
                    # 平滑收敛到 0（desired=√(2·a·d)），停在线前 0.5m
                    d = max(0.1, best["s_stop"] - ego_s)
                    desired = min(target_speed * 0.5,
                                  math.sqrt(2.0 * COMFORT_A * d))
                    if v_long > desired + 0.3 and d > 0.5:
                        a_need = min(MAX_DECEL,
                                     (v_long * v_long - desired * desired) / (2.0 * d))
                    if v_long < 0.5 and d < 1.0:
                        a_need = max(a_need, 1.5)   # 已到停驻点：保持制动防蠕行
                else:
                    desired = target_speed
                    if best["s_stop"] < float("inf"):
                        d = max(0.0, best["s_stop"] - ego_s)
                        desired = min(desired, math.sqrt(2.0 * COMFORT_A * d))
                        # 车速高于剖面期望 → 按剖面所需减速度主动制动
                        # （速度 P 控制的 0.25 油门基线自身减不了速）
                        if v_long > desired + 0.3 and d > 0.5:
                            a_need = min(MAX_DECEL,
                                         (v_long * v_long - desired * desired) / (2.0 * d))
                        if v_long < 0.5 and d < 2.0:
                            a_need = max(a_need, 1.5)   # 近停止线保持制动防蠕行
                plan_traj = [_world(sm[1], sm[2]) for sm in best["samples"]]
                plan_end_l = best["samples"][-1][2] if best["samples"] else ego_l
                avoid_lat_target = best["l_t"]
                avoiding = abs(best["l_t"]) > 0.5
                avoid_side = 1 if best["l_t"] > 0.5 else (-1 if best["l_t"] < -0.5 else 0)
            else:
                # 兜底：无可行轨迹（本道与旁道皆被占/过近）→ 当前走廊内刹停
                s_stop = _corridor_stop(ego_l)
                d = max(0.1, s_stop - ego_s) if s_stop < float("inf") else 0.1
                a_need = min(MAX_DECEL, v_long * v_long / (2.0 * d))
                if v_long < 0.5:
                    a_need = 1.5   # 已停：保持制动防蠕行
                desired = 0.0
                plan_traj = [_world(ego_s + 2.0 * i, ego_l) for i in range(1, 16)]
                plan_end_l = ego_l
                avoiding = False
                avoid_side = 0
                avoid_lat_target = ego_l
                _plan_log(f"EVENT 无可行候选 → 兜底刹停 (rej: 域{_rej_bounds}/"
                          f"红{_rej_red}/碰{_rej_coll}/向{_rej_dir})")

            # 逐帧决策日志（排障主数据：一帧一行，状态+决策+控制全链路）
            _bk = (best["mode"], round(best["l_t"], 1)) if best is not None else ("FALLBACK", None)
            if _bk != _best_prev_key:
                _plan_log(f"EVENT 最优解切换: {_best_prev_key} → {_bk}")
                _best_prev_key = _bk
            _obs_str = ",".join(f"{o['s']:.0f}/{o['l']:+.1f}" for o in obstacles[:3])
            _cost_str = f"/c={best['cost']:.2f}" if best is not None else ""
            _plan_log(
                f"FRM t={t:6.1f} v={spd:5.2f}(lon {v_long:5.2f}) s={ego_s:7.1f} l={ego_l:+5.2f} "
                f"tl={tl_state[:3]}/{tl_dist:5.1f}m obs={len(obstacles)}[{_obs_str}] "
                f"blk={'Y' if _blocked else 'N'} itn={intent_l:+5.2f} nb={[round(x, 1) for x in _neighbors]} "
                f"cands={len(cands)} best={_bk[0]}{_cost_str} "
                f"des={desired:5.2f} a={a_need:5.2f} "
                f"rej:域{_rej_bounds}/红{_rej_red}/碰{_rej_coll}/向{_rej_dir}")

            # SSE 兼容：换道目标车道（鸟瞰图高亮）。
            # 高亮前做同向校验——邻接链查询可能拿到对向/交叉车道，高亮到逆行
            # 车道上会误导观测（规划层已有 _lat_lane_ok 双保险，此处管展示）
            avoid_dest_lane = None
            if avoiding:
                try:
                    _wp_cur = route_wps[wp_idx] if wp_idx < len(route_wps) else None
                    if _wp_cur is not None:
                        nb = _wp_cur.get_left_lane() if avoid_side > 0 else _wp_cur.get_right_lane()
                        if nb is not None and nb.lane_type == carla.LaneType.Driving:
                            _cf = _wp_cur.transform.get_forward_vector()
                            _nf = nb.transform.get_forward_vector()
                            if _nf.x * _cf.x + _nf.y * _cf.y > 0.3:
                                avoid_dest_lane = (nb.road_id, nb.lane_id)
                except Exception:
                    pass

            # 行为状态标签（教学/日志/可视化用；决策本身每帧重估无状态依赖）
            _blocked_now = any(
                abs(o["l"] - ego_l) < ego_half_w + o["half_w"] + 0.25
                and ego_s < o["s"] + o["half_len"] < ego_s + DEC_WIN
                for o in obstacles)
            if avoiding:
                _fsm_new = "LANE_CHANGE"
            elif red_stop_s is not None:
                _fsm_new = "APPROACH_RED"
            elif _blocked_now:
                _fsm_new = "FOLLOW"
            else:
                _fsm_new = "CRUISE"
            # 状态/换道转换日志（教学观测用；CRUISE 回归不刷屏）
            if avoiding and not _avoid_prev:
                _exp_log(f"{'左' if avoid_side > 0 else '右'}侧邻道可行，换道绕行（目标偏移 {abs(avoid_lat_target):.1f}m）")
            elif not avoiding and _avoid_prev:
                _exp_log("绕行完成，回正车道")
            _avoid_prev = avoiding
            if _fsm_new != fsm_state:
                if _fsm_new == "APPROACH_RED":
                    _exp_log(f"前方 {tl_dist:.0f}m 红灯，减速停车（停止线判定）")
                elif _fsm_new == "FOLLOW":
                    _exp_log("本车道被占且邻道不可行，跟停等待")
                fsm_state = _fsm_new

            # 展示用：本道走廊内最近障碍（到后缘的距离，含半长——修"量到中心"缺陷）
            front_obstacle = float("inf")
            for o in obstacles:
                if (abs(o["l"] - ego_l) < ego_half_w + o["half_w"] + 0.25
                        and o["s"] - o["half_len"] > ego_s):
                    d_rear = o["s"] - o["half_len"] - ego_s
                    if d_rear < front_obstacle:
                        front_obstacle = d_rear
                        front_obs_src = o["cls"]

            # 黄灯：软约束（红灯/障碍已由规划层硬约束处理，此处不叠加）
            YELLOW_D = 1.0
            if tl_state == "yellow" and a_need < YELLOW_D:
                a_need = YELLOW_D
                desired = min(desired, target_speed * 0.5)

            # 施加制动：低于阈值视为无需主动刹车（正常巡航/跟车），否则按所需减速度占比输出
            COMFORT_D = 0.8
            if a_need > COMFORT_D:
                brake = min(1.0, max(0.0, (a_need / MAX_DECEL) * brake_force))
            else:
                err = desired - spd
                throttle = max(0.0, min(1.0, 0.25 + 0.12 * err))
            speed_history.append(round(spd, 2))

            # 横向控制 (Pure Pursuit) —— 前视点改取自「规划轨迹」：
            # 1) 先在路线点列上校正 wp_idx（进度推进/障碍窗口/可视化共用）；
            # 2) 前视点 = 规划轨迹上距自车 ≥ lookahead 的第一个点（含换道 S 弯，
            #    替代原「路线点+法向平移」的两段式做法——跟踪的就是规划器输出本身）；
            # 3) 规划轨迹不够远（低速/临近停车）时回退到路线点。
            if route_wp:
                loc = fused_loc   # 用带噪声的融合定位驱动控制；真值仅用于误差评估
                yaw_rad = math.radians(fused_yaw_deg)  # 用融合航向（GNSS+INS 递推），不再用真值航向
                # 1) 在 wp_idx 前 50 点内找最近路点，校正 wp_idx（防止车辆越过路点后落后）
                best = wp_idx
                best_d = float("inf")
                for j in range(wp_idx, min(wp_idx + 50, len(route_wp))):
                    d = math.hypot(loc.x - route_wp[j].x, loc.y - route_wp[j].y)
                    if d < best_d:
                        best_d = d
                        best = j
                wp_idx = best
                # 2) 前视点：优先规划轨迹，回退路线点列
                look_x, look_y = None, None
                for px, py in plan_traj:
                    if math.hypot(px - loc.x, py - loc.y) >= lookahead:
                        look_x, look_y = px, py
                        break
                if look_x is None:
                    for j in range(wp_idx, len(route_wp)):
                        if math.hypot(loc.x - route_wp[j].x, loc.y - route_wp[j].y) >= lookahead:
                            look_x, look_y = route_wp[j].x, route_wp[j].y
                            break
                    else:
                        look_x, look_y = route_wp[-1].x, route_wp[-1].y
                dx = look_x - loc.x
                dy = look_y - loc.y
                # 3) 转向角（标准 Pure Pursuit）。
                # steer_angle/1.22 把前轮角(最大约70°=1.22rad)映射到 [-1,1] 已是合理幅度；
                # kp_steer 为教学灵敏度：前端默认 1.4，除以 1.4 使默认时=标准幅度，
                # 调大更激进、调小更柔和，避免直接乘 kp 导致转度过大冲出路面。
                hdng_alpha = math.atan2(dy, dx) - yaw_rad
                wheelbase = 2.85
                steer_angle = math.atan2(2.0 * wheelbase * math.sin(hdng_alpha), lookahead)
                raw_steer = max(-1.0, min(1.0, (kp_steer / 1.4) * steer_angle / 1.22))
                cte = dx * math.sin(yaw_rad) - dy * math.cos(yaw_rad)
                look_dist = math.hypot(dx, dy)
            else:
                dx = dy = 0.0
                raw_steer = 0.0
                cte = 0.0
                look_dist = 0.0
                hdng_alpha = 0.0
            steer_history.append(raw_steer)
            delay_steps = max(0, int(steer_delay / 0.05))
            steer = steer_history[max(0, len(steer_history) - 1 - delay_steps)] if steer_history else 0
            cte_history.append(round(cte, 3))

            # 应用控制
            ctrl = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
            )
            vehicle.apply_control(ctrl)

            # ── 诊断日志（每 10 帧≈0.5s 输出一次，便于快速定位转向/控制问题）──
            _dbg_frame += 1
            if _dbg_frame % 10 == 0:
                _exp_log(
                    f"[t={t:.1f}s] spd={spd:.2f} des={desired:.1f} thr={throttle:.2f} brk={brake:.2f} "
                    f"out_steer={steer:.2f} raw_steer={raw_steer:.2f} alpha={hdng_alpha:+.3f} "
                    f"look={look_dist:.1f} cte={cte:.2f} err={loc_err:.2f} "
                    f"yaw(gt)={gt_yaw:.1f} yaw(F)={fused_yaw_deg:.1f} gyaw={gnss_yaw_deg:.1f} "
                    f"gyro={gyro_w:.1f}°/s yaw_a={yaw_a:.3f} "
                    f"fused=({fused_loc.x:.1f},{fused_loc.y:.1f}) p={ego_tf.rotation.pitch:.0f} r={ego_tf.rotation.roll:.0f} "
                    f"obs={front_obstacle if front_obstacle != float('inf') else 'inf'}({front_obs_src}) "
                    f"tl={tl_state}/{tl_dist:.0f}m wp={wp_idx}/{len(route_wp)} "
                    f"fsm={fsm_state} avd={'L' if avoid_side > 0 else 'R' if avoid_side < 0 else '-'} "
                    f"s={ego_s:.0f} l={ego_l:+.1f} "
                    f"loc=({gt_loc.x:.1f},{gt_loc.y:.1f})"
                )

            # 到达判断
            dist_to_end = gt_loc.distance(end_loc)
            arrived = dist_to_end < 5.0
            if arrived:
                _exp_log("到达终点")
                break

            # 语义分割帧：把原始 CityScapes 标签转成彩色图写入 _sensor_frames，
            # 供 SSE 推流（前端可在相机视角下拉切到语义画面；未连接/无帧时前端回退 Mock）
            if sem.id in _semantic_raw:
                try:
                    sem_h = int(sem.attributes["image_size_y"])
                    sem_w = int(sem.attributes["image_size_x"])
                    sem_arr = np.frombuffer(_semantic_raw[sem.id], dtype=np.uint8).reshape((sem_h, sem_w, 4))
                    sem_labels = sem_arr[:, :, 2].astype(np.int32)
                    sem_rgb = _colors_from_labels(_label_semantic_level(sem_labels, "L2"), "L2")
                    sem_buf = io.BytesIO()
                    PIL.Image.fromarray(sem_rgb, mode="RGB").save(sem_buf, format="JPEG", quality=85)
                    _sensor_frames[sem.id] = sem_buf.getvalue()
                    _sensor_frame_num[sem.id] = _sensor_frame_num.get(sem.id, 0) + 1
                except Exception:
                    pass

            # ── 鸟瞰可视化数据：当前/目标车道、参考线（含换道 S 弯）、预测轨迹 ──
            cur_lane = None
            try:
                cwp = carla_map.get_waypoint(fused_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if cwp is not None:
                    cur_lane = [cwp.road_id, cwp.lane_id]
            except Exception:
                pass
            tgt_lane = list(avoid_dest_lane) if (avoiding and avoid_dest_lane is not None) else cur_lane

            # 前方车道计划：沿路线 60m 依次将进入的车道（有序去重）+ 各段进入距离，
            # 供鸟瞰图「将要走的车道」渐变高亮（如前方转弯/换道路）
            plan_lanes = []
            _seen_lanes = set()
            for j in range(wp_idx, min(len(route_wp), wp_idx + int(60.0 / max(0.5, sampling_res)) + 10)):
                lk = route_lane_ids[j] if j < len(route_lane_ids) else None
                if lk is None or lk in _seen_lanes:
                    continue
                _seen_lanes.add(lk)
                d = math.hypot(route_wp[j].x - fused_loc.x, route_wp[j].y - fused_loc.y)
                plan_lanes.append({"lane": [lk[0], lk[1]], "dist": round(d, 1)})

            # 参考线 = 规划轨迹（含换道 S 弯）+ 超出轨迹末端的路线点续接。
            # 续接段必须从「轨迹末端距离」接着往后取，而非从车旁重新取起，
            # 否则绘制顺序变成：先画到前方目标车道、再跳回车旁从本道重画一遍。
            ref_path = []
            d_covered = 0.0
            for px, py in plan_traj:
                _d = math.hypot(px - fused_loc.x, py - fused_loc.y)
                if _d <= 60.0:
                    d_covered = max(d_covered, _d)
                    ref_path.append({"x": round(px, 1), "y": round(py, 1)})
            if route_wp and len(ref_path) < 30:
                step_j = max(1, int(round(2.0 / max(0.5, sampling_res))))
                for j in range(wp_idx, len(route_wp), step_j):
                    px, py = route_wp[j].x, route_wp[j].y
                    d = math.hypot(px - fused_loc.x, py - fused_loc.y)
                    if d <= d_covered + 1.0:
                        continue   # 规划轨迹已覆盖的近段不重复，从轨迹末端续接
                    if len(ref_path) >= 5 and d > 60.0:
                        break
                    # 横向偏移从「轨迹末端偏移」平滑过渡到「换道目标偏移」，
                    # 与 S 弯末端无缝衔接（不再从本道 0 偏移重新爬坡）
                    a = route_wp[max(0, j - 1)]
                    b = route_wp[min(len(route_wp) - 1, j + 1)]
                    tx, ty = b.x - a.x, b.y - a.y
                    tl = math.hypot(tx, ty)
                    if tl > 1e-6:
                        blend = min(1.0, max(0.0, (d - d_covered) / 10.0))
                        off = plan_end_l + (avoid_lat_target - plan_end_l) * blend
                        px -= (ty / tl) * off
                        py += (tx / tl) * off
                    ref_path.append({"x": round(px, 1), "y": round(py, 1)})

            # 预测轨迹：自行车模型前推 3s（当前车速 + 实际输出前轮角），
            # 展示「保持当前操作车辆将驶向哪里」；与参考线的偏差即跟踪误差
            pred_path = []
            if spd > 0.2:
                _px, _py = fused_loc.x, fused_loc.y
                _hdg = math.radians(fused_yaw_deg)
                _delta = max(-1.0, min(1.0, steer)) * 1.22   # [-1,1] → 前轮角（1.22rad≈满舵）
                for _ in range(30):
                    _px += spd * math.cos(_hdg) * 0.1
                    _py += spd * math.sin(_hdg) * 0.1
                    _hdg += spd * math.tan(_delta) / 2.85 * 0.1
                    pred_path.append({"x": round(_px, 1), "y": round(_py, 1)})
                    if math.hypot(_px - fused_loc.x, _py - fused_loc.y) > 60.0:
                        break

            # 推送实验数据
            _push_to_sse({
                "experiment": {
                    "id": 10,
                    "status": "running",
                    "t": round(t, 2),
                    "progress": round(min(100, wp_idx / max(1.0, len(route_wp)) * 100.0), 1),
                    "speed": round(spd, 2),
                    "desired_speed": round(desired, 2),
                    "steer": round(steer, 3),
                    "throttle": round(throttle, 3),
                    "brake": round(brake, 3),
                    "cte": round(cte, 3),
                    "loc_err": round(loc_err, 3),
                    "front_obstacle": round(front_obstacle, 1) if front_obstacle != float("inf") else None,
                    "arrived": arrived,
                    "perception": perception_on,
                    "perceived_count": len(perceived),
                    "gt": {"x": round(gt_loc.x, 2), "y": round(gt_loc.y, 2)},
                    "fused": {"x": round(fused_loc.x, 2), "y": round(fused_loc.y, 2)},
                    "gnss": {"x": round(ngx, 2), "y": round(ngy, 2)},
                    "heading": round(fused_yaw_deg, 1),
                    "gt_yaw": round(gt_yaw, 1),
                    "lane": {"cur": cur_lane, "tgt": tgt_lane, "plan": plan_lanes},
                    "ref_path": ref_path,
                    "pred_path": pred_path,
                    "obstacles": obs_list[:5],
                    "planned_obstacles": _EXP10_PLANNED_OBSTACLES,
                    "traffic_light": {"state": tl_state, "distance": round(tl_dist, 1)},
                    "avoid": {"active": avoiding, "side": avoid_side, "offset": round(avoid_lat_target, 2)},
                    "fsm": fsm_state,
                }
            })

            # 同步模式下 tick 已按固定时间步推进并阻塞至该帧完成，无需额外 sleep
            # time.sleep(0.05)  # 20fps

        # 结束：汇总评分报告（到达 / 里程 / 定位与横向误差 / 碰撞事故）
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        avg_loc_err = sum(loc_err_history) / max(1, len(loc_err_history))
        max_loc_err = max(loc_err_history) if loc_err_history else 0.0
        abs_cte = [abs(c) for c in cte_history]
        avg_abs_cte = (sum(abs_cte) / len(abs_cte)) if abs_cte else 0.0
        max_abs_cte = max(abs_cte) if abs_cte else 0.0
        with col_lock:
            incidents = [dict(i) for i in col_incidents]
        col_by_type = {}
        for i in incidents:
            col_by_type[i["cls"]] = col_by_type.get(i["cls"], 0) + 1
        elapsed = time.time() - t0
        report = {
            "arrived": arrived,
            "duration": round(elapsed, 1),
            "distance_m": round(traveled, 1),
            "avg_loc_err": round(avg_loc_err, 3),
            "max_loc_err": round(max_loc_err, 3),
            "avg_abs_cte": round(avg_abs_cte, 3),
            "max_abs_cte": round(max_abs_cte, 3),
            "collisions": {
                "count": len(incidents),
                "by_type": col_by_type,
                "max_impulse": round(max((i["impulse"] for i in incidents), default=0.0), 1),
            },
            "perception_mode": perception_used,
        }
        _exp_log(
            f"实验10结束 · 到达={arrived} · 用时={elapsed:.0f}s · 里程={traveled:.0f}m · "
            f"平均定位误差={avg_loc_err:.2f}m · 平均|CTE|={avg_abs_cte:.2f}m · "
            f"碰撞事故={len(incidents)}次 · 感知闭环={'开' if perception_used else '关'}"
        )
        if incidents:
            _exp_log(f"碰撞明细: {col_by_type} · 最大冲击={report['collisions']['max_impulse']}")
        _push_to_sse({"experiment": {"id": 10, "status": "done", "arrived": arrived,
                                     "avg_loc_err": round(avg_loc_err, 3), "report": report}})

    except Exception as exc:
        _exp_log(f"实验10 异常: {exc}")
        if _plan_log_f is not None:
            try:
                _plan_log(f"EVENT 实验异常: {exc!r}")
            except Exception:
                pass
        _push_to_sse({"experiment": {"id": 10, "status": "error", "message": str(exc)}})
    finally:
        # 规划调试日志收尾：落结束标记后关闭句柄（buffering=1 已逐行落盘）
        if _plan_log_f is not None:
            try:
                _plan_log("=== 实验10 规划调试日志 end ===")
                _plan_log_f.close()
            except Exception:
                pass
            _plan_log_f = None
        # 清理传感器：先停止监听（断开流），再销毁，避免 socket 报错/scope 警告
        _stream_bird = None
        _stream_camera = None
        _stream_semantic = None
        _stream_bbox = None
        for sid in created_sids:
            try:
                a = world.get_actor(sid)
                if a and a.is_alive:
                    if hasattr(a, "stop") and a.is_listening:
                        a.stop()
                    if not a.destroy():
                        _exp_log(f"传感器销毁失败 sid={sid}")
                _sensor_refs.pop(sid, None)
                # 必须同步从 _managed_actors 移除，否则下次运行 _sweep_stale_actors
                # 会对已销毁 actor 重复 destroy，触发 CARLA libcarla 原生 Abort
                with _lock:
                    _managed_actors.discard(sid)
            except Exception as exc:
                _exp_log(f"传感器清理异常 sid={sid}: {exc!r}")
        # 清理行人
        for obj in _EXP10_PEDI:
            try:
                if obj and obj.is_alive:
                    obj.destroy()
                with _lock:
                    _managed_actors.discard(obj.id)
            except Exception:
                pass
        _EXP10_PEDI.clear()
        # 清理车辆
        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
                with _lock:
                    _managed_actors.discard(vehicle.id)
            except Exception:
                pass
        # 恢复世界运行模式（同步→原异步），避免残留同步模式导致其他实验卡住
        try:
            if _exp10_old_settings is not None:
                world.apply_settings(_exp10_old_settings)
                client.get_trafficmanager(8000).set_synchronous_mode(_exp10_old_settings.synchronous_mode)
        except Exception:
            pass
        with _EXP10_LOCK:
            _EXP10_RUNNING = False
            if _EXP10_ABORT:
                _push_to_sse({"experiment": {"id": 10, "status": "stopped"}})
        _EXP10_VEHICLE_ID = None
        _EXP10_ROUTE = []
        with _EXP10_SPAWN_LOCK:
            _sreq_pending = _EXP10_SPAWN_REQ
            _EXP10_SPAWN_REQ = None
        if _sreq_pending is not None and not _sreq_pending["done"].is_set():
            _sreq_pending["result"] = {"ok": False, "message": "实验已结束，无法生成"}
            _sreq_pending["done"].set()
        _EXP_CURRENT_ID = None


def _exp10_spawn_obstacle(world, ego, route, distance):
    """在 ego 沿路线前方 distance 米处生成静止障碍车辆。
    必须在 exp10 主循环线程内调用（CARLA 客户端非线程安全，跨线程并发会崩溃）。"""
    bps = world.get_blueprint_library().filter("vehicle.*")
    bp = np.random.choice(bps)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "obstacle")
    if bp.has_attribute("color"):
        bp.set_attribute("color", np.random.choice(bp.get_attribute("color").recommended_values))

    # ---- 净距语义：distance 表示"自车与障碍物外侧包围盒之间"的净距(edge-to-edge)。----
    # 不能直接把障碍中心放到"自车中心前方 distance"处——那只是中心距，两车仍会紧贴/重叠。
    # 因此障碍中心需再前进一段"两车半长之和"，使两盒之间的净距恰为 distance。
    ego_half = ego.bounding_box.extent.x          # 自车半长（x 为本地前进轴）
    if bp.has_attribute("length"):
        obs_half = 0.5 * float(bp.get_attribute("length").as_float())  # 障碍半长
    else:
        obs_half = ego_half
    center_offset = distance + ego_half + obs_half  # 障碍中心沿路距自车中心的距离

    ego_loc = ego.get_location()
    if len(route) >= 2:
        # ---- 沿导航路线推进：先计算路线各路点的累计弧长，再把自车位置投影到路线上
        #      得到起算弧长 s0，最后沿路线向前数 center_offset 米定位障碍点。
        cum_len = [0.0] * len(route)
        for j in range(1, len(route)):
            cum_len[j] = cum_len[j - 1] + route[j - 1].distance(route[j])

        # 将自车位置投影到路线折线上（取侧向距离最小的投影点），得到起算弧长 s0
        s0, best_d = 0.0, None
        for p in range(len(route) - 1):
            A, B = route[p], route[p + 1]
            abx, aby = B.x - A.x, B.y - A.y
            seg2 = abx * abx + aby * aby
            if seg2 <= 0.0:
                continue
            t = ((ego_loc.x - A.x) * abx + (ego_loc.y - A.y) * aby) / seg2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            px, py = A.x + abx * t, A.y + aby * t
            d = math.hypot(ego_loc.x - px, ego_loc.y - py)
            if best_d is None or d < best_d:
                best_d = d
                s0 = cum_len[p] + (cum_len[p + 1] - cum_len[p]) * t

        # 目标弧长 = 自车投影弧长 + center_offset，并按路界截断
        target = min(s0 + center_offset, cum_len[-1])

        # 定位 target 所在路段并在其内线性插值
        p = 0
        while p < len(route) - 2 and cum_len[p + 1] < target:
            p += 1
        seg = cum_len[p + 1] - cum_len[p]
        t = 0.0 if seg <= 0.0 else (target - cum_len[p]) / seg
        x = route[p].x + (route[p + 1].x - route[p].x) * t
        y = route[p].y + (route[p + 1].y - route[p].y) * t
        yaw = math.degrees(math.atan2(route[p + 1].y - route[p].y, route[p + 1].x - route[p].x))
        loc = carla.Location(x=x, y=y, z=route[p].z)
    else:
        # 无路线时退回沿车头方向直线放置
        tf = ego.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        loc = carla.Location(x=ego_loc.x + fx * center_offset, y=ego_loc.y + fy * center_offset, z=ego_loc.z)
        yaw = tf.rotation.yaw

    obs = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=yaw)))
    if obs is None:
        return {"ok": False, "message": f"路线前方 {distance:.0f}m 处无法放置（可能与 ego 或其它车辆重叠，可调大距离）"}
    obs.set_autopilot(False)
    obs.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    with _lock:
        _managed_actors.add(obs.id)
    return {
        "ok": True, "id": obs.id, "type": obs.type_id, "distance": round(distance, 1),
        "on_route": len(route) >= 2,
        "location": {"x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2)},
    }


def _exp10_spawn_route_obstacles(world, positions):
    """在指定世界坐标生成静态障碍车（每处一辆）。必须在 exp10 主循环线程内调用
    （CARLA 客户端非线程安全，跨线程并发会崩溃）。返回生成结果列表。
    生成点与静态物碰撞导致 try_spawn 失败时，沿路线方向前后微调重试，
    避免路线障碍物「生成的不够」。"""
    result = []
    bps = list(world.get_blueprint_library().filter("vehicle.*"))
    for obc in positions:
        if not bps:
            break
        yaw = math.radians(obc.get("yaw", 0))
        z = obc.get("z", 0)
        # 原位 → 沿路线前后 ±1~4m 依次重试（生成点蹭到护栏/路缘时微调即可成功）
        offsets = (0.0, 1.5, -1.5, 3.0, -3.0)
        obs = None
        used = None
        for ds in offsets:
            loc = carla.Location(
                x=obc["x"] + math.cos(yaw) * ds,
                y=obc["y"] + math.sin(yaw) * ds,
                z=z,
            )
            bp = random.choice(bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "obstacle")
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            obs = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=obc.get("yaw", 0))))
            if obs is not None:
                used = ds
                break
        if obs is None:
            _exp_log(f"障碍物生成失败（重试 {len(offsets)} 次仍碰撞）: ({obc['x']:.1f}, {obc['y']:.1f})")
            result.append({"ok": False, "x": obc["x"], "y": obc["y"]})
            continue
        obs.set_autopilot(False)
        obs.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        with _lock:
            _managed_actors.add(obs.id)
        if used:
            _exp_log(f"障碍物原位生成受阻，已沿路线偏移 {used:+.1f}m 生成")
        result.append({"ok": True, "id": obs.id, "type": obs.type_id,
                       "x": obc["x"], "y": obc["y"]})
    return result


def _clear_exp10_obstacles(world):
    """清除世界中所有遗留的车辆/行人（含上次实验或上次进程残留，服务/前端重启后仍有效），
    但保留本车 ego。直接按蓝图全量扫描，不再依赖进程内记忆的 actor id。
    必须在 exp10 主循环线程内调用（CARLA 客户端非线程安全）。"""
    destroyed = []

    def _kill(a):
        try:
            if a and a.is_alive:
                a.destroy()
                _managed_actors.discard(a.id)
                destroyed.append(a.type_id)
        except Exception:
            pass

    ego_id = _EXP10_VEHICLE_ID
    # 车辆：剔除本车 ego
    for a in world.get_actors().filter("vehicle.*"):
        if ego_id is not None and a.id == ego_id:
            continue
        _kill(a)
    # 行人 AI 控制器 + 行人
    for a in world.get_actors().filter("controller.ai.walker"):
        _kill(a)
    for a in world.get_actors().filter("walker.pedestrian.*"):
        _kill(a)

    if destroyed:
        _exp_log(f"已清理世界中 {len(destroyed)} 个遗留车辆/行人")


@app.route("/experiment/10/spawn_obstacle", methods=["POST"])
def experiment_10_spawn_obstacle():
    """实验10：在前方生成障碍物。仅入队，实际生成由 exp10 主循环线程执行，
    避免 Flask 线程与 world.tick() 跨线程并发调用 CARLA 导致崩溃。"""
    global _EXP10_SPAWN_REQ
    if not _EXP10_RUNNING or not _EXP10_VEHICLE_ID:
        return jsonify({"status": "error", "message": "实验10 未在运行"}), 409

    data = request.get_json(silent=True) or {}
    distance = max(10.0, min(15.0, float(data.get("distance", 12.0))))

    evt = threading.Event()
    with _EXP10_SPAWN_LOCK:
        _EXP10_SPAWN_REQ = {"distance": distance, "done": evt, "result": {}}
    evt.wait(timeout=8.0)
    with _EXP10_SPAWN_LOCK:
        req = _EXP10_SPAWN_REQ
        _EXP10_SPAWN_REQ = None
    if not evt.is_set() or not req:
        return jsonify({"status": "error", "message": "生成超时（实验可能已结束）"}), 500
    res = req["result"]
    if not res.get("ok"):
        return jsonify({"status": "error", "message": res.get("message", "生成失败")}), 409
    return jsonify({
        "status": "ok", "id": res["id"], "type": res["type"],
        "distance": res["distance"], "on_route": res.get("on_route", False),
        "location": res["location"],
    })


@app.route("/experiment/10/start", methods=["POST"])
def experiment_10_start():
    global _EXP10_THREAD, _EXP10_RUNNING, _EXP10_ABORT
    with _EXP10_LOCK:
        if _EXP10_THREAD is not None and _EXP10_THREAD.is_alive():
            if not _EXP10_ABORT:
                # 正在正常运行：直接拒绝。重试/双击产生的重复 start 在此被挡，
                # 绝不 kill-重启实验
                return jsonify({"status": "error", "message": "实验10 已在运行"}), 409
            # 正在停止/收尾：等旧线程完整退出后接管（保留「停止→快速重启」体验）
            _EXP10_THREAD.join(timeout=15.0)
            if _EXP10_THREAD.is_alive():
                return jsonify({"status": "error", "message": "实验10 旧线程仍在收尾，请稍后重试"}), 409
        # RUNNING/ABORT 在持锁的 handler 内提前置位：
        # ① 关闭 /cleanup、/preview/start 守卫在「start 已返回、线程体尚未执行」间的穿透窗口；
        # ② 消除「新线程刚起、ABORT 仍残留旧 stop 的 true」导致后续 start 误判收尾中的窗口
        _EXP10_RUNNING = True
        _EXP10_ABORT = False
        args = request.get_json(silent=True) or {}
        try:
            _EXP10_THREAD = threading.Thread(target=_run_exp10, args=(args,), daemon=True)
            _EXP10_THREAD.start()
        except Exception as exc:
            _EXP10_RUNNING = False
            return jsonify({"status": "error", "message": f"实验10 线程启动失败: {exc}"}), 500
    return jsonify({"status": "ok", "experiment_id": 10, "message": "实验10 (闭环自动驾驶) 已启动"})


@app.route("/experiment/10/stop", methods=["POST"])
def experiment_10_stop():
    global _EXP10_ABORT
    _EXP10_ABORT = True
    return jsonify({"status": "ok", "message": "实验10 停止请求已发送"})


@app.route("/experiment/10/params", methods=["POST"])
def experiment_10_params():
    """实验10运行中实时调整控制参数（前端滑杆即时生效，无需重启实验）"""
    data = request.get_json(silent=True) or {}
    with _EXP10_CTRL_LOCK:
        for key in ("kp_steer", "lookahead", "target_speed", "steer_delay", "brake_force"):
            if key in data and data[key] is not None:
                _EXP10_CTRL[key] = float(data[key])
        # 感知闭环开关：运行中实时切换「bbox 相机感知 / 世界真值」，便于 A/B 对比
        if "perception" in data and data["perception"] is not None:
            _EXP10_CTRL["perception"] = 1.0 if bool(data["perception"]) else 0.0
        snapshot = dict(_EXP10_CTRL)
    return jsonify({"status": "ok", "params": snapshot})

