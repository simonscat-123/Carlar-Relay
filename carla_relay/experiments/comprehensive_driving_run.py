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
            _exp_log(f"路线规划完成: {len(route_wp)} waypoints")
        except Exception as exc:
            _exp_log(f"路线规划失败({exc})，使用直线插值")
            route_wp = [start_loc, end_loc]
        route_lane_ids += [None] * (len(route_wp) - len(route_lane_ids))
        _EXP10_ROUTE = list(route_wp)

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

        # ── 方案A：车道级横向避障状态（只绕「挡在本车道」的障碍 → 换到邻道中心 → 越过回正）──
        avoiding = False          # 是否处于绕行状态
        avoid_side = 0            # +1 向左 / -1 向右
        avoid_lat_target = 0.0    # 目标横向偏移（米，带符号；左正右负）
        avoid_target_id = None    # 正在绕行的障碍 actor id
        avoid_target_lane = None  # 触发绕行时被绕障碍所在车道 (road,lane)（车道级退出判据用）
        avoid_dest_lane = None    # 换道目标车道 (road,lane)（鸟瞰图高亮用）
        avoid_travel = 0.0        # 进入绕行后累计前进距离
        avoid_start_pos = None    # 进入绕行时车辆位置
        AVOID_TRIGGER = 20.0      # 前方障碍进入该距离（m）即触发绕行
        AVOID_OFFSET = 3.5        # 绕行基础横向偏移（m，邻道信息缺失时的回退值）
        ego_half_w = float(vehicle.bounding_box.extent.y)  # 自车半宽（走廊判据用）

        def _route_proj(px, py):
            """把世界坐标投影到路线：在当前路点前方窗口内找最近路点，
            返回 (lat, j)：lat = 相对路线切线的横向偏移（左正，与 lat 约定一致）。
            用于「障碍是否挡在本车道路线走廊内」的几何判据（含车宽）。"""
            j0 = max(0, wp_idx - 5)
            j1 = min(len(route_wp), wp_idx + int(60.0 / max(0.5, sampling_res)) + 10)
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
            return tx * ry - ty * rx, best_j

        def _blocks(fwd_dist, lat_route, lane_id, half_w, lanes_ahead):
            """障碍是否「挡在本车道」：① 障碍所在车道属于路线前方车道集合；
            ② 车道信息缺失时按路线走廊几何判据（含双方半宽 + 0.25m 余量）。
            邻道车辆因不在路线车道、也不在走廊内，不再触发避障/绕行。"""
            if fwd_dist <= 0.0 or fwd_dist > 50.0:
                return False
            if lane_id is not None and lane_id in lanes_ahead:
                return True
            return abs(lat_route) < (ego_half_w + half_w + 0.25)

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

            # 障碍物扫描（车道级判据：只有「挡在本车道/路线走廊」的障碍才计入避障与绕行）
            front_obstacle = float("inf")
            front_obs_src = "none"
            dst_actor2 = 999.0
            obs_list = []
            bbox_cands = []               # 包围框相机候选：50m 内的车辆/行人 (actor, 距离)
            avoid_cand_id = None          # 绕行候选：前方挡道的最近障碍
            avoid_cand_dist = float("inf")
            avoid_cand_lat = 0.0          # 候选障碍相对路线切线的横向偏移（左正右负）
            avoid_cand_lane = None        # 候选障碍所在车道 id
            detected_lane_fwd = []         # 本帧所有已探测障碍的 (lane_id, 前向距离)，选道/退出判据用
            perceived = []                # 感知闭环：bbox 相机单目感知到的障碍物
            fwd = ego_tf.get_forward_vector()
            left = carla.Vector3D(x=fwd.y, y=-fwd.x, z=0)  # 车体系左向（lat 左正约定；(-fwd.y,fwd.x) 是右向，曾致感知坐标镜像）
            # 路线前方 60m 涉及的车道集合（判断障碍是否挡在规划路线上）
            lanes_ahead = set()
            for j in range(wp_idx, min(len(route_lane_ids), wp_idx + int(60.0 / max(0.5, sampling_res)) + 10)):
                if route_lane_ids[j] is not None:
                    lanes_ahead.add(route_lane_ids[j])

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
                    lat_route, _pj = _route_proj(wpx, wpy)
                    lane_id = None
                    try:
                        owp = carla_map.get_waypoint(carla.Location(x=wpx, y=wpy, z=0.0),
                                                     project_to_road=True, lane_type=carla.LaneType.Driving)
                        if owp is not None:
                            lane_id = (owp.road_id, owp.lane_id)
                    except Exception:
                        pass
                    detected_lane_fwd.append((lane_id, fwd_dist))
                    half_w = max(0.3, p["width_m"] / 2.0)
                    if _blocks(fwd_dist, lat_route, lane_id, half_w, lanes_ahead):
                        if fwd_dist < front_obstacle:
                            front_obstacle = fwd_dist
                            front_obs_src = p["cls"]
                        if fwd_dist < avoid_cand_dist:
                            avoid_cand_dist = fwd_dist
                            avoid_cand_lat = lat_route
                            avoid_cand_lane = lane_id
                    obs_list.append({
                        "category": _EXP10_PERC_STYLE.get(p["cls"], ("目标",))[0],
                        "dist": round(p["dist"], 1), "size": round(p["width_m"], 1),
                        "x": round(wpx, 1), "y": round(wpy, 1),  # 感知估计的世界坐标（鸟瞰图标记）
                        "vel": None,  # 单帧感知无速度估计（多帧跟踪是进阶内容）
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
                    # 前向投影（车头系，用于距离）
                    rel = actor.get_location() - fused_loc
                    fwd_dist = fwd.x * rel.x + fwd.y * rel.y
                    # 车道级判据：障碍位置 → 路线系横向偏移 + 所在车道
                    aloc = actor.get_location()
                    lat_route, _pj = _route_proj(aloc.x, aloc.y)
                    lane_id = None
                    try:
                        owp = carla_map.get_waypoint(aloc, project_to_road=True, lane_type=carla.LaneType.Driving)
                        if owp is not None:
                            lane_id = (owp.road_id, owp.lane_id)
                    except Exception:
                        pass
                    detected_lane_fwd.append((lane_id, fwd_dist))
                    if _blocks(fwd_dist, lat_route, lane_id, bb.y, lanes_ahead):
                        if fwd_dist < front_obstacle:
                            front_obstacle = fwd_dist
                            front_obs_src = tid
                        if fwd_dist < avoid_cand_dist:
                            avoid_cand_dist = fwd_dist
                            avoid_cand_lat = lat_route
                            avoid_cand_id = actor.id
                            avoid_cand_lane = lane_id
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

            # 红绿灯：只考虑车辆正前方、大致同车道（侧向<5m）的灯，
            # 避免把路边/身后/交叉口相邻车道的红灯误当成本车道红灯导致误刹
            tl_state = "green"
            tl_dist = 999.0
            try:
                ego_fwd = ego_tf.get_forward_vector()
                tls = world.get_actors().filter("traffic.traffic_light*")
                for tl in tls:
                    rel = tl.get_location() - fused_loc
                    ahead = ego_fwd.x * rel.x + ego_fwd.y * rel.y
                    if ahead <= 0:          # 在后方或侧面，忽略
                        continue
                    lateral = abs(ego_fwd.x * rel.y - ego_fwd.y * rel.x)
                    if lateral > 5.0:       # 不在本车道 / 太偏，忽略
                        continue
                    if ahead < 30 and ahead < tl_dist:
                        tl_dist = ahead
                        tl_state = _TL_STATE_MAP.get(tl.state, "green")
            except Exception:
                pass

            # 车速（先于避障决策与纵向控制，供同帧使用）
            vel = vehicle.get_velocity()
            spd = math.sqrt(vel.x ** 2 + vel.y ** 2)

            # ── 方案A：车道级横向避障决策（先于纵向控制，保证触发/退出与制动同帧生效）──
            # 触发：挡在本车道的障碍进入 AVOID_TRIGGER 即触发（不限车速，避免低速被刹停在障碍前）；
            # 退出：被绕障碍所在车道的前方已无障碍（均已到侧后方）且前进≥10m（或绕行超距兜底）。
            avoid_cand_behind = False
            if avoid_target_lane is not None:
                # 车道级退出：原车道上「还在前方(-1~25m)」的障碍清空即认为已越过
                avoid_cand_behind = not any(
                    l == avoid_target_lane and -1.0 < f < 25.0 for l, f in detected_lane_fwd)
            elif perception_on:
                # 感知闭环回退：相机只能看见前方，视野内已无任何挡道目标即准备回正
                avoid_cand_behind = math.isinf(avoid_cand_dist)
            elif avoid_target_id is not None:
                try:
                    ta = world.get_actor(avoid_target_id)
                    if ta is None:
                        avoid_cand_behind = True
                    else:
                        trel = ta.get_location() - fused_loc
                        if fwd.x * trel.x + fwd.y * trel.y < -3.0:
                            avoid_cand_behind = True
                except Exception:
                    avoid_cand_behind = True

            if avoiding:
                if avoid_start_pos is not None:
                    avoid_travel = fused_loc.distance(avoid_start_pos)
                if (avoid_cand_behind and avoid_travel >= 10.0) or avoid_travel >= 70.0:
                    avoiding = False
                    avoid_side = 0
                    avoid_lat_target = 0.0
                    avoid_target_id = None
                    avoid_target_lane = None
                    avoid_dest_lane = None
                    _exp_log("绕行完成，回正车道")
            elif avoid_cand_dist < AVOID_TRIGGER and (perception_on or avoid_cand_id is not None):
                avoiding = True
                avoid_target_id = avoid_cand_id
                avoid_target_lane = avoid_cand_lane
                avoid_dest_lane = None
                avoid_start_pos = fused_loc
                avoid_travel = 0.0
                # 选边（车道级）：优先「可行驶、同向、前方无障碍」的相邻车道，换道目标=邻道中心
                lane_pick = None
                try:
                    ewp = carla_map.get_waypoint(fused_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if ewp is not None:
                        occupied = {l for l, f in detected_lane_fwd if l is not None and -1.0 < f < 30.0}
                        for nb in (ewp.get_left_lane(), ewp.get_right_lane()):
                            if nb is None or nb.lane_type != carla.LaneType.Driving:
                                continue
                            nf = nb.transform.get_forward_vector()
                            if nf.x * fwd.x + nf.y * fwd.y <= 0.0:
                                continue  # 对向车道，不可选
                            if (nb.road_id, nb.lane_id) in occupied:
                                continue  # 邻道前方有障碍，不可选
                            nlat, _pj = _route_proj(nb.transform.location.x, nb.transform.location.y)
                            lane_pick = (1 if nlat >= 0 else -1,
                                         max(3.0, min(5.5, (ewp.lane_width + nb.lane_width) / 2.0)))
                            avoid_dest_lane = (nb.road_id, nb.lane_id)
                            break
                except Exception:
                    pass
                if lane_pick is not None:
                    avoid_side, mag = lane_pick
                else:
                    # 回退：邻道信息缺失时按几何启发式（障碍偏左就右绕、偏右就左绕；居中默认左）
                    avoid_side = 1 if abs(avoid_cand_lat) < 0.8 else (1 if avoid_cand_lat < 0 else -1)
                    mag = max(3.0, min(5.5, abs(avoid_cand_lat) + AVOID_OFFSET))
                avoid_lat_target = avoid_side * mag
                _exp_log(f"前方 {avoid_cand_dist:.0f}m 本车道内有障碍，换道绕行{'左' if avoid_side > 0 else '右'}（偏移 {mag:.1f}m）")

            # 纵向控制：绕行期间忽略"前方障碍"的制动力（避免被刹停在障碍前），红绿灯制动仍生效。
            # 采用"所需减速度 + 最苛刻约束仲裁"：红绿灯停车线 / 前方障碍 / 黄灯各约束分别
            # 按当前车速，计算为在各自目标点停下所需的最小减速度 a=v²/(2d)，取最苛刻者施加。
            # 物理风险（障碍、红灯）因所需减速度大而天然压过黄灯（规则）——车速不足时所需
            # 减速度增大、刹车自动加重，满足"障碍优先于黄灯、车速不足则刹停"。
            brake_obstacle = float("inf") if avoiding else front_obstacle
            throttle = 0.0
            brake = 0.0

            MAX_DECEL = 4.0            # 最大可用减速度 (m/s²)，对应 brake≈1
            COMFORT_D = 0.8            # 低于此减速度无需制动（正常巡航 / 跟车）
            YELLOW_D = 1.0             # 黄灯"建议减速"幅度 (m/s²)，软约束
            OBS_MARGIN = safe_dist     # 刹停后与前方障碍保持的安全距离
            RED_MARGIN = 3.0           # 距红灯停车线前的停止余量

            a_need = 0.0
            desired = target_speed
            red_state = tl_state == "red"

            # 障碍约束：在距离 d 内降速到 0，并预留 OBS_MARGIN 安全间距
            if brake_obstacle < float("inf"):
                d_obs = max(0.0, brake_obstacle - OBS_MARGIN)
                a_obs = MAX_DECEL if d_obs <= 0.0 else spd * spd / (2.0 * d_obs)
                a_need = max(a_need, a_obs)
                desired = min(desired, math.sqrt(2.0 * MAX_DECEL * d_obs))

            # 红灯停车线约束：在距离 d 内降速到 0
            if red_state:
                d_stop = max(0.0, tl_dist - RED_MARGIN)
                a_red = MAX_DECEL if d_stop <= 0.0 else spd * spd / (2.0 * d_stop)
                a_need = max(a_need, a_red)
                desired = min(desired, math.sqrt(2.0 * MAX_DECEL * d_stop))

            # 黄灯：软约束。仅当没有更紧急的红灯/障碍亟需更高减速度时生效，
            # 保证"障碍优先于黄灯"——障碍已要求更高减速度时，黄灯不再加重刹车。
            if tl_state == "yellow" and not red_state and a_need < YELLOW_D:
                a_need = YELLOW_D
                desired = min(desired, target_speed * 0.5)

            # 施加制动：低于阈值视为无需主动刹车（正常巡航/跟车），否则按所需减速度占比输出
            if a_need > COMFORT_D:
                brake = min(1.0, max(0.0, (a_need / MAX_DECEL) * brake_force))
            else:
                err = desired - spd
                throttle = max(0.0, min(1.0, 0.25 + 0.12 * err))
            speed_history.append(round(spd, 2))

            # 横向控制 (Pure Pursuit) —— 与实验8/9 相同的已验证实现：
            # 1) 前视点以车辆当前位置为基准，向前找第一个距离 ≥ lookahead 的路点；
            #    车辆不动时前视点保持不动（不再每帧前跳），避免前视点被推远导致转向饱和/偏航。
            # 2) 转向角用标准单车模型公式，符号与实验8/9 一致（实测可正常跟线）。
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
                # 2) 前视点：车前第一个距离 ≥ lookahead 的点；找不到则取末尾
                target_idx = wp_idx
                for j in range(wp_idx, len(route_wp)):
                    if math.hypot(loc.x - route_wp[j].x, loc.y - route_wp[j].y) >= lookahead:
                        target_idx = j
                        break
                else:
                    target_idx = len(route_wp) - 1
                look_pt = route_wp[target_idx]
                # 绕行：把前视目标点沿路线法向平移（路线坐标系，避免随车头旋转导致绕圈）
                if avoiding:
                    i0 = max(0, target_idx - 1)
                    i1 = min(len(route_wp) - 1, target_idx + 1)
                    tx0 = route_wp[i1].x - route_wp[i0].x
                    ty0 = route_wp[i1].y - route_wp[i0].y
                    tl0 = math.hypot(tx0, ty0)
                    if tl0 > 1e-6:
                        tx0, ty0 = tx0 / tl0, ty0 / tl0
                    else:
                        tx0, ty0 = fwd.x, fwd.y
                    look_pt = carla.Location(
                        x=look_pt.x + (-ty0) * avoid_lat_target,
                        y=look_pt.y + tx0 * avoid_lat_target,
                        z=look_pt.z,
                    )
                dx = look_pt.x - loc.x
                dy = look_pt.y - loc.y
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
                look_pt = gt_loc
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
                    f"avd={'L' if avoid_side > 0 else 'R' if avoid_side < 0 else '-'}{avoid_target_lane if avoid_target_lane is not None else ''} "
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

            # 参考线：前方 60m 每 2m 一点；避障时沿路线法向平移（自车处偏移 0、
            # 12m 处渐变到全幅，自然呈现 S 形换道轨迹）
            ref_path = []
            if route_wp:
                step_j = max(1, int(round(2.0 / max(0.5, sampling_res))))
                for j in range(wp_idx, len(route_wp), step_j):
                    px, py = route_wp[j].x, route_wp[j].y
                    if len(ref_path) >= 5 and math.hypot(px - fused_loc.x, py - fused_loc.y) > 60.0:
                        break
                    if avoiding:
                        a = route_wp[max(0, j - 1)]
                        b = route_wp[min(len(route_wp) - 1, j + 1)]
                        tx, ty = b.x - a.x, b.y - a.y
                        tl = math.hypot(tx, ty)
                        if tl > 1e-6:
                            d = math.hypot(px - fused_loc.x, py - fused_loc.y)
                            off = avoid_lat_target * max(0.0, min(1.0, d / 12.0))
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
        _push_to_sse({"experiment": {"id": 10, "status": "error", "message": str(exc)}})
    finally:
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
            except Exception as exc:
                _exp_log(f"传感器清理异常 sid={sid}: {exc!r}")
        # 清理行人
        for obj in _EXP10_PEDI:
            try:
                if obj and obj.is_alive:
                    obj.destroy()
            except Exception:
                pass
        _EXP10_PEDI.clear()
        # 清理车辆
        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
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

