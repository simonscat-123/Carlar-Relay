"""定位分析 · 前端关卡 localization（API 实验ID 23，合并历史实验2+3）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 23（合并实验2+3）：服务端运行完整实验 + 互补滤波，SSE 推送轨迹点
# =============================================================================

_EXP23_LOCK = threading.Lock()
_EXP23_RUNNING = False
_EXP23_ABORT = False
_EXP23_THREAD = None

EARTH_RADIUS_M = 6378137.0


def _llh_to_local(lat, lon, alt, lat0, lon0, alt0):
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    z = alt - alt0
    return x, y, z


def _route_cte(route_wps, wp_idx, x, y):
    """点到路线的有符号横向偏差（米）。在 wp_idx 邻域找最近路段做投影，左偏为正。
    仅用于评分统计，不参与控制。"""
    lo = max(0, wp_idx - 5)
    hi = min(len(route_wps) - 2, wp_idx + 10)
    best_d = float("inf")
    best_cross = 0.0
    for j in range(lo, hi + 1):
        ax, ay = route_wps[j]
        bx, by = route_wps[j + 1]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            continue
        s = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
        px, py = ax + s * dx, ay + s * dy
        d = math.hypot(x - px, y - py)
        if d < best_d:
            best_d = d
            best_cross = dx * (y - py) - dy * (x - px)
    if best_d == float("inf"):
        return 0.0
    return math.copysign(best_d, best_cross)


def _run_exp23(args):
    """运行合并实验23的线程函数"""
    global _EXP23_RUNNING, _EXP23_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle
    _EXP_CURRENT_ID = 23
    _EXP23_RUNNING = True
    _EXP23_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验23 (GNSS/INS 融合 · 闭环驾驶) 启动")
    _sweep_stale_actors()
    _exp_log("正在配置仿真环境（同步模式 + 交通管理）…")

    duration = float(args.get("duration", 20.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    settle_seconds = float(args.get("settle_seconds", 1.5))
    launch_seconds = float(args.get("launch_seconds", 2.0))
    launch_throttle = float(args.get("launch_throttle", 0.25))
    alpha = float(args.get("alpha", 0.08))
    gnss_noise = max(0.0, float(args.get("gnss_noise", 0.0)))
    ins_noise = max(0.0, float(args.get("ins_noise", 0.0)))
    seed = int(args.get("seed", 7))
    # 闭环控制参数（冻结，保证定位噪声是唯一变量）
    target_speed = float(args.get("target_speed", 6.0))
    lookahead = float(args.get("lookahead", 6.0))
    # GNSS 失锁窗口 [开始秒, 结束秒]（相对采集起点），None 表示不失锁
    outage = args.get("gnss_outage")
    if isinstance(outage, (list, tuple)) and len(outage) == 2:
        outage = (float(outage[0]), float(outage[1]))
    else:
        outage = None
    if outage:
        _exp_log(f"GNSS 失锁窗口: {outage[0]:.0f}s ~ {outage[1]:.0f}s")

    actors = []
    sensors = {}  # sid → actor ref
    old_settings = None

    try:
        # --- 同步模式 ---
        old_settings = world.get_settings()
        new_settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=fixed_delta,
            no_rendering_mode=False,
        )
        world.apply_settings(new_settings)
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        # --- 生成车辆 ---
        blueprint = world.get_blueprint_library().filter("vehicle.*")[0]
        spawn_pts = world.get_map().get_spawn_points()
        rng = random.Random(seed)
        pt = rng.choice(spawn_pts)
        vehicle = world.spawn_actor(blueprint, pt)
        actors.append(vehicle)
        vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        vehicle_id = vehicle.id
        _exp_log(f"车辆已生成 (id={vehicle_id})")

        # --- 挂载相机 ---
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp,
                                carla.Transform(carla.Location(x=1.5, z=1.8)),
                                attach_to=vehicle)
        actors.append(cam)
        _sensor_refs[cam.id] = cam  # 防止 GC 导致 listen() 回调失效
        cam.listen(lambda data, sid=cam.id: _sensor_callback(sid, "camera", data))
        _stream_camera = cam.id
        _stream_vehicle = vehicle_id

        # --- 挂载 GNSS ---
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", str(fixed_delta))
        gnss = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=1.8)),
                                 attach_to=vehicle)
        actors.append(gnss)
        sensors[gnss.id] = gnss
        # 用 actor.id（稳定整数）而不是 id(对象)（内存地址，对象销毁后地址可能被新对象复用导致串数据）
        gnss.listen(lambda data, sid=gnss.id: _sensor_callback(sid, "gnss", data))
        _sensor_dtype[gnss.id] = "gnss"

        # --- 挂载 IMU ---
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(fixed_delta))
        imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=1.8)),
                                attach_to=vehicle)
        actors.append(imu)
        sensors[imu.id] = imu
        imu.listen(lambda data, sid=imu.id: _sensor_callback(sid, "imu", data))
        _sensor_dtype[imu.id] = "imu"

        # 纳入托管列表（/actors 可见；本线程结束时会在 finally 中自行销毁并移出）
        for a in actors:
            with _lock:
                _managed_actors.add(a.id)

        _exp_log("传感器已挂载")

        # --- 稳定期（先稳定物理，再生成路线）---
        # 注意：路线生成必须在 settle 之后。spawn 后立即调 get_location() 在
        # CARLA 冷启动时可能返回未初始化值（如 (0,0,0)），导致 get_waypoint
        # 投影到错误道路 → 路线起点与车辆实际位置差几十米 → 横向偏差巨大 →
        # 持续满舵（首跑"向右打死"根因）。settle 后物理稳定，位置可靠。
        _exp_log("静置稳定中…")
        settle_ticks = max(0, int(settle_seconds / fixed_delta))
        for i in range(settle_ticks):
            if _EXP23_ABORT:
                raise RuntimeError("已中止")
            world.tick()
        vehicle.apply_control(carla.VehicleControl())
        # 清空静置期间累积的过期传感器数据（对齐 exp02）
        for sid in list(_sensor_frames.keys()):
            if _sensor_dtype.get(sid) in ("gnss", "imu"):
                _sensor_frames.pop(sid, None)
        _exp_log("稳定期结束，起步加速中…")

        # --- 闭环参考路线：沿车道中心线向前采样（基于 settle 后的稳定位置）---
        carla_map = world.get_map()
        ego_loc = vehicle.get_location()
        wp_cur = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        route_wps = [(wp_cur.transform.location.x, wp_cur.transform.location.y)]
        lane_half = (getattr(wp_cur, "lane_width", 3.5) or 3.5) / 2.0
        route_step = 2.0
        route_len = max(80.0, target_speed * duration + 30.0)
        _acc_len = 0.0
        while _acc_len < route_len:
            nxt = wp_cur.next(route_step)
            if not nxt:
                break
            wp_cur = nxt[0]
            route_wps.append((wp_cur.transform.location.x, wp_cur.transform.location.y))
            _acc_len += route_step
        _exp_log(f"闭环路线: {len(route_wps)} 航点 / ~{_acc_len:.0f} m，车道半宽 {lane_half:.1f} m"
                 f"，起点 ({ego_loc.x:.1f}, {ego_loc.y:.1f})")

        # --- 起步 ---
        launch_ticks = max(0, int(launch_seconds / fixed_delta))
        for i in range(launch_ticks):
            if _EXP23_ABORT:
                raise RuntimeError("已中止")
            progress = (i + 1) / max(1, launch_ticks)
            throttle = launch_throttle * progress
            vehicle.apply_control(carla.VehicleControl(throttle=throttle))
            world.tick()

        # --- 闭环控制（不再使用 TM 自动驾驶：转向/油门由融合估计位姿驱动）---
        _exp_log("闭环控制已启用（Pure Pursuit + PID），开始采集")

        # --- 互补滤波器（参照 exp03_gnss_ins_filter.py，首帧不跳过）---
        lat0 = lon0 = alt0 = None
        ins_x = ins_y = ins_vx = ins_vy = 0.0
        gt_ref = None
        last_t = None
        trajectory = []
        total_ticks = int(duration / fixed_delta)

        # 航向融合与闭环控制状态
        fused_yaw_deg = 0.0
        prev_gt_yaw_deg = None
        wp_idx = 0
        pid_prev_err = 0.0
        pid_integral = 0.0
        prev_steer_sign = 0
        steer_reversals = 0
        cte_sum = 0.0
        cte_max = 0.0
        offlane_ticks = 0
        trajectory_pushed_route = False
        diag_ticks = 0          # 首帧诊断日志计数
        diverge_ticks = 0       # 融合发散看门狗计数

        for tick_idx in range(total_ticks):
            if _EXP23_ABORT:
                _exp_log("收到停止请求")
                break

            world.tick()
            snapshot = world.get_snapshot()
            t_sim = snapshot.timestamp.elapsed_seconds

            # 读取传感器数据（按本实验传感器 sid 精确读取；旧实现扫描全局
            # _sensor_frames 抓任意 gnss/imu 帧，可能吃到残留传感器数据）
            gnss_data = None
            imu_data = None
            raw_g = _sensor_frames.get(gnss.id)
            raw_i = _sensor_frames.get(imu.id)
            if raw_g is not None:
                try:
                    gnss_data = json.loads(raw_g.decode())
                except Exception:
                    pass
            if raw_i is not None:
                try:
                    imu_data = json.loads(raw_i.decode())
                except Exception:
                    pass
            if gnss_data is None or imu_data is None:
                continue

            # --- 初始化（首帧不跳过，对齐 exp03：dt=1e-3，从 llh_to_local 零位开始积分）---
            if last_t is None:
                lat0 = gnss_data["latitude"]
                lon0 = gnss_data["longitude"]
                alt0 = gnss_data["altitude"]
                ins_x = ins_y = ins_vx = ins_vy = 0.0
                last_t = t_sim
                dt = 1e-3  # 与 exp03 首帧 dt 一致
            else:
                dt = max(1e-3, t_sim - last_t)
                last_t = t_sim

            # 真值（对齐首个位置）
            loc = vehicle.get_location()
            if gt_ref is None:
                gt_ref = (loc.x, loc.y)
            gt_x_local = loc.x - gt_ref[0]
            gt_y_local = loc.y - gt_ref[1]

            # yaw 变换
            ego_tf = vehicle.get_transform()
            yaw_deg = ego_tf.rotation.yaw

            # --- 重力补偿后的 INS 加速度（核心修复）---
            # CARLA IMU 加速度计含重力分量（静止水平时 z≈+9.81），车身侧倾/俯仰会把
            # 重力泄漏到横向/纵向轴。冷启动时物理引擎未热、悬挂不沉降，车身持续微侧倾
            # → 横向加速度偏置 → INS 双积分横向漂移 → 首跑"向右打死"（物理热后正常）。
            # 处理：用本体系三轴向量把重力精确投影扣除，再做运动学积分。
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            up = ego_tf.get_up_vector()
            GRAV = 9.81
            g_body = (-GRAV * fwd.z, -GRAV * right.z, -GRAV * up.z)
            acc_meas = imu_data["accelerometer"]
            ax_body = acc_meas["x"] + g_body[0] + rng.gauss(0, ins_noise)
            ay_body = acc_meas["y"] + g_body[1] + rng.gauss(0, ins_noise)
            az_body = acc_meas["z"] + g_body[2]
            # 旋转到世界系（完整三轴，含 roll/pitch）
            ax_world = ax_body * fwd.x + ay_body * right.x + az_body * up.x
            ay_world = ax_body * fwd.y + ay_body * right.y + az_body * up.y
            # 野值/物理冷启动毛刺限幅（本实验目标速度 ~6 m/s，>15 m/s² 非物理）
            ax_world = max(-15.0, min(15.0, ax_world))
            ay_world = max(-15.0, min(15.0, ay_world))

            # 首帧诊断日志（排查冷启动异常用，仅前 3 个有效帧）
            if diag_ticks < 3:
                diag_ticks += 1
                _exp_log(f"[diag] IMU原始 acc=({acc_meas['x']:+.2f},{acc_meas['y']:+.2f},"
                         f"{acc_meas['z']:+.2f}) 重力补偿后=({ax_body:+.2f},{ay_body:+.2f}) "
                         f"GNSS=({gnss_data['latitude']:.6f},{gnss_data['longitude']:.6f})")

            # INS 积分（速度限幅防发散）
            ins_vx = max(-25.0, min(25.0, ins_vx + ax_world * dt))
            ins_vy = max(-25.0, min(25.0, ins_vy + ay_world * dt))
            ins_x += ins_vx * dt
            ins_y += ins_vy * dt

            # GNSS → 本地坐标（叠加位置噪声模拟定位误差）
            gx_raw, gy_raw, _ = _llh_to_local(
                gnss_data["latitude"], gnss_data["longitude"], gnss_data["altitude"],
                lat0, lon0, alt0,
            )
            gx = gx_raw + rng.gauss(0, gnss_noise)
            gy = gy_raw + rng.gauss(0, gnss_noise)

            # GNSS 失锁判定：窗口内无观测，融合退化为纯 INS 推算
            elapsed_exp = tick_idx * fixed_delta
            gnss_valid = outage is None or not (outage[0] <= elapsed_exp <= outage[1])

            # 互补滤波校正（GNSS 突跳保护：与 INS 相差 >60 m 视为野值，跳过本帧校正）
            if gnss_valid and math.hypot(gx - ins_x, gy - ins_y) < 60.0:
                ins_x = (1 - alpha) * ins_x + alpha * gx
                ins_y = (1 - alpha) * ins_y + alpha * gy

            # --- 航向融合（互补滤波，与位置同构）：INS 陀螺递推 + GNSS 航向观测 ---
            # 陀螺角速度用「真值航向差分」模拟（CARLA IMU 陀螺读数在此环境不可靠，同实验10 的处理）
            if prev_gt_yaw_deg is None:
                fused_yaw_deg = yaw_deg
                prev_gt_yaw_deg = yaw_deg
            else:
                delta_yaw = ((yaw_deg - prev_gt_yaw_deg + 180.0) % 360.0) - 180.0
                prev_gt_yaw_deg = yaw_deg
                gyro_w = delta_yaw / dt + rng.gauss(0, ins_noise * 30.0)   # °/s
                ins_yaw_deg = fused_yaw_deg + gyro_w * dt
                if gnss_valid:
                    gnss_yaw_deg = yaw_deg + rng.gauss(0, gnss_noise * 4.0)  # 度
                    # 用卷绕后的相位差(innovation)做互补滤波，避免 ±180° 边界跳变
                    innov_deg = ((gnss_yaw_deg - ins_yaw_deg + 180.0) % 360.0) - 180.0
                    fused_yaw_deg = ins_yaw_deg + alpha * innov_deg
                else:
                    fused_yaw_deg = ins_yaw_deg
                fused_yaw_deg = ((fused_yaw_deg + 180.0) % 360.0) - 180.0

            # --- 闭环控制：Pure Pursuit 只吃融合估计位姿，真值严禁流入控制 ---
            fused_tform = carla.Transform(
                carla.Location(x=ins_x + gt_ref[0], y=ins_y + gt_ref[1], z=loc.z),
                carla.Rotation(yaw=fused_yaw_deg),
            )
            steer, wp_idx, _ = _pure_pursuit_steer(fused_tform, route_wps, lookahead, wp_idx)
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            throttle, brake, pid_prev_err, pid_integral = _pid_speed_control(
                speed, target_speed, pid_prev_err, pid_integral, dt)
            if wp_idx >= len(route_wps) - 3:
                # 跑到路线末端：刹停等待计时结束
                throttle, brake = 0.0, 1.0
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            # 转向平稳性统计（0.02 死区内不判反转）
            steer_sign = 1 if steer > 0.02 else (-1 if steer < -0.02 else 0)
            if steer_sign != 0:
                if prev_steer_sign != 0 and steer_sign != prev_steer_sign:
                    steer_reversals += 1
                prev_steer_sign = steer_sign

            err = math.hypot(ins_x - gt_x_local, ins_y - gt_y_local)

            # 融合发散看门狗（安全联锁：真值仅用于检测，控制仍只吃融合位姿）。
            # 冷启动等异常导致 INS 严重发散时，把融合位姿重置到当前 GNSS 观测，
            # 避免车辆持续满舵偏驶。
            if err > 15.0:
                diverge_ticks += 1
            else:
                diverge_ticks = 0
            if diverge_ticks >= 10 and gnss_valid:
                _exp_log(f"融合发散保护触发（err={err:.1f}m），重置到 GNSS 观测")
                ins_x, ins_y = gx, gy
                ins_vx = ins_vy = 0.0
                diverge_ticks = 0

            # 真值横向偏差（评分用，不参与控制）：投影到最近路线段
            cte_gt = _route_cte(route_wps, wp_idx, loc.x, loc.y)
            cte_sum += abs(cte_gt)
            cte_max = max(cte_max, abs(cte_gt))
            if abs(cte_gt) > max(0.5, lane_half - 0.9):
                offlane_ticks += 1

            # 推送到 SSE
            point = {
                "t": round(t_sim, 2),
                "gnss_x": round(gx, 3) if gnss_valid else None,
                "gnss_y": round(gy, 3) if gnss_valid else None,
                "gnss_x_raw": round(gx_raw, 3),
                "gnss_y_raw": round(gy_raw, 3),
                "fused_x": round(ins_x, 3),
                "fused_y": round(ins_y, 3),
                "gt_x": round(gt_x_local, 3),
                "gt_y": round(gt_y_local, 3),
                "ins_accel_x": round(ax_body, 3),
                "ins_accel_y": round(ay_body, 3),
                "ins_accel_x_raw": round(acc_meas["x"], 3),
                "ins_accel_y_raw": round(acc_meas["y"], 3),
                "steer": round(steer, 3),
                "cte": round(cte_gt, 3),
                "error": round(err, 3),
            }
            trajectory.append(point)

            # 每 4 个 tick 推一次轨迹点（减少 SSE 频率），并携带当前进度（供前端展示，替代墙钟估算）
            if tick_idx % 4 == 0:
                push_point = {**point, "progress": round((tick_idx + 1) / total_ticks * 100, 1)}
                # 首个轨迹点附带参考路线（车道中心线），供前端绘制"规划路线"
                # 转换到与轨迹一致的本地坐标系（原点 = 首帧真值位置）
                if not trajectory_pushed_route:
                    trajectory_pushed_route = True
                    push_point = {**push_point, "route": [
                        [round(x - gt_ref[0], 3), round(y - gt_ref[1], 3)] for x, y in route_wps
                    ]}
                _push_to_sse({"experiment": {"id": 23, "trajectory": push_point}})

        # --- 计算统计指标 ---
        if trajectory:
            n_pts = len(trajectory)
            rmse = math.sqrt(sum(p["error"] ** 2 for p in trajectory) / n_pts)
            _exp_log(f"position RMSE = {rmse:.3f} m | 平均横向偏差 {cte_sum / n_pts:.2f} m | "
                     f"最大 {cte_max:.2f} m | 压线 {offlane_ticks / n_pts * 100:.0f}% | "
                     f"转向反转 {steer_reversals} 次")
            _push_to_sse({"experiment": {
                "id": 23,
                "result": {
                    "rmse": round(rmse, 3),
                    "elapsed": round(trajectory[-1]["t"] - trajectory[0]["t"], 1) if n_pts > 1 else 0,
                    "points": n_pts,
                    "avg_cte": round(cte_sum / n_pts, 3),
                    "max_cte": round(cte_max, 3),
                    "offlane_pct": round(offlane_ticks / n_pts * 100, 1),
                    "steer_reversals": steer_reversals,
                },
            }})

        _exp_log("实验23 完成" if not _EXP23_ABORT else "实验23 已停止")

    except Exception as e:
        _exp_log(f"实验23 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None

        # 先 stop 传感器以排空数据流（与实验4 相同的收尾顺序）
        for _a in actors:
            try:
                if _a is not None and getattr(_a, "is_listening", False):
                    _a.stop()
            except Exception:
                pass

        # 恢复世界为异步模式 + TM 异步 + tick，重置到 _init_carla 的基线，
        # 避免实验结束后仿真静止、前端画面永久定格。
        try:
            world.apply_settings(carla.WorldSettings(synchronous_mode=False))
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            for _i in range(5):
                world.tick()
        except Exception:
            pass

        # 销毁前先取消自动驾驶，避免销毁 TM 托管车辆时原生崩溃
        for _a in actors:
            try:
                if _a is not None and getattr(_a, "type_id", "").startswith("vehicle."):
                    _a.set_autopilot(False)
            except Exception:
                pass

        # 销毁本实验生成的所有 actors（线程自行销毁，保证快速重启时无累积泄漏）
        for _a in actors:
            try:
                if _a is not None and _a.is_alive:
                    _a.destroy()
            except Exception:
                pass
            with _lock:
                _managed_actors.discard(_a.id)
            _sensor_frames.pop(_a.id, None)
            _sensor_dtype.pop(_a.id, None)
            _sensor_refs.pop(_a.id, None)

        _EXP23_RUNNING = False
        if _EXP23_ABORT:
            _push_to_sse({"experiment": {"id": 23, "status": "stopped"}})


@app.route("/experiment/23/start", methods=["POST"])
def experiment_23_start():
    global _EXP23_THREAD, _EXP23_RUNNING, _EXP23_ABORT
    with _EXP23_LOCK:
        if _EXP23_THREAD is not None and _EXP23_THREAD.is_alive():
            if not _EXP23_ABORT:
                # 正在正常运行：直接拒绝。重试/双击产生的重复 start 在此被挡，
                # 绝不 kill-重启实验
                return jsonify({"status": "error", "message": "实验23 已在运行"}), 409
            # 正在停止/收尾：等旧线程完整退出后接管（保留「停止→快速重启」体验）
            _EXP23_THREAD.join(timeout=15.0)
            if _EXP23_THREAD.is_alive():
                return jsonify({"status": "error", "message": "实验23 旧线程仍在收尾，请稍后重试"}), 409
        # RUNNING/ABORT 在持锁的 handler 内提前置位：
        # ① 关闭 /cleanup、/preview/start 守卫在「start 已返回、线程体尚未执行」间的穿透窗口；
        # ② 消除「新线程刚起、ABORT 仍残留旧 stop 的 true」导致后续 start 误判收尾中的窗口
        _EXP23_RUNNING = True
        _EXP23_ABORT = False
        args = request.get_json(silent=True) or {}
        try:
            _EXP23_THREAD = threading.Thread(target=_run_exp23, args=(args,), daemon=True)
            _EXP23_THREAD.start()
        except Exception as exc:
            _EXP23_RUNNING = False
            return jsonify({"status": "error", "message": f"实验23 线程启动失败: {exc}"}), 500
    return jsonify({"status": "ok", "experiment_id": 23, "message": "实验23 已启动"})


@app.route("/experiment/23/stop", methods=["POST"])
def experiment_23_stop():
    global _EXP23_ABORT
    _EXP23_ABORT = True
    return jsonify({"status": "ok", "message": "实验23 停止请求已发送"})

