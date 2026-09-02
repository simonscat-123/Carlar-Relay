"""Lidar 检测 · 前端关卡 lidar-detection（API 实验ID 4）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 4: LiDAR 点云与障碍物检测（进程内运行，与 Exp23 同构）
# =============================================================================

_EXP04_LOCK = threading.Lock()
_EXP04_RUNNING = False
_EXP04_ABORT = False
_EXP04_THREAD = None

import time as _time


# #region debug-point util
def _dbg(_hyp, _loc, _msg, **_data):
    try:
        import json, urllib.request as _ur
        _ur.urlopen(_ur.Request("http://127.0.0.1:7777/event",
            data=json.dumps({"sessionId": "experiment4-restart-crash", "runId": "post-fix3",
            "hypothesisId": _hyp, "location": _loc, "msg": "[DEBUG] " + _msg, "data": _data}).encode(),
            headers={"Content-Type": "application/json"}))
    except Exception:
        pass
# #endregion


# 实验4 精确定位验证开关（默认关闭）。开启时会在主循环每 4 tick 做一次 world.cast_ray +
# world.get_actor 射线求交，验证投影精度；该调用是同步 RPC，会阻塞模拟 tick，造成明显卡顿。
_EX04_RAYCAST_VERIFY = False


def _run_exp04(args):
    """运行实验4的线程函数：生成车辆 → 挂载 LiDAR + 三目相机 → 投影到三路画面 + 前方障碍距离估计"""
    global _EXP04_RUNNING, _EXP04_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_camera_left, _stream_camera_right, _stream_vehicle
    _EXP_CURRENT_ID = 4
    _EXP04_RUNNING = True
    _EXP04_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验4 (LiDAR 障碍物检测) 启动")
    _sweep_stale_actors()
    _exp_log("正在配置仿真环境（同步模式 + 交通管理）…")
    _dbg("D", "carla_relay.py:1587", "entry", duration=args.get("duration"), npc=args.get("npc_count"))

    duration = float(args.get("duration", 20.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    lidar_range = float(args.get("range", 60.0))
    lidar_channels = int(args.get("channels", 64))
    lidar_pps = int(args.get("points_per_second", 120000))
    lidar_rf = float(args.get("rotation_frequency", 20.0))
    # 每圈 tick 数 = 仿真频率 ÷ 转速（前端滑条为整数比档位，直接取整即为精确值）
    ticks_per_scan = max(1, round((1.0 / fixed_delta) / lidar_rf)) if lidar_rf > 0 else 1
    lidar_hfov = float(args.get("lidar_hfov", 120.0))  # LiDAR 水平视场角
    camera_fov = float(args.get("camera_fov", 90.0))    # 相机视场角（与官方示例 sensor.camera.rgb 默认 90° 一致）
    npc_count = int(args.get("npc_count", 0))
    seed = int(args.get("seed", 7))

    actors = []
    latest_lidar = {"frame": 0, "count": 0, "dist": float("inf")}
    # 攒圈缓冲：转速低于仿真频率时，单个 tick 只扫出部分扇区（如 5Hz@20Hz → 每 tick 90°），
    # 需攒满一圈（ticks_per_scan 个 tick，整数比保证不重叠不漏扫）才合成整圈点云
    _scan_pts_buf = []      # 各 tick 扇区的点云缓冲
    _scan_parsed_buf = []   # 各 tick 扇区的结构化数据缓冲（含 ObjTag）
    _scan_ticks = 0         # 当前圈已累计的 tick 数

    try:
        # --- 同步模式 ---
        _dbg("E", "carla_relay.py:1617", "setup-begin")
        settings = world.get_settings()
        world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=fixed_delta,
            no_rendering_mode=False,
        ))
        _dbg("E", "carla_relay.py:1625", "apply-settings-ok")
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)
        tm.set_global_distance_to_leading_vehicle(2.5)
        _dbg("E", "carla_relay.py:1628", "tm-ok")

        # --- 生成车辆 ---
        _exp_log("正在生成主车…")
        blueprint = world.get_blueprint_library().filter("vehicle.*")[0]
        spawn_pts = world.get_map().get_spawn_points()
        rng = random.Random(seed)
        pt = rng.choice(spawn_pts)
        vehicle = world.spawn_actor(blueprint, pt)
        actors.append(vehicle)
        vehicle_id = vehicle.id
        _exp_log(f"主车已生成 (id={vehicle_id})")
        _dbg("A", "carla_relay.py:1638", "vehicle-spawned", vid=vehicle_id)

        # --- 生成 NPC 车辆 ---
        tm_port = tm.get_port()
        if npc_count > 0:
            npc_vehicles = []
            vehicle_bps = list(world.get_blueprint_library().filter("vehicle.*"))
            npc_rng = random.Random(seed + 100)
            # 可用的出生点（排除 ego 车附近）
            available_pts = [s for s in spawn_pts if s.location.distance(pt.location) >= 5.0]
            for i in range(npc_count):
                if not available_pts:
                    break
                _exp_log(f"正在生成 NPC 车辆 ({i + 1}/{npc_count})…")
                idx = npc_rng.randrange(len(available_pts))
                npc_pt = available_pts.pop(idx)  # 取出后不再重复使用
                npc_bp = npc_rng.choice(vehicle_bps)
                try:
                    npc = world.try_spawn_actor(npc_bp, npc_pt)
                except RuntimeError:
                    npc = None
                if npc:
                    npc.set_autopilot(True, tm_port)
                    npc_vehicles.append(npc)
                    actors.append(npc)
            _exp_log(f"已生成 {len(npc_vehicles)}/{npc_count} 辆 NPC")

        # --- 挂载三目 RGB 相机（左 / 前 / 右），800x450 @10fps，统一 FOV，形成宽幅全景 ---
        # 降分辨率 + sensor_tick 降频：同步模式下 tick 需等全部传感器渲染+回调完成，
        # 1280x720@20fps 曾导致每 tick ~200ms（仿真时间只有现实的 1/4）
        cam_w, cam_h = 800, 450
        # 每线每圈点数 (PLPR) = 采样点频率 ÷ (通道数 × 转速)，返回给前端做角分辨率显示
        _exp_log("正在挂载三目相机…")
        plpr_value = round(lidar_pps / (lidar_channels * lidar_rf), 2) if lidar_channels > 0 and lidar_rf > 0 else 0.0
        _exp_log(f"LiDAR 参数: PPS={lidar_pps}, channels={lidar_channels}, rotFreq={round(lidar_rf, 2)}Hz（每圈 {ticks_per_scan} tick）→ PLPR={plpr_value}，整圈点数≈{int(lidar_pps / lidar_rf)}，点云更新率={round(lidar_rf, 2)}Hz")

        def _mount_camera(yaw_deg: float):
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(cam_w))
            bp.set_attribute("image_size_y", str(cam_h))
            bp.set_attribute("fov", str(camera_fov))
            bp.set_attribute("sensor_tick", "0.1")  # 10fps 渲染：GPU/编码/传输全链路减负
            tf = carla.Transform(carla.Location(x=1.5, z=1.8),
                                 carla.Rotation(pitch=0.0, yaw=float(yaw_deg), roll=0.0))
            c = world.spawn_actor(bp, tf, attach_to=vehicle)
            actors.append(c)
            _sensor_refs[c.id] = c
            c.listen(lambda data, sid=c.id: _sensor_callback(sid, "camera", data))
            return c

        # 无缝拼接的关键：相邻相机的 yaw 间隔必须等于 FOV。
        # yaw 间隔 < FOV 会重叠（原 -60°/0°/+60° 配 FOV 120° 时每侧重叠 60°）；
        # yaw 间隔 > FOV 会漏拍。参考官方示例 visualize_multiple_sensors.py：
        # FOV 90° + yaw -90°/0°/+90° → 三目恰好覆盖 270°，画面边缘无缝衔接。
        cam_left  = _mount_camera(-camera_fov)  # 左相机：覆盖 [-1.5*FOV, -0.5*FOV]
        cam_front = _mount_camera(  0.0)        # 前相机：覆盖 [-0.5*FOV, +0.5*FOV]
        cam_right = _mount_camera(+camera_fov)  # 右相机：覆盖 [+0.5*FOV, +1.5*FOV]
        _stream_camera_left  = cam_left.id
        _stream_camera       = cam_front.id
        _stream_camera_right = cam_right.id
        _stream_vehicle = vehicle_id
        _dbg("A", "carla_relay.py:1680", "cams-ok",
             cl=cam_left.id, cf=cam_front.id, cr=cam_right.id, plpr=plpr_value)

        # --- 挂载 LiDAR（专用回调，直接解析全量点云做前方障碍检测）---
        _exp_log("正在挂载 LiDAR…")
        # 使用语义 LiDAR（ray_cast 的语义版），每条返回带 ObjTag，可识别最近障碍的类型
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("range", str(lidar_range))
        lidar_bp.set_attribute("channels", str(lidar_channels))
        lidar_bp.set_attribute("points_per_second", str(lidar_pps))
        lidar_bp.set_attribute("rotation_frequency", str(lidar_rf))
        lidar_bp.set_attribute("upper_fov", "20.0")
        lidar_bp.set_attribute("lower_fov", "-30.0")
        lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.4)),
                                  attach_to=vehicle)
        actors.append(lidar)
        _dbg("A", "carla_relay.py:1693", "lidar-spawned", lz=lidar.id)

        def _lidar_cb(data):
            # 每收到一帧 LiDAR 点云就执行一次；data 是 CARLA 推送的该帧数据对象
            # 语义 LiDAR raw_data 每点结构：x,y,z,CosAngle,ObjIdx,ObjTag
            nonlocal _scan_ticks
            try:
                parsed = np.frombuffer(data.raw_data, dtype=np.dtype([
                    ("x", np.float32), ("y", np.float32), ("z", np.float32),
                    ("CosAngle", np.float32), ("ObjIdx", np.uint32), ("ObjTag", np.uint32),
                ]))
                pts = np.empty((parsed.shape[0], 4), dtype=np.float32)
                pts[:, 0] = parsed["x"]; pts[:, 1] = parsed["y"]
                pts[:, 2] = parsed["z"]; pts[:, 3] = parsed["CosAngle"]
                structured = True
            except Exception:
                # 回退：按普通 ray_cast 解析（无对象标签）
                pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
                parsed = None
                structured = False
            # --- 攒圈合并：整数比保证每圈恰好 ticks_per_scan 个 tick；未攒满先缓存返回 ---
            _scan_pts_buf.append(pts)
            if structured:
                _scan_parsed_buf.append(parsed)
            _scan_ticks += 1
            if _scan_ticks < ticks_per_scan:
                return  # 未攒满一圈：latest_lidar 保持上一整圈数据（点云更新率 = 转速）
            pts = np.concatenate(_scan_pts_buf)
            parsed = np.concatenate(_scan_parsed_buf) if structured else None
            _scan_pts_buf.clear()
            _scan_parsed_buf.clear()
            _scan_ticks = 0
            # 水平视场角过滤（±half_fov）：保留前方锥形区域
            if lidar_hfov < 360:
                # 仅当水平 FOV < 360° 时才裁剪；360° 表示保留全向点云
                azi = np.arctan2(pts[:, 1], pts[:, 0])  # 方位角 (rad)，atan2(y, x)，x 前向 / y 侧向
                half = np.radians(lidar_hfov / 2)       # 半视场角（角度 → 弧度）
                keep = np.abs(azi) <= half
                pts = pts[keep]                         # 只保留前方锥形区域内的点
                if structured:
                    parsed = parsed[keep]
            front_mask = (pts[:, 0] > 0.0) & (np.abs(pts[:, 1]) < 3.0) & (pts[:, 2] > -1.7)
            # 更窄的“检测走廊”：正前方(x>0)、横向±3m(|y|<3)、地面以上(z>-1.7)，用于障碍检测
            if np.any(front_mask):
                dists = np.linalg.norm(pts[front_mask][:, :2], axis=1)  # 只算 x,y 平面距离
                i_min = int(np.argmin(dists))                            # 走廊内最近点的下标
                d = float(dists[i_min])                                  # 最近前方障碍距离
                c = int(front_mask.sum())                                # 走廊内点数量
                tag = int(parsed["ObjTag"][np.nonzero(front_mask)[0][i_min]]) if structured else 0
            else:
                d = float("inf")  # 走廊内无点 → 无穷远
                c = 0
                tag = 0
            total = pts.shape[0]           # FOV 过滤后剩余的点数
            step = max(1, total // 8000)   # 均匀抽样步长；不足 8000 点则 step=1（不抽稀）
            sampled = pts[::step, :3]      # 均匀取样并只保留 x,y,z（丢弃强度列）
            latest_lidar["frame"] = data.frame  # 记录本帧帧号
            latest_lidar["count"] = c           # 走廊点数
            latest_lidar["dist"] = d            # 最近障碍距离
            latest_lidar["tag"] = tag           # 最近障碍对象标签
            latest_lidar["total"] = total       # FOV 过滤后点数
            latest_lidar["points"] = sampled  # 保留 numpy 数组供投影计算，不再发给前端

        lidar.listen(_lidar_cb)
        _dbg("A", "carla_relay.py:1745", "lidar-listen-ok")

        # 纳入托管列表
        for a in actors:
            with _lock:
                _managed_actors.add(a.id)

        _exp_log(f"传感器已挂载 (LiDAR range={lidar_range}m, channels={lidar_channels})")

        # --- 稳定期 ---
        settle_ticks = max(0, int(1.0 / fixed_delta))
        for _ in range(settle_ticks):
            if _EXP04_ABORT:
                raise RuntimeError("已中止")
            world.tick()
        _dbg("B", "carla_relay.py:1755", "settle-done")

        # --- 自动驾驶 ---
        vehicle.set_autopilot(True, tm.get_port())
        _exp_log("自动驾驶已启用，开始采集")
        _exp_log("检测走廊: x>0, |y|<3m, z>-1.7m")
        _dbg("B", "carla_relay.py:1764", "autopilot-on")

        # --- 主循环 ---
        total_ticks = int(duration / fixed_delta)
        rows = []

        _exp_log("准备就绪，开始采集数据…")

        # --- 真实时间同步：让仿真以 1x 速率推进，避免仿真跑得比显示快 ---
        # 同步模式下 world.tick() 会尽量快推进，硬件快时会呈现「加速」（如 3s 仿真/1s 现实）。
        # 以墙钟为准：第 k 个 tick 对应仿真时刻 k*fixed_delta，若提前完成则休眠补齐剩余时间。
        _tick_wall0 = _time.monotonic()
        _tick_deadline = _tick_wall0

        for tick_idx in range(total_ticks):
            if _EXP04_ABORT:
                _exp_log("收到停止请求")
                break

            world.tick()
            snapshot = world.get_snapshot()
            t_sim = snapshot.timestamp.elapsed_seconds

            count = latest_lidar["count"]
            dist = latest_lidar["dist"]
            frame_num = latest_lidar["frame"]

            rows.append({
                "frame": frame_num,
                "time": t_sim,
                "front_point_count": count,
                "nearest_front_obstacle_m": dist,
                "total_points": latest_lidar.get("total", 0),
                "nearest_obstacle_type": _lidar_tag_name(latest_lidar.get("tag", 0)),
            })

            # 每 4 tick 推一次轨迹点（降低 SSE 频率），携带进度 + 三目投影 + 雷达点云
            if tick_idx % 4 == 0:
                lidar_pts = latest_lidar.get("points", None)
                # 雷达面板点云（降采样到 ~1000 点，前端画俯视图用；点数随 PPS 变化，密度差异可见）
                radar_list = []
                if lidar_pts is not None and lidar_pts.shape[0] > 0:
                    step_r = max(1, lidar_pts.shape[0] // 1000)
                    for p in lidar_pts[::step_r, :3]:
                        radar_list.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)])

                # ---- LiDAR → Camera 投影（分别计算左/前/右三目相机）----
                def _project_to_camera(cam_actor) -> list:
                    """LiDAR 局部点 → 指定相机图像像素坐标。返回 [{'u':..,'v':..,'d':..}, ...]"""
                    if lidar_pts is None or lidar_pts.shape[0] == 0 or cam_actor is None:
                        return []
                    try:
                        pts_h = np.concatenate([lidar_pts, np.ones((lidar_pts.shape[0], 1))], axis=1)
                        lidar_to_world = np.array(lidar.get_transform().get_matrix())
                        world_to_cam = np.array(cam_actor.get_transform().get_inverse_matrix())
                        lidar_to_cam = world_to_cam @ lidar_to_world
                        cam_pts = (lidar_to_cam @ pts_h.T).T
                        # CARLA cam frame (x→fwd,y→right,z→up) → image frame (z→fwd)
                        xyz = np.stack([cam_pts[:, 1], -cam_pts[:, 2], cam_pts[:, 0]], axis=1)
                        front = xyz[:, 2] > 0.1
                        if not np.any(front):
                            return []
                        xyz_f = xyz[front]
                        w_img, h_img = cam_w, cam_h
                        focal = w_img / (2.0 * math.tan(math.radians(camera_fov) / 2.0))
                        K = np.array([[focal, 0, w_img / 2], [0, focal, h_img / 2], [0, 0, 1]])
                        uvw = (K @ xyz_f.T).T
                        uv = uvw[:, :2] / uvw[:, 2:3]
                        valid = (uv[:, 0] >= 0) & (uv[:, 0] < w_img) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
                        # 降采样投影点（上限 ~2000 点/相机，点密度随 PPS 可见变化），避免渲染卡顿
                        _step_p = max(1, int(valid.sum()) // 2000)
                        _sel = slice(None, None, _step_p)
                        out = []
                        for (u, v), d in zip(uv[valid][_sel], xyz_f[valid, 2][_sel]):
                            out.append({"u": round(float(u), 1), "v": round(float(v), 1), "d": round(float(d), 2)})
                        return out
                    except Exception:
                        return []

                proj_left  = _project_to_camera(cam_left)
                proj_front = _project_to_camera(cam_front)
                proj_right = _project_to_camera(cam_right)

                _push_to_sse({"experiment": {
                    "id": 4,
                    "trajectory": {
                        "frame": frame_num,
                        "t": round(t_sim, 3),
                        "plpr": plpr_value,
                        "front_point_count": count,
                        "nearest_front_obstacle_m": round(dist, 3) if not math.isinf(dist) else None,
                        "total": latest_lidar.get("total", 0),
                        "nearest_obstacle_type": _lidar_tag_name(latest_lidar.get("tag", 0)),
                        "progress": round((tick_idx + 1) / total_ticks * 100, 1),
                        "cameras": [
                            {"key": "left",  "projection": proj_left},
                            {"key": "front", "projection": proj_front},
                            {"key": "right", "projection": proj_right},
                        ],
                        "lidar_points": radar_list,
                    },
                }})

            # 校准仿真到真实时间：未跑满 1x 时长则休眠补齐，渲染变慢时不强制追赶
            _tick_deadline += fixed_delta
            _slack = _tick_deadline - _time.monotonic()
            if _slack > 0:
                _time.sleep(_slack)

        # --- 结果 ---
        elapsed = rows[-1]["time"] - rows[0]["time"] if rows else 0
        _exp_log(f"采集完成 — {len(rows)} 行数据")
        _push_to_sse({"experiment": {
            "id": 4,
            "result": {
                "elapsed": round(elapsed, 1),
                "rows": len(rows),
            },
        }})

        _exp_log("实验4 完成" if not _EXP04_ABORT else "实验4 已停止")

    except Exception as e:
        _exp_log(f"实验4 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        _stream_camera_left = None
        _stream_camera_right = None

        # 清理三目相机引用
        for _cam in (cam_left, cam_front, cam_right):
            try:
                _sensor_frames.pop(_cam.id, None)
                _sensor_refs.pop(_cam.id, None)
            except Exception:
                pass

        # 先 stop 传感器以排空数据流
        for _a in actors:
            try:
                if _a is not None and getattr(_a, "is_listening", False):
                    _a.stop()
            except Exception:
                pass

        # 恢复世界为异步模式 + TM 异步 + tick，重置到 _init_carla 的基线
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

        # 销毁前先取消所有车辆的自动驾驶，让 TrafficManager 释放托管，避免销毁 TM 车辆时原生崩溃
        for _a in actors:
            try:
                if _a is not None and getattr(_a, "type_id", "").startswith("vehicle."):
                    _a.set_autopilot(False)
            except Exception:
                pass

        # 销毁本实验生成的所有 actors（不销毁会导致传感器泄漏）
        for _a in actors:
            try:
                if _a is not None and _a.is_alive:
                    _a.destroy()
            except Exception:
                pass
            with _lock:
                _managed_actors.discard(_a.id)

        _EXP04_RUNNING = False
        if _EXP04_ABORT:
            _push_to_sse({"experiment": {"id": 4, "status": "stopped"}})


# --- 语义 LiDAR 对象标签 → 中文名称（CARLA SemanticTag，仅覆盖常见类别）---
_LIDAR_TAG_NAMES = {
    0: "未标注", 1: "静态物", 2: "行人", 3: "地面", 4: "道路",
    5: "车道线", 6: "人行道", 7: "植被", 8: "护栏", 9: "建筑",
    10: "墙体", 11: "立杆", 12: "交通灯", 13: "交通标志", 14: "车辆",
    15: "行人", 16: "摩托车", 17: "车轮", 18: "隔离墩",
}


def _lidar_tag_name(tag):
    return _LIDAR_TAG_NAMES.get(tag, "未知(tag={})".format(tag))


@app.route("/experiment/4/start", methods=["POST"])
def experiment_4_start():
    global _EXP04_THREAD, _EXP04_RUNNING, _EXP04_ABORT
    with _EXP04_LOCK:
        if _EXP04_THREAD is not None and _EXP04_THREAD.is_alive():
            if not _EXP04_ABORT:
                # 正在正常运行：直接拒绝。重试/双击产生的重复 start 在此被挡，
                # 绝不 kill-重启实验
                return jsonify({"status": "error", "message": "实验4 已在运行"}), 409
            # 正在停止/收尾：等旧线程完整退出后接管（保留「停止→快速重启」体验）
            _exp_log("检测到实验4正在停止，等待旧线程收尾..")
            _EXP04_THREAD.join(timeout=15.0)
            if _EXP04_THREAD.is_alive():
                return jsonify({"status": "error", "message": "实验4 旧线程仍在收尾，请稍后重试"}), 409
        # RUNNING/ABORT 在持锁的 handler 内提前置位：
        # ① 关闭 /cleanup、/preview/start 守卫在「start 已返回、线程体尚未执行」间的穿透窗口；
        # ② 消除「新线程刚起、ABORT 仍残留旧 stop 的 true」导致后续 start 误判收尾中的窗口
        _EXP04_RUNNING = True
        _EXP04_ABORT = False
        args = request.get_json(silent=True) or {}
        try:
            _EXP04_THREAD = threading.Thread(target=_run_exp04, args=(args,), daemon=True)
            _EXP04_THREAD.start()
        except Exception as exc:
            _EXP04_RUNNING = False
            return jsonify({"status": "error", "message": f"实验4 线程启动失败: {exc}"}), 500
    return jsonify({"status": "ok", "experiment_id": 4, "message": "实验4 已启动"})


@app.route("/experiment/4/stop", methods=["POST"])
def experiment_4_stop():
    global _EXP04_ABORT
    _EXP04_ABORT = True
    return jsonify({"status": "ok", "message": "实验4 停止请求已发送"})

