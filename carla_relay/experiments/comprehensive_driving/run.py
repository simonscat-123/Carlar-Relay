"""综合驾驶（下）：闭环自动驾驶编排 · 前端关卡 comprehensive-driving（API 实验ID 10）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。

算法主体已按工业界分层架构拆分至本实验包（comprehensive_driving/）内的
真实模块（可独立 import，经构造函数注入依赖，层间数据经帧契约显式流动）：

    frames.py        层间帧数据契约（LocFrame/PercFrame/...）
    context.py       装配上下文（Exp10Context，调试检视用）
    reference.py     参考线 / Frenet / 可行驶域 / 停止线绑定
    sensor.py        驱动层：传感器装配 + 碰撞记录
    localization.py  定位层：GNSS/INS 互补滤波
    perception.py    感知层：bbox 相机感知 / 真值扫描 / 信号灯
    prediction.py    预测层：恒速外推
    planner.py       规划层：FSM 决策 + 时空联合规划
    control.py       控制层：Pure Pursuit + PI 纵向控制
    viz.py           可视化：bbox 渲染 / 语义帧 / SSE 组装
    actors.py        实验台具：障碍生成 / 行人 / 信号灯时长

本文件只做三件事：
    1) 装配：解析参数 → 生成主车 → 挂载传感器 → GRP 全局路线规划 → 构造各层实例；
    2) 主循环编排：每 tick 按数据流调用各层（tick → 定位 → 感知 → 规划 → 控制）；
    3) SSE 推送 / 评分报告 / 顶层 API 路由（start/stop/params/spawn_obstacle）。
"""
# ── 分层模块（真实 import；server/ 已由引导壳注入 sys.path）──
from carla_relay.experiments.comprehensive_driving.context import Exp10Context
from carla_relay.experiments.comprehensive_driving.reference import ReferenceLine
from carla_relay.experiments.comprehensive_driving.sensor import SensorRig
from carla_relay.experiments.comprehensive_driving.localization import Localizer
from carla_relay.experiments.comprehensive_driving.perception import Perceiver
from carla_relay.experiments.comprehensive_driving.perception_v2 import PerceiverV2
from carla_relay.experiments.comprehensive_driving.prediction import ObstaclePredictor
from carla_relay.experiments.comprehensive_driving.planner import (
    TrajectoryPlanner, RED_MARGIN, MAX_DECEL,
)
from carla_relay.experiments.comprehensive_driving.simple_planner import (
    SimplePlanner, AVOID_MARGIN,
)
from carla_relay.experiments.comprehensive_driving.control import VehicleController
from carla_relay.experiments.comprehensive_driving.control_v2 import VehicleControllerV2
from carla_relay.experiments.comprehensive_driving.viz import (
    render_bbox_overlay, render_semantic_frame, build_sse_payload,
    overlay_3d_boxes, build_gap_viz,
)
from carla_relay.experiments.comprehensive_driving.actors import (
    spawn_obstacle_ahead, refresh_route_obstacles, clear_obstacles,
    spawn_crossing_pedestrian, set_traffic_light_timing,
)

# ── 实验10 共享全局状态（命名空间片段级；/route/plan 等跨片段读写）──
_EXP10_THREAD = None                 # 实验线程句柄
_EXP10_RUNNING = False               # 运行中标志（start/stop/spawn 守卫）
_EXP10_ABORT = False                 # 停止请求（主循环每 tick 检查）
_EXP10_VEHICLE_ID = None             # 主车 actor id（spawn_obstacle 等用）
_EXP10_ROUTE = []                    # 当前路线点列（世界坐标）
_EXP10_ROUTE_SAME_TOL = 5.0          # 起终点重规划容差（m）：小于视为同一路线
_EXP10_LAST_PLAN = None              # 上次规划起终点（{"start": (x,y), "end": (x,y)}）
_EXP10_PLANNED_OBSTACLES = []        # 规划期选定的路线障碍物位置（/route/plan 写入）
_EXP10_PLANNED_CHANGED = False       # 重新规划标志：下次运行清旧障碍并重新生成
_EXP10_OBSTACLE_ACTORS = []          # 已生成障碍物登记 [{"id", "pos"}]
_EXP10_SPAWN_REQ = None              # 前端"生成障碍物"请求（主循环线程内消费）
_EXP10_CTRL = {}                     # 运行中实时可调参数（前端滑杆）
_EXP10_LOCK = threading.Lock()       # RUNNING/ABORT 状态锁
_EXP10_CTRL_LOCK = threading.Lock()  # _EXP10_CTRL 读写锁
_EXP10_SPAWN_LOCK = threading.Lock() # _EXP10_SPAWN_REQ 读写锁


def _run_exp10(args):
    """闭环自动驾驶实验线程（编排层）：
    生成车辆 → 挂载传感器 → 全局路线规划 → 分层闭环主循环 → SSE 推送 → 评分报告
    """
    global _EXP10_RUNNING, _EXP10_ABORT, _EXP_CURRENT_ID, _EXP10_VEHICLE_ID, _EXP10_ROUTE
    global _EXP10_SPAWN_REQ
    global _stream_vehicle, _stream_camera, _stream_semantic, _stream_bird, _stream_bbox
    global _EXP10_PLANNED_CHANGED, _EXP10_OBSTACLE_ACTORS

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
    # alpha 互补增益不再作为前端参数：改为由 IMU/GNSS 噪声自适应（噪声小的一方信任更高），
    # 每帧随噪声波动，见 localization.py 顶部 NOISE_JITTER/ALPHA_MIN/ALPHA_MAX/ALPHA_IDLE。
    _exp_log(f"本次参数: gnssσ={gnss_noise:.2f} insσ={ins_noise:.2f} "
             f"(alpha 自适应) "
             f"target={target_speed:.1f}m/s lookahead={lookahead:.1f}m kp={kp_steer:.2f}")
    # 感知闭环：开启后障碍物距离由 bbox 相机（实例+语义分割）单目估计，
    # 不再查询世界真值；关闭则沿用真值扫描（两种模式可运行中实时切换对比）
    perception_mode = bool(args.get("perception", False))
    _exp_log(f"感知闭环: {'开（bbox 相机单目测距）' if perception_mode else '关（世界真值）'}")
    # 激进驾驶模式：必经车道被障碍堵死且无同向邻道可绕（单车道/邻接链断裂）时，
    # 允许借对向车道绕行（逐点地图级校验仍限 Driving 路面，借道期间限速）。
    # 默认关闭；运行中可经 /experiment/10/params 实时切换
    aggressive_mode = bool(args.get("aggressive", False))
    _exp_log(f"激进驾驶: {'开（允许借对向道绕障）' if aggressive_mode else '关（安全模式）'}")
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
            "aggressive": 1.0 if aggressive_mode else 0.0,
        })
    # 规划器选择：simple=单一几何避障（方案B，构造性生成+地图级校验）；
    # legacy=采样式时空联合规划（默认，保持旧行为）
    planner_kind = str(args.get("planner", "legacy")).lower()
    if planner_kind not in ("simple", "legacy"):
        planner_kind = "legacy"
    _exp_log(f"规划器: {planner_kind}"
             f"{'（单一几何避障）' if planner_kind == 'simple' else '（FSM+采样式规划）'}")
    # 控制器选择：v2=前馈+反馈纵向（无油门基线，停车保持归控制层）；
    # legacy=0.25 基线油门 + 纯前馈制动（保持旧行为，默认）
    controller_kind = str(args.get("controller", "legacy")).lower()
    if controller_kind not in ("v2", "legacy"):
        controller_kind = "legacy"
    _exp_log(f"控制器: {controller_kind}"
             f"{'（前馈+反馈纵向）' if controller_kind == 'v2' else '（基线油门+前馈制动）'}")
    # 感知器（信号灯判定）选择：v2=车道归属+路口committed+越线判定，红/黄
    # 统一停驻点（修：黄灯不停/红灯蠕行闯灯/停在路中间等驶入车道的灯）；
    # legacy=旧判定（保持旧行为，默认）
    perceiver_kind = str(args.get("perceiver", "legacy")).lower()
    if perceiver_kind not in ("v2", "legacy"):
        perceiver_kind = "legacy"
    _exp_log(f"感知器: {perceiver_kind}"
             f"{'（车道归属+红黄统一停驻）' if perceiver_kind == 'v2' else '（旧信号灯判定）'}")
    # 信号灯参数（v2 感知用）：停止余量 / 刹停考虑窗口
    tl_stop_margin = float(args.get("tl_stop_margin", RED_MARGIN))
    tl_brake_window = float(args.get("tl_brake_window", 80.0))
    # 规划/感知可调参数（JSON 覆盖，缺省按各消费层模块头登记默认；构造时注入 params：
    # simple_planner 避障窗 dec_win=25；legacy 规划器决策窗 dec_win=60；感知距离=50）
    exp_params = {
        "dec_win": float(args.get("dec_win",
                                  25.0 if planner_kind == "simple" else 60.0)),
        "perception_range": float(args.get("perception_range", 50.0)),
    }
    gps_failure = bool(args.get("gps_failure", False))
    spawn_pedestrian = bool(args.get("spawn_pedestrian", False))
    sampling_res = float(args.get("sampling_resolution", 2.0))

    # 全局路线规划算法与三类成本惩罚（缺省关闭惩罚，保持原始 A* 行为）
    route_algorithm = str(args.get("algorithm", "astar")).lower()
    if route_algorithm not in ("astar", "dijkstra", "bfs"):
        route_algorithm = "astar"
    lane_change_cost = float(args.get("lane_change_cost", 0.0))
    intersection_cost = float(args.get("intersection_cost", 0.0))
    curvature_gain = float(args.get("curvature_gain", 0.0))

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

    vehicle = None
    rig = None
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
            end_loc_raw = carla.Location(x=float(end_coord.get("x", 0)), y=float(end_coord.get("y", 0)), z=0.0)
            end_tf = _snap(end_loc_raw)
            end_loc = end_tf.location
        else:
            if end_idx >= len(spawn_pts):
                _exp_log(f"生成点索引越界: end={end_idx}")
                _push_to_sse({"experiment": {"id": 10, "status": "error", "message": "spawn idx out of range"}})
                return
            end_tf = spawn_pts[end_idx]
            end_loc = end_tf.location

        # ── 起终点吸附诊断：暴露 road_id/lane_id/yaw，便于排查单行道绕路问题 ──
        try:
            _cm = world.get_map()
            _swp = _cm.get_waypoint(spawn_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            _ewp = _cm.get_waypoint(end_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            _s_snap_d = math.hypot(spawn_tf.location.x - _swp.transform.location.x,
                                   spawn_tf.location.y - _swp.transform.location.y)
            _e_snap_d = math.hypot(end_loc_raw.x - _ewp.transform.location.x,
                                   end_loc_raw.y - _ewp.transform.location.y) if end_coord else 0.0
            _exp_log(
                f"起终点吸附: S raw=({_swp.transform.location.x:.1f},{_swp.transform.location.y:.1f}) "
                f"road={_swp.road_id} lane={_swp.lane_id} sec={_swp.section_id} "
                f"yaw={_swp.transform.rotation.yaw:.1f}° 吸附偏移={_s_snap_d:.2f}m | "
                f"E raw=({end_loc.x:.1f},{end_loc.y:.1f}) "
                f"road={_ewp.road_id} lane={_ewp.lane_id} sec={_ewp.section_id} "
                f"yaw={_ewp.transform.rotation.yaw:.1f}° 吸附偏移={_e_snap_d:.2f}m"
            )
        except Exception as _diag_exc:
            _exp_log(f"起终点吸附诊断失败: {_diag_exc}")

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

        # 2. 挂载传感器（驱动层）
        _exp_log("正在挂载传感器（相机 ×4 / LiDAR / GNSS / IMU）…")
        rig = SensorRig(sensor_callback=_sensor_callback, sensor_refs=_sensor_refs,
                        managed_actors=_managed_actors, lock=_lock, log=_exp_log)
        cam, inst, sem, bird, lidar, gnss, imu, col = rig.spawn(world, vehicle, _push_to_sse)
        _stream_camera = cam.id
        _stream_bbox = inst.id
        _stream_semantic = sem.id
        _stream_bird = bird.id

        # 3.5 只有「重新规划后运行」才清空世界中遗留的车辆/行人（含上次实验残留，重启后仍有效），
        #     并保留本车 ego；反之（停止后调参再启动、未重新规划）则沿用世界已有障碍，不清空。
        need_fresh = _EXP10_PLANNED_CHANGED
        _EXP10_PLANNED_CHANGED = False  # 消费标志
        if need_fresh:
            _exp_log("正在清理上次实验残留…")
            try:
                clear_obstacles(world, _EXP10_VEHICLE_ID, _exp_log, _managed_actors)
            except Exception as exc:
                _exp_log(f"清除上次障碍物失败: {exc}")

        # 3. 全局路线规划（起点终点沿用上面确定的 start_loc/spawn_tf 与 end_loc）
        start_loc = spawn_tf.location if start_coord else spawn_pts[start_idx].location
        _exp_log("正在规划全局路线（构建路网拓扑，可能需要数秒）…")
        carla_map = world.get_map()
        route_lane_ids = []       # 与 route_wp 平行：各路点所在车道 id（车道级避障判据用）
        try:
            from carla_relay.vendor.grp_loader import make_route_planner, get_grp_source
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
            _exp_log(f"全局路线算法={route_algorithm} 变道惩罚={lane_change_cost} 路口惩罚={intersection_cost} 弯道惩罚={curvature_gain}")
            if get_grp_source() == "official":
                _exp_log("警告：宿主 CARLA 未同步修改版 GlobalRoutePlanner，已退化为官方原版；algorithm 与三类惩罚不生效，仅按最短距离寻路。")
            grp = make_route_planner(carla_map, sampling_res, algorithm=route_algorithm, weight_fn=weight_fn)
            path = grp.trace_route(start_loc, end_loc)
            route_wp = [wp.transform.location for wp, _ in path]
            route_lane_ids = [(wp.road_id, wp.lane_id) for wp, _ in path]
            route_wps = [wp for wp, _ in path]   # 保留 waypoint 对象（车道/邻道查询用）
            _exp_log(f"路线规划完成: {len(route_wp)} waypoints")

            # ── 路径端点诊断：暴露第 1/2/倒数 2/1 个 waypoint 的 road/lane/yaw，
            #    便于判断 GRP 实际是从哪个方向进入/退出端点，排查单向道绕路。──
            if path:
                def _wp_info(wp):
                    return (f"road={wp.road_id} lane={wp.lane_id} "
                            f"sec={wp.section_id} yaw={wp.transform.rotation.yaw:.1f}° "
                            f"loc=({wp.transform.location.x:.1f},{wp.transform.location.y:.1f})")
                n = len(path)
                head = [_wp_info(path[0][0])]
                if n > 1:
                    head.append(_wp_info(path[1][0]))
                tail = [_wp_info(path[-1][0])]
                if n > 1:
                    tail.append(_wp_info(path[-2][0]))
                _exp_log(f"路径头: { ' ; '.join(head) }")
                _exp_log(f"路径尾: { ' ; '.join(tail) }")
                # 路径上 road/lane 切换统计：出现次数 >1 的 road 表示存在折返/绕路
                from collections import Counter
                _rd = Counter(wp.road_id for wp, _ in path)
                _repeat = sorted(((r, c) for r, c in _rd.items() if c > 1), key=lambda x: -x[1])[:5]
                if _repeat:
                    _exp_log(f"路径 road 出现次数(>1): {_repeat}")
        except Exception as exc:
            _exp_log(f"路线规划失败({exc})，使用直线插值")
            route_wp = [start_loc, end_loc]
            route_wps = []
        route_lane_ids += [None] * (len(route_wp) - len(route_lane_ids))
        route_wps += [None] * (len(route_wp) - len(route_wps))
        _EXP10_ROUTE = list(route_wp)

        # ── 参考线层：弧长表 + Frenet 变换 + 可行驶域（时空联合规划的基础设施）──
        reference = ReferenceLine(carla_map, route_wp, route_wps, route_lane_ids)
        s_tab = reference.s_tab
        tl_bindings = reference.bind_traffic_lights(world, _exp_log)

        # 4. 横穿行人
        if spawn_pedestrian:
            spawn_crossing_pedestrian(world, vehicle, _EXP10_PEDI, _managed_actors, _lock, _exp_log)

        # 4.5 障碍物与路线绑定：新路线重建 / 同一路线沿用并补齐缺失
        _EXP10_OBSTACLE_ACTORS = refresh_route_obstacles(
            world, need_fresh, _EXP10_PLANNED_OBSTACLES, _EXP10_OBSTACLE_ACTORS,
            _exp_log, _managed_actors, _lock)

        # 5. 统一设置全图信号灯时长，压缩等待、避免超长红灯卡停车辆
        set_traffic_light_timing(world, _exp_log)

        _start_stream_thread()
        # 同步模式下沉稳车辆 + 让 GNSS/IMU 帧先到达
        _exp_log("稳定预热中…")
        for _ in range(20):
            if _EXP10_ABORT:
                break
            world.tick()

        # 6. 分层装配（各层实例构造 + 装配上下文挂载）
        _push_to_sse({"experiment": {"id": 10, "status": "running", "route_len": len(route_wp)}})

        _ego_bb = vehicle.bounding_box.extent
        ego_half_w = float(_ego_bb.y)    # 自车半宽（碰撞检查/走廊判据用）
        ego_half_len = float(_ego_bb.x)  # 自车半长（碰撞检查含障碍长度，修"量到中心"缺陷）

        # ── 规划调试日志系统：逐帧决策细节写入文件（排障用）──
        # SSE 实验日志只推关键事件（模式切换/异常/结果），高频细节全部落盘：
        # 每帧一行（自车状态/信号灯/障碍/候选与代价/拒绝统计/控制输出），
        # 外加事件行（最优解切换/信号灯变化/回退兜底）。文件路径启动时推给前端。
        import tempfile as _tempfile
        try:
            _plan_log_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "logs")
        except NameError:
            _plan_log_dir = os.path.join(_tempfile.gettempdir(), "carla_exp10_logs")
        os.makedirs(_plan_log_dir, exist_ok=True)
        _plan_log_path = os.path.join(
            _plan_log_dir, f"exp10_plan_{time.strftime('%Y%m%d_%H%M%S')}.log")
        _plan_log_f = open(_plan_log_path, "w", encoding="utf-8", buffering=1)

        def _plan_log(msg):
            try:
                _plan_log_f.write(
                    f"[{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d}] {msg}\n")
            except Exception:
                pass

        def _both_log(msg):
            """诊断日志：同时落地到规划日志文件 + SSE 实时日志面板。"""
            _plan_log(msg)
            try:
                _exp_log(msg)
            except Exception:
                pass

        _plan_log(f"=== 实验10 规划调试日志 start ===")
        _plan_log(f"参数: target={target_speed} lookahead={lookahead} kp={kp_steer} "
                  f"safe_dist={safe_dist} brake_force={brake_force} "
                  f"perception={'on' if perception_mode else 'off'} "
                  f"route={len(route_wp)}wp 总弧长={s_tab[-1]:.0f}m "
                  f"tl_bindings={len(tl_bindings)}")
        _exp_log(f"规划调试日志: {_plan_log_path}")

        # ── 各层实例（依赖经构造函数显式注入；层间数据经帧契约流动）──
        localizer = Localizer(gnss_noise, ins_noise, vehicle.get_location())
        predictor = ObstaclePredictor()
        if perceiver_kind == "v2":
            # v2 感知：自带绑定（含车道归属），tl_stop_margin/窗口参数化
            perceiver = PerceiverV2(reference, world, carla_map, _exp_log,
                                    _dynamic_class, tl_stop_margin,
                                    tl_brake_window, _TL_STATE_MAP, _plan_log,
                                    route_lane_ids, params=exp_params)
        else:
            perceiver = Perceiver(reference, world, carla_map, _exp_log,
                                  _dynamic_class, RED_MARGIN, tl_bindings,
                                  _TL_STATE_MAP, _plan_log, params=exp_params)
        if planner_kind == "simple":
            planner = SimplePlanner(reference, predictor, _exp_log, _plan_log,
                                    ego_half_w, ego_half_len, params=exp_params)
        else:
            planner = TrajectoryPlanner(reference, predictor, _exp_log, _plan_log,
                                        ego_half_w, ego_half_len, params=exp_params)
        # 越障横向余量（JSON: avoid_margin 可覆盖）：simple 规划器从注入的
        # exp_params 读取；legacy 规划器无该量 → 回退模块默认，用于可视化同源。
        _avoid_margin = getattr(planner, "avoid_margin", AVOID_MARGIN)
        if controller_kind == "v2":
            controller = VehicleControllerV2(kp_steer, lookahead, steer_delay,
                                             brake_force, MAX_DECEL)
        else:
            controller = VehicleController(kp_steer, lookahead, steer_delay,
                                           brake_force, MAX_DECEL)

        # 装配上下文（调试时可整体检视各层实例与装配产物）
        ctx = Exp10Context()
        ctx.log = _exp_log
        ctx.push = _push_to_sse
        ctx.sensor_frames = _sensor_frames
        ctx.sensor_frame_num = _sensor_frame_num
        ctx.instance_raw = _instance_raw
        ctx.semantic_raw = _semantic_raw
        ctx.sensor_refs = _sensor_refs
        ctx.managed_actors = _managed_actors
        ctx.lock = _lock
        ctx.sensor_callback = _sensor_callback
        ctx.dynamic_class = _dynamic_class
        ctx.tl_state_map = _TL_STATE_MAP
        ctx.client = client
        ctx.world = world
        ctx.vehicle = vehicle
        ctx.carla_map = carla_map
        ctx.sensors = (cam, inst, sem, bird, lidar, gnss, imu, col)
        ctx.reference = reference
        ctx.tl_bindings = getattr(perceiver, "bindings", tl_bindings)
        ctx.rig = rig
        ctx.localizer = localizer
        ctx.perceiver = perceiver
        ctx.predictor = predictor
        ctx.planner = planner
        ctx.controller = controller
        ctx.params = {
            "duration": duration, "target_speed": target_speed, "lookahead": lookahead,
            "kp_steer": kp_steer, "steer_delay": steer_delay, "brake_force": brake_force,
            "safe_dist": safe_dist, "gnss_noise": gnss_noise, "ins_noise": ins_noise,
            "perception": perception_mode, "aggressive": aggressive_mode,
            "gps_failure": gps_failure, "sampling_res": sampling_res,
            "dec_win": exp_params["dec_win"],
            "perception_range": exp_params["perception_range"],
            "planner": planner_kind,
            "controller": controller_kind,
            "perceiver": perceiver_kind,
        }

        # 7. 主循环（编排：tick → 定位 → 感知 → 规划 → 控制 → 可视化/推送）
        t0 = time.time()
        wp_idx = 0
        arrived = False
        _bbox_diag = {"ok": 0, "skip": 0, "err": 0}  # bbox 帧渲染诊断计数
        prev_gt = None                # 上一帧真值位置（里程统计用）
        traveled = 0.0                # 累计行驶里程（m）
        perception_used = False       # 本次运行是否启用过感知闭环（报告用）

        import collections
        loc_err_history = collections.deque(maxlen=240)
        speed_history = collections.deque(maxlen=240)
        cte_history = collections.deque(maxlen=240)
        _dbg_frame = 0  # 诊断日志计数器（每 20 帧≈1s 输出一次）
        _pred_seen = set()  # 预测日志去重：记录已出现过的障碍物 key（首次识别即记录）

        while not _EXP10_ABORT:
            world.tick()  # 同步模式：推进一帧，车辆据此移动

            # 处理前端"生成障碍物"请求：必须在主循环线程内操作 CARLA 客户端，
            # 否则与 world.tick() 跨线程并发调用会导致协议错乱、CARLA 崩溃
            with _EXP10_SPAWN_LOCK:
                _sreq = _EXP10_SPAWN_REQ
            if _sreq is not None and not _sreq["done"].is_set():
                try:
                    _sreq["result"] = spawn_obstacle_ahead(
                        world, vehicle, route_wp, _sreq["distance"],
                        _managed_actors, _lock)
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
                aggressive_on = float(_EXP10_CTRL.get("aggressive", 0.0)) > 0.5
            controller.update_params(kp_steer=kp_steer, lookahead=lookahead,
                                     steer_delay=steer_delay, brake_force=brake_force)

            # 真值（仅供误差评估与观测合成）
            ego_tf = vehicle.get_transform()
            gt_loc = ego_tf.location
            gt_yaw = ego_tf.rotation.yaw
            if prev_gt is not None:
                traveled += prev_gt.distance(gt_loc)
            prev_gt = carla.Location(x=gt_loc.x, y=gt_loc.y, z=gt_loc.z)
            perception_used = perception_used or perception_on

            # ── 定位层：GNSS + 互补滤波（输出 LocFrame：融合位姿 + 诊断量）──
            gnss_data = None
            if gnss.id in _sensor_frames:
                try:
                    gnss_data = json.loads(_sensor_frames[gnss.id].decode())
                except Exception:
                    pass
            loc = localizer.step(gt_loc, gt_yaw, vehicle.get_velocity(),
                                 gnss_data is not None, gps_failure)
            loc_err_history.append(round(loc.loc_err, 3))

            # ── 感知层：障碍统一扫描（全量保留不预筛本道，输出 Frenet 障碍列表）──
            _scan_j0 = max(0, wp_idx - 5)
            _scan_j1 = min(len(route_wp), wp_idx + int(70.0 / max(0.5, sampling_res)) + 10)
            perc = perceiver.step(perception_on, vehicle, ego_tf, loc.fused_loc,
                                  _scan_j0, _scan_j1, inst, sem,
                                  _instance_raw, _semantic_raw)

            # ── 预测层日志：识别到障碍物即预测并记录（id/类别/候选点数/概率最大轨迹）──
            # 恒速模型为确定性单轨迹：候选轨迹点=时间网格采样点(0.25~4s)共16点，
            # 概率最大的轨迹即该名义轨迹本身(prob≈1.0)。首次出现立即记录，之后每10帧一次。
            for _o in perc.obstacles:
                _pred_key = (_o["id"] if _o["id"] is not None
                             else (_o["cls"], round(_o["s"], 1), round(_o["l"], 1)))
                _new_obs = _pred_key not in _pred_seen
                _pred_seen.add(_pred_key)
                if _new_obs or _dbg_frame % 10 == 0:
                    _psum = predictor.predict(_o)
                    _b = _psum["best"]
                    _cat = _dynamic_class(_o["cls"])
                    if abs(_b["v_s"]) < 0.1:
                        _shape = f"静止({_b['t']:.1f}s内s≈{_b['s_start']:.1f}m)"
                    else:
                        _shape = (f"沿参考线匀速直行 l={_o['l']:+.1f}m "
                                  f"{_b['dist']:+.1f}m@{_b['t']:.1f}s")
                    _plan_log(
                        f"PRED obs={_o['id'] if _o['id'] is not None else '-'} "
                        f"cat={_cat} cls={_o['cls']} "
                        f"pos=(s{_o['s']:.1f},l{_o['l']:+.1f}) "
                        f"size={2 * _o['half_len']:.1f}x{2 * _o['half_w']:.1f}m "
                        f"cand={_psum['n_cand']}pt "
                        f"best(prob={_b['prob']:.2f}) s={_b['s_start']:.1f}→{_b['s_end']:.1f} "
                        f"{_shape}")

            # bbox 渲染：扫描之后立即用本 tick 检测结果 + 本 tick 相机帧渲染
            # （避开尾部延迟让 bbox 赶上 SSE 采样；检测框与画面同步）。
            # 返回本 tick 的 bbox 帧，交 overlay 叠 3D 后单次写回，消除闪烁。
            _inst_jpeg = render_bbox_overlay(inst, rig.rgb_raw, _instance_raw,
                                             perception_on, perc.perceived,
                                             perc.bbox_cands, _sensor_frames,
                                             _sensor_frame_num, _bbox_diag, _exp_log)

            # 目标识别相机 + 鸟瞰相机：叠加自车/障碍的 3D 包围框（真实框实线、
            # 带余量框虚线，余量=SEP_MIN/COLL_S）。纯可视化，不改决策逻辑。
            _viz3d = overlay_3d_boxes(vehicle=vehicle, fused_loc=loc.fused_loc,
                                      fused_yaw_deg=loc.fused_yaw_deg,
                                      obstacles=perc.obstacles, reference=reference,
                                      inst=inst, bird=bird, sensor_frames=_sensor_frames,
                                      inst_frame=_inst_jpeg, sensor_frame_num=_sensor_frame_num,
                                      avoid_margin=_avoid_margin,
                                      log=_both_log)

            # 车速 + Frenet 位姿（先于红绿灯判定与决策规划，供同帧使用）
            vel = vehicle.get_velocity()
            spd = math.sqrt(vel.x ** 2 + vel.y ** 2)
            ego_s, ego_l, ego_tx, ego_ty = reference.frenet(
                loc.fused_loc.x, loc.fused_loc.y, _scan_j0, _scan_j1, track=True)
            v_long = max(0.0, vel.x * ego_tx + vel.y * ego_ty)   # 纵向车速（沿参考线）

            # ── 感知层：信号灯状态（停止线绑定 + s 判定，输出 TlFrame）──
            # v2：车道归属 + 路口 committed（自车已在路口内则穿行到底）
            if perceiver_kind == "v2":
                _wp_ego = (route_wps[wp_idx]
                           if wp_idx < len(route_wps) else None)
                tl = perceiver.read_traffic_light(
                    ego_s, wp_idx,
                    _wp_ego is not None and _wp_ego.is_junction)
            else:
                tl = perceiver.read_traffic_light(ego_s)

            # ── 规划层：决策 + 时空联合规划（输出 PlanOutput：轨迹 + 期望 + 可视化状态）──
            # 航向-参考线夹角（日志增强#5）：自车航向 vs 参考线切向。拐点重关联
            # 处该角会瞬间跳变（配合 FRM 的 yaw_e 字段暴露坐标系不连续）。
            # CARLA yaw 为度、左手系：atan2 交叉项取 (tx·sinθ − ty·cosθ)
            _yaw_rad = math.radians(loc.fused_yaw_deg)
            _yaw_err = math.degrees(math.atan2(
                ego_tx * math.sin(_yaw_rad) - ego_ty * math.cos(_yaw_rad),
                ego_tx * math.cos(_yaw_rad) + ego_ty * math.sin(_yaw_rad)))
            plan = planner.step(
                t=t, spd=spd, v_long=v_long, ego_s=ego_s, ego_l=ego_l,
                obstacles=perc.obstacles, tl=tl, aggressive_on=aggressive_on,
                target_speed=target_speed, safe_dist=safe_dist, wp_idx=wp_idx,
                prev_thr=controller.prev_thr, prev_brk=controller.prev_brk,
                yaw_err_deg=_yaw_err)

            # ── 控制层：纵向 PI + 横向 Pure Pursuit（输出 CtrlOutput：转向/油门/刹车）──
            ctrl = controller.step(
                desired=plan.desired, a_need=plan.a_need, spd=spd,
                loc=loc.fused_loc, yaw_rad=math.radians(loc.fused_yaw_deg),
                plan_traj=plan.plan_traj, route_wp=route_wp, wp_idx=wp_idx)
            wp_idx = ctrl.wp_idx

            # ── 横向链路诊断（方向C）：命令 l_t 是否写进轨迹、控制有没有在跟 ──
            # 取「距自车 ≥ lookahead 的第一个规划轨迹点」反投影到 Frenet，得到该点被
            # 命令的横向 cmd_l。与实姿 ego_l 对比即可定位断点：
            #   cmd_l≈+2.2 而 ego_l≈0.5 → 轨迹带了偏移但执行没跟（控制/执行侧）；
            #   cmd_l≈0（尽管 intent_l=+2.2）→ 轨迹本身没写进偏移（规划侧）。
            # 仅在绕行/偏移目标非零时逐帧写规划调试日志，巡航不刷屏。
            _cmd_l = None
            _cmd_s = None
            if plan.avoiding or abs(plan.intent_l) > 1e-6:
                for _px, _py in plan.plan_traj:
                    if math.hypot(_px - loc.fused_loc.x, _py - loc.fused_loc.y) >= lookahead:
                        try:
                            _cmd_s, _cmd_l, *_ = reference.frenet(
                                _px, _py, _scan_j0, _scan_j1, track=True)
                        except Exception:
                            _cmd_l = None
                        break
                _plan_log(f"LAT t={t:.1f} intent_l={plan.intent_l:+.2f} "
                          + (f"cmd_l={_cmd_l:+.2f}@s{_cmd_s:.0f} ego_l={ego_l:+.2f} "
                             f"steer={ctrl.steer:.2f} raw={ctrl.raw_steer:.2f} "
                             f"cte={ctrl.cte:.2f} spd={spd:.2f}"
                             if _cmd_l is not None else "cmd_l=NA"))

            # 应用控制
            vehicle.apply_control(carla.VehicleControl(
                throttle=float(ctrl.throttle),
                steer=float(ctrl.steer),
                brake=float(ctrl.brake),
            ))
            speed_history.append(round(spd, 2))
            cte_history.append(round(ctrl.cte, 3))

            # ── 诊断日志（每 10 帧≈0.5s 输出一次，便于快速定位转向/控制问题）──
            _dbg_frame += 1
            if _dbg_frame % 10 == 0:
                _exp_log(
                    f"[t={t:.1f}s] spd={spd:.2f} des={plan.desired:.1f} thr={ctrl.throttle:.2f} brk={ctrl.brake:.2f} "
                    f"out_steer={ctrl.steer:.2f} raw_steer={ctrl.raw_steer:.2f} alpha={ctrl.hdng_alpha:+.3f} "
                    f"look={ctrl.look_dist:.1f} cte={ctrl.cte:.2f} err={loc.loc_err:.2f} "
                    f"yaw(gt)={gt_yaw:.1f} yaw(F)={loc.fused_yaw_deg:.1f} gyaw={loc.gnss_yaw_deg:.1f} "
                    f"gyro={loc.gyro_w:.1f}°/s yaw_a={loc.yaw_a:.3f} "
                    f"fused=({loc.fused_loc.x:.1f},{loc.fused_loc.y:.1f}) p={ego_tf.rotation.pitch:.0f} r={ego_tf.rotation.roll:.0f} "
                    f"obs={plan.front_obstacle if plan.front_obstacle != float('inf') else 'inf'}({plan.front_obs_src}) "
                    f"tl={tl.state}/{tl.dist:.0f}m wp={wp_idx}/{len(route_wp)} "
                    f"fsm={plan.fsm_state} avd={'L' if plan.avoid_side > 0 else 'R' if plan.avoid_side < 0 else '-'} "
                    f"s={ego_s:.0f} l={ego_l:+.1f} "
                    f"loc=({gt_loc.x:.1f},{gt_loc.y:.1f})"
                )

            # 到达判断
            dist_to_end = gt_loc.distance(end_loc)
            arrived = dist_to_end < 5.0
            if arrived:
                _exp_log("到达终点")
                break

            # 语义分割帧：原始 CityScapes 标签 → 彩色图写入帧缓存（SSE 推流用）
            render_semantic_frame(sem, _semantic_raw, _sensor_frames, _sensor_frame_num,
                                  _label_semantic_classes, _colors_from_labels)

            # 推送实验数据（可视化层组装：鸟瞰车道/参考线/预测轨迹 + 状态帧）
            # 障碍两侧可通行间隙带（口径与 simple_planner 一致 → 所见即决策）。
            # 纯可视化：任何异常不得影响驾驶主循环。
            try:
                _gap_viz = build_gap_viz(reference, perc.obstacles, ego_half_w,
                                     ego_half_len, ego_s=ego_s,
                                     aggressive_on=aggressive_on, log=_both_log,
                                     avoid_margin=_avoid_margin)
            except Exception:
                _gap_viz = None
            _payload = build_sse_payload(
                t=t, wp_idx=wp_idx, route_wp=route_wp, route_lane_ids=route_lane_ids,
                sampling_res=sampling_res, fused_loc=loc.fused_loc,
                fused_yaw_deg=loc.fused_yaw_deg, spd=spd, desired=plan.desired,
                steer=ctrl.steer, throttle=ctrl.throttle, brake=ctrl.brake,
                cte=ctrl.cte, loc_err=loc.loc_err, front_obstacle=plan.front_obstacle,
                arrived=arrived, perception_on=perception_on,
                perceived_count=len(perc.perceived), gt_loc=gt_loc, gt_yaw=gt_yaw,
                ngx=loc.ngx, ngy=loc.ngy, plan=plan, obs_list=perc.obs_list,
                planned_obstacles=_EXP10_PLANNED_OBSTACLES, tl=tl, carla_map=carla_map,
                viz3d=_viz3d, gap_viz=_gap_viz)
            # 调试：与 bbox3d/bird3d 同通道透出 2D 检测框归一化坐标，供前端对比屏幕坐标
            _payload["bbox2d"] = _bbox_diag.get("uv2d", [])
            # 透出当前帧自适应 alpha（GNSS 权重）供状态栏动态展示
            _payload["alpha"] = round(loc.alpha, 3)
            _push_to_sse(_payload)

            # 同步模式下 tick 已按固定时间步推进并阻塞至该帧完成，无需额外 sleep
            # time.sleep(0.05)  # 20fps

        # 结束：汇总评分报告（到达 / 里程 / 定位与横向误差 / 碰撞事故）
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        avg_loc_err = sum(loc_err_history) / max(1, len(loc_err_history))
        max_loc_err = max(loc_err_history) if loc_err_history else 0.0
        abs_cte = [abs(c) for c in cte_history]
        avg_abs_cte = (sum(abs_cte) / len(abs_cte)) if abs_cte else 0.0
        max_abs_cte = max(abs_cte) if abs_cte else 0.0
        incidents = rig.collision.snapshot()
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
        # 清理传感器（驱动层）：先停止监听（断开流），再销毁
        _stream_bird = None
        _stream_camera = None
        _stream_semantic = None
        _stream_bbox = None
        if rig is not None:
            try:
                rig.cleanup(world)
            except Exception as exc:
                _exp_log(f"传感器清理异常: {exc!r}")
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
        # 激进驾驶开关：运行中实时切换「安全模式 / 借对向道绕障」
        if "aggressive" in data and data["aggressive"] is not None:
            _EXP10_CTRL["aggressive"] = 1.0 if bool(data["aggressive"]) else 0.0
        snapshot = dict(_EXP10_CTRL)
    return jsonify({"status": "ok", "params": snapshot})
