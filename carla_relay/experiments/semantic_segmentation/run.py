"""语义分割 · 前端关卡 semantic-segmentation（API 实验ID 5）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""

# =============================================================================
# 实验 5 · 可调参数区（文件头）
#   天气退化参数集：每档天气对应一批数值，统一驱动各感知通路的退化程度
#     det_sev   : 目标识别退化强度 [0,1] → 远处目标随机漏检概率、检测框抖动幅度
#     sem_noise : 语义分割画面高斯噪声 σ（像素值，0=干净）→ 模拟真 RGB 分割器受扰
#   画面天气本体由 _exp5_weather 写入真实 CARLA 世界（set_weather），不再额外拟真。
# =============================================================================
_EXP5_WEATHER_PROFILE = {
    "ClearNoon":    {"det_sev": 0.00, "sem_noise": 0.0},
    "CloudyNoon":   {"det_sev": 0.12, "sem_noise": 3.0},
    "SoftRainNoon": {"det_sev": 0.35, "sem_noise": 8.0},
    "HardRainNoon": {"det_sev": 0.65, "sem_noise": 16.0},
    "FoggyNoon":    {"det_sev": 0.70, "sem_noise": 12.0},
    "ClearNight":   {"det_sev": 0.60, "sem_noise": 11.0},
}
# =============================================================================
# 实验 5: 视觉与语义分割
# =============================================================================

# 感知融合 → BEV 占据栅格世界 → 基于栅格的规划（真实模块，绝对路径导入）
from carla_relay.experiments.semantic_segmentation.fusion import perceive as _exp5_perceive
from carla_relay.experiments.semantic_segmentation.bev import (
    BevBuilder as _exp5_BevBuilder, _project_pixels as _exp5_project)
from carla_relay.experiments.semantic_segmentation.grid_planner import GridPlanner as _exp5_GridPlanner
from carla_relay.experiments.comprehensive_driving.viz import render_perceived_frame as _exp5_render_perceived

import math as _math5
import random as _random5


def _exp5_weather_prof(key):
    """安全取天气预设；未知 key 按晴天处理。"""
    return _EXP5_WEATHER_PROFILE.get(key, _EXP5_WEATHER_PROFILE["ClearNoon"])


def _exp5_weather(key):
    """按预设 key 返回真实 CARLA WeatherParameters；未知 key 返回 None。

    优先使用 CARLA 内置天气预设常量（carla.WeatherParameters.*），并额外提供
    自定义“大雾”预设。web 端下拉与本地 json 的 weather 字段都走本映射。"""
    if not key:
        return None
    wp = getattr(carla.WeatherParameters, key, None)
    if isinstance(wp, carla.WeatherParameters):
        return wp
    if key == "FoggyNoon":
        return carla.WeatherParameters(
            cloudiness=45.0, fog_density=85.0, fog_distance=10.0,
            precipitation=0.0, precipitation_deposits=0.0, wetness=0.0,
            sun_azimuth_angle=100.0, sun_altitude_angle=50.0,
        )
    return None


def _exp5_apply_weather(world, key):
    """把真实天气写入 CARLA 世界；入参为空或未知时返回 False。"""
    wp = _exp5_weather(str(key or ""))
    if wp is None:
        return False
    try:
        world.set_weather(wp)
        return True
    except Exception as exc:
        _exp_log(f"[weather] 设置天气 {key} 失败: {exc!r}")
        return False


def _exp5_degrade_detection(targets, key, rng):
    """识别降级：按天气 det_sev，对较远目标随机漏检（近处基本保留）。

    返回保留的目标列表；此处只做『丢』不动位置，保证 BEV/决策/障碍表一致
    ——恶劣天看到的正是感知退化后的世界。"""
    prof = _exp5_weather_prof(key)
    sev = float(prof.get("det_sev", 0.0))
    if sev <= 0.01:
        return list(targets)
    out = []
    for t in targets:
        d = t.get("dist", 40.0)
        far = min(1.0, max(0.0, (d - 12.0) / 40.0))   # 12m 内基本保留，越远越易漏
        drop = sev * (0.12 + 0.88 * far)
        if rng.random() > drop:
            out.append(t)
    return out


def _exp5_jitter_box(targets, key, rng):
    """识别抖动：恶劣天按 det_sev 对检测框加少量随机像素偏移，呈现『识别不稳』。"""
    prof = _exp5_weather_prof(key)
    j = int(1 + 7 * float(prof.get("det_sev", 0.0)))
    if j <= 0:
        return list(targets)
    out = []
    for t in targets:
        b = t.get("box")
        if not b:
            out.append(t)
            continue
        x0, y0, x1, y1 = b
        jx, jy = rng.uniform(-j, j), rng.uniform(-j, j)
        out.append({**t, "box": (x0 + jx, y0 + jy, x1 + jx, y1 + jy)})
    return out


def _exp5_sem_noise(rgb, key, rng):
    """语义分割画面退化：按天气 sem_noise 叠加高斯噪声。

    模拟『真实 RGB 语义分割器』在恶劣天下同样受扰（画面变花但结构仍在）。
    rgb: (H,W,3) uint8；σ≤0 时原样返回。"""
    prof = _exp5_weather_prof(key)
    std = float(prof.get("sem_noise", 0.0))
    if std <= 0.0 or rgb is None:
        return rgb
    # numpy 标准正态 × σ（random.Random 无 normal，用 gaussian 也没法整幅，故走 numpy）
    n = (np.random.standard_normal(rgb.shape[:2] + (1,)).astype(np.float32) * std)
    out = rgb.astype(np.float32) + n
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


def _exp5_lane_geoms(world, ego_tf, *, front=60.0, step=1.5, fov_deg=90.0):
    """采样当前车道 + 左右相邻车道的中心线，转自车系（前向x、车右y），返回左右边界。

    返回 [{ "left": [[x,y],...], "right": [[x,y],...] }, ...]，仅保留自车前方 90° 视锥内的点
    （fwd>0 且 |y|<=fwd*tan(45°)），与 BEV 视锥范围一致。若当前车道两侧无同向邻道，
    列表只有当前车道一项。
    """
    try:
        wp0 = world.get_map().get_waypoint(ego_tf.location)
    except Exception:
        return []
    if wp0 is None:
        return []

    # CARLA 世界系 x前 y右 → 自车系 x'=前向, y'=车右：x'=cos·dx+sin·dy, y'=-sin·dx+cos·dy
    yaw = _math5.radians(ego_tf.rotation.yaw)
    c0, s0 = _math5.cos(yaw), _math5.sin(yaw)
    ex, ey = ego_tf.location.x, ego_tf.location.y

    def body(dx, dy):
        return (dx * c0 + dy * s0, -dx * s0 + dy * c0)

    # 前方 90° 视锥裁剪（±45°）
    tan_half = _math5.tan(_math5.radians(fov_deg) / 2.0)

    # 当前车道 + 左/右邻道（同 lane_type 才算正向邻道）
    start_wps = [wp0]
    for side_fn in (wp0.get_left_lane, wp0.get_right_lane):
        try:
            nw = side_fn()
            if nw is not None and nw.lane_id != wp0.lane_id \
                    and nw.lane_type == wp0.lane_type:
                start_wps.append(nw)
        except Exception:
            pass

    geoms = []
    for st in start_wps:
        w, centers, dist, guard = st, [], 0.0, 0
        while dist < front and guard < 120:
            cx_w, cy_w = w.transform.location.x, w.transform.location.y
            fwd, lat = body(cx_w - ex, cy_w - ey)
            if fwd > 0 and abs(lat) <= fwd * tan_half:
                centers.append((fwd, lat))
            nxt = w.next(step)
            if not nxt:
                break
            nw = nxt[0] if isinstance(nxt, (list, tuple)) else nxt
            if nw is None:
                break
            dist += _math5.hypot(nw.transform.location.x - cx_w,
                                 nw.transform.location.y - cy_w)
            w, guard = nw, guard + 1
        if len(centers) < 2:
            continue
        hw = (getattr(st, "lane_width", None) or 3.5) / 2.0
        left, right = [], []
        for i, (bx, by) in enumerate(centers):
            prv = centers[i - 1] if i > 0 else centers[0]
            nxtc = centers[i + 1] if i + 1 < len(centers) else centers[-1]
            tx, ty = nxtc[0] - prv[0], nxtc[1] - prv[1]
            n = _math5.hypot(tx, ty) or 1.0
            # 中心线切向(tx,ty)，左法向=(-ty,tx)（= 屏上"左"），但 y_body=车右 → 屏右
            # 这里 lat 已经是车右，所以"左"边界 = lat 减 hw（更靠车中线为小 lat=车左）
            # 物理上：left/right 是相对车而言
            left.append([round(bx, 2), round(by - hw, 2)])   # 车左：lat 更小
            right.append([round(bx, 2), round(by + hw, 2)])  # 车右：lat 更大
        geoms.append({"left": left, "right": right})
    return geoms


def _exp5_actor_metrics(world, ego, aid, t, vel_state, still_cnt):
    """由真实 actor id 反查完整物理量：pose(全局 xyz/自身 yaw 弧度)、
    size(lwh)、velocity/acceleration(转自车系：前向 vx、左向 vy)、is_static。
    vel_state[id]=(vx_w, vy_w, t)；still_cnt[id] 为低速连续计数。"""
    if aid is None:
        return None
    try:
        actor = world.get_actor(aid)
    except Exception as exc:
        if aid not in _EXP05_MISSED:
            _EXP05_MISSED.add(aid)
            _exp_log(f"[metrics] get_actor 异常 aid={aid}: {exc!r}")
        return None
    if actor is None:
        if aid not in _EXP05_MISSED:
            _EXP05_MISSED.add(aid)
            _exp_log(f"[metrics] get_actor 返回 None aid={aid}")
        return None
    out = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0,
           "length": 0.0, "width": 0.0, "height": 0.0,
           "vx": 0.0, "vy": 0.0, "ax": 0.0, "ay": 0.0, "is_static": False}
    tl = actor.get_transform()
    out["x"], out["y"], out["z"] = tl.location.x, tl.location.y, tl.location.z
    out["yaw"] = _math5.radians(tl.rotation.yaw)
    ext = actor.bounding_box.extent
    out["length"], out["width"], out["height"] = 2.0 * ext.x, 2.0 * ext.y, 2.0 * ext.z
    v = actor.get_velocity()
    if aid not in _EXP05_VLOG:
        _EXP05_VLOG.add(aid)
        _exp_log(f"[metrics] aid={aid} type={actor.type_id} "
                 f"vel=({v.x:.2f},{v.y:.2f}) acti={actor.is_alive}")
    yr = _math5.radians(ego.get_transform().rotation.yaw)
    c, s = _math5.cos(yr), _math5.sin(yr)
    out["vx"] = v.x * c + v.y * s      # 自车系前向
    out["vy"] = -v.x * s + v.y * c     # 自车系右向
    prev = vel_state.get(aid)
    if prev is not None:
        dt = max(1e-3, t - prev[2])
        awx = (v.x - prev[0]) / dt
        awy = (v.y - prev[1]) / dt
        out["ax"] = awx * c + awy * s
        out["ay"] = -awx * s + awy * c
    vel_state[aid] = (v.x, v.y, t)
    if v.x * v.x + v.y * v.y < 0.25:          # 速度模 < 0.5 m/s
        still_cnt[aid] = still_cnt.get(aid, 0) + 1
    else:
        still_cnt[aid] = 0
    out["is_static"] = still_cnt.get(aid, 0) >= 5
    return out


def _exp5_ego_project(ego, fwd, lat):
    """车体系 (fwd 前向, lat 右正) → 全局 (x, y)。右向量 = (sin, -cos)（与综合驾驶一致）。"""
    etf = ego.get_transform()
    yr = _math5.radians(etf.rotation.yaw)
    c, s = _math5.cos(yr), _math5.sin(yr)
    gx = etf.location.x + fwd * c + lat * s
    gy = etf.location.y + fwd * s - lat * c
    return gx, gy


def _exp5_static_by_pos(ego, tg, t, pos_hist):
    """跨帧全局位移判定静态：位移速率 < 0.5 m/s 累计 5 帧 → True。
    不依赖 actor 真值速度，actor 反查失败时仍能正确判定静止。"""
    gx, gy = _exp5_ego_project(ego, tg.get("fwd", 0.0), tg.get("lat", 0.0))
    aid = tg.get("id")
    prev = pos_hist.get(aid)
    if prev is not None:
        px, py, pt, pc = prev
        dt = max(1e-3, t - pt)
        sp = _math5.hypot(gx - px, gy - py) / dt
        cnt = pc + 1 if sp < 0.5 else 0
    else:
        cnt = 0
    pos_hist[aid] = (gx, gy, t, cnt)
    return cnt >= 5


def _exp5_fallback_size(ego, tg):
    """actor 反查失败时，用感知估计补尺寸（长度=2×典型半长，宽度=单目宽度，高度=类别典型）。
    位置由车体系投影到全局；vx/vy/ax/ay 无真值置 None（表格显示 —）。"""
    cls = tg["cls"]
    x, y = _exp5_ego_project(ego, tg.get("fwd", 0.0), tg.get("lat", 0.0))
    return {"x": round(x, 2), "y": round(y, 2), "z": 0.0, "yaw": None,
            "length": round(2.0 * tg.get("half_len", 1.5), 2),
            "width": round(tg.get("width_m", 0.8), 2),
            "height": round(1.8 if cls == "walker" else 1.5, 2),
            "vx": None, "vy": None, "ax": None, "ay": None,
            "is_static": False}


def _exp5_depth_gray_jpeg(raw, h, w, max_range):
    """深度 raw BGRA → 对数灰度 JPEG，> max_range(米) 置黑（depth_range 截断视图）。"""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    r = arr[:, :, 2].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 0].astype(np.float32)
    d = np.clip((r + g * 256.0 + b * 256.0 * 256.0) / (2.0 ** 24 - 1.0) * 1000.0,
                0.1, max_range)
    log_range = np.log1p(max_range) - np.log1p(0.1)
    gray = np.clip(255.0 * (1.0 - (np.log1p(d) - np.log1p(0.1)) / log_range),
                   0, 255).astype(np.uint8)
    img = PIL.Image.fromarray(np.stack([gray, gray, gray], axis=-1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _exp5_sem_truncate_by_depth(rgb, depth_raw, h, w, sem_range):
    """按真实远近对语义画面截断：用与语义相机同位姿、同分辨率深度图的深度做真值，
    深度 > sem_range 或深度无效(天空/远景/缺失)的像素置黑，语义感知范围外不再显示。
    避免旧方案用地面平面逆投影(只依赖像素行)造成的"按上下截断"。"""
    arr = np.frombuffer(depth_raw, dtype=np.uint8).reshape((h, w, 4))
    r = arr[:, :, 2].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 0].astype(np.float32)
    d = (r + g * 256.0 + b * 256.0 * 256.0) / (2.0 ** 24 - 1.0) * 1000.0
    valid = (d > 0.05) & (d <= sem_range)
    return np.where(valid[..., None], rgb, 0)


_EXP05_RUNNING = False
_EXP05_MISSED = set()  # 已诊断过的反查失败 actor id（避免每帧刷屏）
_EXP05_VLOG = set()    # 已打印过原始速度的 actor id
_EXP05_ABORT = False
_EXP05_THREAD = None
_EXP05_LOCK = threading.Lock()


def _run_exp05(args):
    global _EXP05_RUNNING, _EXP05_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle, _stream_semantic
    _EXP_CURRENT_ID = 5
    _EXP05_RUNNING = True
    _EXP05_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验5 (语义分割) 启动")
    _sweep_stale_actors()
    _exp_log("正在配置仿真环境（同步模式 + 交通管理）…")

    duration = float(args.get("duration", 20.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    seed = int(args.get("seed", 7))
    classes = str(args.get("classes", "7"))
    if classes not in _SEMANTIC_PRESETS:
        classes = "7"
    # 感知融合 → BEV 构建参数（task 参数可覆盖）
    perc_range = float(args.get("perception_range", 50.0))
    sem_range = float(args.get("semantic_range", 40.0))
    depth_range = float(args.get("depth_range", 80.0))
    # BEV 栅格边长：自车位于栅格中心，前向可见 = span/2 → 默认 120 保证前向 60m
    bev_span = float(args.get("bev_span", 120.0))
    bev_res = float(args.get("bev_res", 0.5))
    npc_count = int(args.get("npc_count", 0))
    walker_count = int(args.get("walker_count", 6))
    weather_key = str(args.get("weather", "ClearNoon"))  # 天气拟真预设（画面+识别扰动）
    weather_rng = _random5.Random()  # 每轮独立随机源：雨丝/漏检/抖动共用，现象更真实

    cam_fov = 90.0   # 语义/实例相机 FOV（度）
    cam_pitch = -5.0  # 相机安装俯仰（负值向下，供地面逆投影 / 单目测距）
    cam_height = 1.7  # 相机离地高度（m）

    _exp5_rgb_raw = {}  # 前相机原始 BGRA（供检测框叠加）
    _exp5_depth_raw = {}  # 深度相机原始 BGRA（供 depth_range 截断渲染）

    # 感知融合 / BEV 世界 / 基于栅格的规划器（每 tick 复用）
    bev = _exp5_BevBuilder(span=bev_span, res=bev_res, cam_fov=cam_fov,
                           cam_pitch=cam_pitch, cam_height=cam_height,
                           perc_range=perc_range, sem_range=sem_range, subsample=2)
    planner = _exp5_GridPlanner()

    actors = []
    walker_pairs = []  # (walker_actor, controller_actor)，收尾先 stop 再 destroy
    _exp5_vel_state = {}   # 障碍 id -> (vx_w, vy_w, t)，求加速度 / 判静态
    _exp5_still_cnt = {}
    _exp5_pos_hist = {}    # 障碍 id -> (gx, gy, t, cnt)，跨帧全局位移判静态
    try:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=fixed_delta))
        # 真实 CARLA 天气：写作世界后改变所有相机的真实光线；收尾还原原天气。
        _exp5_orig_weather = None
        try:
            _exp5_orig_weather = world.get_weather()
        except Exception:
            _exp5_orig_weather = None
        if _exp5_apply_weather(world, weather_key):
            _exp_log(f"已应用真实世界天气: {weather_key}")
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        # TM 全局调参：车距 / 车速差 / 混合物理（官方 generate_traffic 口径）
        tm.set_global_distance_to_leading_vehicle(2.5)
        try:
            tm.global_percentage_speed_difference(25.0)
        except Exception:
            pass
        # 关闭 hybrid physics：该模式下 NPC 无真实物理，get_velocity() 恒为 0，
        # 会让障碍物表格速度/加速度/静态判定全部失真。改为完整物理模拟。
        try:
            tm.set_hybrid_physics_mode(False)
        except Exception:
            pass

        bp = world.get_blueprint_library().filter("vehicle.*")[0]
        rng = random.Random(seed)
        _exp_log("正在生成主车…")
        vehicle = world.spawn_actor(bp, rng.choice(world.get_map().get_spawn_points()))
        actors.append(vehicle)
        v_id = vehicle.id

        _exp_log("正在挂载相机（RGB / 语义 / 实例）…")
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(cam)
        _sensor_refs[cam.id] = cam
        # 相机回调仅缓存原始 BGRA（并保留 dtype 标记），跳过 core 的 JPEG 编码：
        # 该帧会被 run.py 主循环叠加检测框后重编码回写 _sensor_frames[cam.id]，
        # 与深度帧独占渲染的做法一致，避免每 tick 多余的一次 1280×720 q70 编码。
        def _exp5_cam_cb(d, sid=cam.id):
            _sensor_dtype[sid] = "camera"
            _exp5_rgb_raw[sid] = bytes(d.raw_data)
        cam.listen(_exp5_cam_cb)
        _stream_camera = cam.id
        _stream_vehicle = v_id

        sem_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", "800")
        sem_bp.set_attribute("image_size_y", "600")
        sem_bp.set_attribute("fov", "90")
        sem = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(sem)
        _sensor_refs[sem.id] = sem
        sem.listen(lambda d, sid=sem.id: _sensor_callback(sid, "semantic", d))
        _stream_semantic = sem.id

        ins_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        ins_bp.set_attribute("image_size_x", "800")
        ins_bp.set_attribute("image_size_y", "600")
        ins = world.spawn_actor(ins_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(ins)
        _sensor_refs[ins.id] = ins
        ins.listen(lambda d, sid=ins.id: _sensor_callback(sid, "instance", d))

        # 深度相机（core 侧经 _sensor_dtype=="depth" 自动推送 msg["depth"]）
        dep_bp = world.get_blueprint_library().find("sensor.camera.depth")
        dep_bp.set_attribute("image_size_x", "800")
        dep_bp.set_attribute("image_size_y", "600")
        dep_bp.set_attribute("fov", "90")
        dep = world.spawn_actor(dep_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(dep)
        _sensor_refs[dep.id] = dep
        dep.listen(lambda d, sid=dep.id: _sensor_callback(sid, "depth", d)
                   or _exp5_depth_raw.__setitem__(sid, bytes(d.raw_data)))

        for a in actors:
            with _lock:
                _managed_actors.add(a.id)
        _exp_log(f"车辆+RGB+语义+实例+深度相机已挂载 (id={v_id})")

        # ── 生成 NPC 交通车（官方生成 traffic 的方式：batch 生成 + TM 接管）──
        from carla.command import SpawnActor as _cSpawn, SetAutopilot as _cAuto, \
            FutureActor as _cFuture, DestroyActor as _cDestroy
        npc_list = []
        if npc_count > 0:
            car_bps = sorted(world.get_blueprint_library().filter("vehicle.*"),
                         key=lambda b: b.id)
            if car_bps:
                sp = world.get_map().get_spawn_points()
                avail = [s for s in sp if s.location.distance(vehicle.get_location()) >= 5.0]
                n = min(npc_count, len(avail))
                if n > 0:
                    nrng = random.Random(seed + 100)
                    batch = []
                    for rec in nrng.sample(avail, n):
                        bp = nrng.choice(car_bps)
                        batch.append(_cSpawn(bp, rec).then(_cAuto(_cFuture, True, tm.get_port())))
                    for r in client.apply_batch_sync(batch, True):
                        if not r.error:
                            a = world.get_actor(r.actor_id)
                            if a is not None:
                                actors.append(a)
                                npc_list.append(a)
                    if npc_list:
                        _exp_log(f"已生成 {len(npc_list)}/{npc_count} 辆 NPC 交通车")

        # ── 生成行人（官方 flow：walker 实体 + controller.ai.walker）──
        if walker_count > 0:
            wbps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
            if wbps:
                try:
                    world.set_pedestrians_seed(seed + 300)
                except Exception:
                    pass
                wrng = random.Random(seed + 400)
                wlocs, wspeeds = [], []
                for _ in range(walker_count):
                    loc = world.get_random_location_from_navigation()
                    if loc is None:
                        continue
                    loc.z += 0.3
                    bp = wrng.choice(wbps)
                    wlocs.append(loc)
                    sp = 1.4
                    if bp.has_attribute("speed"):
                        sv = bp.get_attribute("speed").recommended_values
                        if len(sv) > 1 and sv[1]:
                            sp = float(sv[1])
                    wspeeds.append(sp)
                wids = [r.actor_id for r in client.apply_batch_sync(
    [_cSpawn(wrng.choice(wbps), carla.Transform(loc)) for loc in wlocs], True) if not r.error]
                cids = [r.actor_id for r in client.apply_batch_sync(
                    [_cSpawn(world.get_blueprint_library().find("controller.ai.walker"),
                             carla.Transform(), wid) for wid in wids], True) if not r.error]
                world.tick()
                for i in range(min(len(wids), len(cids))):
                    wa = world.get_actor(wids[i])
                    ca = world.get_actor(cids[i])
                    if wa is not None and ca is not None:
                        try:
                            ca.start()
                            ca.go_to_location(world.get_random_location_from_navigation())
                            ca.set_max_speed(wspeeds[i] if i < len(wspeeds) else 1.4)
                        except Exception:
                            pass
                        actors.append(wa)
                        actors.append(ca)
                        walker_pairs.append((wa, ca))
                if walker_pairs:
                    _exp_log(f"已生成 {len(walker_pairs)}/{walker_count} 个行人")

        _exp_log("渲染预热中（首帧着色器编译）…")
        settle_ticks = int(1.0 / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP05_ABORT:
                raise RuntimeError("已中止")
            world.tick()

        vehicle.set_autopilot(True, tm.get_port())
        _exp_log("自动驾驶已启用，开始采集")

        total = int(duration / fixed_delta)
        rows = []
        for i in range(total):
            if _EXP05_ABORT:
                break
            world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds

            # 语义标签图（CityScapes）
            sem_labels = None
            if sem.id in _semantic_raw:
                sem_labels = np.frombuffer(_semantic_raw[sem.id], dtype=np.uint8).reshape((600, 800, 4))[:, :, 2].astype(np.int32)

            # 实例分割：R=语义ID, G=actor低字节, B=actor高字节
            instance = None
            if ins.id in _instance_raw:
                arr = np.frombuffer(_instance_raw[ins.id], dtype=np.uint8).reshape((600, 800, 4))
                sem_ids = arr[:, :, 2].astype(np.int32)
                actor_ids = arr[:, :, 1].astype(np.uint16) + (arr[:, :, 0].astype(np.uint16) << 8)
                instance = (sem_ids, actor_ids)

            # ── 感知融合 → BEV 占据栅格世界 → 基于栅格的规划 ──
            targets = []
            bev_payload = None
            grid_stat = None
            decision = None
            rich = {}   # 障碍 id -> 真实 actor 补齐的 pose/size/运动学
            if ins.id in _instance_raw and sem.id in _semantic_raw:
                try:
                    ih = int(ins.attributes["image_size_y"]); iw = int(ins.attributes["image_size_x"])
                    sh = int(sem.attributes["image_size_y"]); sw = int(sem.attributes["image_size_x"])
                    inst_arr = np.frombuffer(_instance_raw[ins.id], dtype=np.uint8).reshape((ih, iw, 4))
                    sem_arr = np.frombuffer(_semantic_raw[sem.id], dtype=np.uint8).reshape((sh, sw, 4))
                    targets = _exp5_perceive(inst_arr, sem_arr, cam_fov=cam_fov,
                                             cam_pitch=cam_pitch, cam_height=cam_height,
                                             perc_range=perc_range, exclude_ids=(vehicle.id,))
                    # 天气 → 识别降级：恶劣天对远目标随机漏检（近处保留）。
                    # 漏检后的 target 同步喂给 BEV/决策/障碍表 → 感知退化的世界是自洽的。
                    targets = _exp5_degrade_detection(targets, weather_key, weather_rng)
                    # BEV 动态障碍统一口径：目标检测到（融合范围内）就绘制上 BEV，
                    # 只受 bev 栅格自身覆盖半径裁剪；不再受 depth_range 二次门控。
                    # 车道/可行驶等静态结构仍来自语义分割、受 semantic_range 截断。
                    cells, grid_stat, bev_payload = bev.build(sem_labels, targets,
                                                              sem_h=sh, sem_w=sw)
                    decision = planner.decide(cells, res=bev.res)
                    # 前视 RGB：真实世界天气直接作用在该相机帧上 → 叠加抖动后的检测框。
                    # 画面即 CARLA 原图；识别几何仍以漏检后 targets 为准。
                    if cam.id in _exp5_rgb_raw:
                        rcw, rch = int(cam.attributes["image_size_x"]), int(cam.attributes["image_size_y"])
                        _rgb_arr = np.frombuffer(_exp5_rgb_raw[cam.id], dtype=np.uint8).reshape((rch, rcw, 4))
                        _disp = _exp5_jitter_box(targets, weather_key, weather_rng)
                        _sensor_frames[cam.id] = _exp5_render_perceived(_rgb_arr, _disp, iw, ih)
                        _sensor_frame_num[cam.id] = _sensor_frame_num.get(cam.id, 0) + 1
                except Exception as _exp5e:
                    _exp_log(f"感知融合异常: {_exp5e!r}")

            # 障碍物 rich 信息：反查真实 actor 补齐 pose/size/速度/加速度/静态判定
            for _tg in targets:
                _d5 = _exp5_actor_metrics(world, vehicle, _tg.get("id"), t,
                                          _exp5_vel_state, _exp5_still_cnt)
                if _d5 is None:
                    # actor 反查失败：尺寸回退感知估计（不再显示 0）
                    _d5 = _exp5_fallback_size(vehicle, _tg)
                # 静态判定统一用跨帧位移（不依赖可能失败的 actor 真值）
                _d5["is_static"] = _exp5_static_by_pos(vehicle, _tg, t, _exp5_pos_hist)
                rich[_tg.get("id")] = _d5

            # 深度相机画面：按 depth_range 对数灰度 + 截断（超距置黑）
            if dep.id in _exp5_depth_raw:
                try:
                    _sensor_frames[dep.id] = _exp5_depth_gray_jpeg(
                        _exp5_depth_raw[dep.id], 600, 800, depth_range)
                    _sensor_frame_num[dep.id] = _sensor_frame_num.get(dep.id, 0) + 1
                except Exception:
                    pass

            ratios = {}
            if sem_labels is not None:
                labeled = _label_semantic_classes(sem_labels, classes, instance, world)
                ratios = _ratios_from_labels(labeled, classes)
                rgb = _colors_from_labels(labeled, classes)
                # 语义画面退化：按天气 sem_noise 叠加随机噪声（程度不同天气不同）
                rgb = _exp5_sem_noise(rgb, weather_key, weather_rng)
                # 语义画面按真实远近截断：用同位姿深度图置黑超出 semantic_range(含天空/远景)的像素
                if dep.id in _exp5_depth_raw:
                    _sh5, _sw5 = sem_labels.shape
                    rgb = _exp5_sem_truncate_by_depth(rgb, _exp5_depth_raw[dep.id],
                                                      _sh5, _sw5, sem_range)
                img = PIL.Image.fromarray(rgb, mode="RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                _sensor_frames[sem.id] = buf.getvalue()
                _sensor_frame_num[sem.id] = i + 1

            rows.append({"frame": i + 1, "time": t, "classes": classes, **ratios})
            if i % 4 == 0:
                pt = {"frame": i + 1, "t": round(t, 3), "progress": round((i + 1) / total * 100, 1), "classes": classes,
                      "weather": weather_key}
                pt.update({k + "_ratio": v for k, v in ratios.items()})
                pt["targets"] = len(targets)
                _obs5 = []
                for _k, _tg in enumerate(targets[:5]):
                    _ec = rich.get(_tg.get("id"), {})
                    # 深度测距是否有效：超过 depth_range 则依赖深度的字段置空；值缺失时亦置空
                    _ok5 = _tg["dist"] <= depth_range
                    def _v5(k, n):
                        v = _ec.get(k)
                        return round(v, n) if (_ok5 and v is not None) else None
                    _obs5.append({
                        "id": _tg.get("id", _k + 1), "cls": _tg["cls"],
                        "dist": round(_tg["dist"], 1) if _ok5 else None,
                        "x": _v5("x", 2), "y": _v5("y", 2), "z": _v5("z", 2),
                        "yaw": _v5("yaw", 3),
                        "length": _v5("length", 2), "width": _v5("width", 2),
                        "height": _v5("height", 2),
                        "vx": _v5("vx", 2), "vy": _v5("vy", 2),
                        "ax": _v5("ax", 2), "ay": _v5("ay", 2),
                        "is_static": (bool(_ec.get("is_static", False))
                                      if _ok5 and _ec.get("is_static") is not None
                                      else None),
                    })
                pt["obstacles"] = _obs5
                pt["grid_stat"] = grid_stat
                pt["bev"] = bev_payload
                if classes == "22":
                    # 车道线仅在 22 类语义分割下绘制（7 类不产出 → BEV 无车道线）
                    try:
                        pt["lane_edges"] = _exp5_lane_geoms(world, vehicle.get_transform(),
                                                            front=sem_range)
                    except Exception:
                        pass
                if decision is not None:
                    pt["decision"] = {
                        "state": decision["state"], "label": decision["label"],
                        "target_lat": decision["target_lat"], "block_m": decision["block_m"],
                        "reason": decision["reason"], "traj": decision["traj"],
                    }
                _push_to_sse({"experiment": {"id": 5, "trajectory": pt}})

        _push_to_sse({"experiment": {"id": 5, "result": {"elapsed": round(t if rows else 0, 1), "rows": len(rows)}}})
        _exp_log(f"实验5 完成 — {len(rows)} 行")
    except Exception as e:
        _exp_log(f"实验5 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        _stream_semantic = None

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

        # 行人 controller 先 stop，再统一销毁（避免持控 walker 时销毁崩溃）
        for _w, _c in walker_pairs:
            try:
                if _c is not None and _c.is_alive:
                    _c.stop()
            except Exception:
                pass

        # 销毁本实验生成的所有 actors（自行销毁，避免反复重启累积相机导致渲染变卡）
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
            _semantic_raw.pop(_a.id, None)
            _instance_raw.pop(_a.id, None)

        _EXP05_RUNNING = False
        # 还原实验前的真实世界天气，避免天气底色残留到后续实验
        if _exp5_orig_weather is not None:
            try:
                world.set_weather(_exp5_orig_weather)
            except Exception:
                pass
        if _EXP05_ABORT:
            _push_to_sse({"experiment": {"id": 5, "status": "stopped"}})


@app.route("/experiment/5/start", methods=["POST"])
def experiment_5_start():
    global _EXP05_THREAD, _EXP05_RUNNING, _EXP05_ABORT
    with _EXP05_LOCK:
        if _EXP05_THREAD is not None and _EXP05_THREAD.is_alive():
            if not _EXP05_ABORT:
                # 正在正常运行：直接拒绝。重试/双击产生的重复 start 在此被挡，
                # 绝不 kill-重启实验
                return jsonify({"status": "error", "message": "实验5 已在运行"}), 409
            # 正在停止/收尾：等旧线程完整退出后接管（保留「停止→快速重启」体验）
            _EXP05_THREAD.join(timeout=15.0)
            if _EXP05_THREAD.is_alive():
                return jsonify({"status": "error", "message": "实验5 旧线程仍在收尾，请稍后重试"}), 409
        # RUNNING/ABORT 在持锁的 handler 内提前置位：
        # ① 关闭 /cleanup、/preview/start 守卫在「start 已返回、线程体尚未执行」间的穿透窗口；
        # ② 消除「新线程刚起、ABORT 仍残留旧 stop 的 true」导致后续 start 误判收尾中的窗口
        _EXP05_RUNNING = True
        _EXP05_ABORT = False
        args = request.get_json(silent=True) or {}
        try:
            _EXP05_THREAD = threading.Thread(target=_run_exp05, args=(args,), daemon=True)
            _EXP05_THREAD.start()
        except Exception as exc:
            _EXP05_RUNNING = False
            return jsonify({"status": "error", "message": f"实验5 线程启动失败: {exc}"}), 500
    return jsonify({"status": "ok", "experiment_id": 5, "message": "实验5 已启动"})


@app.route("/experiment/5/stop", methods=["POST"])
def experiment_5_stop():
    global _EXP05_ABORT
    _EXP05_ABORT = True
    return jsonify({"status": "ok", "message": "实验5 停止请求已发送"})

