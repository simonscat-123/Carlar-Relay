"""实验10 可视化层（HMI）：检测框渲染 + 语义帧转换 + SSE 数据组装。

对标工业界 HMI/可视化模块（感知算法与渲染解耦——改渲染不误伤感知）：

  - render_bbox_frame / render_perceived_frame：在前相机 RGB 帧上叠加 2D
    检测框（真值模式按实例 id 匹配 actor；感知闭环模式只画感知到的目标，
    标注距离为单目几何估计值而非真值——学生看到的画面就是控制器实际
    使用的感知输出）；
  - render_bbox_overlay：主循环内用本 tick 检测结果 + 本 tick 相机帧渲染
    （位置前移到扫描之后、控制之前，避开尾部延迟让 bbox 赶上 SSE 采样；
    用本 tick 数据保证检测框与画面同步）；
  - render_semantic_frame：原始 CityScapes 标签 → 彩色图（SSE 推流用）；
  - build_sse_payload：鸟瞰可视化数据（当前/目标/将要走的车道、参考线含
    换道 S 弯、自行车模型预测轨迹）+ 实验状态帧组装。

依赖 PIL/numpy/carla；平台帧缓存经参数注入。
"""
from __future__ import annotations

import io
import math

import carla
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from carla_relay.experiments.comprehensive_driving.perception import BBOX_CLASSES, EXP10_PERC_STYLE

_font_cache = None


def _font():
    """标签字体：优先中文字体（Windows 微软雅黑/黑体），失败回退 PIL 默认位图字体"""
    global _font_cache
    if _font_cache is None:
        for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
            try:
                _font_cache = PIL.ImageFont.truetype(name, 30)
                break
            except Exception:
                continue
        if _font_cache is None:
            _font_cache = PIL.ImageFont.load_default()
    return _font_cache


def render_bbox_frame(rgb_arr, inst_arr, actors):
    """在前相机 RGB 帧上叠加 2D 检测框，返回 JPEG bytes。

    实现思路同官方 bounding_boxes.py 的 decode_instance_segmentation +
    bbox_2d_for_actor + visualize_2d_bboxes：
    实例分割图解码出每个 actor 的像素掩码（G=actor id 低字节、B=高字节）→
    最小外接矩形 → 按语义类别着色描边 + 类别色底白字标签（含距离）。
    rgb_arr: (H,W,4) BGRA 前相机原始帧；inst_arr: (h,w,4) BGRA 实例分割帧
    （可与 RGB 分辨率不同，坐标自动缩放）；actors: [(actor, 距离m), ...]。
    """
    h, w = inst_arr.shape[:2]
    rgb_h, rgb_w = rgb_arr.shape[:2]
    sx, sy = rgb_w / w, rgb_h / h
    # 解码实例分割 → actor id 图（与官方示例 decode_instance_segmentation 一致）
    actor_ids = inst_arr[:, :, 1].astype(np.uint16) + (inst_arr[:, :, 0].astype(np.uint16) << 8)
    # 单次扫描取出所有实例 id 的 bbox，取代逐 actor 全图掩码（多目标场景 10× 加速）
    from carla_relay.experiments.comprehensive_driving.perception import instance_bboxes
    bboxes = instance_bboxes(actor_ids)
    img = PIL.Image.fromarray(rgb_arr[:, :, :3][:, :, ::-1].copy())  # BGRA → RGB（copy 消除负步长只读视图，避免 fromarray 访问违规）
    draw = PIL.ImageDraw.Draw(img)
    font = _font()
    th = getattr(font, "size", 12)
    for actor, dist in actors:
        try:
            box = bboxes.get(actor.id)
            if box is None:
                continue  # 不在画面内
            xmin, ymin, xmax, ymax = box[:4]  # 7 元组取前 4（其余为计数/语义投票，渲染不用）
            xmin, ymin, xmax, ymax = xmin * sx, ymin * sy, xmax * sx, ymax * sy
            name, color = BBOX_CLASSES.get(actor.semantic_tags[0], ("目标", (85, 170, 255)))
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            # 标签：类别色底 + 白字（同官方 visualize_2d_bboxes 的渲染方式），附距离
            label = f"{name} {dist:.1f}m"
            tw = draw.textlength(label, font=font)
            ty = ymin - th - 10
            if ty < 0:
                ty = ymin + 3  # 贴近画面顶部时标签移入框内
            draw.rectangle([xmin, ty, xmin + tw + 14, ty + th + 8], fill=color)
            draw.text((xmin + 7, ty + 4), label, fill=(255, 255, 255), font=font)
        except Exception:
            continue
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def render_perceived_frame(rgb_arr, perceived, inst_w, inst_h):
    """在前相机 RGB 帧上叠加「感知闭环」检测结果，返回 JPEG bytes。

    与 render_bbox_frame 的区别：只画感知到的目标，标注距离为单目几何估计值
    而非真值——学生看到的画面就是控制器实际使用的感知输出。
    perceived: camera_perceive 的返回值；box 为实例图像素坐标，自动缩放。
    """
    rgb_h, rgb_w = rgb_arr.shape[:2]
    sx, sy = rgb_w / float(inst_w), rgb_h / float(inst_h)
    img = PIL.Image.fromarray(rgb_arr[:, :, :3][:, :, ::-1].copy())  # BGRA → RGB（copy 消除负步长只读视图，避免 fromarray 访问违规）
    draw = PIL.ImageDraw.Draw(img)
    font = _font()
    th = getattr(font, "size", 12)
    for p in perceived:
        xmin, ymin, xmax, ymax = p["box"]
        xmin, ymin, xmax, ymax = xmin * sx, ymin * sy, xmax * sx, ymax * sy
        name, color = EXP10_PERC_STYLE.get(p["cls"], ("目标", (85, 170, 255)))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        label = f"{name} {p['dist']:.1f}m·视觉"
        tw = draw.textlength(label, font=font)
        ty = ymin - th - 6
        if ty < 0:
            ty = ymin + 2
        draw.rectangle([xmin, ty, xmin + tw + 8, ty + th + 4], fill=color)
        draw.text((xmin + 4, ty + 2), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def render_bbox_overlay(inst, rgb_raw, instance_raw, perception_on, perceived,
                        bbox_cands, sensor_frames, sensor_frame_num, diag, log):
    """主循环 bbox 渲染：本 tick 检测结果 + 本 tick 相机帧 → 写入帧缓存。
    diag: {"ok","skip","err"} 渲染诊断计数（就地更新）。"""
    if rgb_raw["raw"] is not None and inst.id in instance_raw:
        try:
            _rgb_arr = np.frombuffer(rgb_raw["raw"], dtype=np.uint8).reshape((rgb_raw["h"], rgb_raw["w"], 4))
            _ih = int(inst.attributes["image_size_y"])
            _iw = int(inst.attributes["image_size_x"])
            if perception_on:
                sensor_frames[inst.id] = render_perceived_frame(_rgb_arr, perceived, _iw, _ih)
            else:
                _inst_arr = np.frombuffer(instance_raw[inst.id], dtype=np.uint8).reshape((_ih, _iw, 4))
                sensor_frames[inst.id] = render_bbox_frame(_rgb_arr, _inst_arr, bbox_cands)
            sensor_frame_num[inst.id] = sensor_frame_num.get(inst.id, 0) + 1
            diag["ok"] += 1
        except Exception as exc:
            diag["err"] += 1
            if diag["err"] <= 3:
                log(f"bbox渲染异常#{diag['err']}: {exc!r}")
    else:
        diag["skip"] += 1


# ── 3D 包围框叠加（自车 + 障碍：真实框实线 / 带余量框虚线）─────────────
# 纯可视化，不改感知/规划/控制。世界系角点 → 相机系 → 像素，透视投影画线框。
# 目标识别相机：画 8 角点立体线框；鸟瞰相机：画地面足迹(实/虚矩形)。
# 尺寸/朝向来源：自车=fused 定位 + 车体常量；障碍=Frenet(s,l)→世界 + 类别高度。
# 余量 = 规划决策用余量 SEP_MIN(横向)/COLL_S(纵向)，所见即决策所用。
SEP_MIN = 0.2            # 与 simple_planner.SEP_MIN 一致的横向分离余量（m）
COLL_S = 0.5             # 与 planner.COLL_S 一致的纵向碰撞余量（m）
EGO_HW = 1.0             # 自车半宽（m）
EGO_HL = 2.45            # 自车半长（m）
EGO_H = 1.5              # 自车高（m）


def _intrinsics(fov, w, h):
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[focal, 0.0, w / 2.0],
                     [0.0, focal, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _world_to_camera(vehicle, cam):
    """世界→相机局部 4x4（同官方 bounding_boxes.py）。

    CARLA 中 attach 到车辆的传感器，get_transform() 返回的已是**世界位姿**
    （非相对父级的局部位姿），直接取 get_inverse_matrix() 即为 w2c；
    再叠加车辆变换会重复复合，导致投影全偏出画面。
    """
    return np.asarray(cam.get_transform().get_inverse_matrix(), dtype=np.float64)


def _project_points(world_pts, vehicle, cam, w, h):
    """世界系角点 → 像素 (x,y)；点位于相机后方/前方阈值内返回 None。"""
    K = _intrinsics(float(cam.attributes.get("fov", 90.0)), w, h)
    w2c = _world_to_camera(vehicle, cam)
    out = []
    for p in world_pts:
        cp = w2c.dot(np.array([p.x, p.y, p.z, 1.0]))
        # UE4 相机局部系（同官方 get_image_point 的重排 (x,y,z)→(y,-z,x)）：
        # 前向=+X 为景深分母，+Y 为列方向，-Z 为行方向。
        z = cp[0]
        if z <= 0.1:
            out.append(None)
            continue
        px = K[0, 0] * cp[1] / z + K[0, 2]
        py = K[1, 1] * (-cp[2]) / z + K[1, 2]
        out.append((px, py))
    return out


def _ego_box_pts(cx, cy, yaw_deg, hl, hw, h_full, ground_z):
    """自车：中心 + 航向旋转的外接 8 角点（底4+顶4）。"""
    phi = math.radians(yaw_deg)
    c, s = math.cos(phi), math.sin(phi)
    fx, fy, rx, ry = c, s, -s, c
    def pt(i, j, z):
        return carla.Location(cx + i * fx + j * rx, cy + i * fy + j * ry, z)
    bottom = [pt(-hl, -hw, ground_z), pt(-hl, hw, ground_z),
              pt(hl, hw, ground_z), pt(hl, -hw, ground_z)]
    top = [carla.Location(p.x, p.y, p.z + h_full) for p in bottom]
    return bottom, top


def _obs_box_pts(ref, o, hl, hw, h_full, ground_z):
    """障碍：沿参考线在 (s,l) 展开的外接 8 角点（无需航向，world(s,l) 即含方向）。"""
    s, l = o["s"], o["l"]
    bottom = []
    for ds, dl in ((-hl, -hw), (-hl, hw), (hl, hw), (hl, -hw)):
        x, y = ref.world(s + ds, l + dl)
        bottom.append(carla.Location(x, y, ground_z))
    top = [carla.Location(p.x, p.y, p.z + h_full) for p in bottom]
    return bottom, top


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),   # 底框
              (4, 5), (5, 6), (6, 7), (7, 4),   # 顶框
              (0, 4), (1, 5), (2, 6), (3, 7)]   # 竖棱


def _dashed(draw, a, b, color, width=2, dash=8, gap=5):
    x0, y0, x1, y1 = a[0], a[1], b[0], b[1]
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-6:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    d = 0.0
    while d < L:
        d2 = min(L, d + dash)
        draw.line([x0 + ux * d, y0 + uy * d, x0 + ux * d2, y0 + uy * d2],
                  fill=color, width=width)
        d = d2 + gap


def _draw_box(draw, pixels, color, dashed=False, full3d=True, width=2):
    """pixels: [8] 像素点。full3d 时画立体线框，否则只画底框足迹。"""
    edges = _BOX_EDGES if full3d else [(0, 1), (1, 2), (2, 3), (3, 0)]
    for ia, ib in edges:
        a, b = pixels[ia], pixels[ib]
        if a is None or b is None:
            continue
        if dashed:
            _dashed(draw, a, b, color, width=width)
        else:
            draw.line([a[0], a[1], b[0], b[1]], fill=color, width=width)


_BOX_DBG = {"n": 0}


def overlay_3d_boxes(*, vehicle, fused_loc, fused_yaw_deg, obstacles, reference,
                     inst, bird, sensor_frames, log):
    """在目标识别相机(inst)与鸟瞰相机(bird)上叠加自车+障碍的 3D/足迹包围框。

    侵入面：仅改写 _sensor_frames[inst.id]/[bird.id] 的编码帧，不动上层决策。
    obstacles: PercFrame.obstacles（每项含 s/l/half_len/half_w/cls）。
    """
    # 无条件入口诊断：证明 overlay 确实被调用、且能看到两个 feed 是否就位
    if _BOX_DBG["n"] < 6:
        _BOX_DBG["n"] += 1
        _has = {c.id: c.id in sensor_frames for c in (inst, bird)}
        log(f"3DBOX 入口 主车+{len(obstacles)}障碍 "
            f"inst帧={'在' if _has.get(inst.id) else '无'} "
            f"bird帧={'在' if _has.get(bird.id) else '无'}")

    ground_z = fused_loc.z - 0.9  # 道路近似高度（自车中心 -0.9 落地）
    specs = []
    eb = _ego_box_pts(fused_loc.x, fused_loc.y, fused_yaw_deg, EGO_HL, EGO_HW, EGO_H, ground_z)
    specs.append({"color": (0, 220, 255), "set": eb})                    # 自车·真实
    eb_m = _ego_box_pts(fused_loc.x, fused_loc.y, fused_yaw_deg,
                        EGO_HL + COLL_S, EGO_HW + SEP_MIN, EGO_H, ground_z)
    specs.append({"color": (0, 120, 255), "set": eb_m, "dashed": True})  # 自车·带余量
    for o in obstacles:
        h_full = 1.8 if str(o["cls"]).startswith("walker") else 1.5
        ob = _obs_box_pts(reference, o, o["half_len"], o["half_w"], h_full, ground_z)
        specs.append({"color": (255, 180, 0), "set": ob})               # 障碍·真实
        ob_m = _obs_box_pts(reference, o, o["half_len"] + COLL_S,
                            o["half_w"] + SEP_MIN, h_full, ground_z)
        specs.append({"color": (255, 60, 60), "set": ob_m, "dashed": True})  # 障碍·带余量

    for cam, full3d in ((inst, True), (bird, False)):
        try:
            jpeg = sensor_frames.get(cam.id)
            if jpeg is None:
                continue
            # 帧可能是被其他渲染改写后的图（如 bbox 叠加把 inst 图换成前相机
            # 1280x720 的 RGB），故用 jpeg 实际像素尺寸投影，避免内参/画面错配。
            img = PIL.Image.open(io.BytesIO(jpeg)).convert("RGB")
            w, h = img.size
            draw = PIL.ImageDraw.Draw(img)
            for sp in specs:
                bottom, top = sp["set"]
                pts = _project_points(bottom + top, vehicle, cam, w, h)
                _draw_box(draw, pts, sp["color"], dashed=sp.get("dashed", False),
                          full3d=full3d)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            sensor_frames[cam.id] = buf.getvalue()
            # 首个可投影自车框：打印自车底框中心像素，便于确认投影是否落在画面上
            if _BOX_DBG["n"] < 12:
                _BOX_DBG["n"] += 1
                ctr = _project_points([eb[1][2]], vehicle, cam, w, h)[0]
                log(f"3DBOX cam={str(cam.id)[:6]} 首帧自车框中心像素={ctr} "
                    f"(画面{w}x{h}) specs={len(specs)} in3d={full3d}")
        except Exception:
            import traceback
            log("3D框渲染异常:\n" + traceback.format_exc())


def render_semantic_frame(sem, semantic_raw, sensor_frames, sensor_frame_num,
                          label_semantic_level, colors_from_labels):
    """语义分割帧：原始 CityScapes 标签 → 彩色图写入帧缓存，供 SSE 推流
    （前端可在相机视角下拉切到语义画面；未连接/无帧时前端回退 Mock）。"""
    if sem.id in semantic_raw:
        try:
            sem_h = int(sem.attributes["image_size_y"])
            sem_w = int(sem.attributes["image_size_x"])
            sem_arr = np.frombuffer(semantic_raw[sem.id], dtype=np.uint8).reshape((sem_h, sem_w, 4))
            sem_labels = sem_arr[:, :, 2].astype(np.int32)
            sem_rgb = colors_from_labels(label_semantic_level(sem_labels, "L2"), "L2")
            sem_buf = io.BytesIO()
            PIL.Image.fromarray(sem_rgb, mode="RGB").save(sem_buf, format="JPEG", quality=85)
            sensor_frames[sem.id] = sem_buf.getvalue()
            sensor_frame_num[sem.id] = sensor_frame_num.get(sem.id, 0) + 1
        except Exception:
            pass


def build_sse_payload(*, t, wp_idx, route_wp, route_lane_ids, sampling_res,
                      fused_loc, fused_yaw_deg, spd, desired, steer, throttle,
                      brake, cte, loc_err, front_obstacle, arrived,
                      perception_on, perceived_count, gt_loc, gt_yaw,
                      ngx, ngy, plan, obs_list, planned_obstacles, tl,
                      carla_map):
    """组装 SSE 实验数据帧（前端零改动的兼容字段集）。
    plan: PlanOutput；tl: TlFrame。"""
    # ── 鸟瞰可视化数据：当前/目标车道、参考线（含换道 S 弯）、预测轨迹 ──
    cur_lane = None
    try:
        cwp = carla_map.get_waypoint(fused_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if cwp is not None:
            cur_lane = [cwp.road_id, cwp.lane_id]
    except Exception:
        pass
    tgt_lane = list(plan.avoid_dest_lane) if (plan.avoiding and plan.avoid_dest_lane is not None) else cur_lane

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
    for px, py in plan.plan_traj:
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
            tl_ = math.hypot(tx, ty)
            if tl_ > 1e-6:
                blend = min(1.0, max(0.0, (d - d_covered) / 10.0))
                off = plan.plan_end_l + (plan.avoid_lat_target - plan.plan_end_l) * blend
                px -= (ty / tl_) * off
                py += (tx / tl_) * off
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

    return {
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
            "perceived_count": perceived_count,
            "gt": {"x": round(gt_loc.x, 2), "y": round(gt_loc.y, 2)},
            "fused": {"x": round(fused_loc.x, 2), "y": round(fused_loc.y, 2)},
            "gnss": {"x": round(ngx, 2), "y": round(ngy, 2)},
            "heading": round(fused_yaw_deg, 1),
            "gt_yaw": round(gt_yaw, 1),
            "lane": {"cur": cur_lane, "tgt": tgt_lane, "plan": plan_lanes},
            "ref_path": ref_path,
            "pred_path": pred_path,
            "obstacles": obs_list[:5],
            "planned_obstacles": planned_obstacles,
            "traffic_light": {"state": tl.state, "distance": round(tl.dist, 1)},
            "avoid": {"active": plan.avoiding, "side": plan.avoid_side,
                      "offset": round(plan.avoid_lat_target, 2)},
            "fsm": plan.fsm_state,
        }
    }
