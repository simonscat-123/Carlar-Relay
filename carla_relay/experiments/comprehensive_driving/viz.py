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
        label = f"{name}"
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
    """主循环 bbox 渲染：本 tick 检测结果 + 本 tick 相机帧 → 返回渲染后的 JPEG。

    不再直接写 sensor_frames[inst.id]，改为**返回** bytes，由后接的
    overlay_3d_boxes 在同一帧上叠 3D 后一次性写回，避免 SSE 线程读到
    "只有 2D 无 3D" 的中间帧导致 3D 框闪烁消失。
    diag: {"ok","skip","err"} 渲染诊断计数（就地更新）。返回 bytes 或 None。
    """
    if rgb_raw["raw"] is not None and inst.id in instance_raw:
        try:
            _rgb_arr = np.frombuffer(rgb_raw["raw"], dtype=np.uint8).reshape((rgb_raw["h"], rgb_raw["w"], 4))
            _ih = int(inst.attributes["image_size_y"])
            _iw = int(inst.attributes["image_size_x"])
            if perception_on:
                out = render_perceived_frame(_rgb_arr, perceived, _iw, _ih)
            else:
                _inst_arr = np.frombuffer(instance_raw[inst.id], dtype=np.uint8).reshape((_ih, _iw, 4))
                out = render_bbox_frame(_rgb_arr, _inst_arr, bbox_cands)
            diag["ok"] += 1
            return out
        except Exception as exc:
            diag["err"] += 1
            if diag["err"] <= 3:
                log(f"bbox渲染异常#{diag['err']}: {exc!r}")
    else:
        diag["skip"] += 1
    return None


# ── 3D 包围框叠加（自车 + 障碍：真实框实线 / 带余量框虚线）─────────────
# 纯可视化，不改感知/规划/控制。世界系角点 → 相机系 → 像素，透视投影画线框。
# 目标识别相机：画 8 角点立体线框；鸟瞰相机：画地面足迹(实/虚矩形)。
# 尺寸/朝向来源：自车=fused 定位 + 车体常量；障碍=Frenet(s,l)→世界 + 类别高度。
# 余量 = simple_planner 实际越障/clr 用的单一余量 AVOID_MARGIN（障碍边→本车边，含
# 执行层跟踪误差预算）。障碍画余量虚线框（禁区边界）；自车画真实框 + 纯车宽
# 参考虚线框（同尺寸不带余量），可过判断由 build_gap_viz 可容带完成（带宽≥车宽）
# → 所见即决策。
from carla_relay.experiments.comprehensive_driving.simple_planner import AVOID_MARGIN, EDGE_MARGIN
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


def _rect_from_verts(vpts, am, ground_z):
    """由真实包围盒 8 顶点求「底面矩形四角」，再向四侧各扩余量 am。

    用真实顶点做 PCA 得到底面的两条主轴（含真实朝向），中心取底面质心，
    半尺寸取顶点沿轴向投影上界 + am。这样余量框严格贴在障碍真实位置并带
    真实朝向，彻底摆脱 `reference.world(s, l)` 沿弧长重建带来的漂移
    （弧长 s 与直线距离不等、感知 s/l 误差放大正是"自车准、障碍漂"的根因）。
    返回 [carla.Location]*4（底面四角，供 bird 足迹连线；右侧相机画 3D 时
    顶面沿用加高度，故这里仅提供底面矩形）。
    """
    # 取 z 最小的 4 点为底面
    bs = sorted(vpts, key=lambda p: p.z)[:4]
    import numpy as np
    M = np.array([[p.x, p.y] for p in bs], dtype=np.float64)
    c = M.mean(0)
    q = M - c
    cov = q.T @ q
    evals, evecs = np.linalg.eigh(cov)
    ax = evecs[:, 1]          # 长轴（最大特征）
    ay = evecs[:, 0]          # 短轴
    h1 = float(np.abs(q @ ax).max()) + am
    h2 = float(np.abs(q @ ay).max()) + am
    out = []
    # 顺时针环序（用直轴长 ±，避免按 s1/s2 嵌套循环产出非相邻对角点 →
    # edges 连线会画成 X 字形）。顺序：(-,-),(-,+),(+,+),(+,-) 与 _obs_box_pts 的
    # bottom 顺序一致，4条边连线即为矩形。
    for a, b in ((-h1, -h2), (-h1, h2), (h1, h2), (h1, -h2)):
        out.append(carla.Location(c[0] + a * ax[0] + b * ay[0],
                                  c[1] + a * ax[1] + b * ay[1],
                                  ground_z))
    return out


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),   # 底框
              (4, 5), (5, 6), (6, 7), (7, 4),   # 顶框
              (0, 4), (1, 5), (2, 6), (3, 7)]   # 竖棱

# 官方 bounding_boxes.py 配套 get_world_vertices() 顺序的连线表（顶点顺序不同，
# 不能与上面手工 bottom+top 表混用）。
_OFFICIAL_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),   # 底面
                   (0, 4), (4, 5), (5, 1), (5, 7),   # 竖棱 + 顶面前边
                   (7, 6), (6, 4), (6, 2), (7, 3)]   # 顶面 + 后竖棱


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


def _draw_box(draw, pixels, color, dashed=False, full3d=True, width=2,
              edges3=_BOX_EDGES, edgesf=((0, 1), (1, 2), (2, 3), (3, 0))):
    """pixels: [8] 像素点。full3d 用 edges3 画立体线框，否则用 edgesf 画底框足迹。"""
    edges = edges3 if full3d else edgesf
    for ia, ib in edges:
        a, b = pixels[ia], pixels[ib]
        if a is None or b is None:
            continue
        if dashed:
            _dashed(draw, a, b, color, width=width)
        else:
            draw.line([a[0], a[1], b[0], b[1]], fill=color, width=width)


_BOX_DBG = {"n": 0, "n2": 0}

# 已输出过宽度诊断的障碍 id（按 id 去重，每障碍只打一次）
_GAP_LOG_IDS = set()


def build_gap_viz(reference, obstacles, ego_half_w, ego_half_len, ego_s=None,
                  aggressive_on=True, limit=4, log=None, avoid_margin=None):
    """鸟瞰「车身可容空间」可视化数据：每个前方障碍所在横断面，用与
    SimplePlanner 同源的 ReferenceLine.probe_drivable 可行驶域区间，按同一
    带宽公式求两侧可容带（可容带 = 车边可到区间：障碍禁区外边 ↔ 域边界
    −EDGE_MARGIN）+ 可容宽 + 该侧能否通过（带宽 ≥ 车宽 2×ego_half_w），
    与规划器候选生成公式严格同源 → 所见即决策。
    返回 {"ego_req_w":…, "sides":[…]}; 无参考线 / 无有效障碍时 sides 为空。"""
    am = AVOID_MARGIN if avoid_margin is None else float(avoid_margin)
    out = {"ego_req_w": round(2 * ego_half_w, 2), "sides": []}
    if reference is None:
        return out
    items = []
    for o in obstacles or []:
        if ego_s is not None:
            # 只看前方未越过的障碍（后缘不落后自车 4m、前缘不超 60m）
            if o["s"] + o["half_len"] < ego_s - 4.0 or o["s"] - o["half_len"] > ego_s + 60.0:
                continue
        items.append(o)
    items.sort(key=lambda o: o["s"] - o["half_len"])   # 由近及远
    for o in items[:limit]:
        s_lo, s_hi = o["s"] - o["half_len"], o["s"] + o["half_len"]
        s_rear_ob = o["s"] - o["half_len"]
        s_mid = s_rear_ob if s_rear_ob > (ego_s or 0.0) else (ego_s or 0.0) + 6.0
        ivs = reference.probe_drivable(s_mid, aggressive_on)   # 与决策同源
        _log_lines = []
        for side in (-1, 1):
            l_lo = o["l"] - o["half_w"]     # 障碍左缘
            l_hi = o["l"] + o["half_w"]     # 障碍右缘
            best_w, best_a, best_b, best_iv = None, None, None, None
            for iv_lo, iv_hi in ivs:
                # 可容带 = 车边可到区间（与 simple_planner 候选公式同源）：
                #   左：外侧 = iv_lo + EDGE，内侧 = min(障碍左缘 − AVOID, iv_hi − EDGE)
                #   右：内侧 = max(障碍右缘 + AVOID, iv_lo + EDGE)，外侧 = iv_hi − EDGE
                if side < 0:
                    a = iv_lo + EDGE_MARGIN
                    b = min(l_lo - am, iv_hi - EDGE_MARGIN)
                else:
                    a = max(l_hi + am, iv_lo + EDGE_MARGIN)
                    b = iv_hi - EDGE_MARGIN
                w = b - a
                if best_w is None or w > best_w:
                    best_w, best_a, best_b, best_iv = w, a, b, (iv_lo, iv_hi)
            if best_iv is None:
                continue                     # 该横断面无可行驶域，跳过
            free = (l_lo - best_iv[0]) if side < 0 else (best_iv[1] - l_hi)
            corridor = best_w if best_w is not None else 0.0
            if corridor > 0.05:
                l_a, l_b = best_a, best_b
            else:
                # 无可容空间：在带宽中点画红色细带，提示本侧不可通过
                l_mid = (best_a + best_b) / 2
                l_a, l_b = l_mid - 0.2, l_mid + 0.2
            # 平行四边形：沿障碍长度 s_lo→s_hi，横向 可容带内/外侧
            # reference.world() 返回 (x, y) 元组（非 carla.Location）
            poly = [reference.world(s_lo, l_a), reference.world(s_hi, l_a),
                    reference.world(s_hi, l_b), reference.world(s_lo, l_b)]
            lx, ly = reference.world(o["s"], (l_a + l_b) / 2)
            # EDGE 边界余量保留区：可容带外侧 → 探测域边界（本车不会进入）
            e_in = l_a if side < 0 else l_b
            e_out = best_iv[0] if side < 0 else best_iv[1]
            e_poly = [reference.world(s_lo, e_in), reference.world(s_hi, e_in),
                      reference.world(s_hi, e_out), reference.world(s_lo, e_out)]
            out["sides"].append({
                "poly": [[round(p[0], 2), round(p[1], 2)] for p in poly],
                "width": round(max(0.0, corridor), 2),
                "pass": corridor >= 2 * ego_half_w,
                "label": [round(lx, 2), round(ly, 2)],
                "edge_poly": [[round(p[0], 2), round(p[1], 2)] for p in e_poly],
                "edge_margin": round(EDGE_MARGIN, 2),
            })
            _log_lines.append((side, free, corridor, corridor >= 2 * ego_half_w))
        # 宽度诊断：每障碍首次出现打一次（探测域/两侧空闲/可容宽/可否通过）
        oid = o.get("id")
        if log is not None and oid is not None and oid not in _GAP_LOG_IDS:
            _GAP_LOG_IDS.add(oid)
            _ivs_txt = ",".join(f"[{a:.1f},{b:.1f}]" for a, b in ivs)
            _side_txt = " ".join(
                f"{'左' if side < 0 else '右'} free={free:.2f} "
                f"fit={corr:.2f} {'可过' if corr is not None and corr >= 2 * ego_half_w else '不可过'}"
                for side, free, corr, _ in _log_lines)
            log(f"GAP 障碍#{oid} {o['cls']}@s={o['s']:.1f} l={o['l']:+.2f} "
                f"半长={o['half_len']:.2f} 半宽={o['half_w']:.2f} ivs={_ivs_txt} | {_side_txt}")
    return out


def overlay_3d_boxes(*, vehicle, fused_loc, fused_yaw_deg, obstacles, reference,
                     inst, bird, sensor_frames, log, inst_frame=None, sensor_frame_num=None,
                     avoid_margin=None):
    """在目标识别相机(inst)与鸟瞰相机(bird)上叠加自车+障碍的 3D/足迹包围框。

    侵入面：仅改写 _sensor_frames[inst.id]/[bird.id] 的编码帧，不动上层决策。
    obstacles: PercFrame.obstacles（每项含 s/l/half_len/half_w/cls）。
    avoid_margin: 障碍余量虚线框的横向余量（m，实验 JSON 配置）；不传时回退
    simple_planner 模块默认 AVOID_MARGIN，与规划器实际越障余量保持一致。
    """
    am = AVOID_MARGIN if avoid_margin is None else float(avoid_margin)
    # 无条件入口诊断：证明 overlay 确实被调用、且能看到两个 feed 是否就位
    if _BOX_DBG["n"] < 6:
        _BOX_DBG["n"] += 1
        _has = {c.id: c.id in sensor_frames for c in (inst, bird)}
        log(f"3DBOX 入口 主车+{len(obstacles)}障碍 "
            f"inst帧={'在' if _has.get(inst.id) else '无'} "
            f"bird帧={'在' if _has.get(bird.id) else '无'}")

    ground_z = fused_loc.z - 0.9  # 道路近似高度（自车中心 -0.9 落地）
    records = []
    out = {"bbox": [], "bird": []}

    def add(col, pts, dashed, truth=False, foot=None, ego=False, obs_real=False):
        """pts: [8] 三维点（carla.Location 或含 x/y/z）→ 投影后画框。
        truth: 用官方连线表（真实框顶点序）；foot: 底面矩形点（bird 足迹专用，
        替代固定索引连线，避免正下视塌缩成线）；ego: 自车框标记（仅鸟瞰画，
        右侧 RGB 相机不画自车 3D 框）；obs_real: 障碍「真实体积」实线框（右侧
        目标识别相机不画实线体积，只在鸟瞰画足迹；体验上 2D bbox + 虚线余量框
        已足够，避免画面被实线条占满）。"""
        e3 = _OFFICIAL_EDGES if truth else _BOX_EDGES
        records.append({"color": col, "pts": pts, "dashed": dashed,
                        "edges3": e3, "foot": foot, "ego": ego,
                        "obs_real": obs_real})

    # 自车：真实框用 bounding_box 世界顶点（官方同款，绝对贴合）；余量框用
    # 融合位姿重建（真实框对齐后，余量框即所见即决策）。
    # 注意：自车 3D 框只画在鸟瞰(bird)；右侧 RGB(inst) 不画自车框（避免挡画面）。
    try:
        ego_verts = [carla.Location(v.x, v.y, v.z)
                     for v in vehicle.bounding_box.get_world_vertices(vehicle.get_transform())]
    except Exception:
        ego_verts = None
    if ego_verts:
        ego_ground = min(v.z for v in ego_verts)   # 自车真实地面高度
    else:
        ego_ground = ground_z
    if ego_verts:
        # bbox 用真实顶点贴物；bird 足迹用「重建底面矩形」（与虚线同法，避免
        # 路面有坡度时真实底面 4 顶点 z 不等、提取不足而塌缩成线）
        _bt = _ego_box_pts(fused_loc.x, fused_loc.y, fused_yaw_deg,
                           EGO_HL, EGO_HW, EGO_H, ego_ground)
        add((0, 220, 255), ego_verts, False, truth=True, foot=_bt[0],
            ego=True)                                                  # 自车·真实
    else:
        _bt = _ego_box_pts(fused_loc.x, fused_loc.y, fused_yaw_deg,
                           EGO_HL, EGO_HW, EGO_H, ego_ground)
        add((0, 220, 255), _bt[0] + _bt[1], False, ego=True)
    # 自车纯车宽参考虚线框（=真实框同尺寸，不带 AVOID 余量；仅作车宽示意，
    # 可过判断仍由 build_gap_viz 可容带完成，不引入"余量虚框"矛盾）
    _bt = _ego_box_pts(fused_loc.x, fused_loc.y, fused_yaw_deg,
                       EGO_HL, EGO_HW, EGO_H, ego_ground)
    add((0, 120, 255), _bt[0] + _bt[1], True, ego=True)                # 自车·纯车宽参考
    for o in obstacles:
        h_full = 1.8 if str(o["cls"]).startswith("walker") else 1.5
        ov = o.get("verts")
        if ov:
            # 真值模式/感知模式补齐的真实包围盒顶点：直接用，位置与物体严格重合（真值）。
            # 余量虚线框也用真实底面矩形向四侧扩 AVOID 余量（保留真实朝向），彻底不依赖
            # reference.world(s,l) 弧长重建，消除"自车准、障碍漂"。
            pts = [carla.Location(v["x"], v["y"], v["z"]) if isinstance(v, dict) else v
                   for v in ov]
            o_ground = min(v.z for v in pts)          # 障碍真实地面高度
            _ob = _obs_box_pts(reference, o, o["half_len"], o["half_w"],
                               h_full, o_ground)
            add((255, 180, 0), pts, False, truth=True, foot=_ob[0],
                obs_real=True)  # 障碍·真实（用真实顶点贴合）
            add((255, 60, 60), _rect_from_verts(pts, am, o_ground), True)  # 障碍·带余量（真值中心）
        else:
            o_ground = ground_z
            _bt = _obs_box_pts(reference, o, o["half_len"],
                               o["half_w"], h_full, o_ground)
            add((255, 180, 0), _bt[0] + _bt[1], False,
                obs_real=True)                           # 障碍·真实(重建，无真值顶点时回退)
            mbt = _obs_box_pts(reference, o, o["half_len"] + am,
                               o["half_w"] + am, h_full, o_ground)
            add((255, 60, 60), mbt[0] + mbt[1], True)                    # 障碍·带余量(重建)

    for cam, full3d in ((inst, True), (bird, False)):
        tag = "bbox" if full3d else "bird"
        try:
            # 投影基底帧：inst 用本 tick 的干净 bbox 帧（含 2D 检测框，不含 3D）；
            # bird 用帧缓存原始俯瞰帧。仅取尺寸用于投影，3D 框不再刻进 JPEG。
            if cam.id == inst.id and inst_frame is not None:
                jpeg = inst_frame
                write_back = True
            else:
                jpeg = sensor_frames.get(cam.id)
                write_back = False
            if jpeg is None:
                if cam.id == inst.id and inst_frame is not None and sensor_frame_num is not None:
                    sensor_frame_num[inst.id] = sensor_frame_num.get(inst.id, 0) + 1
                continue
            # 帧可能是被其他渲染改写后的图（如 bbox 叠加把 inst 图换成前相机
            # 1280x720 的 RGB），故用 jpeg 实际像素尺寸投影，避免内参/画面错配。
            w, h = PIL.Image.open(io.BytesIO(jpeg)).size
            for rec in records:
                # 障碍「真实体积」实线框：两相机都不画（右侧保留 2D bbox +
                # 虚线余量框；鸟瞰保留虚线余量框）。
                if rec.get("obs_real"):
                    continue
                pts_ok = rec["pts"]
                if full3d and len(pts_ok) >= 8:
                    # 立体线框（自车真实框等 8 角点）：右侧相机也画，恢复 3D 显示。
                    pixels = _project_points(pts_ok, vehicle, cam, w, h)
                    edges = rec["edges3"]
                else:
                    # 仅有底面/足迹 4 点的框（障碍余量框）或鸟瞰视角：按矩形连线，
                    # 避免 full3d 对 4 点 rec 用 edges3(面向 8 点) 索引越界，使右侧
                    # bbox 段整段丢失。foot 优先，否则取 pts 前 4 点（地面矩形）。
                    base = (rec.get("foot") if rec.get("foot") is not None
                            else pts_ok[:4])
                    fpix = _project_points(base, vehicle, cam, w, h)
                    edges = [(i, (i + 1) % 4) for i in range(4)]
                    pixels = dict(enumerate(fpix))
                # 归一化 UV 线段下发：客户端用「自己展示该帧的实际宽高」乘回，
                # 使任意显示缩放/适配模式下坐标都精确对齐（不依赖刻进 JPEG 的像素）。
                segs = []
                for ia, ib in edges:
                    a, b = pixels[ia], pixels[ib]
                    if a is None or b is None:
                        continue
                    segs.append([[a[0] / w, a[1] / h],
                                 [b[0] / w, b[1] / h]])
                # 单帧诊断：对比自车框 vs 障碍框的「世界坐标 / 投影 uv」，判定偏移
                # 在数据侧(world 重建)还是前端绘制侧(uv→画布映射)。只打前几帧。
                if _BOX_DBG["n2"] < 20 and segs:
                    _BOX_DBG["n2"] += 1
                    _kind = ("EGO" if rec.get("ego")
                             else "OBS-B" if tuple((rec["color"][0], rec["color"][1], rec["color"][2])) == (255, 60, 60)
                             else "OTH")
                    _like = rec["pts"]
                    _p0 = _like[0] if len(_like) > 0 else None
                    if _kind in ("EGO", "OBS-B") and _p0 is not None:
                        log(f"3DBOX[{_kind}] {tag} world=({_p0.x:.1f},{_p0.y:.1f},z={_p0.z:.2f})"
                            f" uv0={tuple(round(v, 3) for v in segs[0][0])} "
                            f"点多={len(segs)} uvN={tuple(round(v, 3) for v in segs[-1][1])}")
                out[tag].append({"segs": segs,
                                 "color": list(rec["color"]),
                                 "dashed": rec["dashed"]})
            # 干净 bbox 帧写回推流（只含 2D 检测框）；bird 保持原始俯瞰帧，皆不叠 3D。
            if write_back:
                sensor_frames[inst.id] = inst_frame
                if sensor_frame_num is not None:
                    sensor_frame_num[inst.id] = sensor_frame_num.get(inst.id, 0) + 1
        except Exception:
            import traceback
            log("3D框渲染异常:\n" + traceback.format_exc())
    return out


def render_semantic_frame(sem, semantic_raw, sensor_frames, sensor_frame_num,
                          label_semantic_classes, colors_from_labels):
    """语义分割帧：原始 CityScapes 标签 → 彩色图写入帧缓存，供 SSE 推流
    （前端可在相机视角下拉切到语义画面；未连接/无帧时前端回退 Mock）。"""
    if sem.id in semantic_raw:
        try:
            sem_h = int(sem.attributes["image_size_y"])
            sem_w = int(sem.attributes["image_size_x"])
            sem_arr = np.frombuffer(semantic_raw[sem.id], dtype=np.uint8).reshape((sem_h, sem_w, 4))
            sem_labels = sem_arr[:, :, 2].astype(np.int32)
            sem_rgb = colors_from_labels(label_semantic_classes(sem_labels, "7"), "7")
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
                      carla_map, viz3d=None, gap_viz=None):
    """组装 SSE 实验数据帧（前端零改动的兼容字段集）。
    plan: PlanOutput；tl: TlFrame。viz3d: overlay_3d_boxes 返回的 bbox/bird
    投影线段，供 local_runner 与车道线同通道即时绘制（避免 JPEG 闪烁）。"""
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
            "bbox3d": (viz3d or {}).get("bbox", []),
            "bird3d": (viz3d or {}).get("bird", []),
            "gap_viz": gap_viz or {"ego_req_w": 0.0, "sides": []},
        }
    }
