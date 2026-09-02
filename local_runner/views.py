"""pygame 渲染视图：各实验的运行画面 + 实时图表 + HUD。

布局对标前端：
  - 综合驾驶: 左车顶俯瞰(bird，叠加车道/参考线/预测线/障碍物) + 右车前包围框(bbox)
  - 语义分割: RGB(camera) + 语义(semantic) 双画面
  - Lidar检测: 三目相机水平无缝拼接全景（左/前/右 FOV=90° yaw=-90/0/+90）
  - 定位分析: 前视相机
  所有实验底部为实时统计图（charts.TelemetryHistory）。
相机画面只显示图像本身：不加标签、不加边框（用户要求）。
"""
from __future__ import annotations

import math

import pygame

from .charts import TelemetryHistory
from .relay_client import decode_frame_b64, img_bytes_to_surface

HUD_BG = (28, 33, 48)
HUD_TEXT = (215, 225, 245)
HUD_DIM = (140, 152, 176)
OK_COLOR = (120, 220, 150)

# ── 鸟瞰叠加开关（每题一项；False 即不画对应元素，改这里即可）──
VIZ_LANE_BAND = True    # 车道底色带（当前/目标/规划车道半透明带）
VIZ_LANE_EDGE = True    # 车道边界线（实线/虚线）
VIZ_REF_LINE = True     # 参考路线（蓝实线）
VIZ_PRED_LINE = True    # 预测轨迹（橙虚线）
VIZ_OBSTACLE = False     # 障碍物块（红）
VIZ_RANGE = False        # 感知范围圈（约 50m）
VIZ_EGO_MARK = False     # 自车标记（中心圆点）
VIZ_BBOX3D = True       # 3D 包围框（自车真实+纯车宽参考虚线；障碍真实+余量虚线）
VIZ_GAP_BAND = True     # 障碍两侧车身可容空间带（绿=带宽≥车宽/红=不够）+ 可容宽标注
VIZ_EDGE_MARGIN = True  # 可容带外侧 EDGE 边界余量保留区（黄带 + 余量标注）
VIZ_EGO_CLEAR = True    # 自车所需通过宽度标注（车宽 X.Xm）

# 帧缓存：slot -> (b64, surface)，避免同一帧重复 JPEG 解码
_FRAME_CACHE: dict = {}


def _slot_surf(surfaces: dict, slot: str):
    b64 = surfaces.get(slot)
    if not b64:
        return None
    if _FRAME_CACHE.get(slot) == (b64,):
        return _FRAME_CACHE.get("_surf_" + slot)
    raw = decode_frame_b64(b64)
    if raw is None:
        return None
    surf = img_bytes_to_surface(raw)
    _FRAME_CACHE["_surf_" + slot] = surf
    _FRAME_CACHE[slot] = (b64,)
    return surf


def _placeholder(screen, font, rect, text):
    pygame.draw.rect(screen, (15, 18, 28), rect)
    cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2
    # 长文本自动截断，避免越界
    if font.size(text)[0] > rect[2] - 24:
        while text and font.size(text + "…")[0] > rect[2] - 24:
            text = text[:-1]
        text += "…"
    s = font.render(text, True, (90, 105, 130))
    screen.blit(s, (cx - s.get_width() // 2, cy - s.get_height() // 2))


def _loading_text(exp, default="等待画面"):
    """启动 loading 占位文案：优先显示服务端推送的启动日志（对齐前端 starting 状态）。"""
    if not isinstance(exp, dict):
        return default
    if (exp.get("status") in ("done", "stopped", "error")
            or "result" in exp or "report" in exp):
        return "实验已结束"
    log = exp.get("log")
    if log:
        return f"启动中… {log}"
    return default


def _blit_fit(screen, surf, rect):
    """等比缩放居中显示（letterbox，无任何装饰）。"""
    iw, ih = surf.get_size()
    if iw <= 0 or ih <= 0:
        return
    scale = min(rect[2] / iw, rect[3] / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    scaled = pygame.transform.smoothscale(surf, (nw, nh)) if (nw, nh) != (iw, ih) else surf
    screen.blit(scaled, (rect[0] + (rect[2] - nw) // 2, rect[1] + (rect[3] - nh) // 2))


def _blit_cover(screen, surf, rect):
    """等比缩放铺满矩形并居中裁剪（cover）。"""
    iw, ih = surf.get_size()
    if iw <= 0 or ih <= 0:
        return
    scale = max(rect[2] / iw, rect[3] / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    scaled = pygame.transform.smoothscale(surf, (nw, nh)) if (nw, nh) != (iw, ih) else surf
    screen.blit(scaled, (rect[0] + (rect[2] - nw) // 2, rect[1] + (rect[3] - nh) // 2))


def _hud_panel(screen, rect):
    pygame.draw.rect(screen, HUD_BG, rect)


def _hud_lines(screen, font, rect, lines, done_text=None):
    _hud_panel(screen, rect)
    y = rect[1] + 10
    for ln in lines:
        surf = font.render(ln, True, HUD_TEXT)
        screen.blit(surf, (rect[0] + 14, y))
        y += surf.get_height() + 6
    if done_text:
        s = font.render(done_text, True, (255, 220, 120))
        screen.blit(s, (rect[0] + 14, rect[1] + rect[3] - 30))


def _dashed_polyline(screen, color, pts, dash=8, gap=6, width=2):
    """沿折线画虚线。"""
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        t = 0.0
        while t < seg:
            t2 = min(t + dash, seg)
            pygame.draw.line(screen, color,
                             (x0 + (x1 - x0) * t / seg, y0 + (y1 - y0) * t / seg),
                             (x0 + (x1 - x0) * t2 / seg, y0 + (y1 - y0) * t2 / seg), width)
            t = t2 + gap


def _cover_pt(rect, sw, sh, u, v):
    """cover 映射：源图像素(u,v) → 目标 rect 内坐标（与 _blit_cover 同规则）。"""
    x, y, rw, rh = rect
    scale = max(rw / sw, rh / sh)
    nw, nh = sw * scale, sh * scale
    ox = x + (rw - nw) / 2
    oy = y + (rh - nh) / 2
    return ox + u * scale, oy + v * scale


def _overlay_segs(screen, rect, src, frames):
    """用 payload 下发的 3D 框投影线段即时绘制（同车道线通道，避免 JPEG 闪烁）。
    frames: [{segs:[[[u,v],[u,v]]...], color:[r,g,b], dashed:bool}], src=(sw,sh)。"""
    sw, sh = src
    for f in frames or []:
        col = tuple(f.get("color") or (255, 255, 255))
        dashed = bool(f.get("dashed"))
        for a, b in f.get("segs") or []:
            p1 = _cover_pt(rect, sw, sh, a[0], a[1])
            p2 = _cover_pt(rect, sw, sh, b[0], b[1])
            if dashed:
                _dashed_polyline(screen, col, [p1, p2], 7, 5, 2)
            else:
                pygame.draw.line(screen, col, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), 2)


# ═══════════════════════════════════════════════════════════════════
# 实验 23 定位分析
# ═══════════════════════════════════════════════════════════════════
def render_localization(screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx):
    w, h = screen.get_size()
    # 前视相机上移并加高，让位于下方图表与状态面板，互不压盖
    video = (8, 8, w - 16, 430)
    surf = _slot_surf(surfaces, "camera")
    if surf is not None:
        _blit_cover(screen, surf, video)
    else:
        _placeholder(screen, fonts["md"], video, _loading_text(exp, "等待前视相机画面…"))

    chart_y = 452
    chart_h = 300
    hist.exp23_err.draw(screen, fonts["sm"], (8, chart_y, (w - 24) // 2, chart_h))
    hist.exp23_traj.draw(screen, fonts["sm"], (16 + (w - 24) // 2, chart_y, (w - 24) // 2, chart_h))

    pt = exp.get("trajectory") or {}
    res = exp.get("result") or {}
    lines = [
        f"进度 {pt.get('progress', 0):.0f}% · 时刻 {pt.get('t', 0):.1f}s",
        f"定位误差 {pt.get('error', 0):.2f} m · 横向误差 {pt.get('cte', 0):.2f} m · 转向 {pt.get('steer', 0):.2f}",
        f"GNSS {'有效' if pt.get('gnss_x') is not None else '无效（INS 推算）'}",
    ]
    if res:
        lines.insert(0, f"完成 · RMSE {res.get('rmse', '—')} m · 平均横向 {res.get('avg_cte', '—')} m · "
                        f"最大 {res.get('max_cte', '—')} m · 压线 {res.get('offlane_pct', '—')}%")
    done = "实验结束（ESC 退出）" if exp.get("status") in ("done", "stopped") else (
        "出错：" + str(exp.get("message", "")) if exp.get("status") == "error" else None)
    hud_y = chart_y + chart_h + 16
    _hud_lines(screen, fonts["sm"], (8, hud_y, w - 16, h - hud_y - 8), lines, done)


# ═══════════════════════════════════════════════════════════════════
# 实验 4 Lidar 检测（三目无缝拼接全景 + 点云投影叠加）
# ═══════════════════════════════════════════════════════════════════
def _panorama_surfaces(surfaces):
    """取左/前/右三目画面；缺任一则返回 None。"""
    parts = []
    for slot in ("cameraL", "camera", "cameraR"):
        s = _slot_surf(surfaces, slot)
        if s is None:
            return None
        parts.append(s)
    return parts


# 点云深度色阶（对齐前端 COLOR_STOPS：近红/橙 → 远绿，随 LiDAR 量程归一化）
_LIDAR_STOPS = [(0.0, (255, 5, 30)), (0.08, (255, 80, 30)), (0.22, (255, 170, 40)),
                (0.4, (80, 210, 60)), (0.7, (20, 220, 80)), (1.0, (20, 230, 80))]


def _depth_color(d: float, max_range: float):
    nd = min(1.0, max(0.0, d / max(1.0, max_range)))
    lo, hi = _LIDAR_STOPS[0], _LIDAR_STOPS[-1]
    for i in range(1, len(_LIDAR_STOPS)):
        if nd <= _LIDAR_STOPS[i][0]:
            hi = _LIDAR_STOPS[i]
            lo = _LIDAR_STOPS[i - 1]
            break
    t = (nd - lo[0]) / (hi[0] - lo[0]) if hi[0] > lo[0] else 0.0
    t = min(1.0, max(0.0, t))
    return tuple(int(lo[1][k] + (hi[1][k] - lo[1][k]) * t) for k in range(3))


def _paint_projection(screen, traj: dict, layout, max_range: float):
    """把 LiDAR→三目相机投影点按深度着色画到拼接全景上（对齐前端 paintProjection）。

    layout: (x0, y0, pane_w, pane_h, cam_w, cam_h, scale)
    """
    cams = traj.get("cameras") or []
    if not cams:
        return
    x0, y0, pw, ph, cam_w, cam_h, s = layout
    r = max(1, round(2.2 * s))
    key_idx = {"left": 0, "front": 1, "right": 2}
    for cam in cams:
        idx = key_idx.get(cam.get("key"))
        if idx is None:
            continue
        ox = x0 + idx * pw
        for pt in cam.get("projection") or []:
            u, v, d = pt.get("u"), pt.get("v"), pt.get("d")
            if u is None or v is None:
                continue
            px = ox + u * s
            py = y0 + v * s
            if ox <= px < ox + pw and y0 <= py < y0 + ph:
                pygame.draw.circle(screen, _depth_color(d or 0.0, max_range),
                                   (int(px), int(py)), r)


def render_lidar(screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx):
    w, h = screen.get_size()
    video = (8, 8, w - 16, 330)

    parts = _panorama_surfaces(surfaces)
    if parts is not None:
        # 三目等比无缝拼接：统一缩放系数，水平排列（对齐前端 3×CAM_W 拼接）
        cam_w, cam_h = parts[0].get_size()
        s = min(video[2] / (cam_w * 3), video[3] / cam_h)
        pw, ph = max(1, int(cam_w * s)), max(1, int(cam_h * s))
        total_w = pw * 3
        x0 = video[0] + (video[2] - total_w) // 2
        y0 = video[1] + (video[3] - ph) // 2
        for i, p in enumerate(parts):
            scaled = pygame.transform.smoothscale(p, (pw, ph)) if (pw, ph) != p.get_size() else p
            screen.blit(scaled, (x0 + i * pw, y0))
        # LiDAR 点云投影叠加（深度着色）
        traj = exp.get("trajectory") or {}
        _paint_projection(screen, traj, (x0, y0, pw, ph, cam_w, cam_h, s),
                          float(ctx.get("params", {}).get("range", 50)))
    else:
        _placeholder(screen, fonts["md"], video, _loading_text(exp, "等待三目相机画面…"))

    chart_y = 352
    chart_h = 300
    hist.exp4_obstacle.draw(screen, fonts["sm"], (8, chart_y, (w - 24) // 2, chart_h))
    hist.exp4_points.draw(screen, fonts["sm"], (16 + (w - 24) // 2, chart_y, (w - 24) // 2, chart_h))

    traj = exp.get("trajectory") or {}
    res = exp.get("result") or {}
    near = traj.get("nearest_front_obstacle_m")
    lines = [
        f"进度 {traj.get('progress', 0):.0f}% · 时刻 {traj.get('t', 0):.0f}s",
        f"前方最近障碍 {('%.1f m' % near) if near is not None else '—'}"
        f"（{traj.get('nearest_obstacle_type', '无')}）· 前向有效点 {traj.get('front_point_count', '—')}",
        f"每圈点数≈{traj.get('total', '—')} · 每线每圈点数(PLPR) {traj.get('plpr', '—')}",
    ]
    if res:
        lines.insert(0, f"完成 · 采样 {res.get('rows', '—')} 行 · 用时 {res.get('elapsed', '—')}s")
    done = "实验结束（ESC 退出）" if exp.get("status") in ("done", "stopped") else (
        "出错：" + str(exp.get("message", "")) if exp.get("status") == "error" else None)
    hud_y = chart_y + chart_h + 14
    _hud_lines(screen, fonts["sm"], (8, hud_y, w - 16, h - hud_y - 8), lines, done)


# ═══════════════════════════════════════════════════════════════════
# 实验 5 语义分割
# ═══════════════════════════════════════════════════════════════════
def render_semantic(screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx):
    w, h = screen.get_size()
    # 顶部：RGB + 语义 双画面对齐铺满整行
    pane_w = (w - 24) // 2
    for i, slot in enumerate(("camera", "semantic")):
        rect = (8 + i * (pane_w + 8), 8, pane_w, 400)
        surf = _slot_surf(surfaces, slot)
        if surf is not None:
            _blit_cover(screen, surf, rect)
        else:
            _placeholder(screen, fonts["md"], rect, _loading_text(exp, "等待画面…"))

    # 中部：占比图横跨整行
    chart_y = 420
    chart_h = 230
    hist.exp5_ratio.draw(screen, fonts["sm"], (8, chart_y, w - 16, chart_h))

    # 底部：状态面板整行叠放（图表在上、状态在下，上下布局），填满下方空白
    traj = exp.get("trajectory") or {}
    res = exp.get("result") or {}
    # 等级以服务端轨迹推送为准（L2/L3），没有轨迹时回退到参数配置
    level = traj.get("level") or ctx.get("params", {}).get("level", "—")
    lines = [
        f"自动驾驶等级 {level}",
        f"进度 {traj.get('progress', 0):.0f}% · 时刻 {traj.get('t', 0):.0f}s · 帧 {traj.get('frame', '—')}",
    ]
    # 当前占比（取前 4 个非零类别）
    ratios = [(k[:-len("_ratio")], v) for k, v in traj.items()
              if k.endswith("_ratio") and isinstance(v, (int, float))]
    ratios.sort(key=lambda kv: -kv[1])
    if ratios:
        lines.append("占比 " + " · ".join(f"{k} {v * 100:.0f}%" for k, v in ratios[:4]))
    if res:
        lines.insert(0, f"完成 · 采样 {res.get('rows', '—')} 行 · 用时 {res.get('elapsed', '—')}s")
    done = "实验结束（ESC 退出）" if exp.get("status") in ("done", "stopped") else (
        "出错：" + str(exp.get("message", "")) if exp.get("status") == "error" else None)
    hud_y = chart_y + chart_h + 16
    _hud_lines(screen, fonts["sm"], (8, hud_y, w - 16, h - hud_y - 8), lines, done)


# ═══════════════════════════════════════════════════════════════════
# 实验 10 综合驾驶：鸟瞰叠加 + 包围框 + 图表
# ═══════════════════════════════════════════════════════════════════
_LANE_BBOX_CACHE = {"lanes": None, "bbox": []}


def _lane_bboxes(lanes):
    if _LANE_BBOX_CACHE["lanes"] is lanes:
        return _LANE_BBOX_CACHE["bbox"]
    bbs = []
    for lane in lanes:
        pts = lane.get("pts", [])
        if not pts:
            bbs.append(None)
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        bbs.append((min(xs), min(ys), max(xs), max(ys)))
    _LANE_BBOX_CACHE["lanes"] = lanes
    _LANE_BBOX_CACHE["bbox"] = bbs
    return bbs


def _draw_bird_overlay(screen, rect, exp, lanes, font=None):
    """在鸟瞰相机画面上叠加车道 / 参考线 / 预测线 / 障碍物（对齐前端 drawLaneBev）。

    鸟瞰相机：960×960、FOV 90°、挂车顶 z=45 正下视 → 图像中心是自车，
    K = 480/45 px/m；画面随车旋转（挂接车辆），用真值航向对齐世界系。
    """
    pos = exp.get("gt") or exp.get("fused")
    if not pos or not lanes:
        return
    pw, ph = rect[2], rect[3]
    s = min(pw, ph) / 960.0
    ox = rect[0] + (pw - 960 * s) / 2
    oy = rect[1] + (ph - 960 * s) / 2
    K = 480.0 / 45.0
    hr = math.radians(exp.get("gt_yaw", exp.get("heading", 0.0)) or 0.0)
    fx, fy = math.cos(hr), math.sin(hr)
    rx, ry = -math.sin(hr), math.cos(hr)

    def to_scr(x, y):
        dx, dy = x - pos["x"], y - pos["y"]
        fr = dx * fx + dy * fy      # 前向分量 → 图像上方
        rr = dx * rx + dy * ry      # 右向分量 → 图像右侧
        return (ox + (480 + K * rr) * s, oy + (480 - K * fr) * s)

    cx, cy = ox + 480 * s, oy + 480 * s
    # 半透明叠加层（车道底色带）
    overlay = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)

    def band_poly(lane, hw):
        pts = lane.get("pts", [])
        if len(pts) < 2:
            return None
        left, right = [], []
        for k, a in enumerate(pts):
            b = pts[min(k + 1, len(pts) - 1)]
            c = pts[max(k - 1, 0)]
            tx, ty = b["x"] - c["x"], b["y"] - c["y"]
            tl = math.hypot(tx, ty) or 1.0
            tx, ty = tx / tl, ty / tl
            nx, ny = -ty, tx
            left.append(to_scr(a["x"] + nx * hw, a["y"] + ny * hw))
            right.append(to_scr(a["x"] - nx * hw, a["y"] - ny * hw))
        return left + right[::-1]

    def draw_band(key, rgba, stroke_rgba):
        if not key:
            return
        bbs = _lane_bboxes(lanes)
        for i, lane in enumerate(lanes):
            if lane.get("road") != key[0] or lane.get("lane") != key[1]:
                continue
            bb = bbs[i]
            if bb is None:
                continue
            # 只画自车 60m 内的车道
            dx = max(bb[0] - pos["x"], 0, pos["x"] - bb[2])
            dy = max(bb[1] - pos["y"], 0, pos["y"] - bb[3])
            if dx * dx + dy * dy > 60 * 60:
                continue
            hw = (lane.get("width") or 3.5) / 2
            poly = band_poly(lane, hw)
            if poly and len(poly) >= 3:
                pygame.draw.polygon(overlay, rgba, poly)
                pygame.draw.polygon(overlay, stroke_rgba, poly, 1)

    lane_info = exp.get("lane") or {}
    cur, tgt = lane_info.get("cur"), lane_info.get("tgt")
    if VIZ_LANE_BAND:
        for pl in (lane_info.get("plan") or []):
            key = pl.get("lane") if isinstance(pl, dict) else None
            if key and key != cur and key != tgt:
                fade = max(0.25, 1 - (pl.get("dist", 0) or 0) / 60)
                draw_band(key, (54, 89, 255, int(36 * fade)), (54, 89, 255, int(100 * fade)))
        if cur:
            draw_band(cur, (54, 89, 255, 55), (54, 89, 255, 150))
        if tgt and tgt != cur:
            draw_band(tgt, (0, 180, 42, 60), (0, 180, 42, 190))
    screen.blit(overlay, (rect[0], rect[1]))

    # 车道边界线（实线 / 虚线）
    if VIZ_LANE_EDGE:
        bbs = _lane_bboxes(lanes)
        for i, lane in enumerate(lanes):
            bb = bbs[i]
            if bb is None:
                continue
            dx = max(bb[0] - pos["x"], 0, pos["x"] - bb[2])
            dy = max(bb[1] - pos["y"], 0, pos["y"] - bb[3])
            if dx * dx + dy * dy > 60 * 60:
                continue
            hw = (lane.get("width") or 3.5) / 2
            for side in (1, -1):
                pts = lane.get("pts", [])
                edge = []
                for k, a in enumerate(pts):
                    b = pts[min(k + 1, len(pts) - 1)]
                    c = pts[max(k - 1, 0)]
                    tx, ty = b["x"] - c["x"], b["y"] - c["y"]
                    tl = math.hypot(tx, ty) or 1.0
                    nx, ny = -ty / tl * side, tx / tl * side
                    edge.append(to_scr(a["x"] + nx * hw, a["y"] + ny * hw))
                if len(edge) < 2:
                    continue
                if lane.get("marking") == "Broken":
                    _dashed_polyline(screen, (255, 255, 255), edge, 6 * max(s, 0.5), 6, 1)
                else:
                    pygame.draw.lines(screen, (255, 255, 255), False, edge, 1)

    # 参考路线（蓝实线）
    if VIZ_REF_LINE:
        ref = exp.get("ref_path") or []
        if len(ref) > 1:
            pygame.draw.lines(screen, (86, 156, 250), False,
                              [to_scr(p["x"], p["y"]) for p in ref], max(2, round(2.5 * s)))
    # 预测轨迹（橙虚线）
    if VIZ_PRED_LINE:
        pred = exp.get("pred_path") or []
        if len(pred) > 1:
            _dashed_polyline(screen, (255, 125, 0), [to_scr(p["x"], p["y"]) for p in pred], 8, 5, 2)
    # 障碍物（红块）
    if VIZ_OBSTACLE:
        for ob in (exp.get("obstacles") or []):
            q = to_scr(ob.get("x", 0), ob.get("y", 0))
            sz = max(1.5, ob.get("size") or 2) * K * s
            r = pygame.Rect(q[0] - sz / 2, q[1] - sz / 2, sz, sz)
            pygame.draw.rect(screen, (245, 63, 63), r)
            pygame.draw.rect(screen, (255, 213, 213), r, 1)
        for ob in (exp.get("planned_obstacles") or []):
            q = to_scr(ob.get("x", 0), ob.get("y", 0))
            sz = 2.0 * K * s
            r = pygame.Rect(q[0] - sz / 2, q[1] - sz / 2, sz, sz)
            pygame.draw.rect(screen, (245, 120, 63), r, 1)
    # 感知范围圈（约 50m）
    if VIZ_RANGE:
        pygame.draw.circle(screen, (120, 132, 158), (int(cx), int(cy)), int(50 * K * s), 1)
    # 自车标记
    if VIZ_EGO_MARK:
        pygame.draw.circle(screen, (78, 139, 255), (int(cx), int(cy)), max(5, int(7 * s)), 1)

    # 障碍两侧车身可容空间带（后端 build_gap_viz 下发，口径=simple_planner）+
    # 可容宽标注；绿=带宽≥车宽(可过)，红=不够
    if VIZ_GAP_BAND:
        gv = exp.get("gap_viz") or {}
        for sd in gv.get("sides") or []:
            poly = [to_scr(p[0], p[1]) for p in sd.get("poly", [])]
            if len(poly) < 3:
                continue
            if sd.get("pass"):
                fill, stroke = (60, 210, 90, 70), (60, 210, 90)
            else:
                fill, stroke = (245, 63, 63, 70), (245, 63, 63)
            band = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
            pygame.draw.polygon(band, fill, poly)
            pygame.draw.polygon(band, stroke, poly, 1)
            screen.blit(band, (rect[0], rect[1]))
            if font is not None:
                lp = to_scr(sd["label"][0], sd["label"][1])
                txt = font.render(f"{sd['width']:.1f}", True, stroke)
                screen.blit(txt, (int(lp[0] - txt.get_width() / 2),
                                  int(lp[1] - txt.get_height() / 2)))

    # EDGE 边界余量保留区（可容带外侧 → 可行驶域边界，本车不会进入；黄带）
    if VIZ_EDGE_MARGIN:
        gv = exp.get("gap_viz") or {}
        for sd in gv.get("sides") or []:
            ep = sd.get("edge_poly") or []
            if len(ep) < 3:
                continue
            poly = [to_scr(p[0], p[1]) for p in ep]
            band = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
            pygame.draw.polygon(band, (255, 200, 60, 45), poly)
            pygame.draw.polygon(band, (255, 200, 60), poly, 1)
            screen.blit(band, (rect[0], rect[1]))
            # if font is not None:
            #     mx = sum(p[0] for p in poly) / len(poly)
            #     my = sum(p[1] for p in poly) / len(poly)
            #     txt = font.render(f"余量 {sd.get('edge_margin', 0):.0f}m",
            #                       True, (255, 200, 60))
            #     screen.blit(txt, (int(mx - txt.get_width() / 2),
            #                       int(my - txt.get_height() / 2)))

    # 自车车宽标注（真实框宽度 = 2×ego_half_w；与可容带带宽对比判断能否通过）
    if VIZ_EGO_CLEAR and font is not None:
        req_w = (exp.get("gap_viz") or {}).get("ego_req_w")
        if req_w:
            txt = font.render(f"车宽 {req_w}m", True, (0, 200, 255))
            screen.blit(txt, (int(cx) + 10, int(cy) + 8))


def render_comprehensive(screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx):
    w, h = screen.get_size()
    # 顶部左鸟瞰、右车前相机（加大高度，底部图表整体下移，互不压盖）
    top_h = 560
    bird_rect = (0, 0, 630, top_h)
    bbox_rect = (726, 0, w - 726, top_h + 35)

    bird = _slot_surf(surfaces, "bird")
    if bird is not None:
        _blit_cover(screen, bird, bird_rect)
        if exp.get("status") == "running":
            _draw_bird_overlay(screen, bird_rect, exp, ctx.get("lanes") or [], fonts["sm"])
            # 3D 框投影线段：与车道线同通道即时绘制，稳定不闪
            if VIZ_BBOX3D:
                _overlay_segs(screen, bird_rect, bird.get_size(), exp.get("bird3d"))
    else:
        _placeholder(screen, fonts["md"], bird_rect, _loading_text(exp, "等待车顶俯瞰画面…"))

    bbox = _slot_surf(surfaces, "bbox") or _slot_surf(surfaces, "camera")
    if bbox is not None:
        _blit_cover(screen, bbox, bbox_rect)
        if exp.get("status") == "running" and VIZ_BBOX3D:
            _overlay_segs(screen, bbox_rect, bbox.get_size(), exp.get("bbox3d"))
    else:
        _placeholder(screen, fonts["md"], bbox_rect, _loading_text(exp, "等待车前包围框画面…"))

    st = exp.get("status")
    # 结束信息占位行（仅在结束后显示）
    if st in ("done", "stopped", "error"):
        rep = exp.get("report") or {}
        if st == "done":
            msg = (f"实验结束 · 到达={'是' if exp.get('arrived') else '否'} · "
                   f"里程 {rep.get('distance_m', '—')}m · 碰撞 {rep.get('collisions', {}).get('count', 0)} 次")
        elif st == "stopped":
            msg = "已停止（ESC 退出）"
        else:
            msg = "出错：" + str(exp.get("message", ""))
        s = fonts["md"].render(msg, True, (255, 220, 120))
        screen.blit(s, (16, 578))

    # 运行状态条
    st_rect = (0, 595, w, 26)
    pygame.draw.rect(screen, HUD_BG, st_rect)
    avoid = exp.get("avoid") or {}
    tl = exp.get("traffic_light") or {}
    stats = (f"t {exp.get('t', 0):.1f}s · 进度 {exp.get('progress', 0):.0f}% · "
             f"速度 {exp.get('speed', 0):.1f}/{exp.get('desired_speed', 0):.1f} m/s · "
             f"CTE {exp.get('cte', 0):.2f}m · 定位误差 {exp.get('loc_err', 0):.2f}m · "
             f"障碍 {exp.get('front_obstacle') if exp.get('front_obstacle') is not None else '—'}m · "
             f"信号灯 {tl.get('state', '—')}/{tl.get('distance', '—')}m · "
             f"绕行 {'是(' + str(avoid.get('side', '')) + ')' if avoid.get('active') else '否'} · "
             f"感知 {'bbox 相机' if exp.get('perception') else '世界真值'}")
    screen.blit(fonts["sm"].render(stats, True, HUD_TEXT), (st_rect[0] + 10, st_rect[1] + 5))

    # 底部三图：速度 / 误差 / 轨迹
    chart_y = 620
    ch = h - chart_y
    cw = (w) // 3
    hist.exp10_speed.draw(screen, fonts["sm"], (0, chart_y, cw, ch))
    hist.exp10_err.draw(screen, fonts["sm"], (0 + cw, chart_y, cw, ch))
    hist.exp10_traj.draw(screen, fonts["sm"], (0 + cw * 2, chart_y, w - cw * 2, ch))


def dispatch_render(exp_id: int, screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx) -> None:
    if exp_id == 10:
        render_comprehensive(screen, fonts, surfaces, exp, hist, ctx)
    elif exp_id == 5:
        render_semantic(screen, fonts, surfaces, exp, hist, ctx)
    elif exp_id == 4:
        render_lidar(screen, fonts, surfaces, exp, hist, ctx)
    else:
        render_localization(screen, fonts, surfaces, exp, hist, ctx)