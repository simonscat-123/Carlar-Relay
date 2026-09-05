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

import base64
import math

import pygame

from .charts import PANEL_BG, PANEL_BORDER, TEXT, TEXT_DIM, TelemetryHistory
from .relay_client import decode_frame_b64, img_bytes_to_surface

HUD_BG = (28, 33, 48)
HUD_TEXT = (215, 225, 245)
HUD_DIM = (140, 152, 176)
OK_COLOR = (120, 220, 150)

# ── 鸟瞰叠加已由后端烧录进画面，本地不再有鸟瞰叠加开关 ──────────────
VIZ_BBOX3D = True        # 右侧前置相机 3D 包围框（保留前端 overlay 通道）

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


def _draw_image_panel(screen, rect, surf, title, font):
    """等大相机面板：背景 + 边框 + 标题 + cover 居中画面（与 BEV 面板同构）。"""
    pygame.draw.rect(screen, PANEL_BG, rect)
    pygame.draw.rect(screen, PANEL_BORDER, rect, 1)
    if title:
        screen.blit(font.render(title, True, TEXT), (rect[0] + 8, rect[1] + 5))
    inner = (rect[0] + 8, rect[1] + 30, rect[2] - 16, rect[3] - 38)
    if inner[2] < 4 or inner[3] < 4:
        return
    if surf is not None:
        prev = screen.get_clip()
        screen.set_clip(inner)
        _blit_cover(screen, surf, inner)
        screen.set_clip(prev)
    else:
        _placeholder(screen, font, inner, "等待画面…")


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
    frames: [{segs:[[[u,v],[u,v]]...], color:[r,g,b], dashed:bool}]，u/v 为归一化
    [0,1]（对应该帧原图 UV）；src=(sw,sh) 为本端实际展示的帧像素尺寸，先乘回
    像素再 cover 映射到 rect。因 uv 是归一化坐标，任意展示缩放/适配下均精确对齐，
    与 web 端（contain 适配 × 各自帧尺寸）用同一套几何，双端结果一致。此处再
    裁剪到 rect，避免盒子角点投影越界时把边框画到相机画面边界之外。"""
    sw, sh = src
    prev_clip = screen.get_clip()
    screen.set_clip(rect)
    try:
        for f in frames or []:
            col = tuple(f.get("color") or (255, 255, 255))
            dashed = bool(f.get("dashed"))
            for a, b in f.get("segs") or []:
                p1 = _cover_pt(rect, sw, sh, a[0] * sw, a[1] * sh)
                p2 = _cover_pt(rect, sw, sh, b[0] * sw, b[1] * sh)
                if dashed:
                    _dashed_polyline(screen, col, [p1, p2], 7, 5, 2)
                else:
                    pygame.draw.line(screen, col, (int(p1[0]), int(p1[1])),
                                     (int(p2[0]), int(p2[1])), 2)
    finally:
        screen.set_clip(prev_clip)


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
# ── 实验5：语义分割 → 感知融合 → BEV 占据栅格世界 → 规划决策 ─────────
# BEV 单元类别 → 颜色（对齐服务端 bev.py 的类别编码 0/1/2/3/4）
_BEV_CELL_COLORS = {
    0: (26, 30, 44),    # 未知 / 未感知
    1: (50, 190, 118),  # 可行驶
    2: (180, 126, 54),  # 静态障碍
    3: (240, 96, 96),   # 车辆
    4: (210, 132, 232), # 行人
}
_BEV_CACHE = {}         # cell 数据(base64) → 小尺寸栅格 Surface（避免逐帧重建）


def _bev_small_surface(bev):
    """把 base64 栅格解码为 G×G 小 Surface；返回 (surface_or_None, G)。"""
    data = bev.get("data")
    G = bev.get("w") or 0
    if not data or G <= 0:
        return None, 0
    if data in _BEV_CACHE:
        return _BEV_CACHE[data], G
    try:
        raw = base64.b64decode(data)
    except Exception:
        return None, 0
    if len(_BEV_CACHE) > 8:
        _BEV_CACHE.clear()
    surf = pygame.Surface((G, G))
    for idx, b in enumerate(raw):
        a0, a1 = idx // G, idx % G
        # lat=车右（与综合驾驶鸟瞰/官方坐标系一致）：a1→屏x(车右→右)，前向 a0→屏上
        surf.set_at((a1, G - 1 - a0), _BEV_CELL_COLORS.get(b, (26, 30, 44)))
    _BEV_CACHE[data] = surf
    return surf, G


def _draw_bev_panel(screen, rect, traj, font):
    """BEV 鸟瞰栅格面板 + 规划轨迹 / 自车标记叠加。

    仅前方 90° 做了感知融合，画面数据集中在前向可视锥内；自车标记下移至
    面板 4/5 高度，使纵向可视锥露出的可行驶 / 障碍区域更大，便于观察决策。
    """
    pygame.draw.rect(screen, PANEL_BG, rect)
    pygame.draw.rect(screen, PANEL_BORDER, rect, 1)
    screen.blit(font.render("BEV 前融合", True, TEXT), (rect[0] + 8, rect[1] + 5))
    inner = (rect[0] + 8, rect[1] + 30, rect[2] - 16, rect[3] - 38)
    bev = traj.get("bev") or {}
    surf, G = _bev_small_surface(bev)
    if surf is None or inner[2] < 4 or inner[3] < 4:
        _placeholder(screen, font, inner, _loading_text(traj, "等待画面…"))
        return
    span = bev.get("span") or 60.0
    res = (bev.get("res") or 0.5) or 0.5            # 栅格分辨率（m），兜底 0.5
    pm = inner[2] / span                            # 米 → 像素
    cx = inner[0] + inner[2] / 2
    cy = inner[1] + inner[3] * 0.8                  # 自车下移，前方可视锥露出更多

    # 底图栅格同样以 (cx, cy) 为自车中心、按米→像素精确摆放（不再中心平铺），
    # 使栅格自车行 / 叠加轨迹 / 自车标记完全对齐
    cell_px = pm * res
    gpx = int(G * cell_px)
    g0x, g0y = int(cx - gpx / 2), int(cy - gpx / 2)
    if gpx != (inner[2], inner[3]):
        g = pygame.transform.smoothscale(surf, (gpx, gpx))
    prev_clip = screen.get_clip()

    def sc(x, y):                                   # 车体系(x前向, y右) → 屏幕（右在右）
        return (cx + y * pm, cy - x * pm)

    screen.set_clip(inner)
    screen.blit(g, (g0x, g0y))
    # 车道线叠加（只画白线，参考综合驾驶鸟瞰；data 随 SSE lane_edges 下发）
    for _gm in (traj.get("lane_edges") or []):
        for _ekey in ("left", "right"):
            _line = _gm.get(_ekey) or []
            if len(_line) >= 2:
                pygame.draw.lines(screen, (245, 245, 250), False,
                                  [sc(p[0], p[1]) for p in _line], 2)
    screen.set_clip(prev_clip)

    # 自车标记（点 + 朝向小三角）
    ex, ey = sc(0.0, 0.0)
    pygame.draw.circle(screen, (120, 220, 255), (int(ex), int(ey)), 4)
    pygame.draw.polygon(screen, (120, 220, 255),
                        [(ex, ey - 8), (ex - 4, ey - 2), (ex + 4, ey - 2)])
    # 决策角标（右上角）
    dec = traj.get("decision") or {}
    if dec.get("label"):
        lbl = font.render(dec["label"], True, (255, 230, 90))
        screen.blit(lbl, (rect[0] + rect[2] - lbl.get_width() - 8, rect[1] + 5))


_OBSTACLE_CLS = {"vehicle": "车辆", "walker": "行人", "rider": "骑手"}


def _draw_obstacle_list(screen, font, rect, obstacles, total):
    """障碍物列表表格：ID / 类型 / 距离(m)，按距离优先（近→远）最多 5 项，
    右上角显示识别总数。替代原「检测目标数」折线图。"""
    x, y, w, h = rect
    # 右上角：识别总数
    total_txt = f"识别总数 {total}"
    ts = font.render(total_txt, True, (255, 220, 120))
    screen.blit(ts, (x + w - ts.get_width(), y))
    # 表头
    heads = [("ID", x), ("类型", x + 84), ("距离(m)", x + 170)]
    hy = y + 16
    for label, hx in heads:
        screen.blit(font.render(label, True, TEXT_DIM), (hx, hy))
    # 数据行（最多 5 项）
    row_h = 20
    for i, ob in enumerate(obstacles[:5]):
        ry = hy + 22 + i * row_h
        if ry > y + h - 4:
            break
        cls = ob.get("cls", "")
        col = (120, 200, 255) if cls == "vehicle" else (250, 170, 60) if cls == "walker" else TEXT
        screen.blit(font.render(str(ob.get("id", "—")), True, TEXT), (heads[0][1], ry))
        name = _OBSTACLE_CLS.get(cls, cls or "目标")
        screen.blit(font.render(name, True, col), (heads[1][1], ry))
        screen.blit(font.render(f"{ob.get('dist', 0):.1f}", True, TEXT),
                    (heads[2][1], ry))


# 障碍物表格列定义：(表头, 取值 key, 宽, 格式)。value 为行 dict 取数。
_OBST_COLS = [
    ("ID", "id", 44, "raw"),
    ("类型", "cls", 52, "cls"),
    ("距离m", "dist", 56, "1f"),
    ("x", "x", 56, "2f"), ("y", "y", 56, "2f"), ("z", "z", 56, "2f"),
    ("yaw", "yaw", 58, "3f"),
    ("长", "length", 50, "2f"), ("宽", "width", 50, "2f"), ("高", "height", 50, "2f"),
    ("vx", "vx", 52, "2f"), ("vy", "vy", 52, "2f"),
    ("ax", "ax", 52, "2f"), ("ay", "ay", 52, "2f"),
]


def _fmt_obst(ob, key, fmt):
    v = ob.get(key)
    if fmt == "raw":
        return "—" if v is None else str(v)
    if fmt == "cls":
        cls = str(v or "")
        return _OBSTACLE_CLS.get(cls, cls or "目标"), cls
    if fmt == "1f":
        return "—" if v is None else f"{v:.1f}"
    if fmt == "2f":
        return "—" if v is None else f"{v:.2f}"
    if fmt == "3f":
        return "—" if v is None else f"{v:.3f}"
    return "—"


def _draw_obstacle_full_table(screen, fonts, rect, obstacles, total, status):
    """整宽障碍物详细表格：pose(x,y,z,yaw) / size(l,w,h) / velocity(vx,vy) /
    acceleration(ax,ay)，按距离近→远最多 5 行。"""
    x, y, w, h = rect
    pygame.draw.rect(screen, PANEL_BG, rect)
    pygame.draw.rect(screen, PANEL_BORDER, rect, 1)
    cols = _OBST_COLS

    # 标题行 + 右上状态（识别总数 + 等级/决策/进度）
    screen.blit(fonts["md"].render("障碍物详细 (pose · size · velocity · accel)", True, TEXT),
                (x + 8, y + 4))
    st = fonts["sm"].render(status, True, (255, 220, 120))
    screen.blit(st, (x + w - st.get_width() - 8, y + 6))

    # 分组表头
    gy = y + 26
    gsplit = {5: "Pose", 9: "Size", 11: "Velocity(自车系)", 12: "Accel(自车系)"}
    g_labels = {}
    # 列宽按窗口可用宽度等比伸缩，避免右侧留白
    _tw = sum(cw for _l, _k, cw, _f in cols) or 1
    _K = (x + w - 8 - (x + 8)) / _tw        # 内容区宽 = 右缘-8 减 左缘+8
    col_x = []
    cx = x + 8
    for i, (lab, key, cw, fmt) in enumerate(cols):
        col_x.append(cx)
        if i in gsplit:
            g_labels[cx] = gsplit[i]
        cx += cw * _K
    # 画分组标签
    for gcx, glab in g_labels.items():
        t2 = fonts["sm"].render(glab, True, (150, 165, 195))
        screen.blit(t2, (gcx, gy))
    # 画每列表头
    hy = gy + 18
    for i, (lab, key, cw, fmt) in enumerate(cols):
        screen.blit(fonts["sm"].render(lab, True, TEXT_DIM), (col_x[i], hy))

    # 数据行（最多 5 行）
    row_h = 20
    r0 = hy + 22
    for i, ob in enumerate(obstacles[:5]):
        ry = r0 + i * row_h
        if ry > y + h - 2:
            break
        for j, (lab, key, cw, fmt) in enumerate(cols):
            tx = col_x[j]
            if fmt == "cls":
                name, clsk = _fmt_obst(ob, key, fmt)
                col = (120, 200, 255) if clsk == "vehicle" else (250, 170, 60) if clsk == "walker" else TEXT
                screen.blit(fonts["sm"].render(name, True, col), (tx, ry))
            else:
                screen.blit(fonts["sm"].render(_fmt_obst(ob, key, fmt), True, TEXT), (tx, ry))


def render_semantic(screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx):
    w, h = screen.get_size()
    gap, mg = 8, 8
    pane_w = (w - mg * 3) // 2            # 两列，各占约半宽
    row_h = 300
    row1_y, row2_y = 8, 8 + row_h + gap

    # 第1行：目标识别(相机叠框) | 语义分割
    for i, slot in enumerate(("camera", "semantic")):
        rect = (mg + i * (pane_w + gap), row1_y, pane_w, row_h)
        title = "目标识别" if slot == "camera" else "语义分割"
        _draw_image_panel(screen, rect, _slot_surf(surfaces, slot),
                          title, fonts["md"])

    traj = exp.get("trajectory") or {}

    # 第2行：深度相机 | BEV 占用栅格
    drect = (mg, row2_y, pane_w, row_h)
    _draw_image_panel(screen, drect, _slot_surf(surfaces, "depth"),
                      "LiDAR点云深度", fonts["md"])
    _draw_bev_panel(screen, (mg + (pane_w + gap), row2_y, pane_w, row_h),
                    traj, fonts["sm"])

    # 整宽：障碍物信息统计表（字段按窗口宽度自适应）
    ta_y = row2_y + row_h + gap
    tab_h = h - 8 - ta_y
    _dst = traj.get("decision") or None
    _table_status = f"识别总数 {traj.get('targets', 0)}" + (
        (" · 决策 " + _dst.get("state", "—")) if _dst else " · 等待感知")
    _draw_obstacle_full_table(screen, fonts, (mg, ta_y, w - 2 * mg, tab_h),
                              traj.get("obstacles") or [], traj.get("targets", 0), _table_status)


# ═══════════════════════════════════════════════════════════════════
# 实验 10 综合驾驶：鸟瞰叠加 + 包围框 + 图表
# ═══════════════════════════════════════════════════════════════════
# 鸟瞰叠加（当前车道/参考线/预测轨迹/障碍/自车/可容带）已由后端真实投影烧录进
# bird 帧，本地只 blit 该帧（见 render_comprehensive），不再有 _draw_bird_overlay。
def render_comprehensive(screen, fonts, surfaces, exp, hist: TelemetryHistory, ctx):
    w, h = screen.get_size()
    # 顶部左鸟瞰、右车前相机（加大高度，底部图表整体下移，互不压盖）
    top_h = 560
    bird_rect = (0, 0, 630, top_h)
    bbox_rect = (726, 0, w - 726, top_h + 35)

    bird = _slot_surf(surfaces, "bird")
    if bird is not None:
        # 鸟瞰叠加（当前车道/参考线/预测轨迹/障碍/自车/可容带）已由后端烧录进帧，
        # 本地只 blit 已画好的画面，不在 pygame 侧重复叠加。
        _blit_cover(screen, bird, bird_rect)
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