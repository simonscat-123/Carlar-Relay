"""综合驾驶(实验10)：规划阶段 pygame 交互。

对齐前端规划画布：
  - GET /map/render 拉取整城俯视实拍底图作为背景（有投影矩阵 H/Hinv 时
    用单应性做 世界↔图像 坐标变换，正交模式退化为线性映射）
  - 左键点选起点 → 终点，点完自动 POST /route/plan 规划并返回
  - 右键 / R 重置点选；ESC 取消退出
无底图时回退为道路矢量折线图。
"""
from __future__ import annotations

import math

import pygame

from .fonts import load_font
from .relay_client import RelayClient, RelayError, decode_frame_b64, img_bytes_to_surface


def _h_apply(H, x, y):
    """3x3 单应性矩阵（行主序 9 元素）作用于点 (x, y)。"""
    d = (H[6] * x + H[7] * y + H[8]) or 1.0
    return ((H[0] * x + H[1] * y + H[2]) / d,
            (H[3] * x + H[4] * y + H[5]) / d)


def _nearest_lane_point(lanes, wx, wy):
    """在车道折线中找距 (wx,wy) 最近的点，返回 (dist, x, y, yaw)。"""
    best = None
    for lane in lanes:
        pts = lane.get("pts", [])
        for i, p in enumerate(pts):
            d = math.hypot(p["x"] - wx, p["y"] - wy)
            if best is None or d < best[0]:
                nxt = pts[i + 1] if i + 1 < len(pts) else p
                yaw = math.degrees(math.atan2(nxt["y"] - p["y"], nxt["x"] - p["x"]))
                best = (d, p["x"], p["y"], yaw)
    return best


def _load_map_shot(client: RelayClient):
    """拉取 /map/render；返回 (surface|None, H, Hinv, img_w)。"""
    try:
        shot = client.map_render()
    except RelayError:
        return None, None, None, 1280
    if not isinstance(shot, dict) or not shot.get("base64"):
        return None, None, None, 1280
    raw = decode_frame_b64(shot["base64"])
    surf = img_bytes_to_surface(raw) if raw else None
    proj = shot.get("proj") or {}
    H = proj.get("H")
    Hinv = proj.get("Hinv")
    return surf, H, Hinv, shot.get("width") or 1280


def run(client: RelayClient, params: dict, lanes=None, width=1152, height=720) -> dict:
    """运行规划阶段；返回 {start, end, route, obstacles, sampling}。

    取消（ESC/关窗）时返回 {"start": None}，由调用方判断。
    """
    if not lanes:
        net = client.road_network()
        if not net.get("lanes"):
            raise RelayError("获取路网失败：" + str(net.get("message", "empty roads")))
        lanes = net["lanes"]

    # 由车道点集推世界范围（lanes 可能由调用方传入）
    allx = [p["x"] for l in lanes for p in l.get("pts", [])]
    ally = [p["y"] for l in lanes for p in l.get("pts", [])]
    if not allx:
        raise RelayError("路网为空，无法规划")
    bmin_x, bmax_x = min(allx), max(allx)
    bmin_y, bmax_y = min(ally), max(ally)

    print("[plan] 正在获取整城俯视底图（/map/render）…")
    map_surf, H, Hinv, img_w = _load_map_shot(client)
    if map_surf is not None:
        print(f"[plan] 底图已就绪 {img_w}px · 投影={'单应性' if H else '正交线性'}")
    else:
        print("[plan] 未获取到底图，回退道路矢量图")

    pygame.init()
    pygame.display.set_caption("综合驾驶 · 鸟瞰规划（左键点选起点→终点，右键重置，ESC 退出）")
    screen = pygame.display.set_mode((width, height))
    font = load_font(16)

    # 世界→屏幕映射（对齐前端 planTransform）：
    # 正方形画区 side=min(W,H)，世界范围取 x 方向 [min_x, max_x]（前端 worldRange 同款）
    pad = 16
    side = min(width, height) - pad * 2
    r_min, r_max = bmin_x, bmax_x
    scale = side / max(1e-6, r_max - r_min)
    ox = (width - side) / 2
    oy = (height - side) / 2
    S = scale * (r_max - r_min) / img_w     # 图像像素 → 屏幕像素
    use_h = bool(H and Hinv and map_surf is not None)

    def to_screen(x, y):
        if use_h:
            u, v = _h_apply(H, x, y)
            return (ox + u * S, oy + v * S)
        return (ox + (x - r_min) * scale, oy + (y - r_min) * scale)

    def to_world(sx, sy):
        if use_h:
            u = (sx - ox) / S
            v = (sy - oy) / S
            return _h_apply(Hinv, u, v)
        return (r_min + (sx - ox) / scale, r_min + (sy - oy) / scale)

    # 底图预缩放（正交：整图铺满画区；单应性：图像像素坐标 × S 直接对齐）
    base_scaled = None
    if map_surf is not None:
        base_scaled = pygame.transform.smoothscale(map_surf, (int(img_w * S), int(img_w * S)))

    routing = False
    start = None
    end = None
    route = None
    obstacles = []
    clock = pygame.time.Clock()
    error = None
    sampling = float(params.get("sampling_resolution", 2.0))

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return {"start": None}
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                return {"start": None}
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                start = end = route = None
                error = None
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and not routing:
                wx, wy = to_world(*ev.pos)
                hit = _nearest_lane_point(lanes, wx, wy)
                if hit is None:
                    continue
                _, x, y, yaw = hit
                if start is None:
                    start = {"x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 1)}
                elif end is None and (abs(x - start["x"]) > 2 or abs(y - start["y"]) > 2):
                    end = {"x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 1)}
                    routing = True   # 起终点齐了 → 自动规划
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                start = end = route = None
                error = None

        if routing:
            try:
                res = client.route_plan(start, end, sampling)
                route = res.get("route", [])
                obstacles = res.get("obstacles", [])
                routing = False
            except RelayError as exc:
                error = str(exc)
                routing = False
                start = end = route = None

        screen.fill((10, 13, 20))

        # ── 背景层 ──
        if base_scaled is not None:
            screen.blit(base_scaled, (ox, oy))
        else:
            # 回退：网格 + 道路矢量
            step = max(20, math.ceil((r_max - r_min) / 12 / 20) * 20)
            for g in range(math.floor(r_min / step) * step, int(r_max) + 1, step):
                pygame.draw.line(screen, (30, 36, 50), to_screen(g, r_min), to_screen(g, r_max), 1)
                pygame.draw.line(screen, (30, 36, 50), to_screen(r_min, g), to_screen(r_max, g), 1)
            for lane in lanes:
                pts = [to_screen(p["x"], p["y"]) for p in lane.get("pts", [])]
                if len(pts) >= 2:
                    pygame.draw.lines(screen, (70, 84, 110), False, pts, 2)

        # ── 规划结果层 ──
        if route:
            rp = [to_screen(p["x"], p["y"]) for p in route]
            if len(rp) >= 2:
                pygame.draw.lines(screen, (255, 170, 60), False, rp, 3)
            for ob in obstacles:
                op = to_screen(ob.get("x", 0), ob.get("y", 0))
                pygame.draw.circle(screen, (255, 80, 80), (int(op[0]), int(op[1])), 6)
        if start:
            sp = to_screen(start["x"], start["y"])
            pygame.draw.circle(screen, (60, 220, 120), (int(sp[0]), int(sp[1])), 8)
        if end:
            ep = to_screen(end["x"], end["y"])
            pygame.draw.circle(screen, (230, 80, 80), (int(ep[0]), int(ep[1])), 8)

        # ── 提示 ──
        tip = "点击选择起点"
        if start and not end:
            tip = "点击选择终点（点完自动规划并开始实验）"
        if routing:
            tip = "正在规划路线…"
        if route:
            tip = "路线已规划，即将开始自动驾驶…"
        if error:
            tip = "规划失败：" + error + "，右键重新选择"
        screen.blit(font.render(tip, True, (220, 230, 245)), (20, 16))
        if start:
            screen.blit(font.render(f"起点 ({start['x']:.1f}, {start['y']:.1f})", True, (120, 255, 180)), (20, 40))
        if end:
            screen.blit(font.render(f"终点 ({end['x']:.1f}, {end['y']:.1f})", True, (255, 160, 160)), (20, 64))

        pygame.display.flip()
        clock.tick(30)

        if route:
            pygame.time.wait(1000)   # 短暂展示路线后进入运行阶段
            pygame.quit()
            return {"start": start, "end": end, "route": route,
                    "obstacles": obstacles, "sampling": sampling}
