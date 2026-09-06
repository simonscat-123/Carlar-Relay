"""实时统计图表：pygame 实现的折线图 / 轨迹图（替代前端 ECharts）。

由 TelemetryHistory 从各实验 SSE 遥测中提取时序数据，views 层每帧绘制。
"""
from __future__ import annotations

import pygame

PANEL_BG = (18, 22, 32)
PANEL_BORDER = (46, 54, 72)
GRID = (34, 40, 56)
TEXT = (170, 182, 205)
TEXT_DIM = (118, 130, 155)

# 系列配色（与前端 ECharts 主色系对齐）
C_BLUE = (86, 156, 250)
C_ORANGE = (250, 170, 60)
C_GREEN = (80, 210, 130)
C_RED = (245, 100, 100)
C_PURPLE = (180, 130, 250)

# 动态折线配色环（语义类别数不固定时轮询取色）
_PICK_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE,
                (240, 220, 90), (80, 190, 240), (230, 150, 140), (150, 210, 120)]


class LineChart:
    """多序列折线图：自适应 y 轴，图例显示最新值。"""

    def __init__(self, title: str, series, fmt=None, dynamic=False, max_series=8):
        """series: [(名称, 颜色), ...]；fmt: 数值格式化函数
        dynamic=True 时，遥测里出现的未注册类别会自动加为一条折线
        （语义分割的类别数随 7/22 预设变化，固定系列对不上会导致图例恒为 0）。"""
        self.title = title
        self.series = list(series)
        self.data = {name: [] for name, _ in self.series}
        self.fmt = fmt or (lambda v: f"{v:.1f}")
        self.dynamic = dynamic
        self.max_series = max_series
        self._color_i = 0

    def append(self, x, values: dict):
        for name, v in values.items():
            if v is None:
                continue
            if name not in self.data:
                if not self.dynamic or len(self.series) >= self.max_series:
                    continue
                self.series.append((name, _PICK_COLORS[self._color_i % len(_PICK_COLORS)]))
                self._color_i += 1
                self.data[name] = []
            self.data[name].append((float(x), float(v)))
        # 抽稀：超 900 点减半，防止长时间运行后绘制变卡
        for name, d in self.data.items():
            if len(d) > 900:
                self.data[name] = d[::2]

    def draw(self, screen, font, rect):
        x, y, w, h = rect
        pygame.draw.rect(screen, PANEL_BG, rect)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 1)
        screen.blit(font.render(self.title, True, TEXT), (x + 8, y + 5))
        # 图例（右上角，从右往左排）
        lx = x + w - 8
        for name, color in reversed(self.series):
            d = self.data[name]
            latest = self.fmt(d[-1][1]) if d else "—"
            s = font.render(f"{name} {latest}", True, color)
            lx -= s.get_width()
            screen.blit(s, (lx, y + 5))
            lx -= 14
        gx, gy, gw, gh = x + 46, y + 26, w - 54, h - 46
        if gw < 10 or gh < 10:
            return
        pts_all = [p for d in self.data.values() for p in d]
        if not pts_all:
            s = font.render("等待数据…", True, TEXT_DIM)
            screen.blit(s, (gx + gw // 2 - s.get_width() // 2, gy + gh // 2))
            return
        xmin = min(p[0] for p in pts_all)
        xmax = max(p[0] for p in pts_all)
        vals = [p[1] for p in pts_all]
        ymin, ymax = min(vals), max(vals)
        if ymax - ymin < 1e-9:
            ymin, ymax = ymin - 1.0, ymin + 1.0
        padv = (ymax - ymin) * 0.08
        ymin, ymax = ymin - padv, ymax + padv
        # 网格 + y 轴标签
        for i in range(5):
            yy = gy + gh * i / 4
            pygame.draw.line(screen, GRID, (gx, yy), (gx + gw, yy), 1)
            val = ymax - (ymax - ymin) * i / 4
            s = font.render(self.fmt(val), True, TEXT_DIM)
            screen.blit(s, (gx - s.get_width() - 4, yy - s.get_height() // 2))
        # x 轴标签（时间）
        s = font.render(f"{xmin:.0f}s", True, TEXT_DIM)
        screen.blit(s, (gx, gy + gh + 3))
        s = font.render(f"{xmax:.0f}s", True, TEXT_DIM)
        screen.blit(s, (gx + gw - s.get_width(), gy + gh + 3))

        def tx(v):
            return gx + (v - xmin) / (xmax - xmin) * gw if xmax > xmin else gx + gw / 2

        def ty(v):
            return gy + (ymax - v) / (ymax - ymin) * gh

        for name, color in self.series:
            d = self.data[name]
            if len(d) < 2:
                continue
            pygame.draw.lines(screen, color, False, [(tx(a), ty(b)) for a, b in d], 2)


class TrajectoryPlot:
    """等比例轨迹图（GT / 融合 / GNSS 多序列）。"""

    def __init__(self, title: str, series):
        self.title = title
        self.series = list(series)
        self.data = {name: [] for name, _ in self.series}

    def append(self, name, x, y):
        if name in self.data and x is not None and y is not None:
            self.data[name].append((float(x), float(y)))
            if len(self.data[name]) > 2000:
                self.data[name] = self.data[name][::2]

    def draw(self, screen, font, rect):
        x, y, w, h = rect
        pygame.draw.rect(screen, PANEL_BG, rect)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 1)
        screen.blit(font.render(self.title, True, TEXT), (x + 8, y + 5))
        gx, gy, gw, gh = x + 10, y + 26, w - 20, h - 36
        if gw < 10 or gh < 10:
            return
        pts_all = [p for d in self.data.values() for p in d]
        if not pts_all:
            s = font.render("等待数据…", True, TEXT_DIM)
            screen.blit(s, (gx + gw // 2 - s.get_width() // 2, gy + gh // 2))
            return
        xmin = min(p[0] for p in pts_all); xmax = max(p[0] for p in pts_all)
        ymin = min(p[1] for p in pts_all); ymax = max(p[1] for p in pts_all)
        span = max(xmax - xmin, ymax - ymin, 1.0) * 1.1
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        scale = min(gw, gh) / span

        def tp(px, py):
            return (gx + gw / 2 + (px - cx) * scale,
                    gy + gh / 2 - (py - cy) * scale)

        # 起点 + 网格十字
        x0, y0 = tp(pts_all[0][0], pts_all[0][1])
        pygame.draw.circle(screen, TEXT_DIM, (int(x0), int(y0)), 4, 1)
        # 轨迹线
        for name, color in self.series:
            d = self.data[name]
            if len(d) < 2:
                continue
            pygame.draw.lines(screen, color, False, [tp(a, b) for a, b in d], 2)
        # 图例（左下角）
        lx = gx + 6
        ly = gy + gh - 20
        for name, color in self.series:
            pygame.draw.line(screen, color, (lx, ly + 8), (lx + 16, ly + 8), 3)
            s = font.render(name, True, color)
            screen.blit(s, (lx + 20, ly))
            lx += 20 + s.get_width() + 12


class TelemetryHistory:
    """按实验提取 SSE 遥测 → 图表数据；同一 SSE 消息只处理一次（按 seq 去重）。"""

    def __init__(self):
        self._last_seq = -1
        self.charts: dict = {}   # exp_id -> {"charts": [(chart, rect_key)], ...}
        self._ratio_chart = None
        # 各实验图表实例
        self.exp23_err = LineChart("定位 / 横向误差 (m)", [
            ("定位误差", C_BLUE), ("横向误差", C_ORANGE)], fmt=lambda v: f"{v:.2f}")
        self.exp23_traj = TrajectoryPlot("轨迹（本地系）", [
            ("真值", C_GREEN), ("融合", C_BLUE), ("GNSS", C_ORANGE)])
        self.exp4_obstacle = LineChart("前方最近障碍距离 (m)", [
            ("距离", C_BLUE)], fmt=lambda v: f"{v:.0f}")
        self.exp4_points = LineChart("LiDAR 每帧点数", [
            ("点数", C_PURPLE)], fmt=lambda v: f"{v:.0f}")
        self.exp5_ratio = LineChart("语义类别占比", [], dynamic=True, max_series=8,
            fmt=lambda v: f"{v * 100:.0f}%")
        self.exp5_grid = LineChart("BEV 栅格构成", [
            ("可行驶", C_GREEN), ("占据", C_RED), ("未知", C_BLUE)],
            fmt=lambda v: f"{v * 100:.0f}%")
        self.exp5_targets = LineChart("检测目标数", [
            ("目标", C_ORANGE)], fmt=lambda v: f"{v:.0f}")
        self.exp10_speed = LineChart("速度 (m/s)", [
            ("实际", C_BLUE), ("期望", C_ORANGE)], fmt=lambda v: f"{v:.1f}")
        self.exp10_err = LineChart("误差 (m)", [
            ("横向误差", C_BLUE), ("定位误差", C_ORANGE)], fmt=lambda v: f"{v:.2f}")
        self.exp10_traj = TrajectoryPlot("轨迹（世界系）", [
            ("真值", C_GREEN), ("融合", C_BLUE), ("GNSS", C_ORANGE)])

    def update(self, exp_id: int, exp: dict, seq: int):
        if not exp:
            return
        if seq is not None and seq == self._last_seq:
            return
        self._last_seq = seq
        if exp_id == 23:
            pt = exp.get("trajectory")
            if isinstance(pt, dict):
                self.exp23_err.append(pt.get("t", 0), {
                    "定位误差": pt.get("error"), "横向误差": pt.get("cte")})
                self.exp23_traj.append("真值", pt.get("gt_x"), pt.get("gt_y"))
                self.exp23_traj.append("融合", pt.get("fused_x"), pt.get("fused_y"))
                if pt.get("gnss_updated", True) and pt.get("gnss_x") is not None:
                    self.exp23_traj.append("GNSS", pt.get("gnss_x"), pt.get("gnss_y"))
        elif exp_id == 4:
            pt = exp.get("trajectory")
            if isinstance(pt, dict):
                self.exp4_obstacle.append(pt.get("t", 0), {"距离": pt.get("nearest_front_obstacle_m")})
                self.exp4_points.append(pt.get("t", 0), {"点数": pt.get("total")})
        elif exp_id == 5:
            pt = exp.get("trajectory")
            if isinstance(pt, dict):
                # 占比字段形如 "<cls>_ratio"，与图表既有系列按中文名匹配不到时丢弃
                vals = {}
                for k, v in pt.items():
                    if k.endswith("_ratio") and isinstance(v, (int, float)):
                        vals[_ratio_label(k)] = v
                self.exp5_ratio.append(pt.get("t", 0), vals)
                # BEV 栅格构成 + 检测目标数
                gs = pt.get("grid_stat")
                if isinstance(gs, dict):
                    self.exp5_grid.append(pt.get("t", 0), {
                        "可行驶": gs.get("free"), "占据": gs.get("occupied"),
                        "未知": gs.get("unknown")})
                if isinstance(pt.get("targets"), (int, float)):
                    self.exp5_targets.append(pt.get("t", 0), {"目标": pt.get("targets")})
        elif exp_id == 10:
            if exp.get("status") == "running" and exp.get("t") is not None:
                t = exp.get("t", 0)
                self.exp10_speed.append(t, {"实际": exp.get("speed"), "期望": exp.get("desired_speed")})
                self.exp10_err.append(t, {"横向误差": exp.get("cte"), "定位误差": exp.get("loc_err")})
                gt = exp.get("gt") or {}
                fu = exp.get("fused") or {}
                gn = exp.get("gnss") or {}
                self.exp10_traj.append("真值", gt.get("x"), gt.get("y"))
                self.exp10_traj.append("融合", fu.get("x"), fu.get("y"))
                self.exp10_traj.append("GNSS", gn.get("x"), gn.get("y"))


# 语义类别 key → 中文名（对齐服务端 SEMANTIC_CLASSES 与 SEMANTIC_PRESETS 的 7/22 类预设）
_RATIO_LABELS = {
    # 7 类预设
    "background": "背景", "drivable": "可行驶区域", "sidewalk": "人行道",
    "pedestrian": "行人", "vehicle": "车辆", "vehicles": "车辆",
    "traffic_sign": "交通标志", "traffic_light": "信号灯",
    # 22 类预设
    "unlabeled": "未标注", "building": "建筑", "fence": "栅栏", "pole": "杆状物",
    "vegetation": "植被", "sky": "天空", "road": "道路", "road_line": "车道线",
    "wall": "墙体", "rider": "骑行者", "car": "轿车", "truck": "卡车",
    "bus": "公交车", "motorcycle": "摩托车", "bicycle": "自行车",
    "guardrail": "隔离墩", "ground": "地面", "other": "其他",
    # 其他常见语义
    "terrain": "地表", "water": "水面", "static": "静态物",
}


def _ratio_label(key: str) -> str:
    cls = key[:-len("_ratio")]
    return _RATIO_LABELS.get(cls, cls)
