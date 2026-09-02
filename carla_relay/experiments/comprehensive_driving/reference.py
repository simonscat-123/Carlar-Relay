"""实验10 规划层前置基础设施：参考线 + Frenet 坐标系 + 可行驶域 + 停止线绑定。

对标工业界 Reference Line Provider（Apollo）：全局路线（GRP 沿车道中心采样）
即参考线；本模块提供：

  - frenet()/world()：世界坐标 ↔ Frenet (s, l) 互转（l 左正）；
  - drivable_bounds()/bounds_at_s()：从路线当前车道向两侧扩展「同向 Driving」
    车道得到横向边界——对向车道/路缘即边界，逆行轨迹从此无法通过硬约束；
  - lat_lane_ok()/lat_driving()：偏移点的地图级车道校验（换道方向校验 /
    激进借道的 Driving 路面校验）；
  - bind_traffic_lights()：一次性把「管着本路线」的信号灯停止线绑定到
    参考线弧长 s_stop（地图先验，属参考线而非感知）。

本模块为真实模块（可独立 import），仅依赖 carla 与标准库；
路线数据经构造函数注入，不读全局命名空间。
"""
from __future__ import annotations

import math

import carla


class ReferenceLine:
    """全局参考线：弧长表 + Frenet 变换 + 可行驶域查询。"""

    def __init__(self, carla_map, route_wp, route_wps, route_lane_ids):
        self.carla_map = carla_map
        self.route_wp = route_wp          # [carla.Location] 参考线点列
        self.route_wps = route_wps        # [carla.Waypoint|None] 对应 waypoint 对象
        self.route_lane_ids = route_lane_ids  # [(road_id, lane_id)|None] 各点所在车道
        # s_tab: 与 route_wp 平行的累计弧长；参考线本身即车道中心线（GRP 沿车道中心采样）
        self.s_tab = [0.0]
        for _j in range(1, len(route_wp)):
            self.s_tab.append(self.s_tab[-1] + route_wp[_j].distance(route_wp[_j - 1]))
        self._bounds_cache = {}
        # ── frenet 关联稳定化（修路口拐点跳变）──
        # 问题：最近点搜索在急弯/路口处，路线「回头」时（u 形弯、发卡弯，
        # 或转弯前后两段在空间上贴近），自车可能距「转弯后路段」比距「当前
        # 段」更近 → 关联一帧跳到未来路段 → s 突进 / l 突变 / 可行驶域翻转，
        # 进行中的绕行/换道剖面被直接杀死（三次实测均栽在此）。
        # 对策（track=True，仅供主循环自车位姿调用）：
        #   1) 窗口 = [上次关联点 - 15点, +前向 FRENET_FWD m]，只允许在当前
        #      段附近搜索，物理上不可能跳到远处路段；
        #   2) 候选须比当前关联点近 FRENET_HYST（滞回），定位噪声不触发切换；
        #   3) 窗口内允许双向移动（后向 15 点 ≈ 一帧内物理不可能走完的
        #      距离，倒车/校正仍可用）。
        # track=False（默认）：无状态投影，行为与旧实现完全一致——停止线
        # 绑定 / 障碍物扫描等任意位置调用不得污染自车跟踪状态（否则首帧
        # 关联会被停止线位置锚死，自车坐标系整体错位）。
        self._frenet_j = None          # 上次自车关联的最近路点索引

    # ── 坐标变换 ─────────────────────────────────────────────────────────
    # 前向搜索距离：局部窗口上界。须覆盖两次调用间最大位移（车速上限 × 帧距）
    # 留余量即可——过大窗口会重新引入拐点跳变风险
    FRENET_FWD_M = 30.0
    # 切换滞回（m²·距离平方）：候选须比当前关联点近此余量才允许前进切换，
    # 定位噪声（±0.25m）不触发关联点抖动
    FRENET_HYST_D2 = 0.5 ** 2

    def frenet(self, px, py, j0=0, j1=None, track=False):
        """世界坐标 → Frenet (s, l, 切向tx, 切向ty)。s = 弧长 + 切向投影，
        l = 相对切线的横向偏移（左正，与控制层约定一致）。

        track=False（默认）：无状态窗口投影——[j0, j1) 内全局最近点，与
        旧实现完全一致。停止线绑定 / 障碍物扫描等任意位置调用必须用此模式
        （不更新内部关联状态，防止污染自车跟踪锚点）。
        track=True：自车位姿专用——稳定关联（局部窗口 + 滞回，见 __init__
        注释），修路口拐点处关联跳到未来路段导致 s/l 突变、域翻转、绕行
        剖面被杀的问题。仅 run.py 主循环自车位姿调用。"""
        route_wp = self.route_wp
        if j1 is None:
            j1 = len(route_wp)
        if track:
            # ── 稳定关联模式（自车专用）──
            if self._frenet_j is None:
                best_j, best_d2 = j0, float("inf")
                for j in range(j0, j1):
                    d2 = (px - route_wp[j].x) ** 2 + (py - route_wp[j].y) ** 2
                    if d2 < best_d2:
                        best_d2, best_j = d2, j
                self._frenet_j = best_j
            else:
                # 窗口 = [上次关联 - 15点, +前向 FRENET_FWD_M 弧长]；
                # 后向 15 点覆盖控制层 wp_idx 校正滞后与倒车场景
                _lo = max(0, self._frenet_j - 15)
                _s_hi = self.s_tab[min(len(route_wp) - 1, self._frenet_j)] \
                    + self.FRENET_FWD_M
                _hi = _lo
                while (_hi < len(route_wp) - 1
                       and self.s_tab[_hi + 1] <= _s_hi):
                    _hi += 1
                best_j = self._frenet_j
                best_d2 = ((px - route_wp[best_j].x) ** 2
                           + (py - route_wp[best_j].y) ** 2)
                for j in range(max(_lo, j0), min(_hi + 1, j1)):
                    d2 = (px - route_wp[j].x) ** 2 + (py - route_wp[j].y) ** 2
                    if d2 < best_d2 - self.FRENET_HYST_D2:
                        best_d2, best_j = d2, j
                self._frenet_j = best_j
        else:
            # ── 无状态投影模式（旧实现行为）──
            best_j, best_d2 = j0, float("inf")
            for j in range(j0, j1):
                d2 = (px - route_wp[j].x) ** 2 + (py - route_wp[j].y) ** 2
                if d2 < best_d2:
                    best_d2, best_j = d2, j
        a = route_wp[max(0, best_j - 1)]
        b = route_wp[min(len(route_wp) - 1, best_j + 1)]
        tx, ty = b.x - a.x, b.y - a.y
        tl = math.hypot(tx, ty)
        if tl < 1e-6:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = tx / tl, ty / tl
        rx, ry = px - route_wp[best_j].x, py - route_wp[best_j].y
        return (self.s_tab[best_j] + tx * rx + ty * ry,
                tx * ry - ty * rx, tx, ty)

    def world(self, s, l):
        """Frenet (s, l) → 世界坐标 (x, y)。s 超出路线范围时钳到端点。"""
        route_wp, s_tab = self.route_wp, self.s_tab
        s = max(0.0, min(s, s_tab[-1]))
        lo, hi = 0, len(s_tab) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s_tab[mid] <= s:
                lo = mid
            else:
                hi = mid
        nxt = min(lo + 1, len(route_wp) - 1)
        a, b = route_wp[lo], route_wp[nxt]
        seg = s_tab[nxt] - s_tab[lo]
        u = 0.0 if seg < 1e-6 else max(0.0, min(1.0, (s - s_tab[lo]) / seg))
        px, py = a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u
        tx, ty = b.x - a.x, b.y - a.y
        tl = math.hypot(tx, ty)
        if tl < 1e-6:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = tx / tl, ty / tl
        return px - ty * l, py + tx * l   # 左法向 (-ty, tx) × l

    # ── 可行驶域 ─────────────────────────────────────────────────────────
    def drivable_bounds(self, j):
        """route_wp[j] 所在车道的可行驶横向边界（相对该处参考线）。失败返回 None。"""
        key = self.route_lane_ids[j] if j < len(self.route_lane_ids) else None
        if key is None:
            return None
        if key in self._bounds_cache:
            return self._bounds_cache[key]
        wp = self.route_wps[j]
        l_min = l_max = None
        try:
            if wp is not None:
                w0 = wp.lane_width
                edge_l, edge_r = -w0 / 2.0, w0 / 2.0
                cur = wp
                # 向左扩展：仅接受 Driving 且前向同向的车道
                for _ in range(3):
                    nb = cur.get_left_lane()
                    if nb is None or nb.lane_type != carla.LaneType.Driving:
                        break
                    nf = nb.transform.get_forward_vector()
                    cf = cur.transform.get_forward_vector()
                    if nf.x * cf.x + nf.y * cf.y <= 0.0:
                        break   # 对向车道，禁入
                    edge_l -= nb.lane_width
                    cur = nb
                cur = wp
                for _ in range(3):
                    nb = cur.get_right_lane()
                    if nb is None or nb.lane_type != carla.LaneType.Driving:
                        break
                    nf = nb.transform.get_forward_vector()
                    cf = cur.transform.get_forward_vector()
                    if nf.x * cf.x + nf.y * cf.y <= 0.0:
                        break
                    edge_r += nb.lane_width
                    cur = nb
                l_min, l_max = edge_l, edge_r
        except Exception:
            pass
        self._bounds_cache[key] = (l_min, l_max)
        return l_min, l_max

    def bounds_at_s(self, s):
        """弧长 s 处的可行驶域边界（换道轨迹跨多车道段时逐点取当地边界，
        而非只用车头处的边界——前方车道收窄/对向开始处才不会误入）。"""
        s_tab = self.s_tab
        lo, hi = 0, len(s_tab) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s_tab[mid] <= s:
                lo = mid
            else:
                hi = mid
        b = self.drivable_bounds(lo)
        if b is not None and b[0] is not None:
            return b
        return None

    def lat_lane_ok(self, l_t, s_probe):
        """横向偏移 l_t 在弧长 s_probe 处是否落在「同向 Driving 车道」。
        直接取该偏移点的地图车道做前向点积校验——不依赖边界缓存的推断，
        对向车道（含斜向/路口交叉车道，点积≤0.3）一律判不可行。"""
        try:
            px, py = self.world(s_probe, l_t)
            wp2 = self.carla_map.get_waypoint(
                carla.Location(x=px, y=py, z=0.0),
                project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp2 is None:
                return False
            s_tab = self.s_tab
            lo, hi = 0, len(s_tab) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s_tab[mid] <= s_probe:
                    lo = mid
                else:
                    hi = mid
            a = self.route_wp[max(0, lo - 1)]
            b = self.route_wp[min(len(self.route_wp) - 1, lo + 1)]
            tx, ty = b.x - a.x, b.y - a.y
            f = wp2.transform.get_forward_vector()
            return (f.x * tx + f.y * ty) > 0.3
        except Exception:
            return False

    def lat_driving(self, l_t, s_probe):
        """该偏移点是否落在 Driving 车道上（不限方向——激进借道判定用）。
        project_to_road 会投影到最近车道，须校验横向距离，
        防止把远处/邻路的车道投影过来误判为可走。"""
        try:
            px, py = self.world(s_probe, l_t)
            wp2 = self.carla_map.get_waypoint(
                carla.Location(x=px, y=py, z=0.0),
                project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp2 is None:
                return False
            dx = wp2.transform.location.x - px
            dy = wp2.transform.location.y - py
            return math.hypot(dx, dy) < 1.5
        except Exception:
            return False

    def lat_driving_fwd(self, l_t, s_probe):
        """该偏移点是否落在「同向 Driving 车道」上（地图级 + 投影距离校验）。

        供轨迹逐点域校验的过渡段兜底：S 式换道过渡处参考线从旧车道中心
        切到新车道中心，drivable_bounds 缓存的域边界相对「当地车道中心」
        计量，而轨迹横向偏移 l 相对「连续参考曲线」——两坐标系在过渡段
        错位可达数米，直接比较会把本可通行的剖面误判越界。此方法把
        (s, l) 经连续参考曲线转回世界坐标后直接问地图，不受坐标系错位
        影响；投影距离过远（草坪/远处邻路）与对向车道（前向点积≤0.3）
        均判不可行。"""
        try:
            px, py = self.world(s_probe, l_t)
            wp2 = self.carla_map.get_waypoint(
                carla.Location(x=px, y=py, z=0.0),
                project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp2 is None:
                return False
            dx = wp2.transform.location.x - px
            dy = wp2.transform.location.y - py
            if math.hypot(dx, dy) >= 1.5:
                return False
            s_tab = self.s_tab
            lo, hi = 0, len(s_tab) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s_tab[mid] <= s_probe:
                    lo = mid
                else:
                    hi = mid
            a = self.route_wp[max(0, lo - 1)]
            b = self.route_wp[min(len(self.route_wp) - 1, lo + 1)]
            tx, ty = b.x - a.x, b.y - a.y
            f = wp2.transform.get_forward_vector()
            return (f.x * tx + f.y * ty) > 0.3
        except Exception:
            return False

    # ── 停止线绑定（地图先验，一次性）────────────────────────────────────
    def bind_traffic_lights(self, world, log):
        """停止线 → 参考线弧长 s_stop。只绑定「停止线所在车道属于本路线」的灯
        ——管着本路线的灯才有效；运行时用 s 差判定：车越过停止线后该灯自动
        退出考虑（committed 语义），路口内/出口不再被交叉方向的红灯误刹。"""
        tl_bindings = []
        try:
            _route_lane_set = set(l for l in self.route_lane_ids if l is not None)
            for _tl in world.get_actors().filter("traffic.traffic_light*"):
                try:
                    for _swp in _tl.get_stop_waypoints():
                        if (_swp.road_id, _swp.lane_id) in _route_lane_set:
                            _s_stop = self.frenet(_swp.transform.location.x,
                                                  _swp.transform.location.y)[0]
                            tl_bindings.append({"tl": _tl, "s_stop": _s_stop})
                            break
                except Exception:
                    pass
            if tl_bindings:
                tl_bindings.sort(key=lambda b: b["s_stop"])
                log(f"信号灯绑定: {len(tl_bindings)} 处停止线已关联到路线")
        except Exception as exc:
            log(f"信号灯绑定失败: {exc}")
        return tl_bindings
