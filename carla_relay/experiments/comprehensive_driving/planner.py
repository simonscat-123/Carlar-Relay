"""实验10 规划层：行为决策（FSM 意图）+ Frenet 采样式时空联合轨迹规划。

参照 Werling/PythonRobotics 的采样规划 + Apollo 的「参考线 + 候选轨迹」：

  - 每帧重估：行为 FSM 输出意图标签（日志/可视化/教学用）；
  - 候选 = 横向{保持,左换,右换} × 纵向{巡航,停驻}，横向五次多项式剖面
    （最小急动度）+ 纵向逐步积分；
  - 硬约束逐时刻检查全量障碍（含长度、含旁道、含预测层恒速外推）：
    ① 可行驶域（越界即逆行/出路缘，逐点取当地边界）；② 红灯（CRUISE
    不得带速越线）；③ 碰撞（含障碍半长半宽）；换道候选另做地图级
    同向校验（双保险防逆行）；
  - 代价最小者胜出；全部被拒 → 本道刹停兜底（横向不够纵向补）；
  - 激进模式：本道被堵死且无同向邻道可绕时，允许借对向车道绕行
    （逐点地图级校验仍限 Driving 路面，借道期间限速）。

模块级常量为规划调参入口；逐帧决策细节（FRM/EVENT 行）写入规划调试日志。
"""
from __future__ import annotations

import math

import carla

from carla_relay.experiments.comprehensive_driving.frames import PlanOutput

# ── 规划参数（调参入口）──────────────────────────────────────────────────
T_HORIZON = 4.0        # 轨迹展开时长（s）
DT_PLAN = 0.25         # 展开步长（s）
DEC_WIN = 60.0         # 决策窗口：邻道占用/本道被占的检查范围（m）
RED_MARGIN = 0.5       # 红灯停止线前的停止余量（沿 s，停在线前 0.5m）
COLL_S = 0.5           # 纵向碰撞余量（m）
COLL_L = 0.3           # 横向碰撞余量（m）
COMFORT_A = 2.5        # 舒适减速度（STOP 剖面用，m/s²）
MAX_DECEL = 4.0        # 最大可用减速度（仲裁用，brake≈1；控制层共享）


class TrajectoryPlanner:
    """决策 + 时空联合规划器（有状态：意图/借道/FSM 事件检测）。"""

    def __init__(self, ref, predictor, log, plan_log, ego_half_w, ego_half_len):
        self._ref = ref                  # ReferenceLine
        self._predictor = predictor      # ObstaclePredictor（恒速外推）
        self._log = log                  # _exp_log（SSE 关键事件）
        self._plan_log = plan_log        # 规划调试日志（FRM/EVENT 逐帧细节）
        self._ego_half_w = ego_half_w    # 自车半宽（碰撞检查/走廊判据用）
        self._ego_half_len = ego_half_len  # 自车半长（碰撞检查含障碍长度）
        # 事件检测状态（上一帧）
        self._best_prev_key = None       # 上一帧最优解 (mode, l_t)——切换事件检测
        self._borrow_prev = False        # 上一帧是否激进借道——开始/结束事件检测
        self._avoid_prev = False         # 上一帧是否换道中——转换日志用
        self.fsm_state = "CRUISE"        # 行为状态标签：CRUISE/APPROACH_RED/FOLLOW/LANE_CHANGE

    def step(self, *, t, spd, v_long, ego_s, ego_l, obstacles, tl,
             aggressive_on, target_speed, safe_dist, wp_idx,
             prev_thr, prev_brk) -> PlanOutput:
        ref = self._ref
        plan_log = self._plan_log
        red_stop_s = tl.red_stop_s
        tl_state = tl.state
        tl_dist = tl.dist

        # ── 刹停余量（safe_distance 滑杆联动，默认12→6m）──
        STOP_MARGIN = max(3.0, safe_dist * 0.5)
        a_need = 0.0
        desired = target_speed

        # 可行驶域 + 车道宽度（无车道信息时退化为「只保持车道」的安全模式）
        bnds = ref.drivable_bounds(wp_idx) if wp_idx < len(ref.route_wp) else None
        w_lane = 3.5
        try:
            _wp_cur = ref.route_wps[wp_idx] if wp_idx < len(ref.route_wps) else None
            if _wp_cur is not None and _wp_cur.lane_width:
                w_lane = float(_wp_cur.lane_width)
        except Exception:
            pass
        l_min = l_max = None
        if bnds is not None and bnds[0] is not None:
            l_min, l_max = bnds

        # ── 邻道候选（clamp 进可行驶域：对向侧边界自动收缩，无可行邻道则该侧消失）。
        # 每侧独立判定：该侧窄/对向 → 跳过该侧但不中断另一侧（原 break 会连
        # 另一侧一起跳过）；目标偏移点再做一次地图级同向校验（双保险防逆行）
        _neighbors = []
        if l_min is not None:
            lo_b = l_min + self._ego_half_w + COLL_L
            hi_b = l_max - self._ego_half_w - COLL_L
            for l_t in (w_lane, -w_lane):
                if lo_b >= hi_b - 0.2:
                    break      # 域过窄（如仅一条车道），只保持
                l_c = max(lo_b, min(hi_b, l_t))
                if abs(l_c) < 0.5:
                    continue   # clamp 后已贴回本道，不算可行邻道
                # 非激进模式：避让必须开到隔壁车道（全量偏移）。转弯处"蹭着
                # 边线"的部分避让幅度不足，易与障碍/路缘擦碰，且本道一旦清空
                # 又立刻回摆、反复横跳——宁可跟停也不做无效的部分避让。
                if not aggressive_on and abs(l_c) < 0.7 * w_lane:
                    continue
                # 前方 12m 处该横向偏移落点必须是同向 Driving 车道（防对向）
                if not ref.lat_lane_ok(l_c, ego_s + 12.0):
                    plan_log(f"EVENT 邻道 {l_c:+.1f}m 被方向校验否决（对向/交叉车道）")
                    continue
                _neighbors.append(l_c)

        def _obs_stop(l_t):
            """走廊内最近障碍停驻点（后缘-余量，动态障碍 1s 前瞻）；inf = 走廊无障碍"""
            s_stop = float("inf")
            for o in obstacles:
                if abs(o["l"] - l_t) < self._ego_half_w + o["half_w"] + 0.25:
                    rear = o["s"] + max(0.0, o["v_s"]) * 1.0 - o["half_len"]
                    if ego_s + 0.5 < rear < ego_s + DEC_WIN or ego_s + 0.5 < o["s"] + o["half_len"] < ego_s + DEC_WIN:
                        s_stop = min(s_stop, rear - STOP_MARGIN)
            return s_stop

        def _corridor_stop(l_t):
            """走廊内最近停驻点 = min(障碍停驻点, 红灯停止线)"""
            s_obs = _obs_stop(l_t)
            if red_stop_s is not None:
                return red_stop_s if s_obs == float("inf") else min(red_stop_s, s_obs)
            return s_obs

        # ── 行为决策（FSM 意图，修问题3/5 的"逐个处理"与"不查旁道"）──
        # 本道走廊被占 → 选「走廊无障碍且在可行驶域内」的邻道换道（先左后右，
        # 超车靠左惯例）；两侧皆不可行 → 保持+跟停（安全兜底，不冒险切道）。
        # 换道途中每帧重估：邻道新出现障碍/本道清空都会即时改变意图。
        _blocked = _obs_stop(0.0) < float("inf")
        intent_l = 0.0
        if _blocked:
            for l_t in _neighbors:
                if _obs_stop(l_t) == float("inf"):
                    intent_l = l_t
                    break
        # 激进模式兜底：本道被占且无同向邻道可绕（单车道+对向道、或邻接链
        # 断裂致边界收缩）→ 借邻接车道绕行（通常是对向道）。目标偏移取
        # 邻接车道中心（相对参考线），走廊须无障碍；轨迹层逐点校验仍在
        # Driving 路面 + 全量碰撞检查 + 借道限速，绕过障碍后自动回本道
        borrow_l = None
        if aggressive_on and _blocked and intent_l == 0.0:
            _wp_b = ref.route_wps[wp_idx] if wp_idx < len(ref.route_wps) else None
            _b_offs = []
            if _wp_b is not None:
                try:
                    for nb, sign in ((_wp_b.get_left_lane(), 1.0),
                                     (_wp_b.get_right_lane(), -1.0)):
                        if nb is not None and nb.lane_type == carla.LaneType.Driving:
                            _b_offs.append(sign * (_wp_b.lane_width / 2.0
                                                    + nb.lane_width / 2.0))
                except Exception:
                    pass
            if not _b_offs:
                _b_offs = [w_lane, -w_lane]
            for l_t in _b_offs:
                if (abs(l_t) > 0.5 and ref.lat_driving(l_t, ego_s + 12.0)
                        and _obs_stop(l_t) == float("inf")):
                    intent_l = l_t
                    borrow_l = l_t
                    break
        lat_targets = [0.0] if intent_l == 0.0 else [intent_l, 0.0]

        # ── 候选生成与展开 ──
        cands = []
        _rej_bounds = _rej_red = _rej_coll = _rej_dir = 0   # 拒绝统计（日志用）
        _n_steps = int(T_HORIZON / DT_PLAN)
        T_lat = max(2.0, min(4.0, 1.2 * max(2.0, v_long)))
        if intent_l != 0.0:
            # 避让换道：横向过渡须在抵达障碍前完成（全量进入邻道后再与障碍
            # 平行）。默认 T_lat 按 1.2·v 拉长（≈21m 才换完），转弯路段横向
            # 进展又慢，到障碍处只剩 1~2m 偏移，被碰撞硬约束拒掉 → 换道/刹停
            # 横跳。这里按"障碍前缘距离 / 纵向速度 − 0.8s"压缩过渡时长，
            # 保证驶到障碍跟前时已全量进入邻道。
            _obs_front = min((o["s"] - o["half_len"] for o in obstacles
                              if o["s"] - o["half_len"] > ego_s + 0.5),
                             default=float("inf"))
            if _obs_front < float("inf"):
                _t_avail = max(1.2, (_obs_front - ego_s) / max(1.0, v_long) - 0.8)
                T_lat = min(T_lat, _t_avail)
            else:
                T_lat = min(T_lat, 3.0)
        for l_t in lat_targets:
            is_borrow = borrow_l is not None and l_t == borrow_l
            # 借道限速：绕障机动期间降速通过，缩短对向风险暴露时间
            v_cap = max(3.0, target_speed * 0.5) if is_borrow else target_speed
            s_stop = _corridor_stop(l_t)
            for mode in ("CRUISE", "STOP"):
                if mode == "STOP" and s_stop == float("inf"):
                    continue   # 无停驻点则无需 STOP 候选
                if mode == "CRUISE":
                    a_long = max(-2.0, min(1.5, (v_cap - v_long) / 2.0))
                else:
                    d = s_stop - ego_s
                    a_long = 0.0 if d <= 0.5 else max(-MAX_DECEL, -(v_long * v_long) / (2.0 * d))
                # 逐时刻展开：横向五次多项式（最小急动度）+ 纵向逐步积分。
                # CRUISE 按停驻点（红灯/障碍）生成舒适制动剖面
                # v ≤ √(2·COMFORT_A·(s_stop−s))，到停止线恰好停住。
                # 顺序要点（修期望速度横跳）：① 物理减速度下限先施加；
                # ② 剖面钳制最后施加且允许超过舒适值——若剖面放在下限之前，
                # 贴线归零会被下限顶回 v>0.3，整条 CRUISE 被红灯硬约束拒掉，
                # 与 STOP 候选逐帧轮替胜出 → desired 在最大/最小间跳变。
                samples = []
                ok = True
                dl = l_t - ego_l
                s_prev, v_prev = ego_s, v_long
                for k in range(1, _n_steps + 1):
                    tk = k * DT_PLAN
                    tau = min(1.0, tk / T_lat)
                    l_k = ego_l + dl * tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)
                    v_k = max(0.0, v_prev + a_long * DT_PLAN)
                    v_k = max(v_k, v_prev - MAX_DECEL * DT_PLAN)   # 物理极限内
                    if mode == "CRUISE":
                        if a_long > 0.0:
                            v_k = min(v_k, v_cap)
                        if s_stop < float("inf"):
                            v_k = min(v_k, math.sqrt(
                                2.0 * COMFORT_A * max(0.0, s_stop - s_prev)))
                            # 贴线归零：本步内将抵达停止线就停（判据=剩余距离
                            # 小于本步行程，而非固定 0.2m——低速步长 0.5m 会被
                            # 漏判带速越线触发整条拒绝）
                            if s_stop - s_prev <= max(0.25, v_prev * DT_PLAN):
                                v_k = 0.0
                    s_k = s_prev + 0.5 * (v_prev + v_k) * DT_PLAN
                    s_prev, v_prev = s_k, v_k
                    # 硬约束①：可行驶域（越界即逆行/出路缘，整条拒绝）。
                    # 逐点取「当地」边界——轨迹展开 30m+，前方路段可能收窄/
                    # 变两车道，只用车头处边界会放行前方的对向车道（蓝线逆行）。
                    # 借道候选例外：不受同向域限制，改为逐点地图级校验
                    # （Driving 路面即可，不限方向）——对向道/邻接链断裂处
                    # 按此放行，但出路缘仍拒绝
                    if is_borrow:
                        if not ref.lat_driving(l_k, s_k):
                            _rej_bounds += 1
                            ok = False
                            break
                    elif l_min is not None:
                        b_k = ref.bounds_at_s(s_k)
                        if b_k is None:
                            b_k = (l_min, l_max)
                        if (l_k < min(b_k[0] + self._ego_half_w + COLL_L, ego_l - 0.05)
                                or l_k > max(b_k[1] - self._ego_half_w - COLL_L, ego_l + 0.05)):
                            _rej_bounds += 1
                            ok = False
                            break
                    # 硬约束②：红灯（CRUISE 不得带速越过停止线）
                    if (mode == "CRUISE" and red_stop_s is not None
                            and s_k > red_stop_s and v_k > 0.3):
                        _rej_red += 1
                        ok = False
                        break
                    # 硬约束③：碰撞——对全量障碍（含长度、含预测外推、含旁道）
                    for o in obstacles:
                        if o["s"] < ego_s - 1.0 and o["v_s"] > v_long:
                            continue   # 后方更快的超车车辆：后车责任，不因此误刹
                        s_o = self._predictor.extrapolate(o, tk)
                        if (abs(s_k - s_o) < o["half_len"] + self._ego_half_len + COLL_S
                                and abs(l_k - o["l"]) < o["half_w"] + self._ego_half_w + COLL_L):
                            _rej_coll += 1
                            ok = False
                            break
                    if not ok:
                        break
                    samples.append((tk, s_k, l_k, v_k))
                if not ok or not samples:
                    continue
                # 换道候选：中段与末端落点必须是同向 Driving 车道。
                # 边界缓存按「邻接链+点积」推断，路口/车道斜接处可能漏判对向
                # （如左换道落在交叉来车道上）——地图级查询兜底，逆行零容忍。
                # 借道候选（激进模式）显式允许对向，跳过方向校验
                if abs(dl) > 0.5 and not is_borrow:
                    if not (ref.lat_lane_ok(l_t, samples[len(samples) // 2][1])
                            and ref.lat_lane_ok(l_t, samples[-1][1])):
                        _rej_dir += 1
                        continue
                # 代价：意图对齐（决策层意图优先，非意图轨迹仅作兜底）+
                # 偏离参考线 + 横摆平顺 + 舒适性 + 效率 + 换道机动惩罚
                l_arr = [ego_l] + [sm[2] for sm in samples]
                j_lat = sum(x * x for x in l_arr) / len(l_arr)
                j_dl = sum(((l_arr[i + 1] - l_arr[i]) / DT_PLAN) ** 2
                           for i in range(len(l_arr) - 1)) / max(1, len(l_arr) - 1)
                v_end = samples[-1][3]
                cost = (0.5 * j_lat + 0.3 * j_dl + 0.2 * a_long * a_long
                        + 0.4 * max(0.0, target_speed - v_end)
                        + (0.6 if abs(l_t) > 0.5 else 0.0)
                        + (0.0 if l_t == intent_l else 5.0))
                cands.append({"cost": cost, "l_t": l_t, "mode": mode,
                              "s_stop": s_stop, "a_long": a_long,
                              "v_cap": v_cap, "samples": samples})

        best = min(cands, key=lambda c: c["cost"]) if cands else None
        plan_traj = []
        plan_end_l = ego_l
        avoiding = False
        avoid_side = 0
        avoid_lat_target = ego_l
        if best is not None:
            # 由最优候选导出控制量与可视化状态
            if best["mode"] == "STOP":
                # 逼近速度 = 最高速一半（远端），贴近停驻点按舒适制动剖面
                # 平滑收敛到 0（desired=√(2·a·d)），停在线前 0.5m
                d = max(0.1, best["s_stop"] - ego_s)
                desired = min(target_speed * 0.5,
                              math.sqrt(2.0 * COMFORT_A * d))
                if v_long > desired + 0.3 and d > 0.5:
                    a_need = min(MAX_DECEL,
                                 (v_long * v_long - desired * desired) / (2.0 * d))
                if v_long < 0.5 and d < 1.0:
                    a_need = max(a_need, 1.5)   # 已到停驻点：保持制动防蠕行
            else:
                desired = best.get("v_cap", target_speed)
                if best["s_stop"] < float("inf"):
                    d = max(0.0, best["s_stop"] - ego_s)
                    desired = min(desired, math.sqrt(2.0 * COMFORT_A * d))
                    # 车速高于剖面期望 → 按剖面所需减速度主动制动
                    # （速度 P 控制的 0.25 油门基线自身减不了速）
                    if v_long > desired + 0.3 and d > 0.5:
                        a_need = min(MAX_DECEL,
                                     (v_long * v_long - desired * desired) / (2.0 * d))
                    if v_long < 0.5 and d < 2.0:
                        a_need = max(a_need, 1.5)   # 近停止线保持制动防蠕行
            plan_traj = [ref.world(sm[1], sm[2]) for sm in best["samples"]]
            plan_end_l = best["samples"][-1][2] if best["samples"] else ego_l
            avoid_lat_target = best["l_t"]
            avoiding = abs(best["l_t"]) > 0.5
            avoid_side = 1 if best["l_t"] > 0.5 else (-1 if best["l_t"] < -0.5 else 0)
        else:
            # 兜底：无可行轨迹（本道与旁道皆被占/过近）→ 当前走廊内刹停
            s_stop = _corridor_stop(ego_l)
            d = max(0.1, s_stop - ego_s) if s_stop < float("inf") else 0.1
            a_need = min(MAX_DECEL, v_long * v_long / (2.0 * d))
            if v_long < 0.5:
                a_need = 1.5   # 已停：保持制动防蠕行
            desired = 0.0
            plan_traj = [ref.world(ego_s + 2.0 * i, ego_l) for i in range(1, 16)]
            plan_end_l = ego_l
            avoiding = False
            avoid_side = 0
            avoid_lat_target = ego_l
            plan_log(f"EVENT 无可行候选 → 兜底刹停 (rej: 域{_rej_bounds}/"
                     f"红{_rej_red}/碰{_rej_coll}/向{_rej_dir})")

        # 逐帧决策日志（排障主数据：一帧一行，状态+决策+控制全链路）
        _bk = (best["mode"], round(best["l_t"], 1)) if best is not None else ("FALLBACK", None)
        if _bk != self._best_prev_key:
            plan_log(f"EVENT 最优解切换: {self._best_prev_key} → {_bk}")
            self._best_prev_key = _bk
        if (borrow_l is not None) != self._borrow_prev:
            if borrow_l is not None:
                plan_log(f"EVENT 激进借道: 目标偏移 {borrow_l:+.1f}m"
                         f"（本道被占且无同向邻道可绕）")
            else:
                plan_log("EVENT 激进借道结束")
            self._borrow_prev = borrow_l is not None
        _obs_str = ",".join(f"{o['s']:.0f}/{o['l']:+.1f}" for o in obstacles[:3])
        _cost_str = f"/c={best['cost']:.2f}" if best is not None else ""
        plan_log(
            f"FRM t={t:6.1f} v={spd:5.2f}(lon {v_long:5.2f}) s={ego_s:7.1f} l={ego_l:+5.2f} "
            f"tl={tl_state[:3]}/{tl_dist:5.1f}m obs={len(obstacles)}[{_obs_str}] "
            f"blk={'Y' if _blocked else 'N'} agm={'Y' if aggressive_on else 'N'} "
            f"bor={'Y' if borrow_l is not None else 'N'} "
            f"itn={intent_l:+5.2f} nb={[round(x, 1) for x in _neighbors]} "
            f"cands={len(cands)} best={_bk[0]}{_cost_str} "
            f"des={desired:5.2f} a={a_need:5.2f} "
            f"thr={prev_thr:.2f} brk={prev_brk:.2f} "
            f"rej:域{_rej_bounds}/红{_rej_red}/碰{_rej_coll}/向{_rej_dir}")

        # SSE 兼容：换道目标车道（鸟瞰图高亮）。
        # 高亮前做同向校验——邻接链查询可能拿到对向/交叉车道，高亮到逆行
        # 车道上会误导观测（规划层已有 lat_lane_ok 双保险，此处管展示）
        avoid_dest_lane = None
        if avoiding:
            try:
                _wp_cur = ref.route_wps[wp_idx] if wp_idx < len(ref.route_wps) else None
                if _wp_cur is not None:
                    nb = _wp_cur.get_left_lane() if avoid_side > 0 else _wp_cur.get_right_lane()
                    if nb is not None and nb.lane_type == carla.LaneType.Driving:
                        _cf = _wp_cur.transform.get_forward_vector()
                        _nf = nb.transform.get_forward_vector()
                        # 借道期间（激进模式）对向道也高亮，展示实际意图
                        if (_nf.x * _cf.x + _nf.y * _cf.y > 0.3
                                or borrow_l is not None):
                            avoid_dest_lane = (nb.road_id, nb.lane_id)
            except Exception:
                pass

        # 行为状态标签（教学/日志/可视化用；决策本身每帧重估无状态依赖）
        _blocked_now = any(
            abs(o["l"] - ego_l) < self._ego_half_w + o["half_w"] + 0.25
            and ego_s < o["s"] + o["half_len"] < ego_s + DEC_WIN
            for o in obstacles)
        if avoiding:
            _fsm_new = "LANE_CHANGE"
        elif red_stop_s is not None:
            _fsm_new = "APPROACH_RED"
        elif _blocked_now:
            _fsm_new = "FOLLOW"
        else:
            _fsm_new = "CRUISE"
        # 状态/换道转换日志（教学观测用；CRUISE 回归不刷屏）
        if avoiding and not self._avoid_prev:
            self._log(f"{'左' if avoid_side > 0 else '右'}侧邻道可行，换道绕行（目标偏移 {abs(avoid_lat_target):.1f}m）"
                      + ("，激进借对向道（限速通过）" if borrow_l is not None else ""))
        elif not avoiding and self._avoid_prev:
            self._log("绕行完成，回正车道")
        self._avoid_prev = avoiding
        if _fsm_new != self.fsm_state:
            if _fsm_new == "APPROACH_RED":
                self._log(f"前方 {tl_dist:.0f}m 红灯，减速停车（停止线判定）")
            elif _fsm_new == "FOLLOW":
                self._log("本车道被占且邻道不可行，跟停等待")
            self.fsm_state = _fsm_new

        # 展示用：本道走廊内最近障碍（到后缘的距离，含半长——修"量到中心"缺陷）
        front_obstacle = float("inf")
        front_obs_src = "none"
        for o in obstacles:
            if (abs(o["l"] - ego_l) < self._ego_half_w + o["half_w"] + 0.25
                    and o["s"] - o["half_len"] > ego_s):
                d_rear = o["s"] - o["half_len"] - ego_s
                if d_rear < front_obstacle:
                    front_obstacle = d_rear
                    front_obs_src = o["cls"]

        # 黄灯：软约束（红灯/障碍已由规划层硬约束处理，此处不叠加）
        YELLOW_D = 1.0
        if tl_state == "yellow" and a_need < YELLOW_D:
            a_need = YELLOW_D
            desired = min(desired, target_speed * 0.5)

        return PlanOutput(
            best=best,
            plan_traj=plan_traj,
            plan_end_l=plan_end_l,
            intent_l=intent_l,
            borrow_l=borrow_l,
            desired=desired,
            a_need=a_need,
            avoiding=avoiding,
            avoid_side=avoid_side,
            avoid_lat_target=avoid_lat_target,
            avoid_dest_lane=avoid_dest_lane,
            fsm_state=self.fsm_state,
            front_obstacle=front_obstacle,
            front_obs_src=front_obs_src,
            n_cands=len(cands),
            blocked=_blocked,
            rej_bounds=_rej_bounds,
            rej_red=_rej_red,
            rej_coll=_rej_coll,
            rej_dir=_rej_dir,
        )
