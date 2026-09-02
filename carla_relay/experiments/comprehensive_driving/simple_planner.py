"""实验10 简化规划器（方案B：单一几何避障）。

与 legacy TrajectoryPlanner（FSM + 采样式时空联合规划）相对，本规划器把
「换道 / 贴边 / 借道」三种行为统一为一个动作：给控制层一条横向偏移后的
参考轨迹 + 期望速度。设计原则：

  1. 构造性生成，非竞争选择：偏移量直接由障碍几何推出（障碍边 + 本车宽
     + 单一余量），不存在「决策说能过、验证说过不去」的双真相源矛盾；
  2. 唯一真相源 = 地图级校验：可行驶性判定一律用 world 坐标投影
     （lat_driving_fwd / lat_driving），不比较缓存域边界，免疫参考线
     坐标系在 S 弯/路口的重关联错位；
  3. 每帧完全重估、近零内部状态：无意图锁存 / 降级 / 滞回 / 承诺机制，
     剖面每帧从当前位姿重新锚定，天然连续；
  4. 纵向兜底：横向绕不开时减速停车（唯一兜底动作），不做候选竞争。

避障逻辑（每帧）：
  候选生成 = 地图级横向网格探测（唯一真相源，不经缓存域）；
  侧选 = 两级字典序：同向合法优先、幅度最小次之——除非只能逆行否则不逆行；
  无阻挡障碍 → 横向目标 0（回正；回扫走廊内仍有并行障碍时延后回正）；
  有阻挡障碍（静态/低速）→ 包络两侧试偏移 →「渐变—保持—回正」s 基剖面；
  动态障碍（|v_s|>0.5）不几何绕行 → 跟停；
  双安全网：轨迹剖面逐点校验 + 实际位姿盒-盒校验（本车按盒子建模）。

接口与 TrajectoryPlanner 完全兼容（构造参数 / step 关键字 / PlanOutput），
经实验参数 planner="simple" 启用（默认 "legacy" 保持旧行为）。
"""
from __future__ import annotations

import math

import carla

from carla_relay.experiments.comprehensive_driving.frames import PlanOutput

# ── 参数（刻意保持少量）───────────────────────────────────────────────────
DEC_WIN = 60.0        # 前方阻挡障碍检查范围（m）
AVOID_MARGIN = 1.0    # 单一横向余量（m）：障碍边 → 本车边（含执行层跟踪误差预算）
EDGE_MARGIN = 0.3     # 域边界余量（m）：本车边 → 可行驶域边界
SEP_MIN = 0.2         # 碰撞安全网的最小横向分离（m）
OBS_VMAX = 0.5        # 允许几何绕行的障碍最大速度（m/s）
COMFORT_A = 2.5       # 舒适减速度（m/s²）
MAX_DECEL = 4.0       # 最大减速度（m/s²，与控制层共享量级）
RAMP_MIN = 5.0        # 横向渐变段最小长度（m，实际按 1.5·v 拉伸）
RAMP_MIN_S = 2.0      # 距离不足时的渐变段压缩下限（m）
RETURN_X = 2.0        # 回正触发：车尾越过障碍前缘此余量后才开始回正（m，防切角）
HORIZON = 60.0        # 输出轨迹长度（m，与鸟瞰可视范围一致）
STEP_S = 2.0          # 轨迹采样步长（m）
COMMITTED_L = 0.5     # 「已离开本道」判定的横向偏移阈值（m）
PROBE_RANGE = 6.0     # 地图级横向探测半径（m）
PROBE_STEP = 0.25     # 地图级横向探测网格步长（m）
POSE_WIN_AHEAD = 6.0  # 实际位姿安全网：纵向刹停窗口下限（m）


class SimplePlanner:
    """单一几何避障规划器（每帧重估，近零状态：仅事件日志用上一帧标记）。"""

    def __init__(self, ref, predictor, log, plan_log, ego_half_w, ego_half_len):
        self._ref = ref                  # ReferenceLine
        self._predictor = predictor      # ObstaclePredictor（恒速外推，安全网用）
        self._log = log                  # _exp_log（SSE 关键事件）
        self._plan_log = plan_log        # 规划调试日志
        self._ego_half_w = ego_half_w
        self._ego_half_len = ego_half_len
        self._avoid_prev = False         # 上一帧是否绕行中（开始/结束事件）
        self._fsm_state = "CRUISE"       # 展示标签，无决策依赖

    # ── 工具 ────────────────────────────────────────────────────────────
    def _probe_drivable(self, s, aggressive_on):
        """地图级横向网格探测：返回可行驶区间 [(lo,hi),...]。

        非激进只扫同向（lat_driving_fwd）；激进扫全部 Driving 路面
        （lat_driving，含对向），对向候选由后续两级排序排后。区间边界
        按网格保守收缩一个步长（等效附加余量）。
        """
        fn = self._ref.lat_driving if aggressive_on else self._ref.lat_driving_fwd
        ivs = []
        lo = None
        n = int(PROBE_RANGE / PROBE_STEP)
        for i in range(-n, n + 1):
            l = i * PROBE_STEP
            ok = fn(l, s)
            if ok and lo is None:
                lo = l
            elif not ok and lo is not None:
                ivs.append((lo, l - PROBE_STEP))
                lo = None
        if lo is not None:
            ivs.append((lo, PROBE_RANGE))
        return ivs

    def _blockers(self, obstacles, ego_s):
        """前方决策窗内、横向上挡住本道的障碍（含并行中，不含已越过）。

        一律判参考线 l=0 走廊（与 legacy 教训一致）：绕行途中 ego_l 已偏移，
        若用 ego_l 判走廊会中途判空 → 剖面闪断 → 回摆。
        """
        out = []
        for o in obstacles:
            if ego_s + 0.5 >= o["s"] + o["half_len"]:
                continue   # 已越过（车头已过障碍后缘）
            if o["s"] - o["half_len"] > ego_s + DEC_WIN:
                continue   # 决策窗外
            if abs(o["l"]) < self._ego_half_w + o["half_w"] + 0.25:
                out.append(o)
        return out

    def _try_side(self, l_t, blockers, s_mid, aggressive_on):
        """校验一侧偏移目标：分离充分 → 地图级可行驶。

        唯一真相源是 lat_driving_fwd（世界坐标地图级查询）；激进模式放开
        方向限制（lat_driving，允许对向 Driving 路面）。
        """
        ok = all(abs(l_t - o["l"]) >= self._ego_half_w + o["half_w"] + SEP_MIN
                 for o in blockers)
        if not ok:
            return False
        if aggressive_on:
            return self._ref.lat_driving(l_t, s_mid)
        return self._ref.lat_driving_fwd(l_t, s_mid)

    # ── 主入口（接口与 TrajectoryPlanner.step 完全一致） ────────────────
    def step(self, *, t, spd, v_long, ego_s, ego_l, obstacles, tl,
             aggressive_on, target_speed, safe_dist, wp_idx,
             prev_thr, prev_brk, yaw_err_deg=None) -> PlanOutput:
        ref = self._ref
        plan_log = self._plan_log
        red_stop_s = tl.red_stop_s
        tl_state = tl.state
        tl_dist = tl.dist
        STOP_MARGIN = max(3.0, safe_dist * 0.5)

        a_need = 0.0
        desired = target_speed
        s_stop = float("inf")            # 本帧纵向停驻目标（inf=不停）
        avoid_l = None                   # 本帧横向偏移目标（None=回正）
        avoiding = False
        avoid_side = 0
        avoid_reason = ""
        ramp = max(RAMP_MIN, 1.5 * v_long)
        s_rear = s_front = ego_s         # 障碍包络（无障碍时无意义）
        too_close = False

        # ── 阻挡障碍 ──
        blockers = self._blockers(obstacles, ego_s)
        dynamic_block = any(abs(o["v_s"]) > OBS_VMAX for o in blockers)

        if blockers and not dynamic_block:
            # ── F1 几何绕行：地图级横向网格探测生成候选 ──
            # 唯一真相源 = lat_driving(_fwd) 直接扫描（±PROBE_RANGE 网格），
            # 不经缓存域（实测 20260902_170121：缓存域不含右侧同向绿道，
            # 右候选在生成阶段即被杀，只剩逆行左道可选）。障碍联合占用
            # 包络两侧按「清障几何 + 本车宽 + 单一余量」构造目标，并整体
            # clamp 进探测区间（含 EDGE_MARGIN）。
            s_rear = min(o["s"] - o["half_len"] for o in blockers)
            s_front = max(o["s"] + o["half_len"] for o in blockers)
            l_lo = min(o["l"] - o["half_w"] for o in blockers)
            l_hi = max(o["l"] + o["half_w"] for o in blockers)
            # 剖面完成期限：进入障碍包络前必须完成横向分离
            s_sep_end = s_rear - self._ego_half_len - AVOID_MARGIN
            committed = abs(ego_l) > COMMITTED_L
            avail = s_sep_end - ego_s
            if avail < ramp:
                if avail >= RAMP_MIN_S:
                    ramp = avail          # 压缩渐变段，贴边更陡但可行
                elif committed:
                    ramp = RAMP_MIN_S     # 已横移中：尽力陡移，安全网兜底
                else:
                    too_close = True      # 太近且未横移：本帧跟停，下帧重试
            s_mid = s_rear if s_rear > ego_s else ego_s + 6.0
            cands = []
            for iv_lo, iv_hi in self._probe_drivable(s_mid, aggressive_on):
                # 右侧候选：障碍右缘 + 本车宽 + 余量，且整体落入探测区间
                l_t = max(l_hi + self._ego_half_w + AVOID_MARGIN,
                          iv_lo + self._ego_half_w + EDGE_MARGIN)
                if l_t <= iv_hi - self._ego_half_w - EDGE_MARGIN:
                    cands.append((l_t, 1))
                # 左侧候选：障碍左缘 − 本车宽 − 余量，同理
                l_t = min(l_lo - self._ego_half_w - AVOID_MARGIN,
                          iv_hi - self._ego_half_w - EDGE_MARGIN)
                if l_t >= iv_lo + self._ego_half_w + EDGE_MARGIN:
                    cands.append((l_t, -1))
            # 两级字典序：① 同向合法（地图级 lat_driving_fwd 点查）→
            # ② 幅度最小 → ③ 先左。即「除非只能逆行，否则不逆行」；
            # 非激进模式探测本身只含同向路面，① 恒为 0。
            cands.sort(key=lambda c: (
                0 if ref.lat_driving_fwd(c[0], s_mid) else 1, abs(c[0]), -c[1]))
            if not too_close:
                # 注：不做「目标≈当前偏移则跳过」判断（实测 20260902_165209
                # 会误杀唯一可行侧致逐帧翻转）；已到位且校验通过 = 稳定保持
                # 偏移直到越过包络。某侧贴回被占区间由分离校验否决。
                for l_t, side in cands:
                    if self._try_side(l_t, blockers, s_mid, aggressive_on):
                        avoid_l = l_t
                        avoid_side = side
                        break
            if avoid_l is None:
                avoid_reason = "无可行驶侧或地图校验否决"
        elif blockers:
            avoid_reason = "动态障碍：不几何绕行"

        # ── F2 回扫走廊检查：欲回正（avoid_l=None）但回扫横向带内仍有
        # 并行障碍（其包络与自车 ±2m 窗口重叠）时，保持当前偏移直至越过
        # 该障碍再回正——只延后回正时机，不加深、不反向（不影响他人路权）。
        if avoid_l is None and abs(ego_l) > COMMITTED_L:
            _sweep_lo = min(ego_l, 0.0) - self._ego_half_w
            _sweep_hi = max(ego_l, 0.0) + self._ego_half_w
            _hold_front = None
            for o in obstacles:
                if (o["l"] + o["half_w"] <= _sweep_lo
                        or o["l"] - o["half_w"] >= _sweep_hi):
                    continue                      # 不在回扫带内
                if (o["s"] + o["half_len"] < ego_s - 2.0
                        or o["s"] - o["half_len"] > ego_s + self._ego_half_len + 2.0):
                    continue                      # 不在并行窗口内
                _hold_front = max(_hold_front if _hold_front is not None
                                  else float("-inf"), o["s"] + o["half_len"])
            if _hold_front is not None:
                avoid_l = ego_l                   # 原位保持（不加深）
                avoid_side = 1 if ego_l > 0 else -1
                avoiding = True
                s_front = _hold_front             # 供回正起点计算

        # ── 横向剖面（s 基，每帧从当前位姿重锚 → 帧间连续） ──
        if avoid_l is not None:
            avoiding = True
            s_ret_start = s_front + RETURN_X + self._ego_half_len
        else:
            s_ret_start = ego_s           # 回正：从当前位置渐变回 0

        def lat(s):
            if avoid_l is None:
                # 回正剖面：ramp 内 cosine 渐变回 0
                if s <= ego_s + ramp:
                    x = (s - ego_s) / ramp if ramp > 1e-6 else 1.0
                    return ego_l * (1.0 - x * x * (3.0 - 2.0 * x))
                return 0.0
            if s <= ego_s + ramp:
                # 渐变进入偏移（距离不足时 ramp 已压缩）
                x = (s - ego_s) / ramp if ramp > 1e-6 else 1.0
                return ego_l + (avoid_l - ego_l) * x * x * (3.0 - 2.0 * x)
            if s <= s_ret_start:
                return avoid_l            # 保持偏移，与障碍并行
            if s <= s_ret_start + ramp:
                x = (s - s_ret_start) / ramp if ramp > 1e-6 else 1.0
                return avoid_l * (1.0 - x * x * (3.0 - 2.0 * x))
            return 0.0

        # ── 纵向 ──
        if blockers and (avoid_l is None or too_close):
            # 横向绕不开：跟停（唯一兜底，平滑减速不整条丢弃）
            _s_rear = min(o["s"] + max(0.0, o["v_s"]) * 1.0 - o["half_len"]
                          for o in blockers)
            s_stop = _s_rear - STOP_MARGIN
            desired = 0.0
        elif avoiding:
            desired = max(3.0, target_speed * 0.7)   # 绕行期间稍降速
        if red_stop_s is not None:
            s_stop = red_stop_s if s_stop == float("inf") else min(s_stop, red_stop_s)

        # 速度/减速度导出（制动剖面 + 防蠕行，与 legacy 同型）
        if s_stop < float("inf"):
            d = max(0.1, s_stop - ego_s)
            if desired > 0:
                desired = min(desired, math.sqrt(2.0 * COMFORT_A * d))
            if v_long > desired + 0.3 and d > 0.5:
                a_need = min(MAX_DECEL,
                             (v_long * v_long - desired * desired) / (2.0 * d))
            if v_long < 0.5 and d < 2.0:
                a_need = max(a_need, 1.5)

        # ── 轨迹展开（s 网格采样，控制层 Pure Pursuit 用几何点列） ──
        plan_traj = []
        samples = []
        n_pts = int(HORIZON / STEP_S)
        for i in range(1, n_pts + 1):
            s_k = ego_s + i * STEP_S
            l_k = lat(s_k)
            samples.append((s_k, l_k))
            wx, wy = ref.world(s_k, l_k)
            plan_traj.append((wx, wy))

        # ── 逐点安全网：包络内且横向未分离 → 收紧本帧期望速度 ──
        # （平滑逼近而非刹停：给下一帧剖面重锚留收敛机会；横向正在分离
        #   的点视为逃离中放行——渐变段前几点必然仍在窗口内）
        _safe_v = desired
        _bad_s = None
        for s_k, l_k in samples:
            _hit = False
            for o in obstacles:
                if o["s"] < ego_s - 1.0 and o["v_s"] > v_long:
                    continue   # 后方更快超车车辆：后车责任
                t_k = (s_k - ego_s) / max(0.5, v_long)
                s_o = self._predictor.extrapolate(o, t_k)
                if abs(s_k - s_o) >= o["half_len"] + self._ego_half_len:
                    continue   # 纵向不在包络内
                sep = abs(l_k - o["l"])
                if sep >= self._ego_half_w + o["half_w"] + SEP_MIN:
                    continue   # 已横向分离
                if sep > abs(ego_l - o["l"]) + 1e-6:
                    continue   # 横向正在分离（渐变段）：放行
                _hit = True
                break
            if _hit:
                # 未分离且未改善：本帧须能在该点前减速停下
                _safe_v = min(_safe_v, math.sqrt(
                    2.0 * COMFORT_A * max(0.0, s_k - ego_s - 2.0)))
                _bad_s = s_k
                break
        if _safe_v < desired:
            desired = _safe_v
            if v_long > desired + 0.3:
                a_need = min(MAX_DECEL, max(a_need, 1.0))
            plan_log(f"SAFE t={t:.1f} 轨迹点(s={_bad_s:.1f})未分离 → "
                     f"收紧期望 des={desired:.2f}（下帧剖面重锚）")

        # ── F3 实际位姿安全网：规划剖面合格 ≠ 实际没撞（本车是 ~4.9×2.0m
        # 的盒子，不是点/线）。用当前真实位姿对障碍做盒-盒判定：横向净距
        # 不足且纵向刹停距离不够 → 按真实净距收紧本帧期望速度。
        # 实测 20260902_170121：t=15.5 规划剖面仍输出 des=5.60 而实际净距
        # 已 −0.17m，本网激活前全程沉默（只查剖面不查实姿）。──
        for o in obstacles:
            if o["s"] < ego_s - 1.0 and o["v_s"] > v_long:
                continue                          # 后方更快超车车辆
            if o["s"] + o["half_len"] < ego_s + 0.5:
                continue                          # 已越过（避免回切后误刹）
            dl = abs(o["l"] - ego_l) - (self._ego_half_w + o["half_w"])
            if dl >= SEP_MIN:
                continue                          # 横向已实质分离
            ds = (o["s"] - o["half_len"]) - (ego_s + self._ego_half_len)
            if ds >= max(POSE_WIN_AHEAD, v_long * 1.5):
                continue                          # 纵向尚远，剖面分离来得及
            _pose_v = math.sqrt(2.0 * MAX_DECEL * max(0.0, ds - 0.5))
            if _pose_v < desired:
                desired = _pose_v
                if v_long > desired + 0.3:
                    a_need = max(a_need, 1.5)
                plan_log(f"POSE t={t:.1f} 实姿净距不足(纵{ds:+.1f} 横{dl:+.1f} "
                         f"vs {o['cls']}@s{o['s']:.1f}) → des={desired:.2f}")

        # ── 展示/日志状态 ──
        avoid_lat_target = avoid_l if avoid_l is not None else ego_l
        plan_end_l = samples[-1][1] if samples else ego_l

        # 鸟瞰目标车道高亮（展示用，非决策）
        avoid_dest_lane = None
        if avoiding:
            try:
                wp_cur = ref.route_wps[wp_idx] if wp_idx < len(ref.route_wps) else None
                if wp_cur is not None:
                    nb = (wp_cur.get_left_lane() if avoid_side > 0
                          else wp_cur.get_right_lane())
                    if nb is not None and nb.lane_type == carla.LaneType.Driving:
                        cf = wp_cur.transform.get_forward_vector()
                        nf = nb.transform.get_forward_vector()
                        if (nf.x * cf.x + nf.y * cf.y > 0.3 or aggressive_on):
                            avoid_dest_lane = (nb.road_id, nb.lane_id)
            except Exception:
                pass

        # FSM 标签（纯展示，无状态依赖）
        if avoiding:
            _fsm_new = "LANE_CHANGE"
        elif red_stop_s is not None:
            _fsm_new = "APPROACH_RED"
        elif blockers:
            _fsm_new = "FOLLOW"
        else:
            _fsm_new = "CRUISE"
        # 绕行开始/结束事件（SSE 关键事件 + 调试日志）
        if avoiding and not self._avoid_prev:
            self._log(f"绕行障碍（目标偏移 {avoid_l:+.1f}m，降速通过）")
            plan_log(f"EVENT 绕行开始: l_t={avoid_l:+.2f} blockers={len(blockers)} "
                     f"包络=[s{s_rear:.1f}~s{s_front:.1f}] ramp={ramp:.1f}m")
        elif not avoiding and self._avoid_prev:
            self._log("绕行完成，回正车道")
            plan_log("EVENT 绕行结束")
        elif blockers and avoid_l is None and avoid_reason:
            plan_log(f"FRM t={t:.1f} 跟停原因: {avoid_reason}")
        self._avoid_prev = avoiding
        if _fsm_new != self._fsm_state:
            if _fsm_new == "APPROACH_RED":
                self._log(f"前方 {tl_dist:.0f}m 红灯，减速停车")
            elif _fsm_new == "FOLLOW":
                self._log("本车道被占且绕行不可行，跟停等待")
            self._fsm_state = _fsm_new

        # 前方最近障碍（展示用，到后缘距离）
        front_obstacle = float("inf")
        front_obs_src = "none"
        for o in obstacles:
            if (abs(o["l"] - ego_l) < self._ego_half_w + o["half_w"] + 0.25
                    and o["s"] - o["half_len"] > ego_s):
                d_rear = o["s"] - o["half_len"] - ego_s
                if d_rear < front_obstacle:
                    front_obstacle = d_rear
                    front_obs_src = o["cls"]

        # 黄灯软约束（红灯/障碍已由纵向逻辑处理）
        if tl_state == "yellow" and a_need < 1.0:
            a_need = 1.0
            desired = min(desired, target_speed * 0.5)

        # 逐帧一行调试日志（与 legacy FRM 行同位，便于 A/B 对比）
        _obs_str = ",".join(f"{o['s']:.0f}/{o['l']:+.1f}" for o in obstacles[:3])
        plan_log(
            f"FRM t={t:6.1f} v={spd:5.2f}(lon {v_long:5.2f}) s={ego_s:7.1f} l={ego_l:+5.2f} "
            f"wp={wp_idx} tl={tl_state[:3]}/{tl_dist:5.1f}m obs={len(obstacles)}[{_obs_str}] "
            f"blk={len(blockers)} agm={'Y' if aggressive_on else 'N'} "
            f"l_t={avoid_l if avoid_l is not None else 0.0:+5.2f} "
            f"des={desired:5.2f} a={a_need:5.2f} thr={prev_thr:.2f} brk={prev_brk:.2f} "
            f"fsm={self._fsm_state})")

        return PlanOutput(
            best=None,
            plan_traj=plan_traj,
            plan_end_l=plan_end_l,
            intent_l=(avoid_l if avoid_l is not None else 0.0),
            borrow_l=None,
            desired=desired,
            a_need=a_need,
            avoiding=avoiding,
            avoid_side=avoid_side,
            avoid_lat_target=avoid_lat_target,
            avoid_dest_lane=avoid_dest_lane,
            nudge=False,
            nudge_gap=0.0,
            fsm_state=self._fsm_state,
            front_obstacle=front_obstacle,
            front_obs_src=front_obs_src,
            n_cands=1,
            blocked=bool(blockers),
            rej_bounds=0,
            rej_red=0,
            rej_coll=0,
            rej_dir=0,
        )
