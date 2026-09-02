"""实验10 规划层：行为决策（FSM 意图）+ Frenet 采样式时空联合轨迹规划。

参照 Werling/PythonRobotics 的采样规划 + Apollo 的「参考线 + 候选轨迹」：

  - 每帧重估：行为 FSM 输出意图标签（日志/可视化/教学用）；
  - 候选 = 横向{保持,左换,右换} × 纵向{巡航,停驻}，横向五次多项式剖面
    （最小急动度）+ 纵向逐步积分；
  - 贴边绕行（nudge，Autoware avoidance 式 shift 点）：两车道均被占且
    障碍静止/低速时，在横向占用区间中找能容纳本车+余量的空隙，用
    「贴边渐变—保持—回道渐变」的 s 型横向剖面穿越；仅几何触发，安全
    性仍由逐点碰撞硬约束兜底；
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
from carla_relay.experiments.comprehensive_driving.params import read

# ── 规划参数（调参入口）──────────────────────────────────────────────────
# 参数来源登记（key, 默认, 类型, 来源）：来源 JSON=构造注入 params
# （/start body→_run_exp10(args)）；CONST=本常量兜底默认。
P_DEC_WIN = ("dec_win", 60.0, float, "JSON")  # 决策窗口：邻道占用/本道被占的检查范围（m）
T_HORIZON = 4.0        # 轨迹展开时长（s）
DT_PLAN = 0.25         # 展开步长（s）
RED_MARGIN = 0.5       # 红灯停止线前的停止余量（沿 s，停在线前 0.5m）
COLL_S = 0.5           # 纵向碰撞余量（m）
COLL_L = 0.3           # 横向碰撞余量（m）
COMFORT_A = 2.5        # 舒适减速度（STOP 剖面用，m/s²）
MAX_DECEL = 4.0        # 最大可用减速度（仲裁用，brake≈1；控制层共享）

# ── 贴边绕行（nudge）参数（调参入口）──────────────────────────────────────
NUDGE_MARGIN_OBS = 1     # 障碍四周余量（m）：空隙从障碍实际宽度外扩此值起算
NUDGE_MARGIN_EGO = 1    # 本车两侧余量（m）：空隙须容纳本车宽 + 2×此值
NUDGE_MARGIN_EDGE = 0.6    # 路缘余量（m）：本车距可行驶域边界的最小距离
NUDGE_RETURN_X = 1.0       # 回道触发距离（m）：车尾超过障碍头部此值后才渐变回道
NUDGE_OBS_VMAX = 0.5       # 允许贴边绕行的障碍最大速度（m/s，超过则走跟停/换道）
NUDGE_V_CAP_FRAC = 0.5     # 绕行期间限速比例（× target_speed，缩短侧向暴露）
NUDGE_RAMP_MIN = 5.0       # 贴边/回道渐变段最小长度（m，实际按 1.5·v 拉伸）
NUDGE_HYST = 0.3           # 滞回（m）：绕行中空隙门槛放宽此值，防临界抖动
NUDGE_RAMP_MIN_S = 2.0     # 贴边渐变段最短长度（m）：距离不足时压缩至此，再短放弃

# ── 意图失效降级（修死锁）参数 ─────────────────────────────────────────────
# 场景：邻道走廊「看似无障碍」（决策层判据），但换道候选连续被碰撞硬约束
# 拒绝（横向衰减/临界间隙）→ 车逼近障碍后 intent 候选全灭 → 兜底刹停死锁。
# 对策：intent 横向候选连续 INTENT_FAIL_N 帧无幸存 → 判定该意图不可达，
# 降级归零以放行 nudge/借道分支；锁存 INTENT_RETRY_S 秒后自动重试一次
# （防瞬态误判导致永久放弃换道），再被拒则重新锁存。
INTENT_FAIL_N = 4          # intent 候选连续无幸存的帧数阈值
INTENT_RETRY_S = 5.0       # 意图失效锁存时长（s），到期重试


class TrajectoryPlanner:
    """决策 + 时空联合规划器（有状态：意图/借道/FSM 事件检测）。"""

    def __init__(self, ref, predictor, log, plan_log, ego_half_w, ego_half_len,
                 params=None):
        self._ref = ref                  # ReferenceLine
        self._predictor = predictor      # ObstaclePredictor（恒速外推）
        self._log = log                  # _exp_log（SSE 关键事件）
        self._plan_log = plan_log        # 规划调试日志（FRM/EVENT 逐帧细节）
        self._ego_half_w = ego_half_w    # 自车半宽（碰撞检查/走廊判据用）
        self._ego_half_len = ego_half_len  # 自车半长（碰撞检查含障碍长度）
        self.dec_win = read(params, P_DEC_WIN)  # 决策窗（JSON: dec_win 可覆盖）
        # 事件检测状态（上一帧）
        self._best_prev_key = None       # 上一帧最优解 (mode, l_t)——切换事件检测
        self._borrow_prev = False        # 上一帧是否激进借道——开始/结束事件检测
        self._avoid_prev = False       # 上一帧是否换道中——转换日志用
        self._nudge_active = False     # 上一帧是否贴边绕行中——空隙门槛滞回用
        self._nudge_prev = False      # 上一帧最优解是否 nudge——开始/结束事件检测用
        # nudge 承诺锁存：(t, 最近一次 _find_nudge 成功结果)。绕行中途空隙
        # 判定瞬时空转时沿用，防候选闪断 → 兜底刹停 → 死锁振荡（见 step()）
        self._nudge_last = None
        # 意图失效降级状态（修死锁，见 INTENT_FAIL_N 注释）
        self._intent_fail_cnt = 0     # intent 候选连续无幸存帧数
        self._intent_fail_key = None  # 计数对应的 intent 值（换目标即重置）
        self._intent_latch_t = None   # 降级锁存起始时刻（None=未锁存）
        # 参考线跳变检测（上一帧位姿）：帧间 Δs 远超车速×Δt 或 Δl 突变，
        # 即参考线在路口拐点重关联到新路段（s/l/域全部不连续）——两次实测
        # 均在机动中途杀死绕行剖面，是当前最高优先级异常源
        self._prev_pose = None        # (t, ego_s, ego_l, v_long)
        # 停止线-障碍重叠事件检测（上一帧是否重叠）
        self._tl_obs_prev = False
        # nudge 空隙诊断（_find_nudge 每帧填充，CAND 表头用）：全部自由
        # 区间与所需宽度——「为什么选左不选右」「为什么没绕」直接可答
        self._nudge_diag = None
        self.fsm_state = "CRUISE"        # 行为状态标签：CRUISE/APPROACH_RED/FOLLOW/LANE_CHANGE/NUDGE

    @staticmethod
    def _free_intervals(occ, lo, hi):
        """横向占用区间列表 → [lo,hi] 内的自由区间迭代器（区间求差，保守合并）。"""
        ivs = sorted((max(lo, a), min(hi, b)) for a, b in occ
                     if max(lo, a) < min(hi, b))
        cur = lo
        for a, b in ivs:
            if a > cur:
                yield cur, a
            cur = max(cur, b)
        if cur < hi:
            yield cur, hi

    def _find_nudge(self, obstacles, ego_s, ego_l, v_long, l_min, l_max):
        """贴边绕行规划：两车道均被占时在横向空隙中找穿越路径。

        返回 {"l": 空隙中心, "lat": s→l 横向剖面函数, "gap": 空隙宽, "n": 阻挡障碍数}
        或 None（无可用空隙/障碍为动态/距离不足）。仅做几何触发，可行性最终
        由候选展开时的逐点硬约束裁决。
        """
        if l_min is None:
            return None
        # 阻挡障碍：本道走廊内、前方决策窗内。「本道」一律取参考线 l=0（与
        # step() 的 _blocked 判定一致）——绕行途中 ego_l 已偏移进空隙，若用
        # ego_l 判走廊会中途判空 → nudge 自取消 → 兜底急刹
        blockers = [o for o in obstacles
                    if abs(o["l"]) < self._ego_half_w + o["half_w"] + 0.25
                    and ego_s < o["s"] + o["half_len"] < ego_s + self.dec_win]
        if not blockers:
            return None
        # 动态障碍不贴边：空隙随时间关闭，恒速外推的空隙宽度不可靠
        if any(abs(o["v_s"]) > NUDGE_OBS_VMAX for o in blockers):
            return None
        # 横向占用区间 = 决策窗内全部障碍（含邻道，实际宽度 + 四周余量）。
        # 不同 s 处的障碍保守合并到同一横向截面——宁可漏掉错列绕行也不误入
        occ = [(o["l"] - o["half_w"] - NUDGE_MARGIN_OBS,
                o["l"] + o["half_w"] + NUDGE_MARGIN_OBS)
               for o in obstacles
               if o["s"] - o["half_len"] < ego_s + self.dec_win
               and o["s"] + o["half_len"] > ego_s - 2.0]
        # 本车可用横向边界：取「最近阻挡障碍横截面」处的当地可行驶域（与
        # 逐点硬约束同源），而非自车/前视点处的域——S 弯过渡/前视点跨段时
        # 两者错位，会探测出校验侧过不去的空隙（实测：v=0 时候选靠 ego_l
        # 豁免幸存、v>0 时轨迹前进即被拒 → 刹停-爬行-再拒循环振荡）。取不
        # 到当地域时退回当前域
        _b_blk = self._ref.bounds_at_s(min(o["s"] for o in blockers))
        if _b_blk is not None and _b_blk[0] is not None:
            l_min, l_max = _b_blk
        lo = l_min + self._ego_half_w + NUDGE_MARGIN_EDGE
        hi = l_max - self._ego_half_w - NUDGE_MARGIN_EDGE
        if lo >= hi:
            return None
        # 空隙须容纳本车宽 + 两侧余量；绕行中放宽滞回量防临界抖动
        need = 2.0 * (self._ego_half_w + NUDGE_MARGIN_EGO)
        if self._nudge_active:
            need -= NUDGE_HYST
        # 诊断：全部自由区间（含不可用者），CAND 表头输出——回答
        # 「有哪些空隙/各多宽/为什么没选或没绕」
        self._nudge_diag = {
            "gaps": [(a, b) for a, b in self._free_intervals(occ, lo, hi)],
            "need": need}
        best_gap = None
        for a, b in self._nudge_diag["gaps"]:
            if b - a >= need:
                c = (a + b) / 2.0
                if best_gap is None or abs(c - ego_l) < abs(best_gap[0] - ego_l):
                    best_gap = (c, b - a)
        if best_gap is None:
            return None
        gap_l, gap_w = best_gap
        # ── 承诺判定（修中途自取消）：「已离开本道」即视为绕行承诺——一旦
        # 横移开始就不受距离门控约束（中途取消只会剩本道候选 → 兜底急刹 →
        # 死锁）。旧判据 |ego_l−gap_l|>0.3 会在横移到一半（最不该取消的时刻）
        # 误杀剖面；且停死后定位噪声 ±0.25m 反复穿越 0.3 阈值导致闪断
        committed = abs(ego_l) > 0.5
        # ── shift 点横向剖面（Autoware avoidance 式）：
        # 贴边渐变 → 穿越保持 → 车尾超过最前障碍头部 return_x 后渐变回道
        s_rear = min(o["s"] - o["half_len"] for o in blockers)
        s_front = max(o["s"] + o["half_len"] for o in blockers)
        ramp = max(NUDGE_RAMP_MIN, 1.5 * v_long)
        # 贴边完成期限：进入阻挡障碍「碰撞包络」前必须完成横向分离——包络
        # 前缘 = 障碍后缘 − 本车半长 − 纵向余量（再留 0.5m）。完成点晚于
        # 此处，剖面会斜穿包络对角线（纵向已入包络、横向尚未分开），被逐点
        # 碰撞硬约束整条拒掉：CRUISE/NDG 全灭、仅 v≈0 的 STOP/NDG 靠轨迹
        # 不前进苟活 → 刹停-爬行循环卡死在障碍正前方（实测主因之二）。
        # 旧完成点 s_rear−0.5 比包络前缘晚 3m+，从未正确过
        s_shift_end = s_rear - self._ego_half_len - COLL_S - 0.5
        _avail = s_shift_end - ego_s
        if _avail < ramp:
            # 距离不足：压缩渐变段（重锚到当前位置，兼修 v≈0 死锁恢复——
            # 重锚后 lat(s) 从 ego_l 平滑渡到空隙中心，无剖面跳变）
            if _avail >= NUDGE_RAMP_MIN_S:
                ramp = _avail               # 压缩渐变段，贴边更陡但可行
            elif committed:
                ramp = NUDGE_RAMP_MIN_S     # 已横移中：尽力陡移，硬约束兜底
            else:
                return None                 # 太近且未横移：跟停兜底
            s_shift_end = ego_s + ramp
        ret_x = max(NUDGE_RETURN_X, 0.5 * v_long)   # 相对速度越快回道余量越大
        s_ret_start = s_front + ret_x + self._ego_half_len

        def lat(s):
            if s <= s_shift_end - ramp:
                return ego_l
            if s <= s_shift_end:
                x = (s - (s_shift_end - ramp)) / ramp
                return ego_l + (gap_l - ego_l) * x * x * (3.0 - 2.0 * x)
            if s <= s_ret_start:
                return gap_l
            if s <= s_ret_start + ramp:
                x = (s - s_ret_start) / ramp
                return gap_l * (1.0 - x * x * (3.0 - 2.0 * x))
            return 0.0

        return {"l": gap_l, "lat": lat, "gap": gap_w, "n": len(blockers)}

    def step(self, *, t, spd, v_long, ego_s, ego_l, obstacles, tl,
             aggressive_on, target_speed, safe_dist, wp_idx,
             prev_thr, prev_brk, yaw_err_deg=None) -> PlanOutput:
        ref = self._ref
        plan_log = self._plan_log
        red_stop_s = tl.red_stop_s
        tl_state = tl.state
        tl_dist = tl.dist

        # ── 参考线跳变检测（日志增强#2/#8）：帧间 Δs 远超车速×Δt（物理不可
        # 能）或 Δl 突变 → 参考线在路口拐点重关联。两次实测均在机动中途发生，
        # s/l/可行驶域全部不连续（域镜像翻转、自车 l 一帧跳 1.7m），直接
        # 杀死进行中的绕行/换道剖面。打事件时带上 road/lane id 便于定位
        # 重关联发生在哪个路段
        if self._prev_pose is not None:
            _pt, _ps, _pl, _pv = self._prev_pose
            _dt = max(1e-3, t - _pt)
            _ds_ex = _pv * _dt
            _ds_act = ego_s - _ps
            _dl = ego_l - _pl
            if (abs(_ds_act - _ds_ex) > 1.0 or abs(_dl) > 1.0):
                _wp_l = "无"
                try:
                    _wp_cur = ref.route_wps[wp_idx] if wp_idx < len(ref.route_wps) else None
                    if _wp_cur is not None:
                        _wp_l = f"road{_wp_cur.road_id}/lane{_wp_cur.lane_id}"
                except Exception:
                    pass
                plan_log(f"EVENT 参考线跳变: Δs={_ds_act:+.2f}m(预期{_ds_ex:+.2f}m) "
                         f"Δl={_dl:+.2f}m v={_pv:.1f}m/s wp={wp_idx} 车道={_wp_l} "
                         f"——疑似路口重关联，剖面/域不连续")
        self._prev_pose = (t, ego_s, ego_l, v_long)

        # ── 停止线-障碍重叠事件（日志增强#7）：障碍停在红灯停止线上（或极近）
        # 时，横向可通过也过不去——显式绑定两个 s，避免人肉对齐才发现
        _tl_obs = (red_stop_s is not None and any(
            abs(o["s"] - red_stop_s) < o["half_len"] + 2.0 for o in obstacles))
        if _tl_obs and not self._tl_obs_prev:
            _o_tl = min(obstacles, key=lambda o: abs(o["s"] - red_stop_s))
            plan_log(f"EVENT 停止线被障碍占用: 红灯停止线 s={red_stop_s:.1f}m 与 "
                     f"障碍(s={_o_tl['s']:.1f}, l={_o_tl['l']:+.1f}) 重叠——"
                     f"横向通过也会被红灯封死")
        self._tl_obs_prev = _tl_obs

        # ── 刹停余量（safe_distance 滑杆联动，默认12→6m）──
        STOP_MARGIN = max(3.0, safe_dist * 0.5)
        a_need = 0.0
        desired = target_speed
        self._nudge_diag = None   # 本帧空隙诊断（_find_nudge 触发时填充）

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
                    if ego_s + 0.5 < rear < ego_s + self.dec_win or ego_s + 0.5 < o["s"] + o["half_len"] < ego_s + self.dec_win:
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
        # ── 意图失效降级（修死锁）：决策层走廊判据只看「目标横向带内有无
        # 障碍」，不含过渡段几何——横向衰减/临界间隙下 intent 候选会连续被
        # 碰撞硬约束拒绝。此时 intent 实际不可达，若不降级，nudge/借道分支
        # （触发条件 intent_l==0）永远进不去 → 兜底刹停死锁。连续
        # INTENT_FAIL_N 帧无幸存候选 → 本帧降级归零；锁存 INTENT_RETRY_S
        # 秒后自动重试（nudge 进行中推迟重试，防止中途被拽回换道）
        _intent_raw = intent_l           # 降级前的决策层原始意图
        _intent_degraded = False
        if intent_l != 0.0:
            _key = round(intent_l, 2)
            if self._intent_fail_key != _key:
                self._intent_fail_key = _key   # 换了意图目标：计数清零重来
                self._intent_fail_cnt = 0
                self._intent_latch_t = None
            if (self._intent_latch_t is not None
                    and t - self._intent_latch_t >= INTENT_RETRY_S):
                if self._nudge_prev:
                    self._intent_latch_t = t   # nudge 进行中：推迟重试
                else:
                    self._intent_fail_cnt = 0  # 锁存到期：重试该意图
                    self._intent_latch_t = None
            if self._intent_fail_cnt >= INTENT_FAIL_N:
                _intent_degraded = True
                if self._intent_latch_t is None:
                    self._intent_latch_t = t
                    _wp_l = "无"
                    try:
                        _wp_cur = (ref.route_wps[wp_idx]
                                   if wp_idx < len(ref.route_wps) else None)
                        if _wp_cur is not None:
                            _wp_l = f"road{_wp_cur.road_id}/lane{_wp_cur.lane_id}"
                    except Exception:
                        pass
                    plan_log(f"EVENT 意图失效降级: 邻道 {intent_l:+.1f}m 换道候选连续 "
                             f"{INTENT_FAIL_N} 帧被硬约束拒绝，判定不可达；"
                             f"放行贴边绕行/借道分支，{INTENT_RETRY_S:.0f}s 后重试 "
                             f"（wp={wp_idx} 车道={_wp_l}）")
                intent_l = 0.0
        # ── 贴边绕行（nudge）决策：先于激进借道（路内空隙比对向借道安全）。
        # 触发条件：本道被堵 + 无同向邻道可换（intent_l==0，含「邻道不存在」
        # 与「意图失效降级」两种情况）+ 阻挡障碍静止/低速 + 横向存在能容纳
        # 本车+余量的空隙（_find_nudge 内判定）
        nudge = None
        if _blocked and intent_l == 0.0:
            nudge = self._find_nudge(obstacles, ego_s, ego_l, v_long, l_min, l_max)
            # 承诺锁存（防御）：绕行中途（已离开本道）空隙判定瞬时空转时，
            # 沿用 1.5s 内的上一帧剖面——速度/定位噪声让 nudge 候选闪断会
            # 触发兜底刹停 → 死锁振荡。安全性仍由逐点碰撞硬约束兜底
            if nudge is not None:
                self._nudge_last = (t, nudge)
            elif (self._nudge_active and abs(ego_l) > 0.5
                    and self._nudge_last is not None
                    and t - self._nudge_last[0] < 1.5):
                nudge = self._nudge_last[1]
        # 激进模式兜底：本道被占且无同向邻道可绕（单车道+对向道、或邻接链
        # 断裂致边界收缩）→ 借邻接车道绕行（通常是对向道）。目标偏移取
        # 邻接车道中心（相对参考线），走廊须无障碍；轨迹层逐点校验仍在
        # Driving 路面 + 全量碰撞检查 + 借道限速，绕过障碍后自动回本道
        borrow_l = None
        if aggressive_on and _blocked and intent_l == 0.0 and nudge is None:
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
        _cand_dbg = []              # 全部候选尝试（含被拒/跳过）明细：CAND 行用
        _rej_bounds = _rej_red = _rej_coll = _rej_dir = 0   # 拒绝统计（日志用）
        _n_steps = int(T_HORIZON / DT_PLAN)
        T_lat = max(2.0, min(4.0, 1.2 * max(2.0, v_long)))
        if intent_l != 0.0:
            # 避让换道：横向过渡须在抵达障碍前完成（全量进入邻道后再与障碍
            # 平行）。默认 T_lat 按 1.2·v 拉长（≈21m 才换完），转弯路段横向
            # 进展又慢，到障碍处只剩 1~2m 偏移，被碰撞硬约束拒掉 → 换道/刹停
            # 横跳。这里按"障碍前缘距离 / 纵向速度 − 0.8s"压缩过渡时长，
            # 保证驶到障碍跟前时已全量进入邻道。只统计与「本道或目标走廊」
            # 重叠的障碍——邻道自己的障碍不迫近本车换道，不应压缩换道时长
            # 导致不必要的急换道
            _obs_front = min((o["s"] - o["half_len"] for o in obstacles
                              if (abs(o["l"] - ego_l) < self._ego_half_w + o["half_w"] + 0.25
                                  or abs(o["l"] - intent_l) < self._ego_half_w + o["half_w"] + 0.25)
                              and o["s"] - o["half_len"] > ego_s + 0.5),
                             default=float("inf"))
            if _obs_front < float("inf"):
                _t_avail = max(1.2, (_obs_front - ego_s) / max(1.0, v_long) - 0.8)
                T_lat = min(T_lat, _t_avail)
            else:
                T_lat = min(T_lat, 3.0)
        # 候选规格：横向目标 + 各自限速/停驻点/横向剖面 + 意图对齐惩罚。
        # 车道级候选=时间基五次多项式；贴边绕行=s 基 shift 点剖面。
        # 对齐惩罚（P0 修）：4s 时域短视——绕行收益在时域外，时域内只有代价，
        # 若无保护，myopic 代价会让本道 STOP「停车等待」险胜绕行。故 nudge
        # 触发时视其为该场景的意图，本道候选降级为兜底（+5，同换道惯例）；
        # 决策优先级显式编码在代价里，而非隐式依赖时域内收益
        specs = []
        if nudge is not None:
            specs.append({"l_t": nudge["l"], "is_borrow": False, "is_nudge": True,
                          "check_dir": False, "lat_fn": nudge["lat"],
                          # 绕行限速：缩短侧向近距离暴露时间（同借道惯例）
                          "v_cap": max(3.0, target_speed * NUDGE_V_CAP_FRAC),
                          "s_stop": _corridor_stop(nudge["l"]),
                          "align_pen": 0.0})
            specs.append({"l_t": 0.0, "is_borrow": False, "is_nudge": False,
                          "check_dir": False, "lat_fn": None,
                          "v_cap": target_speed, "s_stop": _corridor_stop(0.0),
                          "align_pen": 5.0})
        else:
            for l_t in lat_targets:
                is_borrow = borrow_l is not None and l_t == borrow_l
                # 借道限速：绕障机动期间降速通过，缩短对向风险暴露时间
                specs.append({"l_t": l_t, "is_borrow": is_borrow, "is_nudge": False,
                              "check_dir": True, "lat_fn": None,
                              "v_cap": max(3.0, target_speed * 0.5) if is_borrow else target_speed,
                              "s_stop": _corridor_stop(l_t),
                              "align_pen": 0.0 if l_t == intent_l else 5.0})
        # 当前已处于碰撞窗口内的障碍索引（修死锁吸收态）：贴边绕行进行到一半
        # 或兜底刹停越界时自车与障碍横向重叠，此时逐点碰撞检查会把第一个
        # 采样点也判为碰撞 → 所有候选被拒 → FALLBACK 吸收态无法自愈。
        # 对此类障碍只拒绝「更深入」的采样点（横向分离与纵向距离均未改善），
        # 放行逃离轨迹（横移出窗口/倒拉开距离）
        _coll_now_idx = {
            i for i, o in enumerate(obstacles)
            if (abs(ego_s - o["s"]) < o["half_len"] + self._ego_half_len + COLL_S
                and abs(ego_l - o["l"]) < o["half_w"] + self._ego_half_w + COLL_L)}
        for spec in specs:
            l_t = spec["l_t"]
            is_borrow = spec["is_borrow"]
            lat_fn = spec["lat_fn"]
            v_cap = spec["v_cap"]
            s_stop = spec["s_stop"]
            for mode in ("CRUISE", "STOP"):
                if mode == "STOP" and s_stop == float("inf"):
                    _cand_dbg.append(f"STOP/l={l_t:+.1f} 跳(无停驻点)")
                    continue   # 无停驻点则无需 STOP 候选
                if mode == "CRUISE":
                    a_long = max(-2.0, min(1.5, (v_cap - v_long) / 2.0))
                else:
                    d = s_stop - ego_s
                    a_long = 0.0 if d <= 0.5 else max(-MAX_DECEL, -(v_long * v_long) / (2.0 * d))
                # 逐时刻展开：纵向逐步积分 + 横向剖面（车道级=五次多项式最小
                # 急动度；nudge=shift 点剖面）。
                # CRUISE 按停驻点（红灯/障碍）生成舒适制动剖面
                # v ≤ √(2·COMFORT_A·(s_stop−s))，到停止线恰好停住。
                # 顺序要点（修期望速度横跳）：① 物理减速度下限先施加；
                # ② 剖面钳制最后施加且允许超过舒适值——若剖面放在下限之前，
                # 贴线归零会被下限顶回 v>0.3，整条 CRUISE 被红灯硬约束拒掉，
                # 与 STOP 候选逐帧轮替胜出 → desired 在最大/最小间跳变。
                samples = []
                ok = True
                _why = ""                   # 被拒原因（域/红/碰），CAND 行用
                _rej_pt = None              # 拒绝点 (t, s, l)：拒绝发生在哪
                _rej_obs = None             # 触发碰撞的障碍 (s, l)：撞的是谁
                dl = l_t - ego_l
                s_prev, v_prev = ego_s, v_long
                for k in range(1, _n_steps + 1):
                    tk = k * DT_PLAN
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
                    # 横向：先算 s 再取 l（nudge 剖面是 s 的函数；车道级
                    # 候选是时间基五次多项式，结果与原先一致）
                    if lat_fn is not None:
                        l_k = lat_fn(s_k)
                    else:
                        tau = min(1.0, tk / T_lat)
                        l_k = ego_l + dl * tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)
                    s_prev, v_prev = s_k, v_k
                    # 硬约束①：可行驶域（越界即逆行/出路缘，整条拒绝）。
                    # 逐点取「当地」边界——轨迹展开 30m+，前方路段可能收窄/
                    # 变两车道，只用车头处边界会放行前方的对向车道（蓝线逆行）。
                    # 借道候选例外：不受同向域限制，改为逐点地图级校验
                    # （Driving 路面即可，不限方向）——对向道/邻接链断裂处
                    # 按此放行，但出路缘仍拒绝。nudge 候选走常规同向域校验
                    # （空隙本就在可行驶域内，含路缘余量）
                    if is_borrow:
                        if not ref.lat_driving(l_k, s_k):
                            _rej_bounds += 1
                            _why = "域"
                            _rej_pt = (tk, s_k, l_k)
                            ok = False
                            break
                    elif l_min is not None:
                        b_k = ref.bounds_at_s(s_k)
                        if b_k is None:
                            b_k = (l_min, l_max)
                        if (l_k < min(b_k[0] + self._ego_half_w + COLL_L, ego_l - 0.05)
                                or l_k > max(b_k[1] - self._ego_half_w - COLL_L, ego_l + 0.05)):
                            # 过渡段域错位兜底（修绕行振荡死锁，实测根因）：
                            # S 式换道过渡处参考线已切到新车道中心，缓存域边界
                            # 相对「当地车道中心」计量，而 l_k 相对「连续参考
                            # 曲线」——两坐标系在过渡段错位可达数米，逐点比较
                            # 把本可通行的绕行/换道剖面误拒（v>0 时轨迹前进即
                            # 触发、v=0 时靠 ego_l 豁免幸存 → 刹停-爬行循环）。
                            # 改用地图级同向 Driving 校验裁决（世界坐标，免疫
                            # 坐标系错位）；路缘/对向仍被正确拒绝
                            if not ref.lat_driving_fwd(l_k, s_k):
                                _rej_bounds += 1
                                _why = "域"
                                _rej_pt = (tk, s_k, l_k)
                                ok = False
                                break
                    # 硬约束②：红灯（CRUISE 不得带速越过停止线）
                    if (mode == "CRUISE" and red_stop_s is not None
                            and s_k > red_stop_s and v_k > 0.3):
                        _rej_red += 1
                        _why = "红"
                        _rej_pt = (tk, s_k, l_k)
                        ok = False
                        break
                    # 硬约束③：碰撞——对全量障碍（含长度、含预测外推、含旁道）
                    for _oi, o in enumerate(obstacles):
                        if o["s"] < ego_s - 1.0 and o["v_s"] > v_long:
                            continue   # 后方更快的超车车辆：后车责任，不因此误刹
                        s_o = self._predictor.extrapolate(o, tk)
                        if (abs(s_k - s_o) < o["half_len"] + self._ego_half_len + COLL_S
                                and abs(l_k - o["l"]) < o["half_w"] + self._ego_half_w + COLL_L):
                            if _oi in _coll_now_idx:
                                # 已在窗口内：横向分离或纵向距离任一在改善即视为
                                # 逃离中，放行；两者都停滞/更深才拒绝
                                if (abs(l_k - o["l"]) > abs(ego_l - o["l"]) + 1e-6
                                        or abs(s_k - s_o) > abs(ego_s - o["s"]) + 1e-6):
                                    continue
                            _rej_coll += 1
                            _why = "碰"
                            _rej_pt = (tk, s_k, l_k)
                            _rej_obs = (o["s"], o["l"])
                            ok = False
                            break
                    if not ok:
                        break
                    samples.append((tk, s_k, l_k, v_k))
                if not ok or not samples:
                    _pt = (f"@t{_rej_pt[0]:.2f}s(s{_rej_pt[1]:.1f},l{_rej_pt[2]:+.1f})"
                           if _rej_pt else "")
                    _ob = (f" 撞obs(s{_rej_obs[0]:.1f},l{_rej_obs[1]:+.1f})"
                           if _rej_obs else "")
                    _cand_dbg.append(f"{mode}/l={l_t:+.1f} 拒({_why or '?'}){_pt}{_ob}")
                    continue
                # 换道候选：中段与末端落点必须是同向 Driving 车道。
                # 边界缓存按「邻接链+点积」推断，路口/车道斜接处可能漏判对向
                # （如左换道落在交叉来车道上）——地图级查询兜底，逆行零容忍。
                # 借道候选（激进模式）显式允许对向，跳过方向校验。
                # nudge 候选目标在车道边缘而非车道中心，同向校验不适用，
                # 安全性由空隙计算（余量）+ 逐点边界校验双重保证
                if spec["check_dir"] and abs(dl) > 0.5:
                    if not (ref.lat_lane_ok(l_t, samples[len(samples) // 2][1])
                            and ref.lat_lane_ok(l_t, samples[-1][1])):
                        _rej_dir += 1
                        _cand_dbg.append(f"{mode}/l={l_t:+.1f} 拒(向)")
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
                        + spec["align_pen"])
                cands.append({"cost": cost, "l_t": l_t, "mode": mode,
                              "s_stop": s_stop, "a_long": a_long,
                              "v_cap": v_cap, "samples": samples,
                              "nudge": spec["is_nudge"]})
                # CAND 行数据：代价分项（jl偏离/jd横摆/a舒适/ef效率/mv机动/al对齐）
                # + 限速/停驻点/末端(s,l,v)，ref 用于事后标记最优
                _cand_dbg.append({
                    "mode": mode, "l_t": l_t, "nudge": spec["is_nudge"],
                    "cost": cost,
                    "terms": (0.5 * j_lat, 0.3 * j_dl, 0.2 * a_long * a_long,
                              0.4 * max(0.0, target_speed - v_end),
                              0.6 if abs(l_t) > 0.5 else 0.0, spec["align_pen"]),
                    "v_cap": v_cap, "a_long": a_long, "s_stop": s_stop,
                    "end": samples[-1], "ref": cands[-1]})

        # 意图失效计数更新：intent 横向目标本帧无任何幸存候选（CRUISE 与
        # STOP 全被拒）→ 计数+1；有幸存 → 清零；无意图（本道未堵/障碍清
        # 空）→ 清零并解锁。锁存期间（已降级）计数冻结，等 INTENT_RETRY_S
        # 到期重试
        if _intent_raw == 0.0:
            self._intent_fail_cnt = 0
            self._intent_latch_t = None
        elif not _intent_degraded:
            if any(abs(c["l_t"] - _intent_raw) < 0.15 for c in cands):
                self._intent_fail_cnt = 0
            else:
                self._intent_fail_cnt += 1

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
            avoiding = best.get("nudge", False) or abs(best["l_t"]) > 0.5
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
        # 贴边绕行事件（含空隙滞回状态更新——_find_nudge 下一帧用放宽门槛）
        _best_nudge = best is not None and best.get("nudge", False)
        self._nudge_active = _best_nudge
        if _best_nudge and not self._nudge_prev:
            plan_log(f"EVENT 贴边绕行: 空隙 {nudge['gap']:.1f}m "
                     f"目标偏移 {nudge['l']:+.1f}m（两车道均被占，"
                     f"阻挡障碍 {nudge['n']} 个，限速通过）")
        elif not _best_nudge and self._nudge_prev:
            plan_log("EVENT 贴边绕行结束")
        self._nudge_prev = _best_nudge
        _obs_str = ",".join(f"{o['s']:.0f}/{o['l']:+.1f}" for o in obstacles[:3])
        _cost_str = f"/c={best['cost']:.2f}" if best is not None else ""
        _ndg_str = f"Y({nudge['gap']:.1f}m)" if _best_nudge else "N"
        # nudge 剖面跟踪误差（日志增强#6）：当前 s 处剖面目标横向 vs 实际
        # ego_l——区分「规划要偏」与「跟踪丢了」（上次绕行后 l 漂移 −2.4→−4.3
        # 无法归因就是因为缺这个数）
        _ndg_trk = ""
        if nudge is not None:
            _tgt_l = nudge["lat"](ego_s)
            _ndg_trk = f" ndgT={_tgt_l:+5.2f} err={ego_l - _tgt_l:+5.2f}"
        # 参考线-自车航向夹角（日志增强#5）：拐点重关联处该角会瞬间跳变，
        # 直接暴露坐标系不连续
        _yaw_str = f" yaw_e={yaw_err_deg:+6.1f}" if yaw_err_deg is not None else ""
        plan_log(
            f"FRM t={t:6.1f} v={spd:5.2f}(lon {v_long:5.2f}) s={ego_s:7.1f} l={ego_l:+5.2f} "
            f"wp={wp_idx}{_yaw_str} "
            f"tl={tl_state[:3]}/{tl_dist:5.1f}m obs={len(obstacles)}[{_obs_str}] "
            f"blk={'Y' if _blocked else 'N'} agm={'Y' if aggressive_on else 'N'} "
            f"bor={'Y' if borrow_l is not None else 'N'} ndg={_ndg_str}{_ndg_trk} "
            f"itn={_intent_raw:+5.2f} dg={'Y' if _intent_degraded else 'N'} "
            f"itf={self._intent_fail_cnt}/{INTENT_FAIL_N} "
            f"nb={[round(x, 1) for x in _neighbors]} "
            f"cands={len(cands)} best={_bk[0]}{_cost_str} "
            f"des={desired:5.2f} a={a_need:5.2f} "
            f"thr={prev_thr:.2f} brk={prev_brk:.2f} "
            f"rej:域{_rej_bounds}/红{_rej_red}/碰{_rej_coll}/向{_rej_dir})")

        # 候选明细（排障主数据之一，多行可读格式）：表头一行 + 每个候选一行。
        # 表头：本帧决策上下文（是否被堵/原始意图/降级状态/邻道/可行驶域/
        # 横向过渡时长/nudge 空隙/存活数）。候选行：幸存者含代价分项与末端
        # 状态（★ 标最优），被拒者含拒绝原因、拒绝时刻/位置、触发障碍。
        # 代价分项：jl=偏离参考线 jd=横摆平顺 a=舒适制动 ef=效率
        # mv=换道机动 al=意图对齐；末端=(s,l,v)
        _bnd = f"[{l_min:+.1f},{l_max:+.1f}]" if l_min is not None else "无"
        _nud = (f"Y(空隙{nudge['gap']:.1f}m@{nudge['l']:+.1f}m)" if nudge is not None
                else "N")
        # 空隙全景（日志增强#4）：全部自由区间 + 所需宽度。回答「为什么选左
        # 不选右」「有哪些空隙但太窄没绕」。无 nudge 触发但本帧被堵时同样
        # 输出（若 _find_nudge 已跑过），空隙仲裁一目了然
        _gaps_str = ""
        if self._nudge_diag is not None:
            _g = self._nudge_diag["gaps"]
            _gaps_str = (" 空隙=" + (",".join(f"({a:+.1f},{b:+.1f})宽{b - a:.1f}"
                                             for a, b in _g) if _g else "无")
                         + f" 需宽{self._nudge_diag['need']:.1f}m")
        plan_log(
            f"CAND t={t:6.1f} blk={'Y' if _blocked else 'N'} "
            f"itn={_intent_raw:+.2f}{'(已降级)' if _intent_degraded else ''} "
            f"itf={self._intent_fail_cnt}/{INTENT_FAIL_N} "
            f"nb={[round(x, 1) for x in _neighbors]} 域={_bnd} "
            f"T_lat={T_lat:.1f}s ndg={_nud}{_gaps_str} 存活={len(cands)}")
        for _i, e in enumerate(_cand_dbg, 1):
            if isinstance(e, str):
                plan_log(f"  cand{_i}: {e}")
                continue
            _t = e["terms"]
            _ss = "inf" if e["s_stop"] == float("inf") else f"{e['s_stop']:.1f}"
            _end = e["end"]
            plan_log(
                f"  cand{_i}: {'★' if e['ref'] is best else ' '} {e['mode']}/l={e['l_t']:+.1f}"
                f"{'/NDG' if e['nudge'] else ''} 总代价={e['cost']:.2f} "
                f"[jl{_t[0]:.2f} jd{_t[1]:.2f} a{_t[2]:.2f} ef{_t[3]:.2f} "
                f"mv{_t[4]:.1f} al{_t[5]:.1f}] "
                f"限速={e['v_cap']:.1f} a={e['a_long']:+.2f} 停驻={_ss} "
                f"末端=(s{_end[1]:.0f}, l{_end[2]:+.1f}, v{_end[3]:.1f})")

        # SSE 兼容：换道目标车道（鸟瞰图高亮）。
        # 高亮前做同向校验——邻接链查询可能拿到对向/交叉车道，高亮到逆行
        # 车道上会误导观测（规划层已有 lat_lane_ok 双保险，此处管展示）。
        # 贴边绕行不换车道（在自身可行驶域内穿越），不高亮邻道
        avoid_dest_lane = None
        if avoiding and not _best_nudge:
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
            and ego_s < o["s"] + o["half_len"] < ego_s + self.dec_win
            for o in obstacles)
        if _best_nudge:
            _fsm_new = "NUDGE"
        elif avoiding:
            _fsm_new = "LANE_CHANGE"
        elif red_stop_s is not None:
            _fsm_new = "APPROACH_RED"
        elif _blocked_now:
            _fsm_new = "FOLLOW"
        else:
            _fsm_new = "CRUISE"
        # 状态/换道转换日志（教学观测用；CRUISE 回归不刷屏）
        if avoiding and not self._avoid_prev:
            if _best_nudge:
                self._log(f"两车道均被占，贴边绕行（空隙 {nudge['gap']:.1f}m，"
                          f"目标偏移 {avoid_lat_target:+.1f}m，限速通过）")
            else:
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

        # 黄灯：软约束兜底——仅当感知层未提供黄灯停驻点时生效（perception_v2
        # 红/黄统一输出停驻点，走上方常规停驻剖面；legacy 感知黄灯无停驻点，
        # 保留旧行为）
        YELLOW_D = 1.0
        if (tl_state == "yellow" and red_stop_s is None
                and a_need < YELLOW_D):
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
            nudge=_best_nudge,
            nudge_gap=(nudge["gap"] if _best_nudge else 0.0),
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
