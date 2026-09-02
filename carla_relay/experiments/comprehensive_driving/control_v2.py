"""实验10 控制层 v2：前馈+反馈纵向控制 + 横向 Pure Pursuit（激进重写版）。

与 legacy（control.py）的纵向差异——修三处结构性问题：

  1. 去掉 0.25 油门基线：旧律 err=0 时仍有 0.25 油门，超速 ~2m/s 内完全不
     减速，轻微减速被迫全部走 a_need 强制制动分支（规划层还要为此打补丁，
     见 planner.py「0.25 油门基线自身减不了速」注释）。新律油门 = P·err + I，
     误差为负油门自然归零 → 发动机倒拖减速，轻微减速不再依赖制动；
  2. 制动 = 前馈 + 反馈：a_need（规划剖面前馈，管定点停车精度）+ Kp·超速量
     （反馈，管巡航跟踪），两项合成所需减速度后按占比输出制动——旧律只有
     前馈一条路，规划没请求减速时超速无人管；
  3. 停车保持归位控制层：已停且期望近零 → 持续制动防蠕行（旧实现写在规划
     层 STOP/CRUISE 两处，a_need=1.5 hack）。

横向与 legacy 完全一致（Pure Pursuit 前视点取自规划轨迹 + 转向延迟队列），
接口契约不变：step() 签名、CtrlOutput、prev_thr/prev_brk、update_params
均与 VehicleController 相同，run.py 按 json 的 "controller" 配置切换。
"""
from __future__ import annotations

import math

from carla_relay.experiments.comprehensive_driving.frames import CtrlOutput


class VehicleControllerV2:
    """纵横向控制器 v2（有状态：油门积分项 + 转向延迟队列）。"""

    WHEELBASE = 2.85          # 轴距（m）
    MAX_STEER_RAD = 1.22      # 前轮角极限（≈70°，[-1,1] 满舵映射）
    KP_STEER_BASE = 1.4       # 教学灵敏度基准（前端默认值，除以它归一）

    # ── 纵向参数（调参入口）──
    KP_THR = 0.18             # 油门 P 增益（每 m/s 速度误差）
    KI_THR = 0.02             # 油门 I 增益（消除坡道/滚动阻力稳态误差）
    I_MIN, I_MAX = -0.25, 0.5  # 积分钳位（单一钳位，抗饱和）
    KP_V = 0.35               # 超速→减速度反馈增益
    BRAKE_GATE = 0.3          # 制动门槛（m/s²）：低于此交给油门自然回落
    HOLD_V = 0.3              # 停车保持：期望速度低于此视为「要停」
    HOLD_SPD = 0.5            # 停车保持：车速低于此视为「已停」
    HOLD_BRAKE = 0.5          # 停车保持制动力

    def __init__(self, kp_steer, lookahead, steer_delay, brake_force, max_decel):
        self._kp_steer = kp_steer
        self._lookahead = lookahead
        self._steer_delay = steer_delay
        self._brake_force = brake_force
        self._max_decel = max_decel
        self._thr_i = 0.0                # 油门积分项（PI 控制的 I）
        self.steer_history = []          # 转向延迟队列
        self.prev_thr = 0.0              # 上一帧油门（FRM 日志用，滞后一帧）
        self.prev_brk = 0.0              # 上一帧制动

    def update_params(self, kp_steer=None, lookahead=None, steer_delay=None,
                      brake_force=None):
        """运行中实时调整（前端滑杆 → /experiment/10/params）。"""
        if kp_steer is not None:
            self._kp_steer = kp_steer
        if lookahead is not None:
            self._lookahead = lookahead
        if steer_delay is not None:
            self._steer_delay = steer_delay
        if brake_force is not None:
            self._brake_force = brake_force

    def step(self, *, desired, a_need, spd, loc, yaw_rad, plan_traj, route_wp,
             wp_idx) -> CtrlOutput:
        """一帧控制。loc/yaw_rad 为融合定位/融合航向；plan_traj 为规划轨迹。"""
        # ── 纵向：制动（前馈+反馈）vs 油门 PI（无基线）──
        # 所需减速度 = 规划剖面前馈（a_need，定点停车精度）+ 超速反馈（巡航
        # 跟踪，规划未请求减速时超速也有人管）。低于门槛时不出制动——轻微
        # 减速由油门自然回落（发动机倒拖）覆盖
        a_req = max(0.0, a_need) + self.KP_V * max(0.0, spd - desired)
        throttle = 0.0
        brake = 0.0
        if a_req > self.BRAKE_GATE:
            brake = min(1.0, (a_req / self._max_decel) * self._brake_force)
            self._thr_i = max(0.0, self._thr_i * 0.9)  # 制动期积分衰减（抗饱和）
        else:
            err = desired - spd
            self._thr_i = max(self.I_MIN, min(self.I_MAX,
                                              self._thr_i + self.KI_THR * err))
            # 无基线：误差为负 → 油门归零 → 倒拖减速；正误差按 P 增益给出
            throttle = max(0.0, min(1.0, self.KP_THR * err + self._thr_i))
        # 停车保持（从规划层归位）：已停且期望近零 → 持续制动防蠕行
        if desired < self.HOLD_V and spd < self.HOLD_SPD:
            throttle = 0.0
            brake = max(brake, self.HOLD_BRAKE)
        self.prev_thr, self.prev_brk = throttle, brake

        # ── 横向：Pure Pursuit（前视点取自规划轨迹）──
        # 1) 先在路线点列上校正 wp_idx（进度推进/障碍窗口/可视化共用）；
        # 2) 前视点 = 规划轨迹上距自车 ≥ lookahead 的第一个点（含换道 S 弯，
        #    跟踪的就是规划器输出本身）；
        # 3) 规划轨迹不够远（低速/临近停车）时回退到路线点列。
        lookahead = self._lookahead
        if route_wp:
            # 1) 在 wp_idx 前 50 点内找最近路点，校正 wp_idx（防止车辆越过路点后落后）
            best = wp_idx
            best_d = float("inf")
            for j in range(wp_idx, min(wp_idx + 50, len(route_wp))):
                d = math.hypot(loc.x - route_wp[j].x, loc.y - route_wp[j].y)
                if d < best_d:
                    best_d = d
                    best = j
            wp_idx = best
            # 2) 前视点：优先规划轨迹，回退路线点列
            look_x, look_y = None, None
            for px, py in plan_traj:
                if math.hypot(px - loc.x, py - loc.y) >= lookahead:
                    look_x, look_y = px, py
                    break
            if look_x is None:
                for j in range(wp_idx, len(route_wp)):
                    if math.hypot(loc.x - route_wp[j].x, loc.y - route_wp[j].y) >= lookahead:
                        look_x, look_y = route_wp[j].x, route_wp[j].y
                        break
                else:
                    look_x, look_y = route_wp[-1].x, route_wp[-1].y
            dx = look_x - loc.x
            dy = look_y - loc.y
            # 3) 转向角（标准 Pure Pursuit）。kp_steer 为教学灵敏度：除以
            # 基准 1.4 使默认时=标准幅度，调大更激进、调小更柔和。
            hdng_alpha = math.atan2(dy, dx) - yaw_rad
            steer_angle = math.atan2(2.0 * self.WHEELBASE * math.sin(hdng_alpha), lookahead)
            raw_steer = max(-1.0, min(1.0, (self._kp_steer / self.KP_STEER_BASE)
                                      * steer_angle / self.MAX_STEER_RAD))
            cte = dx * math.sin(yaw_rad) - dy * math.cos(yaw_rad)
            look_dist = math.hypot(dx, dy)
        else:
            dx = dy = 0.0
            raw_steer = 0.0
            cte = 0.0
            look_dist = 0.0
            hdng_alpha = 0.0

        # 转向延迟：指令经 N 步延迟后输出（模拟执行器滞后）
        self.steer_history.append(raw_steer)
        delay_steps = max(0, int(self._steer_delay / 0.05))
        steer = self.steer_history[max(0, len(self.steer_history) - 1 - delay_steps)] if self.steer_history else 0

        return CtrlOutput(
            steer=steer,
            throttle=throttle,
            brake=brake,
            raw_steer=raw_steer,
            cte=cte,
            look_dist=look_dist,
            hdng_alpha=hdng_alpha,
            wp_idx=wp_idx,
        )
