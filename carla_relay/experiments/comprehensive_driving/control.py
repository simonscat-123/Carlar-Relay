"""实验10 控制层：纵向 PI 速度控制 + 横向 Pure Pursuit 转向控制。

对标工业界控制层（Apollo 控制器 = 纵向 PID/横向 LQR 简化版）：

  - 纵向：期望减速度 a_need > 舒适阈值 → 按占比输出制动；否则油门 PI 控制
    （纯 P 在坡道/风阻下存在稳态误差，I 项逐帧累积消除余差；制动期 I 衰减防饱和）；
  - 横向：Pure Pursuit——前视点取自「规划轨迹」上距自车 ≥ lookahead 的第一个
    点（含换道 S 弯，跟踪的就是规划器输出本身）；规划轨迹不够远（低速/临近
    停车）时回退到路线点列；
  - 转向延迟：指令经 N 步延迟队列后输出（模拟执行器滞后，教学参数）；
  - 控制输入全部来自融合定位/融合航向（带噪）；真值仅用于误差评估。

有状态：油门积分项 _thr_i、转向延迟队列 steer_history、上帧油门/制动
（FRM 日志滞后一帧显示）。
"""
from __future__ import annotations

import math

from carla_relay.experiments.comprehensive_driving.frames import CtrlOutput


class VehicleController:
    """纵横向控制器（有状态：PI 积分项 + 转向延迟队列）。"""

    WHEELBASE = 2.85          # 轴距（m）
    MAX_STEER_RAD = 1.22      # 前轮角极限（≈70°，[-1,1] 满舵映射）
    KP_STEER_BASE = 1.4       # 教学灵敏度基准（前端默认值，除以它归一）

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
        # ── 纵向：制动 vs 油门 PI ──
        # 施加制动：低于阈值视为无需主动刹车（正常巡航/跟车），否则按所需减速度占比输出。
        throttle = 0.0
        brake = 0.0
        COMFORT_D = 0.8
        if a_need > COMFORT_D:
            brake = min(1.0, max(0.0, (a_need / self._max_decel) * self._brake_force))
            self._thr_i = max(0.0, self._thr_i * 0.9)   # 制动期积分衰减（抗饱和）
        else:
            err = desired - spd
            self._thr_i = max(-0.3, min(0.6, self._thr_i + 0.015 * err))
            throttle = max(0.0, min(1.0, 0.25 + 0.12 * err + self._thr_i))
        self.prev_thr, self.prev_brk = throttle, brake

        # ── 横向：Pure Pursuit（前视点取自规划轨迹）──
        # 1) 先在路线点列上校正 wp_idx（进度推进/障碍窗口/可视化共用）；
        # 2) 前视点 = 规划轨迹上距自车 ≥ lookahead 的第一个点（含换道 S 弯，
        #    替代原「路线点+法向平移」的两段式做法——跟踪的就是规划器输出本身）；
        # 3) 规划轨迹不够远（低速/临近停车）时回退到路线点。
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
            # 3) 转向角（标准 Pure Pursuit）。
            # steer_angle/1.22 把前轮角(最大约70°=1.22rad)映射到 [-1,1] 已是合理幅度；
            # kp_steer 为教学灵敏度：前端默认 1.4，除以 1.4 使默认时=标准幅度，
            # 调大更激进、调小更柔和，避免直接乘 kp 导致转度过大冲出路面。
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
