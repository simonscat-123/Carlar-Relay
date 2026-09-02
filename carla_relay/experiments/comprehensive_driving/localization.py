"""实验10 定位层：GNSS + INS 互补滤波融合定位（位置 + 航向）。

对标工业界组合导航（GNSS/INS 松耦合 + 地图匹配）：
  - GNSS 给出带噪绝对位置观测（实验用高斯噪声模拟，含 GPS 失效场景）；
  - INS 以「真值车速差分角速度」递推位姿（等效里程计/陀螺，叠加有界噪声）；
  - 互补滤波融合：位置 α·GNSS + (1-α)·INS；航向用卷绕后的相位差
    （innovation）做互补滤波，避免 ±180° 边界的跳变；
  - 防御性限幅：融合结果钳制在地图范围内，防 NaN/异常漂移污染误差统计。

输出 LocFrame（融合位姿 + 诊断量）；真值仅用于误差评估，控制链路
全部消费融合值——与真实系统一致。
"""
from __future__ import annotations

import math
import random

import carla

from carla_relay.experiments.comprehensive_driving.frames import LocFrame


class Localizer:
    """GNSS/INS 互补滤波定位器（有状态：INS 速度/位置递推 + 融合航向）。"""

    def __init__(self, gnss_noise: float, ins_noise: float, alpha: float, init_loc):
        self.gnss_noise = gnss_noise
        self.ins_noise = ins_noise
        self.alpha = alpha
        self.fused_loc = init_loc      # carla.Location，融合位置
        self.ins_vel = carla.Vector3D()
        self.fused_yaw_deg = 0.0       # 融合航向（度）
        self.prev_gt_yaw_deg = None    # 上一帧真值航向，差分出角速度（可靠源）

    def step(self, gt_loc, gt_yaw: float, true_vel, gnss_ok: bool,
             gps_failure: bool) -> LocFrame:
        """一帧融合。gt_loc/gt_yaw/true_vel 为真值（供观测合成与误差评估），
        gnss_ok 表示本帧 GNSS 观测是否可用（帧缓存里有无数据）。"""
        # GNSS 观测（带噪）：GPS 失效场景下 30% 帧完全丢星
        if gnss_ok and not (gps_failure and random.random() < 0.3):
            noise = self.gnss_noise * (20.0 if gps_failure else 4.0)
            ngx = gt_loc.x + random.gauss(0, noise)
            ngy = gt_loc.y + random.gauss(0, noise)
        else:
            ngx = gt_loc.x + random.gauss(0, self.gnss_noise * 10)
            ngy = gt_loc.y + random.gauss(0, self.gnss_noise * 10)

        # INS 推算：以真值车速为基准叠加有界测量噪声（等效里程计/INS 误差）。
        # 速度加噪后限幅，防止个别异常帧把推算值拉飞
        self.ins_vel.x = max(-60.0, min(60.0, true_vel.x + random.gauss(0, self.ins_noise * 8.0)))
        self.ins_vel.y = max(-60.0, min(60.0, true_vel.y + random.gauss(0, self.ins_noise * 8.0)))
        ins_x = self.fused_loc.x + self.ins_vel.x * 0.05
        ins_y = self.fused_loc.y + self.ins_vel.y * 0.05
        a = self.alpha * (0.1 if gps_failure else 1.0)
        self.fused_loc = carla.Location(
            x=a * ngx + (1 - a) * ins_x,
            y=a * ngy + (1 - a) * ins_y,
            z=gt_loc.z,
        )
        # 防御性限幅：融合定位不应超出地图范围（避免 NaN/异常漂移污染误差统计）
        self.fused_loc.x = max(-10000.0, min(10000.0, self.fused_loc.x))
        self.fused_loc.y = max(-10000.0, min(10000.0, self.fused_loc.y))
        loc_err = self.fused_loc.distance(gt_loc)

        # 航向融合（互补滤波，与位置同构）：GNSS 观测航向 + INS 航向递推。
        # INS 角速度源用「真值航向差分」（CARLA 的 get_angular_velocity 在此环境返回
        # 不可靠的巨幅值，0 噪声下也会被积分成航向漂移），再叠加陀螺噪声；GNSS 给出带噪观测。
        if self.prev_gt_yaw_deg is None:
            self.fused_yaw_deg = gt_yaw  # 首帧：直接以当前真值航向初始化，避免初始 90° 大误差
            self.prev_gt_yaw_deg = gt_yaw
            gyro_w = 0.0
            ins_yaw_deg = self.fused_yaw_deg
            gnss_yaw_deg = gt_yaw
            yaw_a = self.alpha * (0.1 if gps_failure else 1.0)
        else:
            delta_deg = ((gt_yaw - self.prev_gt_yaw_deg + 180.0) % 360.0) - 180.0  # 本帧航向变化(度)
            self.prev_gt_yaw_deg = gt_yaw
            gyro_w = delta_deg / 0.05 + random.gauss(0, self.ins_noise * 30.0)      # °/s
            ins_yaw_deg = self.fused_yaw_deg + gyro_w * 0.05
            gnss_yaw_deg = gt_yaw + random.gauss(0, self.gnss_noise * 4.0)          # 度
            yaw_a = self.alpha * (0.1 if gps_failure else 1.0)
            # 用「卷绕后的相位差(innovation)」做互补滤波：若直接对绝对角度求加权，
            # 在 ±180° 边界处会把 179.9° 与 -180.1° 平均成 -0.1°（实则同向），
            # 导致融合航向瞬间跳变，纯跟踪 cte 剧增、车辆跑偏（含 0 噪声场景）。
            innov_deg = gnss_yaw_deg - ins_yaw_deg
            innov_deg = (innov_deg + 180.0) % 360.0 - 180.0   # 卷绕到 [-180,180]
            self.fused_yaw_deg = ins_yaw_deg + yaw_a * innov_deg
            self.fused_yaw_deg = (self.fused_yaw_deg + 180.0) % 360.0 - 180.0  # 归一到 [-180,180]
        if not math.isfinite(loc_err):
            loc_err = 0.0

        return LocFrame(
            fused_loc=self.fused_loc,
            fused_yaw_deg=self.fused_yaw_deg,
            ngx=ngx, ngy=ngy,
            gnss_yaw_deg=gnss_yaw_deg,
            gyro_w=gyro_w,
            yaw_a=yaw_a,
            loc_err=loc_err,
        )
