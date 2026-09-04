"""实验10 层间帧数据契约：每 tick 在各层之间流动的数据结构。

工业界分层架构（Apollo/Autoware 风格）中，层与层之间通过明确定义的消息传递；
本模块即实验10 各层的「消息定义」。任何一层的输入输出都是可打印的数据对象
——排障时可以 dump 任意一层的输入输出做离线复现，不用翻主循环闭包。

数据流（每 tick，由本包 run.py 编排）：

    SensorRig(驱动) → Localizer(定位) → Perceiver(感知) →
    ObstaclePredictor(预测) → TrajectoryPlanner(规划) → VehicleController(控制)

持久状态（滤波器积分项、转向延迟队列、决策事件检测等）不放在帧里，
由各层类实例自行持有（见各层模块）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LocFrame:
    """定位层输出：GNSS+INS 互补滤波融合位姿。"""
    fused_loc: Any            # carla.Location 融合位置（带噪，驱动控制与可视化）
    fused_yaw_deg: float      # 融合航向（度）
    ngx: float                # GNSS 观测位置（带噪，SSE 展示用）
    ngy: float
    gnss_yaw_deg: float       # GNSS 观测航向（度，诊断日志用）
    gyro_w: float             # 陀螺角速度（°/s，诊断日志用）
    yaw_a: float              # 航向互补滤波权重（诊断日志用）
    alpha: float              # 位置互补滤波权重 alpha（GNSS 权重，自适应，SSE 展示用）
    loc_err: float            # 融合位置 vs 真值距离（m，误差评估用）


@dataclass
class PercFrame:
    """感知层输出：统一障碍列表（Frenet 坐标）+ 展示用原始列表。"""
    obstacles: List[Dict]     # [{s,l,half_len,half_w,v_s,lane,cls,id}] 规划/预测消费
    perceived: List[Dict]     # bbox 相机单目感知原始输出（感知闭环模式）
    obs_list: List[Dict]      # SSE obstacles 字段（前 5 个推给前端）
    bbox_cands: List[Tuple]   # [(actor, dist)] 真值模式 bbox 渲染候选


@dataclass
class TlFrame:
    """感知层输出：信号灯状态（停止线绑定 + s 判定）。"""
    state: str                       # green/yellow/red
    dist: float                      # 距停止线距离（m；999=无有效灯）
    red_stop_s: Optional[float]      # 红灯停止线弧长（None=当前非红灯）


@dataclass
class PlanOutput:
    """规划层输出：最优轨迹 + 控制期望 + 可视化状态。"""
    best: Optional[Dict]                  # 最优候选 {cost,l_t,mode,s_stop,a_long,v_cap,samples}
    plan_traj: List[Tuple[float, float]]  # 规划轨迹（世界坐标点列，控制前视+鸟瞰参考线）
    plan_end_l: float                     # 轨迹末端横向偏移（鸟瞰参考线续接用）
    intent_l: float                       # 决策意图横向目标
    borrow_l: Optional[float]             # 激进借道目标偏移（None=未借道）
    desired: float                        # 期望车速（m/s）
    a_need: float                         # 期望减速度（m/s²）
    avoiding: bool                        # 是否换道绕障
    avoid_side: int                       # 1=左 / -1=右 / 0=无
    avoid_lat_target: float               # 换道目标横向偏移
    avoid_dest_lane: Optional[Tuple]      # 换道目标车道 (road_id, lane_id)
    nudge: bool                           # 是否贴边绕行（空隙穿越，shift 点剖面）
    nudge_gap: float                      # 贴边空隙宽度（m；0=未绕行）
    fsm_state: str                        # CRUISE/APPROACH_RED/FOLLOW/LANE_CHANGE/NUDGE
    front_obstacle: float                 # 本道走廊最近障碍后缘距离（inf=无）
    front_obs_src: str                    # 最近障碍类别（SSE 展示用）
    n_cands: int                          # 候选数（FRM 日志用）
    blocked: bool                         # 本道走廊被占（FRM 日志用）
    rej_bounds: int                       # 拒绝统计（FRM 日志用）
    rej_red: int
    rej_coll: int
    rej_dir: int


@dataclass
class CtrlOutput:
    """控制层输出：方向盘 / 油门 / 刹车 + 跟踪诊断量。"""
    steer: float
    throttle: float
    brake: float
    raw_steer: float      # 转向延迟前的指令（诊断日志用）
    cte: float            # 横向跟踪误差（m）
    look_dist: float      # 前视点距离（m）
    hdng_alpha: float     # 前视点方位角差（rad）
    wp_idx: int           # 校正后的路线进度索引（下一 tick 复用）
