"""实验10 预测层：障碍物轨迹预测（恒速模型）。

工业界预测层做目标速度/轨迹估计（占据栅格、意图分类、多项式拟合等）。
本实验的定位是教学闭环，当前实现为最简形式：

  - 感知闭环（相机）模式：单帧检测无速度估计，v_s=0（静止假设）；
  - 世界真值模式：v_s 取实测纵向速度（沿参考线投影）；
  - 预测模型：恒速外推 s_pred = s + v_s·dt（规划层碰撞检查逐时刻调用）。

接口按工业界预测层形状定义（extrapolate），后续升级为多帧跟踪 +
中值滤波 / 多项式预测时，上下游（感知帧、规划器）无需改动。
"""
from __future__ import annotations

from typing import Dict


class ObstaclePredictor:
    """恒速（CV）预测器：规划层展开轨迹时逐时刻外推障碍位置。"""

    def __init__(self) -> None:
        # 多帧 track 历史占位：升级多帧跟踪时在此维护 {id: deque(历史观测)}
        self._tracks: Dict = {}

    def extrapolate(self, obs: Dict, dt: float) -> float:
        """障碍 obs 在 dt 秒后的纵向位置（沿参考线）。"""
        return obs["s"] + obs["v_s"] * dt
