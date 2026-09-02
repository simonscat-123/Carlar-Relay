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

    def predict(self, obs: Dict, *, dt: float = 0.25, horizon: float = 4.0) -> Dict:
        """对单个障碍物生成预测摘要（保持单恒速轨迹）。

        恒速模型为确定性单轨迹，故把时间网格上的采样点作为候选轨迹点集合；
        概率最大的轨迹即该名义轨迹本身（prob=1.0）。
        返回（纯数据，不依赖日志）：调用方负责落日志。
        """
        n = max(1, int(round(horizon / dt)))
        points = [(k * dt, self.extrapolate(obs, k * dt)) for k in range(1, n + 1)]
        s0 = obs["s"]
        vs = obs["v_s"]
        s_end = points[-1][1] if points else s0
        best = {
            "prob": 1.0,                 # 概率最大（也是唯一）的轨迹
            "t": points[-1][0] if points else 0.0,
            "s_start": s0,
            "s_end": s_end,
            "dist": s_end - s0,
            "v_s": vs,
        }
        return {"points": points, "n_cand": len(points), "best": best}
