"""语义分割（实验5）基于 BEV 占据栅格的规划决策。

只读构建好的占据栅格，在车体系前向走廊内检索占用单元，给出决策与一条
可行参考轨迹（车体系，单位米）。局部自由空间对齐，不依赖全局地图/Frenet，
保持实验5 自洽；不接管控制（是否用于主车由 run.py 决定）。

坐标约定：与 bev 一致——前向 fwd 为正前方(m)，横向 lat 左正右负(m)。

决策语义：
  - CRUISE：走廊通畅，匀速直行；
  - SLOW  ：前方远处有占用，减速接近；
  - AVOID ：前方占用但左/右一方有可通行绕带 → 生成横向偏移轨迹；
  - STOP  ：占用已近且两侧均不可通行，停车等待。
"""
from __future__ import annotations

import numpy as np

from carla_relay.experiments.semantic_segmentation.bev import FREE, STATIC, UNK, VEH, WALK

OCC = (STATIC, VEH, WALK)


class GridPlanner:
    """基于栅格的轻量规划决策（车体系，近零状态）。"""

    def __init__(self, *, lane_half=3.5, lookahead=30.0, ego_half_w=1.0,
                 avoid_offset=4.5, slow_dist=25.0, stop_dist=8.0,
                 patch=14.0):
        self.lane_half = lane_half
        self.lookahead = lookahead
        self.ego_half_w = ego_half_w
        self.avoid_offset = avoid_offset
        self.slow_dist = slow_dist
        self.stop_dist = stop_dist
        self.patch = patch

    def decide(self, cells, *, res):
        """cells: (G,G) int8 占据栅格；res: 栅格分辨率(m)。
        返回决策 dict：{state,label,target_lat,block_m,reason,traj:[{x,y},...]}。"""
        G = cells.shape[0]
        center = G // 2
        max_a0 = min(G, center + int(self.lookahead / res))
        lane_steps = max(1, int(self.lane_half / res))

        # 1) 前向走廊（|lat|<lane_half）内检索最近占用
        corridor = cells[center:max_a0, center - lane_steps:center + lane_steps + 1]
        blocked_rows = np.nonzero(np.isin(corridor, OCC).any(axis=1))[0]
        block_m = None
        if blocked_rows.size:
            block_m = blocked_rows[0] * res                # 首个占用的前向距离

        # 2) 侧向可通行检查（仅在存在占用时评估）
        left_free = right_free = False
        if block_m is not None:
            # 检查窗口从首个占用纵向位置起，向前覆盖 patch 距离（行号 = center + fwd/res）
            a0s = max(center + 1, min(G - 1, center + int(block_m / res)))
            a0e = min(G, a0s + int(self.patch / res))
            stab = max(1, int(self.ego_half_w * 2 / res))
            left_free = self._lane_slab_free(cells, center, self.avoid_offset, res, a0s, a0e, stab)
            right_free = self._lane_slab_free(cells, center, -self.avoid_offset, res, a0s, a0e, stab)

        # 3) 决策
        if block_m is None or block_m > self.slow_dist:
            state, label, tgt = "CRUISE", "匀速直行", 0.0
        elif left_free or right_free:
            side = -1 if right_free else 1                 # 右为空→右绕；否则左绕
            state, label, tgt = "AVOID", ("右绕" if side < 0 else "左绕"), side * self.avoid_offset
        elif block_m > self.stop_dist:
            state, label, tgt = "SLOW", "减速接近", 0.0
        else:
            state, label, tgt = "STOP", "停车等待", 0.0

        traj = self._trajectory(state, tgt, block_m)
        reason = ("走廊通畅" if block_m is None
                  else f"前方 {block_m:.0f}m 占用，" +
                       ("右侧可绕行" if right_free else "左侧可绕行" if left_free else "两侧不可通"))
        return {
            "state": state, "label": label, "target_lat": round(tgt, 2),
            "block_m": round(block_m, 1) if block_m is not None else None,
            "reason": reason, "traj": traj,
        }

    def _lane_slab_free(self, cells, center, offset, res, a0s, a0e, stab):
        """检查车体系横向偏移 offset(m) 处一条与自车等宽的纵向带是否可通行。"""
        a1c = center + int(offset / res)
        lo, hi = max(0, a1c - stab), min(cells.shape[1], a1c + stab + 1)
        if lo >= hi or a0e <= a0s:
            return False
        slab = cells[a0s:a0e, lo:hi]
        if np.isin(slab, OCC).any():
            return False
        return bool(np.count_nonzero(slab == FREE))        # 需存在可行驶单元才算可通

    def _trajectory(self, state, tgt_lat, block_m):
        traj = []
        step = 2.0
        x = 0.0
        while x <= self.lookahead:
            y = 0.0
            if state == "AVOID" and block_m is not None:
                start = max(0.0, block_m - 14.0)
                span = 10.0
                k = min(1.0, max(0.0, (x - start) / span))
                y = tgt_lat * (k * k * (3 - 2 * k))        # smoothstep 渐变
            traj.append({"x": round(x, 1), "y": round(y, 2)})
            x += step
        return traj