"""实验10 装配上下文：跨层共享的运行时状态与平台服务挂载点。

Exp10Context 在实验装配期（路线规划完成、传感器挂载完毕后）由编排层
（本包 run.py）构造，承载：

  - 平台服务：日志 / SSE 推送 / 传感器帧缓存 / actor 管理——均为引导壳
    命名空间中的可变对象引用（dict/set 传引用，层内增删即时生效）；
  - 装配产物：主车 / 传感器 rig / 参考线 / 各层实例；
  - 运行参数快照（运行中可变参数仍由 _EXP10_CTRL 全局 + 各层 update 维持）。

各算法层只通过构造函数显式接收自己需要的依赖（不持有整个 ctx），
保证层间数据仍经帧契约显式流动；ctx 用于装配期接线和调试时整体检视。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set


class Exp10Context:
    """实验10 运行时上下文（属性袋：装配期填充，主循环期只读）。"""

    def __init__(self) -> None:
        # ── 平台服务（引导壳命名空间注入，可变对象传引用）──
        self.log: Optional[Callable[[str], None]] = None       # _exp_log
        self.push: Optional[Callable[[dict], None]] = None     # _push_to_sse
        self.sensor_frames: Optional[Dict[int, bytes]] = None  # _sensor_frames
        self.sensor_frame_num: Optional[Dict[int, int]] = None
        self.instance_raw: Optional[Dict[int, bytes]] = None   # _instance_raw
        self.semantic_raw: Optional[Dict[int, bytes]] = None   # _semantic_raw
        self.sensor_refs: Optional[Dict[int, Any]] = None      # _sensor_refs
        self.managed_actors: Optional[Set[int]] = None         # _managed_actors
        self.lock: Any = None                                  # _lock（actor 管理锁）
        self.sensor_callback: Optional[Callable] = None        # _sensor_callback
        self.dynamic_class: Optional[Callable[[str], str]] = None  # 语义类别中文名
        self.tl_state_map: Optional[Dict[int, str]] = None     # CARLA 灯态 → green/yellow/red

        # ── 装配产物 ──
        self.client: Any = None
        self.world: Any = None
        self.vehicle: Any = None
        self.carla_map: Any = None
        self.sensors: Any = None          # SensorRig 产物（cam/inst/sem/bird/lidar/gnss/imu/col）
        self.reference: Any = None        # ReferenceLine（Frenet + 可行驶域 + 停止线绑定）
        self.tl_bindings: list = []       # 信号灯绑定 [{tl, s_stop}]

        # ── 各层实例（装配期末尾挂载）──
        self.rig: Any = None              # 驱动层：传感器装配/清理 + 碰撞记录
        self.localizer: Any = None        # 定位层
        self.perceiver: Any = None        # 感知层
        self.predictor: Any = None        # 预测层
        self.planner: Any = None          # 规划层
        self.controller: Any = None       # 控制层

        # ── 运行参数快照（装配期确定；运行中可变项走 _EXP10_CTRL）──
        self.params: Dict[str, Any] = {}

    def __repr__(self) -> str:  # 调试检视用
        vid = self.vehicle.id if self.vehicle is not None else None
        n_wp = len(self.reference.route_wp) if self.reference is not None else 0
        return (f"<Exp10Context vehicle={vid} waypoints={n_wp} "
                f"layers={'/'.join(n for n in ('localizer', 'perceiver', 'predictor',
                                               'planner', 'controller')
                                 if getattr(self, n) is not None)}>")
