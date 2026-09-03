"""运行时上下文（AppContext）。

目标：收拢散落在核心引导壳中的模块级可变状态（world / client /
traffic_manager / 托管 actor / 传感器帧缓存 / SSE 订阅队列等），消除隐式
全局耦合，使核心层可测试。

当前仅定义结构与锁策略；运行时状态仍由引导壳持有。核心层将逐步切换为
「AppContext 实例经 app.extensions 注入」。
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppContext:
    """relay 进程级运行时状态。

    约定（核心层起生效）：
    - 通过 ``app.extensions["ctx"]`` 注入，路由/服务层不得直接 import 单例；
    - ``lock`` 保护托管 actor 与自动驾驶状态；
    - ``sse_lock`` 保护 SSE 订阅者列表；
    - ``sensor_frames`` 等帧缓存由各传感器回调单写多读，写读均需 ``lock``。
    """

    # CARLA 连接
    client: Optional[Any] = None            # carla.Client
    world: Optional[Any] = None             # carla.World
    traffic_manager: Optional[Any] = None   # carla.TrafficManager

    # 托管资源（由 relay 生成、退出时统一销毁）
    managed_actors: set = field(default_factory=set)
    autopilot_state: Dict[int, bool] = field(default_factory=dict)  # vehicle_id -> autopilot

    # 传感器最新帧缓存: sensor_id -> bytes（JPEG 或 JSON 字节）
    sensor_frames: Dict[int, bytes] = field(default_factory=dict)
    sensor_frame_num: Dict[int, int] = field(default_factory=dict)  # sid -> CARLA 帧号
    sensor_dtype: Dict[int, str] = field(default_factory=dict)      # "camera"|"lidar"|...
    semantic_raw: Dict[int, bytes] = field(default_factory=dict)    # 原始 BGRA（占比统计）
    instance_raw: Dict[int, bytes] = field(default_factory=dict)    # 原始 BGRA（22 类细分）

    # 同步模式切换前的旧 settings（用于恢复）
    old_settings: Any = None

    # SSE 订阅者队列
    sse_subscribers: List["queue.Queue"] = field(default_factory=list)

    # 锁
    lock: threading.Lock = field(default_factory=threading.Lock)
    sse_lock: threading.Lock = field(default_factory=threading.Lock)
