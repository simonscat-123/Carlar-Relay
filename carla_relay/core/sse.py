"""SSE 订阅者 hub。

职责：管理 Server-Sent Events 订阅者队列，向所有订阅者投递消息；
队列积压满的订阅者会被清空积压并投放 None 哨兵强制断开
（否则该连接只发 keepalive “假活”，浏览器 EventSource 不会重连，
前端画面从此定格）。

不依赖 Flask / CARLA，可用 queue.Queue 直接单元测试。
"""
from __future__ import annotations

import queue
import threading
from typing import Any, List


class SSEHub:
    """SSE 订阅者管理与消息广播。"""

    def __init__(self) -> None:
        self._subscribers: List["queue.Queue"] = []
        self._lock = threading.Lock()

    def subscribe(self, q: "queue.Queue") -> None:
        """注册一个订阅者队列（每个 SSE 连接一个，maxsize=10）。"""
        with self._lock:
            self._subscribers.append(q)

    def unsubscribe(self, q: "queue.Queue") -> None:
        """移除订阅者（连接断开时调用，不存在时静默忽略）。"""
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def has_subscribers(self) -> bool:
        """是否存在订阅者（空闲时推流线程可借此跳过组包省 CPU）。"""
        with self._lock:
            return bool(self._subscribers)

    def push(self, msg: dict) -> None:
        """向所有订阅者非阻塞投递消息；积压满的订阅者被踢出并断开。"""
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)
                # 积压被踢的连接必须主动断开（清空积压后投放哨兵）：
                # 否则该连接只发 keepalive “假活”，浏览器 EventSource 不会重连，前端画面从此定格
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass


# 进程级默认 hub 实例
hub = SSEHub()
