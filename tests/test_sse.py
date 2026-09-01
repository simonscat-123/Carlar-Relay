"""core.sse 订阅者 hub 单元测试。"""
from __future__ import annotations

import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_relay.core.sse import SSEHub


def test_subscribe_push_unsubscribe():
    hub = SSEHub()
    q = queue.Queue(maxsize=10)
    hub.subscribe(q)
    assert hub.has_subscribers()
    hub.push({"ts": 1.0})
    assert q.get_nowait() == {"ts": 1.0}
    hub.unsubscribe(q)
    assert not hub.has_subscribers()
    hub.unsubscribe(q)  # 重复退订应静默


def test_push_evicts_full_subscriber():
    """积压满的订阅者被踢出并收到 None 哨兵（强制断连，防前端画面定格）"""
    hub = SSEHub()
    full = queue.Queue(maxsize=1)
    normal = queue.Queue(maxsize=10)
    hub.subscribe(full)
    hub.subscribe(normal)
    full.put_nowait({"ts": 0})  # 占满
    hub.push({"ts": 1.0})
    # full 应被清空积压后收到哨兵
    assert full.get_nowait() is None
    # normal 正常收到消息
    assert normal.get_nowait() == {"ts": 1.0}
    # full 已被移出订阅者，normal 仍在
    hub.push({"ts": 2.0})
    assert normal.get_nowait() == {"ts": 2.0}
    try:
        full.get_nowait()
        raise AssertionError("full 已被踢出，不应再收到消息")
    except queue.Empty:
        pass


def test_push_without_subscribers():
    hub = SSEHub()
    hub.push({"ts": 1.0})  # 不应抛异常


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("全部通过")
