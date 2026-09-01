"""core.sensors 帧序列化单元测试。

用鸭子类型伪造 CARLA 传感器数据对象，无需真实 CARLA。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_relay.core.sensors import serialize_frame


class FakeImage:  # camera / depth 通用：BGRA raw + 尺寸 + 帧号
    def __init__(self, h=2, w=2, frame=7):
        self.height, self.width, self.frame = h, w, frame
        self.raw_data = bytes(np.zeros((h * w * 4), dtype=np.uint8))


class FakeLidar:
    frame, timestamp = 9, 123.4
    raw_data = bytes(np.arange(16, dtype=np.float32))  # 4 个点


class FakeGnss:
    frame, timestamp = 10, 123.5
    latitude, longitude, altitude = 31.1, 121.2, 5.0


class FakeVec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeImu:
    frame, timestamp = 11, 123.6
    accelerometer = FakeVec(0.1, 0.2, 9.8)
    gyroscope = FakeVec(0.0, 0.0, 0.0)
    compass = 90.0


class FakeDet:
    altitude, azimuth, depth, velocity = 0.1, 1.2, 30.5, 2.0


class FakeRadar:
    frame, timestamp = 12, 123.7
    _dets = [FakeDet(), FakeDet()]

    def __iter__(self):
        return iter(self._dets)


class FakeRaw:  # semantic / instance
    raw_data = b"\x01\x02\x03\x04"


def test_camera():
    payload, fnum = serialize_frame("camera", FakeImage(frame=7))
    assert isinstance(payload, bytes) and payload[:2] == b"\xff\xd8"  # JPEG 魔数
    assert fnum == 7


def test_lidar():
    payload, fnum = serialize_frame("lidar", FakeLidar())
    d = json.loads(payload)
    assert d["frame"] == 9 and d["total_points"] == 4 and len(d["points"]) == 4
    assert fnum is None


def test_gnss():
    payload, fnum = serialize_frame("gnss", FakeGnss())
    d = json.loads(payload)
    assert d["latitude"] == 31.1 and d["longitude"] == 121.2
    assert fnum is None


def test_imu():
    payload, _ = serialize_frame("imu", FakeImu())
    d = json.loads(payload)
    assert d["accelerometer"]["z"] == 9.8 and d["compass"] == 90.0


def test_depth():
    payload, fnum = serialize_frame("depth", FakeImage(frame=3))
    assert payload[:2] == b"\xff\xd8" and fnum == 3


def test_radar():
    payload, _ = serialize_frame("radar", FakeRadar())
    d = json.loads(payload)
    assert d["total"] == 2 and len(d["detections"]) == 2
    assert d["detections"][0]["depth"] == 30.5


def test_semantic_instance_passthrough():
    payload, fnum = serialize_frame("semantic", FakeRaw())
    assert payload == b"\x01\x02\x03\x04" and fnum is None
    payload, _ = serialize_frame("instance", FakeRaw())
    assert payload == b"\x01\x02\x03\x04"


def test_unknown_dtype():
    try:
        serialize_frame("xxx", FakeRaw())
        raise AssertionError("应抛出 ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("全部通过")
