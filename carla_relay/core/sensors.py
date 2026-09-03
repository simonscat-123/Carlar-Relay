"""传感器帧序列化。

把 CARLA 各类传感器回调数据统一序列化为字节载荷：
- camera  → JPEG（q70，教学场景画质足够且显著降低编码耗时与 SSE 包体）
- lidar   → JSON（截断至 500 点 + 总点数）
- gnss    → JSON（纬经高）
- imu     → JSON（加计/陀螺/罗盘）
- depth   → 灰度 JPEG（近白远黑，50m 量程）
- radar   → JSON（检测点列表，截断至 200）
- semantic/instance → 原始 BGRA（分级可视化由实验主循环统一解码）

data 参数为鸭子类型（CARLA 传感器数据对象），无需真实 CARLA 即可单测。
返回 (payload, frame_num)：frame_num 仅图像类传感器有意义，其余为 None。
"""
from __future__ import annotations

import io
import json
from typing import Optional, Tuple

import numpy as np
import PIL.Image


def _camera_to_jpeg(data) -> bytes:
    """BGRA 原始帧 → JPEG（quality=70）。"""
    arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
    # 必须显式 copy：frombuffer 产物是只读视图，[::-1] 又引入负步长，
    # 直接交给 fromarray 会按步长算内存偏移越界 → 访问违规崩溃（多路高分辨率相机并发时必现）。
    rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA → RGB，连续可写自有内存
    img = PIL.Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)  # q70 显著降低编码耗时与 SSE 包体，教学场景画质足够
    return buf.getvalue()


def _depth_to_jpeg(data) -> bytes:
    """深度相机：BGRA → 对数灰度 JPEG（近白远黑，0.1~100m 量程）。

    官方文档：3 通道编码 24-bit 距离（less→more: R→G→B）；
    raw_data 字节序为 B,G,R,A，故 R 通道 = arr[:,:,2]。
    使用对数刻度提升近处物体的区分度（CARLA carla.ColorConverter.LogarithmicDepth 口径）。"""
    arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((data.height, data.width, 4))
    r = arr[:, :, 2].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 0].astype(np.float32)
    depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0) * 1000.0  # 米
    d = np.clip(depth, 0.1, 100.0)
    log_range = np.log1p(100.0) - np.log1p(0.1)
    gray = np.clip(255.0 * (1.0 - (np.log1p(d) - np.log1p(0.1)) / log_range),
                   0, 255).astype(np.uint8)
    img = PIL.Image.fromarray(np.stack([gray, gray, gray], axis=-1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def serialize_frame(dtype: str, data) -> Tuple[bytes, Optional[int]]:
    """按传感器类型序列化回调数据，返回 (payload, frame_num)。

    frame_num：camera/depth 返回 data.frame；其余（含 semantic/instance）为 None，
    由调用方决定是否写入帧号缓存。
    """
    if dtype == "camera":
        return _camera_to_jpeg(data), data.frame
    if dtype == "lidar":
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
        payload = json.dumps({
            "frame": data.frame,
            "timestamp": data.timestamp,
            "points": points[:500].tolist(),
            "total_points": len(points),
        }).encode()
        return payload, None
    if dtype == "gnss":
        payload = json.dumps({
            "frame": data.frame,
            "timestamp": data.timestamp,
            "latitude": data.latitude,
            "longitude": data.longitude,
            "altitude": data.altitude,
        }).encode()
        return payload, None
    if dtype == "imu":
        acc = data.accelerometer
        gyro = data.gyroscope
        comp = data.compass
        payload = json.dumps({
            "frame": data.frame,
            "timestamp": data.timestamp,
            "accelerometer": {"x": acc.x, "y": acc.y, "z": acc.z},
            "gyroscope": {"x": gyro.x, "y": gyro.y, "z": gyro.z},
            "compass": comp,
        }).encode()
        return payload, None
    if dtype == "depth":
        return _depth_to_jpeg(data), data.frame
    if dtype == "radar":
        # 毫米波雷达：序列化检测点列表
        detections = []
        for det in data:
            detections.append({
                "altitude": round(det.altitude, 3),
                "azimuth": round(det.azimuth, 3),
                "depth": round(det.depth, 2),
                "velocity": round(det.velocity, 2),
            })
        payload = json.dumps({
            "frame": data.frame,
            "timestamp": data.timestamp,
            "detections": detections[:200],
            "total": len(detections),
        }).encode()
        return payload, None
    if dtype in ("semantic", "instance"):
        # 仅缓存原始 BGRA；分级可视化由实验主循环统一渲染，
        # 避免在此处写原始 CityScapes 彩图覆盖分级结果。
        return bytes(data.raw_data), None
    raise ValueError(f"未知传感器类型: {dtype}")
