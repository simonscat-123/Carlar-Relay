"""语义分割纯函数：类别映射、彩图渲染、占比统计。

语义分割标签解析与映射。
除 label_semantic_classes 传入的 world 参数外不依赖任何运行时状态，
可用 numpy 数组直接进行单元测试。
"""
from __future__ import annotations

import numpy as np

from carla_relay.config import SEMANTIC_PRESETS


def dynamic_class(type_id: str) -> str:
    """根据 CARLA actor.type_id 判断动态参与者类别（CARLA 0.9.16 自带车辆集合）"""
    if not type_id:
        return "car"
    low = type_id.lower()
    if low.startswith("walker."):
        return "pedestrian"
    if any(k in low for k in ("carlacola", "hgv", "firetruck", "ambulance")):
        return "truck"
    if any(k in low for k in ("fusorosa", "bus")):
        return "bus"
    if any(k in low for k in ("kawasaki", "harley", "yamaha", "vespa", "ninja")):
        return "motorcycle"
    if any(k in low for k in ("crossbike", "diamondback", "gazelle", "omafiets", "century")):
        return "bicycle"
    return "car"


def label_semantic_classes(sem_labels, preset, instance=None, world=None):
    """把 CityScapes 语义标签图映射为指定类别预设的 ID 图 (H, W) int32。

    sem_labels: (H, W) CityScapes 标签数组
    preset: 类别预设ID（"7" / "22"）
    instance: (sem_ids, actor_ids) 或 None（7 类不需要实例分割）
    world: CARLA world，22 类动态细分时用于反查 actor.type_id
    """
    cfg = SEMANTIC_PRESETS[preset]
    result = np.full(sem_labels.shape, cfg["default"], dtype=np.int32)

    # 1) 静态标签合并（7 / 22 类均基于语义标签）
    for tid, lid in cfg["static_map"].items():
        result[sem_labels == tid] = lid

    # 2) 22 类动态细分：用实例分割的 actor ID 覆盖动态参与者
    if preset == "22" and instance is not None and world is not None:
        sem_ids, actor_ids = instance
        for aid in np.unique(actor_ids):
            if aid == 0:
                continue
            actor = world.get_actor(int(aid))
            if actor is None:
                continue
            cls = dynamic_class(actor.type_id)
            if cls in cfg["dynamic_map"]:
                result[actor_ids == aid] = cfg["dynamic_map"][cls]
    return result


def colors_from_labels(labeled, preset):
    """类别 ID 图 -> RGB 彩图 (H, W, 3) uint8"""
    colors = SEMANTIC_PRESETS[preset]["colors"]
    h, w = labeled.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lid, color in colors.items():
        rgb[labeled == lid] = color
    return rgb


def ratios_from_labels(labeled, preset):
    """类别 ID 图 -> 各类别像素占比 dict"""
    cfg = SEMANTIC_PRESETS[preset]
    total = labeled.size
    ratios = {}
    for lid, name in cfg["names"].items():
        ratios[name] = round(float(np.count_nonzero(labeled == lid)) / total, 4)
    return ratios
