"""perception.semantics 纯函数单元测试。

不依赖 CARLA / Flask，仅用 numpy 数组验证映射逻辑。
运行：在 server/ 目录下执行 `python -m pytest tests/ -q`
或直接 `python tests/test_semantics.py`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_relay.config import FRAME_INTERVAL, SEMANTIC_PRESETS
from carla_relay.perception.semantics import (
    colors_from_labels,
    dynamic_class,
    label_semantic_classes,
    ratios_from_labels,
)


def test_frame_interval():
    assert FRAME_INTERVAL == 0.05


def test_dynamic_class():
    assert dynamic_class("") == "car"
    assert dynamic_class("walker.pedestrian.0001") == "pedestrian"
    assert dynamic_class("vehicle.carlacola") == "truck"
    assert dynamic_class("vehicle.volkswagen.bus") == "bus"
    assert dynamic_class("vehicle.kawasaki.ninja") == "motorcycle"
    assert dynamic_class("vehicle.diamondback.century") == "bicycle"
    assert dynamic_class("vehicle.tesla.model3") == "car"  # 默认兜底


def test_label_semantic_classes_7():
    # 2x2 标签图：道路(1)、车辆(14)、行人(12)、未标注(0)
    sem = np.array([[1, 14], [12, 0]], dtype=np.int32)
    labeled = label_semantic_classes(sem, "7")
    cfg = SEMANTIC_PRESETS["7"]
    assert labeled.dtype == np.int32
    assert labeled[0, 0] == cfg["static_map"][1]
    assert labeled[0, 1] == cfg["static_map"][14]
    assert labeled[1, 0] == cfg["static_map"][12]
    assert labeled[1, 1] == cfg["default"]


def test_label_semantic_classes_22_static():
    # 22 类无实例分割时退化为静态映射：道路(1)->6、天空(11)->5
    sem = np.array([[1, 11]], dtype=np.int32)
    labeled = label_semantic_classes(sem, "22")
    assert labeled[0, 0] == SEMANTIC_PRESETS["22"]["static_map"][1]
    assert labeled[0, 1] == SEMANTIC_PRESETS["22"]["static_map"][11]


def test_colors_from_labels():
    labeled = np.array([[0, 1], [3, 4]], dtype=np.int32)
    rgb = colors_from_labels(labeled, "7")
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (0, 0, 0)
    assert tuple(rgb[0, 1]) == (128, 64, 128)


def test_ratios_from_labels():
    labeled = np.array([[1, 1], [0, 0]], dtype=np.int32)
    ratios = ratios_from_labels(labeled, "7")
    assert ratios["drivable"] == 0.5
    assert ratios["background"] == 0.5
    # 全部类别占比之和应 <= 1
    assert sum(ratios.values()) <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("全部通过")
