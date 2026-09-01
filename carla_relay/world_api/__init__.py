"""世界查询/管理 API 片段包。

包含：感知查询（障碍物/信号灯）、路径规划、地图（出生点/边界/路网/渲染）、
行人生成、俯瞰视角。均为命名空间片段，机制同 carla_relay.experiments
（详见其 __init__ 文档），不可独立 import。
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent

_ORDER = ["perception", "map_api", "actors_api"]


def load_into(namespace: dict) -> None:
    """按源文件顺序将全部世界 API 片段载入命名空间。"""
    for name in _ORDER:
        path = _DIR / f"{name}.py"
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, namespace)
