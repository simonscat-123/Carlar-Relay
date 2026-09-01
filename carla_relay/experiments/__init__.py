"""实验逻辑片段包。

各 .py 文件为命名空间片段：由引导壳经 load_into(globals()) 以 exec
载入同一命名空间执行，与其它片段共享一份全局状态。

片段命名对齐前端仿真关卡定义（src/utils/virtual-simulation-runtime.ts）：
    localization             定位分析（API 实验ID 23）
    lidar_detection          Lidar 检测（API 实验ID 4）
    semantic_segmentation    语义分割（API 实验ID 5）
    comprehensive_driving_*  综合驾驶（API 实验ID 10，闭环自动驾驶）
其余为历史实验（basic_control / gnss_imu / ins_fusion / lidar_camera_projection /
route_planning / path_following / target_navigation），前端已不再暴露。
URL 中的数字实验 ID 为 API 契约，保持不变。

片段特性（重要）：
- 模块级语句（全局变量 / 锁 / @app.route 注册）落入载体命名空间，
  与其它片段共享同一份全局状态；
- 不可作为独立模块 import（命名空间中无 app / world 等名称）；
- 后续在真机 CARLA 验证基础上，可逐片段重构为 ExperimentRunner + AppContext。

加载顺序（源文件顺序，仅影响可读性；函数间引用在调用期解析）：
common → localization → lidar_detection → basic_control → gnss_imu →
ins_fusion → semantic_segmentation → lidar_camera_projection → route_planning →
controllers → path_following → target_navigation

综合驾驶片段单独经 load_comprehensive_driving_into() 载入：其模块级语句
在世界 API 之后执行，由引导壳按「其余实验 → world_api → 综合驾驶」顺序
调用三个 loader，保持模块级执行顺序。
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent

_ORDER = [
    "common",
    "localization",
    "lidar_detection",
    "basic_control",
    "gnss_imu",
    "ins_fusion",
    "semantic_segmentation",
    "lidar_camera_projection",
    "route_planning",
    "controllers",
    "path_following",
    "target_navigation",
]

_ORDER_COMPREHENSIVE_DRIVING = [
    "comprehensive_driving_perception",
    "comprehensive_driving_run",
]


def load_into(namespace: dict) -> None:
    """按源文件顺序将定位/检测/分割及历史实验片段载入 legacy 命名空间。"""
    for name in _ORDER:
        path = _DIR / f"{name}.py"
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, namespace)


def load_comprehensive_driving_into(namespace: dict) -> None:
    """将综合驾驶（闭环自动驾驶）片段载入 legacy 命名空间（须在 world_api 之后调用）。"""
    for name in _ORDER_COMPREHENSIVE_DRIVING:
        path = _DIR / f"{name}.py"
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, namespace)
