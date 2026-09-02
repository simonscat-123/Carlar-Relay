"""实验逻辑片段包。

目录结构（实验通用 vs 实验专属）：

    common/                     实验通用逻辑（跨实验共享）
        state.py                共享状态：manifest 加载 / 日志 / 状态路由 / 残留清理
        controllers.py          共用控制器：Pure Pursuit + PID（实验 8/9/23 等）
    <实验名>/run.py             各实验专属命名空间片段
    comprehensive_driving/      综合驾驶（实验10）：分层真实模块 + run.py 薄编排片段

各片段 .py 文件为命名空间片段：由引导壳经 load_into(globals()) 以 exec
载入同一命名空间执行，与其它片段共享一份全局状态。

实验文件夹对齐前端仿真关卡定义（src/utils/virtual-simulation-runtime.ts）：
    localization             定位分析（API 实验ID 23）
    lidar_detection          Lidar 检测（API 实验ID 4）
    semantic_segmentation    语义分割（API 实验ID 5）
    comprehensive_driving    综合驾驶（API 实验ID 10，闭环自动驾驶）
其余为历史实验（basic_control / gnss_imu / ins_fusion / lidar_camera_projection /
route_planning / path_following / target_navigation），前端已不再暴露。
URL 中的数字实验 ID 为 API 契约，保持不变。

片段特性（重要）：
- 模块级语句（全局变量 / 锁 / @app.route 注册）落入载体命名空间，
  与其它片段共享同一份全局状态；
- 除 comprehensive_driving/ 下的分层模块（真实模块，可独立 import）外，
  其余片段不可作为独立模块 import（命名空间中无 app / world 等名称）；
- 后续在真机 CARLA 验证基础上，可逐片段重构为 ExperimentRunner + AppContext。

加载顺序（与原扁平结构完全一致，仅影响可读性；函数间引用在调用期解析）：
common/state → localization → lidar_detection → basic_control → gnss_imu →
ins_fusion → semantic_segmentation → lidar_camera_projection → route_planning →
common/controllers → path_following → target_navigation

综合驾驶片段单独经 load_comprehensive_driving_into() 载入：其模块级语句
在世界 API 之后执行，由引导壳按「其余实验 → world_api → 综合驾驶」顺序
调用三个 loader，保持模块级执行顺序。
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent

_ORDER = [
    "common/state",
    "localization/run",
    "lidar_detection/run",
    "basic_control/run",
    "gnss_imu/run",
    "ins_fusion/run",
    "semantic_segmentation/run",
    "lidar_camera_projection/run",
    "route_planning/run",
    "common/controllers",
    "path_following/run",
    "target_navigation/run",
]

# 综合驾驶：仅 run.py 薄编排片段以命名空间片段载入（_EXP10_* 全局状态 +
# @app.route 注册）；其余分层模块（frames/context/.../actors）均为真实模块，
# 由 run 片段经 import 引入，不在此列。
_ORDER_COMPREHENSIVE_DRIVING = [
    "comprehensive_driving/run",
]


def load_into(namespace: dict) -> None:
    """按源文件顺序将通用逻辑与各实验片段载入 legacy 命名空间。"""
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
