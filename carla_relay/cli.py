"""carla_relay 命令行入口。

服务核心引导壳位于 `smart-course/server/carla_relay_core.py`（已随核心逻辑一起
移入 server/ 目录，保证只分发 server/ 即可运行）；本模块以模块方式加载该引导壳
（保持单一事实源，不复制代码），并把命令行参数原样透传给其 `main()`。实验片段
**命名对齐前端仿真关卡**：localization（定位分析）/
lidar_detection（Lidar 检测）/ semantic_segmentation（语义分割）/
comprehensive_driving（综合驾驶）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# 核心引导壳：server/carla_relay_core.py
# server/carla_relay/cli.py -> parents[1] 即 server/
_LEGACY_CORE_PATH = Path(__file__).resolve().parents[1] / "carla_relay_core.py"

# 加载到 sys.path 时使用的模块名（避免与 carla_relay 包自身冲突）
_LEGACY_MODULE_NAME = "carla_relay_legacy"


def load_legacy_core(path: Path = _LEGACY_CORE_PATH) -> ModuleType:
    """按路径加载核心引导壳模块。

    注意：该文件在 import 时即执行 CARLA 根目录解析、sys.path 注入与
    `import carla` 等模块级副作用。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到核心引导壳: {path}\n"
            "引导壳必须存在（experiments/world_api 命名空间片段由其载入）。"
        )
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载核心引导壳: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    """入口：透传命令行参数给引导壳的 main()。

    引导壳的 main() 内部自行 argparse 解析（--host/--port/--carla-root 等），
    这里不重复定义参数。
    """
    if argv is not None:  # 便于测试与程序化调用
        sys.argv = [sys.argv[0], *argv]
    legacy = load_legacy_core()
    legacy.main()


if __name__ == "__main__":
    main()
