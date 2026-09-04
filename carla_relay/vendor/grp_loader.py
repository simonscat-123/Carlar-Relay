"""GlobalRoutePlanner 兼容加载器。

背景：本项目为官方 `global_route_planner.GlobalRoutePlanner` 增加了
`algorithm`/`weight_fn` 参数（放在 flywheel `carla_relay/vendor/global_route_planner.py`）。
但官方这个文件在 CARLA 安装目录下，不在本 server 仓库内，无法随代码分发。
为兼容不同宿主机（对方可能用官方原版，其 __init__ 不接受 algorithm/weight_fn），
本模块提供统一的构造入口：

  1. 优先按路径加载本仓库内 `vendor/global_route_planner.py`（修改版，支持扩展参数）；
  2. 若仓库内副本缺失，回退到 `agents.navigation.global_route_planner` 官方实现；
  3. 无论用哪个实现，都先探测 __init__ 签名，仅在支持 algorithm/weight_fn 时再传这两参数，
     否则退化为 `GlobalRoutePlanner(map, sampling_res)` 的两参调用，保证不报
     `unexpected keyword argument`。

实验代码统一改为：
    from carla_relay.vendor.grp_loader import make_route_planner
    grp = make_route_planner(carla_map, sampling_res,
                             algorithm=route_algorithm, weight_fn=weight_fn)

注意：仓库内副本仍依赖官方 `agents.navigation.local_planner.RoadOption`
（RoadOption 为稳定枚举，官方/修改版一致），因此仍需 CARLA PythonAPI 在 sys.path 上。
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

# 仓库内修改版副本：与 __file__ 同目录下的 global_route_planner.py
_VENDOR_MODULE_PATH = Path(__file__).resolve().with_name("global_route_planner.py")

# 加载进 sys.modules 时使用的模块名（独立命名，避免覆盖官方 agents.navigation 包）
_VENDOR_MODULE_NAME = "carla_relay_vendor_global_route_planner"


def _load_vendored_class():
    """按路径加载仓库内的修改版 GlobalRoutePlanner（不依赖宿主机 CARLA 是否被改动）。"""
    if not _VENDOR_MODULE_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(_VENDOR_MODULE_NAME, _VENDOR_MODULE_PATH)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "GlobalRoutePlanner", None)


def _load_official_class():
    """回退：导入官方 agents.navigation 实现（__init__ 不支持 algorithm/weight_fn）。"""
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner as _Cls
    except Exception:  # 官方导入失败时置空，交由 make_route_planner 报错
        _Cls = None
    return _Cls


def get_global_route_planner_class():
    """返回可用的 GlobalRoutePlanner 类（优先仓库内修改版，其次官方）。"""
    cls = _load_vendored_class()
    if cls is not None:
        return cls
    return _load_official_class()


def get_grp_source() -> str:
    """返回实际使用的实现来源，用于向前端提示“是否退化”。

    - "vendored"：仓库内修改版（支持 algorithm/weight_fn 扩展参数）
    - "official"：官方原版（不支持扩展参数，已退化为两参构造）
    - "none"    ：两者都不可用
    """
    if _load_vendored_class() is not None:
        return "vendored"
    if _load_official_class() is not None:
        return "official"
    return "none"


def _supports_extended_params(cls) -> bool:
    """探测 __init__ 是否支持 algorithm 与 weight_fn 关键字。"""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "algorithm" in params and "weight_fn" in params


def make_route_planner(carla_map, sampling_res, algorithm="astar", weight_fn=None):
    """构造 GlobalRoutePlanner，兼容仓库内修改版与官方原版两种 __init__ 签名。

    - 引入官方原版（宿主机未被我们的修改同步时）：自动退化为
      `GlobalRoutePlanner(carla_map, sampling_res)`，不传 algorithm/weight_fn，
      从而避免 `unexpected keyword argument 'algorithm'`。代价是丢失算法选择与三类惩罚，
      但能正常运行。
    - 仓库内副本存在且支持扩展参数时：完整启用算法/权重功能。
    """
    cls = get_global_route_planner_class()
    if cls is None:
        raise RuntimeError(
            "无法导入 GlobalRoutePlanner：仓库内 vendor/global_route_planner.py 缺失，"
            "且 agents.navigation.global_route_planner 也不可用（请确认 CARLA PythonAPI 已注入 sys.path）。"
        )
    if _supports_extended_params(cls):
        return cls(carla_map, sampling_res, algorithm=algorithm, weight_fn=weight_fn)
    # 官方原版：只传两参，忽略算法/权重扩展
    return cls(carla_map, sampling_res)