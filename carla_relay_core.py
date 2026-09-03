#!/usr/bin/env python3
"""CARLA HTTP + SSE 中继网关

在 Windows 机器上运行，将 REST API 翻译为 CARLA Python API 调用，
并通过 SSE (Server-Sent Events) 实时推送传感器数据流到前端。

启动方式一（纯 API 模式，前端单独托管）：
  pip install flask numpy Pillow
  python carla_relay_core.py [--host 0.0.0.0] [--port 5000] [--carla-port 2000]
  python carla_relay_core.py --carla-root "x:/App/CARLA_0.9.16"   # 指定 CARLA 根目录

启动方式二（单进程部署，前端+API 合一）：
  python carla_relay_core.py --static-dir ../dist

carla 模块由本文件启动时从 CARLA 安装目录的 PythonAPI/carla/dist/carla-*.egg
自动注入 sys.path，无需 pip 安装。等价入口：在 server/ 下 `python -m carla_relay`。

依赖（仅 Windows 端）：
  - flask
  - numpy
  - Pillow
  - pygame / requests（仅 local_runner 命令行体验客户端需要）
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import queue
import random
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 定位 CARLA 根目录（含 CarlaUE4 的目录），并据此把官方 PythonAPI（agents 等）
# 注入 sys.path。这样即使不在 CARLA 根目录下运行，也能正确 `import agents`、`import carla`。
# 优先级：--carla-root 参数 > 环境变量 CARLA_ROOT > 从当前目录向上查找 > 当前目录。
# ---------------------------------------------------------------------------
_CARLA_ROOT: Optional[str] = None


def _resolve_carla_root() -> Optional[str]:
    """解析 CARLA 根目录：
    1) 环境变量 CARLA_ROOT；
    2) 从当前工作目录不断向上查找 CarlaUE4.exe / CarlaUE4.sh；
    3) 均未找到则返回 None（调用方回退到当前目录）。"""
    env = (os.environ.get("CARLA_ROOT") or "").strip()
    if env:
        for exe in ("CarlaUE4.exe", "CarlaUE4.sh"):
            if os.path.isfile(os.path.join(env, exe)):
                return env
    here = os.getcwd()
    while True:
        for exe in ("CarlaUE4.exe", "CarlaUE4.sh"):
            if os.path.isfile(os.path.join(here, exe)):
                return here
        parent = os.path.dirname(here)
        if parent == here:          # 已达文件系统根目录
            return None
        here = parent


def _bootstrap_relay_path(carla_root: Optional[str] = None) -> str:
    """将 CARLA PythonAPI（含 agents 导航包）及 carla 模块的 egg 加入 sys.path。
    返回最终确定使用的 CARLA 根目录。未在 CARLA 根目录运行时也能正确引入依赖。"""
    global _CARLA_ROOT
    if not carla_root:
        carla_root = _resolve_carla_root() or os.getcwd()
    _CARLA_ROOT = carla_root
    _CARLA_ROOT = os.path.abspath(_CARLA_ROOT)

    pyapi = os.path.join(_CARLA_ROOT, "PythonAPI", "carla")
    import glob as _glob
    candidates = [pyapi]                        # agents 从这里导入（import agents）
    candidates += _glob.glob(                   # 官方做法：dist 下 carla-*.egg 直接加载
        os.path.join(pyapi, "dist", "carla-*.egg")
    )
    # 兼容脚本被放置/复制的其它位置（多为 CARLA 根目录内的副本）
    for base in (
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
    ):
        candidates.append(os.path.join(base, "PythonAPI", "carla"))
    for cand in candidates:
        if cand not in sys.path and (os.path.isdir(cand) or os.path.isfile(cand)):
            sys.path.insert(0, cand)
    if _CARLA_ROOT != os.getcwd():
        print(f"[PATH] CARLA 根目录: {_CARLA_ROOT}", flush=True)
    return _CARLA_ROOT


_CARLA_ROOT = _bootstrap_relay_path()

# ---------------------------------------------------------------------------
# 注入 server/ 包目录，使本文件可 import carla_relay 新包。
# 引导壳已随核心逻辑一起移入 server/ 目录，默认包目录即本文件所在目录；
# 若由 public/carla_relay.py 启动器拉取旧副本到 public/ 下运行，则回退到
# 上一级 server/ 目录（兼容旧部署路径）。
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER_PKG_DIR = _HERE
if not os.path.isdir(os.path.join(_SERVER_PKG_DIR, "carla_relay")):
    _legacy_server = os.path.join(os.path.dirname(_HERE), "server")
    if os.path.isdir(os.path.join(_legacy_server, "carla_relay")):
        _SERVER_PKG_DIR = _legacy_server
if _SERVER_PKG_DIR not in sys.path:
    sys.path.insert(0, _SERVER_PKG_DIR)

import carla
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__)


@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


# ---------------------------------------------------------------------------
# P3 迁移：基础路由（health/sync/vehicle/sensors/stream/preview/misc）已抽取至
# carla_relay.routes 蓝图包，URL 与响应格式零变更。蓝图经 app.extensions["legacy"]
# 在请求期动态访问本模块全局（world / 帧缓存 / stream 目标等）。
# ---------------------------------------------------------------------------
import carla_relay.routes as _routes_pkg

_routes_pkg.register_all(app)
app.extensions["legacy"] = sys.modules[__name__]


# =============================================================================
# 全局状态
# =============================================================================

world: Optional[carla.World] = None
client: Optional[carla.Client] = None
traffic_manager: Optional[carla.TrafficManager] = None
_managed_actors: set[int] = set()
_autopilot_state: Dict[int, bool] = {}  # vehicle_id -> autopilot_enabled
_lock: threading.Lock = threading.Lock()

# 传感器最新帧缓存: sensor_id -> bytes (JPEG)
_sensor_frames: Dict[int, bytes] = {}  # sid → 最新帧数据（JPEG 字节或 JSON 字符串字节）
_sensor_frame_num: Dict[int, int] = {}  # sid → CARLA 帧号（用于调试）
_sensor_dtype: Dict[int, str] = {}  # sensor_id -> "camera" | "lidar" | "gnss" | "imu" | "semantic" | "instance"
_semantic_raw: Dict[int, bytes] = {}  # semantic sensor_id -> 原始 BGRA raw（用于统计占比）
_instance_raw: Dict[int, bytes] = {}  # instance sensor_id -> 原始 BGRA raw（用于 22 类动态细分）

# 旧版同步设置（用于恢复）
_old_settings: Any = None

# SSE 订阅者队列已抽取至 carla_relay.core.sse（P2）：订阅者管理与广播由 hub 承载
# =============================================================================
# 以下配置/感知/核心逻辑已抽取至 server/carla_relay 包（P1-P2 绞杀者迁移）：
#   - carla_relay.config: 语义类别表 / 类别预设映射 / 帧间隔常量
#   - carla_relay.perception.semantics: 语义类别映射纯函数
#   - carla_relay.core.sse: SSE 订阅者 hub
#   - carla_relay.core.sensors: 传感器帧序列化
#   - carla_relay.core.carla_client: CARLA 连接与进程管理
# 此处以别名导入保持本文件内部引用不变（行为零变更）。
# =============================================================================
from carla_relay.config import (
    FRAME_INTERVAL,
    SEMANTIC_CLASSES as _SEMANTIC_CLASSES,
    SEMANTIC_PALETTE as _SEMANTIC_PALETTE,
    SEMANTIC_PRESETS as _SEMANTIC_PRESETS,
    SEMANTIC_PRESET_LABELS as _SEMANTIC_PRESET_LABELS,
)
from carla_relay.perception.semantics import (
    dynamic_class as _dynamic_class,
    label_semantic_classes as _label_semantic_classes,
    colors_from_labels as _colors_from_labels,
    ratios_from_labels as _ratios_from_labels,
)
from carla_relay.core.sse import hub as _sse_hub
from carla_relay.core.sensors import serialize_frame as _serialize_sensor_frame
import carla_relay.core.carla_client as _carla_client


def _push_to_sse(msg: dict):
    """向所有 SSE 订阅者推送消息（P2 已抽取至 carla_relay.core.sse.hub）"""
    _sse_hub.push(msg)


# SSE 推送去重：slot → (上次推送帧号, 上次推送时间)。同一帧 2s 内不重复推送：
# 世界暂停时大幅减少重复大包；超过 2s 仍重发一次，保证新订阅/重连的客户端能拿到当前画面
_frame_slot_state: Dict[str, tuple] = {}


def _frame_msg(slot: str, sid: Optional[int]) -> Optional[dict]:
    """组装某个画面槽位的帧消息；帧号未变且距上次推送不足 2s 时返回 None（跳过）"""
    if sid is None or sid not in _sensor_frames:
        _frame_slot_state.pop(slot, None)
        return None
    fn = _sensor_frame_num.get(sid, 0)
    now = time.time()
    st = _frame_slot_state.get(slot)
    if st is not None and st[0] == fn and (now - st[1]) < 2.0:
        return None
    _frame_slot_state[slot] = (fn, now)
    return {
        "sensor_id": sid,
        "base64": base64.b64encode(_sensor_frames[sid]).decode(),
        "frame_num": fn,
    }


def _sse_stream_thread():
    """后台线程：每 50ms 读取全局 _stream_vehicle/_stream_camera，推送给 SSE 订阅者"""
    while True:
        try:
            # 无订阅者时跳过组包（避免空闲时反复做 base64/JPEG 大包序列化浪费 CPU）
            if not _sse_hub.has_subscribers():
                time.sleep(FRAME_INTERVAL)
                continue
            msg = {"ts": time.time()}
            vid = _stream_vehicle
            # 车辆状态
            if vid is not None:
                actor = world.get_actor(vid)
                if actor is not None and actor.is_alive:
                    v = actor
                    loc = v.get_location()
                    vel = v.get_velocity()
                    t = v.get_transform()
                    msg["vehicle"] = {
                        "id": vid,
                        "location": {"x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2)},
                        "transform": {
                            "rotation": {"pitch": round(t.rotation.pitch, 2),
                                         "yaw": round(t.rotation.yaw, 2),
                                         "roll": round(t.rotation.roll, 2)}
                        },
                        "speed_ms": round(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2), 2),
                        "speed_kmh": round(math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6, 1),
                        "is_alive": True,
                        "autopilot": _autopilot_state.get(vid, True),
                    }
                else:
                    msg["vehicle"] = {"id": vid, "is_alive": False}
            # 相机帧（主/前相机）
            m = _frame_msg("camera", _stream_camera)
            if m is not None:
                msg["camera"] = m
            # 左相机（三目实验）
            m = _frame_msg("cameraL", _stream_camera_left)
            if m is not None:
                msg["cameraL"] = m
            # 右相机（三目实验）
            m = _frame_msg("cameraR", _stream_camera_right)
            if m is not None:
                msg["cameraR"] = m
            # 语义分割帧（独立于 RGB 相机流）
            m = _frame_msg("semantic", _stream_semantic)
            if m is not None:
                msg["semantic"] = m
            # GNSS / IMU / LiDAR / Radar 等传感器数据
            sensors = {}
            for sid, raw in _sensor_frames.items():
                dtype = _sensor_dtype.get(sid)
                if dtype in ("gnss", "imu"):
                    try:
                        sensors[dtype] = json.loads(raw.decode())
                    except Exception:
                        pass
                elif dtype == "lidar":
                    try:
                        sensors["lidar"] = json.loads(raw.decode())
                    except Exception:
                        pass
                elif dtype == "radar":
                    try:
                        sensors["radar"] = json.loads(raw.decode())
                    except Exception:
                        pass
            if sensors:
                msg["sensors"] = sensors
            # 深度相机帧（独立于 RGB 相机流；与 camera/semantic 同口径走 _frame_msg
            # 做帧号去重：同一帧 2s 内不重复推送，避免暂停/实验结束后仍以 20Hz
            # 空转重发同一深度帧，超过 2s 重发一次保证新订阅/重连能拿到当前画面）
            for sid in _sensor_frames:
                if _sensor_dtype.get(sid) == "depth":
                    m = _frame_msg("depth", sid)
                    if m is not None:
                        msg["depth"] = m
                    break  # 只推第一个深度相机
            # 高空俯视相机帧（跟随车辆，真实渲染俯瞰画面）
            m = _frame_msg("bird", _stream_bird)
            if m is not None:
                msg["bird"] = m
            # 包围框相机帧（实验10：前置 RGB + 2D 检测框）
            m = _frame_msg("bbox", _stream_bbox)
            if m is not None:
                msg["bbox"] = m
            _push_to_sse(msg)
        except Exception:
            pass
        time.sleep(FRAME_INTERVAL)


def _init_carla(carla_host: str, carla_port: int, auto_manage: bool = True) -> None:
    """连接 CARLA（P2 已抽取至 carla_relay.core.carla_client.connect）并写入全局状态"""
    global world, client, traffic_manager
    client, world, traffic_manager = _carla_client.connect(
        carla_host, carla_port, auto_manage=auto_manage,
        carla_root=_CARLA_ROOT,
        extra_search_dirs=(os.getcwd(), os.path.dirname(os.path.abspath(__file__))),
    )


# =============================================================================
# API: 健康检查 / 地图信息 / 同步模式 —— P3 已迁移至 carla_relay.routes（health.py / sync.py）
# =============================================================================

# =============================================================================
# API: 车辆生成/销毁/控制/自动驾驶/视角 —— P3 已迁移至 carla_relay.routes（vehicle.py）
# =============================================================================

# =============================================================================
# API: 传感器
# =============================================================================

def _sensor_callback(sid: int, dtype: str, data: Any):
    """传感器回调：序列化（P2 已抽取至 carla_relay.core.sensors）并写入全局帧缓存"""
    try:
        payload, fnum = _serialize_sensor_frame(dtype, data)
        if dtype == "semantic":
            # 仅缓存原始 BGRA；分级可视化由实验5主循环统一渲染，
            # 避免在此处写原始 CityScapes 彩图覆盖分级结果。
            _semantic_raw[sid] = payload
        elif dtype == "instance":
            _instance_raw[sid] = payload  # 原始 BGRA，22 类动态细分在主循环解码
        elif dtype == "depth":
            pass  # 深度帧由实验主循环独占渲染（支持距离截断），core 不写原始帧
        else:
            _sensor_frames[sid] = payload
            if fnum is not None:
                _sensor_frame_num[sid] = fnum
        _sensor_dtype[sid] = dtype
    except Exception as exc:
        print(f"[WARN] 传感器回调失败 dtype={dtype} sid={sid}: {exc}")


# 传感器挂载 / 帧查询 / 删除路由 —— P3 已迁移至 carla_relay.routes（sensors.py）

# =============================================================================
# API: SSE 实时数据流
# =============================================================================

_stream_vehicle: Optional[int] = None
_stream_camera: Optional[int] = None       # 前/主相机（实验4=前相机）
_stream_camera_left: Optional[int] = None  # 左相机（实验4三目）
_stream_camera_right: Optional[int] = None # 右相机（实验4三目）
_stream_semantic: Optional[int] = None
_stream_bird: Optional[int] = None  # 高空俯视相机（跟随车辆，真实渲染俯瞰画面）
_stream_bbox: Optional[int] = None  # 包围框相机（实验10：前置 RGB + 2D 检测框，参照官方 bounding_boxes.py）
_semantic_preset: str = "7"  # 当前语义分割类别预设（7 / 22），运行中可切换
_camera_actor_ref: Any = None  # 防止相机 actor 被 GC 导致 listen() 回调失效
_sensor_refs: Dict[int, Any] = {}  # 保持所有带 listen() 的 sensor actor 引用
_stream_thread: Optional[threading.Thread] = None


def _start_stream_thread():
    global _stream_thread
    if _stream_thread is not None and _stream_thread.is_alive():
        return
    _stream_thread = threading.Thread(
        target=_sse_stream_thread, daemon=True
    )
    _stream_thread.start()


# stream / preview / debug / actors / cleanup 路由 —— P3 已迁移至
# carla_relay.routes（stream.py / misc.py）。SSE 线程 _sse_stream_thread 与
# stream 目标全局仍在本文件，蓝图经 app.extensions["legacy"] 动态读写。


# ---------------------------------------------------------------------------
# P4 迁移：实验逻辑与世界 API 已物理拆分至 carla_relay 包（绞杀者迁移）：
#   - carla_relay.experiments: 定位分析(localization) / Lidar检测(lidar_detection) /
#     语义分割(semantic_segmentation) + 历史实验 + 共享状态 + 控制器；
#     综合驾驶(comprehensive_driving) 单独经 load_comprehensive_driving_into 载入
#   - carla_relay.world_api:   感知查询 / 路径规划 / 地图 / 行人 / 俯瞰 API
# 片段经 load_into(globals()) 载入本模块命名空间执行：代码原样迁移，
# __file__ / global 重绑定 / @app.route 注册语义不变，行为零漂移。
# 加载顺序与原文件模块级语句执行顺序完全一致：
# 其余实验 → 世界API → 综合驾驶。URL 中的数字实验 ID 为 API 契约，不变。
# ---------------------------------------------------------------------------
from carla_relay.experiments import load_into as _load_experiment_fragments
from carla_relay.experiments import load_comprehensive_driving_into as _load_comprehensive_driving_fragments
from carla_relay.world_api import load_into as _load_world_api_fragments

_load_experiment_fragments(globals())
_load_world_api_fragments(globals())
_load_comprehensive_driving_fragments(globals())

# =============================================================================
# 综合驾驶（闭环自动驾驶，API 实验ID 10）→ P4 已迁移至
# carla_relay/experiments/comprehensive_driving/（分层真实模块 + run.py 薄编排片段）
# =============================================================================


# =============================================================================
# 静态文件托管（可选，将前端 dist/ 与 relay 合并为单进程应用）
# =============================================================================

STATIC_DIR: Optional[Path] = None


def _setup_static(static_dir: Path):
    """在所有 API 路由之后注册静态文件路由，API 路由优先匹配。"""
    global STATIC_DIR
    STATIC_DIR = static_dir.resolve()
    if not STATIC_DIR.is_dir():
        raise FileNotFoundError(f"静态文件目录不存在: {STATIC_DIR}")

    @app.route("/")
    def _serve_index():
        return send_from_directory(str(STATIC_DIR), "index.html")

    @app.route("/<path:filename>")
    def _serve_static(filename):
        return send_from_directory(str(STATIC_DIR), filename)


# =============================================================================
# 主入口
# =============================================================================


# CARLA 进程管理（P2 已抽取至 carla_relay.core.carla_client）：
# 此处保留别名，供本文件内部调用点（main 等）零改动。
_kill_other_relays = _carla_client.kill_other_relays
_find_carla_executable = _carla_client.find_carla_executable
_list_carla_pids = _carla_client.list_carla_pids
_kill_all_carla = _carla_client.kill_all_carla
_launch_carla = _carla_client.launch_carla
_wait_carla_ready = _carla_client.wait_carla_ready
_ensure_carla_ready = _carla_client.ensure_carla_ready


def main():
    parser = argparse.ArgumentParser(description="CARLA HTTP 中继网关")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=5000, help="HTTP 端口")
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA 服务端地址")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA RPC 端口")
    parser.add_argument(
        "--carla-root", default=None,
        help="CARLA 安装根目录（含 CarlaUE4.exe 的目录），默认从当前目录向上查找，找不到则用当前目录"
    )
    parser.add_argument("--debug", action="store_true", help="Flask debug 模式")
    parser.add_argument(
        "--static-dir", default=None,
        help="前端静态文件目录（如 dist/），设置后将托管前端页面，实现单进程部署"
    )
    parser.add_argument(
        "--no-carla-manage", action="store_true",
        help="禁用启动时的 CARLA 自动检测/清理/重启（默认多实例时会自动全部清除并重启一个）"
    )
    args = parser.parse_args()

    # 显式 --carla-root 时，用其覆盖启动时自动解析的根目录，并重新注入 sys.path
    global _CARLA_ROOT
    if args.carla_root:
        _CARLA_ROOT = _bootstrap_relay_path(args.carla_root)

    _kill_other_relays()

    _init_carla(args.carla_host, args.carla_port, auto_manage=not args.no_carla_manage)

    # #region debug-point util
    import faulthandler, atexit
    faulthandler.enable()
    def _dbg_exit():
        try:
            import json, urllib.request as _ur
            _ur.urlopen(_ur.Request("http://127.0.0.1:7777/event",
                data=json.dumps({"sessionId": "experiment4-restart-crash", "runId": "post-fix3",
                "hypothesisId": "E", "location": "atexit", "msg": "[DEBUG] process-exit"}).encode(),
                headers={"Content-Type": "application/json"}))
        except Exception:
            pass
    atexit.register(_dbg_exit)
    # #endregion

    if args.static_dir:
        _setup_static(Path(args.static_dir))

    print(f"\n{'='*60}")
    print(f"  CARLA 中继网关已启动")
    print(f"  监听: http://{args.host}:{args.port}")
    print(f"  CARLA: {args.carla_host}:{args.carla_port}")
    if args.static_dir:
        print(f"\n  前端页面: http://127.0.0.1:{args.port}/")
        print(f"  静态目录: {Path(args.static_dir).resolve()}")
    print(f"\n  快速测试:")
    print(f"    curl http://127.0.0.1:{args.port}/health")
    print(f"    curl -X POST http://127.0.0.1:{args.port}/vehicle/spawn")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
