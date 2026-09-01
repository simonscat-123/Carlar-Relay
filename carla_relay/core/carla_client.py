"""CARLA 连接与进程管理。

职责：
- relay 进程互斥（启动前清理其它 carla_relay 进程）；
- CARLA 模拟器进程管理（查找可执行文件 / 列举 / 终止 / 启动）；
- 等待模拟器就绪 + 建立客户端连接。

carla 模块在函数内延迟导入（依赖引导壳先完成 egg 路径注入）。
进程管理仅在 Windows 下生效，与原实现一致。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional, Tuple


def kill_other_relays() -> None:
    """启动前清理其它已运行的 relay 进程，避免多个 relay 抢占同一 CARLA 实例导致冲突。
    仅终止命令行中包含 carla_relay（carla_relay.py / carla_relay_core.py）且非自身的
    python 进程（仅 Windows 有效）。"""
    me = os.getpid()
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*carla_relay*' "
        "-and $_.ProcessId -ne @ME@ } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "
        "Write-Output $_.ProcessId }"
    ).replace("@ME@", str(me))
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
        killed = [p for p in out.stdout.split() if p.isdigit()]
        if killed:
            print(f"[RELAY] 检测到并已清理其它 relay 进程: pid={','.join(killed)}", flush=True)
    except Exception as exc:
        print(f"[RELAY] 清理其它 relay 进程失败: {exc}", flush=True)


def _search_up_carla_exe(start: str) -> Optional[str]:
    """从 start 目录不断向上查找 CarlaUE4 可执行文件。"""
    here = start
    while True:
        for exe in ("CarlaUE4.exe", "CarlaUE4.sh"):
            cand = os.path.join(here, exe)
            if os.path.isfile(cand):
                return cand
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def find_carla_executable(
    carla_root: Optional[str] = None,
    extra_search_dirs: Tuple[str, ...] = (),
) -> Optional[str]:
    """定位 CarlaUE4 可执行文件（Windows: CarlaUE4.exe / Linux: CarlaUE4.sh）。
    优先用已解析的 CARLA 根目录，其次在 extra_search_dirs 各目录向上查找。"""
    for start in ([carla_root] if carla_root else []) + list(extra_search_dirs):
        found = _search_up_carla_exe(start)
        if found:
            return found
    return None


def list_carla_pids() -> list:
    """列出正在运行的 CARLA 服务进程 PID（仅 Windows，通过 PowerShell 查询）。
    只统计真正的服务进程（CarlaUE4-Win64-*），启动器 CarlaUE4.exe 不计入实例数。"""
    if sys.platform != "win32":
        return []
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'CarlaUE4-Win64-*' } | "
        "ForEach-Object { Write-Output $_.ProcessId }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
        return [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    except Exception:
        return []


def kill_all_carla() -> list:
    """终止本机所有 CarlaUE4 进程（启动器与服务端）。"""
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'CarlaUE4*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "
        "Write-Output $_.ProcessId }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=15,
        )
        killed = [p for p in out.stdout.split() if p.strip().isdigit()]
        if killed:
            print(f"[CARLA] 已终止全部 CARLA 进程: pid={','.join(killed)}", flush=True)
        return killed
    except Exception as exc:
        print(f"[CARLA] 终止 CARLA 进程失败: {exc}", flush=True)
        return []


def launch_carla(exe: str) -> None:
    """启动 CARLA 模拟器（非阻塞）。"""
    root = os.path.dirname(exe)
    try:
        if sys.platform == "win32":
            subprocess.Popen([exe], cwd=root)
        else:
            subprocess.Popen([exe], cwd=root, shell=True)
        print(f"[CARLA] 已启动模拟器: {exe}", flush=True)
    except Exception as exc:
        print(f"[CARLA] 启动模拟器失败: {exc}", flush=True)


def wait_carla_ready(carla_host: str, carla_port: int, timeout: int = 180) -> None:
    """轮询等待 CARLA 模拟器就绪（RPC 握手 + world 可用），超时抛出带诊断的 RuntimeError。
    注意：每次探测都新建 carla.Client，避免某次超时后客户端内部连接进入坏状态、
    即使模拟器已恢复也永远无法重连（复用同一客户端会出现“一直等待就绪”的假象）。"""
    import carla  # 延迟导入：依赖引导壳先完成 egg 路径注入

    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            probe = carla.Client(carla_host, carla_port)
            probe.set_timeout(3.0)
            ver = probe.get_server_version()
            world = probe.get_world()
            print(f"[CARLA] 模拟器就绪: v{ver}, 地图 {world.get_map().name}", flush=True)
            return
        except Exception:
            pass
        if attempt == 1 or attempt % 5 == 0:
            print(f"[CARLA] 等待模拟器就绪中... ({int(time.time() - start)}s/{timeout}s)", flush=True)
        time.sleep(2)
    raise RuntimeError(
        f"等待 CARLA 模拟器就绪超时（{timeout}s，{carla_host}:{carla_port}）。\n"
        "可能原因：① CARLA 仍在加载或卡死；② 存在多个 CARLA 实例互相抢占资源。\n"
        "建议：打开任务管理器结束所有 CarlaUE4 进程后重新启动 CARLA，再启动 relay。"
    )


def ensure_carla_ready(
    carla_host: str,
    carla_port: int,
    auto_manage: bool = True,
    wait_timeout: int = 180,
    carla_root: Optional[str] = None,
    extra_search_dirs: Tuple[str, ...] = (),
) -> None:
    """启动前检测 CARLA 状态并保证其就绪：
    - 多个实例（>1）: 全部清除后自动重启一个；
    - 无实例: 自动启动一个；
    - 单个实例: 直接等待其就绪（不做破坏性操作）。
    设置 auto_manage=False 可跳过所有自动管理，仅等待就绪。"""
    if not auto_manage:
        wait_carla_ready(carla_host, carla_port, wait_timeout)
        return
    pids = list_carla_pids()
    if len(pids) > 1:
        print(f"[CARLA] 检测到 {len(pids)} 个 CARLA 实例 (pid={pids})，将全部清除并重新启动", flush=True)
        kill_all_carla()
        time.sleep(3)
        pids = list_carla_pids()
    if len(pids) == 0:
        exe = find_carla_executable(carla_root, extra_search_dirs)
        if not exe:
            raise RuntimeError("未找到 CarlaUE4 可执行文件，无法自动启动 CARLA，请手动启动后重试。")
        if not list_carla_pids():
            launch_carla(exe)
        wait_carla_ready(carla_host, carla_port, wait_timeout)
        return
    print(f"[CARLA] 检测到 {len(pids)} 个 CARLA 实例 (pid={pids})，等待其就绪", flush=True)
    wait_carla_ready(carla_host, carla_port, wait_timeout)


def connect(
    carla_host: str,
    carla_port: int,
    auto_manage: bool = True,
    carla_root: Optional[str] = None,
    extra_search_dirs: Tuple[str, ...] = (),
) -> Tuple:
    """建立 CARLA 连接，返回 (client, world, traffic_manager)。
    调用方负责保存引用（当前写入 MkAppContext 全局）。"""
    import carla  # 延迟导入：依赖引导壳先完成 egg 路径注入

    ensure_carla_ready(
        carla_host, carla_port, auto_manage=auto_manage,
        carla_root=carla_root, extra_search_dirs=extra_search_dirs,
    )
    client = carla.Client(carla_host, carla_port)
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(False)
    print(f"[OK] 已连接到 CARLA {client.get_server_version()} (地图: {world.get_map().name})")
    return client, world, traffic_manager
