"""relay 服务 HTTP / SSE 客户端。

仅依赖 requests，不依赖 carla 模块，可在无 CARLA 环境下做参数往返调试。
对应 server/carla_relay 现有路由：
  GET  /experiment/status               运行状态
  POST /experiment/<id>/start           启动实验（body 为前端参数面板同款 JSON）
  POST /experiment/<id>/stop            停止实验
  GET  /map/road_network                地图车道拓扑折线（综合驾驶画鸟瞰图）
  POST /route/plan                      全局路线规划（综合驾驶）
  GET  /stream                          SSE 实时数据流（相机/鸟瞰/实验遥测）
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests

# 参数文件夹 / 结果输出文件夹（相对本文件：server/local_runner/relay_client.py → parent 即 server）
_PARAMS_DIR = Path(__file__).resolve().parent.parent / "experiment_params"
_OUTPUT_DIR = _PARAMS_DIR / "output"


class RelayError(RuntimeError):
    """relay 返回非 ok 状态时抛出，携带 message"""


class RelayClient:
    """封装 relay 的 HTTP 接口与 SSE 数据流解析。"""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self._session = requests.Session()

    # ---------- HTTP 接口 ----------
    def _post(self, path: str, body: Optional[dict] = None) -> dict:
        url = self.base + path
        try:
            r = self._session.post(url, json=body or {}, timeout=20)
        except requests.RequestException as exc:
            raise RelayError(f"请求失败 {url}: {exc}") from exc
        try:
            data = r.json()
        except ValueError as exc:
            raise RelayError(f"响应非 JSON: {exc}") from exc
        if data.get("status") == "error" or (data.get("status") not in (None, "ok")):
            raise RelayError(data.get("message") or data.get("status") or f"HTTP {r.status_code}")
        return data

    def _get(self, path: str) -> dict:
        url = self.base + path
        try:
            r = self._session.get(url, timeout=20)
        except requests.RequestException as exc:
            raise RelayError(f"请求失败 {url}: {exc}") from exc
        return r.json()

    def status(self) -> dict:
        return self._get("/experiment/status")

    def start(self, exp_id: int, params: dict) -> dict:
        return self._post(f"/experiment/{exp_id}/start", params)

    def stop(self, exp_id: int) -> dict:
        return self._post(f"/experiment/{exp_id}/stop")

    def road_network(self) -> dict:
        return self._get("/map/road_network")

    def map_render(self) -> dict:
        """整城俯视底图（base64 + 投影矩阵），综合驾驶规划阶段用。"""
        return self._get("/map/render")

    def route_plan(self, start: dict, end: dict, sampling: float) -> dict:
        return self._post("/route/plan", {
            "start": start,
            "end": end,
            "sampling_resolution": sampling,
        })

    # ---------- SSE 流 ----------
    def stream(self) -> Iterable[dict]:
        """阻塞式迭代 relay /stream 的消息（每条已 JSON 反序列化）。"""
        url = self.base + "/stream"
        try:
            with self._session.get(url, stream=True, timeout=(10, 90)) as resp:
                resp.raise_for_status()
                buf = ""
                for raw in resp.iter_content(chunk_size=4096, decode_unicode=True):
                    if not raw:
                        continue
                    buf += raw
                    while "\n\n" in buf:
                        block, buf = buf.split("\n\n", 1)
                        for line in block.splitlines():
                            if line.startswith("data:"):
                                payload = line[5:].strip()
                                if not payload:
                                    continue
                                try:
                                    yield json.loads(payload)
                                except ValueError:
                                    continue
                                break
        except requests.RequestException as exc:
            raise RelayError(f"SSE 连接断开: {exc}") from exc


def decode_frame_b64(b64: str):
    """把 SSE 消息里的 base64 JPEG 解码为 python 字节，失败返回 None。"""
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def save_output(record: dict) -> Path:
    """把一次实验的记录写入 experiment_params/output/<时间戳>_<slug>.json。

    record 需包含 slug/experiment_id/label 等；返回落盘路径。
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = _OUTPUT_DIR / f"{stamp}_{record.get('slug', 'exp')}_{record.get('experiment_id', 0)}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


class SSEQueue:
    """后台线程拉取 SSE，把最新各槽位消息放入线程安全结构供 pygame 主循环读取。

    由于 pygame 主循环与 SSE 读取是不同线程，这里只缓存"最新一帧"（用锁保护），
    渲染层仅消费最新的快照，不需要对历史做排队。
    """

    def __init__(self, client: RelayClient):
        self._client = client
        self._lock = threading.Lock()
        self._latest: Dict[str, dict] = {}          # 槽位(dict key) → 最新消息
        self._surfaces: Dict[str, str] = {}          # "camera"/"bird"/... → base64
        self._experiment: Dict[str, object] = {}     # msg["experiment"]：最新实验消息
        self._exp_final: Dict[str, object] = {}      # 粘性终态消息（result/report/done…）
        self._exp_seq = -1                           # 实验消息序号（去重用）
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                for msg in self._client.stream():
                    if not self._running:
                        return
                    self._ingest(msg)
            except RelayError:
                time.sleep(0.5)

    def _ingest(self, msg: dict):
        with self._lock:
            # 相机/语义/鸟瞰/包围框等图像槽位
            for slot, key in (("camera", "camera"), ("bird", "bird"),
                              ("semantic", "semantic"), ("bbox", "bbox"),
                              ("cameraL", "cameraL"), ("cameraR", "cameraR"),
                              ("depth", "depth")):
                blk = msg.get(slot)
                if isinstance(blk, dict) and blk.get("base64"):
                    self._surfaces[slot] = blk["base64"]
            # 实验遥测 / 结果 / 日志 / 碰撞
            # 注意：服务端推 result 后紧跟一条 log（"实验完成"），若只保留最新消息
            # result 会被瞬间覆盖导致永远看不到完成 → 终态消息单独粘性保存。
            exp = msg.get("experiment")
            if isinstance(exp, dict):
                self._experiment = exp
                self._exp_seq += 1
                if ("result" in exp or "report" in exp
                        or exp.get("status") in ("done", "stopped", "error")):
                    self._exp_final.update(exp)

    def snapshot(self):
        """返回 (最新图像槽位复制, 实验消息复制(终态字段优先), experiment 消息序号)。"""
        with self._lock:
            exp = dict(self._experiment)
            if self._exp_final:
                exp.update(self._exp_final)
            return dict(self._surfaces), exp, self._exp_seq


def img_bytes_to_surface(raw: bytes):
    """把 JPEG 字节转成 pygame.Surface；失败返回 None（外面处理）。

    .convert() 需要 display 已初始化（如 map_picker 在 pygame.init() 前
    预加载底图），失败时回退原始 Surface。
    """
    try:
        import pygame
        surf = pygame.image.load(io.BytesIO(raw))
        try:
            return surf.convert()
        except Exception:
            return surf
    except Exception:
        return None