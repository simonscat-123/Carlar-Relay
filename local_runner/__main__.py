"""local_runner：命令行 + pygame 的 CARLA 实验纯体验仿真客户端。

用法（在 server 目录下执行）:
    python -m local_runner localization
    python -m local_runner lidar_detection
    python -m local_runner semantic_segmentation
    python -m local_runner comprehensive_driving        # 先点选起止点
    python -m local_runner <slug> --params experiment_params/localization.json
    python -m local_runner <slug> --relay http://127.0.0.1:5000

依赖: pip install requests pygame   (不需要 carla 模块，不启动 CARLA)
流程: 读参数 → 探测 relay → (综合驾驶先规划) → 启动实验 → SSE+pygame 渲染 →
      收到 result 自动写入 experiment_params/output/ → ESC 退出并调 stop。
纯体验仿真：不接入思考题/提交功能。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pygame

from .charts import TelemetryHistory
from .fonts import clear_font_cache, load_font
from .relay_client import (
    _PARAMS_DIR,
    _OUTPUT_DIR,
    RelayClient,
    RelayError,
    SSEQueue,
    save_output,
)
from .views import dispatch_render

# slug → 默认参数文件
DEFAULT_PARAMS = {
    "localization": _PARAMS_DIR / "localization.json",
    "lidar_detection": _PARAMS_DIR / "lidar_detection.json",
    "semantic_segmentation": _PARAMS_DIR / "semantic_segmentation.json",
    "comprehensive_driving": _PARAMS_DIR / "comprehensive_driving.json",
    "route_planning": _PARAMS_DIR / "route_planning.json",
}

# 提高纵向分辨率以容纳下方图表与状态面板，避免压缩/遮挡并减少留白
WINDOW_SIZE = (1400, 900)


def _load_config(slug: str, params_path: str | None) -> dict:
    path = Path(params_path) if params_path else DEFAULT_PARAMS.get(slug)
    if path is None:
        raise SystemExit(
            f"未知实验 '{slug}'。可用: {', '.join(DEFAULT_PARAMS)}；"
            f"或通过 --params 指定参数文件。"
        )
    if not path.is_file():
        raise SystemExit(f"参数文件不存在: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["slug"] = cfg.get("slug", slug)
    cfg["params_file"] = str(path)
    return cfg


def _make_fonts():
    def f(size, bold=False):
        return load_font(size, bold=bold)
    return {"sm": f(16), "md": f(20), "lg": f(26, bold=True)}


def _check_relay(client: RelayClient) -> dict:
    st = client.status()
    if st.get("running"):
        print(f"[relay] 已有实验在运行 (id={st.get('experiment_id')})，请先停止后再体验")
        raise SystemExit(1)
    print(f"[relay] 已连接 {client.base}")
    return st


def _resolve_done(exp) -> str | None:
    """从实验遥测消息推断是否已结束及其状态。

    返回 "done"/"stopped"/"error"，未结束返回 None。
    - 实验 10 完成带 status:"done"，被 stop 带 status:"stopped"，异常带 status:"error"
    - 实验 4/5/23 正常完成只带 result 键（无 status）；被 stop 带 status:"stopped"
    """
    if not exp:
        return None
    st = exp.get("status")
    if st in ("done", "stopped", "error"):
        return st
    if "result" in exp or "report" in exp:
        return "done"
    return None


def _run_experiment(cfg: dict, planner=None) -> int:
    """执行单个实验的主循环。planner 为 None 时不做规划阶段。"""
    exp_id = int(cfg["experiment_id"])
    slug = cfg["slug"]
    relay_url = cfg.get("relay", "http://127.0.0.1:5000")
    client = RelayClient(relay_url)

    _check_relay(client)

    params = dict(cfg.get("params") or {})
    run_meta = {}  # 额外的运行信息（起终点/路线），随结果一起落盘
    lanes = []     # 综合驾驶：车道数据（鸟瞰叠加层用）

    # 综合驾驶：先加载路网 → 进入鸟瞰规划阶段
    if cfg.get("plan_stage"):
        from . import map_picker
        print("[plan] 正在加载路网与整城俯瞰底图…")
        try:
            net = client.road_network()
            lanes = net.get("lanes") or []
        except RelayError as exc:
            print(f"[plan] 路网加载失败: {exc}")
        print("[plan] 请在鸟瞰图上点选起点与终点…")
        plan = map_picker.run(client, params, lanes=lanes,
                              width=1152, height=720)
        if plan.get("start") is None:
            print("[plan] 已取消，未开始实验。")
            return 0
        # 把点选坐标并入启动参数；综合驾驶运行逻辑优先使用 start/end 坐标
        params["start"] = {"x": plan["start"]["x"], "y": plan["start"]["y"]}
        params["end"] = {"x": plan["end"]["x"], "y": plan["end"]["y"]}
        run_meta["plan"] = plan

    pygame.init()
    # 规划阶段可能已 pygame.quit()（map_picker），需清掉失效的字体缓存，防止
    # 复用已释放 SDL_ttf 资源的 Font 对象导致原生崩溃（黑屏闪退）。
    clear_font_cache()
    pygame.display.set_caption(f"CARLA 体验仿真 · {cfg['label']} (实验{exp_id}) [ESC 退出]")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    fonts = _make_fonts()
    queue = SSEQueue(client)
    queue.start()
    hist = TelemetryHistory()
    render_ctx = {"params": params, "lanes": lanes}

    started_at = time.time()
    record = {
        "experiment_id": exp_id,
        "label": cfg["label"],
        "slug": slug,
        "params": params,
        "params_file": cfg.get("params_file"),
        "meta": run_meta,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "elapsed": None,
        "status": "running",
        "result": None,
        "log": [],
    }

    print(f"[run] 启动实验 {slug} (id={exp_id})" + ("（含点选起终点）" if run_meta else ""))
    try:
        resp = client.start(exp_id, params)
        record["start_resp"] = resp
    except RelayError as exc:
        print(f"[run] 启动失败: {exc}")
        queue.stop()
        pygame.quit()
        return 1

    finished = False
    final_status = None
    clock = pygame.time.Clock()
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            surfaces, exp, exp_seq = queue.snapshot()
            hist.update(exp_id, exp, exp_seq)

            # 检测完成状态（done/stopped/error）。
            # 注意：实验 4/5/23 正常跑完只推送 {"result": {...}}，不含 status:"done"；
            # 实验 10 跑完推送 status:"done" + report。因此以「带 result/report 键或 status
            # 为结束态」判定（运行中的遥测推送不带 result/report 键）。
            st = exp.get("status") if exp else None
            done_kind = _resolve_done(exp)
            if not finished and done_kind:
                finished = True
                final_status = done_kind
                record["status"] = done_kind
                record["result"] = (exp.get("result") or exp.get("report")
                                    or exp.get("message") or exp)
                record["elapsed"] = round(time.time() - started_at, 1)
                record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    path = save_output(record)
                    print(f"[result] 实验结束 ({st})，结果已保存 → {path}")
                except Exception as exc:
                    print(f"[result] 写入失败: {exc}")

            dispatch_render(exp_id, screen, fonts, surfaces, exp, hist, render_ctx)
            pygame.display.flip()
            clock.tick(30)
    except KeyboardInterrupt:
        # 结束时主动调 stop，避免 relay 残留运行线程/车辆
        if not finished:
            try:
                client.stop(exp_id)
            except RelayError:
                pass
            record["status"] = "stopped"
            record["result"] = record.get("result") or {"message": "user stopped"}
            record["elapsed"] = round(time.time() - started_at, 1)
            record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                path = save_output(record)
                print(f"[result] 已停止并保存 → {path}")
            except Exception as exc:
                print(f"[result] 写入失败: {exc}")
    finally:
        queue.stop()
        pygame.quit()

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="local_runner",
        description="命令行 + pygame 的 CARLA 实验纯体验仿真客户端",
    )
    parser.add_argument("experiment", nargs="?", help="实验 slug：localization / lidar_detection / semantic_segmentation / comprehensive_driving / route_planning")
    parser.add_argument("--params", help="参数 JSON 文件路径（缺省用 server/experiment_params/ 下同名文件）")
    parser.add_argument("--relay", help="relay 服务地址，如 http://127.0.0.1:5000（覆盖参数文件内 relay 字段）")
    parser.add_argument("--list", action="store_true", help="列出可用实验与参数")
    args = parser.parse_args(argv)

    if args.list:
        print("可用实验:")
        for slug, path in DEFAULT_PARAMS.items():
            print(f"  {slug:<24} {path}")
        return 0

    if not args.experiment:
        parser.print_help()
        return 1

    try:
        cfg = _load_config(args.experiment, args.params)
        if args.relay:
            cfg["relay"] = args.relay
    except SystemExit as exc:
        print(str(exc))
        return 1

    # 确保输出目录存在
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _run_experiment(cfg)


if __name__ == "__main__":
    sys.exit(main())