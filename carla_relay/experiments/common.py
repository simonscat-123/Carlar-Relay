"""实验共享状态：manifest 加载 / 日志 / 状态与输出路由。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验共享状态 (所有实验均在此 relay 进程内运行)
# =============================================================================

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "courses" / "intelligent_driving" / "experiments"
_MANIFEST_PATH = _EXPERIMENTS_DIR / "manifest.json"
_EXP_LOCK = threading.Lock()
_EXP_CURRENT_ID = None
_EXP_LOG = []

_EXP_MANIFEST = None


def _load_manifest():
    global _EXP_MANIFEST
    if _EXP_MANIFEST is not None:
        return _EXP_MANIFEST
    if _MANIFEST_PATH.exists():
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
                _EXP_MANIFEST = json.load(f)
            print(f"[EXP] 已加载 manifest: {len(_EXP_MANIFEST.get('experiments', []))} 个实验")
            return _EXP_MANIFEST
        except Exception as e:
            print(f"[EXP] manifest 加载失败: {e}")
    _EXP_MANIFEST = {"experiments": []}
    return _EXP_MANIFEST


def _get_exp_entry(exp_id):
    for entry in _load_manifest().get("experiments", []):
        if entry.get("id") == exp_id:
            return entry
    return None


# 实验日志落盘路径（server 目录下）
_EXP_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.log")


def _write_log_file(line: str):
    """落盘到 experiments.log；失败时打印原因，不静默吞掉。"""
    try:
        with open(_EXP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[LOG-WRITE-ERROR] 写入 {_EXP_LOG_PATH} 失败: {e!r}")


# 模块加载即写一行，用于确认：只要运行的是本文件，日志文件必然立即生成
_write_log_file("=== CARLA relay 模块加载 === pid=%s" % (os.getpid(),))


def _exp_log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _EXP_LOG.append(line)
    if len(_EXP_LOG) > 200:
        _EXP_LOG[:] = _EXP_LOG[-200:]
    # 仅保留日志逻辑（内存 + 落盘 + SSE 推送到前端日志面板），不再打印到控制台
    _write_log_file(line)
    _push_to_sse({"experiment": {"id": _EXP_CURRENT_ID, "log": line}})


def _any_experiment_running():
    return any([
        _EXP01_RUNNING, _EXP02_RUNNING, _EXP03_RUNNING,
        _EXP04_RUNNING, _EXP05_RUNNING, _EXP06_RUNNING,
        _EXP07_RUNNING, _EXP08_RUNNING, _EXP09_RUNNING,
        _EXP10_RUNNING, _EXP23_RUNNING,
    ])


def _sweep_stale_actors(owner_label="experiment"):
    """实验线程体开头调用：销毁上一实验残留的托管 actor，保证干净起点。

    背景：实验异常收尾（中途停止时 CARLA 短暂无响应等）可能泄漏车辆/传感器，
    残留车停在出生点或路线上会导致新实验首跑被撞偏（如定位实验起步右拐），
    残留 GNSS/IMU 还会持续回调浪费带宽。start handler 已保证旧线程退出后才
    放行新实验，故此处扫到的存活 actor 必为泄漏物，可安全销毁。
    返回销毁数量。
    """
    with _lock:
        stale_ids = list(_managed_actors)
    destroyed = 0
    for aid in stale_ids:
        try:
            actor = world.get_actor(aid)
            if actor is not None and actor.is_alive:
                if getattr(actor, "type_id", "").startswith("vehicle."):
                    try:
                        actor.set_autopilot(False)
                    except Exception:
                        pass
                actor.destroy()
                destroyed += 1
        except Exception:
            pass
        with _lock:
            _managed_actors.discard(aid)
        _sensor_frames.pop(aid, None)
        _sensor_dtype.pop(aid, None)
        _sensor_refs.pop(aid, None)
    if destroyed:
        _exp_log(f"已清理上一实验残留 actor ×{destroyed}")
        # 残留传感器可能仍在 _stream_* 槽位上挂着旧 sid（int）或 actor 引用，一并复位避免串流
        for slot in ("_stream_camera", "_stream_camera_left", "_stream_camera_right",
                     "_stream_semantic", "_stream_bird", "_stream_bbox",
                     "_stream_vehicle", "_camera_actor_ref"):
            val = globals().get(slot)
            if val in stale_ids or getattr(val, "id", None) in stale_ids:
                globals()[slot] = None
    return destroyed


@app.route("/experiments")
def list_experiments():
    return jsonify(_load_manifest())


@app.route("/experiments/<int:exp_id>")
def get_experiment(exp_id):
    entry = _get_exp_entry(exp_id)
    if entry is None:
        return jsonify({"status": "error", "message": f"无效实验ID: {exp_id}"}), 404
    return jsonify(entry)


@app.route("/experiment/status")
def experiment_status():
    running = _any_experiment_running()
    with _EXP_LOCK:
        return jsonify({
            "running": running,
            "experiment_id": _EXP_CURRENT_ID if running else None,
            "logs": _EXP_LOG[-50:],
        })


@app.route("/experiment/outputs")
def experiment_outputs():
    output_dir = _EXPERIMENTS_DIR.parent / "outputs" / "experiments"
    if not output_dir.exists():
        return jsonify({"files": []})
    files = []
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            files.append({
                "name": f.name,
                "path": str(f.relative_to(output_dir)),
                "size": f.stat().st_size,
            })
    return jsonify({"files": files})

