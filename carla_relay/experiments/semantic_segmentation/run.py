"""语义分割 · 前端关卡 semantic-segmentation（API 实验ID 5）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 5: 视觉与语义分割
# =============================================================================

# 感知融合 → BEV 占据栅格世界 → 基于栅格的规划（真实模块，绝对路径导入）
from carla_relay.experiments.semantic_segmentation.fusion import perceive as _exp5_perceive
from carla_relay.experiments.semantic_segmentation.bev import BevBuilder as _exp5_BevBuilder
from carla_relay.experiments.semantic_segmentation.grid_planner import GridPlanner as _exp5_GridPlanner
from carla_relay.experiments.comprehensive_driving.viz import render_perceived_frame as _exp5_render_perceived

_EXP05_RUNNING = False
_EXP05_ABORT = False
_EXP05_THREAD = None
_EXP05_LOCK = threading.Lock()


def _run_exp05(args):
    global _EXP05_RUNNING, _EXP05_ABORT, _EXP_CURRENT_ID, _stream_camera, _stream_vehicle, _stream_semantic, _semantic_level
    _EXP_CURRENT_ID = 5
    _EXP05_RUNNING = True
    _EXP05_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验5 (语义分割) 启动")
    _sweep_stale_actors()
    _exp_log("正在配置仿真环境（同步模式 + 交通管理）…")

    duration = float(args.get("duration", 20.0))
    fixed_delta = float(args.get("fixed_delta", 0.05))
    seed = int(args.get("seed", 7))
    level = str(args.get("level", "L2")).upper()
    if level not in _SEMANTIC_LEVELS:
        level = "L2"
    _semantic_level = level
    # 感知融合 → BEV 构建参数（task 参数可覆盖）
    perc_range = float(args.get("perception_range", 50.0))
    bev_span = float(args.get("bev_span", 60.0))
    bev_res = float(args.get("bev_res", 0.5))

    cam_fov = 90.0   # 语义/实例相机 FOV（度）
    cam_pitch = -5.0  # 相机安装俯仰（负值向下，供地面逆投影 / 单目测距）
    cam_height = 1.7  # 相机离地高度（m）

    _exp5_rgb_raw = {}  # 前相机原始 BGRA（供检测框叠加）

    # 感知融合 / BEV 世界 / 基于栅格的规划器（每 tick 复用）
    bev = _exp5_BevBuilder(span=bev_span, res=bev_res, cam_fov=cam_fov,
                           cam_pitch=cam_pitch, cam_height=cam_height,
                           perc_range=perc_range, subsample=2)
    planner = _exp5_GridPlanner()

    actors = []
    try:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=fixed_delta))
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        bp = world.get_blueprint_library().filter("vehicle.*")[0]
        rng = random.Random(seed)
        _exp_log("正在生成主车…")
        vehicle = world.spawn_actor(bp, rng.choice(world.get_map().get_spawn_points()))
        actors.append(vehicle)
        v_id = vehicle.id

        _exp_log("正在挂载相机（RGB / 语义 / 实例）…")
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(cam)
        _sensor_refs[cam.id] = cam
        cam.listen(lambda d, sid=cam.id: _sensor_callback(sid, "camera", d) or _exp5_rgb_raw.__setitem__(sid, bytes(d.raw_data)))
        _stream_camera = cam.id
        _stream_vehicle = v_id

        sem_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", "800")
        sem_bp.set_attribute("image_size_y", "600")
        sem = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(sem)
        _sensor_refs[sem.id] = sem
        sem.listen(lambda d, sid=sem.id: _sensor_callback(sid, "semantic", d))
        _stream_semantic = sem.id

        ins_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        ins_bp.set_attribute("image_size_x", "800")
        ins_bp.set_attribute("image_size_y", "600")
        ins = world.spawn_actor(ins_bp, carla.Transform(carla.Location(x=1.5, z=cam_height), carla.Rotation(pitch=cam_pitch)), attach_to=vehicle)
        actors.append(ins)
        _sensor_refs[ins.id] = ins
        ins.listen(lambda d, sid=ins.id: _sensor_callback(sid, "instance", d))

        for a in actors:
            with _lock:
                _managed_actors.add(a.id)
        _exp_log(f"车辆+RGB+语义+实例相机已挂载 (id={v_id})")

        _exp_log("渲染预热中（首帧着色器编译）…")
        settle_ticks = int(1.0 / fixed_delta)
        for _ in range(settle_ticks):
            if _EXP05_ABORT:
                raise RuntimeError("已中止")
            world.tick()

        vehicle.set_autopilot(True, tm.get_port())
        _exp_log("自动驾驶已启用，开始采集")

        total = int(duration / fixed_delta)
        rows = []
        for i in range(total):
            if _EXP05_ABORT:
                break
            world.tick()
            snap = world.get_snapshot()
            t = snap.timestamp.elapsed_seconds

            level = _semantic_level  # 当前等级（可被 /experiment/5/level 实时切换）

            # 语义标签图（CityScapes）
            sem_labels = None
            if sem.id in _semantic_raw:
                sem_labels = np.frombuffer(_semantic_raw[sem.id], dtype=np.uint8).reshape((600, 800, 4))[:, :, 2].astype(np.int32)

            # 实例分割：R=语义ID, G=actor低字节, B=actor高字节
            instance = None
            if ins.id in _instance_raw:
                arr = np.frombuffer(_instance_raw[ins.id], dtype=np.uint8).reshape((600, 800, 4))
                sem_ids = arr[:, :, 2].astype(np.int32)
                actor_ids = arr[:, :, 1].astype(np.uint16) + (arr[:, :, 0].astype(np.uint16) << 8)
                instance = (sem_ids, actor_ids)

            # ── 感知融合 → BEV 占据栅格世界 → 基于栅格的规划 ──
            targets = []
            bev_payload = None
            grid_stat = None
            decision = None
            if ins.id in _instance_raw and sem.id in _semantic_raw:
                try:
                    ih = int(ins.attributes["image_size_y"]); iw = int(ins.attributes["image_size_x"])
                    sh = int(sem.attributes["image_size_y"]); sw = int(sem.attributes["image_size_x"])
                    inst_arr = np.frombuffer(_instance_raw[ins.id], dtype=np.uint8).reshape((ih, iw, 4))
                    sem_arr = np.frombuffer(_semantic_raw[sem.id], dtype=np.uint8).reshape((sh, sw, 4))
                    targets = _exp5_perceive(inst_arr, sem_arr, cam_fov=cam_fov,
                                             cam_pitch=cam_pitch, cam_height=cam_height,
                                             perc_range=perc_range, exclude_ids=(vehicle.id,))
                    cells, grid_stat, bev_payload = bev.build(sem_labels, targets,
                                                              sem_h=sh, sem_w=sw)
                    decision = planner.decide(cells, res=bev.res)
                    # 前视 RGB 叠加检测框（复用综合驾驶 viz 渲染；失败不影响主流程）
                    if cam.id in _exp5_rgb_raw:
                        rcw, rch = int(cam.attributes["image_size_x"]), int(cam.attributes["image_size_y"])
                        _rgb_arr = np.frombuffer(_exp5_rgb_raw[cam.id], dtype=np.uint8).reshape((rch, rcw, 4))
                        _sensor_frames[cam.id] = _exp5_render_perceived(_rgb_arr, targets, iw, ih)
                        _sensor_frame_num[cam.id] = _sensor_frame_num.get(cam.id, 0) + 1
                except Exception as _exp5e:
                    _exp_log(f"感知融合异常: {_exp5e!r}")

            ratios = {}
            if sem_labels is not None:
                labeled = _label_semantic_level(sem_labels, level, instance, world)
                ratios = _ratios_from_labels(labeled, level)
                rgb = _colors_from_labels(labeled, level)
                img = PIL.Image.fromarray(rgb, mode="RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                _sensor_frames[sem.id] = buf.getvalue()
                _sensor_frame_num[sem.id] = i + 1

            rows.append({"frame": i + 1, "time": t, "level": level, **ratios})
            if i % 4 == 0:
                pt = {"frame": i + 1, "t": round(t, 3), "progress": round((i + 1) / total * 100, 1), "level": level}
                pt.update({k + "_ratio": v for k, v in ratios.items()})
                pt["targets"] = len(targets)
                pt["grid_stat"] = grid_stat
                pt["bev"] = bev_payload
                if decision is not None:
                    pt["decision"] = {
                        "state": decision["state"], "label": decision["label"],
                        "target_lat": decision["target_lat"], "block_m": decision["block_m"],
                        "reason": decision["reason"], "traj": decision["traj"],
                    }
                _push_to_sse({"experiment": {"id": 5, "trajectory": pt}})

        _push_to_sse({"experiment": {"id": 5, "result": {"elapsed": round(t if rows else 0, 1), "rows": len(rows)}}})
        _exp_log(f"实验5 完成 — {len(rows)} 行")
    except Exception as e:
        _exp_log(f"实验5 错误: {e}")
    finally:
        _stream_vehicle = None
        _stream_camera = None
        _stream_semantic = None

        # 先 stop 传感器以排空数据流（与实验4 相同的收尾顺序）
        for _a in actors:
            try:
                if _a is not None and getattr(_a, "is_listening", False):
                    _a.stop()
            except Exception:
                pass

        # 恢复世界为异步模式 + TM 异步 + tick，重置到 _init_carla 的基线，
        # 避免实验结束后仿真静止、前端画面永久定格。
        try:
            world.apply_settings(carla.WorldSettings(synchronous_mode=False))
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            for _i in range(5):
                world.tick()
        except Exception:
            pass

        # 销毁前先取消自动驾驶，避免销毁 TM 托管车辆时原生崩溃
        for _a in actors:
            try:
                if _a is not None and getattr(_a, "type_id", "").startswith("vehicle."):
                    _a.set_autopilot(False)
            except Exception:
                pass

        # 销毁本实验生成的所有 actors（自行销毁，避免反复重启累积相机导致渲染变卡）
        for _a in actors:
            try:
                if _a is not None and _a.is_alive:
                    _a.destroy()
            except Exception:
                pass
            with _lock:
                _managed_actors.discard(_a.id)
            _sensor_frames.pop(_a.id, None)
            _sensor_dtype.pop(_a.id, None)
            _sensor_refs.pop(_a.id, None)
            _semantic_raw.pop(_a.id, None)
            _instance_raw.pop(_a.id, None)

        _EXP05_RUNNING = False
        if _EXP05_ABORT:
            _push_to_sse({"experiment": {"id": 5, "status": "stopped"}})


@app.route("/experiment/5/start", methods=["POST"])
def experiment_5_start():
    global _EXP05_THREAD, _EXP05_RUNNING, _EXP05_ABORT
    with _EXP05_LOCK:
        if _EXP05_THREAD is not None and _EXP05_THREAD.is_alive():
            if not _EXP05_ABORT:
                # 正在正常运行：直接拒绝。重试/双击产生的重复 start 在此被挡，
                # 绝不 kill-重启实验
                return jsonify({"status": "error", "message": "实验5 已在运行"}), 409
            # 正在停止/收尾：等旧线程完整退出后接管（保留「停止→快速重启」体验）
            _EXP05_THREAD.join(timeout=15.0)
            if _EXP05_THREAD.is_alive():
                return jsonify({"status": "error", "message": "实验5 旧线程仍在收尾，请稍后重试"}), 409
        # RUNNING/ABORT 在持锁的 handler 内提前置位：
        # ① 关闭 /cleanup、/preview/start 守卫在「start 已返回、线程体尚未执行」间的穿透窗口；
        # ② 消除「新线程刚起、ABORT 仍残留旧 stop 的 true」导致后续 start 误判收尾中的窗口
        _EXP05_RUNNING = True
        _EXP05_ABORT = False
        args = request.get_json(silent=True) or {}
        try:
            _EXP05_THREAD = threading.Thread(target=_run_exp05, args=(args,), daemon=True)
            _EXP05_THREAD.start()
        except Exception as exc:
            _EXP05_RUNNING = False
            return jsonify({"status": "error", "message": f"实验5 线程启动失败: {exc}"}), 500
    return jsonify({"status": "ok", "experiment_id": 5, "message": "实验5 已启动"})


@app.route("/experiment/5/stop", methods=["POST"])
def experiment_5_stop():
    global _EXP05_ABORT
    _EXP05_ABORT = True
    return jsonify({"status": "ok", "message": "实验5 停止请求已发送"})


@app.route("/experiment/5/level", methods=["POST"])
def experiment_5_level():
    global _semantic_level
    args = request.get_json(silent=True) or {}
    level = str(args.get("level", "")).upper()
    if level not in _SEMANTIC_LEVELS:
        return jsonify({"status": "error", "message": f"未知等级: {level}", "current": _semantic_level}), 400
    _semantic_level = level
    return jsonify({"status": "ok", "level": _semantic_level, "message": f"已切换到 {level} 语义分割"})

