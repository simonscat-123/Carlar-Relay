"""GNSS/INS 简化组合导航（API 实验ID 3，已并入定位分析）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 3: GNSS/INS 简化组合导航（离线滤波）
# =============================================================================

_EXP03_RUNNING = False
_EXP03_ABORT = False


def _run_exp03(args):
    """离线实验：读取 exp02 CSV，运行互补滤波，输出新的 CSV"""
    global _EXP03_RUNNING, _EXP03_ABORT, _EXP_CURRENT_ID
    _EXP_CURRENT_ID = 3
    _EXP03_RUNNING = True
    _EXP03_ABORT = False
    _EXP_LOG.clear()
    _exp_log("实验3 (GNSS/INS 组合导航) 启动 — 离线处理")

    alpha = float(args.get("alpha", 0.9))
    gnss_noise = float(args.get("gnss_noise", 0.5))
    ins_noise = float(args.get("ins_noise", 0.1))
    seed = int(args.get("seed", 7))

    try:
        # 读取 exp02 CSV（位于 experiments 目录下，由 exp02 子进程生成或默认路径）
        csv_path = _EXPERIMENTS_DIR / "exp02_gnss_imu_log.csv"
        if not csv_path.exists():
            raise RuntimeError(f"找不到 exp02 输出文件: {csv_path}\n请先运行实验2")
        _exp_log(f"读取输入: {csv_path.name}")

        rows_in = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_in.append(row)
        _exp_log(f"输入 {len(rows_in)} 行")

        if len(rows_in) < 2:
            raise RuntimeError("输入数据不足（至少需要2行）")

        rng = random.Random(seed)

        # 第一行作为原点
        r0 = rows_in[0]
        lat0, lon0, alt0 = float(r0["latitude"]), float(r0["longitude"]), float(r0["altitude"])
        gt0_x, gt0_y = float(r0["gt_x"]), float(r0["gt_y"])

        # ENU 转换参数（WGS84 近似）
        RAD = math.pi / 180.0
        lat0r = lat0 * RAD
        meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat0r) + 1.175 * math.cos(4 * lat0r)
        meters_per_deg_lon = 111412.84 * math.cos(lat0r) - 93.5 * math.cos(3 * lat0r)

        prev_gt = None
        fused_x, fused_y, ins_vx, ins_vy = gt0_x, gt0_y, 0.0, 0.0
        rows_out = []

        for idx, row in enumerate(rows_in):
            if _EXP03_ABORT:
                break

            t = float(row["time"])
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
            gt_yaw = float(row["gt_yaw"])
            lat, lon = float(row["latitude"]), float(row["longitude"])

            # GNSS → 局部 ENU
            gnss_x = gt0_x + (lon - lon0) * meters_per_deg_lon
            gnss_y = gt0_y + (lat - lat0) * meters_per_deg_lat
            gnss_x_raw = gnss_x
            gnss_y_raw = gnss_y
            gnss_x += rng.gauss(0, gnss_noise)
            gnss_y += rng.gauss(0, gnss_noise)

            # 局部 GT
            gt_x_local = gt_x - gt0_x
            gt_y_local = gt_y - gt0_y

            # INS 航位推算
            ins_ax = 0.0
            ins_ay = 0.0
            if idx > 0:
                dt = t - float(rows_in[idx - 1]["time"])
                if dt > 0 and prev_gt is not None:
                    ax = float(row["accel_x"]) + rng.gauss(0, ins_noise)
                    ay = float(row["accel_y"]) + rng.gauss(0, ins_noise)
                    ins_ax = ax * math.cos(gt_yaw * RAD) - ay * math.sin(gt_yaw * RAD)
                    ins_ay = ax * math.sin(gt_yaw * RAD) + ay * math.cos(gt_yaw * RAD)
                ins_vx += ins_ax * dt
                ins_vy += ins_ay * dt
                fused_x += ins_vx * dt
                fused_y += ins_vy * dt

            # 互补滤波
            fused_x = (1 - alpha) * fused_x + alpha * gnss_x
            fused_y = (1 - alpha) * fused_y + alpha * gnss_y

            rows_out.append({
                "time": t, "fused_x": round(fused_x, 3), "fused_y": round(fused_y, 3),
                "gnss_x": round(gnss_x, 3), "gnss_y": round(gnss_y, 3),
                "gnss_x_raw": round(gnss_x_raw, 3), "gnss_y_raw": round(gnss_y_raw, 3),
                "ins_accel_x": round(ins_ax, 4), "ins_accel_y": round(ins_ay, 4),
                "gt_x_local": round(gt_x_local, 3), "gt_y_local": round(gt_y_local, 3),
            })

            if idx % 20 == 0:
                _push_to_sse({"experiment": {"id": 3, "trajectory": {
                    "index": idx, "t": round(t, 3), "fused_x": round(fused_x, 3),
                    "fused_y": round(fused_y, 3), "gnss_x": round(gnss_x, 3),
                    "progress": round((idx + 1) / len(rows_in) * 100, 1)}}})

            prev_gt = (gt_x, gt_y)

        # 计算 RMSE
        if rows_out:
            errs = [math.sqrt((r["fused_x"] - gi["gt_x"] + gt0_x) ** 2 + (r["fused_y"] - gi["gt_y"] + gt0_y) ** 2)
                     for r, gi in zip(rows_out, [{"gt_x": float(x["gt_x"]), "gt_y": float(x["gt_y"])} for x in rows_in])]
            rmse = math.sqrt(sum(e ** 2 for e in errs) / len(errs))
            _exp_log(f"RMSE = {rmse:.3f} m")

        # 写输出
        out_path = _EXPERIMENTS_DIR / "exp03_gnss_ins_filter.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            w.writeheader()
            w.writerows(rows_out)
        _exp_log(f"输出: {out_path.name} ({len(rows_out)} 行)")

        elapsed = rows_out[-1]["time"] - rows_out[0]["time"] if rows_out else 0
        _push_to_sse({"experiment": {"id": 3, "result": {"elapsed": round(elapsed, 1), "rows": len(rows_out), "rmse": round(rmse, 3)}}})
        _exp_log("实验3 完成")
    except Exception as e:
        _exp_log(f"实验3 错误: {e}")
    finally:
        _EXP03_RUNNING = False
        if _EXP03_ABORT:
            _push_to_sse({"experiment": {"id": 3, "status": "stopped"}})


@app.route("/experiment/3/start", methods=["POST"])
def experiment_3_start():
    global _EXP03_RUNNING
    if _EXP03_RUNNING:
        return jsonify({"status": "error", "message": "实验3 已在运行"}), 409
    args = request.get_json(silent=True) or {}
    threading.Thread(target=_run_exp03, args=(args,), daemon=True).start()
    return jsonify({"status": "ok", "experiment_id": 3, "message": "实验3 已启动"})


@app.route("/experiment/3/stop", methods=["POST"])
def experiment_3_stop():
    global _EXP03_ABORT
    _EXP03_ABORT = True
    return jsonify({"status": "ok", "message": "实验3 停止请求已发送"})

