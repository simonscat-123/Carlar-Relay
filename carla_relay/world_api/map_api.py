"""地图 API：出生点 / 边界 / 路网 / 渲染。

本文件由 carla_relay.world_api.load_into(globals()) 载入执行，不可独立 import。
"""
@app.route("/map/spawn_points")
def map_spawn_points():
    """返回地图所有生成点（用于前端点选起终点）"""
    spawn_pts = world.get_map().get_spawn_points()
    pts = []
    for i, t in enumerate(spawn_pts):
        pts.append({
            "idx": i,
            "x": round(t.location.x, 1), "y": round(t.location.y, 1), "z": round(t.location.z, 1),
            "yaw": round(t.rotation.yaw, 1),
        })
    return jsonify({"spawn_points": pts, "count": len(pts), "map": world.get_map().name})


@app.route("/map/bounds")
def map_bounds():
    """返回地图大致边界（用于前端 Canvas 坐标映射）"""
    spawn_pts = world.get_map().get_spawn_points()
    if not spawn_pts:
        return jsonify({"status": "error", "message": "no spawn points"}), 500
    xs = [t.location.x for t in spawn_pts]
    ys = [t.location.y for t in spawn_pts]
    return jsonify({
        "min_x": round(min(xs) - 50, 1), "max_x": round(max(xs) + 50, 1),
        "min_y": round(min(ys) - 50, 1), "max_y": round(max(ys) + 50, 1),
        "map": world.get_map().name,
    })


@app.route("/map/road_network")
def map_road_network():
    """返回地图真实道路拓扑折线（含车道元数据）：沿每条道路的车道中心线采样，
    供前端鸟瞰画布绘制真实 CARLA 地图与车道边界线。
    返回 {"roads": [[{x,y},...], ...]（兼容旧格式）,
          "lanes": [{"pts": [...], "road": int, "lane": int, "width": float, "marking": str}, ...],
          "bounds": {...}, "map": name}"""
    m = world.get_map()
    lanes = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    # get_topology() 返回 (start_waypoint, end_waypoint) 的道路线段集合（每条即一个车道）
    for w_start, w_end in m.get_topology():
        pts = []
        wp = w_start
        try:
            marking = w_start.lane_marking_type_left.name   # Broken / Solid / ...
        except Exception:
            marking = "None"
        _guard = 0
        while wp is not None and _guard < 5000:
            loc = wp.transform.location
            pts.append({"x": round(loc.x, 1), "y": round(loc.y, 1)})
            if loc.x < min_x: min_x = loc.x
            if loc.x > max_x: max_x = loc.x
            if loc.y < min_y: min_y = loc.y
            if loc.y > max_y: max_y = loc.y
            _guard += 1
            if _guard > 1 and wp.transform.location.distance(w_end.transform.location) < 2.0:
                break
            # 沿道路前进（取最接近 end 的一个后续航点）
            nxt = wp.next(4.0)
            if not nxt:
                break
            nxt = sorted(nxt, key=lambda w: w.transform.location.distance(w_end.transform.location))
            # 避免来回折返
            if _guard > 2 and len(pts) > 2:
                last = pts[-2]
                if abs(nxt[0].transform.location.x - last["x"]) < 1.0 and abs(nxt[0].transform.location.y - last["y"]) < 1.0:
                    break
            wp = nxt[0]
        if len(pts) >= 2:
            lanes.append({
                "pts": pts,
                "road": w_start.road_id,
                "lane": w_start.lane_id,
                "width": round(float(w_start.lane_width), 1),
                "marking": marking,
            })

    if not lanes:
        return jsonify({"status": "error", "message": "no road network"}), 500

    pad = 40.0
    return jsonify({
        "roads": [l["pts"] for l in lanes],
        "lanes": lanes,
        "map": m.name,
        "bounds": {
            "min_x": round(min_x - pad, 1), "max_x": round(max_x + pad, 1),
            "min_y": round(min_y - pad, 1), "max_y": round(max_y + pad, 1),
        },
    })


@app.route("/map/render")
def map_render():
    """一次性渲染整个 CARLA 城镇的高空俯视截图，作为规划阶段的鸟瞰底图。

    用一只临时 RGB 相机悬于地图正上方，竖直向下拍摄（pitch=-90），
    采集一帧后编码为 JPEG 返回给前端绘制在画布底层。
    返回 {"base64": "...", "width": ..., "height": ...}"""
    m = world.get_map()
    # 1. 计算地图边界（取道路拓扑 + 生成点的包络）
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for w_start, w_end in m.get_topology():
        for lp in (w_start.transform.location, w_end.transform.location):
            if lp.x < min_x: min_x = lp.x
            if lp.x > max_x: max_x = lp.x
            if lp.y < min_y: min_y = lp.y
            if lp.y > max_y: max_y = lp.y
    for sp in m.get_spawn_points():
        l = sp.location
        if l.x < min_x: min_x = l.x
        if l.x > max_x: max_x = l.x
        if l.y < min_y: min_y = l.y
        if l.y > max_y: max_y = l.y
    if min_x == float("inf"):
        return jsonify({"status": "error", "message": "empty map bounds"}), 500

    pad = 30.0

    # 2. 俯拍相机：竖直朝下（pitch=-90）覆盖整图
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_w = (max_x - min_x) / 2 + pad
    half_h = (max_y - min_y) / 2 + pad
    fov = 90.0
    alt = max(half_w, half_h) / math.tan(math.radians(fov / 2))
    alt = max(alt, 25.0)

    img_size = 1280

    # 采集单帧并销毁相机
    def _snap(cam):
        done = threading.Event()
        captured = {}
        try:
            def _on_image(snapshot):
                try:
                    arr = np.frombuffer(snapshot.raw_data, dtype=np.uint8) \
                        .reshape((snapshot.height, snapshot.width, 4))
                    img = PIL.Image.fromarray(arr[:, :, :3][:, :, ::-1].copy())  # BGRA → RGB（copy 消除负步长只读视图，避免 fromarray 访问违规）
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    captured["b64"] = base64.b64encode(buf.getvalue()).decode()
                    captured["w"] = snapshot.width
                    captured["h"] = snapshot.height
                finally:
                    done.set()
            cam.listen(_on_image)
            # 同步模式下相机只有在 world.tick() 时才会出帧；规划阶段无人 tick，
            # 若世界残留同步模式会导致抓帧超时(500)。等待期间按需循环 tick 驱动渲染。
            is_sync = world.get_settings().synchronous_mode
            deadline = time.time() + 8.0
            while not done.is_set() and time.time() < deadline:
                if is_sync:
                    try:
                        world.tick()
                    except Exception:
                        pass
                    time.sleep(0.001)
                time.sleep(0.002)
            if not done.is_set() or "b64" not in captured:
                return None
            return captured
        finally:
            cam.destroy()

    cam_tform = carla.Transform(
        carla.Location(x=cx, y=cy, z=alt),
        carla.Rotation(pitch=-90, yaw=-90, roll=0),
    )

    # 2a. 优先：正射宽角相机（orthographic 平行投影，做 BEV 鸟瞰最合适）
    #     需 CARLA >= 0.9.14 且存在 wide_angle_lens 蓝图 + camera_model 属性；
    #     若当前实例不支持，则回退普通透视相机。
    ortho_cam = None
    try:
        wbp = world.get_blueprint_library().find("sensor.camera.rgb.wide_angle_lens")
        has_model = any(a.id == "camera_model" for a in wbp)
        if has_model:
            wbp.set_attribute("camera_model", "orthographic")
            wbp.set_attribute("image_size_x", str(img_size))
            wbp.set_attribute("image_size_y", str(img_size))
            wbp.set_attribute("fov", str(int(fov)))
            ortho_cam = world.spawn_actor(wbp, cam_tform)
    except Exception:
        ortho_cam = None

    if ortho_cam is not None:
        captured = _snap(ortho_cam)
        proj_mode = "orthographic"
        proj_geo = None
    else:
        # 2b. 回退：普通透视相机覆盖整图
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(img_size))
        bp.set_attribute("image_size_y", str(img_size))
        bp.set_attribute("fov", str(int(fov)))
        bp.set_attribute("lens_circle_multiplier", "0.0")
        cam = world.spawn_actor(bp, cam_tform)
        if cam is None:
            return jsonify({"status": "error", "message": "spawn camera failed"}), 500
        proj_mode = "perspective"

        # 计算地面平面 (z=0) 的投影单应，前端用它做 屏幕<->世界 双向精确转换，
        # 从而让点选坐标与绘制的路线和透视底图严格对齐。
        proj_geo = None
        focal = img_size / (2.0 * math.tan(math.radians(fov) / 2.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = img_size / 2.0
        K[1, 2] = img_size / 2.0
        # 按 CARLA 源码 Transform::GetMatrix 公式，用我们 spawn 用的 cam_tform 直接构造
        # camera->world 矩阵再求逆，避免依赖新出生相机的 get_transform()（可能未同步位姿）。
        _yaw = math.radians(cam_tform.rotation.yaw)
        _pitch = math.radians(cam_tform.rotation.pitch)
        _roll = math.radians(cam_tform.rotation.roll)
        _cy, _sy = math.cos(_yaw), math.sin(_yaw)
        _cr, _sr = math.cos(_roll), math.sin(_roll)
        _cp, _sp = math.cos(_pitch), math.sin(_pitch)
        _lx = cam_tform.location.x
        _ly = cam_tform.location.y
        _lz = cam_tform.location.z
        M_c2w = np.array([
            [_cp * _cy, _cy * _sp * _sr - _sy * _cr, -_cy * _sp * _cr - _sy * _sr, _lx],
            [_cp * _sy, _sy * _sp * _sr + _cy * _cr, -_sy * _sp * _cr + _cy * _sr, _ly],
            [_sp, -_cp * _sr, _cp * _cr, _lz],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        M_w2c = np.linalg.inv(M_c2w)
        try:
            def _proj_ground(x, y):
                # 照官方 lidar_to_camera：UE 相机坐标 (x,y,z)->(y,-z,x)
                col = M_w2c.dot(np.array([x, y, 0.0, 1.0]))
                px, py, pz = col[1], -col[2], col[0]
                if pz <= 0.0:
                    return None
                u = (K[0, 0] * px + K[0, 2] * pz) / pz
                v = (K[1, 1] * py + K[1, 2] * pz) / pz
                return u, v
            corners = [
                (min_x, min_y), (max_x, min_y),
                (max_x, max_y), (min_x, max_y),
            ]
            A, b = [], []
            for x, y in corners:
                uv = _proj_ground(x, y)
                if uv is None:
                    raise ValueError("corner behind camera")
                u, v = uv
                A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
                A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
                b += [u, v]
            h = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
            H = np.append(h, 1.0).reshape(3, 3)
            Hinv = np.linalg.inv(H)
            proj_geo = (H, Hinv)
        except Exception as _e:
            print(f"[map/render] homography 失败: {_e!r}")
            proj_geo = None

        captured = _snap(cam)

    if not captured:
        return jsonify({"status": "error", "message": "capture timeout"}), 500
    print(f"[map/render] 投影模式 = {proj_mode} (alt={alt:.0f}m fov={fov:.0f} homography={'ok' if proj_geo else 'none'})")
    resp = {
        "status": "ok",
        "map": m.name,
        "mode": proj_mode,
        "base64": captured["b64"],
        "width": captured["w"],
        "height": captured["h"],
    }
    if proj_geo is not None:
        resp["proj"] = {
            "H": proj_geo[0].flatten().tolist(),
            "Hinv": proj_geo[1].flatten().tolist(),
        }
    return jsonify(resp)

