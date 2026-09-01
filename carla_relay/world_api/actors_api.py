"""行人生成与俯瞰视角 API。

本文件由 carla_relay.world_api.load_into(globals()) 载入执行，不可独立 import。
"""
# --- 行人生成 ---

_EXP10_PEDI: list = []  # 保持行人和控制器引用


@app.route("/pedestrian/spawn", methods=["POST"])
def pedestrian_spawn():
    """在自车前方生成一个横穿行人。
    Body: {"vehicle_id": 10, "side": "left"}"""
    data = request.get_json(silent=True) or {}
    vid = data.get("vehicle_id")
    side = data.get("side", "left")
    if vid is None:
        return jsonify({"status": "error", "message": "vehicle_id required"}), 400
    ego = world.get_actor(vid)
    if ego is None or not ego.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404

    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = carla.Vector3D(x=-fwd.y, y=fwd.x, z=0)
    side_off = 6.0 if side == "left" else -6.0
    fwd_off = 15.0
    loc = carla.Location(
        x=ego_tf.location.x + fwd.x * fwd_off + right.x * side_off,
        y=ego_tf.location.y + fwd.y * fwd_off + right.y * side_off,
        z=ego_tf.location.z,
    )

    bp_lib = world.get_blueprint_library()
    walker_bp = bp_lib.find("walker.pedestrian.0001")
    walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
    if walker is None:
        return jsonify({"status": "error", "message": "spawn failed"}), 500

    # Walker AI Controller
    controller_bp = bp_lib.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(loc), attach_to=walker)

    # 让行人横穿
    cross_loc = carla.Location(
        x=ego_tf.location.x + fwd.x * 10 - right.x * side_off,
        y=ego_tf.location.y + fwd.y * 10 - right.y * side_off,
        z=ego_tf.location.z,
    )
    controller.start()
    controller.go_to_location(cross_loc)
    controller.set_max_speed(1.5)

    _EXP10_PEDI.append(walker)
    _EXP10_PEDI.append(controller)
    with _lock:
        _managed_actors.add(walker.id)
        _managed_actors.add(controller.id)

    return jsonify({
        "status": "ok", "walker_id": walker.id, "controller_id": controller.id,
        "spawn_location": {"x": round(loc.x, 1), "y": round(loc.y, 1)},
    })


# --- 鸟瞰视角 ---

@app.route("/vehicle/<int:vid>/spectator/topdown", methods=["POST"])
def spectator_topdown(vid: int):
    """将 spectator 设为自车正上方俯瞰视角。Body: {"height": 50, "pitch": -90}"""
    actor = world.get_actor(vid)
    if actor is None or not actor.is_alive:
        return jsonify({"status": "error", "message": "vehicle not found"}), 404
    data = request.get_json(silent=True) or {}
    height = float(data.get("height", 50))
    pitch = float(data.get("pitch", -90))
    tf = actor.get_transform()
    spec_loc = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + height)
    spec_rot = carla.Rotation(pitch=pitch, yaw=0, roll=0)
    world.get_spectator().set_transform(carla.Transform(spec_loc, spec_rot))
    return jsonify({"status": "ok", "location": {"x": round(spec_loc.x, 1), "y": round(spec_loc.y, 1), "z": round(spec_loc.z, 1)}})

