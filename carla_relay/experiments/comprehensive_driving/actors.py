"""实验10 实验台具：障碍物生成/清理、横穿行人、信号灯时长设置、路线障碍绑定。

不属于自动驾驶算法闭环（感知-规划-控制），而是实验环境编排：

  - spawn_obstacle_ahead：前端"生成障碍物"——在 ego 沿路线前方 distance 米处
    生成静止障碍车（净距语义：distance 为两车包围盒之间的边到边距离）；
  - spawn_route_obstacles / clear_obstacles / refresh_route_obstacles：
    规划期选定的路线障碍物的生成 / 世界清理 / 跨实验沿用与补齐
    （障碍与路线绑定：只有重新规划才清空重建，同一路线重复运行沿用）；
  - spawn_crossing_pedestrian：横穿行人（AI 控制器驱动）；
  - set_traffic_light_timing：统一设置全图信号灯时长，压缩等待。

所有函数必须在 exp10 主循环线程内调用（CARLA 客户端非线程安全，
跨线程并发会崩溃）。平台服务（日志/actor 管理）经参数注入。
"""
from __future__ import annotations

import math
import random

import carla
import numpy as np


def spawn_obstacle_ahead(world, ego, route, distance, managed_actors, lock):
    """在 ego 沿路线前方 distance 米处生成静止障碍车辆。
    返回 {"ok", id?, type?, distance, on_route, location?} / {"ok": False, message}。"""
    bps = world.get_blueprint_library().filter("vehicle.*")
    bp = np.random.choice(bps)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "obstacle")
    if bp.has_attribute("color"):
        bp.set_attribute("color", np.random.choice(bp.get_attribute("color").recommended_values))

    # ---- 净距语义：distance 表示"自车与障碍物外侧包围盒之间"的净距(edge-to-edge)。----
    # 不能直接把障碍中心放到"自车中心前方 distance"处——那只是中心距，两车仍会紧贴/重叠。
    # 因此障碍中心需再前进一段"两车半长之和"，使两盒之间的净距恰为 distance。
    ego_half = ego.bounding_box.extent.x          # 自车半长（x 为本地前进轴）
    if bp.has_attribute("length"):
        obs_half = 0.5 * float(bp.get_attribute("length").as_float())  # 障碍半长
    else:
        obs_half = ego_half
    center_offset = distance + ego_half + obs_half  # 障碍中心沿路距自车中心的距离

    ego_loc = ego.get_location()
    if len(route) >= 2:
        # ---- 沿导航路线推进：先计算路线各路点的累计弧长，再把自车位置投影到路线上
        #      得到起算弧长 s0，最后沿路线向前数 center_offset 米定位障碍点。
        cum_len = [0.0] * len(route)
        for j in range(1, len(route)):
            cum_len[j] = cum_len[j - 1] + route[j - 1].distance(route[j])

        # 将自车位置投影到路线折线上（取侧向距离最小的投影点），得到起算弧长 s0
        s0, best_d = 0.0, None
        for p in range(len(route) - 1):
            A, B = route[p], route[p + 1]
            abx, aby = B.x - A.x, B.y - A.y
            seg2 = abx * abx + aby * aby
            if seg2 <= 0.0:
                continue
            t = ((ego_loc.x - A.x) * abx + (ego_loc.y - A.y) * aby) / seg2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            px, py = A.x + abx * t, A.y + aby * t
            d = math.hypot(ego_loc.x - px, ego_loc.y - py)
            if best_d is None or d < best_d:
                best_d = d
                s0 = cum_len[p] + (cum_len[p + 1] - cum_len[p]) * t

        # 目标弧长 = 自车投影弧长 + center_offset，并按路界截断
        target = min(s0 + center_offset, cum_len[-1])

        # 定位 target 所在路段并在其内线性插值
        p = 0
        while p < len(route) - 2 and cum_len[p + 1] < target:
            p += 1
        seg = cum_len[p + 1] - cum_len[p]
        t = 0.0 if seg <= 0.0 else (target - cum_len[p]) / seg
        x = route[p].x + (route[p + 1].x - route[p].x) * t
        y = route[p].y + (route[p + 1].y - route[p].y) * t
        yaw = math.degrees(math.atan2(route[p + 1].y - route[p].y, route[p + 1].x - route[p].x))
        loc = carla.Location(x=x, y=y, z=route[p].z)
    else:
        # 无路线时退回沿车头方向直线放置
        tf = ego.get_transform()
        yaw_rad = math.radians(tf.rotation.yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        loc = carla.Location(x=ego_loc.x + fx * center_offset, y=ego_loc.y + fy * center_offset, z=ego_loc.z)
        yaw = tf.rotation.yaw

    obs = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=yaw)))
    if obs is None:
        return {"ok": False, "message": f"路线前方 {distance:.0f}m 处无法放置（可能与 ego 或其它车辆重叠，可调大距离）"}
    obs.set_autopilot(False)
    obs.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    with lock:
        managed_actors.add(obs.id)
    return {
        "ok": True, "id": obs.id, "type": obs.type_id, "distance": round(distance, 1),
        "on_route": len(route) >= 2,
        "location": {"x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2)},
    }


def spawn_route_obstacles(world, positions, log, managed_actors, lock):
    """在指定世界坐标生成静态障碍车（每处一辆）。返回生成结果列表。
    生成点与静态物碰撞导致 try_spawn 失败时，沿路线方向前后微调重试，
    避免路线障碍物「生成的不够」。"""
    result = []
    bps = list(world.get_blueprint_library().filter("vehicle.*"))
    for obc in positions:
        if not bps:
            break
        yaw = math.radians(obc.get("yaw", 0))
        z = obc.get("z", 0)
        # 原位 → 沿路线前后 ±1~4m 依次重试（生成点蹭到护栏/路缘时微调即可成功）
        offsets = (0.0, 1.5, -1.5, 3.0, -3.0)
        obs = None
        used = None
        for ds in offsets:
            loc = carla.Location(
                x=obc["x"] + math.cos(yaw) * ds,
                y=obc["y"] + math.sin(yaw) * ds,
                z=z,
            )
            bp = random.choice(bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "obstacle")
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            obs = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=obc.get("yaw", 0))))
            if obs is not None:
                used = ds
                break
        if obs is None:
            log(f"障碍物生成失败（重试 {len(offsets)} 次仍碰撞）: ({obc['x']:.1f}, {obc['y']:.1f})")
            result.append({"ok": False, "x": obc["x"], "y": obc["y"]})
            continue
        obs.set_autopilot(False)
        obs.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        with lock:
            managed_actors.add(obs.id)
        if used:
            log(f"障碍物原位生成受阻，已沿路线偏移 {used:+.1f}m 生成")
        result.append({"ok": True, "id": obs.id, "type": obs.type_id,
                       "x": obc["x"], "y": obc["y"]})
    return result


def clear_obstacles(world, ego_id, log, managed_actors):
    """清除世界中所有遗留的车辆/行人（含上次实验或上次进程残留，服务/前端重启后
    仍有效），但保留本车 ego。直接按蓝图全量扫描，不依赖进程内记忆的 actor id。"""

    destroyed = []

    def _kill(a):
        try:
            if a and a.is_alive:
                a.destroy()
                managed_actors.discard(a.id)
                destroyed.append(a.type_id)
        except Exception:
            pass

    # 车辆：剔除本车 ego
    for a in world.get_actors().filter("vehicle.*"):
        if ego_id is not None and a.id == ego_id:
            continue
        _kill(a)
    # 行人 AI 控制器 + 行人
    for a in world.get_actors().filter("controller.ai.walker"):
        _kill(a)
    for a in world.get_actors().filter("walker.pedestrian.*"):
        _kill(a)

    if destroyed:
        log(f"已清理世界中 {len(destroyed)} 个遗留车辆/行人")


def refresh_route_obstacles(world, need_fresh, planned_obstacles, obstacle_actors,
                            log, managed_actors, lock):
    """障碍物与路线绑定：
      - 规划了新路线（need_fresh）→ 清掉旧障碍并按新规划生成；
      - 同一路线再次运行 → 沿用世界中已有障碍，不清理不重生；若其间被
        其它实验/清理销毁，则按原规划位置补齐缺失的障碍。
    返回更新后的障碍物登记表 [{"id", "pos"}]（调用方写回全局）。"""
    if need_fresh:
        try:
            spawned = spawn_route_obstacles(world, planned_obstacles, log, managed_actors, lock)
            n_ok = sum(1 for r in spawned if r.get("ok"))
            if spawned:
                log(f"已自动生成 {n_ok}/{len(spawned)} 个路线障碍物")
        except Exception as exc:
            log(f"路线障碍物生成失败: {exc}")
            spawned = []
        return [
            {"id": r["id"], "pos": p}
            for r, p in zip(spawned, planned_obstacles)
            if r.get("ok")
        ]

    alive = []
    missing = []
    for ent in obstacle_actors:
        try:
            a = world.get_actor(ent["id"])
        except Exception:
            a = None
        if a is not None and a.is_alive:
            # 被撞离原位（上次运行避障剐蹭/物理推移）的障碍视为缺失：
            # 不在路线上的障碍没有避障意义，销毁后按原规划位置重生成
            disp = math.hypot(a.get_location().x - ent["pos"]["x"],
                              a.get_location().y - ent["pos"]["y"])
            if disp <= 3.0:
                alive.append(ent)
            else:
                log(f"检测到障碍物被撞离原位 {disp:.1f}m，按原位置重生成")
                try:
                    a.destroy()
                except Exception:
                    pass
                missing.append(ent["pos"])
        else:
            missing.append(ent["pos"])
    # 路线未变但障碍从未生成过（如全部生成失败），按规划全量补齐
    if not obstacle_actors and planned_obstacles:
        missing = list(planned_obstacles)
    if missing:
        try:
            respawned = spawn_route_obstacles(world, missing, log, managed_actors, lock)
            n_ok = sum(1 for r in respawned if r.get("ok"))
            if n_ok:
                log(f"检测到 {len(missing)} 个路线障碍物缺失，已按原位置补齐 {n_ok} 个")
            alive.extend(
                {"id": r["id"], "pos": p}
                for r, p in zip(respawned, missing)
                if r.get("ok")
            )
        except Exception as exc:
            log(f"补齐路线障碍物失败: {exc}")
    return alive


def spawn_crossing_pedestrian(world, vehicle, pedi_list, managed_actors, lock, log):
    """在主车前方 15m、右侧 6m 生成横穿行人（AI 控制器驱动，步行至对侧）。"""
    try:
        ego_fwd = vehicle.get_transform().get_forward_vector()
        ego_tf = vehicle.get_transform()
        right = carla.Vector3D(x=-ego_fwd.y, y=ego_fwd.x, z=0)
        ped_loc = carla.Location(
            x=ego_tf.location.x + ego_fwd.x * 15 + right.x * 6,
            y=ego_tf.location.y + ego_fwd.y * 15 + right.y * 6,
            z=ego_tf.location.z,
        )
        walker_bp = world.get_blueprint_library().find("walker.pedestrian.0001")
        walker = world.try_spawn_actor(walker_bp, carla.Transform(ped_loc))
        if walker:
            ctl_bp = world.get_blueprint_library().find("controller.ai.walker")
            ctl = world.spawn_actor(ctl_bp, carla.Transform(ped_loc), attach_to=walker)
            cross = carla.Location(x=ego_tf.location.x + ego_fwd.x * 10 - right.x * 6,
                                   y=ego_tf.location.y + ego_fwd.y * 10 - right.y * 6)
            ctl.start(); ctl.go_to_location(cross); ctl.set_max_speed(1.5)
            pedi_list.extend([walker, ctl])
            with lock:
                managed_actors.update([walker.id, ctl.id])
            log("横穿行人生成成功")
    except Exception as exc:
        log(f"行人生成失败: {exc}")


def set_traffic_light_timing(world, log):
    """统一设置全图信号灯时长，压缩等待、避免超长红灯卡停车辆
    （简化逻辑：不区分是否贴近路线）。"""
    try:
        _tl_set = 0
        for tl in world.get_actors().filter("traffic.traffic_light*"):
            try:
                tl.set_frozen(False)
                tl.set_green_time(15.0)
                tl.set_yellow_time(2.0)
                tl.set_red_time(5.0)
                _tl_set += 1
            except Exception:
                pass
        if _tl_set:
            log(f"已设置全图 {_tl_set} 处信号灯：红 5s / 黄 2s / 绿 15s")
    except Exception as exc:
        log(f"设置信号灯失败: {exc}")
