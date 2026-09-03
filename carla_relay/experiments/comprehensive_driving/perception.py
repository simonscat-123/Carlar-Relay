"""实验10 感知层：目标检测（bbox 相机单目感知）+ 障碍统一扫描 + 信号灯状态。

对标工业界感知层（检测/分类/测距 + 传感器坐标反算 + 车道归属）：

  - 检测算法：实例分割掩码 → 2D 检测框，语义分割投票分类（车辆/行人），
    接地点针孔模型单目测距 + 像素方位角（不查询任何世界真值）；
  - Perceiver.step：双模式障碍扫描——
      感知闭环（perception_on）：只来自 bbox 相机，视野限制即真实限制
      （FOV ±45°、识别距离上限 perception_range）；由感知结果反算世界
      坐标（基准=融合定位），再用 HD Map 做车道归属，与真实系统一致；
      世界真值模式：全量扫描 perception_range 内车辆/行人（含包围盒
      尺寸/实测速度）。
    输出统一 Frenet 障碍列表（全量保留不预筛选本道——筛选交给规划器的
    碰撞检查，旁道/远端障碍天然进入时空联合检查）。
  - Perceiver.read_traffic_light：读取绑定到本路线的信号灯当前灯色
    （停止线绑定属参考线/地图先验，见 reference.py）。

渲染（bbox 叠框等 HMI 职责）在 viz.py。
本模块为真实模块，仅依赖 carla/numpy/math；平台服务经构造函数注入。
"""
from __future__ import annotations

import math

import carla
import numpy as np

from carla_relay.experiments.comprehensive_driving.frames import PercFrame, TlFrame
from carla_relay.experiments.comprehensive_driving.params import read

# ── 感知识别范围（识别距离上限）────────────────────────────────────────
# 参数来源登记（key, 默认, 类型, 来源）；来源 JSON=构造注入 params
# （/start body→_run_exp10(args)）。两种模式统一封顶：真值扫描 /
# bbox 单目感知（perception_on）。
P_PERC_RANGE = ("perception_range", 50.0, float, "JSON")  # 障碍识别距离上限（m）

# ── bbox 相机（参照官方示例 PythonAPI/examples/bounding_boxes.py）─────────────
# CARLA 语义标签 → (类别名, 框颜色)。颜色沿用官方 SEMANTIC_MAP（CityScapes 调色板），
# 仅保留实验10 关心的动态目标类别（车辆/行人）。
BBOX_CLASSES = {
    12: ("行人", (220, 20, 60)),
    13: ("骑手", (255, 0, 0)),
    14: ("汽车", (0, 0, 142)),
    15: ("卡车", (0, 0, 70)),
    16: ("巴士", (0, 60, 100)),
    17: ("列车", (0, 80, 100)),
    18: ("摩托车", (0, 0, 230)),
    19: ("自行车", (119, 11, 32)),
}

# ── 实验10 感知闭环：bbox 相机单目感知 ─────────────────────────────
# CARLA 语义标签（CityScapes 编码）：12/13 = 行人/骑手，14-19 = 各类车辆
EXP10_PERC_VEHICLE_LABELS = (14, 15, 16, 17, 18, 19)
EXP10_PERC_WALKER_LABELS = (12, 13)
# 前相机安装参数（与传感器装配的前相机/实例相机一致）：
# FOV 90°，俯仰 pitch=-5°（CARLA 约定负值向下），镜头离地约 1.7m
EXP10_CAM_FOV = 90.0
EXP10_CAM_PITCH = -5.0
EXP10_CAM_HEIGHT = 1.7

EXP10_PERC_STYLE = {
    "walker": ("行人", (220, 20, 60)),
    "vehicle": ("车辆", (0, 0, 142)),
}

# 感知模式下无真值尺寸，按类别取典型半长（保守补偿，修"只算障碍中心"缺陷）
PERC_HALF_LEN = {"walker": 0.3, "vehicle": 2.4}


def instance_bboxes(actor_ids, is_vehicle_px=None, is_walker_px=None):
    """单次扫描实例 id 图，返回每个非零 id 的统计信息。

    替代逐 actor/逐 id 的 ``actor_ids == id`` 全图掩码扫描（O(N·P)→O(P log P)），
    多动态目标场景下显著加速（包围框渲染与相机感知共用此函数）。

    actor_ids: (H,W) uint16 实例 id 图（G 低字节 + B 高字节，0=背景）。
    is_vehicle_px/is_walker_px: (H,W) bool 语义投票掩码；缺省时相应计数为 0。
    返回 {id: (xmin,ymin,xmax,ymax,count,n_veh,n_wal)}，像素坐标、actor_ids 原分辨率。
    """
    h, w = actor_ids.shape
    flat = actor_ids.reshape(-1)
    nz = flat != 0
    if not nz.any():
        return {}
    ids = flat[nz]
    idx = np.nonzero(nz)[0]
    ys = (idx // w).astype(np.int64)
    xs = (idx % w).astype(np.int64)
    # 按实例 id 稳定排序后分组，分组内即可一次性求 bbox/计数/语义投票
    order = np.argsort(ids, kind="stable")
    ids = ids[order]; ys = ys[order]; xs = xs[order]
    if is_vehicle_px is not None:
        veh = is_vehicle_px.reshape(-1)[nz][order].astype(np.int64)
    if is_walker_px is not None:
        wal = is_walker_px.reshape(-1)[nz][order].astype(np.int64)
    if len(ids) > 1:
        starts = np.concatenate([[0], np.flatnonzero(np.diff(ids)) + 1]).astype(np.int64)
    else:
        starts = np.array([0], dtype=np.int64)
    gid = ids[starts]
    count = np.append(starts[1:], len(ids)) - starts
    xmin = np.minimum.reduceat(xs, starts)
    xmax = np.maximum.reduceat(xs, starts)
    ymin = np.minimum.reduceat(ys, starts)
    ymax = np.maximum.reduceat(ys, starts)
    n_veh = np.add.reduceat(veh, starts) if is_vehicle_px is not None else np.zeros(len(starts), dtype=np.int64)
    n_wal = np.add.reduceat(wal, starts) if is_walker_px is not None else np.zeros(len(starts), dtype=np.int64)
    return {int(gid[k]): (int(xmin[k]), int(ymin[k]), int(xmax[k]), int(ymax[k]),
                          int(count[k]), int(n_veh[k]), int(n_wal[k])) for k in range(len(starts))}


def camera_perceive(inst_arr, sem_arr, exclude_ids=()):
    """bbox 相机单目感知（实验10 感知闭环）：从相机图像估计前方障碍物距离/方位。

    教学化的「检测 + 分类 + 单目测距」流程，不查询任何世界真值：
      1) 实例分割图解码出各目标的像素掩码 → 最小外接矩形（2D 检测框）；
      2) 语义分割图在掩码内投票分类：车辆 / 行人（其余类别如信号灯丢弃）；
      3) 检测框底边中点视为接地点，针孔模型求其相对光轴俯角，加上相机俯仰
         得到相对水平面的俯角，再用 相机离地高度 / tan(俯角) 反解地面距离；
      4) 框中心列坐标 → 水平方位角，得到目标在车体坐标系的前向/横向位置。
    exclude_ids: 需要剔除的实例 id（如自车，避免把自己的引擎盖当作障碍）。
    返回按距离升序的目标列表：[{cls, dist, fwd, lat, width_m,
    box=(xmin,ymin,xmax,ymax)（实例图像素坐标）}]。
    """
    h, w = inst_arr.shape[:2]
    # 实例 id 图（与官方 decode_instance_segmentation 一致：G 低字节 + B 高字节）
    actor_ids = inst_arr[:, :, 1].astype(np.uint16) + (inst_arr[:, :, 0].astype(np.uint16) << 8)
    sem_labels = sem_arr[:, :, 2].astype(np.int32)
    is_vehicle_px = np.isin(sem_labels, EXP10_PERC_VEHICLE_LABELS)
    is_walker_px = np.isin(sem_labels, EXP10_PERC_WALKER_LABELS)

    fx = (w / 2.0) / math.tan(math.radians(EXP10_CAM_FOV / 2.0))  # 针孔模型焦距（像素）
    u0, v0 = w / 2.0, h / 2.0
    targets = []
    # 单次扫描取出每个实例 id 的 bbox/像素数/语义投票，取代逐 id 全图掩码（O(N·P)→O(P log P)）
    for iid, st in instance_bboxes(actor_ids, is_vehicle_px, is_walker_px).items():
        if iid in exclude_ids:
            continue  # 0 已被 instance_bboxes 排除
        xmin, ymin, xmax, ymax, cnt, n_veh, n_wal = st
        if cnt < 8 or (n_veh == 0 and n_wal == 0):
            continue  # 过小掩码视为噪声；两者皆非则丢弃
        # 3) 单目测距：底边中点 = 接地点 → 距离 = 相机高度 / tan(俯角)
        delta = math.atan((ymax - v0) / fx)              # 相对光轴的俯角（向下为正）
        theta = math.radians(-EXP10_CAM_PITCH) + delta   # 相对水平面的总俯角
        if theta <= 0.02:
            continue  # 接地点在视平线附近，几何无解（远景/异常帧）
        dist = EXP10_CAM_HEIGHT / math.tan(theta)
        # 4) 方位 → 车体系前向/横向（左正右负，与真值扫描的 lat 约定一致）
        u_c = (xmin + xmax) / 2.0
        psi = math.atan2(u0 - u_c, fx)
        targets.append({
            "cls": "walker" if n_wal > n_veh else "vehicle",
            "dist": dist,
            "fwd": dist * math.cos(psi),
            "lat": dist * math.sin(psi),
            "width_m": (xmax - xmin) * dist / fx,  # 单目尺寸估计：像宽×距离/焦距
            "box": (xmin, ymin, xmax, ymax),
        })
    targets.sort(key=lambda t: t["dist"])
    return targets


def _near_actor_verts(wpx, wpy, world, exclude_id, radius=2.0):
    """在 (wpx,wpy) 附近 radius 内匹配最近的真实 vehicle/walker，返回其真实包围盒
    世界顶点（[{x,y,z}]*8）——用于感知模式下给障碍物补齐**真值**顶点，使鸟瞰/检测
    相机的 3D 框严格贴物，不再依赖 reference.world(s,l) 弧长重建。找不到则返回 None。"""
    try:
        if world is None:
            return None
        best, bd = None, float("inf")
        for a in world.get_actors():
            if a.id == exclude_id:
                continue
            tid = a.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                continue
            al = a.get_location()
            d = (al.x - wpx) ** 2 + (al.y - wpy) ** 2   # 避免建 Location 对象，平方距离
            if d < bd:
                bd, best = d, a
        if best is None or bd > radius * radius:
            return None
        return [{"x": v.x, "y": v.y, "z": v.z}
                for v in best.bounding_box.get_world_vertices(best.get_transform())]
    except Exception:
        return None


class Perceiver:
    """感知层：每 tick 扫描障碍（相机闭环 / 世界真值）+ 读取信号灯状态。"""

    def __init__(self, ref, world, carla_map, log, dynamic_class, red_margin,
                 tl_bindings, tl_state_map, plan_log, params=None):
        self._ref = ref                    # ReferenceLine（Frenet/地图查询）
        self._world = world                # CARLA world（真值模式 actor 扫描用）
        self._carla_map = carla_map
        self._log = log                    # _exp_log（SSE 实验日志）
        self._dynamic_class = dynamic_class
        self._red_margin = red_margin      # 红灯停止线余量（规划参数，红灯判定用）
        self._tl_bindings = tl_bindings    # 参考线停止线绑定（装配期一次性）
        self._tl_state_map = tl_state_map
        self._plan_log = plan_log          # 规划调试日志（EVENT 信号灯变化）
        self._perc_range = read(params, P_PERC_RANGE)  # 识别距离上限（JSON: perception_range 可覆盖）
        self._diag = {"err": 0}            # 相机感知异常诊断计数
        self._tl_state_prev = None         # 上一帧信号灯状态——变化事件检测

    def step(self, perception_on, vehicle, ego_tf, fused_loc, scan_j0, scan_j1,
             inst, sem, instance_raw, semantic_raw) -> PercFrame:
        """障碍统一扫描：全量保留（不预筛选本道），输出 Frenet 障碍列表。
        每个障碍含半长/半宽（修"距离只算到障碍中心"的问题）。"""
        obstacles = []            # 统一障碍列表：[{s,l,half_len,half_w,v_s,lane,cls,id}]
        bbox_cands = []           # 包围框相机候选：perception_range 内的车辆/行人 (actor, 距离)
        perceived = []            # 感知闭环：bbox 相机单目感知到的障碍物
        fwd = ego_tf.get_forward_vector()
        left = carla.Vector3D(x=fwd.y, y=-fwd.x, z=0)  # 车体系左向（lat 左正约定；(-fwd.y,fwd.x) 是右向，曾致感知坐标镜像）

        if perception_on:
            # ── 感知闭环：不查询世界真值，障碍物只来自 bbox 相机（实例+语义）──
            # 视野限制即真实限制：FOV ±45°、识别距离上限 perception_range 内的目标才能被"看见"
            try:
                if inst.id in instance_raw and sem.id in semantic_raw:
                    ih = int(inst.attributes["image_size_y"])
                    iw = int(inst.attributes["image_size_x"])
                    inst_arr = np.frombuffer(instance_raw[inst.id], dtype=np.uint8).reshape((ih, iw, 4))
                    sh = int(sem.attributes["image_size_y"])
                    sw = int(sem.attributes["image_size_x"])
                    sem_arr = np.frombuffer(semantic_raw[sem.id], dtype=np.uint8).reshape((sh, sw, 4))
                    perceived = camera_perceive(inst_arr, sem_arr, exclude_ids=(vehicle.id,))
            except Exception as exc:
                self._diag["err"] += 1
                if self._diag["err"] <= 3:
                    self._log(f"相机感知异常#{self._diag['err']}: {exc!r}")
            obs_list = []
            for p in perceived:
                if p["dist"] > self._perc_range:
                    continue   # 超过识别距离上限（perception_range，本车太远不可信）
                fwd_dist, lat = p["fwd"], p["lat"]
                # 由感知结果反算世界坐标（基准=融合定位；再用 HD Map 做车道归属），
                # 与真实系统一致：相机检测目标 → 地图匹配 → 车道级行为决策
                wpx = fused_loc.x + fwd.x * fwd_dist + left.x * lat
                wpy = fused_loc.y + fwd.y * fwd_dist + left.y * lat
                s_o, l_o, _tx, _ty = self._ref.frenet(wpx, wpy, scan_j0, scan_j1)
                lane_id = None
                try:
                    owp = self._carla_map.get_waypoint(carla.Location(x=wpx, y=wpy, z=0.0),
                                                        project_to_road=True, lane_type=carla.LaneType.Driving)
                    if owp is not None:
                        lane_id = (owp.road_id, owp.lane_id)
                except Exception:
                    pass
                obstacles.append({
                    "s": s_o, "l": l_o,
                    "half_len": PERC_HALF_LEN.get(p["cls"], 2.0),  # 类别典型半长（保守）
                    "half_w": max(0.3, p["width_m"] / 2.0),
                    "v_s": 0.0,   # 单帧感知无速度估计（多帧跟踪是进阶内容）
                    "lane": lane_id, "cls": p["cls"], "id": None,
                    "verts": _near_actor_verts(wpx, wpy, self._world, vehicle.id),
                })
                obs_list.append({
                    "category": EXP10_PERC_STYLE.get(p["cls"], ("目标",))[0],
                    "dist": round(p["dist"], 1), "size": round(p["width_m"], 1),
                    "x": round(wpx, 1), "y": round(wpy, 1),  # 感知估计的世界坐标（鸟瞰图标记）
                    "vel": None,
                })
        else:
            obs_list = []
            for actor in self._world.get_actors():
                if actor.id == vehicle.id:
                    continue
                tid = actor.type_id
                if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                    continue
                dist = actor.get_location().distance(fused_loc)
                if dist > self._perc_range:
                    continue
                vel = actor.get_velocity()
                speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
                bb = actor.bounding_box.extent
                aloc = actor.get_location()
                s_o, l_o, tx_o, ty_o = self._ref.frenet(aloc.x, aloc.y, scan_j0, scan_j1)
                lane_id = None
                try:
                    owp = self._carla_map.get_waypoint(aloc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if owp is not None:
                        lane_id = (owp.road_id, owp.lane_id)
                except Exception:
                    pass
                try:
                    _wv = [{"x": v.x, "y": v.y, "z": v.z}
                           for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
                except Exception:
                    _wv = None
                obstacles.append({
                    "s": s_o, "l": l_o,
                    "half_len": float(bb.x),   # 真值模式：包围盒半长（extent 为半尺寸）
                    "half_w": float(bb.y),
                    "verts": _wv,              # 真实包围盒世界顶点（HMI 精确投影）
                    "v_s": vel.x * tx_o + vel.y * ty_o,  # 沿参考线的纵向速度
                    "lane": lane_id, "cls": tid, "id": actor.id,
                })
                bbox_cands.append((actor, dist))
                obs_list.append({
                    "category": self._dynamic_class(tid),
                    "dist": round(dist, 1), "size": round(max(bb.x, bb.y) * 2, 1),
                    "x": round(aloc.x, 1), "y": round(aloc.y, 1),
                    "vel": round(speed, 1),
                })
        return PercFrame(obstacles=obstacles, perceived=perceived,
                         obs_list=obs_list, bbox_cands=bbox_cands)

    def read_traffic_light(self, ego_s: float) -> TlFrame:
        """信号灯状态：有效灯 = 绑定到本路线、且停止线仍在前方(0.5~80m)的最近一个；
        车越过停止线后 s 差变负，该灯自动退出考虑——路口内/出口不再被
        交叉方向的红灯误刹（committed 语义由 s 比较天然实现）。"""
        tl_state = "green"
        tl_dist = 999.0
        red_stop_s = None
        for b in self._tl_bindings:
            d = b["s_stop"] - ego_s
            if 0.5 < d < 80.0 and d < tl_dist:
                try:
                    st = self._tl_state_map.get(b["tl"].state, "green")
                except Exception:
                    continue
                tl_dist, tl_state = d, st
                red_stop_s = (b["s_stop"] - self._red_margin) if st == "red" else None
        if tl_state != self._tl_state_prev:
            self._plan_log(f"EVENT 信号灯: {self._tl_state_prev} → {tl_state} "
                           f"@前方{tl_dist:.0f}m (red_stop_s="
                           f"{f'{red_stop_s:.1f}' if red_stop_s is not None else 'None'})")
            self._tl_state_prev = tl_state
        return TlFrame(state=tl_state, dist=tl_dist, red_stop_s=red_stop_s)
