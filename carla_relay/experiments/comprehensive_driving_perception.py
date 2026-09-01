"""综合驾驶（上）：包围框相机感知与渲染 · 前端关卡 comprehensive-driving（API 实验ID 10）。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 10: 闭环自动驾驶（Pure Pursuit + PID + 互补滤波定位）
# =============================================================================

# ── 包围框相机（参照官方示例 PythonAPI/examples/bounding_boxes.py）─────────────
# CARLA 语义标签 → (类别名, 框颜色)。颜色沿用官方 SEMANTIC_MAP（CityScapes 调色板），
# 仅保留实验10 关心的动态目标类别（车辆/行人）。
_BBOX_CLASSES = {
    12: ("行人", (220, 20, 60)),
    13: ("骑手", (255, 0, 0)),
    14: ("汽车", (0, 0, 142)),
    15: ("卡车", (0, 0, 70)),
    16: ("巴士", (0, 60, 100)),
    17: ("列车", (0, 80, 100)),
    18: ("摩托车", (0, 0, 230)),
    19: ("自行车", (119, 11, 32)),
}

_bbox_font_cache = None


def _bbox_font():
    """标签字体：优先中文字体（Windows 微软雅黑/黑体），失败回退 PIL 默认位图字体"""
    global _bbox_font_cache
    if _bbox_font_cache is None:
        for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
            try:
                _bbox_font_cache = PIL.ImageFont.truetype(name, 30)
                break
            except Exception:
                continue
        if _bbox_font_cache is None:
            _bbox_font_cache = PIL.ImageFont.load_default()
    return _bbox_font_cache


def _instance_bboxes(actor_ids, is_vehicle_px=None, is_walker_px=None):
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


def _render_bbox_frame(rgb_arr, inst_arr, actors):
    """在前相机 RGB 帧上叠加 2D 检测框，返回 JPEG bytes。

    实现思路同官方 bounding_boxes.py 的 decode_instance_segmentation +
    bbox_2d_for_actor + visualize_2d_bboxes：
    实例分割图解码出每个 actor 的像素掩码（G=actor id 低字节、B=高字节）→
    最小外接矩形 → 按语义类别着色描边 + 类别色底白字标签（含距离）。
    rgb_arr: (H,W,4) BGRA 前相机原始帧；inst_arr: (h,w,4) BGRA 实例分割帧
    （可与 RGB 分辨率不同，坐标自动缩放）；actors: [(actor, 距离m), ...]。
    """
    h, w = inst_arr.shape[:2]
    rgb_h, rgb_w = rgb_arr.shape[:2]
    sx, sy = rgb_w / w, rgb_h / h
    # 解码实例分割 → actor id 图（与官方示例 decode_instance_segmentation 一致）
    actor_ids = inst_arr[:, :, 1].astype(np.uint16) + (inst_arr[:, :, 0].astype(np.uint16) << 8)
    # 单次扫描取出所有实例 id 的 bbox，取代逐 actor 全图掩码（多目标场景 10× 加速）
    bboxes = _instance_bboxes(actor_ids)
    img = PIL.Image.fromarray(rgb_arr[:, :, :3][:, :, ::-1].copy())  # BGRA → RGB（copy 消除负步长只读视图，避免 fromarray 访问违规）
    draw = PIL.ImageDraw.Draw(img)
    font = _bbox_font()
    th = getattr(font, "size", 12)
    for actor, dist in actors:
        try:
            box = bboxes.get(actor.id)
            if box is None:
                continue  # 不在画面内
            xmin, ymin, xmax, ymax = box[:4]  # 7 元组取前 4（其余为计数/语义投票，渲染不用）
            xmin, ymin, xmax, ymax = xmin * sx, ymin * sy, xmax * sx, ymax * sy
            name, color = _BBOX_CLASSES.get(actor.semantic_tags[0], ("目标", (85, 170, 255)))
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            # 标签：类别色底 + 白字（同官方 visualize_2d_bboxes 的渲染方式），附距离
            label = f"{name} {dist:.1f}m"
            tw = draw.textlength(label, font=font)
            ty = ymin - th - 10
            if ty < 0:
                ty = ymin + 3  # 贴近画面顶部时标签移入框内
            draw.rectangle([xmin, ty, xmin + tw + 14, ty + th + 8], fill=color)
            draw.text((xmin + 7, ty + 4), label, fill=(255, 255, 255), font=font)
        except Exception:
            continue
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


# ── 实验10 感知闭环：bbox 相机单目感知 ─────────────────────────────
# CARLA 语义标签（CityScapes 编码）：12/13 = 行人/骑手，14-19 = 各类车辆
_EXP10_PERC_VEHICLE_LABELS = (14, 15, 16, 17, 18, 19)
_EXP10_PERC_WALKER_LABELS = (12, 13)
# 前相机安装参数（与 _run_exp10 挂载的前相机/实例相机一致）：
# FOV 90°，俯仰 pitch=-5°（CARLA 约定负值向下），镜头离地约 1.7m
_EXP10_CAM_FOV = 90.0
_EXP10_CAM_PITCH = -5.0
_EXP10_CAM_HEIGHT = 1.7

_EXP10_PERC_STYLE = {
    "walker": ("行人", (220, 20, 60)),
    "vehicle": ("车辆", (0, 0, 142)),
}


def _exp10_camera_perceive(inst_arr, sem_arr, exclude_ids=()):
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
    is_vehicle_px = np.isin(sem_labels, _EXP10_PERC_VEHICLE_LABELS)
    is_walker_px = np.isin(sem_labels, _EXP10_PERC_WALKER_LABELS)

    fx = (w / 2.0) / math.tan(math.radians(_EXP10_CAM_FOV / 2.0))  # 针孔模型焦距（像素）
    u0, v0 = w / 2.0, h / 2.0
    targets = []
    # 单次扫描取出每个实例 id 的 bbox/像素数/语义投票，取代逐 id 全图掩码（O(N·P)→O(P log P)）
    for iid, st in _instance_bboxes(actor_ids, is_vehicle_px, is_walker_px).items():
        if iid in exclude_ids:
            continue  # 0 已被 _instance_bboxes 排除
        xmin, ymin, xmax, ymax, cnt, n_veh, n_wal = st
        if cnt < 8 or (n_veh == 0 and n_wal == 0):
            continue  # 过小掩码视为噪声；两者皆非则丢弃
        # 3) 单目测距：底边中点 = 接地点 → 距离 = 相机高度 / tan(俯角)
        delta = math.atan((ymax - v0) / fx)              # 相对光轴的俯角（向下为正）
        theta = math.radians(-_EXP10_CAM_PITCH) + delta  # 相对水平面的总俯角
        if theta <= 0.02:
            continue  # 接地点在视平线附近，几何无解（远景/异常帧）
        dist = _EXP10_CAM_HEIGHT / math.tan(theta)
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


def _render_perceived_frame(rgb_arr, perceived, inst_w, inst_h):
    """在前相机 RGB 帧上叠加「感知闭环」检测结果，返回 JPEG bytes。

    与 _render_bbox_frame 的区别：只画感知到的目标，标注距离为单目几何估计值
    而非真值——学生看到的画面就是控制器实际使用的感知输出。
    perceived: _exp10_camera_perceive 的返回值；box 为实例图像素坐标，自动缩放。
    """
    rgb_h, rgb_w = rgb_arr.shape[:2]
    sx, sy = rgb_w / float(inst_w), rgb_h / float(inst_h)
    img = PIL.Image.fromarray(rgb_arr[:, :, :3][:, :, ::-1].copy())  # BGRA → RGB（copy 消除负步长只读视图，避免 fromarray 访问违规）
    draw = PIL.ImageDraw.Draw(img)
    font = _bbox_font()
    th = getattr(font, "size", 12)
    for p in perceived:
        xmin, ymin, xmax, ymax = p["box"]
        xmin, ymin, xmax, ymax = xmin * sx, ymin * sy, xmax * sx, ymax * sy
        name, color = _EXP10_PERC_STYLE.get(p["cls"], ("目标", (85, 170, 255)))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        label = f"{name} {p['dist']:.1f}m·视觉"
        tw = draw.textlength(label, font=font)
        ty = ymin - th - 6
        if ty < 0:
            ty = ymin + 2
        draw.rectangle([xmin, ty, xmin + tw + 8, ty + th + 4], fill=color)
        draw.text((xmin + 4, ty + 2), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


_EXP10_LOCK = threading.Lock()
_EXP10_RUNNING = False
_EXP10_ABORT = False
_EXP10_THREAD = None
# 实验运行中可实时调整的控制参数（前端滑杆 → /experiment/10/params，无需重启实验）
_EXP10_CTRL = {}
_EXP10_CTRL_LOCK = threading.Lock()
# 实验10 ego 车辆 id（供"前方生成障碍物"接口定位）
_EXP10_VEHICLE_ID = None
# 实验10 规划路线的路点（供障碍物沿路线前方生成）
_EXP10_ROUTE = []
# 前端"生成障碍物"请求队列：由 exp10 主循环线程执行，
# 避免 Flask 线程与 world.tick() 跨线程并发调用 CARLA 客户端导致崩溃
_EXP10_SPAWN_REQ = None
_EXP10_SPAWN_LOCK = threading.Lock()
# 规划阶段在路线上随机选定的 3 个障碍物位置（世界坐标），供实验启动时统一生成
_EXP10_PLANNED_OBSTACLES = []
# 是否重新规划过：为 True 表示前端重新规划了路线（障碍位置已更新），
# 下次运行实验时应清空世界旧障碍并重新生成；为 False(未重新规划)则沿用世界已有障碍，不清不重生成。
_EXP10_PLANNED_CHANGED = False
# 上一次 /route/plan 的起终点（世界坐标）：起终点均在容差内视为「同一路线重复规划」，
# 此时障碍物与路线绑定，沿用已规划的障碍位置，不触发清理重建。
_EXP10_LAST_PLAN = None
_EXP10_ROUTE_SAME_TOL = 5.0  # 米：起终点均落在该容差内判定为同一路线
# 与当前路线绑定的障碍物 [{"id": actor_id, "pos": {...}}]，跨实验沿用；
# 若其间被其它实验/清理销毁，实验启动时按原规划位置补齐。
_EXP10_OBSTACLE_ACTORS = []

