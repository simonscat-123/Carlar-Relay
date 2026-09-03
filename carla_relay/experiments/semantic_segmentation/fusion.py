"""语义分割（实验5）感知融合层：目标识别 + 语义分割 → 车体系障碍列表。

对标工业界感知层（检测/分类/测距 + 传感器坐标反算）：
  - 检测：实例分割图解码出各目标像素掩码 → 最小外接矩形（2D 检测框）；
  - 分类：在每目标掩码内用语义分割图投票（车辆 / 行人）；其余类别丢弃；
  - 测距：检测框底边中点视为接地点，针孔模型求相对光轴俯角，叠加相机安装
    俯仰后由 相机高度 / tan(俯角) 反解地面距离；框中心列坐标 → 水平方位角，
    得到目标在**车体系**的前向 fwd / 横向 lat（左正右负，与真值扫描约定一致）；
  - 融合语义：把目标的类别标签与语义分割在掩码上的投票对齐，作为统一障碍。

输出（车体系前视图，按距离升序）：
  [{cls, dist, fwd, lat, width_m, half_len, half_w, box}, ...]
  全程不查询任何世界真值，视野/FOV/感知范围即真实限制。

单位工作：单帧感知，不含多帧跟踪/速度估计（进阶内容）。
"""
from __future__ import annotations

import math

import numpy as np

# 复用综合驾驶的实例 bbox 扫描与语义类别集合（单源实现，避免重复）
from carla_relay.experiments.comprehensive_driving.perception import (
    EXP10_PERC_VEHICLE_LABELS,
    EXP10_PERC_WALKER_LABELS,
    instance_bboxes,
)

# 感知模式无真值尺寸，按类别取典型半长（保守补偿障碍占用）
PERC_HALF_LEN = {"walker": 0.4, "vehicle": 2.2}
# 感知模式无真值宽度，单目宽度估计的最小截断（避免过窄栅格单元）
PERC_MIN_HALF_W = 0.45


def perceive(inst_arr, sem_arr, *, cam_fov=90.0, cam_pitch=-5.0,
             cam_height=1.7, perc_range=50.0, exclude_ids=()):
    """从实例分割 + 语义分割图估计前方障碍物（车体系），返回目标列表。

    inst_arr: (H,W,4) BGRA 实例分割原始帧；sem_arr: (H,W,4) BGRA 语义分割帧。
    cam_pitch: CARLA 约定负值向下（镜头安装俯仰角，度）。
    返回按距离升序目标 [{cls,dist,fwd,lat,width_m,half_len,half_w,box}]。
    """
    h, w = inst_arr.shape[:2]
    fx = (w / 2.0) / math.tan(math.radians(cam_fov) / 2.0)  # 针孔焦距（像素）
    u0, v0 = w / 2.0, h / 2.0
    # 实例 id 图（G 低字节 + B 高字节，同官方 decode）
    actor_ids = inst_arr[:, :, 1].astype(np.uint16) + (inst_arr[:, :, 0].astype(np.uint16) << 8)
    sem_labels = sem_arr[:, :, 2].astype(np.int32)
    is_vehicle_px = np.isin(sem_labels, EXP10_PERC_VEHICLE_LABELS)
    is_walker_px = np.isin(sem_labels, EXP10_PERC_WALKER_LABELS)

    targets = []
    for iid, (xmin, ymin, xmax, ymax, cnt, n_veh, n_wal) in instance_bboxes(
            actor_ids, is_vehicle_px, is_walker_px).items():
        if iid in exclude_ids:
            continue
        if cnt < 8 or (n_veh == 0 and n_wal == 0):
            continue  # 过小掩码视为噪声；两者皆非则丢弃
        # 单目测距：底边中点 = 接地点 → 距离 = 相机高度 / tan(总俯角)
        delta = math.atan((ymax - v0) / fx)               # 相对光轴向下俯角
        theta = math.radians(-cam_pitch) + delta          # 相对水平面总俯角
        if theta <= 0.02:
            continue                                       # 接地点近视平线，几何无解
        dist = cam_height / math.tan(theta)
        if dist > perc_range:
            continue                                       # 超出识别距离上限不可信
        u_c = (xmin + xmax) / 2.0
        psi = math.atan2(u0 - u_c, fx)                     # 水平方位角（左正右负）
        cls = "walker" if n_wal > n_veh else "vehicle"
        targets.append({
            "id": iid,
            "cls": cls,
            "dist": dist,
            "fwd": dist * math.cos(psi),
            "lat": dist * math.sin(psi),
            "width_m": max((xmax - xmin) * dist / fx, PERC_MIN_HALF_W),
            "half_len": PERC_HALF_LEN.get(cls, 1.5),
            "half_w": PERC_HALF_LEN.get(cls, 1.5) * 0.35,  # 估宽缺省：半长×典型长宽比
            "box": (xmin, ymin, xmax, ymax),
        })
    targets.sort(key=lambda t: t["dist"])
    return targets