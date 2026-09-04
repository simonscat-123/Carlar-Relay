"""语义分割（实验5）BEV 占据栅格世界构建。

把前视感知结果（语义分割静态可行驶/障碍 + 目标识别动态障碍）用针孔模型
逆投影回地面俯视空间，构建以自车为中心的二维占据栅格：

   栅格坐标：axis0 = 前向 fwd（自车中心为 G//2，越大越远）
             axis1 = 横向 lat（右正，越大越靠车右，与综合驾驶一致）
   单元类别：0=未知(视锥外/未感知)  1=可行驶(freespace)  2=静态障碍
             3=车辆  4=行人（2/3/4 均视为占用）

关键设计：
  - 视觉可见性约束：只有落在相机视锥、且在感知范围内的区域才被"看见"，
    视野外一律标未知 → 让"感知世界"有真实边界，而非全知地图；
  - 静态可行驶/障碍来自语义分割投影（道路/车道线=可行驶，建筑/墙/护栏=占用）；
  - 动态障碍来自融合感知目标（instance+semantic 投票 + 单目测距），占用足迹
    以目标估计半长/半宽覆盖栅格。

输出：cells (G,G) int8 类别图 + 前端渲染 payload（base64 紧凑编码）。
"""
from __future__ import annotations

import base64
import math

import numpy as np

# 单元类别
UNK = 0      # 未知 / 未感知
FREE = 1     # 可行驶（freespace）
STATIC = 2   # 静态障碍（语义）
VEH = 3      # 动态车辆（目标识别）
WALK = 4     # 动态行人（目标识别）

# 可行驶 / 静态障碍的 CityScapes 标签集合
DRIVABLE_LABELS = (1, 24)                       # Road / RoadLine
STATIC_BARRIER_LABELS = (3, 4, 5, 6, 10)        # Building / Wall / Fence / Pole / Vegetation(粗)


def _project_pixels(mask_ys, mask_xs, fx, u0, v0, tilt_rad, cam_height):
    """针孔模型把一组像素投影到地面，返回 (fwd, lat) 车体系坐标数组。"""
    v = mask_ys.astype(np.float64)
    u = mask_xs.astype(np.float64)
    delta = np.arctan((v - v0) / fx)                       # 向下俯角
    theta = tilt_rad + delta                               # 相对水平面总俯角
    valid = theta > 0.02
    # 无效像素用大有限哨兵(1e6 m)，避免 inf*0=NaN 污染 lat/fwd；
    # 该距离远超任意栅格覆盖半径，填入时会被范围判断自然剔除
    d = np.where(valid, cam_height / np.tan(np.maximum(theta, 0.02)), 1e6)
    psi = np.arctan2(u - u0, fx)                           # 水平方位角（右正，与综合驾驶一致）
    fwd = d * np.cos(psi)
    lat = d * np.sin(psi)
    return fwd, lat


class BevBuilder:
    """以自车为中心的前向占据栅格构建器。"""

    def __init__(self, *, span=60.0, res=0.5, cam_fov=90.0, cam_pitch=-5.0,
                 cam_height=1.7, perc_range=50.0, sem_range=40.0, subsample=2):
        self.res = res
        self.span = span
        self.G = int(round(span / res))
        self.half = span / 2.0
        self.cam_fov = cam_fov
        self.cam_pitch = cam_pitch
        self.cam_height = cam_height
        self.perc_range = perc_range
        self.sem_range = sem_range                    # 语义像素融合距离上限（m）
        self.subsample = int(max(1, subsample))
        self.tilt_rad = math.radians(-cam_pitch)
        self.fx = None        # 首次 build 时按语义图尺寸缓存
        self.u0 = self.v0 = 0.0
        self._sem_h = 0

    def _ensure_intrinsics(self, w):
        if self.fx is None or self._sem_w != w:
            self._sem_w = w
            self.fx = (w / 2.0) / math.tan(math.radians(self.cam_fov) / 2.0)
            self.u0 = w / 2.0
            self.v0 = self._sem_h / 2.0

    def build(self, sem_labels, targets, *, sem_h=None, sem_w=None):
        """sem_labels: (H,W) int32 CityScapes 标签；targets: fusion.perceive 结果。
        返回 (cells (G,G) int8, payload dict)。"""
        G = self.G
        cells = np.full((G, G), UNK, dtype=np.int8)
        cells[G // 2, G // 2] = FREE                    # 自车位视为可行驶

        if sem_labels is not None and sem_labels.size and sem_h:
            self._sem_h = int(sem_h)
            self._sem_w = int(sem_w or sem_labels.shape[1])
            self._ensure_intrinsics(self._sem_w)
            ssub = self.subsample
            # 语义投影统一按 sem_range 截断（前向距离内才融合进栅格）
            sem_m = self.sem_range
            # 可行驶投影（道路/车道线）
            ys, xs = np.nonzero(np.isin(sem_labels, DRIVABLE_LABELS))
            if ys.size:
                fwd, lat = _project_pixels(
                    ys[::ssub], xs[::ssub], self.fx, self.u0, self.v0,
                    self.tilt_rad, self.cam_height)
                keep = fwd <= sem_m
                self._paint(cells, fwd[keep], lat[keep], FREE)
            # 静态障碍投影（建筑/墙/护栏…）
            ys, xs = np.nonzero(np.isin(sem_labels, STATIC_BARRIER_LABELS))
            if ys.size:
                fwd, lat = _project_pixels(
                    ys[::ssub], xs[::ssub], self.fx, self.u0, self.v0,
                    self.tilt_rad, self.cam_height)
                keep = fwd <= sem_m
                self._paint(cells, fwd[keep], lat[keep], STATIC)

        # 动态障碍：按目标估计足迹覆盖栅格（覆盖同位置的纯可行驶标记）
        for t in targets:
            a0c = int(t["fwd"] / self.res + G / 2)
            a1c = int(t["lat"] / self.res + G / 2)
            hl = int(t.get("half_len", 1.5) / self.res)
            hw = int(t.get("half_w", 0.7) / self.res)
            cls = VEH if t.get("cls") == "vehicle" else WALK
            c0_s = max(0, a0c - hl); c0_e = min(G, a0c + hl + 1)
            c1_s = max(0, a1c - hw); c1_e = min(G, a1c + hw + 1)
            if c0_e > c0_s and c1_e > c1_s:
                cells[c0_s:c0_e, c1_s:c1_e] = cls

        payload = self._payload(cells)
        return cells, payload

    def _paint(self, cells, fwd, lat, cls):
        """把投影点批量标入栅格（越界/超出纵向范围丢弃）。"""
        G = self.G
        a0 = (fwd / self.res) + G / 2
        a1 = (lat / self.res) + G / 2
        m = (fwd >= 0.0) & (a0 >= 0) & (a0 < G) & (a1 >= 0) & (a1 < G)
        if m.any():
            cells[a0[m].astype(np.int64), a1[m].astype(np.int64)] = cls

    def _payload(self, cells):
        return {
            "res": self.res,
            "span": self.span,
            "w": self.G,
            "data": base64.b64encode(cells.tobytes()).decode(),
        }