"""实验10 感知层 v2：信号灯有效灯判定重设计（修三个实测缺陷）。

与 legacy（perception.py 的 read_traffic_light）的差异——有效灯须同时满足：

  ① 车道归属：停止线所在车道 == 自车当前路线车道。驶入车道的灯（停止线
     在路口出口段/连接段车道上）在换到那条车道前不可见——修「停在路中间
     等驶入车道的灯」（现象2）。取「路线车道」而非物理车道：绕障换道期间
     wp_idx 仍在路线上，判定不受影响；
  ② 路口 committed：自车 waypoint 已在路口内 → 本帧无有效灯，穿行到底
     ——与①双保险，修现象2；
  ③ 越线判定 d > 0：越过停止线才失效，无 0.5m 下限。旧版 0.5 下限与
     规划停止余量 0.5 正好相消：车停在停驻点时灯在有效边界上闪烁，红灯
     期间蠕行，挪出停止线后灯永久失效 → 提速闯灯（现象3）；
  ④ 窗口 d < tl_brake_window（参数，替代硬编码 80m）。

红/黄统一：red 和 yellow 都输出停驻点 stop_s = 停止线 − tl_stop_margin
（参数），green/off 无——黄灯刹停由规划层现有停驻点逻辑（STOP/CRUISE
舒适剖面）自动完成，不再依赖恒定 a_need 软约束。减速起点由舒适剖面
「恰好能停住」自动给出（v=√(2·a·d)），停在停止线前 tl_stop_margin。

障碍扫描（step）与 legacy 完全一致（继承）。绑定自带车道归属信息。
接口：read_traffic_light(ego_s, wp_idx, in_junction)——比 legacy 多两个
参数，由 run.py 按配置分支调用；PercFrame/TlFrame 契约不变。
"""
from __future__ import annotations

from carla_relay.experiments.comprehensive_driving.frames import TlFrame
from carla_relay.experiments.comprehensive_driving.perception import Perceiver


class PerceiverV2(Perceiver):
    """感知层 v2：障碍扫描同 legacy，信号灯判定重设计（见模块注释）。"""

    def __init__(self, ref, world, carla_map, log, dynamic_class,
                 tl_stop_margin, tl_brake_window, tl_state_map, plan_log,
                 route_lane_ids, params=None):
        self._tl_window = tl_brake_window
        self._route_lane_ids = route_lane_ids
        # 绑定：与 reference.bind_traffic_lights 同源，但携带车道归属
        self.bindings = self._bind(ref, world, log)
        super().__init__(ref, world, carla_map, log, dynamic_class,
                         tl_stop_margin, self.bindings, tl_state_map, plan_log,
                         params)

    def _bind(self, ref, world, log):
        """停止线 → 参考线弧长 s_stop + 所在车道（地图先验，一次性）。"""
        bindings = []
        try:
            _route_lane_set = set(l for l in self._route_lane_ids
                                  if l is not None)
            for _tl in world.get_actors().filter("traffic.traffic_light*"):
                try:
                    for _swp in _tl.get_stop_waypoints():
                        if (_swp.road_id, _swp.lane_id) in _route_lane_set:
                            _s_stop = ref.frenet(
                                _swp.transform.location.x,
                                _swp.transform.location.y)[0]
                            bindings.append({
                                "tl": _tl, "s_stop": _s_stop,
                                "lane": (_swp.road_id, _swp.lane_id)})
                            break
                except Exception:
                    pass
            bindings.sort(key=lambda b: b["s_stop"])
            if bindings:
                log(f"信号灯绑定(v2): {len(bindings)} 处停止线（含车道归属）")
        except Exception as exc:
            log(f"信号灯绑定(v2)失败: {exc}")
        return bindings

    def read_traffic_light(self, ego_s: float, wp_idx=None,
                           in_junction: bool = False) -> TlFrame:
        """信号灯状态（v2 判定，见模块注释）。红/黄统一输出停驻点。"""
        tl_state = "green"
        tl_dist = 999.0
        stop_s = None
        cur_lane = (self._route_lane_ids[wp_idx]
                    if wp_idx is not None and wp_idx < len(self._route_lane_ids)
                    else None)
        if not in_junction:   # ② 路口内无有效灯（committed 穿行到底）
            for b in self.bindings:
                # ① 车道归属：只看当前路线车道的出口灯（当前车道未知时退化为
                # 仅窗口判定，保持可用性）
                if cur_lane is not None and b["lane"] != cur_lane:
                    continue
                d = b["s_stop"] - ego_s
                # ③ 越线才失效（d>0）；④ 刹停考虑窗口
                if d <= 0.0 or d >= self._tl_window:
                    continue
                if d >= tl_dist:
                    continue
                try:
                    st = self._tl_state_map.get(b["tl"].state, "green")
                except Exception:
                    continue
                tl_dist, tl_state = d, st
                # 红/黄统一：都输出停驻点（黄灯刹停走规划层常规剖面）
                stop_s = (b["s_stop"] - self._red_margin
                          if st in ("red", "yellow") else None)
        if tl_state != self._tl_state_prev:
            self._plan_log(f"EVENT 信号灯(v2): {self._tl_state_prev} → {tl_state} "
                           f"@前方{tl_dist:.0f}m 车道={cur_lane} "
                           f"(stop_s={f'{stop_s:.1f}' if stop_s is not None else 'None'})")
            self._tl_state_prev = tl_state
        return TlFrame(state=tl_state, dist=tl_dist, red_stop_s=stop_s)
