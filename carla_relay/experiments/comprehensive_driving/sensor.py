"""实验10 驱动层：传感器装配 + 帧数据注入 + 碰撞事件记录。

对标工业界驱动层（传感器 HAL / 驱动节点）：CARLA 的传感器即硬件抽象，
world.tick() 即驱动节拍。本层职责：

  - SensorRig.spawn：挂载全套传感器（前相机 RGB ×1 / 实例分割 / 语义分割 /
    高空俯视 / LiDAR / GNSS / IMU / 碰撞），回调把帧数据注入平台帧缓存
    （_sensor_frames 等，SSE 推流消费）；
  - 前相机原始 BGRA 帧缓存：包围框渲染需要未编码帧（_sensor_callback 只存
    JPEG），rgb_raw 由 rig 持有；
  - CollisionRecorder：碰撞事件 → 事故记录（1 秒内的连续接触合并为一次事故
    ）+ SSE 实时告警。回调在传感器线程触发，只做轻量追加；
  - SensorRig.cleanup：先停止监听（断开流），再销毁，避免 socket 报错/scope
    警告；并同步从 _managed_actors 移除，否则下次运行 _sweep_stale_actors
    会对已销毁 actor 重复 destroy，触发 CARLA libcarla 原生 Abort。

平台服务（帧缓存/actor 管理/回调）经构造函数注入，不读全局命名空间。
"""
from __future__ import annotations

import math
import threading
import time

import carla


class CollisionRecorder:
    """碰撞传感器回调 + 事故合并（评分报告数据源）。"""

    def __init__(self, push):
        self.events = []        # 原始事件 [{wall, cls, impulse}]
        self.incidents = []     # 合并后的事故 [{wall, cls, impulse, contacts}]
        self.lock = threading.Lock()
        self._push = push       # _push_to_sse

    def make_callback(self):
        def _on_collision(event):
            try:
                imp = event.normal_impulse
                mag = math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)
                other = event.other_actor
                tid = other.type_id if other is not None else ""
                if tid.startswith("walker."):
                    cls = "行人"
                elif tid.startswith("vehicle."):
                    cls = "车辆"
                else:
                    cls = "静态物体"
                now = time.time()
                with self.lock:
                    self.events.append({"wall": now, "cls": cls, "impulse": round(mag, 1)})
                    if self.incidents and now - self.incidents[-1]["wall"] <= 1.0:
                        inc = self.incidents[-1]
                        inc["impulse"] = max(inc["impulse"], round(mag, 1))
                        inc["contacts"] += 1
                        is_new = False
                    else:
                        self.incidents.append({"wall": now, "cls": cls,
                                               "impulse": round(mag, 1), "contacts": 1})
                        is_new = True
                if is_new:
                    self._push({"experiment": {"id": 10, "collision": {
                        "cls": cls, "impulse": round(mag, 1), "count": len(self.incidents),
                    }}})
            except Exception:
                pass
        return _on_collision

    def snapshot(self):
        """合并后的事故快照（报告用）。"""
        with self.lock:
            return [dict(i) for i in self.incidents]


class SensorRig:
    """驱动层：传感器装配 / 帧注入 / 清理。"""

    def __init__(self, *, sensor_callback, sensor_refs, managed_actors, lock, log):
        self._sensor_callback = sensor_callback
        self._sensor_refs = sensor_refs
        self._managed_actors = managed_actors
        self._lock = lock
        self._log = log
        self.created_sids = []
        self.collision = CollisionRecorder(None)   # push 在 spawn 时注入（需 world 就绪）
        # 前相机原始 BGRA 帧缓存：包围框渲染需要未编码帧
        self.rgb_raw = {"raw": None, "w": 1280, "h": 720}

    def spawn(self, world, vehicle, push, *, bird_w=960, bird_h=960, bird_fov=90):
        """挂载全套传感器，返回 (cam, inst, sem, bird, lidar, gnss, imu, col)。

        bird_w/bird_h/bird_fov：鸟瞰相机分辨率与 FOV，由实验 JSON/前端传入，
        仅启动时生效（不参与运行中热调）。分辨率做非法值防护后应用。
        """
        self.collision._push = push
        log = self._log

        # 前相机 RGB
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "90")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0)), attach_to=vehicle)
        self._sensor_refs[cam.id] = cam

        def _on_front_rgb(d, sid=cam.id):
            self.rgb_raw["raw"] = d.raw_data
            self.rgb_raw["w"] = d.width
            self.rgb_raw["h"] = d.height
            self._sensor_callback(sid, "camera", d)

        cam.listen(_on_front_rgb)
        self._register(cam)

        # 实例分割相机（包围框检测，参照官方 bounding_boxes.py；与前相机同位姿，
        # 半分辨率降低掩码计算量，渲染时坐标自动缩放到 RGB 帧）
        inst_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        inst_bp.set_attribute("image_size_x", "640")
        inst_bp.set_attribute("image_size_y", "360")
        inst = world.spawn_actor(inst_bp, carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0)), attach_to=vehicle)
        self._sensor_refs[inst.id] = inst
        inst.listen(lambda d, sid=inst.id: self._sensor_callback(sid, "instance", d))
        self._register(inst)

        # 语义分割相机
        sem_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", "640")
        sem_bp.set_attribute("image_size_y", "360")
        sem = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-5.0)), attach_to=vehicle)
        self._sensor_refs[sem.id] = sem
        sem.listen(lambda d, sid=sem.id: self._sensor_callback(sid, "semantic", d))
        self._register(sem)

        # 高空俯视相机（跟随车辆，真实渲染俯瞰画面）。
        # 分辨率由实验 JSON/前端参数控制；防非法值（限宽高范围）后设置。
        try:
            bird_w = int(bird_w)
            bird_h = int(bird_h)
        except Exception:
            bird_w, bird_h = 960, 960
        bird_w = min(max(bird_w, 320), 1920)
        bird_h = min(max(bird_h, 320), 1920)
        bird_fov = float(bird_fov) if not isinstance(bird_fov, bool) else 90.0
        bird_fov = min(max(bird_fov, 30.0), 120.0)
        bird_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bird_bp.set_attribute("image_size_x", str(bird_w))
        bird_bp.set_attribute("image_size_y", str(bird_h))
        bird_bp.set_attribute("fov", str(bird_fov))
        bird = world.spawn_actor(
            bird_bp,
            carla.Transform(carla.Location(z=45), carla.Rotation(pitch=-90, yaw=0, roll=0)),
            attach_to=vehicle,
        )
        self._sensor_refs[bird.id] = bird
        bird.listen(lambda d, sid=bird.id: self._sensor_callback(sid, "camera", d))
        self._register(bird)

        # LiDAR
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "50000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.4)), attach_to=vehicle)
        self._sensor_refs[lidar.id] = lidar
        lidar.listen(lambda d, sid=lidar.id: self._sensor_callback(sid, "lidar", d))
        self._register(lidar)

        # GNSS
        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
        self._sensor_refs[gnss.id] = gnss
        gnss.listen(lambda d, sid=gnss.id: self._sensor_callback(sid, "gnss", d))
        self._register(gnss)

        # IMU
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        self._sensor_refs[imu.id] = imu
        imu.listen(lambda d, sid=imu.id: self._sensor_callback(sid, "imu", d))
        self._register(imu)

        # 碰撞传感器（评分报告用）：碰撞事件 → 事故记录 + SSE 实时告警。
        # 回调在传感器线程触发，只做轻量追加；1 秒内的连续接触合并为一次事故。
        col_bp = world.get_blueprint_library().find("sensor.other.collision")
        col = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        self._sensor_refs[col.id] = col
        col.listen(self.collision.make_callback())
        self._register(col)

        log(f"车辆+传感器就绪: vid={vehicle.id} cam={cam.id} inst={inst.id} "
            f"lidar={lidar.id} gnss={gnss.id} imu={imu.id} collision={col.id}")
        return cam, inst, sem, bird, lidar, gnss, imu, col

    def _register(self, sensor):
        self.created_sids.append(sensor.id)
        with self._lock:
            self._managed_actors.add(sensor.id)

    def cleanup(self, world):
        """清理传感器：先停止监听（断开流），再销毁。"""
        for sid in self.created_sids:
            try:
                a = world.get_actor(sid)
                if a and a.is_alive:
                    if hasattr(a, "stop") and a.is_listening:
                        a.stop()
                    if not a.destroy():
                        self._log(f"传感器销毁失败 sid={sid}")
                self._sensor_refs.pop(sid, None)
                # 必须同步从 _managed_actors 移除，否则下次运行 _sweep_stale_actors
                # 会对已销毁 actor 重复 destroy，触发 CARLA libcarla 原生 Abort
                with self._lock:
                    self._managed_actors.discard(sid)
            except Exception as exc:
                self._log(f"传感器清理异常 sid={sid}: {exc!r}")
