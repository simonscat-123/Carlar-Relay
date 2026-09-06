"""定位实验的轻量 GNSS/INS 卡尔曼滤波。"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def velocity_increment_std(velocity_random_walk_std: float, dt: float) -> float:
    """把速度随机游走强度离散为单帧速度增量标准差。"""
    return max(0.0, float(velocity_random_walk_std)) * math.sqrt(max(0.0, float(dt)))


@dataclass
class SensorFrameGate:
    """只放行尚未处理过的传感器帧。"""

    last_frame: int | None = None

    def accept(self, frame: int | None) -> bool:
        if frame is None or frame == self.last_frame:
            return False
        self.last_frame = frame
        return True


@dataclass
class AxisKalmanFilter:
    """单轴位置/速度卡尔曼滤波器，状态为 ``[p, v]``。"""

    position: float = 0.0
    velocity: float = 0.0
    position_variance: float = 1.0
    position_velocity_covariance: float = 0.0
    velocity_variance: float = 1.0

    def predict(
        self,
        acceleration: float,
        dt: float,
        velocity_random_walk_std: float,
        velocity_increment_error: float = 0.0,
    ) -> None:
        """使用 IMU 加速度和本帧人工速度误差预测状态及协方差。"""
        dt = max(0.0, float(dt))
        acceleration = float(acceleration)
        self.position += self.velocity * dt + 0.5 * acceleration * dt * dt
        # 人工 INS 误差只进入速度状态，后续帧再由速度递推进入位置。
        self.velocity += acceleration * dt + float(velocity_increment_error)

        p_pp = self.position_variance
        p_pv = self.position_velocity_covariance
        p_vv = self.velocity_variance
        q = max(0.0, float(velocity_random_walk_std)) ** 2
        # 本帧随机误差施加在速度更新末端，Q 只增加速度方差；之后通过 F
        # 自然传播到位置，和上面的状态误差注入方式保持一致。
        self.position_variance = p_pp + 2.0 * dt * p_pv + dt * dt * p_vv
        self.position_velocity_covariance = p_pv + dt * p_vv
        self.velocity_variance = p_vv + q * dt

    def update(self, measured_position: float, gnss_position_std: float) -> tuple[float, float, float]:
        """用 GNSS 位置更新，返回 ``(位置增益, 速度增益, 创新)``。"""
        measurement_variance = max(0.0, float(gnss_position_std)) ** 2
        innovation = float(measured_position) - self.position
        innovation_variance = self.position_variance + measurement_variance
        if innovation_variance <= 1e-15:
            self.position = float(measured_position)
            return 1.0, 0.0, innovation

        gain_position = self.position_variance / innovation_variance
        gain_velocity = self.position_velocity_covariance / innovation_variance
        self.position += gain_position * innovation
        self.velocity += gain_velocity * innovation

        old_pp = self.position_variance
        old_pv = self.position_velocity_covariance
        old_vv = self.velocity_variance
        self.position_variance = max(0.0, (1.0 - gain_position) * old_pp)
        self.position_velocity_covariance = (1.0 - gain_position) * old_pv
        self.velocity_variance = max(0.0, old_vv - gain_velocity * old_pv)
        return gain_position, gain_velocity, innovation


class PositionKalmanFilter2D:
    """x/y 两轴位置—速度卡尔曼滤波器。"""

    def __init__(self, initial_position_std: float, initial_velocity_std: float):
        p_var = max(0.0, float(initial_position_std)) ** 2
        v_var = max(0.0, float(initial_velocity_std)) ** 2
        self.x = AxisKalmanFilter(position_variance=p_var, velocity_variance=v_var)
        self.y = AxisKalmanFilter(position_variance=p_var, velocity_variance=v_var)

    def predict(
        self,
        ax: float,
        ay: float,
        dt: float,
        velocity_random_walk_std: float,
        velocity_error_x: float = 0.0,
        velocity_error_y: float = 0.0,
    ) -> None:
        self.x.predict(ax, dt, velocity_random_walk_std, velocity_error_x)
        self.y.predict(ay, dt, velocity_random_walk_std, velocity_error_y)

    def update(self, gx: float, gy: float, gnss_position_std: float) -> dict:
        kx, kvx, innovation_x = self.x.update(gx, gnss_position_std)
        ky, kvy, innovation_y = self.y.update(gy, gnss_position_std)
        return {
            "gain_position_x": kx,
            "gain_position_y": ky,
            "gain_velocity_x": kvx,
            "gain_velocity_y": kvy,
            "innovation_x": innovation_x,
            "innovation_y": innovation_y,
        }


class YawKalmanFilter:
    """处理角度卷绕的一维航向卡尔曼滤波器。"""

    def __init__(self, initial_angle_deg: float, initial_std_deg: float):
        self.angle_deg = _wrap_deg(initial_angle_deg)
        self.variance = max(0.0, float(initial_std_deg)) ** 2

    def predict(self, gyro_rate_deg_s: float, dt: float, gyro_rate_std_deg_s: float) -> None:
        dt = max(0.0, float(dt))
        self.angle_deg = _wrap_deg(self.angle_deg + float(gyro_rate_deg_s) * dt)
        self.variance += (max(0.0, float(gyro_rate_std_deg_s)) * dt) ** 2

    def update(self, measured_yaw_deg: float, measurement_std_deg: float) -> tuple[float, float]:
        measurement_variance = max(0.0, float(measurement_std_deg)) ** 2
        innovation = _wrap_deg(float(measured_yaw_deg) - self.angle_deg)
        innovation_variance = self.variance + measurement_variance
        if innovation_variance <= 1e-15:
            self.angle_deg = _wrap_deg(measured_yaw_deg)
            return 1.0, innovation
        gain = self.variance / innovation_variance
        self.angle_deg = _wrap_deg(self.angle_deg + gain * innovation)
        self.variance = max(0.0, (1.0 - gain) * self.variance)
        return gain, innovation
