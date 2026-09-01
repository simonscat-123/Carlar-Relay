"""路径跟踪 / 目标导航 / 综合驾驶共用控制器：Pure Pursuit + PID。

本文件由 carla_relay.experiments.load_into(globals()) 载入执行，不可独立 import。
"""
# =============================================================================
# 实验 8/9/10 共用: Pure Pursuit + PID 控制
# =============================================================================

def _pure_pursuit_steer(vehicle_tform, route_wps, lookahead, wp_idx):
    """计算 Pure Pursuit 横向控制输出 [-1, 1]"""
    loc = vehicle_tform.location
    yaw = math.radians(vehicle_tform.rotation.yaw)
    # 找最近路点
    best = wp_idx
    best_d = float("inf")
    for j in range(wp_idx, min(wp_idx + 50, len(route_wps))):
        d = math.hypot(loc.x - route_wps[j][0], loc.y - route_wps[j][1])
        if d < best_d:
            best_d = d
            best = j
    wp_idx = best
    # 找前视点
    target_idx = wp_idx
    for j in range(wp_idx, len(route_wps)):
        if math.hypot(loc.x - route_wps[j][0], loc.y - route_wps[j][1]) >= lookahead:
            target_idx = j
            break
    else:
        target_idx = len(route_wps) - 1
    tx, ty = route_wps[target_idx]
    # 横向误差
    dx, dy = tx - loc.x, ty - loc.y
    cte = dx * math.sin(yaw) - dy * math.cos(yaw)
    # 转向角
    alpha = math.atan2(dy, dx) - yaw
    wheelbase = 2.85
    steer_angle = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
    steer = max(-1.0, min(1.0, steer_angle / 1.22))  # 归一化到 [-1, 1]
    return steer, wp_idx, cte


def _pid_speed_control(current_speed, target_speed, prev_error, integral, dt,
                        kp=0.5, ki=0.05, kd=0.1):
    """PID 纵向控制，返回 (throttle, brake, prev_error, integral)"""
    error = target_speed - current_speed
    integral += error * dt
    integral = max(-5.0, min(5.0, integral))
    derivative = (error - prev_error) / dt if dt > 0 else 0
    control = kp * error + ki * integral + kd * derivative
    if control >= 0:
        return min(1.0, control), 0.0, error, integral
    else:
        return 0.0, min(1.0, -control), error, integral

