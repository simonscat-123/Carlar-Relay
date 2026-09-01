"""carla_relay — CARLA HTTP + SSE 中继网关。

本包为 CARLA 中继服务的核心模块。引导壳（`server/carla_relay_core.py`）
承载 bootstrap / 全局状态 / SSE 线程 / 传感器回调 / 静态托管 / main()，
其余逻辑位于本包：

    config.py       常量 / 语义类别 / 级别映射
    core/           CARLA 客户端、传感器装配、SSE hub
    perception/     语义分割解析
    routes/         健康检查 / 同步 / 车辆 / 传感器 / 流 / 杂项蓝图
    experiments/    实验逻辑 + 共享状态 + 控制器（命名对齐前端关卡：
                    localization 定位分析 / lidar_detection Lidar 检测 /
                    semantic_segmentation 语义分割 / comprehensive_driving_* 综合驾驶
                    + 历史实验）
    world_api/      感知查询 / 路径规划 / 地图 / 行人 / 俯瞰 API

experiments/ 与 world_api/ 内的 .py 为命名空间片段：由引导壳经
load_into(globals()) 以 exec 载入原命名空间执行，不可独立 import。
后续可在真机 CARLA 验证基础上逐步重构为 ExperimentRunner + AppContext 形态。

回归门禁：
    tools/route_snapshot.py check   路由清单对比
    tools/p4_smoke_baseline.py      多端点响应 + 全局快照对比

启动方式：
    python -m carla_relay [--host 0.0.0.0] [--port 5000] [--carla-port 2000]
    python -m carla_relay --carla-root "CARLA 安装根目录"
    python -m carla_relay --static-dir ../dist
"""
