# CARLA_Relay

CARLA HTTP + SSE 中继服务,承载自动驾驶教学实验的完整逻辑（感知 / 定位 / 规划 / 控制）。

## 整体架构

```
server/                          # 分发本目录即可
├── carla_relay_core.py          # 引导壳：bootstrap / 全局状态 / SSE 线程 /
│                                # 传感器回调 / 静态托管 / main()
├── carla_relay/                 # 核心包
│   ├── __main__.py             # python -m carla_relay 入口
│   ├── cli.py                  # 按路径加载引导壳并透传 CLI 参数
│   ├── runtime.py              # 运行时公共工具（AppContext）
│   ├── config.py               # 常量 / 语义类别 / 类别预设映射
│   ├── core/                   # 核心层
│   │   ├── carla_client.py     #   CARLA 客户端连接 / 进程管理
│   │   ├── sensors.py          #   传感器装配与帧序列化
│   │   └── sse.py              #   SSE 订阅者 hub
│   ├── perception/             # 感知层
│   │   └── semantics.py        #   语义分割标签解析
│   ├── routes/                 # 路由蓝图（Flask Blueprint）
│   │   ├── health.py           #   /health /experiments 等只读端点
│   │   ├── sync.py             #   同步模式切换 / tick
│   │   ├── vehicle.py          #   车辆生成 / 销毁 / 控制 / autopilot
│   │   ├── sensors.py          #   相机 / GNSS / IMU / LiDAR 装配与取帧
│   │   ├── stream.py           #   SSE 流订阅 / stream 目标管理
│   │   └── misc.py             #   preview / debug / actors / cleanup
│   ├── experiments/            # 实验逻辑（common/ 通用 + 各实验名文件夹）
│   │   ├── common/             #   实验通用逻辑（命名空间片段）
│   │   │   ├── state.py        #     共享状态：manifest / 日志 / 状态路由 / 残留清理
│   │   │   └── controllers.py  #     Pure Pursuit + PID 控制器（多实验共用）
│   │   ├── localization/       #   定位分析 · 前端关卡 localization（实验ID 23）
│   │   ├── lidar_detection/    #   Lidar 检测 · 前端关卡 lidar-detection（实验ID 4）
│   │   ├── semantic_segmentation/  #   语义分割 · 前端关卡 semantic-segmentation（实验ID 5）
│   │   ├── comprehensive_driving/  #   综合驾驶 · 前端关卡 comprehensive-driving（实验ID 10）：
│   │   │   └── ...             #     分层真实模块（frames/planner/control/...）
│   │   └── <历史实验>/run.py   #   历史实验：basic_control / gnss_imu / ins_fusion /
│   │                           #   lidar_camera_projection / route_planning /
│   │                           #   path_following / target_navigation
│   └── world_api/              # 世界查询/管理 API（命名空间片段）
│       ├── perception.py       #   障碍物 / 信号灯查询 / 路径障碍采样
│       ├── map_api.py          #   出生点 / 边界 / 路网 / 地图渲染
│       └── actors_api.py       #   行人生成 / 俯瞰视角
├── tests/                      # 单元测试（SSE / 传感器 / 语义解析）
├── tools/                      # 回归门禁工具
├── experiment_params/          # 命令行体验仿真：各实验参数 JSON + output/ 结果目录
└── local_runner/               # 命令行 + pygame 纯体验仿真客户端
```

## 依赖项

| 依赖                | 说明                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Python ≥ 3.10** | 运行环境                                                                                                               |
| **CARLA 0.9.x**   | 仿真器本体（含 `CarlaUE4.exe` 的安装目录）。PythonAPI 的 `carla` 模块由服务启动时自动从其 `PythonAPI/carla/dist/carla-*.egg` 注入，**无需 pip 安装** |
| **Flask**         | HTTP 服务与路由                                                                                                         |
| **NumPy**         | 点云 / 图像 / 矩阵运算                                                                                                     |
| **Pillow**        | 相机帧编码、渲染叠加                                                                                                         |
| **pygame**        | 仅 `local_runner` 命令行体验客户端需要                                                                                        |
| **requests**      | 仅 `local_runner` 命令行体验客户端需要                                                                                        |

安装 pip 依赖：

```bash
pip install flask numpy pillow          # 仅中继服务
pip install requests pygame             # 仅使用 local_runner 时需要
```

启动时按以下优先级定位 CARLA 根目录（用于注入 PythonAPI 与管理仿真器进程）：

1. `--carla-root` 命令行参数
2. `CARLA_ROOT` 环境变量
3. 从当前工作目录逐级向上查找 `CarlaUE4.exe` / `CarlaUE4.sh`
4. 均未找到则回退到当前目录

## 如何启动

```bash
cd server/

# 最简启动（自动定位 CARLA 并在需要时启动/重启仿真器）
python -m carla_relay

# 常用参数
python -m carla_relay \
    --host 0.0.0.0 \            # HTTP 监听地址
    --port 5000 \               # HTTP 端口
    --carla-host 127.0.0.1 \    # CARLA 服务端地址
    --carla-port 2000 \         # CARLA RPC 端口
    --carla-root "CARLA 安装根目录" \  # 显式指定 CARLA 安装目录
    --static-dir ../dist \      # 托管前端构建产物，单进程部署
    --debug \                   # Flask debug 模式
    --no-carla-manage           # 禁用启动时的 CARLA 自动检测/清理/重启
```

启动后快速自检：

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/vehicle/spawn
```

## 命令行体验仿真（local\_runner，不接前端思考题 / 提交）

纯体验仿真客户端：**先确保 relay 服务已启动**，再通过命令行运行任意实验，
用 pygame 窗口实时渲染 relay 推送的画面（相机 / 语义 / 车顶俯瞰 / 包围框），
实验跑完自动把结果写入 `experiment_params/output/`。全程不依赖 `carla` 模块，
`pip install requests pygame` 即可。

```bash
cd server/

# 定位分析（GNSS/INS 融合）
python -m local_runner localization

# Lidar 检测
python -m local_runner lidar_detection

# 语义分割
python -m local_runner semantic_segmentation

# 综合驾驶（闭环自动驾驶）——启动后自动绘制鸟瞰图，点选起止点后自动开始
python -m local_runner comprehensive_driving
```

### 每个实验的启动方式

| 实验             | 命令                                             | 参数来源                                           | 运行画面                    |
| -------------- | ---------------------------------------------- | ---------------------------------------------- | ----------------------- |
| 定位分析（ID 23）    | `python -m local_runner localization`          | `experiment_params/localization.json`          | 前视相机 + 定位/横向误差 HUD      |
| Lidar 检测（ID 4） | `python -m local_runner lidar_detection`       | `experiment_params/lidar_detection.json`       | 左/前/右三目 + 点云与障碍统计 HUD   |
| 语义分割（ID 5）     | `python -m local_runner semantic_segmentation` | `experiment_params/semantic_segmentation.json` | RGB 与语义双画面              |
| 综合驾驶（ID 10）    | `python -m local_runner comprehensive_driving` | `experiment_params/comprehensive_driving.json` | 左车顶俯瞰 + 右车前包围框 + 底部 HUD |

参数面板每项可配置值即对应 `params` JSON 字段（与传统前端一致），直接编辑
对应 JSON 即改变参数。综合驾驶另有 `start`/`end` 起止点坐标，由规划阶段点选
产生，运行时会并入请求，不需（也不应）写死在 JSON 里。

常用选项：

```bash
# 指定参数文件 / relay 地址
python -m local_runner localization --params experiment_params/localization.json
python -m local_runner <slug> --relay http://127.0.0.1:5000

# 查看可用的实验与参数文件路径
python -m local_runner --list
```

## LICENSE
本项目采用 **知识共享署名‑非商业性使用‑相同方式共享 4.0 国际许可协议（CC BY‑NC‑SA 4.0）**。

✅ 允许行为：下载、学习、Fork、修改、分发本项目，仅限用于学习、研究等非商业用途。
❌ 禁止行为：禁止任何形式的商业使用，包括但不限于售卖、集成到商业产品、付费服务、广告场景。
⚠️ 衍生作品约束：如果你分发修改后的衍生项目，必须继续使用本协议，保留原作者署名。

完整法律协议详见仓库根目录 `LICENSE` 文件（以英文官方文本为法律依据）。
协议官网：https://creativecommons.org/licenses/by‑nc‑sa/4.0/legalcode
