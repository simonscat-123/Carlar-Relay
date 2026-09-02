"""综合驾驶（闭环自动驾驶，API 实验ID 10）实验包。

包内两类成员：

  - 真实模块（可独立 import，依赖经构造函数注入，层间数据经帧契约流动）：
      frames / context / reference / sensor / localization / perception /
      prediction / planner / control / viz / actors
  - run.py：命名空间片段（由 experiments.load_comprehensive_driving_into
    以 exec 载入引导壳命名空间）——装配各层实例 + 主循环编排 +
    /experiment/10/* API 路由与 _EXP10_* 全局状态。

分层架构详见 run.py 模块文档字符串。
"""
