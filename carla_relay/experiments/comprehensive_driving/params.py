"""实验运行参数读取（统一规范）。

背景：规划/感知层的可调量曾以各模块「模块级常量」硬编码（如 simple_planner
的 DEC_WIN、perception 的识别距离 50m）。本模块把它们提升为「显式参数」，
按声明式 spec 读取，来源和默认集中在各消费模块文件头的「参数来源登记表」。

【声明约定】凡从 JSON / 前端滑杆获取的参数，一律用 spec 声明：
    (key, 默认值, 类型, 来源标记)
在模块头登记 + 在构造/读取处用 read(params, SPEC) 取回。来源标记：
    JSON    = 静态 json（experiment_params/*.json → /start body → _run_exp10(args)
              → 构造各层 params），启动时读一次，改动需重启实验
    SLIDER  = 前端滑杆运行中热更（POST /params → _EXP10_CTRL，主循环每 tick 读）
    CONST   = 模块级硬编码兜底默认（不经 params 注入）

【参数来源对照表】（全部为显式参数，规格见各消费模块文件头登记）：
    参数              来源    类型    默认(按消费模块)        消费位置
    dec_win           JSON     float   25.0 simple_planner     simple_planner._blockers
                                       60.0 planner(legacy)    planner._find_nudge 等
    perception_range  JSON     float   50.0                    perception 感知扫描/bbox 识别距离上限
"""


def read(params, spec):
    """按声明式 spec 读取实验参数，缺省或不合法回退默认。

    spec = (key, default, cast, source)：
      key        JSON/滑杆/常量键名
      default    兜底默认值
      cast       类型转换（int/float/bool/str），None 时不转换
      source     来源标记（"JSON"/"SLIDER"/"CONST"，仅登记用）
    params: 装配期注入的参数字典（run.py 从 json args 构造），可为 None。
    """
    key, default, cast, _source = spec
    if params is None:
        return default
    v = params.get(key, default)
    if cast is None:
        return v
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default