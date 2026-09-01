"""路由清单快照工具 — 回归门禁。

两种模式：
    runtime（默认）: 加载核心模块（含 carla_relay.routes 蓝图注册），
                     从 Flask app.url_map 导出真实路由清单。
    ast:            AST 静态扫描单个源文件的 @app.route 装饰器。

用法（在 server/ 目录下）：
    python tools/route_snapshot.py snapshot              # 生成基线 route_baseline.json
    python tools/route_snapshot.py check                 # 当前路由与基线对比
    python tools/route_snapshot.py check --strict-exit 0 # 仅打印差异不阻断
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_DEFAULT_TARGET = _TOOLS_DIR.parent / "carla_relay_core.py"  # server/carla_relay_core.py
_DEFAULT_BASELINE = _TOOLS_DIR / "route_baseline.json"

# Flask 自动追加的隐式方法，对比时排除
_IMPLICIT_METHODS = {"HEAD", "OPTIONS"}


# =============================================================================
# runtime 模式：从 Flask url_map 导出
# =============================================================================

def _load_app(target: Path):
    """加载旧核心模块（含蓝图注册），返回其 Flask app。"""
    server_dir = _TOOLS_DIR.parent
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))
    from carla_relay.cli import load_legacy_core
    legacy = load_legacy_core(target)
    return legacy.app


def extract_routes_runtime(target: Path) -> list[dict]:
    """从 Flask app.url_map 导出路由清单 [{rule, methods, endpoint}]（已排序）。"""
    app = _load_app(target)
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":  # Flask 默认静态规则，不属于业务 API
            continue
        methods = sorted(m for m in rule.methods if m not in _IMPLICIT_METHODS)
        routes.append({"rule": str(rule), "methods": methods, "endpoint": rule.endpoint})
    return sorted(routes, key=lambda r: (r["rule"], tuple(r["methods"])))


# =============================================================================
# ast 模式：静态扫描 @app.route 装饰器
# =============================================================================

def _decorator_route_info(dec: ast.expr) -> dict | None:
    if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "route"):
        return None
    info: dict = {"rule": None, "methods": None}
    if dec.args:
        info["rule"] = dec.args[0]
    for kw in dec.keywords:
        if kw.arg == "rule":
            info["rule"] = kw.value
        elif kw.arg == "methods":
            info["methods"] = kw.value
    return info if info["rule"] is not None else None


def _const_str(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _methods_list(node: ast.expr | None) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [_const_str(e) for e in node.elts]
        return sorted(v for v in vals if v) or ["GET"]
    return ["GET"]


def extract_routes_ast(source_path: Path) -> list[dict]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    routes: list[dict] = []

    def _visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                info = _decorator_route_info(dec)
                if info is not None:
                    rule = _const_str(info["rule"])
                    if rule is None:
                        raise ValueError(f"{source_path.name}:{node.lineno} 路由 rule 非字符串常量")
                    routes.append({
                        "rule": rule,
                        "methods": _methods_list(info["methods"]),
                        "func": node.name,
                        "line": node.lineno,
                    })
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return sorted(routes, key=lambda r: (r["rule"], tuple(r["methods"])))


# =============================================================================
# 快照 / 对比
# =============================================================================

def _key(r: dict) -> tuple:
    return (r["rule"], tuple(r["methods"]))


def _print_diff(added: list[dict], removed: list[dict]) -> None:
    for r in added:
        print(f"  + {r['rule']}  [{','.join(r['methods'])}]  ({r.get('endpoint') or r.get('func')})")
    for r in removed:
        print(f"  - {r['rule']}  [{','.join(r['methods'])}]  ({r.get('endpoint') or r.get('func')})")


def cmd_snapshot(target: Path, baseline: Path, mode: str) -> int:
    routes = extract_routes_runtime(target) if mode == "runtime" else extract_routes_ast(target)
    baseline.write_text(
        json.dumps({"mode": mode, "target": str(target), "count": len(routes), "routes": routes},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[snapshot:{mode}] 已生成基线: {baseline}（{len(routes)} 条路由）")
    return 0


def cmd_check(target: Path, baseline: Path, mode: str, strict_exit: int) -> int:
    if not baseline.is_file():
        print(f"[check] 基线不存在: {baseline}，请先执行 snapshot 子命令")
        return 1
    data = json.loads(baseline.read_text(encoding="utf-8"))
    base_routes = data.get("routes", [])
    if data.get("mode", "ast") != mode:
        print(f"[check] 警告: 基线模式({data.get('mode')})与当前模式({mode})不一致，请重新生成基线")

    current = {_key(r): r for r in (
        extract_routes_runtime(target) if mode == "runtime" else extract_routes_ast(target))}
    base = {_key(r): r for r in base_routes}
    added = [current[k] for k in sorted(set(current) - set(base))]
    removed = [base[k] for k in sorted(set(base) - set(current))]

    print(f"[check:{mode}] 当前 {len(current)} 条 / 基线 {len(base)} 条路由")
    if not added and not removed:
        print("[check] PASS — 路由清单与基线完全一致")
        return 0
    print("[check] FAIL — 路由清单存在差异:")
    _print_diff(added, removed)
    return strict_exit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="路由清单快照与对比（回归门禁）")
    parser.add_argument("command", choices=["snapshot", "check"])
    parser.add_argument("target", nargs="?", default=str(_DEFAULT_TARGET))
    parser.add_argument("--baseline", default=str(_DEFAULT_BASELINE))
    parser.add_argument("--mode", choices=["runtime", "ast"], default="runtime",
                        help="runtime=url_map 运行时导出（默认）；ast=静态扫描")
    parser.add_argument("--strict-exit", type=int, default=1)
    args = parser.parse_args(argv)

    target = Path(args.target)
    if not target.is_file():
        print(f"[error] 目标文件不存在: {target}")
        return 1
    baseline = Path(args.baseline)
    if args.command == "snapshot":
        return cmd_snapshot(target, baseline, args.mode)
    return cmd_check(target, baseline, args.mode, args.strict_exit)


if __name__ == "__main__":
    sys.exit(main())
