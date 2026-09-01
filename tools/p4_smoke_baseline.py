"""冒烟基线采集。

对一组代表性端点采集（状态码 + 响应体前 200 字符 + 关键全局快照），
输出 JSON 到 stdout，供回归对比使用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carla_relay.cli import load_legacy_core

CASES = [
    ("GET", "/health", None),
    ("GET", "/experiments", None),
    ("GET", "/experiments/1", None),
    ("GET", "/experiment/status", None),
    ("GET", "/experiment/outputs", None),
    ("GET", "/perception/obstacles", None),
    ("GET", "/perception/obstacles?vehicle_id=1", None),
    ("GET", "/perception/traffic_lights", None),
    ("GET", "/map/spawn_points", None),
    ("GET", "/map/bounds", None),
    ("GET", "/map/road_network", None),
    ("GET", "/debug/bbox", None),
    ("GET", "/actors", None),
    ("GET", "/vehicle/999/autopilot", None),
    ("GET", "/sensor/999/frame", None),
    ("POST", "/experiment/5/level", {"level": "L9"}),
    ("POST", "/experiment/5/level", {"level": "L3"}),
    ("POST", "/experiment/10/params", {"kp_steer": 0.5, "perception": True}),
    ("POST", "/experiment/1/stop", None),
    ("POST", "/experiment/23/stop", None),
    ("POST", "/experiment/10/stop", None),
    ("POST", "/sync/tick", None),
    ("POST", "/stream/setup", {"vehicle_id": 11, "camera_sid": 22}),
    ("DELETE", "/cleanup", None),
]


def main() -> int:
    m = load_legacy_core()
    c = m.app.test_client()
    out = {"cases": []}
    for method, url, body in CASES:
        r = c.open(url, method=method, json=body)
        out["cases"].append({
            "req": f"{method} {url} {json.dumps(body, sort_keys=True) if body else ''}".strip(),
            "status": r.status_code,
            "body": r.get_data(as_text=True)[:200],
        })
    # 关键全局快照
    out["globals"] = {
        "stream_vehicle": m._stream_vehicle,
        "stream_camera": m._stream_camera,
        "semantic_level": m._semantic_level,
        "exp10_ctrl": {k: m._EXP10_CTRL[k] for k in sorted(m._EXP10_CTRL)},
        "exp_log_tail": m._EXP_LOG[-3:],
    }
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("p4_smoke.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[smoke] 已写入 {out_path}（{len(out['cases'])} 个用例）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
