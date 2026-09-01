# -*- coding: utf-8 -*-
"""local_runner 端到端联调脚本（dummy 视频驱动，无窗口）。
用法: python -m local_runner.e2e_test [localization|semantic_segmentation|lidar_detection] [duration]
"""
import json
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from local_runner.relay_client import RelayClient, SSEQueue, save_output
from local_runner.charts import TelemetryHistory
from local_runner.fonts import load_font
from local_runner import views

SLUG = sys.argv[1] if len(sys.argv) > 1 else "localization"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

cfg = json.load(open(f"experiment_params/{SLUG}.json", encoding="utf-8"))
cfg["params"]["duration"] = DURATION
exp_id = cfg["experiment_id"]

client = RelayClient(cfg["relay"])
print("relay running:", client.status().get("running"))

pygame.init()
screen = pygame.display.set_mode((1280, 800))
fonts = {"sm": load_font(16), "md": load_font(20)}
q = SSEQueue(client)
q.start()
hist = TelemetryHistory()

lanes = []
if exp_id == 10:
    # 自动选一起一终两车道点并规划（替代交互点选）
    net = client.road_network()
    lanes = net.get("lanes") or []
    pts = [p for l in lanes[:3] for p in l.get("pts", [])]
    start = pts[0]
    end = pts[-1]
    plan = client.route_plan({"x": start["x"], "y": start["y"]},
                             {"x": end["x"], "y": end["y"]},
                             cfg["params"].get("sampling_resolution", 2.0))
    print(f"plan: {len(plan.get('route', []))} 路点 · {len(plan.get('obstacles', []))} 障碍")
    cfg["params"]["start"] = {"x": start["x"], "y": start["y"]}
    cfg["params"]["end"] = {"x": end["x"], "y": end["y"]}

print("start:", client.start(exp_id, cfg["params"]).get("status"))
t0 = time.time()
result_saved = False
while time.time() - t0 < DURATION + 90:
    surfaces, exp, seq = q.snapshot()
    hist.update(exp_id, exp, seq)
    views.dispatch_render(exp_id, screen, fonts, surfaces, exp, hist,
                          {"params": cfg["params"], "lanes": lanes})
    pygame.display.flip()
    if not result_saved and ("result" in exp or "report" in exp
                             or exp.get("status") in ("done", "stopped", "error")):
        result_saved = True
        rec = {"experiment_id": exp_id, "slug": SLUG, "status": "done",
               "result": exp.get("result") or exp.get("report")}
        p = save_output(rec)
        res = exp.get("result") or exp.get("report") or {}
        print(f"DONE at {time.time()-t0:.1f}s result_keys={sorted(res.keys())[:8]}")
        print("saved ->", p.name)
        break
    time.sleep(0.03)
else:
    print("TIMEOUT: experiment did not finish")

q.stop()
pygame.quit()
slots = list({"camera": 0}.keys())
print("experiment msg sample keys:", sorted(hist and {} or {}))
# 图表数据量验证
for name, chart in [("exp23_err", hist.exp23_err), ("exp4_obstacle", hist.exp4_obstacle),
                    ("exp5_ratio", hist.exp5_ratio), ("exp10_speed", hist.exp10_speed)]:
    counts = {k: len(v) for k, v in chart.data.items()}
    if any(counts.values()):
        print(f"{name}: {counts}")