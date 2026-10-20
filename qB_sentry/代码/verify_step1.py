#!/usr/bin/env python3
"""验证第一步：坐标性质、全图安全区及原检查器的端点判断。

端点探针使用两个相同坐标，只用于调用原检查器的碰撞逻辑，不是规划路径。
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from inspect_map import ROOT, load_map, world_to_cell, cell_to_world, build_safety, inspect_queries


def main():
    grid = load_map(ROOT/"input/map/rmuc_2025.yaml")
    queries = json.loads((ROOT/"input/map/queries.json").read_text())
    radius = queries["robot_radius_m"]
    safety = build_safety(grid, radius)
    report = {"scope": "地图预检；不代表完整路径通过检查", "checks": []}

    # 坐标属性：格子中心转换为世界坐标后，必须回到同一个格子。
    rng = np.random.default_rng(20261017)
    cells = [(0, 0), (0, grid["width"]-1), (grid["height"]-1, 0),
             (grid["height"]-1, grid["width"]-1)]
    cells += list(zip(rng.integers(0, grid["height"], 1000).tolist(),
                      rng.integers(0, grid["width"], 1000).tolist()))
    for row, col in cells:
        assert world_to_cell(*cell_to_world(row, col, grid), grid) == (row, col)
    assert world_to_cell(-2.26, -1.36, grid) == (162, 26)
    report["checks"].append({"name": "1004 个格子中心坐标往返及已知起点", "passed": True})

    xmin, xmax, ymin, ymax = grid["extent"]
    for x, y in [(xmin-1e-6, ymin+1), (xmax+1e-6, ymin+1),
                 (xmin+1, ymin-1e-6), (xmin+1, ymax+1e-6)]:
        row, col = world_to_cell(x, y, grid)
        assert not (0 <= row < grid["height"] and 0 <= col < grid["width"])
    report["checks"].append({"name": "地图四边外侧越界识别", "passed": True})

    # 独立计算法：枚举半径内的整数位移，构造圆盘核并膨胀障碍。
    # 它直接对应原检查器的 hypot(dr, dc)*resolution < radius，
    # 与主程序的距离变换实现不同，可发现半径阈值或距离算法错误。
    rpx = int(math.ceil(radius/grid["resolution"]))
    dy, dx = np.mgrid[-rpx:rpx+1, -rpx:rpx+1]
    kernel = (np.hypot(dx, dy)*grid["resolution"] < radius).astype(np.uint8)
    collisions = cv2.dilate((~grid["free"]).astype(np.uint8), kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    difference = int(np.count_nonzero((~collisions) != safety["checker_safe"]))
    assert difference == 0, f"安全区域有 {difference} 格与直接圆盘判定不同"
    report["checks"].append({"name": "188892 个格子的安全区与圆盘膨胀交叉验证", "passed": True,
                              "compared_cells": int(grid["free"].size), "different_cells": difference})

    # 实际执行原检查器，逐一核对 14 个端点。探针文件只放临时目录。
    diagnostics = inspect_queries(queries["queries"], grid, safety)
    probes = []
    with tempfile.TemporaryDirectory(prefix="rmvision-step1-") as temporary:
        for item in diagnostics:
            for which in ["start", "goal"]:
                endpoint = item[which]
                x, y = endpoint["world"]
                csv = Path(temporary)/f"endpoint_{item['query']}_{which}.csv"
                csv.write_text(f"x,y\n{x},{y}\n{x},{y}\n")
                command = [sys.executable, "-B", str(ROOT/"tools/check_path.py"),
                           str(ROOT/"input/map/rmuc_2025.yaml"), str(csv)]
                run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if run.returncode != 0:
                    raise RuntimeError(run.stderr or run.stdout)
                passed = run.stdout.strip().splitlines()[-1] == "通过"
                expected = bool(endpoint["inside_map"] and
                                safety["checker_safe"][endpoint["row"], endpoint["col"]])
                assert passed == expected, (item["query"], which, run.stdout)
                probes.append({"query": item["query"], "endpoint": which,
                               "checker_passed": passed, "matched": True,
                               "checker_stdout": run.stdout.strip()})
    report["checks"].append({"name": "原检查器 14 个端点探针", "passed": True, "probes": probes})
    assert all(item["same_component4"] for item in diagnostics[:5])
    assert not diagnostics[5]["goal"]["safe"]
    assert not diagnostics[6]["goal"]["inside_map"]
    report["checks"].append({"name": "7 组查询的前置条件", "passed": True})
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    for check in report["checks"]:
        print("通过：", check["name"])
    print("这些检查验证地图预处理，不代表完整路径规划已经完成。")


if __name__ == "__main__":
    main()
