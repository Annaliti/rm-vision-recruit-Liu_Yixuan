"""在点云自建地图上运行相同7组任务，并与提供地图的结果比较。"""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from inspect_map import ROOT, load_map, build_safety, inspect_queries
from planner import search_grid, cells_to_path, validate_world_path


def run():
    out = ROOT/"results_cloud_planning"
    for directory in (out, out/"paths", out/"checker_logs"):
        directory.mkdir(exist_ok=True)
    grid_path = ROOT/"results_cloud/cloud_map.yaml"
    grid = load_map(grid_path)
    queries = json.loads((ROOT/"input/map/queries.json").read_text())
    safety = build_safety(grid, queries["robot_radius_m"])
    checks = inspect_queries(queries["queries"], grid, safety)
    reference = json.loads((ROOT/"results_step2/summary.json").read_text())
    rows = []
    for check, ref in zip(checks, reference["queries"]):
        n = check["query"]
        record = dict(query=n, start=check["start"], goal=check["goal"], algorithms={},
                      reference_status=ref["status"],
                      reference_length_m=ref["algorithms"].get("astar", {}).get("csv_length_m"))
        if not check["start"]["safe"] or not check["goal"]["safe"]:
            record.update(status="无法到达", reason=check["status"])
            for key, label in (("start", "起点"), ("goal", "终点")):
                point = check[key]
                if not point["safe"] and point["inside_map"]:
                    bad = np.argwhere(~grid["free"])
                    delta = bad-np.array([point["row"], point["col"]])
                    nearest = bad[np.argmin(np.sum(delta**2, axis=1))]
                    kind = "未知格（点云支撑不足）" if grid["unknown"][tuple(nearest)] else "障碍格（地形或车身净空限制）"
                    record["reason"] = f"{label}距最近{kind} {point['obstacle_clearance_m']:.3f}m，小于0.30m半径"
                    break
        else:
            start = (check["start"]["row"], check["start"]["col"])
            goal = (check["goal"]["row"], check["goal"]["col"])
            for algorithm in ("astar", "dijkstra"):
                result = search_grid(safety["safe"], start, goal, grid["resolution"], algorithm, False)
                if not result.found:
                    record.update(status="无法到达", reason="安全八邻域图不连通，队列耗尽；未移动任务端点")
                    continue
                path = cells_to_path(result.cells, check["start"]["world"], check["goal"]["world"], grid)
                path_file = out/"paths"/f"{algorithm}_path{n}.csv"
                np.savetxt(path_file, path, delimiter=",", header="x,y", comments="", fmt="%.10f")
                saved = np.loadtxt(path_file, delimiter=",", skiprows=1)
                validation = validate_world_path(saved, check["start"]["world"], check["goal"]["world"], grid, safety)
                check_run = subprocess.run([sys.executable, "-B", str(ROOT/"tools/check_path.py"), str(grid_path), str(path_file)],
                                           capture_output=True, text=True, timeout=60)
                (out/"checker_logs"/f"{algorithm}_q{n}.txt").write_text(check_run.stdout+check_run.stderr)
                assert check_run.returncode == 0 and check_run.stdout.strip().splitlines()[-1] == "通过"
                record["algorithms"][algorithm] = dict(path_file=str(path_file.relative_to(ROOT)),
                    length_m=validation["length_m"], grid_cost_m=result.cost_m,
                    expanded=result.expanded, elapsed_ms=result.elapsed_s*1000,
                    checker_passed=True, geometry_validation=validation)
                record.update(status="路径通过检查", reason="")
            if len(record["algorithms"]) == 2:
                assert abs(record["algorithms"]["astar"]["grid_cost_m"]-record["algorithms"]["dijkstra"]["grid_cost_m"]) < 1e-8
        rows.append(record)
        print(f"Q{n}: {record['status']} {record['reason']} "+str({k:round(v['length_m'],3) for k,v in record['algorithms'].items()}), flush=True)
    report = dict(map_file="results_cloud/cloud_map.yaml", queries=rows,
                  timing="每种算法单次测量，供自建地图实验记录，不作为稳定性能排名",
                  limitation="可达性与路径仅针对自建离散地图；参考图不是地形真值，自建未知区较保守，不能移动原始端点来凑通过数")
    (out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 自建地图与提供地图对比", "", "两种地图使用相同世界坐标任务、0.30m半径、八邻域和禁止穿角规则。", "",
             "|查询|提供地图长度/m|自建地图A*长度/m|自建地图结果|", "|---|---:|---:|---|"]
    for row in rows:
        old = f"{row['reference_length_m']:.3f}" if row['reference_length_m'] is not None else "无法到达"
        a = row["algorithms"].get("astar")
        new = f"{a['length_m']:.3f}" if a else "—"
        lines.append(f"|{row['query']}|{old}|{new}|{row['reason'] or 'A*与Dijkstra均通过'}|")
    lines += ["", report["limitation"], "", "检查器读取的是自建地图YAML。自建路径并不要求在提供图上通过，因为两者地形分类不同。"]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    return report


if __name__ == "__main__":
    run()
