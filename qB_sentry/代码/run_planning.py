"""运行7组查询、对比两种手写算法、保存实际路径和原检查器日志。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from inspect_map import ROOT, load_map, build_safety, inspect_queries
from planner import search_grid, cells_to_path, validate_world_path


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def run_all(repeats=3):
    if repeats < 1:
        raise ValueError("重复次数至少为1")
    map_file = ROOT/"input/map/rmuc_2025.yaml"
    grid = load_map(map_file)
    queries = json.loads((ROOT/"input/map/queries.json").read_text())
    safety = build_safety(grid, queries["robot_radius_m"])
    checks = inspect_queries(queries["queries"], grid, safety)
    output = ROOT/"results_step2"
    paths = ROOT/"paths"
    for directory in (output, output/"checker_logs", output/"search_traces", paths, paths/"dijkstra"):
        directory.mkdir(parents=True, exist_ok=True)
    report = dict(stage="第二步：提供地图上的路径规划", repeats=repeats,
                  cost_model="八邻域：直走0.05m、斜走sqrt(2)*0.05m；禁止穿角",
                  heuristic="A*使用欧氏距离；Dijkstra使用0",
                  timing_scope="搜索含数组初始化和路径回溯；不含读图、检查、写文件、绘图",
                  expanded_definition="有效弹出并确定的节点数，包含起点和终点，不含过期堆项",
                  optimality_scope="安全格中心组成的离散图；真实起终点通过固定的格子中心连接",
                  queries=[])
    for check in checks:
        number = check["query"]
        start_xy, goal_xy = check["start"]["world"], check["goal"]["world"]
        row = dict(query=number, start=start_xy, goal=goal_xy, algorithms={})
        if not check["start"]["safe"] or not check["goal"]["safe"]:
            row.update(status="无法到达", reason=check["status"], searched=False)
            report["queries"].append(row)
            print(f"Q{number}: {row['reason']}", flush=True)
            continue
        start = (check["start"]["row"], check["start"]["col"])
        goal = (check["goal"]["row"], check["goal"]["col"])
        runs = {"dijkstra": [], "astar": []}
        for repeat in range(repeats):
            # 交替次序，减少固定先后顺序对计时的影响。
            algorithms = ("dijkstra", "astar") if repeat % 2 == 0 else ("astar", "dijkstra")
            for name in algorithms:
                result = search_grid(safety["safe"], start, goal, grid["resolution"],
                                     algorithm=name, record_trace=repeat == 0)
                runs[name].append(result)
        if not runs["astar"][0].found or not runs["dijkstra"][0].found:
            if runs["astar"][0].found != runs["dijkstra"][0].found:
                raise AssertionError("两种算法的可达性结论不一致")
            row.update(status="无法到达", reason="合法邻接图中搜索队列耗尽", searched=True)
            report["queries"].append(row)
            continue

        for name in ("dijkstra", "astar"):
            first = runs[name][0]
            if any(abs(r.cost_m-first.cost_m) > 1e-8 or r.cells != first.cells for r in runs[name]):
                raise AssertionError("相同输入重复运行得到不同路线")
            path = cells_to_path(first.cells, start_xy, goal_xy, grid)
            csv_path = paths/f"path{number}.csv" if name == "astar" else paths/"dijkstra"/f"path{number}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows((f"{x:.10f}", f"{y:.10f}") for x, y in path)
            # 验证实际落盘的CSV，包含十进制写入/读取误差，而不是只检查内存数据。
            saved = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            validation = validate_world_path(saved, start_xy, goal_xy, grid, safety)
            command = [sys.executable, "-B", str(ROOT/"tools/check_path.py"), str(map_file), str(csv_path)]
            checked = subprocess.run(command, capture_output=True, text=True, timeout=60)
            log_path = output/"checker_logs"/f"{name}_q{number}.txt"
            log_path.write_text(checked.stdout+("\nSTDERR:\n"+checked.stderr if checked.stderr else ""), encoding="utf-8")
            output_lines = checked.stdout.strip().splitlines()
            passed = checked.returncode == 0 and bool(output_lines) and output_lines[-1] == "通过"
            if not passed:
                raise AssertionError(f"原检查器未通过：{log_path}\n{checked.stdout}\n{checked.stderr}")
            np.savez_compressed(output/"search_traces"/f"{name}_q{number}.npz",
                                expanded_cells=np.asarray(first.expansion_order, dtype=np.int32),
                                path_cells=np.asarray(first.cells, dtype=np.int32))
            times = [r.elapsed_s*1000 for r in runs[name]]
            row["algorithms"][name] = dict(
                grid_cost_m=first.cost_m, csv_length_m=validation["length_m"],
                point_count=len(saved), expanded_nodes=first.expanded,
                elapsed_ms_runs=times, elapsed_ms_median=statistics.median(times),
                path_file=str(csv_path.relative_to(ROOT)),
                path_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                checker_passed=passed, checker_log=str(log_path.relative_to(ROOT)),
                geometry_validation=validation)
        difference = abs(row["algorithms"]["astar"]["grid_cost_m"]-row["algorithms"]["dijkstra"]["grid_cost_m"])
        if difference > 1e-8:
            raise AssertionError(f"Q{number}两种算法最短成本不同")
        row.update(status="路径通过检查", searched=True, grid_cost_difference_m=difference)
        report["queries"].append(row)
        a, d = row["algorithms"]["astar"], row["algorithms"]["dijkstra"]
        print(f"Q{number}: {a['csv_length_m']:.3f} m；扩展 Dijkstra={d['expanded_nodes']}, A*={a['expanded_nodes']}；两条路径均通过", flush=True)
    write_json(output/"summary.json", report)
    unreachable = [r for r in report["queries"] if r["status"] == "无法到达"]
    write_json(output/"unreachable.json", unreachable)
    lines = ["# 第二步实际运行结果", "", "本报告针对提供的二维地图；未完成点云建图。", "",
             "路径长度为CSV真实折线长度，含原始起终点与所在格子中心的连接。", "",
             f"耗时为同机 {repeats} 次搜索的中位数（可用 --repeats 修改），不包含画图和检查器耗时。", "",
             "| 查询 | A*长度/m | Dijkstra长度/m | A*扩展数 | Dijkstra扩展数 | A*耗时/ms | Dijkstra耗时/ms | 结果 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in report["queries"]:
        if not r["algorithms"]:
            lines.append(f"| {r['query']} | — | — | — | — | — | — | {r['reason']} |")
        else:
            a, d = r["algorithms"]["astar"], r["algorithms"]["dijkstra"]
            lines.append(f"| {r['query']} | {a['csv_length_m']:.3f} | {d['csv_length_m']:.3f} | {a['expanded_nodes']} | {d['expanded_nodes']} | {a['elapsed_ms_median']:.1f} | {d['elapsed_ms_median']:.1f} | 两者均通过 |")
    lines += ["", "第6、7组没有生成路径CSV；不可达原因见表格和 unreachable.json。", "",
              "实际原检查器标准输出保存在 checker_logs/。额外几何检查采用障碍格中心模型，验证了完整线段和真实端点。", "",
              "A*与Dijkstra的长度一致是针对本程序的离散邻接图，并不表示得到了连续空间全局最短曲线。"]
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "algorithm", "status", "length_m", "expanded_nodes", "median_ms", "checker_passed", "reason"])
        for r in report["queries"]:
            for name in ("astar", "dijkstra"):
                a = r["algorithms"].get(name)
                w.writerow([r["query"], name, r["status"], a["csv_length_m"] if a else "",
                            a["expanded_nodes"] if a else "", a["elapsed_ms_median"] if a else "",
                            a["checker_passed"] if a else "", r.get("reason", "")])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    run_all(parser.parse_args().repeats)
