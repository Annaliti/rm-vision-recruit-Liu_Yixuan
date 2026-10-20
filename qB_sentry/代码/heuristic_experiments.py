"""问题④：比较欧氏距离、曼哈顿距离、2倍欧氏距离的 A*。

所有变体共享八邻域、禁止穿角、半径0.30米安全图和原始起终点。
这里允许重新打开节点；高估型启发函数仍然不保证最短路。
运行：python heuristic_experiments.py；只运行小图验证：加 --test-only。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from inspect_map import ROOT, build_safety, inspect_queries, load_map
from planner import MOVES, cells_to_path, search_grid, validate_world_path

VARIANTS = ("euclidean", "manhattan", "weighted2")
LABELS = {"dijkstra": "Dijkstra：h=0", "euclidean": "A*：欧氏距离", 
          "manhattan": "A*：曼哈顿距离", "weighted2": "A*：2倍欧氏距离"}


@dataclass
class VariantResult:
    found: bool
    cells: list
    cost_m: float | None
    expanded: int
    unique_expanded: int
    reopened_expansions: int
    elapsed_s: float


def search_variant(safe, start, goal, resolution, heuristic="euclidean"):
    """手写 A*：优先队列中的记录为(f, g, node)。

    expanded_g 记录某节点上次扩展时的 g。若它后来得到更小 g，允许再次
    扩展（重新打开）。重新打开能传播改进；不能使高估 h 自动变为可采纳。
    """
    if heuristic not in VARIANTS:
        raise ValueError(f"未知启发函数：{heuristic}")
    if safe.ndim != 2 or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("需要二维安全图和正分辨率")
    tick = perf_counter()
    height, width = safe.shape
    for r, c in (start, goal):
        if not (0 <= r < height and 0 <= c < width and safe[r, c]):
            return VariantResult(False, [], None, 0, 0, 0, perf_counter()-tick)
    sr, sc = start
    gr, gc = goal
    start_id, goal_id = sr*width+sc, gr*width+gc
    best_g = np.full(height*width, np.inf)
    expanded_g = np.full(height*width, np.inf)
    parent = np.full(height*width, -1, dtype=np.int32)
    best_g[start_id] = 0.0

    def h(r, c):
        dr, dc = abs(r-gr), abs(c-gc)
        if heuristic == "manhattan":
            return (dr+dc)*resolution
        return math.hypot(dr, dc)*resolution*(2 if heuristic == "weighted2" else 1)

    queue = [(h(sr, sc), 0.0, start_id)]
    expanded, reopened = 0, 0
    reached = False
    while queue:
        _, queued_g, current = heapq.heappop(queue)
        # 跳过过期记录，或没有比上一次扩展更好的记录。
        if queued_g > best_g[current]+1e-12 or queued_g >= expanded_g[current]-1e-12:
            continue
        if math.isfinite(expanded_g[current]):
            reopened += 1
        expanded_g[current] = queued_g
        expanded += 1
        r, c = divmod(current, width)
        if current == goal_id:
            reached = True
            break
        for dr, dc in MOVES:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < height and 0 <= nc < width and safe[nr, nc]):
                continue
            if dr and dc and (not safe[r+dr, c] or not safe[r, c+dc]):
                continue
            neighbor = nr*width+nc
            candidate = queued_g+resolution*(math.sqrt(2) if dr and dc else 1)
            # 不因 neighbor 扩展过就跳过它；找到更好 g 时重新加入队列。
            if candidate < best_g[neighbor]-1e-12:
                best_g[neighbor], parent[neighbor] = candidate, current
                heapq.heappush(queue, (candidate+h(nr, nc), candidate, neighbor))
    cells = []
    if reached:
        node = goal_id
        while node != -1:
            cells.append(divmod(int(node), width))
            node = int(parent[node])
            if len(cells) > safe.size:
                raise AssertionError("父节点链出现环")
        cells.reverse()
    # 高估 h 提前停止时，父链中的祖先可能刚得到改进，尚未来得及传播到
    # 所有后代；以实际回溯路线的边长作为返回成本，并验证不大于队列成本。
    cost = sum(math.hypot(b[0]-a[0], b[1]-a[1])*resolution
               for a, b in zip(cells, cells[1:])) if reached else None
    if reached and cost > best_g[goal_id]+1e-8:
        raise AssertionError("实际回溯成本大于终点成本")
    return VariantResult(reached, cells, cost, expanded,
                         expanded-reopened, reopened, perf_counter()-tick)


def self_tests():
    """已知最优成本、不可达/穿角反例和随机图对照；不依赖现成规划库。"""
    checked = 0
    for kind in VARIANTS:
        empty = np.ones((8, 8), dtype=bool)
        ans = search_variant(empty, (0, 0), (7, 7), 1.0, kind)
        assert ans.found and math.isclose(ans.cost_m, 7*math.sqrt(2), abs_tol=1e-9)
        assert search_variant(empty, (3, 3), (3, 3), 1.0, kind).cost_m == 0
        assert not search_variant(empty, (-1, 0), (7, 7), 1.0, kind).found
        corners = np.array([[True, False], [False, True]])
        assert not search_variant(corners, (0, 0), (1, 1), 1.0, kind).found
        blocked = empty.copy()
        blocked[4, :] = False
        assert not search_variant(blocked, (0, 0), (7, 7), 1.0, kind).found
        checked += 5
    rng = np.random.default_rng(20261019)
    reopened_cases = 0
    for _ in range(35):
        safe = rng.random((12, 13)) > .18
        safe[0, 0] = safe[-1, -1] = True
        start, goal = (0, 0), (11, 12)
        reference = search_grid(safe, start, goal, .05, "dijkstra", False)
        for kind in VARIANTS:
            ans = search_variant(safe, start, goal, .05, kind)
            assert ans.found == reference.found
            if ans.found:
                assert ans.cells[0] == start and ans.cells[-1] == goal
                assert ans.cost_m >= reference.cost_m-1e-9
                if kind == "euclidean":
                    assert math.isclose(ans.cost_m, reference.cost_m, abs_tol=1e-9)
                    assert ans.reopened_expansions == 0
                for a, b in zip(ans.cells, ans.cells[1:]):
                    dr, dc = b[0]-a[0], b[1]-a[1]
                    assert max(abs(dr), abs(dc)) == 1 and safe[b]
                    if dr and dc:
                        assert safe[a[0]+dr, a[1]] and safe[a[0], a[1]+dc]
            reopened_cases += int(ans.reopened_expansions > 0)
        checked += 1
    assert reopened_cases > 0, "随机例必须覆盖重新打开逻辑"
    return dict(passed=True, fixed_cases=15, random_maps=35,
                total_scenarios=checked, random_variant_runs=105,
                runs_exercising_reopen=reopened_cases,
                oracle="已知空地图最优值 + 原手写 Dijkstra；邻接合法性逐段复核")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def write_table(report, output):
    header = ["query", "algorithm", "length_m", "grid_cost_m", "optimal_ratio", "expanded",
              "unique_expanded", "reopened_expansions", "median_ms", "checker_passed", "reason"]
    table = ["# 启发函数实验：本机实际运行结果", "",
             "同一八邻域安全图、禁止穿角，真实起终点保持不变。扩展数包括重新扩展；",
             "最优比=实际网格路径成本/Dijkstra网格最优成本。CSV长度另含固定端点连接。", "",
             f"计时为各算法交错运行{report['repeats']}次的中位数，不含文件、检查器或绘图。", "",
             "| 查询 | 算法 | CSV长度/m | 最优比 | 扩展次数 | 重开后扩展次数 | 耗时/ms | 原检查器 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in report["queries"]:
            if not row["algorithms"]:
                table.append(f"| {row['query']} | 全部 | — | — | — | — | — | {row['reason']} |")
                writer.writerow([row["query"], "all", "", "", "", "", "", "", "", "", row["reason"]])
                continue
            for name, values in row["algorithms"].items():
                table.append(f"| {row['query']} | {LABELS[name]} | {values['length_m']:.3f} | "
                             f"{values['optimal_ratio']:.5f} | {values['expanded']} | "
                             f"{values['reopened_expansions']} | {values['median_ms']:.1f} | "
                             f"{'通过' if values['checker_passed'] else '未检查'} |")
                writer.writerow([row["query"], name, values["length_m"], values["grid_cost_m"],
                                 values["optimal_ratio"], values["expanded"], values["unique_expanded"],
                                 values["reopened_expansions"], values["median_ms"],
                                 values["checker_passed"], ""])
    table += ["", "两种高估启发函数不保证最短，也不保证更快；重开开销可能超过少走的探索。",
              "欧氏距离与原 A* 的路线/成本逐组核对；所有可达路径均经过真实端点、完整线段、间距验证和原检查器。",
              "第6、7组不改动终点、不创建伪造路径。更多原理见项目根目录的《启发函数实验讲解.md》。"]
    (output/"summary.md").write_text("\n".join(table)+"\n", encoding="utf-8")


def plot_report(report, output, grid):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"rmvision-heuristic-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "font.weight": 300, "axes.titleweight": 600})
    rows = [r for r in report["queries"] if r["algorithms"]]
    colors = {"dijkstra": "#586779", "euclidean": "#16878b", "manhattan": "#b27322", "weighted2": "#a14f82"}
    names = ("dijkstra",)+VARIANTS
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    x = np.arange(len(rows))
    for j, name in enumerate(names):
        for ax, field in zip(axes, ("optimal_ratio", "expanded", "median_ms")):
            vals = [row["algorithms"][name][field] for row in rows]
            ax.bar(x+(j-1.5)*.19, vals, width=.18, color=colors[name], label=LABELS[name])
    for ax in axes:
        ax.set_xticks(x, [f"第{r['query']}组" for r in rows])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylim(.98, max(r["algorithms"][name]["optimal_ratio"] for r in rows for name in names)+.04)
    axes[0].axhline(1, color="#333333", lw=.8, ls="--")
    axes[0].set_ylabel("网格路径长度 / 最优长度")
    axes[0].set_title("路径质量、扩展工作量和实际耗时需要分别比较")
    axes[0].legend(ncols=2, loc="upper center")
    axes[1].set_ylabel("扩展次数（包含重新扩展）")
    axes[2].set_ylabel(f"{report['repeats']}次搜索中位数 / ms")
    axes[2].set_xlabel("同机计时会随后台负载变化；不等同于机器人实时性能")
    fig.savefig(output/"01_启发函数对比.png", dpi=160)
    plt.close(fig)

    chosen = max(rows, key=lambda r: max(r["algorithms"][v]["optimal_ratio"] for v in VARIANTS))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    for ax, name in zip(axes, VARIANTS):
        values = chosen["algorithms"][name]
        points = np.loadtxt(ROOT/values["path_file"], delimiter=",", skiprows=1)
        ax.imshow(grid["gray"], cmap="gray", vmin=0, vmax=255, extent=grid["extent"], origin="upper")
        ax.plot(points[:, 0], points[:, 1], color=colors[name], lw=1.7)
        ax.scatter(*chosen["start"], color="#16878b", marker="o", s=40, zorder=5)
        ax.scatter(*chosen["goal"], color="#b43e3e", marker="X", s=55, zorder=5)
        ax.set(title=f"{LABELS[name]}\n长度 {values['length_m']:.3f}m，最优比 {values['optimal_ratio']:.4f}",
               xlabel="世界 x / m", ylabel="世界 y / m", aspect="equal")
    fig.suptitle(f"第{chosen['query']}组：同一起终点、不同启发函数的实际路线")
    fig.savefig(output/"02_路线变化.png", dpi=160)
    plt.close(fig)


def run_experiments(repeats=3):
    if repeats < 1:
        raise ValueError("repeats 至少为1")
    output = ROOT/"results_heuristics"
    for folder in (output, output/"paths", output/"checker_logs"):
        folder.mkdir(parents=True, exist_ok=True)
    tests = self_tests()
    write_json(output/"self_tests.json", tests)
    print(f"小图验证通过，{tests['runs_exercising_reopen']}次随机变体运行触发了重开。", flush=True)
    map_file = ROOT/"input/map/rmuc_2025.yaml"
    grid = load_map(map_file)
    queries = json.loads((ROOT/"input/map/queries.json").read_text())
    safety = build_safety(grid, queries["robot_radius_m"])
    checks = inspect_queries(queries["queries"], grid, safety)
    report = dict(repeats=repeats, tests=tests, queries=[],
                  priority="(g+h, g, node_id)；相同优先级使用稳定平局规则",
                  cost_model="八邻域；禁止穿角；直行resolution，斜行sqrt(2)*resolution",
                  variants="欧氏、曼哈顿、2倍欧氏；所有A*变体允许更低g重新打开节点",
                  timing="搜索含初始化及回溯，不含I/O、检查器、绘图；算法顺序循环交错",
                  expansion="每次有效扩展记1；重开后扩展也计入，独立节点数量另列",
                  optimal_ratio="实际网格路线成本 / 新运行的现有Dijkstra成本；不含固定端点连接",
                  validity_scope="提供地图、障碍格中心模型；未验证实车动力学")
    for check in checks:
        number = check["query"]
        row = dict(query=number, start=check["start"]["world"], goal=check["goal"]["world"], algorithms={})
        if not check["start"]["safe"] or not check["goal"]["safe"]:
            row.update(status="无法到达", reason=check["status"])
            report["queries"].append(row)
            print(f"Q{number}：{row['reason']}", flush=True)
            continue
        start = (check["start"]["row"], check["start"]["col"])
        goal = (check["goal"]["row"], check["goal"]["col"])
        names = ("dijkstra",)+VARIANTS
        runs = {name: [] for name in names}
        for repeat in range(repeats):
            rotation = repeat % len(names)
            for name in names[rotation:]+names[:rotation]:
                result = (search_grid(safety["safe"], start, goal, grid["resolution"], "dijkstra", False)
                          if name == "dijkstra" else
                          search_variant(safety["safe"], start, goal, grid["resolution"], name))
                runs[name].append(result)
        reference = runs["dijkstra"][0]
        assert reference.found, "预检可达但Dijkstra不可达"
        old_astar = search_grid(safety["safe"], start, goal, grid["resolution"], "astar", False)
        assert old_astar.cells == runs["euclidean"][0].cells
        for name in names:
            first = runs[name][0]
            assert first.found and all(r.cells == first.cells for r in runs[name])
            assert first.cost_m >= reference.cost_m-1e-8
            path_file = output/"paths"/f"{name}_path{number}.csv"
            points = cells_to_path(first.cells, row["start"], row["goal"], grid)
            with path_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows((f"{x:.10f}", f"{y:.10f}") for x, y in points)
            saved = np.loadtxt(path_file, delimiter=",", skiprows=1)
            validation = validate_world_path(saved, row["start"], row["goal"], grid, safety)
            checked = subprocess.run([sys.executable, "-B", str(ROOT/"tools/check_path.py"),
                                      str(map_file), str(path_file)], capture_output=True, text=True, timeout=60)
            log_file = output/"checker_logs"/f"{name}_q{number}.txt"
            log_file.write_text(checked.stdout+("\nSTDERR\n"+checked.stderr if checked.stderr else ""), encoding="utf-8")
            lines = checked.stdout.strip().splitlines()
            passed = checked.returncode == 0 and bool(lines) and lines[-1] == "通过"
            if not passed:
                raise AssertionError(f"检查器失败，见{log_file}")
            values = dict(grid_cost_m=first.cost_m, length_m=validation["length_m"],
                          optimal_ratio=first.cost_m/reference.cost_m,
                          expanded=first.expanded,
                          unique_expanded=first.expanded if name == "dijkstra" else first.unique_expanded,
                          reopened_expansions=0 if name == "dijkstra" else first.reopened_expansions,
                          elapsed_ms_runs=[r.elapsed_s*1000 for r in runs[name]],
                          median_ms=statistics.median(r.elapsed_s*1000 for r in runs[name]),
                          path_file=str(path_file.relative_to(ROOT)),
                          checker_log=str(log_file.relative_to(ROOT)), checker_passed=passed,
                          geometry_validation=validation)
            row["algorithms"][name] = values
        assert math.isclose(runs["euclidean"][0].cost_m, reference.cost_m, abs_tol=1e-8)
        row.update(status="全部路径通过检查", euclidean_matches_existing_astar=True)
        report["queries"].append(row)
        print(f"Q{number}："+"；".join(f"{name} {v['length_m']:.3f}m / {v['expanded']}次 / 重开{v['reopened_expansions']}"
                                   for name, v in row["algorithms"].items()), flush=True)
    write_json(output/"summary.json", report)
    write_table(report, output)
    plot_report(report, output, grid)
    print(f"已输出 {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only:
        print(json.dumps(self_tests(), ensure_ascii=False, indent=2))
    else:
        run_experiments(args.repeats)
