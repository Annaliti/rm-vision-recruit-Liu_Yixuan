"""验证搜索的已知答案、随机小图，以及实际保存的10条任务路径。

用法：python verify_step2.py
先运行 run_planning.py。结果写到 results_step2/validation.json。
"""
import hashlib
import json
import math

import numpy as np

from inspect_map import ROOT, load_map, build_safety
from planner import search_grid, validate_world_path


def reference_cost(safe, start, goal, resolution):
    """小图参考答案：用反复遍历全部边的 Bellman-Ford，不用堆和启发函数。

    很慢，只用于几十分之一大小的小图，不能拿它替代正式地图搜索。
    这里独立建立边，再反复松弛，交叉检查正式实现的队列和终止逻辑。
    """
    nodes = [tuple(map(int, p)) for p in np.argwhere(safe)]
    if start not in nodes or goal not in nodes:
        return math.inf
    edges = []
    for a in nodes:
        for b in nodes:
            dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
            if max(dr, dc) != 1:
                continue
            if dr == dc == 1 and not (safe[a[0], b[1]] and safe[b[0], a[1]]):
                continue
            edges.append((a, b, math.hypot(dr, dc)*resolution))
    costs = {node: math.inf for node in nodes}
    costs[start] = 0.0
    for _ in range(len(nodes)-1):
        changed = False
        for a, b, weight in edges:
            candidate = costs[a]+weight
            if candidate < costs[b]-1e-12:
                costs[b] = candidate
                changed = True
        if not changed:
            break
    return costs[goal]


def expect_cost(safe, start, goal, expected):
    for algorithm in ("dijkstra", "astar"):
        result = search_grid(safe, start, goal, 0.05, algorithm)
        assert result.found == math.isfinite(expected), (algorithm, expected, result)
        if result.found:
            assert math.isclose(result.cost_m, expected, abs_tol=1e-10)
            assert result.cells[0] == start and result.cells[-1] == goal
            assert len(set(result.expansion_order)) == result.expanded


def verify():
    cases = [
        (np.ones((5, 6), bool), (0, 0), (4, 5), (4*math.sqrt(2)+1)*0.05),
        (np.ones((3, 3), bool), (1, 1), (1, 1), 0.0),
        (np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], bool), (0, 0), (2, 0), 0.30),
        (np.array([[1, 0], [0, 1]], bool), (0, 0), (1, 1), math.inf),
        (np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], bool), (0, 0), (2, 2), math.inf),
        (np.array([[0, 1], [1, 1]], bool), (0, 0), (1, 1), math.inf),
        (np.ones((3, 3), bool), (-1, 0), (1, 1), math.inf),
    ]
    for case in cases:
        expect_cost(*case)
    rng = np.random.default_rng(20251019)
    for _ in range(25):
        small = rng.random((6, 7)) > 0.30
        small[0, 0] = small[-1, -1] = True
        expected = reference_cost(small, (0, 0), (5, 6), 0.05)
        expect_cost(small, (0, 0), (5, 6), expected)

    # 端点各自安全，但连接线从圆形禁区内切过：额外线段检查应能发现。
    free = np.ones((200, 200), bool)
    free[99, 100] = False  # 障碍中心为(1.005, 1.005)。
    artificial = dict(free=free, height=200, width=200, resolution=0.01,
                      ox=0.0, oy=0.0, extent=[0.0, 2.0, 0.0, 2.0])
    safety = build_safety(artificial, 0.30)
    dangerous = [[0.94, 1.3001], [1.08, 1.3001]]
    assert all(math.dist(p, (1.005, 1.005)) >= 0.30 for p in dangerous)
    try:
        validate_world_path(dangerous, dangerous[0], dangerous[-1], artificial, safety)
    except ValueError as error:
        assert "线段" in str(error), str(error)
    else:
        raise AssertionError("没有发现端点之间的线段碰撞")

    grid = load_map(ROOT/"input/map/rmuc_2025.yaml")
    tasks = json.loads((ROOT/"input/map/queries.json").read_text())
    safety = build_safety(grid, tasks["robot_radius_m"])
    report = json.loads((ROOT/"results_step2/summary.json").read_text())
    assert [r["query"] for r in report["queries"]] == list(range(1, 8))
    checked_paths = 0
    maximum_step = 0.0
    for row in report["queries"]:
        task = tasks["queries"][row["query"]-1]
        if row["query"] in (6, 7):
            assert row["status"] == "无法到达" and not row["algorithms"]
            assert not (ROOT/f"paths/path{row['query']}.csv").exists()
            assert not (ROOT/f"paths/dijkstra/path{row['query']}.csv").exists()
            continue
        assert row["status"] == "路径通过检查" and len(row["algorithms"]) == 2
        for name, item in row["algorithms"].items():
            path_file = ROOT/item["path_file"]
            assert hashlib.sha256(path_file.read_bytes()).hexdigest() == item["path_sha256"]
            path = np.loadtxt(path_file, delimiter=",", skiprows=1)
            result = validate_world_path(path, task["start"], task["goal"], grid, safety)
            assert math.isclose(result["length_m"], item["csv_length_m"], abs_tol=1e-10)
            assert (ROOT/item["checker_log"]).read_text().strip().splitlines()[-1] == "通过"
            trace = np.load(ROOT/f"results_step2/search_traces/{name}_q{row['query']}.npz")
            assert len(trace["expanded_cells"]) == item["expanded_nodes"]
            maximum_step = max(maximum_step, result["max_step_m"])
            checked_paths += 1
    assert checked_paths == 10
    outcome = dict(passed=True, known_answer_cases=len(cases),
                   bellman_ford_crosschecks=25, algorithms_checked_per_case=2,
                   segment_collision_counterexample_detected=True,
                   actual_saved_paths_checked=checked_paths,
                   maximum_csv_step_m=maximum_step,
                   unreachable_queries=[6, 7],
                   note="原检查器由run_planning.py实际执行；本脚本复核日志、文件哈希和完整几何条件")
    (ROOT/"results_step2/validation.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
