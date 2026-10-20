"""第二步核心：手写 Dijkstra / A*；只用 heapq 管理优先队列。

输入是第一步得到的 safe 布尔图。节点为格子中心，采用八邻域，禁止斜穿角。
最短路保证针对这个离散图；它不是连续空间中所有可能曲线的全局最短路。
"""
from dataclasses import dataclass
import heapq
import math
from time import perf_counter

import numpy as np

from inspect_map import cell_to_world, world_to_cell

# 行、列变化量；直行长度1格，斜行长度sqrt(2)格。
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class SearchResult:
    found: bool
    cells: list
    cost_m: float | None
    expanded: int
    elapsed_s: float
    expansion_order: list


def search_grid(safe, start, goal, resolution, algorithm="astar", record_trace=True):
    """在同一个网格图上运行两种算法；区别仅在启发函数 h。

    Dijkstra: h=0，按已经走过的成本g排队。
    A*: h=到终点的欧氏距离，按g+h排队。
    两种h都满足“一致性”，因此节点第一次被有效弹出时即可确定其最小g。
    """
    if algorithm not in ("astar", "dijkstra"):
        raise ValueError("algorithm 必须为 astar 或 dijkstra")
    if safe.ndim != 2 or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("需要二维安全图和正分辨率")
    tick = perf_counter()
    height, width = safe.shape
    for r, c in (start, goal):
        if not (0 <= r < height and 0 <= c < width and safe[r, c]):
            return SearchResult(False, [], None, 0, perf_counter()-tick, [])

    sr, sc = start
    gr, gc = goal
    start_id, goal_id = sr*width+sc, gr*width+gc
    # 把(row,col)压成单个整数编号，便于记录父节点及优先队列。
    best_g = np.full(height*width, np.inf)
    parent = np.full(height*width, -1, dtype=np.int32)
    settled = np.zeros(height*width, dtype=bool)
    best_g[start_id] = 0.0
    use_h = algorithm == "astar"

    def heuristic(r, c):
        return math.hypot(r-gr, c-gc)*resolution if use_h else 0.0

    # 队列元素是(f,g,节点编号)，heapq 总是取字典序最小的一项。
    # g和编号负责在f相同的时候稳定地打破平局，便于重复实验。
    queue = [(heuristic(sr, sc), 0.0, start_id)]
    order = []
    expanded = 0
    reached = False
    while queue:
        _, queued_g, current = heapq.heappop(queue)
        # 找到更短路线后旧队列项仍在堆里；弹出时丢弃过期项。
        if settled[current] or queued_g > best_g[current] + 1e-12:
            continue
        settled[current] = True
        expanded += 1  # 包含起点、成功弹出的终点；不计过期堆项。
        r, c = divmod(current, width)
        if record_trace:
            order.append((r, c))
        if current == goal_id:
            reached = True
            break  # 终点必须从堆中有效弹出，不能一发现它就结束。

        for dr, dc in MOVES:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < height and 0 <= nc < width and safe[nr, nc]):
                continue
            if dr and dc and (not safe[r+dr, c] or not safe[r, c+dc]):
                continue  # 斜向移动的两侧正交格也必须安全。
            neighbor = nr*width+nc
            if settled[neighbor]:
                continue
            step = resolution*(math.sqrt(2.0) if dr and dc else 1.0)
            tentative_g = queued_g + step
            if tentative_g < best_g[neighbor] - 1e-12:
                best_g[neighbor] = tentative_g
                parent[neighbor] = current
                heapq.heappush(queue, (tentative_g+heuristic(nr, nc), tentative_g, neighbor))

    cells = []
    if reached:
        node = goal_id
        while node != -1:
            cells.append(divmod(int(node), width))
            node = int(parent[node])
        cells.reverse()  # parent从终点指向起点，翻转为正常行走顺序。
    return SearchResult(reached, cells, float(best_g[goal_id]) if reached else None,
                        expanded, perf_counter()-tick, order)


def cells_to_path(cells, start_xy, goal_xy, grid):
    """保留真实起终点，并通过所在格子中心连接到网格路径，不做平滑。"""
    if not cells:
        return []
    candidates = [tuple(map(float, start_xy))]
    candidates.extend(cell_to_world(r, c, grid) for r, c in cells)
    candidates.append(tuple(map(float, goal_xy)))
    path = []
    for point in candidates:
        if not path or math.dist(path[-1], point) > 1e-10:
            path.append(point)
    return path


def validate_world_path(path, start_xy, goal_xy, grid, safety, max_step=0.15):
    """额外验证真实端点、完整线段和边界；采用与检查器相同的障碍格中心模型。

    原检查器只在路径点处检查障碍距离，本函数还解析地计算每段线段到
    周围障碍格中心的最近距离。此模型未覆盖真实场地几何误差等因素。
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("路径至少需要两个二维点")
    if not np.isfinite(points).all():
        raise ValueError("路径含无穷大或NaN")
    endpoint_error = max(math.dist(points[0], start_xy), math.dist(points[-1], goal_xy))
    if endpoint_error > 1e-8:
        raise ValueError("路径没有保持题目指定的起终点")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if steps.max() > max_step+1e-6:
        raise ValueError("相邻点间距超过0.15米")
    radius, res = safety["radius"], grid["resolution"]
    xmin, xmax, ymin, ymax = grid["extent"]
    boundary = np.minimum.reduce([points[:, 0]-xmin, xmax-points[:, 0],
                                  points[:, 1]-ymin, ymax-points[:, 1]])
    if boundary.min() < radius-1e-9:
        raise ValueError("机器人圆形足迹超过地图边界")
    for x, y in points:
        r, c = world_to_cell(x, y, grid)
        if not (0 <= r < grid["height"] and 0 <= c < grid["width"] and safety["safe"][r, c]):
            raise ValueError("路径点位于非安全格")

    minimum = math.inf
    # 查询线段半径邻域内的障碍格；外围加一格，避免舍入漏掉边界。
    for index, (a, b) in enumerate(zip(points, points[1:])):
        low = np.minimum(a, b)-radius-res
        high = np.maximum(a, b)+radius+res
        r_top, c_left = world_to_cell(float(low[0]), float(high[1]), grid)
        r_bottom, c_right = world_to_cell(float(high[0]), float(low[1]), grid)
        r0, r1 = max(0, r_top), min(grid["height"], r_bottom+1)
        c0, c1 = max(0, c_left), min(grid["width"], c_right+1)
        locations = np.argwhere(~grid["free"][r0:r1, c0:c1])
        if not len(locations):
            continue
        rows, cols = locations[:, 0]+r0, locations[:, 1]+c0
        obstacles = np.column_stack((xmin+(cols+0.5)*res,
                                     ymin+(grid["height"]-1-rows+0.5)*res))
        v = b-a
        vv = float(v @ v)
        t = np.clip((obstacles-a) @ v / vv, 0.0, 1.0) if vv else np.zeros(len(obstacles))
        nearest = a+t[:, None]*v
        distance = float(np.linalg.norm(obstacles-nearest, axis=1).min())
        minimum = min(minimum, distance)
        if distance < radius-1e-9:
            raise ValueError(f"第{index}段线段与障碍格中心距离{distance:.6f}米，不足半径")
    return dict(passed=True, point_count=len(points), length_m=float(steps.sum()),
                max_step_m=float(steps.max()), endpoint_error_m=endpoint_error,
                minimum_checked_segment_clearance_m=minimum if math.isfinite(minimum) else None,
                segment_check="解析点到线段距离；障碍采用格子中心模型",
                boundary_clearance_m=float(boundary.min()))
