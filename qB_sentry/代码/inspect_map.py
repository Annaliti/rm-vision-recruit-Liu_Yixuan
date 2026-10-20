#!/usr/bin/env python3
"""B 题第一步：地图读取 → 坐标转换 → 机器人安全区域 → 查询预检。

本步骤输出地图数组和查询预检结果；A* / Dijkstra 路径规划见 run_planning.py。
阅读顺序：load_map → world_to_cell → build_safety → inspect_queries → main。
所有距离单位都是米；数组索引使用 [row, col]，世界坐标使用 (x, y)。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def load_map(yaml_path: Path) -> dict:
    """第 1 步：图片给出灰度，YAML 给出这些像素的物理含义。"""
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    # 提供的检查器忽略 yaw 和 negate；本教学版本明确限制为题目原配置。
    if cfg.get("mode", "trinary") != "trinary":
        raise ValueError("本步骤只处理 trinary 地图。")
    if cfg.get("negate", 0) != 0 or abs(float(cfg["origin"][2])) > 1e-12:
        raise ValueError("本步骤要求 negate=0、origin 的 yaw=0，不能直接用于旋转后的坐标系。")
    resolution = float(cfg["resolution"])
    low, high = float(cfg["free_thresh"]), float(cfg["occupied_thresh"])
    if not (resolution > 0 and 0 <= low < high <= 1):
        raise ValueError("分辨率或占据阈值不合法。")
    image_path = yaml_path.parent / cfg["image"]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"无法读取地图：{image_path}")
    height, width = gray.shape  # 注意：shape 先高度，后宽度。

    # 先转浮点数再计算：白色 255 → 0，黑色 0 → 1。
    # 这是该地图编码定义的占据概率/占据值，不是从传感器重新估计概率。
    occupancy = (255.0 - gray.astype(np.float64)) / 255.0
    free = occupancy < low
    occupied = occupancy > high
    unknown = ~(free | occupied)  # 等于阈值也归未知。
    classes = np.where(free, 0, np.where(occupied, 2, 1)).astype(np.uint8)
    ox, oy = map(float, cfg["origin"][:2])
    return dict(cfg=cfg, gray=gray, occupancy=occupancy, free=free,
                occupied=occupied, unknown=unknown, classes=classes,
                height=height, width=width, resolution=resolution,
                ox=ox, oy=oy, extent=[ox, ox+width*resolution, oy, oy+height*resolution])


def world_to_cell(x: float, y: float, grid: dict) -> tuple[int, int]:
    """第 2 步：世界坐标（米）→ 图片行列号；返回 (row, col)。"""
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("坐标必须是有限数。")
    # floor 表示找“包含这个点的格子”，负数时不能用 int() 代替。
    col = math.floor((x-grid["ox"])/grid["resolution"])
    from_bottom = math.floor((y-grid["oy"])/grid["resolution"])
    # 世界 y 向上增加，图片 row 向下增加，所以必须翻转。
    row = grid["height"] - 1 - from_bottom
    return row, col


def cell_to_world(row: int, col: int, grid: dict) -> tuple[float, float]:
    """图片行列号 → 格子中心（米）；它不一定等于输入的原始坐标。"""
    s = grid["resolution"]
    x = grid["ox"] + (col+0.5)*s
    y = grid["oy"] + (grid["height"]-1-row+0.5)*s
    return x, y


def build_safety(grid: dict, radius: float) -> dict:
    """第 3 步：计算机器人中心可以进入哪些格子。"""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("机器人半径必须为正数。")
    free = grid["free"]
    # OpenCV 距离变换：每个非零像素到最近零像素的欧氏距离（单位：像素）。
    # 非空闲区（占据+未知）都设为 0，与提供的 check_path.py 一致。
    if np.any(~free):
        dist_px = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2,
                                       cv2.DIST_MASK_PRECISE)
        # 网格中心间的平方距离应是整数。四舍五入消除 float32 在 6 格边界的误差。
        clearance = np.sqrt(np.rint(dist_px.astype(np.float64)**2))*grid["resolution"]
    else:
        clearance = np.full(free.shape, np.inf)
    checker_safe = free & (clearance >= radius)

    # 检查器只检查路径点是否出界；这里额外检查格子中心到地图边界的距离，
    # 避免机器人中心在图内、身体却伸到地图外。原地图边缘本身也有障碍。
    rows, cols = np.indices(free.shape)
    edge_px = np.minimum.reduce([cols+0.5, grid["width"]-cols-0.5,
                                rows+0.5, grid["height"]-rows-0.5])
    edge_clearance = edge_px*grid["resolution"]
    safe = checker_safe & (edge_clearance >= radius)

    # 连通区仅做前置检查，不计算路线，更没有代替要求自己写的搜索算法。
    count, labels = cv2.connectedComponents(safe.astype(np.uint8), connectivity=4)
    return dict(clearance=clearance, edge_clearance=edge_clearance,
                checker_safe=checker_safe, safe=safe, labels=labels,
                component_count=int(count-1), radius=radius)


def inspect_endpoint(xy: list, grid: dict, safety: dict) -> dict:
    x, y = map(float, xy)
    row, col = world_to_cell(x, y, grid)
    inside = 0 <= row < grid["height"] and 0 <= col < grid["width"]
    result = dict(world=[x, y], row=row, col=col, inside_map=inside)
    if not inside:
        return result | dict(safe=False, reason="坐标在地图外")
    names = ["空闲", "未知", "占据"]
    distance = float(safety["clearance"][row, col])
    edge = float(safety["edge_clearance"][row, col])
    result.update(cell_center=list(cell_to_world(row, col, grid)),
                  gray=int(grid["gray"][row, col]),
                  occupancy=float(grid["occupancy"][row, col]),
                  cell_class=names[int(grid["classes"][row, col])],
                  obstacle_clearance_m=distance if math.isfinite(distance) else None,
                  boundary_clearance_m=edge,
                  component4=int(safety["labels"][row, col]),
                  safe=bool(safety["safe"][row, col]))
    if not grid["free"][row, col]:
        result["reason"] = "落在占据或未知格子"
    elif distance < safety["radius"]:
        result["reason"] = f"障碍距离 {distance:.4f} m 小于半径 {safety['radius']:.2f} m"
    elif edge < safety["radius"]:
        result["reason"] = "距离地图边界不足一个机器人半径"
    else:
        result["reason"] = "格子中心符合离散安全条件"
    return result


def inspect_queries(queries: list, grid: dict, safety: dict) -> list:
    """第 4 步：逐一检查起终点；这里只作离散网格预检。"""
    results = []
    for number, query in enumerate(queries, 1):
        start = inspect_endpoint(query["start"], grid, safety)
        goal = inspect_endpoint(query["goal"], grid, safety)
        same = bool(start["safe"] and goal["safe"] and
                    start.get("component4") == goal.get("component4"))
        if not start["safe"]:
            status = "无法到达：起点" + start["reason"]
        elif not goal["safe"]:
            status = "无法到达：终点" + goal["reason"]
        elif not same:
            status = "四邻域安全网格不连通，需结合最终邻接规则分析"
        else:
            status = "预检通过：同一安全连通区；尚未生成路径"
        results.append(dict(query=number, start=start, goal=goal,
                            same_component4=same, status=status))
    return results


def render_figures(grid: dict, safety: dict, results: list, out: Path) -> None:
    """结果图只用来帮助理解，规划应读取布尔数组而非重新识别截图颜色。"""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"rmvision-step1-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle, Patch
    font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    colors = ["#007a82", "#4066b2", "#a46b15", "#8a4d9e", "#ad4b37", "#c3373c", "#813a39"]
    rgb = np.ones((*grid["free"].shape, 3))
    rgb[~grid["free"]] = [.18, .23, .28]
    rgb[grid["free"] & ~safety["safe"]] = [1., .77, .44]
    xmin, xmax, ymin, ymax = grid["extent"]

    def base(ax, data, raw=False):
        ax.imshow(data, extent=grid["extent"], origin="upper", interpolation="nearest",
                  **dict(cmap="gray", vmin=0, vmax=255) if raw else {})
        ax.set(xlabel="世界 x / 米", ylabel="世界 y / 米")
        ax.set_aspect("equal")

    def markers(ax, show_outside):
        starts = {}
        for item in results:
            n = item["query"]
            starts.setdefault(tuple(item["start"]["world"]), []).append(n)
            if show_outside or item["goal"]["inside_map"]:
                xy = item["goal"]["world"]
                ax.scatter(*xy, marker="X", s=80, color=colors[(n-1)%len(colors)], zorder=5)
                ax.annotate(f"G{n}", xy, xytext=(5, 8), textcoords="offset points", weight="bold")
        for xy, ids in starts.items():
            ax.scatter(*xy, s=70, facecolors="white", edgecolors="#2364a4", linewidths=2, zorder=6)
            ax.annotate("S"+",".join(map(str, ids)), xy, xytext=(6, -16),
                        textcoords="offset points", color="#164b80")

    fig, ax = plt.subplots(figsize=(12, 7.1))
    base(ax, grid["gray"], raw=True)
    markers(ax, True)
    ax.set(xlim=(xmin-1, max(xmax, max(q["goal"]["world"][0] for q in results))+2),
           ylim=(ymin-.8, ymax+.8), title="01 原始地图与 7 组任务点")
    ax.text(29.2, 1.5, "G7 在地图外", color="#a52b2b", ha="center")
    fig.text(.1, .025, "圆圈 S = 起点；叉号 G = 终点。该图未绘制任何规划路径。")
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(out/"01_原始地图与任务点.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    base(ax, rgb)
    markers(ax, False)
    ax.set(xlim=(xmin-.5, xmax+.5), ylim=(ymin-.5, ymax+.5),
           title=f"02 机器人中心安全区域：半径 {safety['radius']:.2f} 米")
    ax.legend(handles=[Patch(facecolor="#2e3b47", label="占据或未知"),
                       Patch(facecolor="#ffc471", label="空闲，但机器人中心不可进入"),
                       Patch(facecolor="white", edgecolor="#777777", label="中心安全区域")],
              loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(out/"02_机器人中心安全区域.png", dpi=160)
    plt.close(fig)

    # 本题的第 6 组专门说明：目标像素空闲，不代表机器人放得下。
    if len(results) >= 6 and results[5]["goal"]["inside_map"]:
        goal = results[5]["goal"]
        x, y = goal["world"]
        cx, cy = goal["cell_center"]
        fig, ax = plt.subplots(figsize=(7.4, 7.4))
        base(ax, rgb)
        ax.add_patch(Circle((x, y), safety["radius"], fill=False, color="#c3373c", linewidth=2))
        ax.scatter(x, y, marker="X", s=90, color="#c3373c", label="原始终点 G6")
        ax.scatter(cx, cy, marker="+", s=80, color="#202a35", label="所在格子中心")
        ax.set(xlim=(x-.85, x+.85), ylim=(y-.85, y+.85), title="03 第 6 组：空闲像素 ≠ 可容纳机器人")
        ax.legend(loc="lower right")
        fig.text(.5, .025, f"按检查器的格子中心模型：障碍距离 {goal['obstacle_clearance_m']:.2f} m < 半径 {safety['radius']:.2f} m",
                 ha="center", color="#a52b2b")
        fig.tight_layout(rect=[0, .06, 1, 1])
        fig.savefig(out/"03_第6组终点局部解释.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=ROOT/"input/map/rmuc_2025.yaml")
    parser.add_argument("--queries", type=Path, default=ROOT/"input/map/queries.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    grid = load_map(args.map)
    query_config = json.loads(args.queries.read_text(encoding="utf-8"))
    radius = float(query_config["robot_radius_m"])
    safety = build_safety(grid, radius)
    results = inspect_queries(query_config["queries"], grid, safety)
    summary = dict(width=grid["width"], height=grid["height"], resolution_m=grid["resolution"],
                   origin=grid["cfg"]["origin"], extent_xy=grid["extent"],
                   robot_radius_m=radius, robot_radius_pixels=round(radius/grid["resolution"], 10),
                   free_cells=int(grid["free"].sum()), occupied_cells=int(grid["occupied"].sum()),
                   unknown_cells=int(grid["unknown"].sum()), safe_center_cells=int(safety["safe"].sum()),
                   inflated_free_cells=int((grid["free"] & ~safety["safe"]).sum()),
                   boundary_extra_blocked_cells=int((safety["checker_safe"] & ~safety["safe"]).sum()),
                   safe_components4=safety["component_count"], path_planning_done=False)
    (out/"map_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    (out/"query_checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    # 下一步 A* / Dijkstra 可直接加载 safe_centers；无需重新处理图片。
    np.savez_compressed(out/"map_arrays.npz", gray=grid["gray"], classes=grid["classes"],
                        free=grid["free"], safe_centers=safety["safe"],
                        clearance_m=safety["clearance"], component4=safety["labels"])
    versions = {"python": sys.version.split()[0], "executable": sys.executable}
    for package in ["numpy", "opencv-python", "matplotlib", "pyyaml", "open3d"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "未安装（本步骤不需要 Open3D）"
    (out/"environment.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2)+"\n")
    render_figures(grid, safety, results, out)
    print(f"地图 {grid['width']} × {grid['height']}，每格 {grid['resolution']} m")
    print(f"空闲格 {summary['free_cells']}；考虑机器人半径后安全格 {summary['safe_center_cells']}")
    for item in results:
        print(f"第 {item['query']} 组：{item['status']}")
    print("已完成地图预检；尚未运行 A* / Dijkstra，也未生成提交用路径。")


if __name__ == "__main__":
    main()
