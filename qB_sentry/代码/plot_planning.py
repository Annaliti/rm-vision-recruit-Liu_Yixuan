"""用已保存的真实搜索记录画图；绘图不混入算法耗时。

python plot_planning.py           # 静态图 + 搜索过程GIF
python plot_planning.py --no-gif  # 只画静态图
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"rmvision-step2-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, font_manager
from matplotlib.patches import Patch
import numpy as np

from inspect_map import ROOT, load_map, build_safety, cell_to_world

BLUE = "#087eab"
ORANGE = "#ba6822"
RED = "#c03954"


def configure_fonts():
    font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "font.weight": 300, "axes.titleweight": 600,
                         "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff"})


def make_base(grid, safety):
    rgb = np.full((*grid["free"].shape, 3), 0.985)
    rgb[~grid["free"]] = [0.15, 0.19, 0.23]
    rgb[grid["free"] & ~safety["safe"]] = [0.91, 0.88, 0.81]
    return rgb


def style_map(ax, grid, rgb):
    artist = ax.imshow(rgb, extent=grid["extent"], origin="upper", interpolation="nearest")
    ax.set_xlabel("世界坐标 x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    return artist


def draw_endpoints(ax, row):
    ax.scatter(*row["start"], s=38, color=BLUE, edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter(*row["goal"], s=80, marker="*", color=RED, edgecolors="white", linewidths=0.5, zorder=5)


def main(make_gif=True):
    configure_fonts()
    out = ROOT/"results_step2"
    report = json.loads((out/"summary.json").read_text())
    grid = load_map(ROOT/"input/map/rmuc_2025.yaml")
    tasks = json.loads((ROOT/"input/map/queries.json").read_text())
    safety = build_safety(grid, tasks["robot_radius_m"])
    base = make_base(grid, safety)
    feasible = [r for r in report["queries"] if r["algorithms"]]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, hspace=0.05)
    for ax, row in zip(axes.flat, feasible):
        style_map(ax, grid, base)
        path = np.loadtxt(ROOT/row["algorithms"]["astar"]["path_file"], delimiter=",", skiprows=1)
        ax.plot(path[:, 0], path[:, 1], color=BLUE, linewidth=1.8)
        draw_endpoints(ax, row)
        ax.set_title(f"第 {row['query']} 组  ·  {row['algorithms']['astar']['csv_length_m']:.3f} m")
    last = axes.flat[-1]
    last.axis("off")
    last.text(0.04, 0.94, "第 6、7 组：无法到达", fontsize=19, weight=600, va="top", transform=last.transAxes)
    last.text(0.04, 0.77, "第 6 组：终点格离障碍 0.25 m < 半径 0.30 m。\n目标点空闲，但放不下整个机器人。",
              fontsize=12, linespacing=1.5, va="top", transform=last.transAxes)
    last.text(0.04, 0.52, "第 7 组：终点 x = 30 m > 地图右边界 25.57 m。\n终点在地图外，无法给出图内路线。",
              fontsize=12, linespacing=1.5, va="top", transform=last.transAxes)
    last.legend(handles=[Patch(color=BLUE, label="蓝色圆点 / 实线：起点 / A*路径"),
                         Patch(color=RED, label="红色星标：终点"),
                         Patch(color="#262f3b", label="深色：障碍或未知"),
                         Patch(color="#e8e0cf", label="米色：空闲但距障碍不足一个半径")],
                loc="lower left", frameon=False, fontsize=10.5)
    fig.suptitle("提供地图上的 7 组查询  |  前 5 组 A* 路径", fontsize=21, weight=600)
    fig.savefig(out/"04_五组规划路径.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), layout="constrained")
    x = np.arange(len(feasible))
    for ax, metric, title, unit in [
        (axes[0], "expanded_nodes", "扩展了多少个格子？", "扩展节点数"),
        (axes[1], "elapsed_ms_median", "搜索花了多少时间？", "搜索时间 / ms"),
    ]:
        for offset, name, label, color in [(-0.19, "dijkstra", "Dijkstra", ORANGE), (0.19, "astar", "A*", BLUE)]:
            values = [r["algorithms"][name][metric] for r in feasible]
            bars = ax.bar(x+offset, values, width=0.36, color=color, label=label)
            labels = [f"{v/1000:.1f}k" if metric == "expanded_nodes" else f"{v:.0f}" for v in values]
            ax.bar_label(bars, labels=labels, fontsize=10, padding=3)
        ax.set_xticks(x, [f"第 {r['query']} 组" for r in feasible])
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=16)
        ax.set_ylim(0, ax.get_ylim()[1]*1.12)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(f"相同地图、相同八邻域规则  |  搜索耗时取 {report['repeats']} 次运行中位数", fontsize=18)
    fig.savefig(out/"05_算法性能对比.png", dpi=150)
    plt.close(fig)

    # 直接排版实际stdout，明确它是日志图，不冒充桌面终端截图。
    fig, axes = plt.subplots(len(feasible), 2, figsize=(13, 10), layout="constrained")
    for row_axes, row in zip(axes, feasible):
        for ax, name, label in zip(row_axes, ("astar", "dijkstra"), ("A*", "Dijkstra")):
            ax.axis("off")
            log = (ROOT/row["algorithms"][name]["checker_log"]).read_text().strip()
            ax.text(0.03, 0.86, f"第 {row['query']} 组  ·  {label}", fontsize=15, color=BLUE, weight=600, transform=ax.transAxes)
            ax.text(0.03, 0.65, log, fontsize=13, linespacing=1.6, va="top", transform=ax.transAxes)
            ax.axhline(0.08, color="#d7e2e7", linewidth=0.8)
    fig.suptitle("原检查器实际输出汇总\n直接读取 checker_logs 中的标准输出；此图为日志排版图", fontsize=18)
    fig.savefig(out/"08_原检查器输出汇总.png", dpi=150)
    plt.close(fig)

    query = feasible[0]
    traces, paths, ranks = {}, {}, {}
    for name in ("dijkstra", "astar"):
        saved = np.load(out/f"search_traces/{name}_q1.npz")
        traces[name] = saved["expanded_cells"]
        paths[name] = np.array([cell_to_world(int(r), int(c), grid) for r, c in saved["path_cells"]])
        ranks[name] = np.full(grid["free"].shape, np.iinfo(np.int32).max, dtype=np.int32)
        ranks[name][traces[name][:, 0], traces[name][:, 1]] = np.arange(len(traces[name]))

    def progress_image(name, budget):
        rgb = base.copy()
        rgb[ranks[name] < budget] = [0.49, 0.78, 0.89] if name == "astar" else [0.94, 0.70, 0.44]
        return rgb

    max_count = max(map(len, traces.values()))
    budgets = [5000, 20000, max_count]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.2), layout="constrained")
    for i, (name, label) in enumerate((("dijkstra", "Dijkstra"), ("astar", "A*"))):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            style_map(ax, grid, progress_image(name, budget))
            finished = budget >= len(traces[name])
            if finished:
                ax.plot(paths[name][:, 0], paths[name][:, 1], color="#6b214e", linewidth=1.6)
            draw_endpoints(ax, query)
            ax.set_title(f"{label}  ·  扩展 {min(budget, len(traces[name])):,} 个"+("（已找到终点）" if finished else ""))
    fig.suptitle("第 1 组的真实搜索过程  |  着色部分是已经扩展过的格子", fontsize=21, weight=600)
    fig.savefig(out/"06_搜索过程对比.png", dpi=150)
    plt.close(fig)

    if make_gif:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        images, lines, titles = {}, {}, {}
        for ax, name in zip(axes, ("dijkstra", "astar")):
            images[name] = style_map(ax, grid, base)
            lines[name], = ax.plot([], [], color="#6b214e", linewidth=1.8)
            draw_endpoints(ax, query)
            titles[name] = ax.set_title(name)
        fig.suptitle("第 1 组  |  按扩展节点数推进的搜索回放", fontsize=18, weight=600)
        fig.text(0.5, 0.015, "每帧使用相同的节点数上限；不是实际运行速度。到达终点后停止搜索。",
                 ha="center", fontsize=10)
        frames = np.concatenate([np.linspace(0, max_count, 48, dtype=int), np.repeat(max_count, 9)])

        def update(budget):
            for name, label in (("dijkstra", "Dijkstra"), ("astar", "A*")):
                images[name].set_data(progress_image(name, int(budget)))
                finished = budget >= len(traces[name])
                titles[name].set_text(f"{label}：{min(int(budget), len(traces[name])):,} 个节点"+("  已到达" if finished else ""))
                if finished:
                    lines[name].set_data(paths[name][:, 0], paths[name][:, 1])
                else:
                    lines[name].set_data([], [])
            return list(images.values())+list(lines.values())+list(titles.values())

        replay = animation.FuncAnimation(fig, update, frames=frames, interval=140, blit=False)
        replay.save(out/"07_搜索过程回放.gif", writer=animation.PillowWriter(fps=7), dpi=95)
        plt.close(fig)
    print("第二步静态图与搜索回放已生成。" if make_gif else "第二步静态图已生成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gif", action="store_true")
    main(make_gif=not parser.parse_args().no_gif)
