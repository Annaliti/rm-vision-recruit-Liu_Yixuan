"""绘制实际点云处理、异常诊断、坡道台阶剖面及自建地图规划结果。"""
import json
import numpy as np
from plot_planning import configure_fonts
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from inspect_map import ROOT, load_map


def main():
    configure_fonts()
    out = ROOT/"results_cloud"
    prep = json.loads((out/"preprocessing.json").read_text())
    anomaly = json.loads((out/"anomaly_analysis.json").read_text())
    data = np.load(out/"prepared_cloud.npz")
    p = data["points"]
    rotation = np.array(prep["rotation_lidar_to_boundary"])
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 5.8), layout="constrained")
    raw = data["raw_sample"]
    first = data["first_plane"]
    ground = data["ground_plane"]
    ax = fig.add_subplot(131, projection="3d")
    q = raw[::4]
    ax.scatter(*q.T, c=q[:, 2], s=.15, cmap="viridis", rasterized=True)
    ax.set_title("① 原始雷达坐标：场地倾斜")
    ax = fig.add_subplot(132, projection="3d")
    ax.scatter(*first[::10].T, s=.2, color="#b94f55", label="第一次RANSAC：墙面")
    ax.scatter(*ground[::3].T, s=.25, color="#188e91", label="地面候选")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("② 最大平面不是地面")
    ax = fig.add_subplot(133, projection="3d")
    bounds = prep["boundary_extent"]
    q = p[(p[:, 2]>-.1)&(p[:, 2]<1.2)&(p[:, 0]>bounds[0]+.15)&(p[:, 0]<bounds[1]-.15)
          &(p[:, 1]>bounds[2]+.15)&(p[:, 1]<bounds[3]-.15)][::3]
    ax.scatter(*q.T, c=q[:, 2], s=.25, cmap="turbo", vmin=0, vmax=.6)
    ax.set_title("③ 调平并沿边界对正（显示低处结构）")
    ax.set_box_aspect((29, 16, 3))
    ax.set_zticks([0, .6, 1.2])
    for ax in fig.axes:
        ax.set_xlabel("x / m"); ax.set_ylabel("y / m"); ax.set_zlabel("z / m")
        ax.view_init(elev=28, azim=-62)
    fig.suptitle(f"点云处理实测：{prep['raw_points']:,} → 体素{prep['downsampled_points']:,} → 滤波{prep['filtered_points']:,} 点", fontsize=19)
    fig.savefig(figures/"01_点云处理过程.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), layout="constrained")
    q = raw@rotation.T
    q[:, 2] += prep["height_offset_m"]
    removed = data["removed"]
    axes[0].scatter(q[:, 0], q[:, 2], s=.15, color="#adb8c1", label="原始抽样")
    axes[0].scatter(removed[:, 0], removed[:, 2], s=.7, color="#bd4c52", label="统计滤波剔除点")
    axes[0].axhline(-.1, color="#164c76", ls="--", label="地下异常阈值 -0.10 m")
    axes[0].set(xlabel="边界坐标 x / m", ylabel="离地高度 / m", title="孤立、地下及场外异常点")
    axes[0].legend(fontsize=10)
    axes[1].hist(p[:, 2], bins=250, color="#347f9b")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="调平后离地高度 / m", ylabel="点数（对数）", title="约18.5m高矩形围挡：几何存在，语义不明")
    axes[1].text(.04, .95, f"统计滤波剔除：{anomaly['sor_removed']:,} 点\n后续地下/场外剔除：{anomaly['union_removed_after_sor']:,} 点\n\n保留围挡近地碰撞截面。\n不能凭高度断言真实物体不存在。", va="top", transform=axes[1].transAxes, fontsize=12,
                 bbox=dict(facecolor="white", alpha=.9, edgecolor="#d5dce1", pad=8))
    fig.suptitle("异常分析：可验证的几何证据与成因推测分开记录", fontsize=19)
    fig.savefig(figures/"02_异常点与高围挡诊断.png", dpi=150)
    plt.close(fig)

    native = np.load(out/"terrain_native.npz")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    specs = [(1.0, (0, 7), "同一高地：左侧台阶，右侧约10°坡道"),
             (10.6, (16, 24), "另一侧高地：连续坡道与垂直边缘"),
             (6.0, (6, 17), "中央高地：0.00/0.20/0.30m不同支撑面"),
             (.8, (7, 16), "高地边缘：需要按局部高差和车身净空判断")]
    profile_records = []
    for ax, (y, xrange, title) in zip(axes.flat, specs):
        sample = p[(abs(p[:, 1]-y)<.08)&(p[:, 0]>xrange[0])&(p[:, 0]<xrange[1])&(p[:, 2]>-.08)&(p[:, 2]<.8)]
        ax.scatter(sample[:, 0], sample[:, 2], s=3, color="#177fa6", alpha=.6)
        ax.set(xlim=xrange, ylim=(-.05, .7), xlabel=f"边界坐标 x / m；截面 y={y:.1f}±0.08m", ylabel="离地高度 / m", title=title)
        ax.axhline(.12, color="#a6464f", ls="--", lw=.8)
        ax.grid(alpha=.2)
        profile_records.append(dict(y_native_m=y, x_range=xrange, points=len(sample)))
    fig.suptitle("真实点云剖面：绝对高度高于12cm并不意味着不可通行", fontsize=19)
    fig.savefig(figures/"03_坡道与台阶实际剖面.png", dpi=150)
    plt.close(fig)
    (out/"profiles.json").write_text(json.dumps(profile_records, ensure_ascii=False, indent=2)+"\n")

    terrain = np.load(out/"terrain.npz")
    ref = load_map(ROOT/"input/map/rmuc_2025.yaml")
    cmap = ListedColormap(["#f8faf9", "#aebbc4", "#26323d"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    z = np.where(terrain["support"], terrain["elevation"], np.nan)
    artist = axes[0, 0].imshow(z, extent=terrain["extent"], vmin=0, vmax=.5, cmap="terrain")
    axes[0, 0].set_title("局部支撑面高度（灰白空洞为未知）")
    fig.colorbar(artist, ax=axes[0, 0], label="m", shrink=.8)
    slope = np.where(terrain["support"], terrain["slope_deg"], np.nan)
    artist = axes[0, 1].imshow(slope, extent=terrain["extent"], vmin=0, vmax=30, cmap="magma")
    axes[0, 1].set_title("局部坡度；最大允许25°为教学设定")
    fig.colorbar(artist, ax=axes[0, 1], label="度", shrink=.8)
    axes[1, 0].imshow(terrain["classes"], extent=terrain["extent"], cmap=cmap, vmin=0, vmax=2)
    axes[1, 0].set_title("自建图：白=可通行，灰=未知，深色=障碍")
    axes[1, 1].imshow(ref["classes"], extent=ref["extent"], cmap=cmap, vmin=0, vmax=2)
    axes[1, 1].set_title("提供图：仅用于共享坐标校准与结果比较")
    for ax in axes.flat:
        ax.set_xlabel("查询世界坐标 x / m"); ax.set_ylabel("y / m")
    fig.suptitle("0.05m 自建地图与提供地图：格子标签独立由点云生成", fontsize=19)
    fig.savefig(figures/"04_自建地形与地图对比.png", dpi=150)
    plt.close(fig)

    results = json.loads((ROOT/"results_cloud_planning/summary.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
    fig.get_layout_engine().set(hspace=.04)
    for ax, row in zip(axes.flat, results["queries"][:6]):
        ax.imshow(terrain["classes"], extent=terrain["extent"], cmap=cmap, vmin=0, vmax=2)
        a = row["algorithms"].get("astar")
        if a:
            route = np.loadtxt(ROOT/a["path_file"], delimiter=",", skiprows=1)
            ax.plot(route[:, 0], route[:, 1], color="#078ca1", lw=1.7)
        ax.scatter(*row["start"]["world"], s=24, color="#066da1")
        ax.scatter(*row["goal"]["world"], s=50, color="#b53658", marker="*")
        note = f"{a['length_m']:.3f} m / 两算法均通过" if a else "无法到达（详见逐组原因）"
        ax.set(title=f"第 {row['query']} 组：{note}", xlabel="x / m", ylabel="y / m")
    fig.suptitle("自建地图上的原始查询；第7组终点在图外", fontsize=19)
    fig.savefig(figures/"05_自建地图规划结果.png", dpi=150)
    plt.close(fig)
    print("点云处理、异常、地形剖面及规划图已生成")


if __name__ == "__main__":
    main()
