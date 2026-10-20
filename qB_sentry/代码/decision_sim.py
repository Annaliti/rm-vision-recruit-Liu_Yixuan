#!/usr/bin/env python3
"""B 题发挥：可解释的有限状态机 + 基于现有 A* 的二维教学仿真。

所有战术阈值、补给速度和攻击范围都是教学假设，不代表 RMUC 官方规则。
环境几何使用题目原 PGM；机器人每个时间步沿已验证网格路径移动一格。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import numpy as np

from inspect_map import build_safety, cell_to_world, load_map, world_to_cell
from planner import search_grid, validate_world_path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_decision"
STATES = ("PATROL", "ATTACK", "RETREAT", "RESUPPLY", "DEFEND", "ENDGAME", "STOP")
NAMES = dict(PATROL="巡逻", ATTACK="接敌", RETREAT="撤退恢复", RESUPPLY="补弹",
             DEFEND="保守防守", ENDGAME="末段集结", STOP="结束")
COLORS = dict(zip(STATES, ("#168585", "#d55443", "#885da8", "#cd9831", "#456eb3", "#bc5585", "#555555")))


@dataclass(frozen=True)
class Parameters:
    dt_s: float = 0.1
    hp_max: float = 200.0
    ammo_max: int = 100
    low_hp: float = 60.0
    recovered_hp: float = 140.0
    low_ammo: int = 8
    recovered_ammo: int = 60
    enemy_enter_m: float = 4.0
    enemy_leave_m: float = 5.0
    fire_range_m: float = 3.5
    fire_interval_s: float = 0.5
    endgame_s: float = 30.0
    heal_per_s: float = 25.0
    reload_per_s: float = 20.0
    incoming_damage_per_s: float = 2.0
    radius_m: float = 0.30


def choose_state(previous, hp, ammo, enemy_distance, remaining_s, outpost_alive, p):
    """每周期从上往下匹配；前面规则优先。返回状态和可审计原因。

    血量 60/140、弹量 8/60、接敌距离 4/5 使用不同进入/退出阈值。
    例如撤退后 HP 升到 61 不会立刻出发，必须恢复到 140。
    """
    if previous == "STOP" or remaining_s <= 0:
        return "STOP", "比赛时间耗尽；结束状态保持"
    if hp <= p.low_hp or (previous == "RETREAT" and hp < p.recovered_hp):
        return "RETREAT", "低血量或撤退恢复尚未达到退出阈值"
    if ammo <= p.low_ammo or (previous == "RESUPPLY" and ammo < p.recovered_ammo):
        return "RESUPPLY", "低弹量或补弹尚未达到退出阈值"
    if not outpost_alive:
        return "DEFEND", "己方前哨站被毁，转入保守防守"
    if remaining_s <= p.endgame_s:
        return "ENDGAME", "进入教学设定的最后 30 秒，前往集结点"
    limit = p.enemy_leave_m if previous == "ATTACK" else p.enemy_enter_m
    if enemy_distance is not None and enemy_distance <= limit:
        return "ATTACK", "敌方位置进入接敌阈值；接敌状态采用较大退出阈值"
    return "PATROL", "没有更高优先级条件，继续巡逻"


def line_of_sight(a, b, grid):
    """射线采样只用于教学攻击判定；路径安全由独立线段验证负责。"""
    count = max(2, math.ceil(math.dist(a, b) / (grid["resolution"] / 2)) + 1)
    for x, y in np.linspace(a, b, count):
        r, c = world_to_cell(float(x), float(y), grid)
        if not (0 <= r < grid["height"] and 0 <= c < grid["width"] and grid["free"][r, c]):
            return False
    return True


class Simulation:
    def __init__(self, parameters=None):
        self.p = parameters or Parameters()
        self.grid = load_map(ROOT / "input/map/rmuc_2025.yaml")
        self.safety = build_safety(self.grid, self.p.radius_m)
        self.safe_cells = np.argwhere(self.safety["safe"])
        self.safe_xy = np.array([cell_to_world(*rc, self.grid) for rc in self.safe_cells])
        # 这些是演示用策略点，并非已识别的真实基地/补给区/前哨站位置。
        requested = dict(home=(-2.26, -1.36), patrol=(2.0, -2.0),
                         defend=(0.4, 2.0), endgame=(5.0, -4.0), enemy=(0.3, -0.7))
        self.anchors = {name: self.nearest_cell(point) for name, point in requested.items()}
        self.route_cache = {}

    def nearest_cell(self, xy):
        index = int(np.argmin(np.sum((self.safe_xy - xy)**2, axis=1)))
        return tuple(map(int, self.safe_cells[index]))

    def target_for(self, state, current, enemy_xy, patrol_index):
        if state in ("RETREAT", "RESUPPLY"):
            return self.anchors["home"]
        if state == "DEFEND":
            return self.anchors["defend"]
        if state == "ENDGAME":
            return self.anchors["endgame"]
        if state == "ATTACK":
            # 选择离敌约 1.5 m 的安全格作为接近位置；敌方位置不直接当可走目标。
            current_xy = np.array(cell_to_world(*current, self.grid))
            enemy_distance = np.linalg.norm(self.safe_xy - enemy_xy, axis=1)
            score = abs(enemy_distance - 1.5) + 0.08 * np.linalg.norm(self.safe_xy-current_xy, axis=1)
            return tuple(map(int, self.safe_cells[int(np.argmin(score))]))
        return self.anchors[("patrol", "home")[patrol_index % 2]]

    def route(self, start, goal):
        key = (start, goal)
        if key not in self.route_cache:
            result = search_grid(self.safety["safe"], start, goal, self.grid["resolution"],
                                 algorithm="astar", record_trace=False)
            if not result.found:
                raise RuntimeError(f"教学策略目标不可到达：{start} -> {goal}")
            points = [cell_to_world(*cell, self.grid) for cell in result.cells]
            # 停在原处时仍用两个相同点验证圆形足迹。
            checked_points = points if len(points) > 1 else points * 2
            validation = validate_world_path(checked_points, points[0], points[-1], self.grid, self.safety)
            self.route_cache[key] = (result.cells, validation)
        return self.route_cache[key]

    def run(self, scenario):
        p = self.p
        current = self.anchors[scenario.get("start", "patrol")]
        hp, ammo = float(scenario.get("hp", 200)), float(scenario.get("ammo", 100))
        previous, patrol_index = "PATROL", 0
        active_goal, route_cells, route_index = None, [], 0
        fire_cooldown = 0.0
        records, transitions, used_routes = [], [], []
        for tick in range(round(scenario["duration_s"] / p.dt_s) + 1):
            t = tick * p.dt_s
            remaining = max(0.0, scenario["remaining_s"] - t)
            outpost_alive = t < scenario.get("outpost_destroyed_at_s", math.inf)
            enemy_window = scenario.get("enemy_window_s")
            enemy_xy = None
            if enemy_window and enemy_window[0] <= t < enemy_window[1]:
                enemy_xy = cell_to_world(*self.anchors["enemy"], self.grid)
            before = cell_to_world(*current, self.grid)
            distance = math.dist(before, enemy_xy) if enemy_xy else None
            state, reason = choose_state(previous, hp, ammo, distance, remaining, outpost_alive, p)
            changed = state != previous
            if tick == 0 or changed:
                transitions.append(dict(time_s=round(t, 4), source=previous, target=state, reason=reason))
            if state == "PATROL" and current == self.anchors[("patrol", "home")[patrol_index % 2]]:
                patrol_index += 1
            goal = current if state == "STOP" else self.target_for(state, current, enemy_xy, patrol_index)
            # 接敌目标在一次接敌期内保持；避免每格移动都重新选择稍有不同的最优点。
            if state == "ATTACK" and not changed and active_goal is not None:
                goal = active_goal
            if goal != active_goal or changed:
                route_cells, validation = self.route(current, goal)
                route_index, active_goal = 0, goal
                used_routes.append(dict(time_s=round(t, 4), state=state,
                                        start=list(current), goal=list(goal), validation=validation))
            action = "保持位置"
            # 一步只走到下一个网格中心，不跨过拐角。最大速度 sqrt(2)*0.05/0.1。
            if state != "STOP" and route_index + 1 < len(route_cells):
                route_index += 1
                current = route_cells[route_index]
                action = "沿已验证 A* 路径移动一格"
            after = cell_to_world(*current, self.grid)
            hp_before, ammo_before = hp, ammo
            at_home = current == self.anchors["home"]
            if state == "RETREAT" and at_home:
                hp = min(p.hp_max, hp + p.heal_per_s * p.dt_s)
                action = "在教学恢复点恢复血量"
            if state == "RESUPPLY" and at_home:
                ammo = min(p.ammo_max, ammo + p.reload_per_s * p.dt_s)
                action = "在教学恢复点补弹"
            fire_cooldown = max(0.0, fire_cooldown - p.dt_s)
            can_engage = (state in ("ATTACK", "DEFEND") and enemy_xy is not None
                          and math.dist(after, enemy_xy) <= p.fire_range_m
                          and line_of_sight(after, enemy_xy, self.grid))
            fired = False
            if can_engage:
                hp = max(0.0, hp - p.incoming_damage_per_s*p.dt_s)
                if fire_cooldown <= 1e-9 and ammo >= 1:
                    ammo -= 1
                    fire_cooldown = p.fire_interval_s
                    fired = True
                    action += "；教学模拟射击一次"
            records.append(dict(t_s=round(t, 4), state_before=previous, state=state, reason=reason,
                                hp_input=hp_before, ammo_input=ammo_before, hp_after=hp, ammo_after=ammo,
                                enemy_xy=list(enemy_xy) if enemy_xy else None,
                                enemy_distance_m=distance, remaining_s=remaining,
                                outpost_alive=outpost_alive, xy_before=list(before), xy=list(after),
                                cell=list(current), goal_cell=list(goal), fired=fired, action=action))
            previous = state
            if state == "STOP":
                break
        return dict(scenario=scenario, transitions=transitions, records=records, routes=used_routes,
                    states_seen=sorted({record["state"] for record in records}))


def scenarios():
    return [
        dict(id="01_patrol_enemy", name="巡逻与敌方位置变化", start="home", duration_s=45,
             remaining_s=240, enemy_window_s=[8, 22]),
        dict(id="02_low_hp", name="低血量撤退及恢复滞回", duration_s=60, remaining_s=240, hp=45, ammo=70),
        dict(id="03_low_ammo", name="低弹量补给及退出阈值", duration_s=60, remaining_s=240, ammo=4),
        dict(id="04_outpost", name="前哨站被毁触发防守", duration_s=50, remaining_s=240,
             outpost_destroyed_at_s=8),
        dict(id="05_endgame", name="剩余时间触发末段集结及结束", duration_s=50, remaining_s=45),
        dict(id="06_priorities", name="血量弹量前哨站同时告急", duration_s=50, remaining_s=45,
             start="home", hp=45, ammo=4, outpost_destroyed_at_s=0),
    ]


def plot_results(sim, runs, out):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rmvision-decision-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch
    font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "font.weight": 300,
                         "axes.titleweight": 600})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, run in zip(axes.flat, runs):
        ax.imshow(sim.grid["gray"], extent=sim.grid["extent"], origin="upper", cmap="gray", vmin=0, vmax=255)
        xy = np.array([r["xy"] for r in run["records"]])
        states = np.array([r["state"] for r in run["records"]])
        ax.plot(xy[:, 0], xy[:, 1], color="#a0a0a0", linewidth=1)
        for state in STATES:
            selected = states == state
            if selected.any():
                ax.scatter(xy[selected, 0], xy[selected, 1], c=COLORS[state], s=5, label=NAMES[state])
        for name in ("home", "defend", "endgame"):
            x, y = cell_to_world(*sim.anchors[name], sim.grid)
            ax.scatter(x, y, marker="x", c="#111111", s=30)
        ax.set(xlim=(-3.5, 8), ylim=(-6, 4.5), xlabel="世界 x / m", ylabel="世界 y / m",
               title=run["scenario"]["name"], aspect="equal")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("六组确定性教学仿真：颜色为决策状态；运动使用原地图与自写 A*", fontsize=15)
    fig.savefig(out / "01_六场景运动轨迹.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(6, 1, figsize=(13, 10), sharex=False, constrained_layout=True)
    for ax, run in zip(axes, runs):
        records = run["records"]
        for state in STATES:
            times = np.array([r["t_s"] for r in records if r["state"] == state])
            if len(times):
                ax.scatter(times, np.zeros_like(times), color=COLORS[state], marker="s", s=26, label=NAMES[state])
        ax.set(title=run["scenario"]["name"], yticks=[], ylim=(-1, 1), xlabel="仿真时间 / s")
        ax.legend(loc="upper right", ncol=5, fontsize=8)
    fig.suptitle("状态时间线：时间步长 0.1 s；条件按固定优先级判断", fontsize=15)
    fig.savefig(out / "02_状态时间线.png", dpi=160)
    plt.close(fig)
    # 监督式 FSM 图：每个活动状态返回优先级判定器，STOP 无出边。
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set(xlim=(0, 13), ylim=(0, 8.5)); ax.axis("off")
    ax.text(6.5, 8.15, "有限状态机：每 0.1 s 从上向下匹配，首个成立条件生效", ha="center", fontsize=16)
    ax.add_patch(FancyBboxPatch((0.4, 2.5), 2.5, 3.0, boxstyle="round,pad=.15", facecolor="#e7eef5", edgecolor="#456eb3"))
    ax.text(1.65, 4.0, "读取输入\n血量 / 弹量\n敌方位置\n比赛剩余时间\n己方前哨站状态\n上一周期状态", ha="center", va="center", linespacing=1.8)
    guards = ["① 时间 ≤ 0 或已结束", "② HP ≤ 60；撤退中则保持到 HP ≥ 140",
              "③ 弹量 ≤ 8；补弹中则保持到弹量 ≥ 60", "④ 己方前哨站被毁", "⑤ 剩余时间 ≤ 30 s",
              "⑥ 敌距 ≤ 4 m；接敌中放宽到 ≤ 5 m", "⑦ 上述条件均不成立"]
    ordered = ["STOP", "RETREAT", "RESUPPLY", "DEFEND", "ENDGAME", "ATTACK", "PATROL"]
    for i, (state, guard) in enumerate(zip(ordered, guards)):
        y = 7.15-i*.87
        ax.add_patch(FancyBboxPatch((3.6, y-.22), 6.0, .48, boxstyle="round,pad=.08", facecolor="#f5f7f9", edgecolor="#aab3bd"))
        ax.text(3.75, y+.02, guard, fontsize=10, va="center")
        ax.annotate("", xy=(10.1, y+.02), xytext=(9.7, y+.02), arrowprops=dict(arrowstyle="->", color=COLORS[state]))
        ax.add_patch(FancyBboxPatch((10.2, y-.22), 2.1, .48, boxstyle="round,pad=.08", facecolor=COLORS[state], edgecolor="none"))
        ax.text(11.25, y+.02, NAMES[state], ha="center", va="center", color="white")
    ax.annotate("", xy=(3.5, 7.17), xytext=(2.9, 5.0), arrowprops=dict(arrowstyle="->", color="#456eb3"))
    for i in range(6):
        ax.annotate("否则", xy=(6.6, 6.57-i*.87), xytext=(6.6, 6.83-i*.87), ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#77818b"))
    # 活动状态明确连到右侧回路；STOP 不加入回路。
    for i in range(1, 7):
        y = 7.17-i*.87
        ax.plot([12.4, 12.7], [y, y], color="#456eb3", linewidth=1)
    ax.plot([12.7, 12.7, 1.65], [6.3, 1.05, 1.05], color="#456eb3", linewidth=1)
    ax.annotate("", xy=(1.65, 2.28), xytext=(1.65, 1.05),
                arrowprops=dict(arrowstyle="->", color="#456eb3"))
    ax.text(6.1, .67, "所有活动状态执行一次动作后返回读取输入；结束状态保持且不移动", ha="center", fontsize=11)
    ax.text(6.5, .18, "策略点、阈值、伤害及补给速度均为教学假设；并非官方比赛规则或实机验证", ha="center", fontsize=10, color="#666666")
    fig.savefig(out / "03_有限状态转移图.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / "03_有限状态转移图.svg", bbox_inches="tight")
    plt.close(fig)
    # GIF 来自已保存的仿真记录，两个场景共用同一仿真时间，不新增模拟行为。
    from PIL import Image
    from matplotlib.patches import Circle
    selected_runs = [runs[0], runs[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    artists = []
    for ax, run in zip(axes, selected_runs):
        ax.imshow(sim.grid["gray"], extent=sim.grid["extent"], origin="upper", cmap="gray", vmin=0, vmax=255)
        ax.set(xlim=(-3.5, 6), ylim=(-4.5, 4.5), aspect="equal", xlabel="世界 x / m", ylabel="世界 y / m",
               title=run["scenario"]["name"])
        line, = ax.plot([], [], color="#168585", linewidth=1.4)
        body = Circle((0, 0), sim.p.radius_m, color="#168585", alpha=.85)
        ax.add_patch(body)
        enemy, = ax.plot([], [], "x", color="#d55443", markersize=8, markeredgewidth=2)
        label = ax.text(.02, .98, "", transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=.9, edgecolor="#cccccc"))
        artists.append((line, body, enemy, label))
    frames = []
    for time_s in np.arange(0, 45.01, .5):
        for run, (line, body, enemy, label) in zip(selected_runs, artists):
            index = min(round(time_s/sim.p.dt_s), len(run["records"])-1)
            record = run["records"][index]
            trace = np.array([r["xy"] for r in run["records"][:index+1]])
            line.set_data(trace[:, 0], trace[:, 1])
            body.center = record["xy"]
            body.set_color(COLORS[record["state"]])
            enemy.set_data([record["enemy_xy"][0]], [record["enemy_xy"][1]]) if record["enemy_xy"] else enemy.set_data([], [])
            label.set_text(f"t={record['t_s']:.1f}s | {NAMES[record['state']]}\n"
                           f"HP={record['hp_after']:.0f}  弹量={record['ammo_after']:.0f}\n"
                           f"剩余={record['remaining_s']:.1f}s  前哨站={'存活' if record['outpost_alive'] else '被毁'}")
        fig.suptitle("二维教学仿真回放（5 倍速）；圆半径 0.30 m，红叉表示已知敌方位置", fontsize=12)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE)
        frames.append(frame)
    frames[0].save(out / "04_二维仿真回放.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    sim = Simulation()
    runs = [sim.run(scenario) for scenario in scenarios()]
    for run in runs:
        file = OUT / (run["scenario"]["id"] + ".json")
        file.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        print(run["scenario"]["id"], " -> ".join(t["target"] for t in run["transitions"]), flush=True)
    summary = dict(kind="确定性二维教学仿真；没有接入实机或比赛系统", random="无随机数；全部情景显式定义",
                   parameters=asdict(sim.p), map="input/map/rmuc_2025.yaml",
                   motion="每 0.1 s 沿 A* 路径移动至下一个安全格中心；最大约 0.707 m/s",
                   anchors={k: dict(cell=list(v), xy=list(cell_to_world(*v, sim.grid))) for k,v in sim.anchors.items()},
                   states_seen=sorted(set().union(*(set(run["states_seen"]) for run in runs))),
                   runs=[dict(id=r["scenario"]["id"], name=r["scenario"]["name"],
                              steps=len(r["records"]), states=r["states_seen"], transitions=r["transitions"],
                              verified_routes=len(r["routes"])) for r in runs])
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(sim, runs, OUT)
    print("结果已写入", OUT)


if __name__ == "__main__":
    main()
