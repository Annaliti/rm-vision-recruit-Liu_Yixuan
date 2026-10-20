#!/usr/bin/env python3
"""验证教学 FSM 的优先级/滞回，以及实际仿真日志和整段运动轨迹。"""
import json
import math

from decision_sim import OUT, Parameters, Simulation, STATES, choose_state, scenarios
from planner import validate_world_path


def main():
    p = Parameters()
    # 输入元组：上一状态、HP、弹量、敌距或None、剩余时间、前哨站存活。
    cases = [
        (("PATROL", 200, 100, None, 240, True), "PATROL"),
        (("PATROL", 60, 100, None, 240, True), "RETREAT"),
        (("PATROL", 61, 100, None, 240, True), "PATROL"),
        (("RETREAT", 61, 100, None, 240, True), "RETREAT"),
        (("RETREAT", 139.9, 100, None, 240, True), "RETREAT"),
        (("RETREAT", 140, 100, None, 240, True), "PATROL"),
        (("PATROL", 200, 8, None, 240, True), "RESUPPLY"),
        (("PATROL", 200, 9, None, 240, True), "PATROL"),
        (("RESUPPLY", 200, 9, None, 240, True), "RESUPPLY"),
        (("RESUPPLY", 200, 59, None, 240, True), "RESUPPLY"),
        (("RESUPPLY", 200, 60, None, 240, True), "PATROL"),
        (("PATROL", 200, 100, 4.0, 240, True), "ATTACK"),
        (("PATROL", 200, 100, 4.1, 240, True), "PATROL"),
        (("ATTACK", 200, 100, 4.1, 240, True), "ATTACK"),
        (("ATTACK", 200, 100, 5.0, 240, True), "ATTACK"),
        (("ATTACK", 200, 100, 5.1, 240, True), "PATROL"),
        (("ATTACK", 200, 100, None, 240, True), "PATROL"),
        (("PATROL", 200, 100, None, 240, False), "DEFEND"),
        (("PATROL", 200, 100, None, 30, True), "ENDGAME"),
        (("PATROL", 200, 100, None, 30.1, True), "PATROL"),
        (("ATTACK", 200, 100, 1, 30, True), "ENDGAME"),
        (("PATROL", 45, 4, 1, 20, False), "RETREAT"),
        (("RETREAT", 140, 4, 1, 20, False), "RESUPPLY"),
        (("RESUPPLY", 140, 60, 1, 20, False), "DEFEND"),
        (("RESUPPLY", 45, 30, 1, 20, False), "RETREAT"),
        (("PATROL", 45, 4, 1, 0, False), "STOP"),
        (("STOP", 200, 100, None, 240, True), "STOP"),
    ]
    for inputs, expected in cases:
        actual, _ = choose_state(*inputs, p)
        assert actual == expected, (inputs, expected, actual)
    sim = Simulation(p)
    states_seen, summaries = set(), []
    expected_transitions = {
        "01_patrol_enemy": ["PATROL", "ATTACK", "PATROL"],
        "02_low_hp": ["RETREAT", "PATROL"],
        "03_low_ammo": ["RESUPPLY", "PATROL"],
        "04_outpost": ["PATROL", "DEFEND"],
        "05_endgame": ["PATROL", "ENDGAME", "STOP"],
        "06_priorities": ["RETREAT", "RESUPPLY", "DEFEND", "STOP"],
    }
    total_steps, total_fired = 0, 0
    for scenario in scenarios():
        data = json.loads((OUT / (scenario["id"] + ".json")).read_text(encoding="utf-8"))
        assert [t["target"] for t in data["transitions"]] == expected_transitions[scenario["id"]]
        records = data["records"]
        route = [records[0]["xy_before"]] + [row["xy"] for row in records]
        geometry = validate_world_path(route, route[0], route[-1], sim.grid, sim.safety)
        previous = "PATROL"
        for index, row in enumerate(records):
            assert row["state_before"] == previous
            predicted, _ = choose_state(previous, row["hp_input"], row["ammo_input"],
                                        row["enemy_distance_m"], row["remaining_s"], row["outpost_alive"], p)
            assert predicted == row["state"]
            if row["enemy_xy"] is not None:
                assert abs(math.dist(row["xy_before"], row["enemy_xy"]) - row["enemy_distance_m"]) < 1e-9
            assert 0 <= row["hp_after"] <= p.hp_max
            assert 0 <= row["ammo_after"] <= p.ammo_max
            assert math.dist(row["xy_before"], row["xy"]) <= math.sqrt(2)*sim.grid["resolution"] + 1e-9
            if row["state"] == "STOP":
                assert row["xy_before"] == row["xy"] and not row["fired"]
            if row["hp_after"] > row["hp_input"]:
                assert row["state"] == "RETREAT" and tuple(row["cell"]) == sim.anchors["home"]
            if row["ammo_after"] > row["ammo_input"]:
                assert row["state"] == "RESUPPLY" and tuple(row["cell"]) == sim.anchors["home"]
            if index:
                assert row["xy_before"] == records[index-1]["xy"]
                assert abs(row["hp_input"] - records[index-1]["hp_after"]) < 1e-9
                assert abs(row["ammo_input"] - records[index-1]["ammo_after"]) < 1e-9
            states_seen.add(row["state"])
            total_fired += row["fired"]
            previous = row["state"]
        total_steps += len(records)
        summaries.append(dict(scenario=scenario["id"], steps=len(records), whole_trajectory=geometry,
                              transitions_match_expected=True, all_inputs_replayed=True))
    assert states_seen == set(STATES)
    assert total_fired > 0
    # 无随机数：重新执行有射击、有状态变化的场景，核对完整轨迹与资源消耗。
    replay = sim.run(scenarios()[0])
    stored = json.loads((OUT / "01_patrol_enemy.json").read_text(encoding="utf-8"))
    assert replay["records"] == stored["records"]
    report = dict(passed=True, boundary_and_priority_cases=len(cases), scenarios=len(summaries),
                  total_simulation_steps=total_steps, states_covered=sorted(states_seen),
                  shots=total_fired, deterministic_replay=True, runs=summaries,
                  limitations="验证为教学二维栅格/障碍中心模型；没有验证真实底盘动力学与官方比赛规则")
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"通过：{len(cases)} 个边界/优先级用例；6 个场景；{total_steps} 步运动；7 个状态；完整日志重放一致。")


if __name__ == "__main__":
    main()
