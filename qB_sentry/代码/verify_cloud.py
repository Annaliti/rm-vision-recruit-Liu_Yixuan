"""验证地形规则的可解释例子、真实坡道/边缘和全部自建地图路径。"""
import json
import math
import numpy as np

from build_cloud_map import terrain_from_points
from inspect_map import ROOT, load_map, build_safety
from planner import validate_world_path


def synthetic_cloud(heights, overhead=None):
    rows, cols = np.indices(heights.shape)
    h, w = heights.shape
    a = np.column_stack(((cols.ravel()+.25)*.05, (h-rows.ravel()-.25)*.05, heights.ravel()))
    b = a.copy(); b[:, 0] += .02; b[:, 1] -= .02
    points = np.concatenate([a, b])
    if overhead is not None:
        upper = points.copy(); upper[:, 2] += overhead
        points = np.concatenate([points, upper])
    return points, [0, w*.05, 0, h*.05]


def main():
    height = np.zeros((40, 80))
    tests = []
    points, extent = synthetic_cloud(height)
    flat = terrain_from_points(points, extent)
    assert np.all(flat["classes"][5:-5, 5:-5]==0)
    tests.append("平地可通行")
    ramp_heights = np.tile(np.clip((np.arange(80)*.05-1.0)*.2, 0, .2), (40, 1))
    points, extent = synthetic_cloud(ramp_heights)
    ramp = terrain_from_points(points, extent)
    assert np.all(ramp["classes"][8:-8, 24:36]==0)
    assert np.all(ramp["ramp"][8:-8, 24:36])
    tests.append("约11.3度连续坡道可通行，即使平台高于0.12m")
    step_heights = height.copy(); step_heights[:, 40:] = .20
    points, extent = synthetic_cloud(step_heights)
    step = terrain_from_points(points, extent)
    assert np.all(np.any(step["step"][8:-8, 36:44], axis=1))
    assert np.all(step["classes"][8:-8, 39:41]==2)
    tests.append("0.20m台阶边缘不可跨越")
    for clearance, expected in ((.30, 2), (.75, 0)):
        points, extent = synthetic_cloud(height, overhead=clearance)
        answer = terrain_from_points(points, extent)
        assert np.all(answer["classes"][8:-8, 8:-8]==expected)
    tests.append("同一支撑面上0.30m处结构碰撞，0.75m净空超过0.60m车身")
    # 宽1m的未观测区域不会被局部补孔填成可走。
    points, extent = synthetic_cloud(height)
    points = points[(points[:, 0]<1.0)|(points[:, 0]>2.0)]
    unknown = terrain_from_points(points, extent)
    assert np.all(unknown["classes"][8:-8, 27:33]==1)
    tests.append("大块无观测区域保留未知")

    out = ROOT/"results_cloud"
    report = json.loads((out/"preprocessing.json").read_text())
    transform = np.array(json.loads((out/"coordinate_transform.json").read_text())["T_lidar_to_query_world"])
    assert np.allclose(transform[:3, :3]@transform[:3, :3].T, np.eye(3), atol=1e-8)
    assert math.isclose(np.linalg.det(transform[:3, :3]), 1, abs_tol=1e-8)
    assert 0.30 < report["height_offset_m"] < .50
    assert report["first_plane_angle_to_up_deg"] > 80
    assert report["ground_normal_angle_to_up_deg"] < 1
    ground = np.load(out/"prepared_cloud.npz")["ground_plane"]
    residual = ground@transform[2, :3]+transform[2, 3]
    # RANSAC候选允许0.035m阈值；用95%残差统计验证调平。
    assert np.quantile(abs(residual), .95) < .035
    native = np.load(out/"terrain_native.npz")
    extent = native["extent"]; h, w = native["classes"].shape
    def index(x, y):
        return h-1-int((y-extent[2])/.05), int((x-extent[0])/.05)
    real_ramps = [(x, 1.0) for x in (4.2, 4.5, 4.8, 5.0)] + [(x, 10.6) for x in (18.7, 19.0, 19.3, 19.6)]
    for xy in real_ramps:
        assert native["classes"][index(*xy)] == 0
        assert 3 < native["slope_deg"][index(*xy)] < 25
    real_edges = [(1.5, 1.0), (1.6, 1.0), (7.5, 6.0), (16.5, 6.0)]
    for xy in real_edges:
        assert native["classes"][index(*xy)] == 2

    grid = load_map(out/"cloud_map.yaml")
    assert grid["resolution"] == .05
    assert grid["height"] == 324 and grid["width"] == 583
    assert set(np.unique(grid["gray"])) <= {0, 127, 254}
    safety = build_safety(grid, .30)
    paths = json.loads((ROOT/"results_cloud_planning/summary.json").read_text())
    checked = 0
    for row in paths["queries"]:
        for name, item in row["algorithms"].items():
            saved = np.loadtxt(ROOT/item["path_file"], delimiter=",", skiprows=1)
            validate_world_path(saved, row["start"]["world"], row["goal"]["world"], grid, safety)
            log = ROOT/f"results_cloud_planning/checker_logs/{name}_q{row['query']}.txt"
            assert log.read_text().strip().splitlines()[-1] == "通过"
            checked += 1
    outcome = dict(passed=True, synthetic_tests=tests, real_ramp_probes=len(real_ramps),
                   real_impassable_edge_probes=len(real_edges),
                   ground_residual_p95_m=float(np.quantile(abs(residual), .95)),
                   rotation_is_rigid=True, saved_cloud_paths_checked=checked,
                   note="仅验证明示模型、实际剖面及规划输出；点云异常物语义与真实底盘坡度能力无标注真值")
    (out/"validation.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
