"""由原始点云独立生成 0.05 m 地形与可通行地图。

先识别地面和边界方向，再在局部地面上判断坡度、台阶及车身空间。
提供的 PGM 仅用于估计坐标平移和比较，不用于生成自建地图的格子标签。
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
import yaml

from inspect_map import ROOT, load_map


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def prepare_cloud(pcd_path, out):
    """最大平面不等于地面；用高度先验及墙面/地面正交性共同判断。"""
    tick = time.perf_counter()
    o3d.utility.random.seed(20251019)
    raw = o3d.io.read_point_cloud(str(pcd_path))
    raw_count = len(raw.points)
    raw.remove_non_finite_points()
    coarse = raw.voxel_down_sample(0.08)
    remaining = coarse
    records, plane_points = [], []
    for i in range(8):
        plane, indices = remaining.segment_plane(0.035, 3, 700)
        plane = np.asarray(plane)
        if plane[2] < 0:
            plane = -plane
        selected = np.asarray(remaining.points)[indices]
        records.append(dict(index=i, count=len(indices), equation=plane.tolist()))
        plane_points.append(selected)
        remaining = remaining.select_by_index(indices, invert=True)
    # 前两个不同方向的大竖墙法向的叉积，给出独立于层高的向上方向。
    normal_a = np.array(records[0]["equation"][:3])
    other = next(i for i in range(1, 4) if abs(normal_a @ np.array(records[i]["equation"][:3])) < 0.2)
    normal_b = np.array(records[other]["equation"][:3])
    up = np.cross(normal_a, normal_b)
    up /= np.linalg.norm(up)
    if up[2] < 0:
        up = -up
    candidates = [i for i, p in enumerate(records)
                  if abs(np.array(p["equation"][:3])@up) > math.cos(math.radians(10))
                  and 0.20 < abs(p["equation"][3]) < 0.65]
    if not candidates:
        raise RuntimeError("没有找到符合雷达离地约0.4m且与墙正交的地面候选")
    ground_index = min(candidates, key=lambda i: abs(abs(records[i]["equation"][3])-0.4))
    # 用地面点的投影中位数定零，不直接硬编码0.4m。
    height_offset = -float(np.median(plane_points[ground_index] @ up))
    axis_x = normal_b - (normal_b@up)*up
    axis_x /= np.linalg.norm(axis_x)
    if axis_x[1] > 0:
        axis_x = -axis_x
    axis_y = np.cross(up, axis_x)
    rotation = np.stack([axis_x, axis_y, up])
    assert np.allclose(rotation@rotation.T, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) > 0.999999
    print(f"原始 {raw_count:,} 点；首次平面 {records[0]['count']:,} 点；地面候选第 {ground_index+1} 个；传感器高度 {height_offset:.4f} m", flush=True)

    down = raw.voxel_down_sample(0.04)
    before_filter = len(down.points)
    cleaned, kept = down.remove_statistical_outlier(nb_neighbors=24, std_ratio=3.0)
    clean_raw = np.asarray(cleaned.points)
    aligned = clean_raw@rotation.T
    aligned[:, 2] += height_offset
    removed = np.asarray(down.points)[np.setdiff1d(np.arange(before_filter), kept)]
    removed = removed@rotation.T
    removed[:, 2] += height_offset
    sample = np.asarray(raw.points)[::max(1, raw_count//100000)]
    np.savez_compressed(out/"prepared_cloud.npz", points=aligned.astype(np.float32),
                        removed=removed.astype(np.float32), raw_sample=sample.astype(np.float32),
                        first_plane=plane_points[0].astype(np.float32),
                        ground_plane=plane_points[ground_index].astype(np.float32))
    # 边界位置由墙面自身的点决定，后续只裁掉明显在场地外的噪声。
    wall_coordinates = [p@rotation.T for p in plane_points[:4]]
    wall_x = sorted(float(np.median(p[:, 0])) for p in wall_coordinates
                    if np.std(p[:, 0]) < 0.10)
    wall_y = sorted(float(np.median(p[:, 1])) for p in wall_coordinates
                    if np.std(p[:, 1]) < 0.10)
    bounds = [wall_x[0], wall_x[-1], wall_y[0], wall_y[-1]]
    report = dict(raw_points=raw_count, finite_points=len(raw.points), coarse_voxel_m=0.08,
                  voxel_m=0.04, downsampled_points=before_filter,
                  filtered_points=len(aligned), sor_removed_points=len(removed),
                  sor_neighbors=24, sor_std_ratio=3.0, planes=records,
                  ground_plane_index=ground_index, height_offset_m=height_offset,
                  up_vector=up.tolist(), rotation_lidar_to_boundary=rotation.tolist(),
                  sensor_tilt_deg=math.degrees(math.acos(up[2])),
                  first_plane_angle_to_up_deg=math.degrees(math.acos(abs(normal_a@up))),
                  ground_normal_angle_to_up_deg=math.degrees(math.acos(min(1.0, abs(np.array(records[ground_index]["equation"][:3])@up)))),
                  boundary_extent=bounds, preparation_seconds=time.perf_counter()-tick,
                  assumptions="竖直正交边界墙、雷达离地约0.4m；用实测地面候选复核，不是盲目将最大平面当地面")
    save_json(out/"preprocessing.json", report)
    print(f"降采样 {before_filter:,} → 统计滤波 {len(aligned):,}；边界 {bounds}", flush=True)
    return aligned.astype(np.float32), report


def raster_quantiles(points, extent, resolution):
    """按xy分格并在格内按z排序；保留高度分布，而非全图固定高度截断。"""
    xmin, xmax, ymin, ymax = extent
    width, height = int(round((xmax-xmin)/resolution)), int(round((ymax-ymin)/resolution))
    cols = np.floor((points[:, 0]-xmin)/resolution).astype(int)
    rows = height-1-np.floor((points[:, 1]-ymin)/resolution).astype(int)
    inside = (cols>=0)&(cols<width)&(rows>=0)&(rows<height)
    pts, rows, cols = points[inside], rows[inside], cols[inside]
    ids = rows*width+cols
    order = np.lexsort((pts[:, 2], ids))
    ids, pts = ids[order], pts[order]
    unique, start, count = np.unique(ids, return_index=True, return_counts=True)
    arrays = {}
    for name, q in (("z10", 0.10), ("z50", 0.50), ("z90", 0.90)):
        a = np.full(width*height, np.nan)
        a[unique] = pts[start+np.floor((count-1)*q).astype(int), 2]
        arrays[name] = a.reshape(height, width)
    counts = np.zeros(width*height, np.int32)
    counts[unique] = count
    arrays["count"] = counts.reshape(height, width)
    return arrays


def terrain_from_points(points, extent, resolution=0.05):
    """生成地面高度、坡度、台阶边缘、车身障碍与可通行分类。"""
    # 低于地面0.1m明显不合理；2m以上不是本场地的支撑地形。
    # 高墙仍保留其低处截面；这不是用一个高度阈值把所有高地变成障碍。
    usable = points[(points[:, 2] >= -0.10)&(points[:, 2] <= 2.0)]
    arrays = raster_quantiles(usable, extent, resolution)
    count, z10, z50, z90 = (arrays[k] for k in ("count", "z10", "z50", "z90"))
    observed = count >= 1
    # 先使用最低一簇点估计支撑面；竖直结构随后由车身空间检查拦截。
    raw_surface = z10.copy()
    valid = observed & np.isfinite(raw_surface)
    distance, nearest = ndimage.distance_transform_edt(~valid, return_indices=True)
    nearest_height = raw_surface[tuple(nearest)]
    filled = (~valid)&(distance*resolution <= 0.15)
    # 扫描线之间的小孔：只有周边观测高度一致、孔被观测包围，才补到0.30m。
    # 这项依据局部表面证据，不根据查询点是否通过来修改地图。
    ry, rx = np.mgrid[-6:7, -6:7]
    disk = rx*rx+ry*ry <= 36
    local_max = ndimage.maximum_filter(np.where(valid, raw_surface, -np.inf), footprint=disk)
    local_min = ndimage.minimum_filter(np.where(valid, raw_surface, np.inf), footprint=disk)
    local_count = ndimage.convolve(valid.astype(np.int16), disk.astype(np.int16))
    enclosed = ndimage.binary_closing(valid, structure=disk)
    planar_fill = (~valid)&enclosed&(distance*resolution <= .30)&(local_count>=8)&((local_max-local_min)<.08)
    filled |= planar_fill
    support = valid|filled
    elevation = np.where(valid, raw_surface, nearest_height)
    # 中值去掉少数低离群值；不会大范围补全未观测区域。
    elevation = ndimage.median_filter(elevation, size=5, mode="nearest")
    smooth = ndimage.gaussian_filter(elevation, sigma=1.4)
    dzdy, dzdx = np.gradient(smooth, resolution)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    step = np.zeros(support.shape, bool)
    max_jump = np.zeros(support.shape)
    for dr, dc in ((1, 0), (0, 1)):
        a = (slice(None, -dr or None), slice(None, -dc or None))
        b = (slice(dr, None), slice(dc, None))
        jump = np.abs(elevation[a]-elevation[b])
        bad = support[a]&support[b]&(jump > 0.12+1e-6)
        step[a] |= bad
        step[b] |= bad
        max_jump[a] = np.maximum(max_jump[a], jump)
        max_jump[b] = np.maximum(max_jump[b], jump)
    # 在本地支撑面以上0.12~0.60m内有点，意味着车身会撞上垂直结构。
    xmin, xmax, ymin, ymax = extent
    h, w = support.shape
    cols = np.floor((usable[:, 0]-xmin)/resolution).astype(int)
    rows = h-1-np.floor((usable[:, 1]-ymin)/resolution).astype(int)
    mask = (cols>=0)&(cols<w)&(rows>=0)&(rows<h)
    pts, r, c = usable[mask], rows[mask], cols[mask]
    above = pts[:, 2]-elevation[r, c]
    hit = (above > 0.12)&(above < 0.60)
    body_count = np.zeros((h, w), np.int32)
    np.add.at(body_count, (r[hit], c[hit]), 1)
    body = body_count >= 2
    steep = slope_deg > 25.0  # 题目未给最大坡度；明确采用25度教学参数。
    blocked = support&(step|body|steep)
    traversable = support&~blocked
    ramp = traversable&(slope_deg >= 3.0)
    classes = np.full((h, w), 1, np.uint8)  # 0可通行、1未知、2障碍
    classes[traversable] = 0
    classes[blocked] = 2
    arrays.update(elevation=elevation, slope_deg=slope_deg, support=support, filled=filled,
                  step=step, max_jump=max_jump, body=body, steep=steep, planar_fill=planar_fill,
                  ramp=ramp, classes=classes)
    return arrays


def register_translation(points, boundary, reference):
    """边界已对正，仅估计平移；匹配实测竖直结构，绝不复制参考图格子标签。"""
    from scipy.optimize import differential_evolution
    p = points
    xmin, xmax, ymin, ymax = boundary
    selected = p[(p[:, 2]>.55)&(p[:, 2]<1.45)&(p[:, 0]>xmin+.3)&(p[:, 0]<xmax-.3)
                 &(p[:, 1]>ymin+.3)&(p[:, 1]<ymax-.3), :2]
    selected = np.unique(np.floor(selected/.08), axis=0)*.08+.04
    distance = ndimage.distance_transform_edt(reference["free"])*reference["resolution"]
    center = np.array([(reference["extent"][0]+reference["extent"][1]-xmin-xmax)/2,
                       (reference["extent"][2]+reference["extent"][3]-ymin-ymax)/2])

    def residual(t):
        q = selected+t
        coordinates = np.stack([reference["height"]-.5-(q[:, 1]-reference["oy"])/reference["resolution"],
                                (q[:, 0]-reference["ox"])/reference["resolution"]-.5])
        return ndimage.map_coordinates(distance, coordinates, order=1, mode="constant", cval=2)

    fit = differential_evolution(lambda t: np.mean(np.minimum(residual(t), .5)**2),
                                 [(x-1, x+1) for x in center], seed=42, popsize=10,
                                 maxiter=60, tol=1e-7, workers=1)
    distances = residual(fit.x)
    return fit.x, dict(method="点云内场0.55~1.45m结构与参考占据区的截断距离匹配，仅估计二维平移",
                       initial_translation_m=center.tolist(), translation_m=fit.x.tolist(),
                       matching_points=len(selected), truncated_rmse_m=float(np.sqrt(fit.fun)),
                       within_0_15_m_fraction=float(np.mean(distances<=.15)),
                       independent_truth=False,
                       limitation="参考地图只校准共享坐标，不生成地形标签；结构有差异且场地近似中心对称，配准需检查叠加图")


def write_grid(out, name, arrays, extent):
    pixels = np.select([arrays["classes"]==0, arrays["classes"]==2], [254, 0], default=127).astype(np.uint8)
    cv2.imwrite(str(out/f"{name}.pgm"), pixels)
    cv2.imwrite(str(out/f"{name}.png"), pixels)
    config = dict(image=f"{name}.pgm", mode="trinary", resolution=.05,
                  origin=[float(extent[0]), float(extent[2]), 0.0], negate=0,
                  occupied_thresh=.65, free_thresh=.25)
    (out/f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def make_map(points, report, out):
    xmin, xmax, ymin, ymax = report["boundary_extent"]
    outside = ((points[:, 0]<xmin-.10)|(points[:, 0]>xmax+.10)|
               (points[:, 1]<ymin-.10)|(points[:, 1]>ymax+.10))
    below = points[:, 2] < -.10
    clean = points[~outside&~below]
    boundary_distance = np.minimum.reduce([abs(points[:, 0]-xmin), abs(points[:, 0]-xmax),
                                           abs(points[:, 1]-ymin), abs(points[:, 1]-ymax)])
    high_boundary = (boundary_distance<.08)&(points[:, 2]>2.0)
    anomaly = dict(sor_removed=report["sor_removed_points"],
                   outside_boundary_after_sor=int(outside.sum()),
                   below_ground_after_sor=int(below.sum()),
                   union_removed_after_sor=int((outside|below).sum()),
                   retained_points=len(clean),
                   high_boundary_points_ignored_for_ground_map=int(high_boundary.sum()),
                   boundary_height_p99_m=float(np.quantile(points[high_boundary, 2], .99)),
                   tall_boundary="检测到约18.5m高矩形围挡，保留近地碰撞截面；其仿真语义未经元数据证实，不贸然删除整面墙",
                   cause="孤立散点、地下/场外点与仿真表面不一致，可能来自注入噪声、异常回波或累积配准误差；仅PCD不能证明具体来源",
                   object_semantics="不能仅凭孤立或高度判定物体不存在；无法证实的实体保留，净空足够的高处点不投影为车身障碍")
    save_json(out/"anomaly_analysis.json", anomaly)
    extent = [math.floor((xmin-.10)/.05)*.05, math.ceil((xmax+.10)/.05)*.05,
              math.floor((ymin-.10)/.05)*.05, math.ceil((ymax+.10)/.05)*.05]
    arrays = terrain_from_points(clean, extent)
    np.savez_compressed(out/"terrain_native.npz", **arrays, extent=extent, resolution=.05)
    write_grid(out, "cloud_map_native", arrays, extent)
    reference = load_map(ROOT/"input/map/rmuc_2025.yaml")
    translation, registration = register_translation(clean, report["boundary_extent"], reference)
    aligned = clean.copy()
    aligned[:, :2] += translation
    shared = terrain_from_points(aligned, reference["extent"])
    np.savez_compressed(out/"terrain.npz", **shared, extent=reference["extent"], resolution=.05)
    write_grid(out, "cloud_map", shared, reference["extent"])
    transform = np.eye(4)
    transform[:3, :3] = report["rotation_lidar_to_boundary"]
    transform[:3, 3] = [*translation, report["height_offset_m"]]
    registration["T_lidar_to_query_world"] = transform.tolist()
    registration["T_query_world_to_lidar"] = np.linalg.inv(transform).tolist()
    save_json(out/"coordinate_transform.json", registration)
    # 保存稀疏着色点云便于在Open3D里自行旋转查看，原始点云在input/cloud保留。
    preview = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
    preview = preview.voxel_down_sample(.08)
    o3d.io.write_point_cloud(str(out/"aligned_cloud.ply"), preview, compressed=True)
    free = shared["classes"] == 0
    known = shared["classes"] != 1
    stats = dict(resolution_m=.05, shape=list(free.shape),
                 traversable_cells=int(free.sum()), unknown_cells=int((~known).sum()),
                 blocked_cells=int((shared["classes"]==2).sum()),
                 ramp_cells=int(shared["ramp"].sum()), step_edge_cells=int(shared["step"].sum()),
                 body_obstacle_cells=int(shared["body"].sum()),
                 gap_filled_cells=int(shared["filled"].sum()),
                 max_step_m=.12, chassis_height_m=.60, max_slope_deg=25,
                 nearest_gap_fill_m=.15, planar_enclosed_gap_fill_m=.30,
                 planar_fill_cells=int(shared["planar_fill"].sum()), slope_is_assumption=True,
                 agreement_with_reference=float(np.mean(free==reference["free"])),
                 free_iou=float(np.sum(free&reference["free"])/np.sum(free|reference["free"])),
                 comparison_note="参考地图非标注真值，IoU仅描述差异；自建图的占据标签全部由点云计算")
    save_json(out/"map_summary.json", stats)
    print("自建地图已导出", stats, flush=True)
    return shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, default=ROOT/"input/cloud/rmuc_2025_map.pcd")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = ROOT/"results_cloud"
    out.mkdir(exist_ok=True)
    if args.reuse_prepared and (out/"prepared_cloud.npz").exists():
        points = np.load(out/"prepared_cloud.npz")["points"]
        report = json.loads((out/"preprocessing.json").read_text())
    else:
        points, report = prepare_cloud(args.pcd, out)
    if args.prepare_only:
        return
    make_map(points, report, out)


if __name__ == "__main__":
    main()
