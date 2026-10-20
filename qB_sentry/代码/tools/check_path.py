"""
路径检查器：检查你规划出来的路径能不能让哨兵真的走过去。

用法：
    python check_path.py ../data/map/rmuc_2025.yaml your_path.csv

your_path.csv 格式：每行一个路径点，世界坐标，单位米，第一行可以是表头
    x,y
    -2.26,-1.36
    -2.21,-1.31
    ...

检查内容：
  1. 相邻两个路径点之间不能跳太远（<= 0.15 m）
  2. 哨兵底盘按半径 0.30 m 的圆算，任何时候都不能碰到障碍物或未知区域
只告诉你哪里有问题，不告诉你怎么改。
"""
import math
import sys

import cv2
import numpy as np
import yaml

ROBOT_RADIUS = 0.30
MAX_STEP = 0.15

cfg_path, csv_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path))
img = cv2.imread(cfg_path.rsplit("/", 1)[0] + "/" + cfg["image"], cv2.IMREAD_GRAYSCALE)
res, (ox, oy) = cfg["resolution"], cfg["origin"][:2]
h, w = img.shape
p = (255 - img.astype(float)) / 255.0
bad = ~(p < cfg["free_thresh"])  # 占据或未知
bad_pts = np.argwhere(bad)

rows = []
for line in open(csv_path):
    line = line.strip()
    if not line:
        continue
    try:
        x, y = map(float, line.split(",")[:2])
    except ValueError:
        continue  # 表头
    rows.append((x, y))
if len(rows) < 2:
    sys.exit("路径点太少")

problems = 0
for i, (a, b) in enumerate(zip(rows, rows[1:])):
    if math.dist(a, b) > MAX_STEP + 1e-6:
        print(f"[跳跃] 第 {i} -> {i + 1} 个点距离 {math.dist(a, b):.3f} m")
        problems += 1

r_px = int(math.ceil(ROBOT_RADIUS / res))
for i, (x, y) in enumerate(rows):
    c = int(math.floor((x - ox) / res))
    r = h - 1 - int(math.floor((y - oy) / res))
    if not (0 <= r < h and 0 <= c < w):
        print(f"[出界] 第 {i} 个点 ({x:.2f}, {y:.2f}) 在地图外")
        problems += 1
        continue
    r0, r1, c0, c1 = max(r - r_px, 0), min(r + r_px + 1, h), max(c - r_px, 0), min(c + r_px + 1, w)
    win = np.argwhere(bad[r0:r1, c0:c1]) + [r0, c0]
    if len(win):
        dmin = np.min(np.hypot(win[:, 0] - r, win[:, 1] - c)) * res
        if dmin < ROBOT_RADIUS:
            print(f"[碰撞] 第 {i} 个点 ({x:.2f}, {y:.2f}) 离障碍物只有 {dmin:.2f} m")
            problems += 1

length = sum(math.dist(a, b) for a, b in zip(rows, rows[1:]))
print(f"\n路径点 {len(rows)} 个，总长 {length:.2f} m，问题 {problems} 处")
print("通过" if problems == 0 else "没通过")
