"""使用本人采集的棋盘格照片进行标定；不填写格长时平移单位为格。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


def calibrate(input_dir, output_dir, square_mm=None, cols=9, rows=6):
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not files:
        raise ValueError('输入目录没有照片')
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= square_mm / 1000 if square_mm is not None else 1.0
    objects, images, accepted, rejected = [], [], [], []
    size = None
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            rejected.append({'image': path.name, 'reason': '无法读取'})
            continue
        current_size = (frame.shape[1], frame.shape[0])
        if size is None:
            size = current_size
        if size != current_size:
            raise ValueError(f'分辨率不一致：{path.name}，禁止混用')
        found, corners = cv2.findChessboardCornersSB(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (cols, rows),
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            rejected.append({'image': path.name, 'reason': '未找到完整角点'})
            continue
        objects.append(obj.copy())
        images.append(corners)
        accepted.append(path)
    if len(accepted) < 5:
        raise ValueError(f'有效照片仅 {len(accepted)} 张，不足以进行本次检查')
    rms, K, distortion, rotations, translations, std_intrinsic, _, per_view = cv2.calibrateCameraExtended(
        objects, images, size, None, None
    )
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('标定输出无效')
    output_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir = output_dir / 'corners'
    annotated_dir.mkdir()
    records = []
    total_sq_error = 0.0
    for i, path in enumerate(accepted):
        projection, _ = cv2.projectPoints(objects[i], rotations[i], translations[i], K, distortion)
        residual = images[i].reshape(-1, 2) - projection.reshape(-1, 2)
        squared = np.sum(residual.astype(np.float64) ** 2, axis=1)
        error = float(np.sqrt(np.mean(squared)))
        total_sq_error += float(squared.sum())
        rotation_matrix, _ = cv2.Rodrigues(rotations[i])
        tilt = float(np.degrees(np.arccos(np.clip(abs(rotation_matrix[2, 2]), 0, 1))))
        points = images[i].reshape(-1, 2)
        coverage = float(cv2.contourArea(cv2.convexHull(points)) / (size[0] * size[1]))
        records.append({'image': path.name, 'rms_px': error, 'estimated_board_tilt_deg': tilt,
                        'inner_corner_hull_image_fraction': coverage,
                        'rvec': rotations[i].ravel().tolist(), 'tvec': translations[i].ravel().tolist()})
        annotated = cv2.imread(str(path))
        cv2.drawChessboardCorners(annotated, (cols, rows), images[i], True)
        cv2.imwrite(str(annotated_dir / path.name), annotated)
    independent_rms = float(np.sqrt(total_sq_error / (len(accepted) * cols * rows)))
    if not np.isclose(independent_rms, rms, atol=1e-4):
        raise RuntimeError('独立复算 RMS 与 OpenCV 结果不一致')
    result = {
        'created_at': datetime.now().astimezone().isoformat(),
        'input_dir': str(input_dir.resolve()), 'image_size': list(size),
        'input_count': len(files), 'accepted_count': len(accepted), 'rejected': rejected,
        'inner_corners': [cols, rows], 'square_mm_measured': square_mm,
        'corner_detector': 'findChessboardCornersSB + CALIB_CB_NORMALIZE_IMAGE; no cornerSubPix',
        'translation_unit': 'm' if square_mm is not None else 'checkerboard_square',
        'camera_matrix': K.tolist(), 'distortion_coefficients': distortion.ravel().tolist(),
        'distortion_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
        'rms_px': float(rms), 'independently_recomputed_rms_px': independent_rms,
        'intrinsic_standard_deviation_opencv_order': std_intrinsic.ravel().tolist(),
        'views': records,
        'notes': ['无实测格长时，内参/畸变仍可估计；棋盘格平移值不能解释为米。',
                  '低重投影误差不等于测距准确；需要后续 A4 实测验证。',
                  '输入视图未按误差自动删除；估计倾角来自标定结果，不是实测角度。'],
    }
    (output_dir / 'calibration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(output_dir / 'camera_params.npz', camera_matrix=K, dist_coeffs=distortion,
             image_size=np.array(size))
    # 原始像素上使用原 K 和畸变；对比图只展示，不作为后续测距输入。
    index = max(range(len(records)), key=lambda i: records[i]['inner_corner_hull_image_fraction'])
    original = cv2.imread(str(accepted[index]))
    corrected = cv2.undistort(original, K, distortion, None, K)
    cv2.imwrite(str(output_dir / 'undistorted.png'), corrected)
    comparison = np.hstack([original, corrected])
    cv2.putText(comparison, 'Original', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(comparison, 'Undistorted (same K, no crop)', (size[0]+20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(str(output_dir / 'undistortion_comparison.jpg'), comparison)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'q2-calib-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(records)+1), [r['rms_px'] for r in records], color='#267b90')
    ax.axhline(rms, color='#b44732', linestyle='--', label=f'Global RMS = {rms:.3f} px')
    ax.set(xlabel='Image index (filename order)', ylabel='Reprojection RMS (pixels)', title='Calibration reprojection error')
    ax.set_xticks(range(1, len(records)+1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'reprojection_errors.png', dpi=160)
    plt.close(fig)
    report = ['# 相机标定结果', '', f'有效照片：{len(accepted)} / {len(files)}；分辨率：{size[0]}×{size[1]}。',
              f'总体重投影 RMS：**{rms:.4f} 像素**。', '',
              f'实测方格边长：{str(square_mm) + " mm" if square_mm is not None else "待补充；此次使用一格为单位"}。', '',
              '## 内参矩阵', '', '```text', np.array2string(K, precision=6), '```', '',
              '畸变系数（k1, k2, p1, p2, k3）：', '', '```text',
              np.array2string(distortion.ravel(), precision=8), '```', '',
              f'去畸变对比使用：{accepted[index].name}。左右保持同一 K、相同尺寸，无裁剪；黑边属于重映射结果。', '',
              '## 逐图结果', '', '| 序号 | 文件 | RMS / px | 估计倾角 / 度 |', '|---|---|---|---|']
    report += [f"| {i+1} | {r['image']} | {r['rms_px']:.4f} | {r['estimated_board_tilt_deg']:.1f} |" for i,r in enumerate(records)]
    if rejected:
        report += ['', '## 未纳入的照片', '', '使用 findChessboardCornersSB 未成功检测完整棋盘格，原始照片保留，未删除：', '']
        report += [f"- {r['image']}：{r['reason']}" for r in rejected]
    report += ['', '## 说明', '',
               'fx、fy 是水平和垂直方向以像素计的焦距；cx、cy 是主点在图像中的坐标。像素近似正方形且图像未非等比缩放时，fx 与 fy 通常接近。', '',
               '重投影 RMS = sqrt(所有角点二维像素残差平方和 / 角点总数)，不是世界坐标误差，也不是米。', '',
               '仅使用这批照片的拟合误差不能证明距离测量准确。下一步需用正确 A4 尺寸、原始分辨率图像、此处内参和畸变进行 PnP，并与本人卷尺数据比较。', '',
               '未填写格长时，不得把棋盘格平移数值解释为米，也不得将模板设计尺寸当作本人实测尺寸。']
    (output_dir / '标定结果.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps({'output_dir': str(output_dir.resolve()), 'accepted':len(accepted), 'rms_px':rms,
                      'K':K.tolist(), 'distortion':distortion.ravel().tolist(),
                      'rms_range':[min(r['rms_px'] for r in records),max(r['rms_px'] for r in records)],
                      'tilt_range':[min(r['estimated_board_tilt_deg'] for r in records),max(r['estimated_board_tilt_deg'] for r in records)]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--square-mm',type=float)
    a=p.parse_args()
    if a.square_mm is not None and a.square_mm <= 0:
        p.error('--square-mm 必须大于 0')
    calibrate(a.input,a.out,a.square_mm)
