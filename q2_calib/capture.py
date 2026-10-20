"""Mac 摄像头采集：原始照片与拍摄信息；不会自动开始保存。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2


def detect_board(frame, pattern=(9, 6)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    return found, corners


def save_frame(directory, frame, metadata):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    image_path = directory / (stamp + '.png')
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f'无法保存照片：{image_path}')
    metadata = {**metadata, 'image': image_path.name,
                'captured_at': datetime.now().astimezone().isoformat()}
    image_path.with_suffix('.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return image_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0, help='摄像头编号；看预览确认是内置摄像头')
    parser.add_argument('--mode', choices=['board', 'a4'], default='board')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--cols', type=int, default=9, help='横向内角点数')
    parser.add_argument('--rows', type=int, default=6, help='纵向内角点数')
    parser.add_argument('--square-mm', type=float, help='实际量得的方格边长，毫米；未知可暂不填写')
    parser.add_argument('--distance-m', type=float, help='A4 模式：卷尺实测镜头附近到纸中心的距离，米')
    parser.add_argument('--angle-deg', type=float, help='可选：实际测得的纸张倾斜角度')
    parser.add_argument('--group', choices=['main', 'front_only'], default='main')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    if min(args.width, args.height, args.cols, args.rows) <= 0:
        parser.error('分辨率、内角点数量必须为正数')
    if args.square_mm is not None and args.square_mm <= 0:
        parser.error('方格边长必须大于 0')
    if args.mode == 'a4' and (args.distance_m is None or args.distance_m <= 0):
        parser.error('A4 模式请填写真实卷尺读数：--distance-m 距离值')

    session = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = args.out / args.mode / args.group / session
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    window = 'Q2 capture | S: save in 5s | A: auto 5s | Q: quit'
    try:
        if not cap.isOpened():
            raise RuntimeError('无法打开摄像头。请检查启动程序的终端/编辑器的相机权限、关闭占用相机的软件，或更换 --camera 编号。')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        directory.mkdir(parents=True, exist_ok=False)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 540)
        print('先确认画面来自 Mac 内置摄像头。照片不做镜像、裁剪或缩放。')
        print('点击预览窗口后按 S：5 秒后拍一张；A：每 5 秒拍摄/暂停；Q：退出。')
        print('棋盘格模式只保存检测成功的帧；自动采集到本次共 25 张时暂停。')
        print('保存目录：', directory)
        saved = 0
        automatic = False
        due = None
        shape = None
        message = 'S: capture in 5s | A: auto | Q: quit'
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('摄像头读取失败；已保存的照片保留。')
            if shape is None:
                shape = frame.shape[:2]
                print(f'实际分辨率：{shape[1]} x {shape[0]}；后续测距必须保持一致。')
            elif frame.shape[:2] != shape:
                raise RuntimeError('相机分辨率发生变化，停止采集以避免混用。')
            display = frame.copy()
            found, corners = False, None
            if args.mode == 'board':
                found, corners = detect_board(frame, (args.cols, args.rows))
                if found:
                    cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, found)
            now = time.monotonic()
            if due is not None and now >= due:
                if args.mode == 'a4' or found:
                    path = save_frame(directory, frame, {
                        'mode': args.mode, 'group': args.group,
                        'camera_index': args.camera, 'backend': cap.getBackendName(),
                        'resolution': [shape[1], shape[0]],
                        'inner_corners': [args.cols, args.rows] if args.mode == 'board' else None,
                        'square_mm_measured': args.square_mm if args.mode == 'board' else None,
                        'distance_m_measured': args.distance_m if args.mode == 'a4' else None,
                        'angle_deg_measured': args.angle_deg if args.mode == 'a4' else None,
                        'opencv_version': cv2.__version__,
                    })
                    saved += 1
                    message = f'Saved {saved}: {path.name}'
                    print(message)
                else:
                    message = 'Skipped: full checkerboard not detected'
                    print('本次未保存：未检测到完整棋盘格，请调整角度、距离或光线。')
                if saved >= 25:
                    automatic = False
                due = now + 5 if automatic else None
            status = f'{shape[1]}x{shape[0]} | saved={saved} | {args.mode}'
            if args.mode == 'board':
                status += ' | BOARD OK' if found else ' | BOARD NOT FOUND'
            if due is not None:
                status += f' | {max(0, due - now):.1f}s'
            cv2.rectangle(display, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
            cv2.putText(display, message, (10, 53), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('s'), ord('S')):
                automatic = False
                due = time.monotonic() + 5
            if key in (ord('a'), ord('A')):
                automatic = not automatic
                due = time.monotonic() + 5 if automatic else None
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
