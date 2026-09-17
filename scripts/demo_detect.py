# -*- coding: utf-8 -*-
"""命令行快速检测脚本，替代原先散落的 imgTest / VideoTest / CameraTest。

用法：
    python scripts/demo_detect.py --source img --input TestFiles/mature_1.png
    python scripts/demo_detect.py --source video --input path/to/video.mp4
    python scripts/demo_detect.py --source camera --input 0
    python scripts/demo_detect.py --source dir --input TestFiles --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Config  # noqa: E402
import detect_tools as tools  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='命令行版 YOLOv8 葡萄成熟度检测')
    parser.add_argument('--source', choices=('img', 'video', 'camera', 'dir'), default='img')
    parser.add_argument('--input', default=str(ROOT / 'TestFiles'), help='图片/视频/文件夹路径或摄像头编号')
    parser.add_argument('--conf', type=float, default=Config.DEFAULT_CONF_THRES)
    parser.add_argument('--iou', type=float, default=Config.DEFAULT_IOU_THRES)
    parser.add_argument('--save', action='store_true', help='把结果写入 Config.save_path')
    return parser.parse_args()


def detect_image(model, path: str, conf: float, iou: float, save: bool) -> None:
    frame = tools.img_cvread(path)
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
    detection = tools.DetectionResult.from_ultralytics(results, path)
    annotated = tools.annotate_image(frame, detection.boxes, detection.classes, detection.confs)
    print(f'{path}: {len(detection)} 个目标')
    if save:
        target = Path(Config.save_path) / tools.target_path_for(path).name
        print('已保存：', tools.img_cvwrite(target, annotated))
    cv2.imshow('YOLOv8 Detection', annotated)
    cv2.waitKey(0)


def detect_stream(model, source, conf: float, iou: float) -> None:
    cap = cv2.VideoCapture(int(source) if isinstance(source, str) and source.isdigit() else source)
    if not cap.isOpened():
        print(f'无法打开视频源：{source}')
        return
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
            detection = tools.DetectionResult.from_ultralytics(results, str(source))
            annotated = tools.annotate_image(frame, detection.boxes, detection.classes, detection.confs)
            cv2.imshow('YOLOv8 Inference', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(Config.model_path)
    if args.source == 'img':
        detect_image(model, args.input, args.conf, args.iou, args.save)
    elif args.source == 'dir':
        for path in tools.list_image_files(args.input):
            detect_image(model, str(path), args.conf, args.iou, args.save)
    else:
        detect_stream(model, args.input, args.conf, args.iou)


if __name__ == '__main__':
    main()
