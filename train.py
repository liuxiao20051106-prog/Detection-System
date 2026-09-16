# -*- coding: utf-8 -*-
"""模型训练入口。

用法示例：
    python train.py --data datasets/data.yaml --weights yolov8n.pt --epochs 100 --batch 4
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='训练葡萄成熟度检测模型')
    parser.add_argument('--data', default='datasets/data.yaml', help='数据集配置文件')
    parser.add_argument('--weights', default='yolov8n.pt', help='预训练权重')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=4, help='批大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='训练设备，留空时自动选择（有 GPU 会优先用 GPU）')
    parser.add_argument('--project', default='runs/detect', help='输出目录')
    parser.add_argument('--name', default='train', help='本次训练的任务名')
    parser.add_argument('--export', action='store_true', help='训练完成后导出 ONNX')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        project=args.project,
        name=args.name,
    )
    if args.export:
        print(model.export(format='onnx'))


if __name__ == '__main__':
    main()
