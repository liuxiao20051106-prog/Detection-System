"""可复现的 YOLO 训练入口。"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练葡萄成熟度检测模型")
    parser.add_argument("--data", required=True, help="YOLO 数据集 YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型或本地权重")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = Path(args.data).expanduser().resolve(strict=True)
    if data.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--data 必须是 YAML 文件")
    YOLO(args.model).train(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
