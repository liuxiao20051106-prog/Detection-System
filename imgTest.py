"""单张图片命令行检测工具。"""

import argparse
from pathlib import Path

from ultralytics import YOLO

import Config
from detection_core import image_write, make_output_path, validate_image_file, validate_model_file


def main():
    parser = argparse.ArgumentParser(description="检测单张葡萄图片")
    parser.add_argument("image", help="待检测图片")
    parser.add_argument("--output", default=Config.save_path, help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    image = validate_image_file(args.image)
    model_path = validate_model_file(Config.model_path, Config.model_sha256)
    result = YOLO(str(model_path), task="detect")(
        image, conf=args.conf, iou=args.iou, verbose=False
    )[0]
    output = make_output_path(args.image, args.output)
    image_write(output, result.plot())
    print(Path(output))


if __name__ == "__main__":
    main()
