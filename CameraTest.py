"""摄像头命令行预览工具。按 q 结束。"""

import argparse

import cv2
from ultralytics import YOLO

import Config
from detection_core import validate_model_file


def main():
    parser = argparse.ArgumentParser(description="预览摄像头检测")
    parser.add_argument("--camera", type=int, default=Config.camera_id)
    args = parser.parse_args()

    model_path = validate_model_file(Config.model_path, Config.model_sha256)
    model = YOLO(str(model_path), task="detect")
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")
    try:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            result = model(frame, verbose=False)[0]
            cv2.imshow("Detection-System", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
