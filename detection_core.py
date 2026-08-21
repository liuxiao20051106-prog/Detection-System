"""与界面无关的输入校验、结果转换和安全导出工具。"""

from __future__ import annotations

import csv
import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".wmv", ".mkv", ".mov", ".m4v"}
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
MAX_VIDEO_BYTES = 20 * 1024 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 4 * 60 * 60
MAX_TABLE_ROWS = 500


class ValidationError(ValueError):
    """外部文件或配置未通过安全校验。"""


@dataclass(frozen=True)
class DetectionItem:
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class DetectionFrame:
    source: str
    original_image: np.ndarray
    annotated_image: np.ndarray
    items: tuple[DetectionItem, ...]
    inference_seconds: float


@dataclass(frozen=True)
class BatchRecord:
    source: str
    cached_image: str
    items: tuple[DetectionItem, ...]


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_existing(path: os.PathLike | str, kind: str) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValidationError(f"{kind}不存在或无法访问：{path}") from error


def validate_model_file(path: os.PathLike | str, expected_sha256: str) -> Path:
    model_path = _resolve_existing(path, "模型")
    if not model_path.is_file():
        raise ValidationError(f"模型路径不是文件：{model_path}")
    if model_path.suffix.lower() not in {".pt", ".onnx"}:
        raise ValidationError("仅支持经过校验的 .pt 或 .onnx 模型。")
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValidationError("必须通过 DETECTION_MODEL_SHA256 提供64位模型哈希。")
    actual = sha256_file(model_path)
    if not hmac.compare_digest(actual.lower(), expected_sha256.lower()):
        raise ValidationError(
            f"模型完整性校验失败。拒绝加载可能被替换或来源不明的模型。\n实际 SHA-256：{actual}"
        )
    return model_path


def validate_image_file(path: os.PathLike | str) -> np.ndarray:
    image_path = _resolve_existing(path, "图片")
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValidationError(f"不支持的图片文件：{image_path.name}")
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValidationError(
            f"图片超过 {MAX_IMAGE_BYTES // 1024 // 1024} MiB 限制：{image_path.name}"
        )
    try:
        with Image.open(image_path) as header:
            width, height = header.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValidationError(f"图片无法解码或内容已损坏：{image_path.name}") from error
    if height <= 0 or width <= 0 or height * width > MAX_IMAGE_PIXELS:
        raise ValidationError(f"图片像素尺寸超出限制：{image_path.name}")
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError(f"图片无法解码或内容已损坏：{image_path.name}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or height * width > MAX_IMAGE_PIXELS:
        raise ValidationError(f"图片像素尺寸超出限制：{image_path.name}")
    return image


def validate_video_file(path: os.PathLike | str) -> VideoInfo:
    video_path = _resolve_existing(path, "视频")
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValidationError(f"不支持的视频文件：{video_path.name}")
    if video_path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValidationError(f"视频超过 {MAX_VIDEO_BYTES // 1024 // 1024 // 1024} GiB 限制。")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValidationError(f"视频无法打开或编码不受支持：{video_path.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValidationError(f"视频分辨率无效或超出限制：{video_path.name}")
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        if frame_count > 0 and frame_count / fps > MAX_VIDEO_DURATION_SECONDS:
            raise ValidationError("视频时长超过4小时限制。")
        return VideoInfo(str(video_path), width, height, fps, max(frame_count, 0))
    finally:
        capture.release()


def list_image_files(directory: os.PathLike | str) -> list[str]:
    root = _resolve_existing(directory, "图片目录")
    if not root.is_dir():
        raise ValidationError(f"图片目录不存在：{root}")
    return [
        str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def make_output_path(
    source: os.PathLike | str,
    output_directory: os.PathLike | str,
    suffix: str = "_detect_result",
    extension: str | None = None,
) -> Path:
    source_path = Path(source)
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_extension = extension or source_path.suffix.lower() or ".png"
    candidate = output_root / f"{source_path.stem}{suffix}{output_extension}"
    counter = 2
    while candidate.exists():
        candidate = output_root / f"{source_path.stem}{suffix}_{counter}{output_extension}"
        counter += 1
    return candidate


def image_write(path: os.PathLike | str, image: np.ndarray) -> Path:
    if image is None:
        raise ValidationError("没有可保存的图片。")
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"无法编码图片：{output_path.name}")
    encoded.tofile(output_path)
    return output_path


def csv_safe(value: object) -> str:
    text = str(value)
    if text.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def write_detection_csv(
    path: os.PathLike | str,
    rows: Iterable[Sequence[object]],
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["序号", "文件", "类别", "置信度", "xmin", "ymin", "xmax", "ymax"])
        for index, row in enumerate(rows, 1):
            writer.writerow([index, *(csv_safe(value) for value in row)])
    return output_path


def frame_csv_rows(frame: DetectionFrame | BatchRecord) -> list[list[object]]:
    return [
        [
            frame.source,
            item.display_name,
            f"{item.confidence:.4f}",
            *item.box,
        ]
        for item in frame.items
    ]
