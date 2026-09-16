# -*- coding: utf-8 -*-
"""图像处理、结果绘制与文件读写工具集。

模块刻意只依赖 numpy / OpenCV / Pillow：Qt 相关转换延迟到函数内部导入，
这样命令行脚本和单元测试在没有安装 PyQt5 的环境下也能复用本模块。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import Config

IMAGE_EXTENSIONS = frozenset(Config.IMAGE_EXTENSIONS)


# --------------------------------------------------------------------------- #
# 字体与颜色
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    """按字号缓存中文字体。

    字体文件有十几 MB，逐张图片重新解析会明显拖慢绘制，这里按字号做缓存。
    """
    return ImageFont.truetype(Config.font_path, size, 0)


class Colors:
    """用于绘制不同类别的颜色表（YOLO 官方调色板）。"""

    def __init__(self) -> None:
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, index: int, bgr: bool = False) -> Tuple[int, int, int]:
        """返回 RGB 元组；bgr=True 时返回 OpenCV 需要的 BGR 顺序。"""
        color = self.palette[int(index) % self.n]
        return (color[2], color[1], color[0]) if bgr else color

    @staticmethod
    def hex2rgb(hex_color: str) -> Tuple[int, int, int]:
        return tuple(int(hex_color[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


# --------------------------------------------------------------------------- #
# 检测结果数据结构
# --------------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    """一帧结果：只保存纯 Python 数据，可安全跨线程传递。

    ultralytics 的 Results 对象持有 torch 张量并依赖推理上下文，
    交给界面线程长期持有既不高效也不安全，因此在这里做一次“落地”转换。
    """

    source: str = ''
    boxes: List[List[int]] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    duration: float = 0.0

    def __len__(self) -> int:
        return len(self.classes)

    @classmethod
    def from_ultralytics(cls, result, source: str = '', duration: float = 0.0) -> 'DetectionResult':
        boxes = result.boxes
        return cls(
            source=source,
            boxes=[[int(round(value)) for value in box] for box in boxes.xyxy.tolist()],
            classes=[int(each) for each in boxes.cls.tolist()],
            confs=[float(each) for each in boxes.conf.tolist()],
            duration=duration,
        )

    def label(self, index: int) -> str:
        """取指定目标的中文类别名，类别 id 越界时退化成数字，避免崩溃。"""
        class_id = self.classes[index]
        if 0 <= class_id < len(Config.CH_names):
            return Config.CH_names[class_id]
        return str(class_id)

    def conf_text(self, index: int) -> str:
        return '%.2f %%' % (self.confs[index] * 100)

    def option_texts(self) -> List[str]:
        """下拉框选项文本：`<英文类别>_<序号>`，序号用于反查目标。"""
        result = []
        for index, class_id in enumerate(self.classes):
            english = Config.names.get(class_id, str(class_id))
            result.append(f'{english}_{index}')
        return result

    def csv_rows(self, start_index: int = 1) -> List[list]:
        """导出为 CSV 行数据。"""
        rows = []
        for index in range(len(self.classes)):
            rows.append([
                start_index + index,
                self.source,
                self.label(index),
                self.conf_text(index),
                str(self.boxes[index]),
            ])
        return rows


def class_name(class_id: int) -> str:
    """根据类别 id 取中文名称。"""
    if 0 <= class_id < len(Config.CH_names):
        return Config.CH_names[class_id]
    return str(class_id)


# --------------------------------------------------------------------------- #
# 文件读写
# --------------------------------------------------------------------------- #
def img_cvread(path) -> np.ndarray:
    """读取图片，兼容中文路径。

    cv2.imread 在 Windows 上无法处理非 ASCII 路径，这里统一走 np.fromfile + imdecode。
    """
    try:
        buffer = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError as exc:  # 文件不存在、路径是目录等情况
        raise FileNotFoundError(f'无法读取图片：{path}') from exc

    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法解码图片（可能不是图片文件或已损坏）：{path}')
    return img


def img_cvwrite(path, image: np.ndarray) -> str:
    """保存图片，同样兼容 Windows 中文路径。"""
    if image is None:
        raise ValueError('没有可保存的图片')
    extension = os.path.splitext(os.fspath(path))[1] or '.png'
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f'无法编码图片：{path}')
    os.makedirs(os.path.dirname(os.path.abspath(os.fspath(path))), exist_ok=True)
    encoded.tofile(os.fspath(path))
    return os.fspath(path)


def list_image_files(directory) -> List[Path]:
    """返回目录下按名称排序的受支持图片（不递归）。"""
    root = Path(directory)
    with os.scandir(root) as entries:
        files = [
            Path(entry.path) for entry in entries
            if entry.is_file() and entry.name.lower().endswith(tuple(IMAGE_EXTENSIONS))
        ]
    return sorted(files, key=lambda item: item.name.lower())


def target_path_for(source, suffix: str = '_detect_result') -> Path:
    """根据源文件生成 `<原名>_detect_result.<扩展名>` 的输出路径。"""
    source_path = Path(source)
    return source_path.with_name(f'{source_path.stem}{suffix}{source_path.suffix}')


def write_csv(path, rows: Sequence[Sequence], header: Iterable[str] = Config.CSV_HEADERS) -> str:
    """一次性写出 CSV（utf-8-sig 便于 Excel 直接打开）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    return str(target)


# --------------------------------------------------------------------------- #
# 尺寸换算与 Qt 图像转换
# --------------------------------------------------------------------------- #
def fit_size(image: np.ndarray, max_width: int, max_height: int) -> Tuple[int, int]:
    """计算等比缩放后能放进 max_width × max_height 的尺寸。"""
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return max(1, max_width), max(1, max_height)
    ratio = width / height
    if ratio >= max_width / max_height:
        return max(1, max_width), max(1, int(round(max_width / ratio)))
    return max(1, int(round(max_height * ratio))), max(1, max_height)


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """等比缩放；缩小时用 INTER_AREA，画质和耗时都优于默认插值。"""
    width, height = fit_size(image, max_width, max_height)
    if (width, height) == (image.shape[1], image.shape[0]):
        return image
    interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def cvimg_to_qpiximg(cvimg: np.ndarray):
    """把 H×W×3 的 BGR 图像转换成 QPixmap。"""
    from PyQt5.QtGui import QImage, QPixmap  # 延迟导入：命令行脚本无需依赖 Qt

    if cvimg is None or cvimg.ndim != 3 or cvimg.shape[2] not in (3, 4):
        raise ValueError('cvimg 必须是 H×W×3(BGR) 或 H×W×4(BGRA) 的图像')
    height, width = cvimg.shape[:2]
    channels = cvimg.shape[2]
    if channels == 4:
        rgb = cv2.cvtColor(cvimg, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgb.data, width, height, width * 4, QImage.Format_RGBA8888)
    else:
        rgb = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
    # QImage 只是引用 numpy 的内存，转成 QPixmap 拷贝一次可避免图像销毁后花屏。
    return QPixmap(qimg.copy())


# --------------------------------------------------------------------------- #
# 绘制
# --------------------------------------------------------------------------- #
def _clip_box(box: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1:
        x1, x2 = max(0, x1 - 1), x2 + 1
    if y2 <= y1:
        y1, y2 = max(0, y1 - 1), y2 + 1
    return x1, y1, x2, y2


def annotate_image(image: np.ndarray,
                   boxes: Sequence[Sequence[int]],
                   class_ids: Sequence[int],
                   confs: Optional[Sequence[float]] = None,
                   *,
                   colors: Optional[Colors] = None,
                   show_labels: bool = True,
                   font_size: Optional[int] = None,
                   indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """在 BGR 图像上绘制检测框与中文标签，返回新的 BGR 图像。

    :param image: 原始图像（不会被修改）
    :param boxes: 目标框 [[x1, y1, x2, y2], ...]
    :param class_ids: 与 boxes 对应的类别 id
    :param confs: 与 boxes 对应的置信度（0~1），为 None 时不显示百分比
    :param show_labels: 为 False 时只画框
    :param indices: 只绘制指定下标的目标，None 表示全部
    """
    canvas = np.ascontiguousarray(image)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]

    targets = range(len(boxes)) if indices is None else [i for i in indices if 0 <= i < len(boxes)]
    if not boxes or not list(targets):
        return canvas

    palette = colors or Colors()
    line_width = max(2, int(round(min(height, width) / 320)))

    if not show_labels:
        for index in targets:
            x1, y1, x2, y2 = _clip_box(boxes[index], width, height)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), palette(class_ids[index], True), line_width)
        return canvas

    if font_size is None:
        font_size = max(14, int(round(min(height, width) * 0.032)))
    font = load_font(font_size)
    padding = max(2, font_size // 6)

    # PIL 侧统一用 RGB；只对画布做一次转换，避免逐框来回转换导致画面偏色。
    pil_image = Image.fromarray(canvas[..., ::-1])
    draw = ImageDraw.Draw(pil_image)

    for index in targets:
        class_id = class_ids[index]
        color = palette(class_id)  # RGB
        if confs is None:
            text = class_name(class_id)
        else:
            text = f'{class_name(class_id)} {float(confs[index]) * 100:.0f}%'

        x1, y1, x2, y2 = _clip_box(boxes[index], width, height)
        text_left, text_top, text_right, text_bottom = font.getbbox(text)
        bar_height = (text_bottom - text_top) + padding * 2
        bar_width = (text_right - text_left) + padding * 2

        bar_top = y1 - bar_height
        if bar_top < 0:  # 目标贴顶时把标签放到框内侧，避免被裁掉
            bar_top = y1
        if bar_top + bar_height > height:
            bar_top = max(0, min(height - bar_height, y2 - bar_height))
        bar_right = min(width, x1 + bar_width)
        if bar_right - x1 < bar_width:  # 目标贴右边界时向左收缩
            bar_right, bar_left = x1, max(0, x1 - bar_width)
        else:
            bar_left = x1

        draw.rectangle([bar_left, bar_top, bar_right, bar_top + bar_height], fill=color)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        draw.text((bar_left + padding, bar_top + padding - text_top), text, fill=(255, 255, 255), font=font)

    return np.ascontiguousarray(np.asarray(pil_image)[..., ::-1])


# --------------------------------------------------------------------------- #
# YOLO 标注格式互转
# --------------------------------------------------------------------------- #
def yolo_to_location(width: int, height: int, yolo_data: Sequence[float]) -> List[int]:
    """YOLO 归一化格式 (cx, cy, w, h) 转左上角/右下角两点坐标，并裁剪到图像范围内。"""
    x_, y_, w_, h_ = yolo_data
    x1 = int(width * x_ - 0.5 * width * w_)
    x2 = int(width * x_ + 0.5 * width * w_)
    y1 = int(height * y_ - 0.5 * height * h_)
    y2 = int(height * y_ + 0.5 * height * h_)
    return [
        max(0, min(int(width - 1), x1)),
        max(0, min(int(height - 1), y1)),
        max(0, min(int(width - 1), x2)),
        max(0, min(int(height - 1), y2)),
    ]


def location_to_yolo(width: int, height: int, locations: Sequence[float]) -> List[float]:
    """左上角/右下角两点坐标转 YOLO 归一化格式，保留 5 位小数。"""
    x1, y1, x2, y2 = locations
    x_ = float('%.5f' % ((x1 + x2) / 2 / width))
    y_ = float('%.5f' % ((y1 + y2) / 2 / height))
    w_ = float('%.5f' % ((x2 - x1) / width))
    h_ = float('%.5f' % ((y2 - y1) / height))
    return [x_, y_, w_, h_]
