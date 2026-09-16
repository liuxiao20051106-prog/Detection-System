# -*- coding: utf-8 -*-
"""后台推理线程。

界面线程只负责渲染与交互，耗时的 YOLO 推理、绘框和批量写盘都放在这里，
这样批量检测、实时视频或导出视频时窗口不会被卡住。

设计要点：

1. 线程之间共用同一个 YOLO 实例（显存占用更友好），所有 ``predict`` 调用都
   串行化，统一通过 ``BaseDetectionWorker.predict()`` 加锁；
2. 跨线程只传递纯 Python 数据（``DetectionResult``）和 numpy 图像，
   不让 torch 张量或 ultralytics 的 Results 留在界面线程里；
3. **视频帧的采集留在主线程**（见 ``open_video_capture``）：OpenCV 的视频后端
   （Windows 的 MSMF / DirectShow 等）对“在哪个线程初始化”很敏感，放到工作
   线程里可能出现卡死或设备打不开；取帧本身开销极小，交给主线程最稳。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import Config
import detect_tools as tools

LOGGER = logging.getLogger(__name__)

SourceType = Union[str, int]

# Windows 下用 DirectShow 打开摄像头通常比默认后端快很多。
_CAMERA_BACKENDS = tuple(
    backend for backend in (getattr(cv2, 'CAP_DSHOW', None), getattr(cv2, 'CAP_ANY', None)) if backend is not None
) or (0,)


def open_video_capture(source: SourceType, camera_backends=_CAMERA_BACKENDS) -> Optional[cv2.VideoCapture]:
    """在主线程里打开视频源；返回 None 表示打不开。

    摄像头会依次尝试 DirectShow 等后端；视频文件则用默认后端一次打开。
    """
    if isinstance(source, bool):
        return None
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        index = int(source)
        for backend in camera_backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    cap = cv2.VideoCapture(str(source))
    if cap.isOpened():
        return cap
    cap.release()
    return None


class BaseDetectionWorker(QThread):
    """线程基类：统一模型调用锁、阈值参数、停止标志与异常上报。"""

    failed = pyqtSignal(str)

    def __init__(self, model, lock: Optional[threading.Lock] = None, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._lock = lock or threading.Lock()
        self._colors = tools.Colors()
        self._conf = Config.DEFAULT_CONF_THRES
        self._iou = Config.DEFAULT_IOU_THRES
        self._show_labels = True
        self._running = True

    def configure_parameters(self, conf: float, iou: float, show_labels: bool = True) -> None:
        """可在线程运行过程中更新阈值，下一张图/下一帧即可生效。"""
        self._conf = conf
        self._iou = iou
        self._show_labels = show_labels

    def predict(self, image):
        """加锁调用 YOLO，返回 (单个结果, 推理耗时)。"""
        with self._lock:
            started = time.perf_counter()
            results = self._model.predict(image, conf=self._conf, iou=self._iou, verbose=False)
            elapsed = time.perf_counter() - started
        return results[0], elapsed

    def detect(self, image, source: str = ''):
        """推理 + 结果落地，返回 (DetectionResult, 推理耗时)。"""
        results, elapsed = self.predict(image)
        return tools.DetectionResult.from_ultralytics(results, source, elapsed), elapsed

    def annotate(self, image, detection: tools.DetectionResult, indices=None) -> np.ndarray:
        return tools.annotate_image(
            image,
            detection.boxes,
            detection.classes,
            detection.confs,
            colors=self._colors,
            show_labels=self._show_labels,
            indices=indices,
        )

    def stop(self) -> None:
        """请求线程尽快结束（下一次循环检查时退出）。"""
        self._running = False


class ImageDetectionWorker(BaseDetectionWorker):
    """单张或批量图片检测。

    ``persist=True`` 时同时把标注结果写入 ``Config.save_path`` 并汇总一份 CSV，
    用于“文件夹导出”，不必先检测一遍、保存时再检测一遍。
    """

    image_done = pyqtSignal(object, object, object, int, int)  # 结果 / 原图 / 标注图 / 当前序号 / 总数
    progress = pyqtSignal(int, int)
    finished_all = pyqtSignal(str)  # 任务结束时带上输出目录

    def __init__(self, model, lock=None, parent=None) -> None:
        super().__init__(model, lock, parent)
        self._paths: List[str] = []
        self._persist = False
        self._csv_rows: List[list] = []

    def configure(self,
                  paths: Sequence[str],
                  conf: float,
                  iou: float,
                  show_labels: bool = True,
                  persist: bool = False) -> None:
        self.configure_parameters(conf, iou, show_labels)
        self._paths = [str(path) for path in paths]
        self._persist = persist
        self._csv_rows = []
        self._running = True

    def run(self) -> None:
        total = len(self._paths)
        for index, path in enumerate(self._paths, start=1):
            if not self._running:
                LOGGER.info('图片检测已取消，剩余 %s/%s 张未处理', total - index + 1, total)
                break
            try:
                frame = tools.img_cvread(path)
                detection, _ = self.detect(frame, str(path))
                annotated = self.annotate(frame, detection)
            except Exception as exc:  # 单张失败不应中断整批任务
                LOGGER.exception('图片检测失败：%s', path)
                self.failed.emit(f'检测失败：{path}\n{exc}')
                continue

            if self._persist:
                self._save_result(annotated, detection)
            self.image_done.emit(detection, frame, annotated, index, total)
            self.progress.emit(index, total)

        if self._persist:
            csv_path = self._flush_csv()
            if csv_path:
                LOGGER.info('导出 CSV：%s', csv_path)
        self.finished_all.emit(Config.save_path)

    # ------------------------------------------------------------------ #
    def _save_result(self, annotated, detection: tools.DetectionResult) -> str:
        target = Path(Config.save_path) / tools.target_path_for(detection.source).name
        tools.img_cvwrite(target, annotated)
        self._csv_rows.extend(detection.csv_rows(start_index=len(self._csv_rows) + 1))
        return str(target)

    def _flush_csv(self) -> str:
        if not self._csv_rows:
            return ''
        target = Path(Config.save_path) / 'batch_detect_result.csv'
        return tools.write_csv(target, self._csv_rows)


class FrameDetectionWorker(BaseDetectionWorker):
    """逐帧推理线程：只负责推理与绘框，取帧由主线程投递。

    队列上限刻意设得很小：推理速度跟不上实时画面时自动丢帧，
    既能保持画面“跟得上”，也不会让待处理帧堆积导致内存上涨。
    """

    frame_done = pyqtSignal(object, object, object)  # DetectionResult / 标注帧 / 原帧
    QUEUE_SIZE = 2

    def __init__(self, model, lock=None, parent=None) -> None:
        super().__init__(model, lock, parent)
        self._frames: 'queue.Queue' = queue.Queue(self.QUEUE_SIZE)

    def configure(self, conf: float, iou: float, show_labels: bool = True) -> None:
        self.configure_parameters(conf, iou, show_labels)
        self._frames = queue.Queue(self.QUEUE_SIZE)
        self._running = True

    def submit(self, frame) -> bool:
        """投递一帧；队列满（说明推理还没跟上）时返回 False，调用方下一拍再投。"""
        if not self.isRunning():
            return False
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            return False
        return True

    def has_pending(self) -> bool:
        return self.isRunning() or not self._frames.empty()

    def run(self) -> None:
        while self._running or not self._frames.empty():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                detection, _ = self.detect(frame)
                annotated = self.annotate(frame, detection)
            except Exception as exc:  # 某一帧出错不必中断整段视频
                LOGGER.exception('实时检测失败')
                self.failed.emit(f'检测失败：{exc}')
                continue
            self.frame_done.emit(detection, annotated, frame)
