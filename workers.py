"""后台检测与导出任务。每个线程独占自己的模型实例。"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

import Config
from detection_core import (
    BatchRecord,
    DetectionFrame,
    DetectionItem,
    ValidationError,
    image_write,
    make_output_path,
    validate_image_file,
    validate_model_file,
    validate_video_file,
)


class DetectionWorker(QThread):
    error_signal = pyqtSignal(str)
    canceled_signal = pyqtSignal()

    def __init__(self, conf: float, iou: float, show_labels: bool = True, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._options_lock = threading.Lock()
        self.conf = conf
        self.iou = iou
        self.show_labels = show_labels

    def stop(self):
        self._stop_event.set()

    def update_options(self, conf: float, iou: float, show_labels: bool):
        with self._options_lock:
            self.conf = conf
            self.iou = iou
            self.show_labels = show_labels

    def _options(self):
        with self._options_lock:
            return self.conf, self.iou, self.show_labels

    def _load_model(self):
        model_path = validate_model_file(Config.model_path, Config.model_sha256)
        from ultralytics import YOLO

        model = YOLO(str(model_path), task="detect")
        actual_names = {int(key): value for key, value in model.names.items()}
        if actual_names != Config.names:
            raise ValidationError(
                f"模型类别与配置不一致。模型：{actual_names}；配置：{Config.names}"
            )
        return model

    def _detect(self, model, source: str, image) -> DetectionFrame:
        conf, iou, show_labels = self._options()
        started = time.perf_counter()
        result = model(image, conf=conf, iou=iou, verbose=False)[0]
        elapsed = time.perf_counter() - started
        class_ids = [int(value) for value in result.boxes.cls.tolist()]
        confidences = [float(value) for value in result.boxes.conf.tolist()]
        boxes = [tuple(map(int, value)) for value in result.boxes.xyxy.tolist()]
        items = []
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            if class_id not in Config.names:
                raise ValidationError(f"模型返回未知类别编号：{class_id}")
            items.append(
                DetectionItem(
                    class_id=class_id,
                    class_name=Config.names[class_id],
                    display_name=Config.CH_names[class_id],
                    confidence=confidence,
                    box=box,
                )
            )
        annotated = result.plot(labels=show_labels, conf=show_labels)
        return DetectionFrame(source, image, annotated, tuple(items), elapsed)


class ImageDetectionWorker(DetectionWorker):
    result_signal = pyqtSignal(object)
    completed_signal = pyqtSignal()

    def __init__(self, source: str, conf: float, iou: float, show_labels: bool, parent=None):
        super().__init__(conf, iou, show_labels, parent)
        self.source = source

    def run(self):
        try:
            image = validate_image_file(self.source)
            if self._stop_event.is_set():
                self.canceled_signal.emit()
                return
            model = self._load_model()
            if self._stop_event.is_set():
                self.canceled_signal.emit()
                return
            self.result_signal.emit(self._detect(model, self.source, image))
            self.completed_signal.emit()
        except Exception as error:
            self.error_signal.emit(str(error))


class BatchDetectionWorker(DetectionWorker):
    item_signal = pyqtSignal(object, int, int)
    item_error_signal = pyqtSignal(str, str, int, int)
    completed_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, sources: list[str], conf: float, iou: float, show_labels: bool, parent=None):
        super().__init__(conf, iou, show_labels, parent)
        self.sources = sources
        self.cache_directory = tempfile.mkdtemp(prefix="detection-system-batch-")

    def run(self):
        records = []
        try:
            model = self._load_model()
            total = len(self.sources)
            for index, source in enumerate(self.sources, 1):
                if self._stop_event.is_set():
                    shutil.rmtree(self.cache_directory, ignore_errors=True)
                    self.canceled_signal.emit()
                    return
                try:
                    image = validate_image_file(source)
                    frame = self._detect(model, source, image)
                    cache_path = (
                        Path(self.cache_directory) / f"{index:06d}{Path(source).suffix.lower()}"
                    )
                    image_write(cache_path, frame.annotated_image)
                    records.append(BatchRecord(source, str(cache_path), frame.items))
                    self.item_signal.emit(frame, index, total)
                except Exception as error:
                    self.item_error_signal.emit(source, str(error), index, total)
                self.progress_signal.emit(index, total)
            self.completed_signal.emit(records)
        except Exception as error:
            shutil.rmtree(self.cache_directory, ignore_errors=True)
            self.error_signal.emit(str(error))


class VideoDetectionWorker(DetectionWorker):
    frame_signal = pyqtSignal(object)
    completed_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, int)

    def __init__(self, source, conf: float, iou: float, show_labels: bool, parent=None):
        super().__init__(conf, iou, show_labels, parent)
        self.source = source

    def run(self):
        capture = None
        try:
            if isinstance(self.source, int):
                source = self.source
                fps = 25.0
                total = 0
            else:
                info = validate_video_file(self.source)
                source = info.path
                fps = info.fps
                total = info.frame_count
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise ValidationError("无法打开视频源。")
            model = self._load_model()
            frame_interval = 1.0 / max(fps, 1.0)
            current = 0
            while not self._stop_event.is_set():
                started = time.perf_counter()
                success, image = capture.read()
                if not success:
                    break
                current += 1
                frame = self._detect(model, str(self.source), image)
                self.frame_signal.emit(frame)
                self.progress_signal.emit(current, total)
                remaining = frame_interval - (time.perf_counter() - started)
                if remaining > 0:
                    self.msleep(int(remaining * 1000))
            if self._stop_event.is_set():
                self.canceled_signal.emit()
            else:
                self.completed_signal.emit()
        except Exception as error:
            self.error_signal.emit(str(error))
        finally:
            if capture is not None:
                capture.release()


class VideoExportWorker(DetectionWorker):
    progress_signal = pyqtSignal(int, int)
    completed_signal = pyqtSignal(str)

    def __init__(
        self,
        source: str,
        output_directory: str,
        conf: float,
        iou: float,
        show_labels: bool,
        parent=None,
    ):
        super().__init__(conf, iou, show_labels, parent)
        self.source = source
        self.output_directory = output_directory

    def run(self):
        capture = None
        writer = None
        temporary_path = None
        final_path = None
        processed = 0
        error_message = None
        try:
            info = validate_video_file(self.source)
            model = self._load_model()
            final_path = make_output_path(info.path, self.output_directory, extension=".avi")
            temporary_path = final_path.with_name(
                f".{final_path.stem}.{uuid.uuid4().hex}.partial.avi"
            )
            capture = cv2.VideoCapture(info.path)
            writer = cv2.VideoWriter(
                str(temporary_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                info.fps,
                (info.width, info.height),
            )
            if not capture.isOpened() or not writer.isOpened():
                raise OSError("无法创建视频输出，请检查编码器和输出目录权限。")

            while not self._stop_event.is_set():
                success, image = capture.read()
                if not success:
                    break
                frame = self._detect(model, info.path, image)
                writer.write(frame.annotated_image)
                processed += 1
                self.progress_signal.emit(processed, info.frame_count)
        except Exception as error:
            error_message = str(error)
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()

        if self._stop_event.is_set():
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.canceled_signal.emit()
            return
        if error_message:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.error_signal.emit(error_message)
            return
        try:
            if temporary_path is None or final_path is None or processed == 0:
                raise OSError("没有生成可用的视频帧。")
            verification = cv2.VideoCapture(str(temporary_path))
            try:
                if not verification.isOpened():
                    raise OSError("导出视频无法重新打开验证。")
                written_frames = int(verification.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(verification.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(verification.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                verification.release()
            if written_frames <= 0 or width != info.width or height != info.height:
                raise OSError("导出视频完整性验证失败。")
            os.replace(temporary_path, final_path)
            self.completed_signal.emit(str(final_path))
        except Exception as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self.error_signal.emit(str(error))
