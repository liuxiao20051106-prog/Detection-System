# -*- coding: utf-8 -*-
"""葡萄成熟度检测系统桌面端入口。

线程分工（这是本次重构最核心的一点）：

- **界面线程**：取视频帧、渲染画面、写结果文件；
- **workers.py 的后台线程**：YOLO 推理与绘框。

之所以不让工作线程去碰 ``cv2.VideoCapture``：OpenCV 的视频后端（Windows 的
MSMF / DirectShow）对“在哪个线程里初始化”很敏感，放到 QThread 里容易出现卡死
或设备打不开；而取帧本身开销很小，留在界面线程最稳，只需要 QTimer 驱动即可。
真正耗时的推理仍然全部在后台，所以窗口不会卡。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
from PyQt5.QtCore import QCoreApplication, Qt, QTimer
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QFileDialog,
                             QHeaderView, QMainWindow, QMessageBox,
                             QTableWidgetItem)
from ultralytics import YOLO

import Config
import detect_tools as tools
from UIProgram.QssLoader import QSSLoader
from UIProgram.UiMain import Ui_MainWindow
from UIProgram.precess_bar import ProgressBar
from UIProgram import ui_sources_rc  # noqa: F401 - 注册 Qt 资源
from workers import (ImageDetectionWorker, FrameDetectionWorker,
                     open_video_capture)

LOGGER = logging.getLogger(__name__)

ALL_TARGETS = '全部'
CAMERA_FRAME_MS = 33  # 摄像头画面约 30 FPS，与界面刷新节奏一致


class MainWindow(QMainWindow, Ui_MainWindow):
    """主窗口：负责界面状态与信号编排，不直接做推理。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        for hidden_label in ('label_2', 'label_12'):
            widget = getattr(self, hidden_label, None)
            if widget is not None:
                widget.hide()

        self.init_state()
        self.init_table()
        self.init_workers()
        self.init_ui()
        self.signalconnect()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def init_state(self) -> None:
        """初始化业务状态。"""
        self.show_width = 770   # 与 UI 里 label_show 的显示区域保持一致
        self.show_height = 480

        self.org_path: Optional[str] = None
        self.org_img = None            # 最近一次检测的原图（缓存用于重绘）
        self.draw_img = None           # 最近一次检测带框的图（用于保存）
        self.last_result = tools.DetectionResult()

        self.conf_thres = Config.DEFAULT_CONF_THRES
        self.iou_thres = Config.DEFAULT_IOU_THRES
        self.show_labels = True

        self._batch_mode = False
        self._exporting = False        # 图片/文件夹导出任务进行中
        self._streaming = False        # 摄像头/视频画面进行中
        self._camera_open = False
        self._pending_frame = None
        self._combo_signature: Optional[tuple] = None
        self._last_table_update = 0.0
        self.progress_bar: Optional[ProgressBar] = None

        os.makedirs(Config.save_path, exist_ok=True)
        if not os.path.isfile(Config.model_path):
            raise FileNotFoundError(
                f'未找到模型文件：{Config.model_path}\n'
                '请将模型放到 models/best.pt，或设置环境变量 DETECTION_MODEL_PATH。'
            )

        # GPU 由 ultralytics 自动选择，因此界面进程不再依赖 torch。
        self.model = YOLO(Config.model_path, task='detect')
        self.model_lock = threading.Lock()

        # 视频/导出流水线：主线程取帧 -> 投给工作线程
        self.cap = None
        self.writer = None
        self.export_path = ''
        self.export_total = 0
        self.export_current = 0

        # 帧渲染节流：高帧率时用定时器抽帧显示，中间帧自动丢弃。
        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(Config.FRAME_RENDER_INTERVAL_MS)
        self.frame_timer.timeout.connect(self.flush_pending_frame)

        # 视频取帧节拍
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_tick)

        # 导出收尾轮询：视频帧取完后要等工作线程把队列排空
        self.drain_timer = QTimer(self)
        self.drain_timer.setInterval(100)
        self.drain_timer.timeout.connect(self.check_export_finished)

        # 阈值连续拖动时去抖，避免每一格都触发一次推理
        self.redetect_timer = QTimer(self)
        self.redetect_timer.setSingleShot(True)
        self.redetect_timer.setInterval(Config.REDETECT_DEBOUNCE_MS)
        self.redetect_timer.timeout.connect(self.redetect_current_image)

    def init_table(self) -> None:
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tableWidget.verticalHeader().setDefaultSectionSize(40)
        for column, width in enumerate((80, 200, 150, 90, 230)):
            self.tableWidget.setColumnWidth(column, width)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.setAlternatingRowColors(True)

    def init_workers(self) -> None:
        """所有推理线程集中创建并连接信号，界面只消费结果。"""
        self.image_worker = ImageDetectionWorker(self.model, self.model_lock, self)
        self.image_worker.image_done.connect(self.on_image_detected)
        self.image_worker.progress.connect(self.on_progress)
        self.image_worker.finished_all.connect(self.on_image_task_finished)
        self.image_worker.failed.connect(self.on_worker_failed)

        self.frame_worker = FrameDetectionWorker(self.model, self.model_lock, self)
        self.frame_worker.frame_done.connect(self.on_frame_result)
        self.frame_worker.failed.connect(self.on_worker_failed)

    def init_ui(self) -> None:
        """样式、阈值控件初始状态。"""
        style_file = os.path.join(str(Config.PROJECT_ROOT), 'UIProgram', 'style.css')
        self.setStyleSheet(QSSLoader.read_qss_file(style_file))

        for box, default in ((self.doubleSpinBox, self.conf_thres), (self.doubleSpinBox_2, self.iou_thres)):
            box.setRange(0.0, 1.0)
            box.setSingleStep(Config.THRES_STEP)
            box.setValue(default)
        self.checkBox.setChecked(self.show_labels)

    def signalconnect(self) -> None:
        self.PicBtn.clicked.connect(self.open_img)
        self.FilesBtn.clicked.connect(self.detect_batch_imgs)
        self.VideoBtn.clicked.connect(self.vedio_show)
        self.CapBtn.clicked.connect(self.camera_show)
        self.SaveBtn.clicked.connect(self.save_result)
        self.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.comboBox.activated.connect(self.combox_change)

        self.doubleSpinBox.valueChanged.connect(self.update_conf_thres)
        self.doubleSpinBox_2.valueChanged.connect(self.update_iou_thres)
        self.checkBox.stateChanged.connect(self.update_show_labels)

    # ------------------------------------------------------------------ #
    # 图片 / 文件夹检测
    # ------------------------------------------------------------------ #
    def open_img(self) -> None:
        """选择单张图片并检测。"""
        self.stop_stream()
        path, _ = QFileDialog.getOpenFileName(self, '打开图片', str(Config.PROJECT_ROOT), Config.IMAGE_FILTER)
        if not path:
            return
        self.prepare_single_source(path)
        self.detect_images([path], batch=False)

    def detect_batch_imgs(self) -> None:
        """选择文件夹批量检测。"""
        self.stop_stream()
        directory = QFileDialog.getExistingDirectory(self, '选取文件夹', str(Config.PROJECT_ROOT))
        if not directory:
            return
        images = tools.list_image_files(directory)
        if not images:
            QMessageBox.information(self, '提示', '所选文件夹中没有支持的图片。')
            return

        self.org_path = directory
        self._batch_mode = True
        self.reset_table()
        self.PiclineEdit.setText(directory)
        LOGGER.info('开始批量检测：%s 张图片', len(images))
        self.run_image_worker(images, batch=True, persist=False)

    def prepare_single_source(self, path: str) -> None:
        """切到单张图片模式前重置相关状态。"""
        self._batch_mode = False
        self.org_path = path
        self.PiclineEdit.setText(path)
        self.reset_table()
        self._combo_signature = None
        self.comboBox.setDisabled(False)

    def run_image_worker(self, paths: Sequence[str], batch: bool, persist: bool) -> None:
        """启动图片检测线程；同时只允许一个任务在跑。"""
        if self.image_worker.isRunning():
            QMessageBox.information(self, '提示', '正在处理上一次任务，请稍候。')
            return
        self._batch_mode = batch
        self._exporting = persist
        if persist or len(paths) > 5:
            self.show_progress(self.image_worker)
        self.image_worker.configure(
            [str(path) for path in paths],
            self.conf_thres,
            self.iou_thres,
            self.show_labels,
            persist=persist,
        )
        self.image_worker.start()

    def detect_images(self, paths: Sequence[str], batch: bool) -> None:
        self.run_image_worker(paths, batch=batch, persist=False)

    def on_image_detected(self, detection, frame, annotated, index: int, total: int) -> None:
        """图片/批量模式的结果回调（后台线程 -> 界面线程）。"""
        if self._exporting:  # 导出任务只更新进度，不刷新画面
            return

        self.last_result = detection
        self.org_img = frame
        self.draw_img = annotated
        self.org_path = detection.source

        if not self._batch_mode:
            self.reset_table()
            self.PiclineEdit.setText(detection.source)

        self.show_inference_time(detection)
        self.label_nums.setText(str(len(detection)))
        self.update_target_panel(detection, 0)
        self.update_combo_box(detection)
        self.queue_frame(frame, annotated, detection)
        self.append_table(detection, replace=not self._batch_mode)

    def on_image_task_finished(self, save_dir: str) -> None:
        """图片/批量任务收尾：关进度条，导出任务再弹提示。"""
        exporting = self._exporting
        self._exporting = False
        self.close_progress()
        if exporting:
            QMessageBox.about(self, '提示', f'图片保存成功！\n文件路径：{save_dir}')

    # ------------------------------------------------------------------ #
    # 视频 / 摄像头：主线程取帧，工作线程推理
    # ------------------------------------------------------------------ #
    def vedio_show(self) -> None:
        """打开视频文件并逐帧检测。"""
        self.stop_stream()
        path, _ = QFileDialog.getOpenFileName(self, '打开视频', str(Config.PROJECT_ROOT), Config.VIDEO_FILTER)
        if not path:
            return
        self.org_path = path
        self.VideolineEdit.setText(path)
        self.start_stream(path)

    def camera_show(self) -> None:
        """开关摄像头。"""
        if self._camera_open:
            self._camera_open = False
            self.CaplineEdit.setText('摄像头未开启')
            self.stop_stream()
            self.label_show.clear()
            self.label_show.setText('')
            return

        self.stop_stream()
        if self.start_stream(0):
            self._camera_open = True
            self.CaplineEdit.setText('摄像头开启')
        else:
            self.CaplineEdit.setText('摄像头未开启')

    def start_stream(self, source) -> bool:
        """启动实时画面；返回是否成功打开视频源。"""
        cap = open_video_capture(source)
        if cap is None:
            QMessageBox.warning(self, '打开失败', f'无法打开视频源：{source}\n请检查文件是否损坏或设备是否被占用。')
            return False

        self.cap = cap
        self.org_path = str(source)
        self.org_img = None
        self._combo_signature = None
        self._exporting = False
        self._streaming = True
        self.reset_table()
        self.comboBox.setDisabled(True)
        self.frame_worker.configure(self.conf_thres, self.iou_thres, self.show_labels)
        self.frame_worker.start()
        self.capture_timer.start(self.frame_interval(source))
        return True

    def frame_interval(self, source) -> int:
        """按视频自身帧率取帧；摄像头统一按 30 FPS。"""
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return CAMERA_FRAME_MS
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 0
        if fps and 0 < fps <= 120:
            return max(5, int(1000 / fps))
        return CAMERA_FRAME_MS

    def capture_tick(self) -> None:
        """取一帧交给工作线程；队列满时本帧丢弃，下一拍继续。"""
        if self.cap is None or not self.cap.isOpened():
            self.finish_stream()
            return
        ok, frame = self.cap.read()
        if not ok:
            self.finish_stream()
            return
        self.frame_worker.submit(frame)

    def on_frame_result(self, detection, annotated, frame) -> None:
        """实时/导出模式下每帧结果的回调。"""
        self.last_result = detection
        self.draw_img = annotated
        self.org_img = frame

        if self._exporting:
            self.write_export_frame(annotated)
            return

        self.show_inference_time(detection)
        self.label_nums.setText(str(len(detection)))
        self.update_target_panel(detection, 0)
        self.update_combo_box(detection)
        self.queue_frame(frame, annotated, detection)

    def finish_stream(self) -> None:
        """视频取完最后一帧。"""
        self.capture_timer.stop()
        if self._exporting:
            self.frame_worker.stop()  # 停止投递，等待队列排空
            self.drain_timer.start()
            return
        self.stop_stream()

    def stop_stream(self) -> None:
        """停止实时画面并释放采集/写盘资源。"""
        self.capture_timer.stop()
        self.drain_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.frame_worker.isRunning():
            self.frame_worker.stop()
            self.frame_worker.wait(3000)
        self.release_writer()

        self._streaming = False
        self._camera_open = False
        self._pending_frame = None
        self._combo_signature = None
        self.frame_timer.stop()
        self.comboBox.setDisabled(False)

    # ------------------------------------------------------------------ #
    # 画面渲染
    # ------------------------------------------------------------------ #
    def queue_frame(self, frame, annotated, detection) -> None:
        """登记待渲染帧，真正绘制在定时器里按 30 FPS 节流完成。"""
        self._pending_frame = (annotated, detection, frame)
        if not self.frame_timer.isActive():
            self.frame_timer.start()

    def flush_pending_frame(self) -> None:
        if self._pending_frame is None:
            self.frame_timer.stop()
            return
        annotated, detection, frame = self._pending_frame
        self._pending_frame = None
        self.render_image(annotated)

        if self._streaming:
            now = time.monotonic()
            if now - self._last_table_update >= Config.STREAM_TABLE_INTERVAL_MS / 1000:
                self._last_table_update = now
                self.append_table(detection, replace=True)
        self.frame_timer.stop()

    def render_image(self, image) -> None:
        self.label_show.setPixmap(tools.cvimg_to_qpiximg(tools.resize_to_fit(image, self.show_width, self.show_height)))
        self.label_show.setAlignment(Qt.AlignCenter)

    def show_inference_time(self, detection: tools.DetectionResult) -> None:
        duration = detection.duration
        self.time_lb.setText(f'{duration * 1000:.0f} ms' if duration < 1 else f'{duration:.3f} s')

    # ------------------------------------------------------------------ #
    # 结果面板
    # ------------------------------------------------------------------ #
    def update_target_panel(self, detection: tools.DetectionResult, index: Optional[int]) -> None:
        """刷新右侧类别、置信度和坐标；index 为 None 表示清空。"""
        if index is None or len(detection) == 0:
            self.type_lb.setText('')
            self.label_conf.setText('')
            for label in (self.label_xmin, self.label_ymin, self.label_xmax, self.label_ymax):
                label.setText('')
            return
        index = max(0, min(index, len(detection) - 1))
        x1, y1, x2, y2 = detection.boxes[index]
        self.type_lb.setText(detection.label(index))
        self.label_conf.setText(detection.conf_text(index))
        self.label_xmin.setText(str(x1))
        self.label_ymin.setText(str(y1))
        self.label_xmax.setText(str(x2))
        self.label_ymax.setText(str(y2))

    def update_combo_box(self, detection: tools.DetectionResult) -> None:
        """仅在目标构成变化时重建下拉框（每帧重建会明显掉帧）。"""
        signature = tuple(detection.classes)
        if signature == self._combo_signature:
            return
        self._combo_signature = signature

        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        self.comboBox.addItems([ALL_TARGETS] + detection.option_texts())
        self.comboBox.setCurrentIndex(0)
        self.comboBox.blockSignals(False)

    def combox_change(self) -> None:
        """切换“全部 / 单个目标”时重绘画面。"""
        if self.org_img is None or len(self.last_result) == 0:
            return
        text = self.comboBox.currentText()
        target_index = 0
        indices = None
        if text and text != ALL_TARGETS:
            try:
                target_index = int(text.rsplit('_', 1)[-1])
            except ValueError:
                return
            if not 0 <= target_index < len(self.last_result.boxes):
                return
            indices = [target_index]

        self.draw_img = tools.annotate_image(
            self.org_img,
            self.last_result.boxes,
            self.last_result.classes,
            self.last_result.confs,
            show_labels=self.show_labels,
            indices=indices,
        )
        self.render_image(self.draw_img)
        self.update_target_panel(self.last_result, target_index)

    # ------------------------------------------------------------------ #
    # 表格
    # ------------------------------------------------------------------ #
    def reset_table(self) -> None:
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()

    def append_table(self, detection: tools.DetectionResult, replace: bool = False) -> None:
        """写入结果行；replace=True 时先清空旧行。"""
        if replace:
            self.reset_table()
        count = len(detection)
        if count == 0:
            return

        start = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(start + count)
        for offset in range(count):
            self.set_table_row(start + offset, start + offset + 1, detection, offset)
        self.trim_table()
        self.tableWidget.scrollToBottom()

    def set_table_row(self, row: int, serial: int, detection: tools.DetectionResult, index: int) -> None:
        values = (
            str(serial),
            detection.source,
            detection.label(index),
            detection.conf_text(index),
            str(detection.boxes[index]),
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column in (0, 2, 3):
                item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row, column, item)

    def trim_table(self) -> None:
        for _ in range(max(0, self.tableWidget.rowCount() - Config.MAX_TABLE_ROWS)):
            self.tableWidget.removeRow(0)

    # ------------------------------------------------------------------ #
    # 保存 / 导出
    # ------------------------------------------------------------------ #
    def save_result(self) -> None:
        """根据当前数据源导出图片、文件夹结果或视频。"""
        if self.image_worker.isRunning() or self._exporting:
            QMessageBox.information(self, '提示', '正在处理上一次任务，请稍候。')
            return
        if self._camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存！')
            return

        source = self.org_path
        if not source:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if os.path.isdir(source):
            self.export_batch_images(source)
        elif os.path.isfile(source):
            if source.lower().endswith(tuple(Config.VIDEO_EXTENSIONS)):
                self.export_video(source)
            else:
                self.export_single_image(source)
        else:
            QMessageBox.about(self, '提示', '请先打开有效的图片、视频或文件夹。')

    def export_single_image(self, source: str) -> None:
        if self.draw_img is None:
            QMessageBox.about(self, '提示', '请先检测一张图片再保存。')
            return
        saved = tools.img_cvwrite(Path(Config.save_path) / tools.target_path_for(source).name, self.draw_img)
        csv_path = tools.write_csv(
            Path(Config.save_path) / f'{Path(source).stem}_detect_result.csv',
            self.last_result.csv_rows(),
        )
        QMessageBox.about(self, '提示', f'图片保存成功！\n图片：{saved}\n数据：{csv_path}')

    def export_batch_images(self, directory: str) -> None:
        images = tools.list_image_files(directory)
        if not images:
            QMessageBox.information(self, '提示', '所选文件夹中没有支持的图片。')
            return
        answer = QMessageBox.question(
            self, '提示',
            f'共 {len(images)} 张图片，导出检测后的图片与 CSV 可能需要较长时间，是否继续？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        self.run_image_worker(images, batch=False, persist=True)

    def export_video(self, source: str) -> None:
        answer = QMessageBox.question(
            self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        cap = open_video_capture(source)
        if cap is None:
            QMessageBox.critical(self, '打开失败', f'无法读取原视频：{source}')
            return

        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.export_path = str(
            Path(Config.save_path) / tools.target_path_for(source).with_suffix('.avi').name
        )
        writer = cv2.VideoWriter(self.export_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        if not writer.isOpened():
            cap.release()
            QMessageBox.critical(self, '保存失败', f'无法创建输出视频：{self.export_path}')
            return

        self.cap = cap
        self.writer = writer
        self.export_total = total
        self.export_current = 0
        self._exporting = True
        self._streaming = True  # 复用取帧流水线，但不做画面刷新
        self._cancel_requested = False

        self.show_progress(self.cancel_export)
        self.frame_worker.configure(self.conf_thres, self.iou_thres, self.show_labels)
        self.frame_worker.start()
        self.capture_timer.start(0)  # 间隔 0：事件队列空闲即触发，等价于“尽快处理”

    def cancel_export(self) -> None:
        """点击“取消保存”：停止取帧与推理，已写入的部分保留。"""
        self._cancel_requested = True
        self.capture_timer.stop()
        self.frame_worker.stop()

    def write_export_frame(self, annotated) -> None:
        if self.writer is None or self._cancel_requested:
            return
        self.writer.write(annotated)
        self.export_current += 1
        self.on_progress(self.export_current, self.export_total)

    def check_export_finished(self) -> None:
        """等待最后一帧推理完成，然后收尾。"""
        if self.frame_worker.has_pending():
            return
        self.drain_timer.stop()
        self.release_writer()
        if self.frame_worker.isRunning():
            self.frame_worker.wait(3000)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        saved, canceled = self.export_path, self._cancel_requested
        finished = self.export_current >= self.export_total and not canceled
        self.export_path = ''
        self.export_total = 0
        self.export_current = 0
        self._exporting = False
        self._streaming = False
        self._cancel_requested = False
        self.close_progress()

        if canceled:
            QMessageBox.information(self, '提示', f'已取消视频保存，已写入的部分在：\n{saved}')
        elif finished:
            QMessageBox.about(self, '提示', f'视频保存成功！\n文件路径：{saved}')
        else:
            QMessageBox.warning(self, '提示', f'视频未处理完（{self.export_total} 帧中有部分失败）：\n{saved}')

    def release_writer(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    # ------------------------------------------------------------------ #
    # 进度条 / 线程回调
    # ------------------------------------------------------------------ #
    def show_progress(self, on_cancel) -> None:
        """显示进度条；on_cancel 在点“取消”时调用（用于停止后台任务）。"""
        if self.progress_bar is None:
            self.progress_bar = ProgressBar(self, on_cancel=on_cancel)
        self.progress_bar.reset()
        self.progress_bar.show()

    def on_progress(self, current: int, total: int) -> None:
        if self.progress_bar is None or not self.progress_bar.isVisible():
            return
        self.progress_bar.setValue(current, total, int(current / max(total, 1) * 100))

    def on_worker_failed(self, message: str) -> None:
        LOGGER.error('后台任务出错：%s', message)
        self.close_progress()
        if self._streaming:
            self.stop_stream()
        self._exporting = False
        QMessageBox.critical(self, '任务失败', message)

    def close_progress(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    # ------------------------------------------------------------------ #
    # 参数调整
    # ------------------------------------------------------------------ #
    def update_conf_thres(self, value: float) -> None:
        self.conf_thres = value
        self.sync_worker_parameters()
        self.redetect_timer.start()

    def update_iou_thres(self, value: float) -> None:
        self.iou_thres = value
        self.sync_worker_parameters()
        self.redetect_timer.start()

    def update_show_labels(self, state) -> None:
        self.show_labels = state == Qt.Checked
        self.sync_worker_parameters()
        if self._streaming:  # 实时画面的下一帧自动生效
            return
        if self.org_img is None or len(self.last_result) == 0:
            return
        self.draw_img = tools.annotate_image(
            self.org_img,
            self.last_result.boxes,
            self.last_result.classes,
            self.last_result.confs,
            show_labels=self.show_labels,
        )
        self.render_image(self.draw_img)
        self.comboBox.blockSignals(True)
        self.comboBox.setCurrentIndex(0)
        self.comboBox.blockSignals(False)
        self.update_target_panel(self.last_result, 0)

    def sync_worker_parameters(self) -> None:
        for worker in (self.image_worker, self.frame_worker):
            worker.configure_parameters(self.conf_thres, self.iou_thres, self.show_labels)

    def redetect_current_image(self) -> None:
        """阈值变化后对当前图片重新推理（去重抖后调用）。"""
        if self._streaming or self._exporting or self.image_worker.isRunning():
            return
        source = self.last_result.source
        if source and os.path.isfile(source):
            self.detect_images([source], batch=False)

    # ------------------------------------------------------------------ #
    # 退出
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        """退出前释放摄像头、视频对象和后台线程。"""
        self.export_path = ''
        self.stop_stream()
        if self.image_worker.isRunning():
            self.image_worker.stop()
            self.image_worker.wait(3000)
        event.accept()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
    except FileNotFoundError as exc:
        QMessageBox.critical(None, '启动失败', str(exc))
        return 1
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
