# -*- coding: utf-8 -*-
"""葡萄成熟度检测系统桌面入口。"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401 - Windows 上需在 PyQt5 前初始化 Torch DLL
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWIDGETSIZE_MAX,
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTableWidgetItem,
)

import Config
import detect_tools as tools
from detection_core import (
    MAX_TABLE_ROWS,
    DetectionFrame,
    frame_csv_rows,
    image_write,
    list_image_files,
    make_output_path,
    validate_model_file,
    write_detection_csv,
)
from UIProgram import ui_sources_rc  # noqa: F401 - 注册 Qt 资源
from UIProgram.precess_bar import ProgressBar
from UIProgram.QssLoader import QSSLoader
from UIProgram.UiMain import Ui_MainWindow
from workers import (
    BatchDetectionWorker,
    ImageDetectionWorker,
    VideoDetectionWorker,
    VideoExportWorker,
)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._configure_scrollable_window()
        self._initialize_state()
        self._configure_controls()
        self._connect_signals()
        self._load_styles()
        self._validate_runtime()

    def _configure_scrollable_window(self):
        content = self.takeCentralWidget()
        content.setFixedSize(1250, 780)
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(False)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.setCentralWidget(scroll_area)
        self.setMinimumSize(900, 650)
        self.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        self.resize(1250, 830)

    def _initialize_state(self):
        self.show_width = 770
        self.show_height = 480
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.show_labels = True
        self.mode = None
        self.current_source = None
        self.current_frame = None
        self.batch_records = []
        self.batch_errors = []
        self.batch_cache_directory = None
        self.active_worker = None
        self.preview_worker = None
        self.export_worker = None
        self.progress_bar = None
        self._row_counter = 0
        self._closing = False
        self._redetect_timer = QTimer(self)
        self._redetect_timer.setSingleShot(True)
        self._redetect_timer.setInterval(300)
        self._redetect_timer.timeout.connect(self._redetect_current_image)

    def _configure_controls(self):
        for widget, value in (
            (self.doubleSpinBox, self.conf_thres),
            (self.doubleSpinBox_2, self.iou_thres),
        ):
            widget.blockSignals(True)
            widget.setRange(0.0, 1.0)
            widget.setSingleStep(0.05)
            widget.setValue(value)
            widget.blockSignals(False)
        self.checkBox.blockSignals(True)
        self.checkBox.setChecked(self.show_labels)
        self.checkBox.blockSignals(False)
        self.comboBox.setDisabled(True)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tableWidget.verticalHeader().setDefaultSectionSize(40)
        for column, width in enumerate((80, 200, 150, 90, 230)):
            self.tableWidget.setColumnWidth(column, width)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setAlternatingRowColors(True)
        for name in ("label_2", "label_12"):
            if hasattr(self, name):
                getattr(self, name).hide()

    def _connect_signals(self):
        self.PicBtn.clicked.connect(self.open_img)
        self.FilesBtn.clicked.connect(self.detect_batch_images)
        self.VideoBtn.clicked.connect(self.video_show)
        self.CapBtn.clicked.connect(self.camera_show)
        self.SaveBtn.clicked.connect(self.save_results)
        self.ExitBtn.clicked.connect(self.close)
        self.comboBox.activated.connect(self.combox_change)
        self.doubleSpinBox.valueChanged.connect(self.update_conf_thres)
        self.doubleSpinBox_2.valueChanged.connect(self.update_iou_thres)
        self.checkBox.stateChanged.connect(self.update_show_labels)

    def _load_styles(self):
        style_file = Config.PROJECT_ROOT / "UIProgram" / "style.css"
        self.setStyleSheet(QSSLoader.read_qss_file(str(style_file)))

    def _validate_runtime(self):
        Path(Config.save_path).mkdir(parents=True, exist_ok=True)
        model_path = validate_model_file(Config.model_path, Config.model_sha256)
        self.statusBar().showMessage(f"模型校验通过：{model_path.name}")

    def _set_detection_busy(self, busy: bool):
        for widget in (
            self.PicBtn,
            self.FilesBtn,
            self.VideoBtn,
            self.CapBtn,
            self.SaveBtn,
            self.doubleSpinBox,
            self.doubleSpinBox_2,
        ):
            widget.setDisabled(busy)

    def _set_export_busy(self, busy: bool):
        for widget in (
            self.PicBtn,
            self.FilesBtn,
            self.VideoBtn,
            self.CapBtn,
            self.SaveBtn,
            self.doubleSpinBox,
            self.doubleSpinBox_2,
            self.checkBox,
        ):
            widget.setDisabled(busy)

    def _stop_worker(self, worker, timeout=15000):
        if worker is None or not worker.isRunning():
            return True
        worker.stop()
        return worker.wait(timeout)

    def _stop_active_worker(self):
        worker = self.active_worker
        if worker is None:
            return True
        if not self._stop_worker(worker):
            QMessageBox.warning(self, "正在停止", "后台任务尚未安全结束，请稍后重试。")
            return False
        return True

    def _stop_preview(self):
        worker = self.preview_worker
        if worker is None:
            return True
        if not self._stop_worker(worker):
            QMessageBox.warning(self, "正在停止", "视频任务尚未安全结束，请稍后重试。")
            return False
        return True

    def _prepare_new_source(self):
        if self.export_worker is not None and self.export_worker.isRunning():
            QMessageBox.information(self, "任务进行中", "请先完成或取消视频导出。")
            return False
        if not self._stop_active_worker() or not self._stop_preview():
            return False
        self._clear_batch_cache()
        return True

    def _clear_batch_cache(self):
        directories = {str(Path(record.cached_image).parent) for record in self.batch_records}
        if self.batch_cache_directory:
            directories.add(self.batch_cache_directory)
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)
        self.batch_records = []
        self.batch_errors = []
        self.batch_cache_directory = None

    def open_img(self):
        if not self._prepare_new_source():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)",
        )
        if file_path:
            self.mode = "image"
            self.current_source = file_path
            self.PiclineEdit.setText(file_path)
            self._start_image_detection(file_path)

    def _start_image_detection(self, source):
        self._stop_active_worker()
        worker = ImageDetectionWorker(
            source,
            self.conf_thres,
            self.iou_thres,
            self.show_labels,
            self,
        )
        self.active_worker = worker
        worker.result_signal.connect(self._show_single_frame)
        worker.error_signal.connect(self._worker_failed)
        worker.canceled_signal.connect(lambda: self.statusBar().showMessage("检测已取消"))
        worker.finished.connect(lambda: self._active_worker_finished(worker))
        self._set_detection_busy(True)
        self.statusBar().showMessage("正在检测图片…")
        worker.start()

    def detect_batch_images(self):
        if not self._prepare_new_source():
            return
        directory = QFileDialog.getExistingDirectory(self, "选取图片文件夹", "")
        if not directory:
            return
        try:
            image_files = list_image_files(directory)
        except Exception as error:
            self._worker_failed(str(error))
            return
        if not image_files:
            QMessageBox.information(self, "提示", "所选文件夹中没有支持的图片。")
            return
        self.mode = "batch"
        self.current_source = directory
        self.batch_errors = []
        self._reset_table()
        worker = BatchDetectionWorker(
            image_files,
            self.conf_thres,
            self.iou_thres,
            self.show_labels,
            self,
        )
        self.active_worker = worker
        self.batch_cache_directory = worker.cache_directory
        worker.item_signal.connect(self._show_batch_frame)
        worker.item_error_signal.connect(self._batch_item_failed)
        worker.progress_signal.connect(self._batch_progress)
        worker.completed_signal.connect(self._batch_completed)
        worker.error_signal.connect(self._worker_failed)
        worker.canceled_signal.connect(lambda: self.statusBar().showMessage("批量检测已取消"))
        worker.finished.connect(lambda: self._active_worker_finished(worker))
        self._set_detection_busy(True)
        self.statusBar().showMessage(f"正在检测 {len(image_files)} 张图片…")
        worker.start()

    def _show_single_frame(self, frame: DetectionFrame):
        self.current_frame = frame
        self._display_frame(frame, reset_table=True)
        self.statusBar().showMessage("图片检测完成")

    def _show_batch_frame(self, frame: DetectionFrame, current: int, total: int):
        self.current_frame = frame
        self._display_frame(frame, reset_table=False)
        self.statusBar().showMessage(f"批量检测：{current}/{total}")

    def _batch_item_failed(self, source: str, message: str, current: int, total: int):
        self.batch_errors.append((source, message))
        self.statusBar().showMessage(f"跳过损坏文件 {current}/{total}：{Path(source).name}")

    def _batch_progress(self, current: int, total: int):
        if current == total:
            self.statusBar().showMessage(f"批量检测完成：{total} 个文件")

    def _batch_completed(self, records):
        self.batch_records = list(records)
        message = f"成功 {len(self.batch_records)} 个"
        if self.batch_errors:
            message += f"，跳过 {len(self.batch_errors)} 个"
        self.statusBar().showMessage(f"批量检测完成：{message}")

    def video_show(self):
        if not self._prepare_new_source():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开视频",
            "",
            "视频文件 (*.avi *.mp4 *.wmv *.mkv *.mov *.m4v)",
        )
        if not file_path:
            return
        self.mode = "video"
        self.current_source = file_path
        self.VideolineEdit.setText(file_path)
        self._start_video_preview(file_path)

    def camera_show(self):
        if self.mode == "camera" and self.preview_worker is not None:
            if not self._stop_preview():
                return
            self.mode = None
            self.CaplineEdit.setText("摄像头未开启")
            self.statusBar().showMessage("摄像头已关闭")
            return
        if not self._prepare_new_source():
            return
        self.mode = "camera"
        self.current_source = Config.camera_id
        self.CaplineEdit.setText(f"摄像头 {Config.camera_id} 开启")
        self._start_video_preview(Config.camera_id)

    def _start_video_preview(self, source):
        self._reset_table()
        self.comboBox.clear()
        self.comboBox.setDisabled(True)
        worker = VideoDetectionWorker(
            source,
            self.conf_thres,
            self.iou_thres,
            self.show_labels,
            self,
        )
        self.preview_worker = worker
        worker.frame_signal.connect(self._show_video_frame)
        worker.completed_signal.connect(self._video_finished)
        worker.canceled_signal.connect(lambda: self.statusBar().showMessage("视频检测已停止"))
        worker.error_signal.connect(self._preview_failed)
        worker.finished.connect(lambda: self._preview_worker_finished(worker))
        self.statusBar().showMessage("正在启动视频检测…")
        worker.start()

    def _show_video_frame(self, frame: DetectionFrame):
        self.current_frame = frame
        self._display_frame(frame, reset_table=True)

    def _video_finished(self):
        self.statusBar().showMessage("视频播放完成")

    def _preview_failed(self, message):
        if self.mode == "camera":
            self.CaplineEdit.setText("摄像头未开启")
        self._worker_failed(message)

    def _active_worker_finished(self, worker):
        if self.active_worker is worker:
            self.active_worker = None
            self._set_detection_busy(False)
        worker.deleteLater()

    def _preview_worker_finished(self, worker):
        if self.preview_worker is worker:
            self.preview_worker = None
        worker.deleteLater()

    def _worker_failed(self, message):
        self.statusBar().showMessage("任务失败")
        QMessageBox.warning(self, "处理失败", message)

    def _display_frame(self, frame: DetectionFrame, reset_table: bool):
        self.time_lb.setText(f"{frame.inference_seconds:.3f} s")
        self.label_nums.setText(str(len(frame.items)))
        self.PiclineEdit.setText(frame.source)
        self._render_image(frame.annotated_image)
        self._populate_combo(frame)
        self._show_item_details(frame.items[0] if frame.items else None)
        if reset_table:
            self._reset_table()
        self._append_table_items(frame.source, frame.items)

    def _render_image(self, image):
        width, height = self.get_resize_size(image)
        resized = cv2.resize(image, (width, height))
        self.label_show.setPixmap(tools.cvimg_to_qpiximg(resized))
        self.label_show.setAlignment(Qt.AlignCenter)

    def _populate_combo(self, frame: DetectionFrame):
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        if frame.items:
            choices = ["全部"] + [
                f"{item.class_name}_{index}" for index, item in enumerate(frame.items)
            ]
            self.comboBox.addItems(choices)
            self.comboBox.setDisabled(self.mode in {"video", "camera"})
        else:
            self.comboBox.setDisabled(True)
        self.comboBox.blockSignals(False)

    def _show_item_details(self, item):
        if item is None:
            values = ("", "", "", "", "", "")
        else:
            values = (
                item.display_name,
                f"{item.confidence * 100:.2f} %",
                *map(str, item.box),
            )
        widgets = (
            self.type_lb,
            self.label_conf,
            self.label_xmin,
            self.label_ymin,
            self.label_xmax,
            self.label_ymax,
        )
        for widget, value in zip(widgets, values):
            widget.setText(value)

    def _reset_table(self):
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self._row_counter = 0

    def _append_table_items(self, source, items):
        for item in items:
            while self.tableWidget.rowCount() >= MAX_TABLE_ROWS:
                self.tableWidget.removeRow(0)
            self._row_counter += 1
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            values = (
                self._row_counter,
                source,
                item.display_name,
                f"{item.confidence * 100:.2f} %",
                str(list(item.box)),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if column != 1:
                    cell.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.tableWidget.setItem(row, column, cell)
        self.tableWidget.scrollToBottom()

    def combox_change(self):
        frame = self.current_frame
        if frame is None or not frame.items:
            return
        text = self.comboBox.currentText()
        if text == "全部":
            self._show_item_details(frame.items[0])
            self._render_image(frame.annotated_image)
            return
        index = int(text.rsplit("_", 1)[1])
        item = frame.items[index]
        image = frame.original_image.copy()
        x1, y1, x2, y2 = item.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.show_labels:
            cv2.putText(
                image,
                f"{item.class_name} {item.confidence:.2f}",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        self._show_item_details(item)
        self._render_image(image)

    def get_resize_size(self, image):
        height, width = image.shape[:2]
        ratio = width / height
        if ratio >= self.show_width / self.show_height:
            return self.show_width, max(int(self.show_width / ratio), 1)
        return max(int(self.show_height * ratio), 1), self.show_height

    def update_conf_thres(self, value):
        self.conf_thres = value
        self._update_preview_options()
        if self.mode == "image" and self.current_source and self.active_worker is None:
            self._redetect_timer.start()

    def update_iou_thres(self, value):
        self.iou_thres = value
        self._update_preview_options()
        if self.mode == "image" and self.current_source and self.active_worker is None:
            self._redetect_timer.start()

    def update_show_labels(self, state):
        self.show_labels = state == Qt.Checked
        self._update_preview_options()
        if self.mode == "image" and self.current_source and self.active_worker is None:
            self._redetect_timer.start()

    def _update_preview_options(self):
        if self.preview_worker is not None:
            self.preview_worker.update_options(
                self.conf_thres,
                self.iou_thres,
                self.show_labels,
            )

    def _redetect_current_image(self):
        if self.mode == "image" and self.current_source and self.active_worker is None:
            self._start_image_detection(self.current_source)

    def save_results(self):
        if self.mode is None or self.current_source is None:
            QMessageBox.information(self, "提示", "请先完成检测。")
            return
        if self.mode == "camera":
            QMessageBox.information(self, "提示", "摄像头预览暂不支持直接保存。")
            return
        if self.mode == "video":
            self._start_video_export()
            return
        try:
            if self.mode == "image":
                self._save_image_result()
            elif self.mode == "batch":
                self._save_batch_results()
        except Exception as error:
            QMessageBox.critical(self, "保存失败", str(error))

    def _save_image_result(self):
        if self.current_frame is None:
            raise ValueError("当前没有可保存的图片结果。")
        image_path = make_output_path(self.current_source, Config.save_path)
        csv_path = make_output_path(
            self.current_source,
            Config.save_path,
            suffix="_detections",
            extension=".csv",
        )
        image_write(image_path, self.current_frame.annotated_image)
        write_detection_csv(csv_path, frame_csv_rows(self.current_frame))
        QMessageBox.information(
            self,
            "保存成功",
            f"图片：{image_path}\n数据：{csv_path}",
        )

    def _save_batch_results(self):
        if not self.batch_records:
            raise ValueError("当前没有可保存的批量结果。")
        rows = []
        for record in self.batch_records:
            output_path = make_output_path(record.source, Config.save_path)
            shutil.copy2(record.cached_image, output_path)
            rows.extend(frame_csv_rows(record))
        csv_source = f"{Path(self.current_source).name}.csv"
        csv_path = make_output_path(
            csv_source,
            Config.save_path,
            suffix="_detect_results",
            extension=".csv",
        )
        write_detection_csv(csv_path, rows)
        QMessageBox.information(
            self,
            "保存成功",
            f"已保存 {len(self.batch_records)} 张图片。\n数据：{csv_path}",
        )

    def _start_video_export(self):
        if self.export_worker is not None and self.export_worker.isRunning():
            return
        answer = QMessageBox.question(
            self,
            "导出视频",
            "导出会重新处理完整视频，并且当前版本不保留原音频。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        if not self._stop_preview():
            return
        worker = VideoExportWorker(
            str(self.current_source),
            Config.save_path,
            self.conf_thres,
            self.iou_thres,
            self.show_labels,
            self,
        )
        self.export_worker = worker
        self.progress_bar = ProgressBar(self)
        self.progress_bar.cancel_requested.connect(worker.stop)
        worker.progress_signal.connect(self._update_export_progress)
        worker.completed_signal.connect(self._video_export_completed)
        worker.canceled_signal.connect(self._video_export_canceled)
        worker.error_signal.connect(self._video_export_failed)
        worker.finished.connect(lambda: self._export_worker_finished(worker))
        self._set_export_busy(True)
        self.progress_bar.show()
        self.statusBar().showMessage("正在导出视频…")
        worker.start()

    def _update_export_progress(self, current: int, total: int):
        if self.progress_bar is not None:
            self.progress_bar.setValue(current, total)

    def _close_progress(self):
        if self.progress_bar is not None:
            self.progress_bar.finish()
            self.progress_bar = None

    def _video_export_completed(self, path):
        self._close_progress()
        self.statusBar().showMessage("视频导出完成")
        QMessageBox.information(self, "导出成功", f"视频已验证并保存：\n{path}")

    def _video_export_canceled(self):
        self._close_progress()
        self.statusBar().showMessage("视频导出已取消，未保留部分文件")

    def _video_export_failed(self, message):
        self._close_progress()
        self.statusBar().showMessage("视频导出失败")
        QMessageBox.critical(self, "视频导出失败", message)

    def _export_worker_finished(self, worker):
        if self.export_worker is worker:
            self.export_worker = None
        self._set_export_busy(False)
        worker.deleteLater()

    def closeEvent(self, event):
        if self._closing:
            event.ignore()
            return
        self._closing = True
        workers = [self.active_worker, self.preview_worker, self.export_worker]
        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.stop()
        unfinished = [
            worker
            for worker in workers
            if worker is not None and worker.isRunning() and not worker.wait(15000)
        ]
        if unfinished:
            self._closing = False
            event.ignore()
            QMessageBox.warning(self, "正在停止", "后台任务尚未安全结束，请稍后再次关闭。")
            return
        self._clear_batch_cache()
        self._close_progress()
        event.accept()


def main():
    if "--self-test" in sys.argv:
        report = Path(tempfile.gettempdir()) / "Detection-System-self-test.log"
        report.unlink(missing_ok=True)
        try:
            model_path = validate_model_file(Config.model_path, Config.model_sha256)
            from ultralytics import YOLO

            model = YOLO(str(model_path), task="detect")
            actual_names = {int(key): value for key, value in model.names.items()}
            if actual_names != Config.names:
                raise RuntimeError(f"模型类别不一致：{actual_names}")
            result = model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)[0]
            result.plot()
            report.write_text("OK\n", encoding="utf-8")
            return 0
        except Exception:
            report.write_text(traceback.format_exc(), encoding="utf-8")
            return 1

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Detection-System")
    try:
        window = MainWindow()
    except Exception as error:
        QMessageBox.critical(None, "启动失败", str(error))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
