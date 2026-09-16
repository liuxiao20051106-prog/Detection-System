# -*- coding: utf-8 -*-
"""后台线程层的单元测试。

这里用桩模型替代真正的 YOLO，因此不需要 torch / ultralytics 也能验证
“启动 -> 推理 -> 结果落地 -> 信号抛出”这一整条链路。
"""

import logging
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PyQt5.QtWidgets import QApplication
except ImportError:  # 缺少 GUI / OpenCV 依赖时跳过
    cv2 = None

import Config  # noqa: E402 - 需要先判定 GUI 依赖是否可用

if cv2 is not None:
    import detect_tools as tools
    import workers

# 失败用例里会故意触发异常，抬高日志级别避免测试输出被 traceback 淹没。
logging.getLogger('workers').addHandler(logging.NullHandler())
logging.getLogger('workers').propagate = False


class _StubBoxes:
    """模拟 ultralytics Results.boxes 的张量接口。"""

    class _Tensor:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    def __init__(self, xyxy, cls, conf):
        self.xyxy = self._Tensor(xyxy)
        self.cls = self._Tensor(cls)
        self.conf = self._Tensor(conf)


class _StubResults:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubModel:
    """假装自己会推理的 YOLO。"""

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = 0

    def predict(self, image, conf=None, iou=None, verbose=False):
        self.calls += 1
        return [_StubResults(self._boxes)]


APP = None


def get_app():
    """模块级持有 QApplication：局部变量会被回收，回收后 Qt 对象会一起失效。"""
    global APP
    if APP is None:
        APP = QApplication([])
    return APP


def pump_events(worker, timeout_seconds=20):
    """启动线程并泵事件队列直到线程结束（加一点休眠，避免忙循环把 CPU 占满）。"""
    import time

    app = get_app()
    worker.start()
    deadline = time.time() + timeout_seconds
    while worker.isRunning() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    app.processEvents()


@unittest.skipIf(cv2 is None, '需要 OpenCV 与 PyQt5')
class ImageWorkerTests(unittest.TestCase):
    boxes = _StubBoxes([[4, 6, 30, 40], [2, 2, 20, 20]], [0, 2], [0.87654, 0.5])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_save_path = Config.save_path
        Config.save_path = self.tmp.name
        self.image_path = Path(self.tmp.name) / '葡萄.png'
        tools.img_cvwrite(self.image_path, np.zeros((64, 96, 3), dtype=np.uint8))

    def tearDown(self):
        Config.save_path = self.original_save_path
        self.tmp.cleanup()

    def test_detection_emits_python_data(self):
        model = _StubModel(self.boxes)
        worker = workers.ImageDetectionWorker(model)
        results = []
        worker.image_done.connect(lambda detection, *_rest: results.append(detection))

        worker.configure([str(self.image_path)], 0.25, 0.45)
        pump_events(worker)

        self.assertEqual(model.calls, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].classes, [0, 2])
        self.assertEqual(results[0].conf_text(0), '87.65 %')
        self.assertEqual(results[0].boxes[0], [4, 6, 30, 40])

    def test_persist_mode_writes_image_and_csv(self):
        worker = workers.ImageDetectionWorker(_StubModel(self.boxes))
        finished = []
        worker.finished_all.connect(finished.append)

        worker.configure([str(self.image_path)], 0.25, 0.45, persist=True)
        pump_events(worker)

        self.assertTrue((Path(self.tmp.name) / '葡萄_detect_result.png').is_file())
        csv_lines = (Path(self.tmp.name) / 'batch_detect_result.csv').read_text(encoding='utf-8-sig')
        self.assertIn(Config.CH_names[0], csv_lines)
        self.assertEqual(len(finished), 1)

    def test_single_failure_does_not_break_the_batch(self):
        worker = workers.ImageDetectionWorker(_StubModel(self.boxes))
        failures, done = [], []
        worker.failed.connect(failures.append)
        worker.image_done.connect(lambda detection, *_rest: done.append(detection))

        worker.configure([str(Path(self.tmp.name) / '不存在.png'), str(self.image_path)], 0.25, 0.45)
        pump_events(worker)

        self.assertEqual(len(failures), 1)
        self.assertEqual(len(done), 1)  # 第二张仍然被处理

    def test_stop_flag_skips_remaining_items(self):
        worker = workers.ImageDetectionWorker(_StubModel(self.boxes))
        done = []
        worker.image_done.connect(lambda detection, *_rest: done.append(detection))

        worker.configure([str(self.image_path)] * 3, 0.25, 0.45)
        worker.stop()
        pump_events(worker)

        self.assertEqual(done, [])


@unittest.skipIf(cv2 is None, '需要 OpenCV 与 PyQt5')
class FrameWorkerTests(unittest.TestCase):
    """逐帧推理线程：主线程投递帧，线程回传结果。"""

    boxes = _StubBoxes([[1, 1, 10, 10]], [1], [0.9])

    def make_worker(self):
        worker = workers.FrameDetectionWorker(_StubModel(self.boxes))
        worker.configure(0.25, 0.45)
        self.worker = worker  # 保持引用，避免 Qt 对象被垃圾回收后无法访问
        return worker

    def test_every_submitted_frame_gets_a_result(self):
        worker = self.make_worker()
        seen = []
        worker.frame_done.connect(lambda detection, annotated, frame: seen.append(detection))

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        worker.start()
        for _ in range(5):
            worker.submit(frame)  # 队列满会自动丢帧，这里等回调即可
        worker.stop()
        pump_events(worker)

        self.assertTrue(seen)
        for detection in seen:
            self.assertEqual(detection.classes, [1])
            self.assertEqual(detection.boxes, [[1, 1, 10, 10]])

    def test_submit_rejects_when_not_running(self):
        worker = self.make_worker()
        self.assertFalse(worker.submit(np.zeros((8, 8, 3), dtype=np.uint8)))

    def test_has_pending_is_false_when_queue_drained(self):
        worker = self.make_worker()
        worker.start()
        worker.submit(np.zeros((16, 16, 3), dtype=np.uint8))
        worker.stop()
        pump_events(worker)
        self.assertFalse(worker.has_pending())


@unittest.skipIf(cv2 is None, '需要 OpenCV 与 PyQt5')
class CaptureHelperTests(unittest.TestCase):
    def test_open_video_source(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            video = Path(tmp.name) / 'clip.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (64, 48))
            if not writer.isOpened():
                self.skipTest('当前环境无法初始化视频编码器')
            try:
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            finally:
                writer.release()

            cap = workers.open_video_capture(str(video))
            self.assertIsNotNone(cap)
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (48, 64, 3))
        finally:
            tmp.cleanup()

    def test_open_missing_video_returns_none(self):
        self.assertIsNone(workers.open_video_capture(str(Path(tempfile.gettempdir()) / 'not_exists.mp4')))


if __name__ == '__main__':
    unittest.main()
