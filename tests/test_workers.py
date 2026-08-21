import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detection_core import DetectionFrame
from workers import VideoExportWorker


def make_video(path: Path, frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("测试环境没有可用的 MJPG 编码器")
    for value in range(frames):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


class FakeVideoExportWorker(VideoExportWorker):
    def __init__(self, *args, cancel_after_first=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_after_first = cancel_after_first
        self.detected = 0

    def _load_model(self):
        return object()

    def _detect(self, model, source, image):
        self.detected += 1
        if self.cancel_after_first and self.detected == 1:
            self.stop()
        return DetectionFrame(source, image, image.copy(), (), 0.001)


class WorkerTests(unittest.TestCase):
    def test_video_completion_is_emitted_after_file_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.avi"
            output = Path(directory) / "output"
            make_video(source)
            worker = FakeVideoExportWorker(str(source), str(output), 0.25, 0.45, True)
            completed = []

            def verify(path):
                capture = cv2.VideoCapture(path)
                completed.append(
                    capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
                )
                capture.release()

            worker.completed_signal.connect(verify)
            worker.run()
            self.assertEqual(completed, [True])
            self.assertFalse(list(output.glob("*.partial.avi")))

    def test_cancel_removes_partial_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.avi"
            output = Path(directory) / "output"
            make_video(source)
            worker = FakeVideoExportWorker(
                str(source), str(output), 0.25, 0.45, True, cancel_after_first=True
            )
            canceled = []
            worker.canceled_signal.connect(lambda: canceled.append(True))
            worker.run()
            self.assertEqual(canceled, [True])
            self.assertFalse(list(output.glob("*.avi")))


if __name__ == "__main__":
    unittest.main()
