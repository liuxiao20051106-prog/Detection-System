import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch  # noqa: F401 - Windows must initialize Torch before PyQt5
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QApplication

from detection_core import MAX_TABLE_ROWS, DetectionItem
from MainProgram import MainWindow
from UIProgram import ui_sources_rc  # noqa: F401
from UIProgram.precess_bar import ProgressBar


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_background_is_embedded(self):
        self.assertTrue(QFile.exists(":/bgs/ui_imgs/bg22.png"))

    def test_progress_cancel_is_emitted_once(self):
        progress = ProgressBar()
        canceled = []
        progress.cancel_requested.connect(lambda: canceled.append(True))
        progress.onCancel()
        progress.onCancel()
        progress.finish()
        self.assertEqual(canceled, [True])

    def test_window_is_resizable_and_table_is_bounded(self):
        window = MainWindow()
        try:
            self.assertLess(window.minimumWidth(), 1250)
            self.assertGreater(window.maximumWidth(), 1250)
            item = DetectionItem(0, "immature", "未成熟", 0.8, (1, 2, 3, 4))
            window._append_table_items("sample.png", [item] * (MAX_TABLE_ROWS + 10))
            self.assertEqual(window.tableWidget.rowCount(), MAX_TABLE_ROWS)
        finally:
            window.close()

    def test_camera_state_is_preserved_when_worker_cannot_stop(self):
        class StubbornWorker:
            stopped = False

            def isRunning(self):
                return True

            def stop(self):
                self.stopped = True

            def wait(self, _timeout):
                return False

        window = MainWindow()
        worker = StubbornWorker()
        try:
            window.mode = "camera"
            window.preview_worker = worker
            window.CaplineEdit.setText("摄像头 0 开启")
            with patch("MainProgram.QMessageBox.warning"):
                window.camera_show()
            self.assertTrue(worker.stopped)
            self.assertEqual(window.mode, "camera")
            self.assertEqual(window.CaplineEdit.text(), "摄像头 0 开启")
        finally:
            window.preview_worker = None
            window.close()


if __name__ == "__main__":
    unittest.main()
