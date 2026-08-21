import tempfile
import unittest
from pathlib import Path

import numpy as np

import detect_tools


class DetectToolsTests(unittest.TestCase):
    def test_coordinate_round_trip(self):
        yolo = detect_tools.location_to_yolo(1000, 500, [100, 50, 500, 250])
        self.assertEqual(detect_tools.yolo_to_location(1000, 500, yolo), [100, 50, 500, 250])

    def test_unicode_image_write_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "葡萄检测.png"
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            detect_tools.img_cvwrite(path, image)
            loaded = detect_tools.img_cvread(path)
            self.assertEqual(loaded.shape, image.shape)

    def test_list_image_files_is_filtered_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.PNG").touch()
            (root / "A.jpg").touch()
            (root / "notes.txt").touch()
            names = [Path(path).name for path in detect_tools.list_image_files(root)]
            self.assertEqual(names, ["A.jpg", "b.PNG"])


if __name__ == "__main__":
    unittest.main()
