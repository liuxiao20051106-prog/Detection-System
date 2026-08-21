import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detection_core import (
    ValidationError,
    make_output_path,
    validate_image_file,
    validate_model_file,
    validate_video_file,
    write_detection_csv,
)


class DetectionCoreTests(unittest.TestCase):
    def test_model_requires_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.pt"
            model.write_bytes(b"trusted model")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            self.assertEqual(validate_model_file(model, digest), model.resolve())
            with self.assertRaisesRegex(ValidationError, "完整性校验失败"):
                validate_model_file(model, "0" * 64)

    def test_missing_file_uses_validation_error(self):
        with self.assertRaisesRegex(ValidationError, "不存在或无法访问"):
            validate_image_file("definitely-missing.png")

    def test_corrupt_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_bytes(b"not-an-image")
            with self.assertRaisesRegex(ValidationError, "无法解码"):
                validate_image_file(path)

    def test_output_path_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grape.png"
            first = make_output_path(source, directory)
            first.touch()
            second = make_output_path(source, directory)
            self.assertNotEqual(first, second)
            self.assertEqual(second.name, "grape_detect_result_2.png")

    def test_csv_neutralizes_spreadsheet_formulas(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            write_detection_csv(output, [["\t=cmd|calc", "成熟", 0.9, 1, 2, 3, 4]])
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[1][1], "'\t=cmd|calc")

    def test_video_metadata_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            info = validate_video_file(path)
            self.assertEqual((info.width, info.height), (32, 24))
            self.assertGreater(info.frame_count, 0)


if __name__ == "__main__":
    unittest.main()
