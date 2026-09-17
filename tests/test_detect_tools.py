# -*- coding: utf-8 -*-
"""detect_tools 与配置层的单元测试（不依赖 torch / ultralytics）。"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import Config
import detect_tools as tools


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


class CoordinateTests(unittest.TestCase):
    def test_coordinate_round_trip(self):
        yolo = tools.location_to_yolo(1000, 500, [100, 50, 500, 250])
        self.assertEqual(tools.yolo_to_location(1000, 500, yolo), [100, 50, 500, 250])

    def test_yolo_location_is_clamped_to_image(self):
        location = tools.yolo_to_location(100, 100, [0.5, 0.5, 1.5, 1.5])
        self.assertEqual(location, [0, 0, 99, 99])


class ImageFileTests(unittest.TestCase):
    def test_unicode_image_write_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '葡萄检测.png'
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            tools.img_cvwrite(path, image)
            loaded = tools.img_cvread(path)
            self.assertEqual(loaded.shape, image.shape)

    def test_read_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                tools.img_cvread(Path(directory) / 'not_exists.png')

    def test_list_image_files_is_filtered_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'b.PNG').touch()
            (root / 'A.jpg').touch()
            (root / 'notes.txt').touch()
            names = [path.name for path in tools.list_image_files(root)]
            self.assertEqual(names, ['A.jpg', 'b.PNG'])

    def test_target_path_for_keeps_extension(self):
        target = tools.target_path_for('dir/葡萄.png')
        self.assertEqual(target.name, '葡萄_detect_result.png')


class DetectionResultTests(unittest.TestCase):
    def make_result(self):
        boxes = _StubBoxes([[10, 20, 50, 60], [5, 5, 40, 40]], [0, 2], [0.9123, 0.5])
        return tools.DetectionResult.from_ultralytics(
            _StubResults(boxes), source='dir/葡萄.png', duration=0.0123)

    def test_single_dispatch_converts_tensors_to_python(self):
        detection = self.make_result()
        self.assertEqual(len(detection), 2)
        self.assertEqual(detection.classes, [0, 2])
        self.assertEqual(detection.boxes, [[10, 20, 50, 60], [5, 5, 40, 40]])
        self.assertEqual(detection.source, 'dir/葡萄.png')

    def test_labels_and_option_texts(self):
        detection = self.make_result()
        self.assertEqual(detection.label(0), Config.CH_names[0])
        self.assertEqual(detection.conf_text(0), '91.23 %')
        self.assertEqual(detection.option_texts(), ['immature_0', 'mature_1'])

    def test_unknown_class_id_falls_back_to_number(self):
        boxes = _StubBoxes([[0, 0, 1, 1]], [99], [0.1])
        detection = tools.DetectionResult.from_ultralytics(_StubResults(boxes))
        self.assertEqual(detection.label(0), '99')

    def test_csv_rows_are_numbered(self):
        rows = self.make_result().csv_rows()
        self.assertEqual([row[0] for row in rows], [1, 2])
        self.assertEqual(rows[0][2], Config.CH_names[0])


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_annotate_keeps_shape_and_colors(self):
        annotated = tools.annotate_image(self.image, [[10, 10, 60, 60]], [1], [0.87])
        self.assertEqual(annotated.shape, self.image.shape)
        self.assertTrue(annotated.any())  # 确实画上了东西

    def test_annotate_without_boxes_returns_original_values(self):
        annotated = tools.annotate_image(self.image, [], [])
        np.testing.assert_array_equal(annotated, self.image)

    def test_annotate_without_labels_only_draws_boxes(self):
        annotated = tools.annotate_image(self.image, [[10, 10, 60, 60]], [1], show_labels=False)
        self.assertEqual(annotated.shape, self.image.shape)

    def test_annotate_handles_target_at_top_edge(self):
        """目标贴着上边界时标签不能被裁掉，也不能越界报错。"""
        annotated = tools.annotate_image(self.image, [[0, 0, 30, 30]], [0], [0.9])
        self.assertEqual(annotated.shape, self.image.shape)

    def test_font_is_cached(self):
        self.assertIs(tools.load_font(20), tools.load_font(20))


class SizeTests(unittest.TestCase):
    def test_fit_size_keeps_ratio(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        width, height = tools.fit_size(image, 320, 240)
        self.assertLessEqual(width, 320)
        self.assertLessEqual(height, 240)
        self.assertAlmostEqual(width / height, 640 / 480, delta=0.02)

    def test_resize_to_fit_enlarges_small_images_to_display_area(self):
        """小图会被放大填满显示区域（与界面历史行为一致）。"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = tools.resize_to_fit(image, 770, 480)
        self.assertEqual(resized.shape, (480, 480, 3))

    def test_resize_to_fit_skips_when_size_already_matches(self):
        image = np.zeros((480, 770, 3), dtype=np.uint8)
        resized = tools.resize_to_fit(image, 770, 480)
        self.assertEqual(resized.shape, image.shape)


class CsvTests(unittest.TestCase):
    def test_write_csv_is_excel_friendly(self):
        detection = tools.DetectionResult(source='a.png', boxes=[[1, 2, 3, 4]], classes=[0], confs=[0.5])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / '葡萄.csv'
            tools.write_csv(target, detection.csv_rows())
            raw = target.read_bytes()
            self.assertTrue(raw.startswith(b'\xef\xbb\xbf'))  # utf-8-sig
            self.assertIn(Config.CH_names[0].encode('utf-8'), raw)


class ConfigTests(unittest.TestCase):
    def test_classes_are_the_single_source_of_truth(self):
        self.assertEqual(list(Config.names), list(range(len(Config.CH_names))))
        self.assertEqual([english for _id, english, _chinese in Config.CLASSES],
                         [Config.names[class_id] for class_id in Config.names])

    def test_paths_are_absolute_and_independent_from_cwd(self):
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            import importlib
            reloaded = importlib.reload(Config)
            self.assertTrue(os.path.isabs(reloaded.save_path))
            self.assertTrue(os.path.isabs(reloaded.model_path))
        finally:
            os.chdir(cwd)
            importlib.reload(Config)


if __name__ == '__main__':
    unittest.main()
