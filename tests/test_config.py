import os
import unittest
from pathlib import Path

import Config


class ConfigTests(unittest.TestCase):
    def test_project_assets_exist(self):
        self.assertTrue(Path(Config.model_path).is_file())
        self.assertTrue(Path(Config.font_path).is_file())

    def test_save_path_is_absolute(self):
        self.assertTrue(os.path.isabs(Config.save_path))

    def test_class_names_stay_aligned(self):
        self.assertEqual(list(Config.names), list(range(len(Config.CH_names))))


if __name__ == "__main__":
    unittest.main()
