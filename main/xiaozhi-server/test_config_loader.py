import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config.config_loader as config_loader


class ConfigLoaderTest(unittest.TestCase):
    def test_get_default_config_path_falls_back_to_config_back_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            config_back_path = project_dir / "config_back.yaml"
            config_back_path.write_text("log: {}\n", encoding="utf-8")

            with mock.patch.object(
                config_loader,
                "get_project_dir",
                return_value=f"{project_dir}{os.sep}",
            ):
                self.assertEqual(
                    str(config_back_path),
                    config_loader.get_default_config_path(),
                )


if __name__ == "__main__":
    unittest.main()
