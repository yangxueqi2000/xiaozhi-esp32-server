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

    def test_ensure_directories_creates_selected_provider_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            config = {
                "log": {"log_dir": "logs"},
                "selected_module": {"LLM": "codex_app_server"},
                "LLM": {
                    "codex_app_server": {"output_dir": "models/codex_app_server"}
                },
            }

            with mock.patch.object(
                config_loader,
                "get_project_dir",
                return_value=f"{project_dir}{os.sep}",
            ):
                config_loader.ensure_directories(config)

            self.assertTrue((project_dir / "models" / "codex_app_server").is_dir())


if __name__ == "__main__":
    unittest.main()
