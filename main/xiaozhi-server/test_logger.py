import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config.logger as logger_module


class SetupLoggingTest(unittest.TestCase):
    def test_setup_logging_uses_defaults_when_log_section_is_missing(self):
        fake_logger = mock.Mock()
        fake_sink = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                with (
                    mock.patch.object(logger_module, "_logger_initialized", False),
                    mock.patch.object(logger_module, "check_config_file"),
                    mock.patch.object(logger_module, "load_config", return_value={}),
                    mock.patch.object(
                        logger_module,
                        "SafeRotatingFileSink",
                        return_value=fake_sink,
                    ) as sink_cls,
                    mock.patch.object(logger_module, "logger", fake_logger),
                ):
                    returned_logger = logger_module.setup_logging()
                    tmp_dir_exists = (Path(temp_dir) / "tmp").is_dir()
                    data_dir_exists = (Path(temp_dir) / "data").is_dir()
            finally:
                os.chdir(original_cwd)

        self.assertIs(returned_logger, fake_logger)
        fake_logger.configure.assert_called_once_with(
            extra={"selected_module": "00000000000000"}
        )
        fake_logger.remove.assert_called_once_with()
        self.assertEqual(2, fake_logger.add.call_count)
        sink_cls.assert_called_once_with(
            os.path.join("tmp", "server.log"),
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
        )

        console_call = fake_logger.add.call_args_list[0]
        self.assertIs(console_call.args[0], sys.stdout)
        self.assertEqual("INFO", console_call.kwargs["level"])

        file_call = fake_logger.add.call_args_list[1]
        self.assertIs(file_call.args[0], fake_sink)
        self.assertEqual("INFO", file_call.kwargs["level"])

        self.assertTrue(tmp_dir_exists)
        self.assertTrue(data_dir_exists)


if __name__ == "__main__":
    unittest.main()
