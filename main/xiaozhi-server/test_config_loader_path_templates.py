import sys
import types
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


fake_manage_api_client = types.ModuleType("config.manage_api_client")
fake_manage_api_client.init_service = lambda *args, **kwargs: None
fake_manage_api_client.get_server_config = lambda *args, **kwargs: None
fake_manage_api_client.get_agent_models = lambda *args, **kwargs: None
sys.modules.setdefault("config.manage_api_client", fake_manage_api_client)


from config.config_loader import apply_config_path_templates


class ConfigLoaderPathTemplatesTest(unittest.TestCase):
    def test_apply_config_path_templates_expands_experiment_paths(self):
        config = {
            "experiment_paths": {
                "workspace_root": "C:/demo/workspace",
                "lab_runs_root": "${workspace_root}/lab_runs",
                "experiment_name": "exp_demo",
                "experiment_root": "${lab_runs_root}/${experiment_name}",
                "experiment_config_root": "${experiment_root}/configs",
                "experiment_data_root": "${experiment_root}/data",
            },
            "prompt_template": "${experiment_config_root}/local_prompt.txt",
            "uvvis_scan_output_root": "${experiment_data_root}/uv_data_common",
            "LLM": {
                "codex_app_server": {
                    "workspace": "${workspace_root}",
                    "yaml_path": "${experiment_yaml_path}",
                    "stream_log_path": "${experiment_data_root}/{device_id}/{device_id}.log",
                }
            },
        }

        expanded = apply_config_path_templates(config)

        self.assertEqual(
            "C:/demo/workspace/lab_runs/exp_demo/configs/local_prompt.txt",
            expanded["prompt_template"].replace("\\", "/"),
        )
        self.assertEqual(
            "C:/demo/workspace/lab_runs/exp_demo/data/uv_data_common",
            expanded["uvvis_scan_output_root"].replace("\\", "/"),
        )
        self.assertEqual(
            "C:/demo/workspace/lab_runs/exp_demo/configs/experiments.yaml",
            expanded["LLM"]["codex_app_server"]["yaml_path"].replace("\\", "/"),
        )
        self.assertEqual(
            "C:/demo/workspace/lab_runs/exp_demo/data",
            expanded["experiment_paths"]["experiment_data_root"].replace("\\", "/"),
        )


if __name__ == "__main__":
    unittest.main()
