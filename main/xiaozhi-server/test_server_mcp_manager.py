import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: _FakeLogger()
sys.modules.setdefault("config.logger", fake_logger_module)

fake_config_loader_module = types.ModuleType("config.config_loader")
fake_config_loader_module.get_project_dir = lambda: str(SCRIPT_DIR) + "/"
sys.modules.setdefault("config.config_loader", fake_config_loader_module)

fake_mcp_types_module = types.ModuleType("mcp.types")
fake_mcp_types_module.LoggingMessageNotificationParams = object
sys.modules.setdefault("mcp.types", fake_mcp_types_module)

fake_mcp_client_module = types.ModuleType(
    "core.providers.tools.server_mcp.mcp_client"
)


class _FakeServerMCPClient:
    def __init__(self, *args, **kwargs):
        pass


fake_mcp_client_module.ServerMCPClient = _FakeServerMCPClient
fake_mcp_client_module._coerce_timeout_value = (
    lambda raw_value, default=None, allow_unbounded=False: default
)
sys.modules.setdefault(
    "core.providers.tools.server_mcp.mcp_client",
    fake_mcp_client_module,
)

fake_mcp_executor_module = types.ModuleType(
    "core.providers.tools.server_mcp.mcp_executor"
)
fake_mcp_executor_module.ServerMCPExecutor = object
sys.modules.setdefault(
    "core.providers.tools.server_mcp.mcp_executor",
    fake_mcp_executor_module,
)


from core.providers.tools.server_mcp.mcp_manager import ServerMCPManager


class _FakeConn:
    def __init__(self, yaml_path: str):
        self.experiment_yaml_path = yaml_path


class ServerMCPManagerConfigTest(unittest.TestCase):
    def test_load_config_injects_experiment_yaml_path_from_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            yaml_path = root / "lab_runs" / "exp_demo" / "configs" / "experiments.yaml"
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text("name: demo\n", encoding="utf-8")

            settings_path = root / ".mcp_server_settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "experiment-graph": {
                                "command": "python",
                                "args": ["experiment_graph_mcp_server.py"],
                                "env": {
                                    "PYTHONUTF8": "1",
                                    "PYTHONIOENCODING": "utf-8",
                                },
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            manager = ServerMCPManager(_FakeConn(str(yaml_path)))
            manager.config_path = str(settings_path)

            config = manager.load_config()

        self.assertEqual(
            str(yaml_path.resolve()),
            config["experiment-graph"]["env"]["EXPERIMENT_YAML_PATH"],
        )


if __name__ == "__main__":
    unittest.main()
