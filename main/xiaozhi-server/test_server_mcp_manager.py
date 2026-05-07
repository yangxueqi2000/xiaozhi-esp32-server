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
        self.device_id = ""
        self.headers = {}
        self.session_id = ""
        self.experiment_session_id = ""


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


class _FakeRuntimeClient:
    def __init__(self):
        self.calls = []

    def is_connected(self):
        return True

    async def call_tool(self, tool_name, arguments, progress_callback=None, meta=None):
        self.calls.append(
            {
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "meta": dict(meta or {}),
            }
        )
        return {"ok": True}


class ServerMCPManagerRecoveryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ServerMCPManager._shared_clients = {}
        ServerMCPManager._shared_client_tools = {}
        ServerMCPManager._shared_tool_to_client = {}
        ServerMCPManager._shared_initialized = False
        ServerMCPManager._shared_ref_count = 0
        ServerMCPManager._shared_init_lock = None
        ServerMCPManager._shared_reconnect_locks = {}

    async def test_execute_tool_refreshes_uvvis_mapping_when_tool_owner_is_missing(self):
        conn = _FakeConn("")
        manager = ServerMCPManager(conn)
        fake_client = _FakeRuntimeClient()
        refreshed = []

        async def fake_ensure_client_initialized(client_name):
            refreshed.append(client_name)
            ServerMCPManager._shared_clients["uvvis"] = fake_client
            ServerMCPManager._shared_client_tools["uvvis"] = [
                {
                    "function": {
                        "name": "uvvis_prepare_dark_current",
                    }
                }
            ]
            ServerMCPManager._shared_tool_to_client[
                "uvvis_prepare_dark_current"
            ] = "uvvis"
            return True

        manager.ensure_client_initialized = fake_ensure_client_initialized

        result = await manager.execute_tool(
            "uvvis_prepare_dark_current",
            {"session_key": "lease-1"},
        )

        self.assertEqual(["uvvis"], refreshed)
        self.assertEqual({"ok": True}, result)
        self.assertEqual(1, len(fake_client.calls))
        self.assertEqual(
            {
                "tool_name": "uvvis_prepare_dark_current",
                "arguments": {"session_key": "lease-1"},
                "meta": {},
            },
            fake_client.calls[0],
        )


if __name__ == "__main__":
    unittest.main()
