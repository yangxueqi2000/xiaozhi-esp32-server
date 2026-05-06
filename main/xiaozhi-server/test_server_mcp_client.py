import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import MethodType


SCRIPT_DIR = Path(__file__).resolve().parent


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: _FakeLogger()
sys.modules["config.logger"] = fake_logger_module

core_module = sys.modules.setdefault("core", types.ModuleType("core"))
core_module.__path__ = getattr(core_module, "__path__", [])
utils_module = sys.modules.setdefault("core.utils", types.ModuleType("core.utils"))
utils_module.__path__ = getattr(utils_module, "__path__", [])
fake_util_module = types.ModuleType("core.utils.util")
fake_util_module.sanitize_tool_name = lambda name: str(name or "").strip()
sys.modules["core.utils.util"] = fake_util_module

mcp_module = types.ModuleType("mcp")
mcp_module.ClientSession = object
mcp_module.StdioServerParameters = object
mcp_module.Implementation = object
sys.modules["mcp"] = mcp_module

fake_client_session_module = types.ModuleType("mcp.client.session")
fake_client_session_module.SamplingFnT = object
fake_client_session_module.ElicitationFnT = object
fake_client_session_module.ListRootsFnT = object
fake_client_session_module.LoggingFnT = object
fake_client_session_module.MessageHandlerFnT = object
sys.modules["mcp.client.session"] = fake_client_session_module

fake_stdio_module = types.ModuleType("mcp.client.stdio")
fake_stdio_module.stdio_client = object
sys.modules["mcp.client.stdio"] = fake_stdio_module

fake_sse_module = types.ModuleType("mcp.client.sse")
fake_sse_module.sse_client = object
sys.modules["mcp.client.sse"] = fake_sse_module

fake_streamable_http_module = types.ModuleType("mcp.client.streamable_http")
fake_streamable_http_module.streamablehttp_client = object
sys.modules["mcp.client.streamable_http"] = fake_streamable_http_module

fake_shared_session_module = types.ModuleType("mcp.shared.session")
fake_shared_session_module.ProgressFnT = object
sys.modules["mcp.shared.session"] = fake_shared_session_module

MODULE_PATH = SCRIPT_DIR / "core" / "providers" / "tools" / "server_mcp" / "mcp_client.py"
MODULE_SPEC = importlib.util.spec_from_file_location("test_server_mcp_client_module", MODULE_PATH)
MCP_CLIENT_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC and MODULE_SPEC.loader
MODULE_SPEC.loader.exec_module(MCP_CLIENT_MODULE)

ServerMCPClient = MCP_CLIENT_MODULE.ServerMCPClient


class ServerMCPClientLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_raises_when_worker_fails_before_ready(self):
        client = ServerMCPClient({"command": "fake"})

        async def fake_worker(self, **kwargs):
            try:
                self._worker_error = RuntimeError("Connection closed")
                self._ready_evt.set()
                raise self._worker_error
            finally:
                self._reset_runtime_state()

        client._worker = MethodType(fake_worker, client)

        with self.assertRaisesRegex(RuntimeError, "Connection closed"):
            await client.initialize()

        self.assertFalse(client.is_connected())
        self.assertIsNone(client._worker_task)
        self.assertEqual([], client.get_available_tools())

    async def test_cleanup_clears_connected_session_state(self):
        client = ServerMCPClient({"command": "fake"})
        fake_tool = types.SimpleNamespace(
            name="uvvis_session",
            description="UV-Vis session tool",
            inputSchema={"type": "object"},
        )

        async def fake_worker(self, **kwargs):
            self.session = object()
            self.tools = [fake_tool]
            self.tools_dict = {"uvvis_session": fake_tool}
            self.name_mapping = {"uvvis_session": "uvvis_session"}
            self._ready_evt.set()
            try:
                await self._shutdown_evt.wait()
            finally:
                self._reset_runtime_state()

        client._worker = MethodType(fake_worker, client)

        await client.initialize()
        self.assertTrue(client.is_connected())
        self.assertEqual(
            ["uvvis_session"],
            [tool["function"]["name"] for tool in client.get_available_tools()],
        )

        await client.cleanup()

        self.assertFalse(client.is_connected())
        self.assertIsNone(client._worker_task)
        self.assertIsNone(client.session)
        self.assertEqual([], client.get_available_tools())


if __name__ == "__main__":
    unittest.main()
