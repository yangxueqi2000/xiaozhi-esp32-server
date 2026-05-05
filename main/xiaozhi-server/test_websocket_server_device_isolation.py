import asyncio
import re
import sys
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

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: _FakeLogger()
sys.modules.setdefault("config.logger", fake_logger_module)

fake_config_loader_module = types.ModuleType("config.config_loader")


async def _fake_get_config_from_api_async(config):
    return config


fake_config_loader_module.get_config_from_api_async = _fake_get_config_from_api_async
fake_config_loader_module.get_project_dir = lambda: str(SCRIPT_DIR)
sys.modules.setdefault("config.config_loader", fake_config_loader_module)

fake_auth_module = types.ModuleType("core.auth")


class _FakeAuthManager:
    def __init__(self, *args, **kwargs):
        pass


class _FakeAuthenticationError(Exception):
    pass


fake_auth_module.AuthManager = _FakeAuthManager
fake_auth_module.AuthenticationError = _FakeAuthenticationError
sys.modules.setdefault("core.auth", fake_auth_module)

fake_connection_module = types.ModuleType("core.connection")
fake_connection_module.ConnectionHandler = object
sys.modules.setdefault("core.connection", fake_connection_module)

fake_modules_initialize_module = types.ModuleType("core.utils.modules_initialize")
fake_modules_initialize_module.initialize_modules = lambda *args, **kwargs: {}
sys.modules.setdefault(
    "core.utils.modules_initialize",
    fake_modules_initialize_module,
)

fake_util_module = types.ModuleType("core.utils.util")
fake_util_module.check_asr_update = lambda *args, **kwargs: False
fake_util_module.check_vad_update = lambda *args, **kwargs: False


def _fake_remove_punctuation_and_length(text):
    normalized = re.sub(r"[\W_]+", "", str(text or ""), flags=re.UNICODE)
    return len(normalized), normalized


fake_util_module.remove_punctuation_and_length = _fake_remove_punctuation_and_length
fake_util_module.sanitize_tool_name = lambda name: str(name or "").strip()
sys.modules.setdefault("core.utils.util", fake_util_module)

from core.websocket_server import WebSocketServer


class _FakeHandler:
    def __init__(self, *, session_id: str, reconnect_ready: bool):
        self.session_id = session_id
        self._reconnect_ready = reconnect_ready

    async def wait_until_transport_detached(self, timeout: float = 2.0) -> bool:
        return self._reconnect_ready


class _FakeWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.close_calls = []

    async def send(self, text):
        self.sent_messages.append(text)

    async def close(self, code=None, reason=None):
        self.close_calls.append({"code": code, "reason": reason})


class WebSocketServerDeviceIsolationTest(unittest.IsolatedAsyncioTestCase):
    def _build_server(self):
        server = object.__new__(WebSocketServer)
        server.logger = _FakeLogger()
        server.connections_lock = asyncio.Lock()
        server.connections_by_session = {}
        server.connections_by_device = {}
        return server

    async def test_duplicate_live_device_connection_is_rejected(self):
        server = self._build_server()
        existing_handler = _FakeHandler(
            session_id="session-live-1",
            reconnect_ready=False,
        )
        server.connections_by_device["device-a"] = existing_handler
        websocket = _FakeWebSocket()

        handler, rejected = await server._resolve_existing_device_handler(
            websocket,
            "device-a",
        )

        self.assertIsNone(handler)
        self.assertTrue(rejected)
        self.assertEqual(
            ["same device-id is already connected"],
            websocket.sent_messages,
        )
        self.assertEqual(1, len(websocket.close_calls))
        self.assertEqual(1008, websocket.close_calls[0]["code"])

    async def test_detached_device_connection_is_reused(self):
        server = self._build_server()
        existing_handler = _FakeHandler(
            session_id="session-detached-1",
            reconnect_ready=True,
        )
        server.connections_by_device["device-a"] = existing_handler
        websocket = _FakeWebSocket()

        handler, rejected = await server._resolve_existing_device_handler(
            websocket,
            "device-a",
        )

        self.assertIs(handler, existing_handler)
        self.assertFalse(rejected)
        self.assertEqual([], websocket.sent_messages)
        self.assertEqual([], websocket.close_calls)

    async def test_different_device_id_is_not_rejected(self):
        server = self._build_server()
        existing_handler = _FakeHandler(
            session_id="session-live-1",
            reconnect_ready=False,
        )
        server.connections_by_device["device-a"] = existing_handler
        websocket = _FakeWebSocket()

        handler, rejected = await server._resolve_existing_device_handler(
            websocket,
            "device-b",
        )

        self.assertIsNone(handler)
        self.assertFalse(rejected)
        self.assertEqual([], websocket.sent_messages)
        self.assertEqual([], websocket.close_calls)
