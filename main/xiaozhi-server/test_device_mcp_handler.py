import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


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
sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)


from core.api.device_mcp_handler import DeviceMCPHandler


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _FakeMCPClient:
    async def is_ready(self):
        return True


class _FakeConn:
    def __init__(self, session_id: str, device_id: str, websocket=object()):
        self.session_id = session_id
        self.device_id = device_id
        self.mcp_client = _FakeMCPClient()
        self.websocket = websocket


class _FakeWSServer:
    def __init__(self, conn):
        self._conn = conn

    async def get_connection(self, session_id=None, device_id=None):
        return self._conn


class DeviceMCPHandlerTest(unittest.IsolatedAsyncioTestCase):
    async def test_take_photo_uses_configured_timeout_when_request_omits_timeout(self):
        config = {
            "server": {"auth_key": "demo", "http_port": 8003},
            "device_mcp_shortcuts": {
                "server_photo_timeout": 20,
            },
        }
        conn = _FakeConn("session-1", "94:a9:90:27:3c:84")
        ws_server = _FakeWSServer(conn)
        handler = DeviceMCPHandler(config, ws_server)
        captured = {}

        async def fake_call_mcp_tool(
            _conn, _mcp_client, tool_name, tool_args, **kwargs
        ):
            captured["tool_name"] = tool_name
            captured["tool_args"] = dict(tool_args)
            captured["timeout"] = kwargs.get("timeout")
            return {"ok": True}

        request = _FakeRequest(
            {
                "session_id": "session-1",
                "device_id": "94:a9:90:27:3c:84",
                "question": "Please take a test photo.",
            }
        )

        with patch(
            "core.api.device_mcp_handler.call_mcp_tool",
            fake_call_mcp_tool,
        ):
            response = await handler.handle_post(request)

        payload = json.loads(response.text)
        self.assertEqual(200, response.status)
        self.assertTrue(payload["success"])
        self.assertEqual(20, captured["timeout"])

    async def test_take_photo_only_sends_question_to_device_tool(self):
        config = {
            "server": {"auth_key": "demo", "http_port": 8003},
        }
        conn = _FakeConn("session-1", "94:a9:90:27:3c:84")
        ws_server = _FakeWSServer(conn)
        handler = DeviceMCPHandler(config, ws_server)
        captured = {}

        async def fake_call_mcp_tool(
            _conn, _mcp_client, tool_name, tool_args, **kwargs
        ):
            captured["tool_name"] = tool_name
            captured["tool_args"] = dict(tool_args)
            return {"ok": True}

        request = _FakeRequest(
            {
                "session_id": "session-1",
                "device_id": "94:a9:90:27:3c:84",
                "question": "Please photograph sample 1.",
                "photo_name": "sample_1",
                "timeout": 5,
            }
        )

        with patch(
            "core.api.device_mcp_handler.call_mcp_tool",
            fake_call_mcp_tool,
        ):
            response = await handler.handle_post(request)

        payload = json.loads(response.text)
        self.assertEqual(200, response.status)
        self.assertTrue(payload["success"])
        self.assertEqual("self_camera_take_photo", captured["tool_name"])
        self.assertEqual(
            {
                "question": (
                    'Please photograph sample 1.\n'
                    '[XIAOZHI_META]{"photo_name":"sample_1"}'
                )
            },
            captured["tool_args"],
        )

    async def test_take_photo_dedupes_same_photo_for_same_utterance(self):
        config = {
            "server": {"auth_key": "demo", "http_port": 8003},
        }
        conn = _FakeConn("session-1", "94:a9:90:27:3c:84")
        conn._latest_clean_user_utterance_text = "我想重新拍一张二号样品的照片。"
        conn._latest_clean_user_utterance_logged_at = 12345.0
        ws_server = _FakeWSServer(conn)
        handler = DeviceMCPHandler(config, ws_server)
        call_count = 0

        async def fake_call_mcp_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"ok": True}

        request_body = {
            "session_id": "session-1",
            "device_id": "94:a9:90:27:3c:84",
            "question": "Please photograph sample 2.",
            "photo_name": "2号样品照片_20260513_163621",
            "timeout": 5,
        }

        with patch(
            "core.api.device_mcp_handler.call_mcp_tool",
            fake_call_mcp_tool,
        ):
            first_response = await handler.handle_post(_FakeRequest(request_body))
            second_body = dict(request_body)
            second_body["photo_name"] = "2号样品照片_20260513_163657"
            second_response = await handler.handle_post(_FakeRequest(second_body))

        first_payload = json.loads(first_response.text)
        second_payload = json.loads(second_response.text)
        self.assertEqual(200, first_response.status)
        self.assertEqual(200, second_response.status)
        self.assertTrue(first_payload["success"])
        self.assertTrue(second_payload["success"])
        self.assertTrue(second_payload["duplicate_suppressed"])
        self.assertEqual(1, call_count)

    async def test_take_photo_rejects_disconnected_websocket_before_timeout(self):
        config = {
            "server": {"auth_key": "demo", "http_port": 8003},
        }
        conn = _FakeConn("session-1", "94:a9:90:27:3c:84", websocket=None)
        ws_server = _FakeWSServer(conn)
        handler = DeviceMCPHandler(config, ws_server)

        request = _FakeRequest(
            {
                "session_id": "session-1",
                "device_id": "94:a9:90:27:3c:84",
                "question": "Please photograph sample 1.",
                "photo_name": "sample_1",
                "timeout": 5,
            }
        )

        response = await handler.handle_post(request)
        payload = json.loads(response.text)

        self.assertEqual(409, response.status)
        self.assertFalse(payload["success"])
        self.assertIn("offline", payload["message"])

    async def test_take_photo_timeout_recovers_saved_photo_from_device_directory(self):
        device_id = "94:a9:90:27:3c:84"

        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            data_root = workspace / "lab_runs" / "exp1" / "data"
            safe_device = "94_a9_90_27_3c_84"
            device_dir = data_root / safe_device
            device_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "server": {"auth_key": "demo", "http_port": 8003},
                "selected_module": {"LLM": "codex_app_server"},
                "LLM": {
                    "codex_app_server": {
                        "workspace": str(workspace),
                        "yaml_path": "lab_runs/exp1/configs/experiments.yaml",
                    }
                },
                "device_mcp_shortcuts": {
                    "server_photo_recovery_window_seconds": 0.2,
                },
            }

            conn = _FakeConn("session-1", device_id)
            ws_server = _FakeWSServer(conn)
            handler = DeviceMCPHandler(config, ws_server)

            async def fake_call_mcp_tool(*args, **kwargs):
                saved_path = device_dir / "sample_1_20260430_150000.png"
                saved_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
                raise TimeoutError("tool call timed out")

            request = _FakeRequest(
                {
                    "session_id": "session-1",
                    "device_id": device_id,
                    "question": "Please photograph sample 1.",
                    "photo_name": "sample_1_20260430_150000",
                    "timeout": 5,
                }
            )

            with patch(
                "core.api.device_mcp_handler.call_mcp_tool",
                fake_call_mcp_tool,
            ):
                response = await handler.handle_post(request)

            payload = json.loads(response.text)
            self.assertEqual(200, response.status)
            self.assertTrue(payload["success"])
            self.assertTrue(payload["recovered_after_timeout"])
            self.assertEqual(
                str(device_dir / "sample_1_20260430_150000.png"),
                payload["saved_photo_path"],
            )
            self.assertEqual(
                "sample_1_20260430_150000.png",
                payload["photo_meta"]["file_name"],
            )


if __name__ == "__main__":
    unittest.main()
