import sys
import types
import unittest
from pathlib import Path
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

fake_hello_module = types.ModuleType("core.handle.helloHandle")


async def _fake_check_wakeup_words(conn, filtered_text):
    return False


fake_hello_module.checkWakeupWords = _fake_check_wakeup_words
fake_hello_module.handleHelloMessage = lambda *args, **kwargs: None
sys.modules.setdefault("core.handle.helloHandle", fake_hello_module)

fake_device_mcp_module = types.ModuleType("core.providers.tools.device_mcp")
fake_device_mcp_module.call_mcp_tool = lambda *args, **kwargs: None
fake_device_mcp_module.handle_mcp_message = lambda *args, **kwargs: None
sys.modules.setdefault("core.providers.tools.device_mcp", fake_device_mcp_module)

fake_send_audio_module = types.ModuleType("core.handle.sendAudioHandle")


async def _fake_send_stt_message(conn, text):
    return None


fake_send_audio_module.send_stt_message = _fake_send_stt_message
fake_send_audio_module.send_tts_message = lambda *args, **kwargs: None
fake_send_audio_module.sendAudioMessage = lambda *args, **kwargs: None
fake_send_audio_module.SentenceType = types.SimpleNamespace(
    FIRST="FIRST",
    LAST="LAST",
)
sys.modules.setdefault("core.handle.sendAudioHandle", fake_send_audio_module)

sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)

from core.handle import intentHandler


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.device_id = ""
        self.headers = {}
        self.session_id = ""
        self.experiment_session_id = "exp-1"


class UVVisToolRefreshTest(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_uvvis_session_key_refreshes_uvvis_client_when_dark_tool_missing(self):
        conn = _FakeConn()
        refreshed = []

        class _FakeManager:
            def __init__(self):
                self.tool_names = {
                    "uvvis_session",
                    "uvvis_measure_spectra",
                    "uvvis_measure_kinetics",
                }

            def is_mcp_tool(self, name):
                return name in self.tool_names

            async def ensure_client_initialized(self, client_name):
                refreshed.append(client_name)
                self.tool_names.add("uvvis_prepare_dark_current")
                return True

        async def fake_execute(_conn, tool_name, arguments):
            self.assertEqual("uvvis_session", tool_name)
            self.assertEqual({"action": "acquire"}, arguments)
            return {"session_key": "lease-1"}

        manager = _FakeManager()

        with patch.object(intentHandler, "_get_server_mcp_manager", return_value=manager):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                session_key, busy_reply = await intentHandler._ensure_uvvis_session_key(conn)

        self.assertEqual(["uvvis"], refreshed)
        self.assertEqual("lease-1", session_key)
        self.assertEqual("", busy_reply)
        self.assertEqual("lease-1", getattr(conn, "_uvvis_session_key", ""))
        self.assertTrue(manager.is_mcp_tool("uvvis_prepare_dark_current"))


if __name__ == "__main__":
    unittest.main()
