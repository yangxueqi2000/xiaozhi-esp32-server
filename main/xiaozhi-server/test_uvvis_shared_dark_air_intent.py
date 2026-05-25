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


class _FakeDialogue:
    def __init__(self):
        self.dialogue = []

    def put(self, message):
        self.dialogue.append(message)


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
from core.utils import textUtils


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.dialogue = _FakeDialogue()
        self.client_abort = False
        self.sentence_id = None
        self.device_id = "94:a9:90:28:ea:58"
        self.headers = {}
        self.session_id = ""
        self.experiment_session_id = "exp-1"
        self.experiment_current_step_id = intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID
        self.experiment_current_step = None
        self.experiment_progress_summary = None
        self.intent_type = "function_call"
        self.config = {}
        self.enriched = False

    def enrich_latest_clean_user_utterance_snapshot(self):
        self.enriched = True


class UVVisSharedDarkAirIntentTest(unittest.IsolatedAsyncioTestCase):
    def test_shared_dark_air_step_reply_mentions_start_scan(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_3_uv_vis_shared_dark_air_prep",
                    "title": "1-5号样品：共享暗电流和空气能量校正",
                    "prompts": {
                        "instruction": (
                            "先检查 1-5 号样品位都为空，参比位也不要放任何液体。"
                            "都空了就告诉我。可以开始时直接说“开始扫描”。"
                        ),
                    },
                }
            }
        }

        meta = intentHandler._get_cached_experiment_step_meta(conn)
        reply = intentHandler._compose_experiment_step_reply(meta, mode="guide")
        spoken = textUtils.prepare_runtime_spoken_text(reply)

        self.assertIn("都空了", spoken)
        self.assertIn("开始扫描", spoken)
        self.assertNotIn("uvvis_prepare_dark_current", spoken)

    async def test_scan_start_calls_dark_then_measurement(self):
        conn = _FakeConn()
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
            "phase": "await_scan_start",
        }
        spoken = []
        executed = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "下一步：装入比色皿。"

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "开始扫描",
                            "开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_prepare_dark_current",
                    {
                        "session_key": "lease-1",
                        "output_dir": str(Path("data").resolve() / "uv_data_common" / "94_a9_90_28_ea_58"),
                        "shared_output_dir": str((Path("data") / "uv_data_common").resolve()),
                    },
                ),
            ],
            executed,
        )
        self.assertEqual(1, len(completed))
        self.assertEqual(["下一步：装入比色皿。"], spoken)

    async def test_empty_then_start_in_one_reply_runs_shared_prep(self):
        conn = _FakeConn()
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
            "phase": "await_empty_positions",
        }
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            return True, "下一步：装入比色皿。"

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "都空了，开始扫描",
                            "都空了，开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_prepare_dark_current",
                    {
                        "session_key": "lease-1",
                        "output_dir": str(Path("data").resolve() / "uv_data_common" / "94_a9_90_28_ea_58"),
                        "shared_output_dir": str((Path("data") / "uv_data_common").resolve()),
                    },
                ),
            ],
            executed,
        )
        self.assertEqual(["下一步：装入比色皿。"], spoken)


    async def test_start_scan_reply_runs_shared_prep_after_empty_prompt(self):
        conn = _FakeConn()
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
            "phase": "await_empty_positions",
        }
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            return True, "下一步：装入比色皿。"

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "开始扫描",
                            "开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_prepare_dark_current",
                    {
                        "session_key": "lease-1",
                        "output_dir": str(Path("data").resolve() / "uv_data_common" / "94_a9_90_28_ea_58"),
                        "shared_output_dir": str((Path("data") / "uv_data_common").resolve()),
                    },
                ),
            ],
            executed,
        )
        self.assertEqual(["下一步：装入比色皿。"], spoken)

    async def test_spectra_measurement_missing_blank_redirects_to_shared_dark_air_when_blank_step_absent(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID
        conn._experiment_yaml_steps_cache = [
            {"id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID},
            {"id": intentHandler._UVVIS_SAMPLE_LOAD_STEP_ID},
            {"id": intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID},
        ]
        spoken = []
        redirected = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_start_turn(_conn, _text):
            return None

        async def fake_execute(_conn, tool_name, arguments):
            self.assertEqual("uvvis_measure_spectra", tool_name)
            self.assertEqual(
                {
                    "session_key": "lease-1",
                    "sample_positions": [1, 2, 3, 4, 5],
                    "ready_for_samples": True,
                    "output_dir": str(
                        Path("data").resolve() / "uv_data_common" / "94_a9_90_28_ea_58"
                    ),
                },
                arguments,
            )
            return {"message": "missing pure water blank"}

        async def fake_call(_conn, tool_name, arguments, *, priority="foreground"):
            redirected.append((tool_name, dict(arguments), priority))
            return {"result": {"ok": True}}

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_start_direct_intent_turn", fake_start_turn):
                with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                    with patch.object(intentHandler, "_call_experiment_graph_tool_fast", fake_call):
                        with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                            handled = await intentHandler.handle_direct_uvvis_intent(
                                conn,
                                "开始扫描",
                                "开始扫描",
                            )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "redirect_to_step",
                    {
                        "session_id": "exp-1",
                        "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
                    },
                    "foreground",
                )
            ],
            redirected,
        )
        self.assertEqual(
            ["这一步缺少共享前置校正，我先退回共享暗电流和空气能量校正。请先把样品位和参比位都清空，再告诉我开始扫描。"],
            spoken,
        )


if __name__ == "__main__":
    unittest.main()
