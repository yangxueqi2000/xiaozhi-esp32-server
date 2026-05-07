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
fake_send_audio_module.SentenceType = types.SimpleNamespace(FIRST="FIRST", LAST="LAST")
sys.modules.setdefault("core.handle.sendAudioHandle", fake_send_audio_module)

sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)

from core.handle import intentHandler


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.dialogue = _FakeDialogue()
        self.client_abort = False
        self.sentence_id = None
        self.chat_session_id = ""
        self.model_session_key = ""
        self.user_id = ""
        self.device_id = "test-device"
        self.prompt = ""
        self.llm = None
        self.experiment_session_id = "exp-1"
        self.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        self.experiment_current_step = None
        self.experiment_progress_summary = None
        self.intent_type = "function_call"
        self.config = {}
        self.tts = None
        self.cmd_exit = []


class UvvisSharedBlankFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_blank_prep_reuses_existing_shared_pure_water_blank(self):
        conn = _FakeConn()
        spoken = []
        completed = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_start_turn(_conn, _text):
            return None

        async def fake_complete(_conn, *, observations, fallback_reply):
            completed.append(
                {
                    "observations": observations,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "当前批次纯水空白可复用，接下来装入样品比色皿。"

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"unexpected": True}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with TemporaryDirectory() as temp_dir:
            shared_blank_csv = Path(temp_dir) / "shared_pure_water_blank.csv"
            shared_blank_csv.write_text("wavelength,absorbance\n400,0.0\n", encoding="utf-8")

            with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
                with patch.object(intentHandler, "_start_direct_intent_turn", fake_start_turn):
                    with patch.object(intentHandler, "_complete_uvvis_shared_blank_step", fake_complete):
                        with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                                with patch.object(
                                    intentHandler,
                                    "_resolve_uvvis_shared_blank_dirs",
                                    lambda _conn: [Path(temp_dir)],
                                ):
                                    handled = await intentHandler._handle_uvvis_shared_blank_prep_v2(
                                        conn,
                                        "开始扫描",
                                        "开始扫描",
                                        {},
                                    )

        self.assertTrue(handled)
        self.assertEqual([], executed)
        self.assertEqual(
            [
                {
                    "observations": "当前批次纯水空白已确认可复用，后续测量将保留或重新放好参比位纯水比色皿。",
                    "fallback_reply": "当前批次纯水空白可复用，接下来装入样品比色皿。",
                }
            ],
            completed,
        )
        self.assertEqual(
            ["当前批次纯水空白可复用，接下来装入样品比色皿。"],
            spoken,
        )

    async def test_blank_prep_runs_shared_prep_before_prompting_for_pure_water(self):
        conn = _FakeConn()
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_start_turn(_conn, _text):
            return None

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True, "phase": "shared_prep_ready", "liquid_blank_exists": False}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_start_direct_intent_turn", fake_start_turn):
                with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        with patch.object(
                            intentHandler,
                            "_uvvis_shared_liquid_blank_exists",
                            lambda _conn, payload=None: False,
                        ):
                            handled = await intentHandler._handle_uvvis_shared_blank_prep_v2(
                                conn,
                                "开始扫描",
                                "开始扫描",
                                {},
                            )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": False,
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                "phase": "await_pure_water_blank",
                "session_key": "lease-1",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(
            [
                "共享前置校正已经准备好。请在 1 到 5 号样品位和参比位各放 1 支纯水比色皿，共 6 支，放好了告诉我可以开始扫描。"
            ],
            spoken,
        )

    async def test_blank_prep_ready_reply_triggers_pure_water_scan(self):
        conn = _FakeConn()
        spoken = []
        completed = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_start_turn(_conn, _text):
            return None

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True, "phase": "liquid_blank_ready", "liquid_blank_exists": True}

        async def fake_complete(_conn, *, observations, fallback_reply):
            completed.append(
                {
                    "observations": observations,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "纯水空白已经准备好了。"

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_start_direct_intent_turn", fake_start_turn):
                with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                    with patch.object(intentHandler, "_complete_uvvis_shared_blank_step", fake_complete):
                        with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                            handled = await intentHandler._handle_uvvis_shared_blank_prep_v2(
                                conn,
                                "放好了",
                                "放好了",
                                {"phase": "await_pure_water_blank"},
                            )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": True,
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            [
                {
                    "observations": "当前批次纯水空白已记录完成，参比位纯水比色皿可继续用于后续测量。",
                    "fallback_reply": "纯水空白已经准备好了。",
                }
            ],
            completed,
        )
        self.assertEqual(["纯水空白已经准备好了。"], spoken)

    async def test_dark_current_prep_first_prompts_for_empty_positions(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID
        spoken = []
        executed = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_start_turn(_conn, _text):
            return None

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "暗电流校正已经完成。"

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_start_direct_intent_turn", fake_start_turn):
                with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                    with patch.object(intentHandler, "_complete_experiment_step_with_fields", fake_complete):
                        with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                            handled = await intentHandler._handle_uvvis_shared_dark_air_prep(
                                conn,
                                "开始扫描",
                                "开始扫描",
                            )

        self.assertTrue(handled)
        self.assertEqual([], executed)
        self.assertEqual([], completed)
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
                "phase": "await_empty_positions",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(
            [
                "先检查1到5号样品位都为空，参比位也不要放任何液体。都空了就告诉我。可以开始时直接说“开始扫描”。",
            ],
            spoken,
        )
