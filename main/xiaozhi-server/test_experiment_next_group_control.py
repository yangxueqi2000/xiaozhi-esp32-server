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


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.dialogue = _FakeDialogue()
        self.client_abort = False
        self.sentence_id = None
        self.experiment_session_id = "exp-1"
        self.experiment_current_step_id = "step_6_kinetics_combined_measurement"
        self.experiment_current_step = None
        self.experiment_progress_summary = None
        self.config = {"experiment_fast_path_enabled": False}
        self.intent_type = "function_call"
        self.tts = None

    def enrich_latest_clean_user_utterance_snapshot(self):
        return None


class ExperimentNextGroupControlTest(unittest.IsolatedAsyncioTestCase):
    def test_next_group_phrases_are_advance_controls(self):
        conn = _FakeConn()

        for text in (
            "\u4e0b\u4e00\u7ec4",
            "\u4e0b\u4e00\u7ec4\u7ee7\u7eed",
            "\u7ee7\u7eed\u4e0b\u4e00\u7ec4",
            "\u6362\u4e0b\u4e00\u7ec4",
            "\u4e0b\u4e00\u6279",
        ):
            with self.subTest(text=text):
                action = intentHandler._classify_short_experiment_control(conn, text)
                self.assertEqual("advance", action)

    def test_next_group_questions_do_not_advance(self):
        conn = _FakeConn()

        for text in (
            "\u4e0b\u4e00\u7ec4\u662f\u4ec0\u4e48",
            "\u4e0b\u4e00\u7ec4\u8981\u505a\u4ec0\u4e48",
            "\u4e0b\u4e00\u6b65\u662f\u4ec0\u4e48",
        ):
            with self.subTest(text=text):
                action = intentHandler._classify_short_experiment_control(conn, text)
                self.assertEqual("", action)

    async def test_next_group_strict_graph_path_advances_when_fast_path_disabled(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        advanced = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_apply_current_confirmation_report(_conn, filtered_text):
            return ""

        async def fake_advance(_conn, session_id):
            advanced.append(session_id)
            return (
                "\u4e0b\u4e00\u7ec4\u4ece\u4e00\u5230\u56db\u53f7"
                "\u6837\u54c1\u88c5\u6837\u5f00\u59cb\u3002"
            )

        text = "\u4e0b\u4e00\u7ec4\u7ee7\u7eed"

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                with patch.object(
                    intentHandler,
                    "_try_apply_current_confirmation_report",
                    fake_apply_current_confirmation_report,
                ):
                    with patch.object(
                        intentHandler,
                        "_advance_experiment_step_fast",
                        fake_advance,
                    ):
                        handled = (
                            await intentHandler.handle_experiment_control_strict_graph_intent(
                                conn,
                                text,
                                text,
                            )
                        )

        self.assertTrue(handled)
        self.assertEqual([text], sent)
        self.assertEqual(["exp-1"], advanced)
        self.assertEqual(1, len(spoken))


if __name__ == "__main__":
    unittest.main()
