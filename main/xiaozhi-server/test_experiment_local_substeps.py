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
from core.utils import textUtils
from core.utils.dialogue import Message
from core.session import (
    load_experiment_session_binding,
    save_experiment_session_binding,
)
from core.utils import experiment_resume


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.dialogue = _FakeDialogue()
        self.client_abort = False
        self.sentence_id = None
        self.chat_session_id = ""
        self.model_session_key = ""
        self.user_id = ""
        self.device_id = ""
        self.prompt = ""
        self.llm = None
        self.experiment_session_id = ""
        self.experiment_current_step_id = "step_01_alkaline_analysis"
        self.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_01_alkaline_analysis",
                    "title": "工业碱总碱度的分析",
                    "prompts": {
                        "instruction": "请按照以下细分操作完成：工业碱总碱度的分析",
                        "safety": ["HCl具有腐蚀性。"],
                    },
                }
            }
        }
        self.experiment_progress_summary = None
        self.intent_type = "function_call"
        self.config = {
            "experiment_fast_path_enabled": True,
            "experiment_opening_fast_path_enabled": True,
        }
        self.tts = None
        self.enriched = False
        self.cmd_exit = []
        self.experiment_resume_recovery_required = False
        self.experiment_resume_latest_current_step_id = ""
        self.experiment_resume_log_path = ""
        self.experiment_resume_turn_count = "0"
        self.experiment_resume_latest_session_id = ""
        self._experiment_yaml_steps_cache = [
            {
                "id": "step_01_alkaline_analysis",
                "title": "工业碱总碱度的分析",
                "prompts": {
                    "instruction": "请按照以下细分操作完成：工业碱总碱度的分析",
                    "safety": ["HCl具有腐蚀性。"],
                    "tips": ["记录原始数据。"],
                },
                "substeps": [
                    {
                        "step_id": "step_01_alkaline_analysis_s01",
                        "instruction": "先移取25 mL 2 mol/L HCl至500 mL容量瓶中。",
                        "description": "先移取25 mL 2 mol/L HCl至500 mL容量瓶中。",
                    },
                    {
                        "step_id": "step_01_alkaline_analysis_s02",
                        "instruction": "加水稀释至500 mL刻度，配成约0.1 mol/L HCl溶液。",
                        "description": "加水稀释至500 mL刻度，配成约0.1 mol/L HCl溶液。",
                    },
                ],
            }
        ]

    def enrich_latest_clean_user_utterance_snapshot(self):
        self.enriched = True

    def _extract_experiment_current_step_id(self, *payloads):
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            body = payload.get("result", payload)
            if not isinstance(body, dict):
                continue
            step = body.get("step")
            if isinstance(step, dict):
                step_id = str(step.get("id", "")).strip()
                if step_id:
                    return step_id
        return ""


class ExperimentLocalSubstepTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_session_advance_uses_next_yaml_substep(self):
        conn = _FakeConn()
        spoken = []
        sent = []

        base_meta = intentHandler._load_fallback_experiment_step_meta_from_yaml(
            conn,
            step_id=conn.experiment_current_step_id,
            prefer_first_substep=True,
        )
        current_meta = intentHandler._resolve_local_experiment_substep_view(
            conn,
            base_meta,
        )["step_meta"]
        self.assertIn("先移取25 mL", current_meta["instruction"])

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "做好了",
                    "做好了",
                )

        self.assertTrue(handled)
        self.assertEqual(["做好了"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("加水稀释至500 mL刻度", spoken[0])
        self.assertNotIn("请按照以下细分操作完成", spoken[0])
        self.assertEqual(
            "step_01_alkaline_analysis",
            getattr(conn, "_experiment_local_substep_step_id", ""),
        )
        self.assertEqual(1, getattr(conn, "_experiment_local_substep_index", -1))

    async def test_binding_roundtrip_preserves_local_substep_state(self):
        with TemporaryDirectory() as temp_dir:
            config = {
                "experiment_session_registry": {
                    "local_store": str(Path(temp_dir) / "registry.json"),
                }
            }

            await save_experiment_session_binding(
                config,
                chat_session_id="chat-1",
                yaml_path="C:\\demo\\experiments.yaml",
                experiment_session_id="exp-1",
                current_step_id="step_01_alkaline_analysis",
                local_substep_step_id="step_01_alkaline_analysis",
                local_substep_index="12",
                status="active",
            )

            loaded = await load_experiment_session_binding(
                config,
                "chat-1",
                "C:\\demo\\experiments.yaml",
            )

            self.assertIsNotNone(loaded)
            self.assertEqual(
                "step_01_alkaline_analysis",
                loaded.get("local_substep_step_id", ""),
            )
            self.assertEqual("12", loaded.get("local_substep_index", ""))

    def test_build_resume_context_reads_local_substep_from_user_log(self):
        with TemporaryDirectory() as temp_dir:
            config = {
                "LLM": {
                    "codex_app_server": {
                        "type": "codex",
                        "stream_log_path": str(Path(temp_dir) / "{device_id}.log"),
                    }
                }
            }
            device_id = "94:a9:90:28:eb:58"

            log_path = experiment_resume.append_user_utterance_log(
                config,
                device_id,
                "准备好了",
                source="listen_text",
                experiment_session_id="exp-1",
                current_step_id="step_01_alkaline_analysis",
                experiment_yaml_path="C:\\demo\\experiments.yaml",
                local_substep_step_id="step_01_alkaline_analysis",
                local_substep_index="12",
            )

            self.assertTrue(log_path)

            resume_context = experiment_resume.build_resume_context(
                config,
                device_id,
            )

            self.assertIsNotNone(resume_context)
            self.assertEqual(
                "step_01_alkaline_analysis",
                resume_context.get("latest_local_substep_step_id", ""),
            )
            self.assertEqual("12", resume_context.get("latest_local_substep_index", ""))

    def test_graph_alignment_guard_does_not_override_local_substep_guidance(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_01_alkaline_analysis"
        conn._experiment_local_substep_step_id = "step_01_alkaline_analysis"
        conn._experiment_local_substep_index = 1
        conn.dialogue.put(Message(role="user", content="继续下一步"))

        prepared = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "现在做这一步：加水稀释至500 mL刻度，配成约0.1 mol/L HCl溶液。做好后告诉我。",
        )

        self.assertIn("加水稀释至500 mL刻度", prepared)
        self.assertNotIn("请按照以下细分操作完成", prepared)


    def test_fast_path_survives_with_only_local_substep_state(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = ""
        conn.experiment_current_step = None
        conn._experiment_local_substep_step_id = "step_01_alkaline_analysis"
        conn._experiment_local_substep_index = 1

        self.assertTrue(intentHandler._is_experiment_fast_path_available(conn))
        step_meta = intentHandler._load_current_local_experiment_substep_meta_from_yaml(
            conn
        )
        self.assertTrue(step_meta.get("is_local_substep"))
        self.assertEqual(2, step_meta.get("substep_index"))
        self.assertIn("0.1 mol/L HCl", step_meta.get("instruction", ""))


if __name__ == "__main__":
    unittest.main()
