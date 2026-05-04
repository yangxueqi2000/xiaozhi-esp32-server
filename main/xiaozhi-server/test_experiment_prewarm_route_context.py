import sys
import time
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
fake_logger_module.build_module_string = lambda *args, **kwargs: ""
fake_logger_module.create_connection_logger = lambda *args, **kwargs: _FakeLogger()
sys.modules.setdefault("config.logger", fake_logger_module)
sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))
sys.modules.setdefault("portalocker", types.ModuleType("portalocker"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)

fake_hello_module = types.ModuleType("core.handle.helloHandle")


async def _fake_check_wakeup_words(conn, filtered_text):
    return False


fake_hello_module.checkWakeupWords = _fake_check_wakeup_words
fake_hello_module.handleHelloMessage = lambda *args, **kwargs: None
sys.modules.setdefault("core.handle.helloHandle", fake_hello_module)

fake_device_mcp_module = types.ModuleType("core.providers.tools.device_mcp")
fake_device_mcp_module.call_mcp_tool = lambda *args, **kwargs: None
fake_device_mcp_module.handle_mcp_message = lambda *args, **kwargs: None
fake_device_mcp_module.DeviceMCPExecutor = object
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

fake_loadplugins_module = types.ModuleType("plugins_func.loadplugins")
fake_loadplugins_module.auto_import_modules = lambda *args, **kwargs: None
sys.modules.setdefault("plugins_func.loadplugins", fake_loadplugins_module)

from core.connection import ConnectionHandler


class ExperimentPrewarmRouteContextTest(unittest.TestCase):
    def _make_handler(self) -> ConnectionHandler:
        handler = object.__new__(ConnectionHandler)
        handler.experiment_prewarm_session_adopted = False
        handler.experiment_prewarm_status = "idle"
        handler.experiment_prewarm_ready_level = "none"
        handler.experiment_prewarm_trigger = ""
        handler.experiment_prewarm_error = ""
        handler.experiment_deep_prefetch_status = "idle"
        handler.experiment_deep_prefetch_error = ""
        handler.experiment_deep_prefetch_focus = ""
        handler.experiment_deep_prefetch_query = ""
        handler.experiment_yaml_path = ""
        handler.experiment_session_id = ""
        handler.experiment_current_step_id = ""
        handler.experiment_overview = None
        handler.experiment_current_step = None
        handler.experiment_progress_summary = None
        handler.experiment_list_steps = None
        handler.experiment_schema = None
        handler.experiment_reference = None
        handler.experiment_resume_recovery_required = False
        handler.experiment_resume_recovery_source = ""
        handler.experiment_resume_previous_session_id = ""
        handler.experiment_resume_reason = ""
        handler.experiment_resume_log_path = ""
        handler.experiment_resume_turn_count = ""
        handler.experiment_resume_latest_session_id = ""
        handler.experiment_resume_latest_current_step_id = ""
        handler.experiment_resume_context_excerpt = ""
        return handler

    def test_incomplete_prewarm_without_wait_result_does_not_expose_context(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "minimal_warming"
        handler.experiment_session_id = "exp-1"
        handler.experiment_current_step_id = "step_prepare"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"

        context = handler._experiment_prewarm_route_context()

        self.assertEqual({}, context)

    def test_minimal_ready_context_persists_for_later_turns_after_timeout(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_prewarm_trigger = "hello"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-2"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        handler.experiment_overview = {
            "title": "Silver nanoparticle synthesis",
            "current": "prepare setup",
        }
        handler.experiment_current_step = {
            "step_id": "step_prepare_setup_all",
            "title": "Prepare all vessels and stir bars",
        }

        context = handler._experiment_prewarm_route_context()

        self.assertEqual("completed", context["experiment_prewarm_status"])
        self.assertEqual("completed", context["experiment_prewarm_ready_level"])
        self.assertEqual("hello", context["experiment_prewarm_trigger"])
        self.assertEqual("C:/demo/experiments.yaml", context["experiment_yaml_path"])
        self.assertEqual("exp-2", context["experiment_session_id"])
        self.assertEqual(
            "step_prepare_setup_all", context["experiment_current_step_id"]
        )
        self.assertIn(
            "Silver nanoparticle synthesis",
            context["experiment_overview_summary"],
        )
        self.assertIn(
            "Prepare all vessels and stir bars",
            context["experiment_current_step_summary"],
        )

    def test_ready_context_includes_deep_prefetch_summaries(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-3"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        handler.experiment_deep_prefetch_status = "ready"
        handler.experiment_deep_prefetch_focus = "workflow,schema"
        handler.experiment_deep_prefetch_query = (
            "\u628a\u540e\u7eed\u6b65\u9aa4\u548c\u5b57\u6bb5\u5b9a\u4e49\u8bf4\u4e00\u4e0b"
        )
        handler.experiment_list_steps = {"tool": "list_steps", "count": 12}
        handler.experiment_schema = {"tool": "get_schema", "fields": ["temperature"]}

        context = handler._experiment_prewarm_route_context(deep_wait_result="ready")

        self.assertEqual("ready", context["experiment_deep_prefetch_wait_result"])
        self.assertEqual("ready", context["experiment_deep_prefetch_status"])
        self.assertEqual("workflow,schema", context["experiment_deep_prefetch_focus"])
        self.assertIn("list_steps", context["experiment_list_steps_summary"])
        self.assertIn("get_schema", context["experiment_schema_summary"])

    def test_completed_background_warming_exposes_common_context_summaries(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-4"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        handler.experiment_list_steps = {"tool": "list_steps", "count": 12}
        handler.experiment_schema = {"tool": "get_schema", "fields": ["temperature"]}

        context = handler._experiment_prewarm_route_context()

        self.assertIn("list_steps", context["experiment_list_steps_summary"])
        self.assertIn("get_schema", context["experiment_schema_summary"])

    def test_recent_unadvanced_photo_confirmation_forces_foreground_refresh_once(self):
        handler = self._make_handler()
        handler.experiment_session_id = "exp-photo-1"
        handler.experiment_current_step_id = "step_prepare_setup_all"
        captured_at = time.time()
        handler._recent_server_photo_confirmation = {
            "captured_at": captured_at,
            "graph_advanced": False,
            "graph_refresh_checked_at": 0.0,
        }

        self.assertTrue(handler._should_refresh_experiment_state_before_llm())

        handler._recent_server_photo_confirmation["graph_refresh_checked_at"] = (
            captured_at + 1.0
        )
        self.assertFalse(handler._should_refresh_experiment_state_before_llm())

    def test_recent_unadvanced_photo_context_mentions_graph_not_advanced(self):
        handler = self._make_handler()
        handler._recent_server_photo_confirmation = {
            "captured_at": time.time(),
            "graph_advanced": False,
            "graph_status_reason": "redirect_did_not_land_on_photo_step",
            "current_step_id": "step_prepare_setup_all",
            "current_step_title": "Prepare all vessels and stir bars",
            "photo_meta": {"file_name": "sample1.png"},
        }

        context = handler._recent_server_photo_confirmation_route_context()

        self.assertEqual("false", context["experiment_recent_photo_graph_advanced"])
        self.assertIn(
            "not confirmed as advanced",
            context["experiment_recent_photo_confirmation_summary"],
        )
        self.assertIn(
            "Prepare all vessels and stir bars",
            context["experiment_recent_photo_confirmation_summary"],
        )


if __name__ == "__main__":
    unittest.main()
