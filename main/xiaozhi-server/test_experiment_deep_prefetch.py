import asyncio
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))
fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)


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

fake_hello_module = types.ModuleType("core.handle.helloHandle")


async def _fake_check_wakeup_words(conn, filtered_text):
    return False


fake_hello_module.checkWakeupWords = _fake_check_wakeup_words
fake_hello_module.handleHelloMessage = lambda *args, **kwargs: None
sys.modules.setdefault("core.handle.helloHandle", fake_hello_module)

fake_send_audio_module = types.ModuleType("core.handle.sendAudioHandle")
fake_send_audio_module.send_stt_message = lambda *args, **kwargs: None
fake_send_audio_module.send_tts_message = lambda *args, **kwargs: None
fake_send_audio_module.sendAudioMessage = lambda *args, **kwargs: None
fake_send_audio_module.SentenceType = types.SimpleNamespace(
    FIRST="FIRST",
    LAST="LAST",
)
sys.modules.setdefault("core.handle.sendAudioHandle", fake_send_audio_module)

fake_device_mcp_module = types.ModuleType("core.providers.tools.device_mcp")
fake_device_mcp_module.call_mcp_tool = lambda *args, **kwargs: None
fake_device_mcp_module.handle_mcp_message = lambda *args, **kwargs: None
fake_device_mcp_module.DeviceMCPExecutor = object
sys.modules.setdefault("core.providers.tools.device_mcp", fake_device_mcp_module)
sys.modules.setdefault("portalocker", types.ModuleType("portalocker"))

fake_loadplugins_module = types.ModuleType("plugins_func.loadplugins")
fake_loadplugins_module.auto_import_modules = lambda *args, **kwargs: None
sys.modules.setdefault("plugins_func.loadplugins", fake_loadplugins_module)

fake_register_module = types.ModuleType("plugins_func.register")
fake_register_module.Action = object
fake_register_module.ActionResponse = object
fake_register_module.all_function_registry = {}
sys.modules.setdefault("plugins_func.register", fake_register_module)

from core.connection import ConnectionHandler


class ExperimentDeepPrefetchTest(unittest.IsolatedAsyncioTestCase):
    def _make_handler(self) -> ConnectionHandler:
        handler = object.__new__(ConnectionHandler)
        handler.logger = _FakeLogger()
        handler.device_id = "device-1"
        handler.config = {"codex_app": {"prewarm_on_hello": True}}
        handler.experiment_prewarm_session_adopted = False
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_prewarm_trigger = "hello"
        handler.experiment_prewarm_error = ""
        handler.experiment_yaml_path = "C:/demo/experiments.yaml"
        handler.experiment_session_id = "exp-1"
        handler.experiment_current_step_id = "step_prepare"
        handler.experiment_overview = None
        handler.experiment_current_step = None
        handler.experiment_resume_recovery_required = False
        handler.experiment_resume_recovery_source = ""
        handler.experiment_resume_previous_session_id = ""
        handler.experiment_resume_reason = ""
        handler.experiment_resume_log_path = ""
        handler.experiment_resume_turn_count = ""
        handler.experiment_resume_latest_session_id = ""
        handler.experiment_resume_latest_current_step_id = ""
        handler.experiment_resume_context_excerpt = ""
        handler.experiment_deep_prefetch_task = None
        handler.experiment_deep_prefetch_lock = None
        handler.experiment_deep_prefetch_status = "idle"
        handler.experiment_deep_prefetch_error = ""
        handler.experiment_deep_prefetch_focus = ""
        handler.experiment_deep_prefetch_query = ""
        handler.experiment_deep_prefetch_started_at = 0.0
        handler.experiment_deep_prefetch_completed_at = 0.0
        handler.experiment_graph_priority_lock = None
        handler.experiment_graph_background_resume_event = None
        handler.experiment_graph_foreground_active = 0
        handler.experiment_list_steps = None
        handler.experiment_schema = None
        handler.experiment_reference = None
        handler.experiment_reference_query = ""
        return handler

    async def test_wait_returns_ready_when_micro_budget_is_enough(self):
        handler = self._make_handler()

        async def fake_call(tool_name, arguments, priority="foreground"):
            await asyncio.sleep(0.005)
            return {"tool": tool_name, "arguments": arguments, "priority": priority}

        handler._call_experiment_graph_tool = fake_call

        context = await handler.wait_for_experiment_deep_prefetch(
            "\u628a\u540e\u7eed\u6b65\u9aa4\u548c\u5b57\u6bb5\u5b9a\u4e49\u8bf4\u4e00\u4e0b",
            0.05,
        )

        self.assertEqual("ready", context["experiment_deep_prefetch_wait_result"])
        self.assertEqual("ready", context["experiment_deep_prefetch_status"])
        self.assertEqual("workflow,schema", context["experiment_deep_prefetch_focus"])
        self.assertIn("list_steps", context["experiment_list_steps_summary"])
        self.assertIn("get_schema", context["experiment_schema_summary"])

    async def test_timeout_keeps_background_prefetch_running_for_later_turns(self):
        handler = self._make_handler()

        async def fake_call(tool_name, arguments, priority="foreground"):
            await asyncio.sleep(0.05)
            return {"tool": tool_name, "arguments": arguments, "priority": priority}

        handler._call_experiment_graph_tool = fake_call

        context = await handler.wait_for_experiment_deep_prefetch(
            "\u8bb2\u4e00\u4e0b\u8fd9\u4e00\u6b65\u7684\u53cd\u5e94\u539f\u7406",
            0.001,
        )

        self.assertEqual("timeout", context["experiment_deep_prefetch_wait_result"])
        self.assertEqual("running", context["experiment_deep_prefetch_status"])
        task = handler.experiment_deep_prefetch_task
        self.assertIsNotNone(task)
        await task

        later_context = handler._experiment_prewarm_route_context()
        self.assertEqual("ready", later_context["experiment_deep_prefetch_status"])
        self.assertIn(
            "search_experiment_reference",
            later_context["experiment_reference_summary"],
        )

    async def test_theory_turn_uses_local_sidecar_before_mcp_search(self):
        handler = self._make_handler()
        called_tools = []

        async def fake_call(tool_name, arguments, priority="foreground"):
            called_tools.append(tool_name)
            self.fail(f"unexpected MCP tool call: {tool_name}")

        handler._call_experiment_graph_tool = fake_call
        handler._load_experiment_reference_sidecar = lambda query: {
            "source": "local_static_sidecar",
            "query": "\u539f\u7406",
            "title": "Ag NPs",
            "principle_notes": ["LSPR"],
            "matched_sections": ["principle_notes"],
        }

        context = await handler.wait_for_experiment_deep_prefetch(
            "\u8bb2\u4e00\u4e0b\u8fd9\u4e00\u6b65\u7684\u53cd\u5e94\u539f\u7406",
            0.05,
        )

        self.assertEqual("ready", context["experiment_deep_prefetch_wait_result"])
        self.assertEqual("ready", context["experiment_deep_prefetch_status"])
        self.assertIn("local_static_sidecar", context["experiment_reference_summary"])
        self.assertNotIn(
            "search_experiment_reference",
            context["experiment_reference_summary"],
        )
        self.assertEqual([], called_tools)

    def test_sidecar_loader_keeps_safety_only_payload_focused(self):
        handler = self._make_handler()
        sample_payload = {
            "title": "Ag NPs",
            "description": "General description that should stay out of safety-only prompts.",
            "principle_notes": ["Theory note 1", "Theory note 2"],
            "safety_notes": ["Safety note 1", "Safety note 2", "Safety note 3", "Safety note 4"],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "experiments.yaml").write_text("name: demo\n", encoding="utf-8")
            (tmp_path / "experiments.json").write_text(
                json.dumps(sample_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            handler.experiment_yaml_path = str(tmp_path / "experiments.yaml")

            payload = handler._load_experiment_reference_sidecar("注意事项")

        self.assertIsNotNone(payload)
        self.assertEqual("注意事项", payload["query"])
        self.assertEqual(["safety_notes"], payload["matched_sections"])
        self.assertIn("safety_notes", payload)
        self.assertNotIn("principle_notes", payload)
        self.assertNotIn("description", payload)
        self.assertEqual(3, len(payload["safety_notes"]))

    def test_sidecar_loader_keeps_theory_only_context_richer(self):
        handler = self._make_handler()
        sample_payload = {
            "title": "Ag NPs",
            "description": "Theory description that should stay available for theory-only prompts.",
            "principle_notes": ["Theory note 1", "Theory note 2"],
            "safety_notes": ["Safety note 1"],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "experiments.yaml").write_text("name: demo\n", encoding="utf-8")
            (tmp_path / "experiments.json").write_text(
                json.dumps(sample_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            handler.experiment_yaml_path = str(tmp_path / "experiments.yaml")

            payload = handler._load_experiment_reference_sidecar("原理")

        self.assertIsNotNone(payload)
        self.assertEqual("原理", payload["query"])
        self.assertEqual(["principle_notes"], payload["matched_sections"])
        self.assertIn("description", payload)
        self.assertIn("principle_notes", payload)
        self.assertNotIn("safety_notes", payload)

    async def test_background_deep_warming_loads_common_context_without_user_gap(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "deep_warming"
        handler.experiment_prewarm_ready_level = "minimal_ready"
        handler.experiment_prewarm_trigger = "hello"

        async def fake_call(tool_name, arguments, priority="foreground"):
            await asyncio.sleep(0)
            return {"tool": tool_name, "arguments": arguments, "priority": priority}

        handler._call_experiment_graph_tool = fake_call

        await handler._run_experiment_prewarm_deep_stage(
            session_id="exp-1",
            progress_summary_payload={"tool": "get_progress_summary"},
        )

        self.assertEqual("completed", handler.experiment_prewarm_status)
        self.assertEqual("completed", handler.experiment_prewarm_ready_level)
        self.assertEqual("get_overview", handler.experiment_overview["tool"])
        self.assertEqual("get_step", handler.experiment_current_step["tool"])
        self.assertEqual("list_steps", handler.experiment_list_steps["tool"])
        self.assertEqual("get_schema", handler.experiment_schema["tool"])
        self.assertIsNone(handler.experiment_reference)

    async def test_prewarm_wait_budget_is_maximum_not_fixed_delay(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "completed"
        handler.experiment_prewarm_ready_level = "completed"
        handler.experiment_prewarm_started_at = time.time() - 0.02
        handler.experiment_prewarm_minimal_ready_at = time.time() - 0.01

        started = time.perf_counter()
        context = await handler.wait_for_experiment_prewarm_for_real_user_turn(3.0)
        elapsed = time.perf_counter() - started

        self.assertEqual("ready", context["experiment_prewarm_wait_result"])
        self.assertLess(elapsed, 0.2)

    async def test_prewarm_wait_timeout_respects_budget_when_not_ready(self):
        handler = self._make_handler()
        handler.experiment_prewarm_status = "minimal_warming"
        handler.experiment_prewarm_ready_level = "none"
        handler.experiment_session_id = ""
        handler.experiment_current_step_id = ""
        handler.experiment_prewarm_started_at = time.time()
        handler.experiment_prewarm_minimal_ready_at = 0.0
        handler.experiment_prewarm_completed_at = 0.0
        handler.experiment_prewarm_minimal_ready_event = asyncio.Event()
        handler.experiment_prewarm_task = asyncio.create_task(asyncio.sleep(0.05))

        started = time.perf_counter()
        try:
            context = await handler.wait_for_experiment_prewarm_for_real_user_turn(0.01)
        finally:
            await handler.experiment_prewarm_task
        elapsed = time.perf_counter() - started

        self.assertEqual("timeout", context["experiment_prewarm_wait_result"])
        self.assertGreaterEqual(elapsed, 0.005)
        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
