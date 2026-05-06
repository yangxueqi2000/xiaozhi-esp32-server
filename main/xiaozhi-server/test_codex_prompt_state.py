import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import MethodType


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

from core.providers.llm.codex.codex import (
    _CodexSession,
    _decode_stderr_line,
    _experiment_prompt_block,
    _load_codex_mcp_config_overrides,
    _recoverable_stderr_reason,
    _should_suppress_stderr_warning,
)


class CodexPromptStateTest(unittest.TestCase):
    def _make_session(self) -> _CodexSession:
        session = _CodexSession(
            {
                "codex_bin": "codex.cmd",
                "model_name": "gpt-5.3-codex",
                "workspace": str(SCRIPT_DIR),
                "system_prompt_mode": "first_turn",
                "bootstrap_mode": "none",
                "auto_approve": True,
                "network_access": True,
                "export_api_key": False,
            },
            "test-session",
        )
        session._captured_prompts = []
        session._fake_started = False

        def fake_start(self):
            if self._fake_started:
                return
            self._fake_started = True
            self.proc = object()
            self.thread_id = "thread-1"
            self._system_prompt_sent = False
            self._bootstrap_history = True

        def fake_stream_turn(self, prompt_text, emit_events, user_text=None, **kwargs):
            self.start()
            self._captured_prompts.append(
                {
                    "prompt_text": prompt_text,
                    "user_text": user_text,
                    "system_prompt_sent": self._system_prompt_sent,
                }
            )
            yield "ok"

        session.start = MethodType(fake_start, session)
        session._stream_turn = MethodType(fake_stream_turn, session)
        return session

    def _first_prompt_with_experiment_context(self, user_text: str, **kwargs) -> str:
        session = self._make_session()
        dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": user_text},
        ]
        list(session.stream_response(dialogue, **kwargs))
        return session._captured_prompts[0]["prompt_text"]

    def test_first_turn_only_sends_system_prompt_once(self):
        session = self._make_session()

        first_dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ]
        list(session.stream_response(first_dialogue))

        self.assertTrue(session._system_prompt_sent)
        self.assertEqual("SYS\n\nhello", session._captured_prompts[0]["prompt_text"])
        self.assertTrue(session._captured_prompts[0]["system_prompt_sent"])

        second_dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
        ]
        list(session.stream_response(second_dialogue))

        self.assertEqual("again", session._captured_prompts[1]["prompt_text"])
        self.assertEqual("again", session._captured_prompts[1]["user_text"])

    def test_wham_transport_request_failure_is_suppressed_as_recoverable_noise(self):
        warning_text = (
            "2026-05-05T07:35:50.838000Z ERROR rmcp::transport::worker: "
            "worker quit with fatal: Transport channel closed, when "
            "Client(HttpRequest(HttpRequest(\"http/request failed: error sending "
            "request for url (https://chatgpt.com/backend-api/wham/apps)\")))"
        )

        self.assertTrue(_should_suppress_stderr_warning(warning_text))
        self.assertEqual("wham_transport_eof", _recoverable_stderr_reason(warning_text))

    def test_windows_localized_process_cleanup_warning_is_suppressed(self):
        warning_text = '错误: 没有找到进程 "14392"。'

        self.assertTrue(_should_suppress_stderr_warning(warning_text))
        self.assertIsNone(_recoverable_stderr_reason(warning_text))

    def test_windows_cp936_stderr_decodes_cleanly(self):
        raw_line = '错误: 没有找到进程 "14392"。\r\n'.encode("gbk")

        self.assertEqual('错误: 没有找到进程 "14392"。\r\n', _decode_stderr_line(raw_line))

    def test_powershell_profile_execution_policy_warning_is_suppressed(self):
        warning_text = (
            ". : Cannot load file "
            "C:\\Users\\11979\\Documents\\WindowsPowerShell\\profile.ps1 "
            "because running scripts is disabled on this system.\n"
            "    + CategoryInfo          : SecurityError: (:) [], PSSecurityException\n"
            "    + FullyQualifiedErrorId : UnauthorizedAccess"
        )

        self.assertTrue(_should_suppress_stderr_warning(warning_text))

    def test_load_codex_mcp_config_overrides_reads_stdio_env_from_settings_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / ".mcp_server_settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "experiment-graph": {
                                "command": "C:/Python/python.exe",
                                "args": ["C:/repo/experiment_graph_mcp_server.py"],
                                "env": {
                                    "EXPERIMENT_YAML_PATH": "C:/repo/experiments.yaml",
                                    "PYTHONUTF8": "1",
                                },
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            overrides = _load_codex_mcp_config_overrides(settings_path)

        self.assertIn(
            'mcp_servers.experiment-graph.command="C:/Python/python.exe"',
            overrides,
        )
        self.assertIn(
            'mcp_servers.experiment-graph.args=["C:/repo/experiment_graph_mcp_server.py"]',
            overrides,
        )
        env_override = next(
            item
            for item in overrides
            if item.startswith("mcp_servers.experiment-graph.env=")
        )
        self.assertIn('EXPERIMENT_YAML_PATH = "C:/repo/experiments.yaml"', env_override)
        self.assertIn('PYTHONUTF8 = "1"', env_override)

    def test_timeout_without_current_step_uses_operation_template(self):
        prompt_text = self._first_prompt_with_experiment_context(
            "\u6211\u73b0\u5728\u4e0b\u4e00\u6b65\u8be5\u505a\u4ec0\u4e48",
            experiment_prewarm_wait_result="timeout",
            experiment_prewarm_status="minimal_warming",
            experiment_prewarm_ready_level="none",
            experiment_yaml_path="C:/demo/experiments.yaml",
        )
        self.assertIn(
            "Timeout first-turn template: operation or next-step guidance.",
            prompt_text,
        )
        self.assertNotIn("Timeout first-turn template: recording intake.", prompt_text)
        self.assertNotIn(
            "Timeout first-turn template: theory or full-workflow request.",
            prompt_text,
        )

    def test_timeout_without_current_step_uses_record_template(self):
        prompt_text = self._first_prompt_with_experiment_context(
            "\u5e2e\u6211\u8bb0\u5f55\u4e00\u4e0b\u6e29\u5ea628\u5ea6",
            experiment_prewarm_wait_result="timeout",
            experiment_prewarm_status="minimal_warming",
            experiment_prewarm_ready_level="none",
            experiment_yaml_path="C:/demo/experiments.yaml",
        )
        self.assertIn(
            "Timeout first-turn template: recording intake.",
            prompt_text,
        )
        self.assertIn(
            "continue directly with the recording flow instead of detouring into theory or later steps.",
            prompt_text,
        )

    def test_timeout_without_current_step_uses_theory_template(self):
        prompt_text = self._first_prompt_with_experiment_context(
            "\u8bb2\u4e00\u4e0b\u8fd9\u4e00\u6b65\u7684\u53cd\u5e94\u539f\u7406",
            experiment_prewarm_wait_result="timeout",
            experiment_prewarm_status="minimal_warming",
            experiment_prewarm_ready_level="none",
            experiment_yaml_path="C:/demo/experiments.yaml",
        )
        self.assertIn(
            "Timeout first-turn template: theory or full-workflow request.",
            prompt_text,
        )
        self.assertIn(
            "ask one short current-state question before expanding.",
            prompt_text,
        )

    def test_prompt_includes_deep_prefetch_detail_block(self):
        prompt_text = self._first_prompt_with_experiment_context(
            "\u628a\u540e\u7eed\u6b65\u9aa4\u548c\u5b57\u6bb5\u5b9a\u4e49\u8bf4\u4e00\u4e0b",
            experiment_prewarm_wait_result="ready",
            experiment_prewarm_status="completed",
            experiment_prewarm_ready_level="completed",
            experiment_session_id="exp-1",
            experiment_current_step_id="step_prepare",
            experiment_deep_prefetch_wait_result="ready",
            experiment_deep_prefetch_status="ready",
            experiment_deep_prefetch_focus="workflow,schema",
            experiment_deep_prefetch_query="\u628a\u540e\u7eed\u6b65\u9aa4\u548c\u5b57\u6bb5\u5b9a\u4e49\u8bf4\u4e00\u4e0b",
            experiment_list_steps_summary='{"tool":"list_steps"}',
            experiment_schema_summary='{"tool":"get_schema"}',
        )
        self.assertIn(
            "Deep-prefetched experiment detail context from server (trusted):",
            prompt_text,
        )
        self.assertIn(
            "Use this deep-prefetched detail context first before calling list_steps",
            prompt_text,
        )

    def test_experiment_prompt_block_includes_graph_alignment_and_uvvis_execution_guards(self):
        prompt_text = _experiment_prompt_block(
            {
                "experiment_prewarm_wait_result": "ready",
                "experiment_prewarm_status": "completed",
                "experiment_prewarm_ready_level": "completed",
                "experiment_session_id": "exp-1",
                "experiment_current_step_id": "step_prepare_setup_all",
                "experiment_current_step_summary": '{"step":"prepare"}',
            },
            "\u5f00\u59cb\u626b\u63cf",
        )

        self.assertIn("Experiment graph alignment guard:", prompt_text)
        self.assertIn(
            "Do not verbally move the student to a later experiment step unless the current turn actually called experiment_graph state/flow tools",
            prompt_text,
        )
        self.assertIn(
            "Do not narrate backend bookkeeping such as '我先记下…'",
            prompt_text,
        )
        self.assertIn("MCP execution guard:", prompt_text)
        self.assertIn("Do not launch local MCP server scripts", prompt_text)
        self.assertIn("call the connected MCP tools directly instead", prompt_text)
        self.assertIn("UV-Vis execution guard:", prompt_text)
        self.assertIn("uvvis_prepare_dark_current", prompt_text)
        self.assertIn("inspect the shared uv_data_common directory", prompt_text)
        self.assertIn("uvvis_measure_kinetics", prompt_text)
        self.assertIn("Do not verbalize internal orchestration rules", prompt_text)

    def test_recent_photo_confirmation_context_is_included_on_later_turn(self):
        session = self._make_session()

        first_dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ]
        list(session.stream_response(first_dialogue))

        second_dialogue = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "\u53ef\u4ee5\u62cd\u7167"},
            {"role": "assistant", "content": "\u62cd\u597d\u4e86\uff0c\u5df2\u7ecf\u4fdd\u5b58\u3002"},
            {"role": "user", "content": "\u7ee7\u7eed\u4e0b\u4e00\u6b65"},
        ]
        list(
            session.stream_response(
                second_dialogue,
                experiment_prewarm_wait_result="ready",
                experiment_prewarm_status="completed",
                experiment_prewarm_ready_level="completed",
                experiment_session_id="exp-1",
                experiment_current_step_id="step_photo_confirm_sample_1",
                experiment_recent_photo_confirmation_summary=(
                    "Recent server photo confirmation succeeded about 3 seconds ago "
                    "for sample 1. The saved file name was sample1.png. Unless the "
                    "user explicitly wants a retake, do not ask to retake the same "
                    "sample photo again."
                ),
                experiment_recent_photo_graph_advanced="true",
                experiment_recent_photo_sample_name="\u4e00\u53f7\u6837\u54c1",
                experiment_recent_photo_next_step_id="step_sample_2_prepare",
                experiment_recent_photo_next_step_title="\u4e8c\u53f7\u6837\u54c1\uff1a\u540e\u7eed\u64cd\u4f5c",
            )
        )

        prompt_text = session._captured_prompts[1]["prompt_text"]
        self.assertIn("Recent trusted photo confirmation context from server:", prompt_text)
        self.assertIn("experiment_recent_photo_graph_advanced: true", prompt_text)
        self.assertIn(
            "Do not ask to retake the same sample photo unless the user explicitly asks for a retake",
            prompt_text,
        )

    def test_stale_vscode_extension_codex_bin_auto_discovers_newer_binary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "workspace"
            workspace.mkdir()

            stale_bin = (
                root
                / ".vscode"
                / "extensions"
                / "openai.chatgpt-26.422.21459-win32-x64"
                / "bin"
                / "windows-x86_64"
                / "codex.exe"
            )
            live_bin = (
                root
                / ".vscode"
                / "extensions"
                / "openai.chatgpt-26.422.62136-win32-x64"
                / "bin"
                / "windows-x86_64"
                / "codex.exe"
            )
            live_bin.parent.mkdir(parents=True, exist_ok=True)
            live_bin.write_text("", encoding="utf-8")

            session = _CodexSession(
                {
                    "codex_bin": str(stale_bin),
                    "model_name": "gpt-5.4",
                    "workspace": str(workspace),
                    "system_prompt_mode": "first_turn",
                    "bootstrap_mode": "none",
                    "auto_approve": True,
                    "network_access": True,
                    "export_api_key": False,
                },
                "test-session",
            )

        self.assertEqual(str(live_bin), session.codex_bin)


if __name__ == "__main__":
    unittest.main()
