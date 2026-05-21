import json
import sys
import tempfile
import types
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

from core.utils import experiment_resume


def test_resume_request_is_intent_but_not_passive_llm_log_context():
    text = "继续刚才的实验步骤"

    assert experiment_resume.is_resume_experiment_request(text)
    assert not experiment_resume.should_load_device_log_context(text)


def test_export_request_can_still_load_device_log_context():
    assert experiment_resume.should_load_device_log_context("生成实验报告")


def test_snapshot_prefers_real_local_substep_progress_over_later_resume_zero():
    entries = [
        {
            "text": "0.16",
            "experiment_session_id": "",
            "current_step_id": "step_01_alkaline_analysis",
            "experiment_yaml_path": "C:/demo/experiments.yaml",
            "local_substep_step_id": "step_01_alkaline_analysis",
            "local_substep_index": "5",
        },
        {
            "text": "25",
            "experiment_session_id": "",
            "current_step_id": "step_01_alkaline_analysis",
            "experiment_yaml_path": "C:/demo/experiments.yaml",
            "local_substep_step_id": "step_01_alkaline_analysis",
            "local_substep_index": "8",
        },
        {
            "text": "继续刚才实验",
            "experiment_session_id": "",
            "current_step_id": "",
            "experiment_yaml_path": "C:/demo/experiments.yaml",
            "local_substep_step_id": "",
            "local_substep_index": "",
        },
        {
            "text": "做到称量std2无水碳酸",
            "experiment_session_id": "",
            "current_step_id": "step_01_alkaline_analysis",
            "experiment_yaml_path": "C:/demo/experiments.yaml",
            "local_substep_step_id": "step_01_alkaline_analysis",
            "local_substep_index": "0",
        },
    ]

    snapshot = experiment_resume._extract_latest_user_utterance_snapshot(entries)

    assert snapshot["latest_current_step_id"] == "step_01_alkaline_analysis"
    assert snapshot["latest_local_substep_step_id"] == "step_01_alkaline_analysis"
    assert snapshot["latest_local_substep_index"] == "8"


def test_build_resume_context_uses_wider_snapshot_window_for_local_substeps():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        config = {
            "codex_app": {"llm_name": "codex-main"},
            "LLM": {
                "codex-main": {
                    "type": "codex",
                    "stream_log_path": str(root / "{device_id}.log"),
                    "user_utterance_log_path": str(
                        root / "{device_id}_utterances.jsonl"
                    ),
                }
            },
        }
        target_user_log = root / "34_EF_D7_93_57_12_utterances.jsonl"
        target_stream_log = root / "34_EF_D7_93_57_12.log"

        entries = [
            {
                "ts": "2026-05-20T18:45:15.013+08:00",
                "text": "0.16",
                "experiment_session_id": "",
                "current_step_id": "step_01_alkaline_analysis",
                "experiment_yaml_path": "C:/demo/experiments.yaml",
                "local_substep_step_id": "step_01_alkaline_analysis",
                "local_substep_index": "5",
            },
            {
                "ts": "2026-05-20T18:46:29.367+08:00",
                "text": "25",
                "experiment_session_id": "",
                "current_step_id": "step_01_alkaline_analysis",
                "experiment_yaml_path": "C:/demo/experiments.yaml",
                "local_substep_step_id": "step_01_alkaline_analysis",
                "local_substep_index": "8",
            },
            {
                "ts": "2026-05-20T19:19:05.802+08:00",
                "text": "继续刚才实验",
                "experiment_session_id": "",
                "current_step_id": "",
                "experiment_yaml_path": "C:/demo/experiments.yaml",
                "local_substep_step_id": "",
                "local_substep_index": "",
            },
            {
                "ts": "2026-05-20T19:21:28.152+08:00",
                "text": "做到称量std2无水碳酸",
                "experiment_session_id": "",
                "current_step_id": "step_01_alkaline_analysis",
                "experiment_yaml_path": "C:/demo/experiments.yaml",
                "local_substep_step_id": "step_01_alkaline_analysis",
                "local_substep_index": "0",
            },
        ]

        target_user_log.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n",
            encoding="utf-8",
        )
        target_stream_log.write_text("", encoding="utf-8")

        context = experiment_resume.build_resume_context(
            config,
            "34:EF:D7:93:57:12",
            max_turns=2,
        )

    assert context is not None
    assert context["latest_current_step_id"] == "step_01_alkaline_analysis"
    assert context["latest_local_substep_step_id"] == "step_01_alkaline_analysis"
    assert context["latest_local_substep_index"] == "8"
    assert "local_substep_index=8" in context["context_text"]
