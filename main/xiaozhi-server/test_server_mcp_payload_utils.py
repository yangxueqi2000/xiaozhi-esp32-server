import sys
import tempfile
import unittest
from pathlib import Path
import types


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


from core.providers.tools.server_mcp.payload_utils import (
    build_server_mcp_spoken_response,
    extract_experiment_message,
    extract_experiment_session_id,
    finalize_server_mcp_payload,
    sync_server_mcp_payload_state,
)


class ServerMCPPayloadUtilsTest(unittest.TestCase):
    def test_extract_experiment_session_id_reads_state_payload(self):
        payload = {
            "result": {
                "state": {
                    "session_id": "exp-42",
                }
            }
        }

        self.assertEqual("exp-42", extract_experiment_session_id(payload))

    def test_extract_experiment_message_reads_nested_result_message(self):
        payload = {
            "result": {
                "ok": False,
                "message": "failed to create session: title is required",
            }
        }

        self.assertEqual(
            "failed to create session: title is required",
            extract_experiment_message(payload),
        )

    def test_export_records_to_yaml_validation_marks_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            yaml_path = root / "experiment.yaml"
            pdf_path = root / "experiment.pdf"
            yaml_path.write_text("session: demo\n", encoding="utf-8")
            pdf_path.write_bytes(b"%PDF-1.4\n")

            payload = {
                "result": {
                    "yaml_path": str(yaml_path),
                    "pdf_path": str(pdf_path),
                    "pdf_generated": True,
                }
            }

            finalized = finalize_server_mcp_payload(
                payload,
                tool_name="export_records_to_yaml",
                arguments={"output_path": str(root)},
            )

            validation = finalized.get("artifact_validation", {})
            self.assertTrue(validation.get("all_expected_outputs_exist"))
            self.assertTrue(validation.get("all_reported_files_exist"))
            self.assertTrue(validation.get("all_reported_directories_exist"))

    def test_export_records_to_yaml_validation_detects_missing_pdf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            yaml_path = root / "experiment.yaml"
            yaml_path.write_text("session: demo\n", encoding="utf-8")
            missing_pdf_path = root / "experiment.pdf"

            payload = {
                "result": {
                    "yaml_path": str(yaml_path),
                    "pdf_path": str(missing_pdf_path),
                    "pdf_generated": True,
                }
            }

            finalized = finalize_server_mcp_payload(
                payload,
                tool_name="export_records_to_yaml",
                arguments={"output_path": str(root)},
            )

            validation = finalized.get("artifact_validation", {})
            self.assertFalse(validation.get("all_expected_outputs_exist"))
            checked_files = validation.get("checked_files", [])
            self.assertEqual(2, len(checked_files))
            self.assertFalse(
                next(
                    item["exists"]
                    for item in checked_files
                    if item.get("field") == "pdf_path"
                )
            )

    def test_build_server_mcp_spoken_response_for_export_failure(self):
        payload = {
            "artifact_validation": {
                "all_expected_outputs_exist": False,
            }
        }

        reply = build_server_mcp_spoken_response(
            "export_records_to_yaml",
            payload,
        )

        self.assertEqual("实验报告还没有完整生成成功，请稍后再试。", reply)

    def test_build_server_mcp_spoken_response_for_uvvis_scan_result(self):
        payload = {
            "result": {
                "lambda_max_nm": 546.0,
                "max_absorbance": 0.823,
            }
        }

        reply = build_server_mcp_spoken_response(
            "uvvis_scan_result",
            payload,
        )

        self.assertIn("最大吸收波长在546.0纳米", reply)
        self.assertIn("最大吸光度是0.823", reply)

    def test_build_server_mcp_spoken_response_blocks_uvvis_success_without_saved_files(self):
        payload = {
            "result": {
                "lambda_max_nm": 546.0,
                "max_absorbance": 0.823,
            },
            "artifact_validation": {
                "checked_files": [
                    {
                        "field": "absorbance_output_csv",
                        "path": "C:/missing.csv",
                        "exists": False,
                    }
                ],
                "all_expected_outputs_exist": False,
            },
        }

        reply = build_server_mcp_spoken_response(
            "uvvis_scan_result",
            payload,
        )

        self.assertEqual("这次扫描结果还没有保存到指定位置，请稍后再试。", reply)

    def test_sync_server_mcp_payload_state_tracks_uvvis_blank_baseline_meta(self):
        class _Conn:
            pass

        conn = _Conn()
        payload = {
            "blank_baseline_exists": True,
            "blank_baseline_status": "reused",
            "blank_baseline_csv": "C:/demo/air_blank_latest.csv",
            "blank_baseline_manifest_json": "C:/demo/latest_air_blank_manifest.json",
        }

        sync_server_mcp_payload_state(
            conn,
            tool_name="uvvis_measure_spectra",
            payload=payload,
        )

        self.assertEqual("uvvis_measure_spectra", getattr(conn, "_last_server_mcp_tool_name"))
        self.assertEqual(payload, getattr(conn, "_last_server_mcp_payload"))
        self.assertEqual(
            {
                "blank_baseline_exists": True,
                "blank_baseline_status": "reused",
                "blank_baseline_csv": "C:/demo/air_blank_latest.csv",
                "blank_baseline_manifest_json": "C:/demo/latest_air_blank_manifest.json",
            },
            getattr(conn, "_last_uvvis_blank_baseline_state"),
        )

    def test_sync_server_mcp_payload_state_updates_experiment_step_from_progress_summary(self):
        class _Conn:
            pass

        conn = _Conn()
        payload = {
            "result": {
                "summary": {
                    "current_step": {
                        "step_id": "step_sample1_2_add_kbr_water_nabh4",
                    }
                }
            }
        }

        sync_server_mcp_payload_state(
            conn,
            tool_name="get_progress_summary",
            payload=payload,
            arguments={"session_id": "exp-1"},
        )

        self.assertEqual("exp-1", getattr(conn, "experiment_session_id"))
        self.assertEqual(
            "step_sample1_2_add_kbr_water_nabh4",
            getattr(conn, "experiment_current_step_id"),
        )
        self.assertEqual(payload, getattr(conn, "experiment_progress_summary"))
        self.assertFalse(getattr(conn, "_experiment_graph_refresh_required", True))

    def test_sync_server_mcp_payload_state_marks_refresh_required_after_mutation_without_step(self):
        class _Conn:
            pass

        conn = _Conn()
        payload = {"result": {"ok": True, "message": "advanced"}}

        sync_server_mcp_payload_state(
            conn,
            tool_name="proceed_to_next_step",
            payload=payload,
            arguments={"session_id": "exp-1"},
        )

        self.assertEqual("exp-1", getattr(conn, "experiment_session_id"))
        self.assertTrue(getattr(conn, "_experiment_graph_refresh_required", False))
        self.assertEqual(
            "proceed_to_next_step",
            getattr(conn, "_last_experiment_graph_mutation_tool"),
        )

    def test_sync_server_mcp_payload_state_tracks_current_turn_tool_names_by_sentence(self):
        class _Conn:
            pass

        conn = _Conn()
        conn.sentence_id = "turn-1"

        sync_server_mcp_payload_state(
            conn,
            tool_name="get_step",
            payload={"result": {"step": {"id": "step_prepare_setup_all"}}},
            arguments={"session_id": "exp-1"},
        )
        sync_server_mcp_payload_state(
            conn,
            tool_name="uvvis_measure_spectra",
            payload={"result": {"ok": True}},
            arguments={"session_key": "lease-1"},
        )

        self.assertEqual("turn-1", getattr(conn, "_current_turn_server_mcp_sentence_id"))
        self.assertEqual(
            ["get_step", "uvvis_measure_spectra"],
            getattr(conn, "_current_turn_server_mcp_tool_names"),
        )

        conn.sentence_id = "turn-2"
        sync_server_mcp_payload_state(
            conn,
            tool_name="get_progress_summary",
            payload={"result": {"summary": {"current_step": {"step_id": "step-2"}}}},
            arguments={"session_id": "exp-1"},
        )

        self.assertEqual("turn-2", getattr(conn, "_current_turn_server_mcp_sentence_id"))
        self.assertEqual(
            ["get_progress_summary"],
            getattr(conn, "_current_turn_server_mcp_tool_names"),
        )

    def test_build_server_mcp_spoken_response_for_missing_uvvis_blank_baseline(self):
        payload = {
            "success": False,
            "phase": "baseline_missing",
            "blank_baseline_exists": False,
            "blank_baseline_status": "missing",
            "blank_baseline_csv": "C:/demo/air_blank_latest.csv",
        }

        reply = build_server_mcp_spoken_response(
            "uvvis_measure_spectra",
            payload,
        )

        self.assertEqual("还没有空白基线，请先确认空白基线已经准备好，再继续。", reply)


if __name__ == "__main__":
    unittest.main()
