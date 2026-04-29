import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


from core.providers.tools.server_mcp.payload_utils import (
    build_server_mcp_spoken_response,
    finalize_server_mcp_payload,
    sync_server_mcp_payload_state,
)


class ServerMCPPayloadUtilsTest(unittest.TestCase):
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
