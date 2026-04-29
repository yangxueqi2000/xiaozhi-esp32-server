import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


from core.providers.tools.server_mcp.payload_utils import (
    build_server_mcp_spoken_response,
    finalize_server_mcp_payload,
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


if __name__ == "__main__":
    unittest.main()
