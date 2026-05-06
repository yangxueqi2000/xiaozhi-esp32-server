import sys
import tempfile
import unittest
from pathlib import Path
import types


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: _FakeLogger()
sys.modules.setdefault("config.logger", fake_logger_module)
sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))
fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)


from plugins_func.register import Action
from core.providers.tools.server_mcp.payload_utils import finalize_server_mcp_payload
from core.providers.tools.server_mcp.uvvis_scan_rule import UVVisScanRule


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.config = {}
        self.device_id = "94:a9:90:28:ea:58"


class UVVisScanRuleTest(unittest.IsolatedAsyncioTestCase):
    def test_prepare_arguments_routes_shared_pure_water_blank_to_common_output_dir(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_3_uv_vis_shared_dark_blank_prep"
        rule = UVVisScanRule(conn, lambda: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            conn.config = {"uvvis_scan_output_root": tmp_dir}
            arguments = {
                "sample_positions": [1, 2, 3, 4, 5],
                "ready_for_samples": True,
            }

            rule.prepare_arguments("uvvis_measure_spectra", arguments)

        self.assertEqual(str(Path(tmp_dir).resolve()), arguments["output_dir"])

    def test_prepare_arguments_keeps_batch_sample_spectra_in_device_dir(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_3_uv_vis_sample1-5_record_data"
        rule = UVVisScanRule(conn, lambda: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            conn.config = {"uvvis_scan_output_root": tmp_dir}
            arguments = {
                "sample_positions": [1, 2, 3, 4, 5],
                "ready_for_samples": True,
            }

            rule.prepare_arguments("uvvis_measure_spectra", arguments)

        expected = str((Path(tmp_dir).resolve() / "94_a9_90_28_ea_58").resolve())
        self.assertEqual(expected, arguments["output_dir"])

    def test_prepare_arguments_routes_shared_kinetics_baseline_to_common_output_dir(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_4_kinetics_sample2_measurement"
        rule = UVVisScanRule(conn, lambda: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            conn.config = {"uvvis_scan_output_root": tmp_dir}
            arguments = {
                "wavelength_nm": 400,
                "duration_minutes": 34,
                "interval_seconds": 60,
                "run_name": "sample2",
                "ready_for_samples": False,
                "sample_positions": [1],
            }

            rule.prepare_arguments("uvvis_measure_kinetics", arguments)

        self.assertEqual(str(Path(tmp_dir).resolve()), arguments["output_dir"])

    def test_before_execute_requires_scan_task_context(self):
        conn = _FakeConn()
        rule = UVVisScanRule(conn, lambda: None)

        response = rule.before_execute("uvvis_scan_result", {})

        self.assertIsNotNone(response)
        self.assertEqual(Action.RESPONSE, response.action)
        self.assertEqual("还没有可查询的扫描任务，请先开始扫描。", response.response)

    async def test_after_execute_returns_spoken_scan_start_failure(self):
        conn = _FakeConn()
        rule = UVVisScanRule(conn, lambda: None)

        response = await rule.after_execute(
            "uvvis_scan_start",
            {},
            {"success": False, "error": "设备忙"},
        )

        self.assertIsNotNone(response)
        self.assertEqual(Action.ERROR, response.action)
        self.assertEqual("扫描没有启动成功：设备忙", response.response)

    async def test_after_execute_returns_peak_from_saved_absorbance_csv(self):
        conn = _FakeConn()
        rule = UVVisScanRule(conn, lambda: None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            absorbance_path = Path(tmp_dir) / "absorbance.csv"
            output_path = Path(tmp_dir) / "raw.csv"
            absorbance_path.write_text(
                "wavelength_nm,absorbance\n540,0.612\n546,0.823\n550,0.701\n",
                encoding="utf-8",
            )
            output_path.write_text("wavelength_nm,sample,reference\n", encoding="utf-8")

            payload = finalize_server_mcp_payload(
                {
                    "task_id": "task-1",
                    "state": "succeeded",
                    "result": {
                        "output_csv": str(output_path),
                        "absorbance_output_csv": str(absorbance_path),
                    },
                },
                tool_name="uvvis_scan_result",
                arguments={"task_id": "task-1"},
            )

            response = await rule.after_execute(
                "uvvis_scan_result",
                {"task_id": "task-1"},
                payload,
            )

        self.assertIsNotNone(response)
        self.assertEqual(Action.RESPONSE, response.action)
        self.assertIn("最大吸收波长在546.0纳米", response.response)
        self.assertIn("最大吸光度是0.823", response.response)

    async def test_after_execute_blocks_uvvis_result_without_saved_artifact(self):
        conn = _FakeConn()
        rule = UVVisScanRule(conn, lambda: None)

        payload = finalize_server_mcp_payload(
            {
                "task_id": "task-2",
                "state": "succeeded",
                "result": {
                    "lambda_max_nm": 546.0,
                    "max_absorbance": 0.823,
                    "output_csv": "C:/missing/raw.csv",
                    "absorbance_output_csv": "C:/missing/absorbance.csv",
                },
            },
            tool_name="uvvis_scan_result",
            arguments={"task_id": "task-2"},
        )

        response = await rule.after_execute(
            "uvvis_scan_result",
            {"task_id": "task-2"},
            payload,
        )

        self.assertIsNotNone(response)
        self.assertEqual(Action.RESPONSE, response.action)
        self.assertEqual("这次扫描结果还没有保存到指定位置，请稍后再试。", response.response)

    async def test_after_execute_reminds_when_uvvis_blank_baseline_is_missing(self):
        conn = _FakeConn()
        rule = UVVisScanRule(conn, lambda: None)

        payload = {
            "success": False,
            "phase": "baseline_missing",
            "blank_baseline_exists": False,
            "blank_baseline_status": "missing",
            "blank_baseline_csv": "C:/missing/air_blank_latest.csv",
        }

        response = await rule.after_execute(
            "uvvis_measure_spectra",
            {"ready_for_samples": True},
            payload,
        )

        self.assertIsNotNone(response)
        self.assertEqual(Action.RESPONSE, response.action)
        self.assertEqual("还没有空白基线，请先确认空白基线已经准备好，再继续。", response.response)


if __name__ == "__main__":
    unittest.main()
