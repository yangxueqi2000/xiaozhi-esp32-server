import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


from plugins_func.register import Action
from core.providers.tools.server_mcp.payload_utils import finalize_server_mcp_payload
from core.providers.tools.server_mcp.uvvis_scan_rule import UVVisScanRule


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.config = {}
        self.device_id = "94:a9:90:28:ea:58"


class UVVisScanRuleTest(unittest.IsolatedAsyncioTestCase):
    def test_before_execute_requires_scan_task_context(self):
        conn = _FakeConn()
        rule = UVVisScanRule(conn, lambda: None)

        response = rule.before_execute("uvvis_scan_result", {})

        self.assertIsNotNone(response)
        self.assertEqual(Action.RESPONSE, response.action)
        self.assertEqual("还没有可查询的扫描任务，请先开始扫描。", response.response)

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


if __name__ == "__main__":
    unittest.main()
