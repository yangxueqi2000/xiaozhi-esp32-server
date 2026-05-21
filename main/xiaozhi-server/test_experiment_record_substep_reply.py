import sys
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
fake_send_audio_module.send_stt_message = lambda *args, **kwargs: None
fake_send_audio_module.send_tts_message = lambda *args, **kwargs: None
fake_send_audio_module.sendAudioMessage = lambda *args, **kwargs: None
fake_send_audio_module.SentenceType = types.SimpleNamespace(FIRST="FIRST", LAST="LAST")
sys.modules.setdefault("core.handle.sendAudioHandle", fake_send_audio_module)

sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)

from core.handle import intentHandler


class RecordSubstepReplyTest(unittest.TestCase):
    def test_non_local_step_reply_stays_on_original_big_step_behavior(self):
        reply = intentHandler._compose_experiment_step_reply(
            {
                "step_id": "step_01_alkaline_analysis",
                "title": "工业碱总碱度的分析",
                "instruction": "请按照以下细分操作完成：工业碱总碱度的分析",
                "safety": "HCl具有腐蚀性。",
            },
            mode="guide",
        )

        self.assertIn("现在做这一步：请按照以下细分操作完成：工业碱总碱度的分析。", reply)
        self.assertIn("做好后告诉我。", reply)
        self.assertNotIn("把工业碱总碱度的分析告诉我", reply)
        self.assertEqual(
            {},
            intentHandler._extract_local_substep_report_requirement(
                {
                    "instruction": "请按照以下细分操作完成：工业碱总碱度的分析",
                }
            ),
        )

    def test_weighing_substep_requires_mass_report_in_reply(self):
        reply = intentHandler._compose_experiment_step_reply(
            {
                "step_id": "step_01_alkaline_analysis",
                "title": "工业碱总碱度的分析",
                "parent_step_title": "工业碱总碱度的分析",
                "instruction": "称取0.15~0.2 g无水Na2CO3于std_1，并记录实际质量。",
                "is_local_substep": True,
                "safety": "HCl具有腐蚀性。",
            },
            mode="next",
        )

        self.assertIn("记录完告诉我std_1的实际质量", reply)
        self.assertNotIn("做好后告诉我", reply)

    def test_record_instruction_is_rewritten_for_student_facing_speech(self):
        reply = intentHandler._compose_experiment_step_reply(
            {
                "step_id": "step_01_alkaline_analysis",
                "title": "工业碱总碱度的分析",
                "instruction": "记录std_1本次滴定的起始读数。",
                "safety": "HCl具有腐蚀性。",
            },
            mode="next",
        )

        self.assertIn("先把std_1本次滴定的起始读数记下来", reply)
        self.assertNotEqual(
            "接下来做这一步：工业碱总碱度的分析。注意HCl具有腐蚀性。做好后告诉我。",
            reply,
        )

    def test_local_substep_reply_omits_parent_step_title(self):
        reply = intentHandler._compose_experiment_step_reply(
            {
                "step_id": "step_01_alkaline_analysis",
                "title": "工业碱总碱度的分析",
                "parent_step_title": "工业碱总碱度的分析",
                "instruction": "向滴定管装入待标定HCl溶液，并排尽尖嘴气泡。",
                "is_local_substep": True,
                "safety": "HCl具有腐蚀性。",
            },
            mode="next",
        )

        self.assertTrue(reply.startswith("接下来做这一步：向滴定管装入待标定HCl溶液"))
        self.assertNotIn("接下来做这一步：工业碱总碱度的分析。", reply)

    def test_local_substep_mass_report_needs_number(self):
        requirement = intentHandler._extract_local_substep_report_requirement(
            {
                "instruction": "称取0.15~0.2 g无水Na2CO3于std_1，并记录实际质量。",
                "is_local_substep": True,
            }
        )

        self.assertFalse(
            intentHandler._local_substep_report_is_satisfied("做好了", requirement)
        )
        self.assertTrue(
            intentHandler._local_substep_report_is_satisfied(
                "std_1 是 0.1823 克",
                requirement,
            )
        )

    def test_rewritten_record_instruction_keeps_record_completion_prompt(self):
        requirement = intentHandler._extract_local_substep_report_requirement(
            {
                "instruction": "先把std_1本次滴定的终止读数、HCl体积、终点观察和有效性记下来。",
                "is_local_substep": True,
            }
        )

        self.assertEqual(
            "记录完告诉我std_1本次滴定的终止读数、HCl体积、终点观察和有效性。",
            requirement.get("completion_prompt"),
        )
        self.assertTrue(
            intentHandler._local_substep_report_is_satisfied(
                "终止读数 23.6，体积 1.4",
                requirement,
            )
        )


    def test_exp3_mass_report_builds_structured_weighing_and_trial_data(self):
        payload = intentHandler._build_exp3_local_substep_write_fields(
            {
                "step_id": "step_01_alkaline_analysis",
                "substep_index": 5,
                "instruction": "称取0.15~0.2 g无水Na2CO3于std_1，并记录实际质量。",
                "is_local_substep": True,
            },
            "0.16",
            {},
        )

        self.assertEqual("pending_measured_in_real_run", payload["real_measurement_status"])
        self.assertEqual(0.16, payload["weighing_records"][0]["actual_mass_g"])
        self.assertEqual("std_1", payload["weighing_records"][0]["record_id"])
        self.assertEqual(0.16, payload["standardization_trials"][0]["standard_mass_g"])
        self.assertEqual("std_1", payload["standardization_trials"][0]["container_label"])

    def test_exp3_initial_burette_report_merges_into_standardization_trial(self):
        payload = intentHandler._build_exp3_local_substep_write_fields(
            {
                "step_id": "step_01_alkaline_analysis",
                "substep_index": 8,
                "instruction": "记录std_1本次滴定的起始读数。",
                "is_local_substep": True,
            },
            "25",
            {
                "weighing_records": [
                    {
                        "record_id": "std_1",
                        "material_name": "无水Na2CO3",
                        "actual_mass_g": 0.16,
                        "weighing_valid": True,
                    }
                ]
            },
        )

        trial = payload["standardization_trials"][0]
        self.assertEqual(25.0, trial["burette_initial_ml"])
        self.assertEqual(0.16, trial["standard_mass_g"])
        self.assertEqual("pending_measured_in_real_run", trial["measurement_status"])

    def test_exp3_final_burette_report_computes_volume_and_marks_measured(self):
        payload = intentHandler._build_exp3_local_substep_write_fields(
            {
                "step_id": "step_01_alkaline_analysis",
                "substep_index": 10,
                "instruction": "记录std_1的终止读数、HCl体积、终点观察和有效性。",
                "is_local_substep": True,
            },
            "终止读数 23.6，黄色恰变橙色，有效",
            {
                "standardization_trials": [
                    {
                        "trial_id": "1",
                        "container_label": "std_1",
                        "standard_material_label": "std_1",
                        "standard_mass_g": 0.16,
                        "burette_initial_ml": 2.2,
                    }
                ]
            },
        )

        trial = payload["standardization_trials"][0]
        self.assertEqual(23.6, trial["burette_final_ml"])
        self.assertEqual(21.4, trial["hcl_volume_used_ml"])
        self.assertEqual("黄色恰变橙色", trial["endpoint_observation"])
        self.assertTrue(trial["endpoint_confirmed"])
        self.assertTrue(trial["trial_valid"])
        self.assertEqual("measured", trial["measurement_status"])
        self.assertTrue(payload["titration_endpoint_rule_confirmed"])


if __name__ == "__main__":
    unittest.main()
