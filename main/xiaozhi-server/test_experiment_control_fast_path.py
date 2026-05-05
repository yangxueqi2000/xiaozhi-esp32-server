import csv
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch


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


class _FakeDialogue:
    def __init__(self):
        self.dialogue = []

    def put(self, message):
        self.dialogue.append(message)


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

sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)

from core.handle import intentHandler
from core.utils.dialogue import Message
from core.utils import experiment_resume, textUtils


class _FakeConn:
    def __init__(self):
        self.logger = _FakeLogger()
        self.dialogue = _FakeDialogue()
        self.client_abort = False
        self.sentence_id = None
        self.chat_session_id = ""
        self.model_session_key = ""
        self.user_id = ""
        self.device_id = ""
        self.prompt = ""
        self.llm = None
        self.experiment_session_id = "exp-1"
        self.experiment_current_step_id = "step_prepare_setup_all"
        self.experiment_current_step = None
        self.experiment_progress_summary = None
        self.intent_type = "function_call"
        self.config = {}
        self.tts = None
        self.enriched = False
        self.cmd_exit = []
        self.experiment_resume_recovery_required = False
        self.experiment_resume_latest_current_step_id = ""
        self.experiment_resume_log_path = ""
        self.experiment_resume_turn_count = "0"
        self.experiment_resume_latest_session_id = ""

    def enrich_latest_clean_user_utterance_snapshot(self):
        self.enriched = True

    def _extract_experiment_current_step_id(self, *payloads):
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            body = payload.get("result", payload)
            if not isinstance(body, dict):
                continue
            step = body.get("step")
            if isinstance(step, dict):
                step_id = str(step.get("id", "")).strip()
                if step_id:
                    return step_id
            summary = body.get("summary")
            if isinstance(summary, dict):
                current_step = summary.get("current_step")
                if isinstance(current_step, dict):
                    step_id = str(current_step.get("step_id", "")).strip()
                    if step_id:
                        return step_id
        return ""


class ExperimentControlFastPathTest(unittest.IsolatedAsyncioTestCase):
    def test_runtime_spoken_text_keeps_only_canonical_opening(self):
        text = (
            "今天我们做《银纳米粒子实验》。你准备好开始了吗？"
            "我先读取一下当前步骤。"
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "今天我们做《银纳米粒子实验》。你准备好开始了吗？",
            result,
        )

    def test_append_experiment_interaction_log_writes_transcript_line(self):
        with TemporaryDirectory() as temp_dir:
            config = {
                "LLM": {
                    "codex_app_server": {
                        "type": "codex",
                        "stream_log_path": str(Path(temp_dir) / "{device_id}.log"),
                    }
                }
            }

            log_path = experiment_resume.append_experiment_interaction_log(
                config,
                "94:a9:90:27:3c:84",
                "可以拍照。",
                role="USER",
                source="asr",
                current_step_id="step_photo_confirm_sample_1",
            )

            self.assertTrue(log_path)
            self.assertEqual(
                Path(temp_dir) / "94_a9_90_27_3c_84.log",
                Path(log_path),
            )
            content = Path(log_path).read_text(encoding="utf-8")
            self.assertIn("[TRANSCRIPT] [USER]", content)
            self.assertIn("[source=asr]", content)
            self.assertIn("[current_step_id=step_photo_confirm_sample_1]", content)
            self.assertIn("可以拍照。", content)

    def test_build_resume_context_extracts_latest_step_from_transcript_log(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "94_a9_90_27_3c_84.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-05-05T14:34:44.431+08:00] [TRANSCRIPT] [USER] [source=asr] "
                        "[experiment_session_id=dfcdd64e1f7947b48915e6a4c985bd1f] "
                        "[current_step_id=step_prepare_setup_all] "
                        "[yaml=C:\\demo\\experiments.yaml] \u5f00\u59cb\u4eca\u5929\u7684\u5b9e\u9a8c\u3002",
                        "[2026-05-05T14:35:58.428+08:00] [TRANSCRIPT] [ASSISTANT] [source=speak_txt] "
                        "[experiment_session_id=dfcdd64e1f7947b48915e6a4c985bd1f] "
                        "[current_step_id=step_sample1_2_add_kbr_water_nabh4] "
                        "[yaml=C:\\demo\\experiments.yaml] \u7ee7\u7eed\u524d\u8fd8\u5dee\u8fd9\u51e0\u4e2a\u786e\u8ba4\u3002",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "LLM": {
                    "codex_app_server": {
                        "type": "codex",
                        "stream_log_path": str(Path(temp_dir) / "{device_id}.log"),
                    }
                }
            }

            context = experiment_resume.build_resume_context(
                config,
                "94:a9:90:27:3c:84",
            )

            self.assertIsNotNone(context)
            self.assertEqual(
                "step_sample1_2_add_kbr_water_nabh4",
                context.get("latest_current_step_id", ""),
            )
            self.assertEqual(
                "dfcdd64e1f7947b48915e6a4c985bd1f",
                context.get("latest_experiment_session_id", ""),
            )

    def test_read_transcript_entries_keeps_step_and_role_metadata(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "94_a9_90_27_3c_84.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-05-05T14:34:44.431+08:00] [TRANSCRIPT] [USER] [source=asr] "
                        "[experiment_session_id=exp-1] [current_step_id=step_prepare_setup_all] "
                        "[yaml=C:\\demo\\experiments.yaml] 全部完成。",
                        "[2026-05-05T14:34:57.063+08:00] [TRANSCRIPT] [ASSISTANT] [source=speak_txt] "
                        "[experiment_session_id=exp-1] [current_step_id=step_add_sodium_citrate_all] "
                        "[yaml=C:\\demo\\experiments.yaml] 现在做这一步。",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            entries = experiment_resume.read_transcript_entries(log_path)

            self.assertEqual(2, len(entries))
            self.assertEqual("USER", entries[0]["role"])
            self.assertEqual("step_prepare_setup_all", entries[0]["current_step_id"])
            self.assertEqual("全部完成。", entries[0]["text"])
            self.assertEqual("ASSISTANT", entries[1]["role"])
            self.assertEqual(
                "step_add_sodium_citrate_all",
                entries[1]["current_step_id"],
            )

    def test_runtime_spoken_text_strips_technical_details(self):
        text = (
            "实验报告已经生成，pdf_path=C:\\demo\\report.pdf，"
            "session_id=abc123。"
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual("实验报告已经生成。", result)

    def test_runtime_spoken_text_limits_to_two_sentences(self):
        text = "现在做这一步。注意不要污染。做好后告诉我。"

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual("现在做这一步。注意不要污染，做好后告诉我。", result)

    def test_conn_runtime_spoken_text_blocks_false_export_success(self):
        conn = _FakeConn()
        conn._pending_export_report_validation = {
            "active": True,
            "all_expected_outputs_exist": False,
        }

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "实验报告已经生成完成。",
        )

        self.assertEqual("实验报告还没有完整生成成功，请稍后再试。", result)

    def test_conn_runtime_spoken_text_rewrites_speculative_uvvis_occupation(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-uvvis-1"

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "这里暂时还没能直接启动校正，你先检查一下光谱仪有没有被别的程序占用，确认后告诉我继续。",
        )

        self.assertEqual("UV-Vis 这边还没准备好，请稍后再试。", result)

    def test_conn_runtime_spoken_text_strips_speculative_interface_outage(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-1"

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "实验图谱接口这轮没接通。",
        )

        self.assertEqual("", result)

    def test_conn_runtime_spoken_text_reanchors_speculative_future_step_to_current_graph_step(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-align-1"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "先完成 1-5 号烧杯编号和磁转子放置。",
                    },
                }
            }
        }
        conn.dialogue.put(Message(role="user", content="继续下一步。"))

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "现在做丁达尔现象观察：把环境调暗，用激光笔从侧面照射样品。看完后告诉我。",
        )

        self.assertIn("准备烧杯与磁转子", result)
        self.assertIn("烧杯编号", result)
        self.assertNotIn("丁达尔", result)

    def test_conn_runtime_spoken_text_reanchors_future_step_even_after_graph_tools_ran(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-align-2"
        conn._current_turn_server_mcp_sentence_id = "turn-graph-align-2"
        conn._current_turn_server_mcp_tool_names = [
            "get_step",
            "get_progress_summary",
        ]
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_add_h2o2_all",
                    "title": "1-5号样品：统一加入H2O2",
                    "prompts": {
                        "instruction": "按 1 到 5 号顺序统一完成 H2O2 加入。",
                    },
                }
            }
        }
        conn.dialogue.put(Message(role="user", content="继续下一步。"))

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "现在做丁达尔现象观察：把环境调暗，用激光笔从侧面照射样品。看完后告诉我。",
        )

        self.assertIn("统一加入H2O2", result)
        self.assertIn("H2O2", result)
        self.assertNotIn("丁达尔", result)

    def test_conn_runtime_spoken_text_bypasses_ready_guard_for_same_sentence(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做这个实验。你准备好开始了吗？",
            )
        )
        conn.sentence_id = "turn-1"
        conn._experiment_ready_guard_bypass_sentence_id = "turn-1"

        guidance = "现在做这一步：完成 1 到 5 号烧杯编号。做好后告诉我。"
        first = textUtils.prepare_runtime_spoken_text_for_conn(conn, guidance)
        second = textUtils.prepare_runtime_spoken_text_for_conn(conn, guidance)

        self.assertIn("现在做这一步", first)
        self.assertIn("现在做这一步", second)
        self.assertEqual(
            "turn-1",
            getattr(conn, "_experiment_ready_guard_bypass_sentence_id", ""),
        )

        conn.sentence_id = "turn-2"
        third = textUtils.prepare_runtime_spoken_text_for_conn(conn, guidance)
        self.assertEqual(first, third)

    def test_activate_ready_guard_bypass_binds_pending_turn_to_current_sentence(self):
        conn = _FakeConn()
        conn._experiment_ready_guard_bypass_pending_turns = 1
        conn.sentence_id = "turn-ready-1"

        activated = textUtils.activate_experiment_ready_guard_bypass_for_current_sentence(
            conn
        )

        self.assertTrue(activated)
        self.assertEqual(
            "turn-ready-1",
            getattr(conn, "_experiment_ready_guard_bypass_sentence_id", ""),
        )
        self.assertEqual(
            0,
            getattr(conn, "_experiment_ready_guard_bypass_pending_turns", 0),
        )

    def test_normalize_tts_text_reads_decimals_digit_by_digit(self):
        result = textUtils.normalize_tts_text("加入1.849mL AgNO3。")

        self.assertIn("一点八四九毫升", result)
        self.assertIn("硝酸银", result)

    def test_normalize_tts_text_reads_numeric_time_ranges_as_to(self):
        result = textUtils.normalize_tts_text("静置1-15分钟后观察。")

        self.assertEqual("静置1到15分钟后观察。", result)

    def test_normalize_tts_text_reads_decimal_volume_ranges_as_to(self):
        result = textUtils.normalize_tts_text("加入0.5-1.0mL AgNO3。")

        self.assertIn("零点五到一点零毫升", result)
        self.assertIn("硝酸银", result)

    def test_normalize_tts_text_supports_generic_chemistry_pronunciation(self):
        result = textUtils.normalize_tts_text(
            "Ag（银）纳米粒子的制备及其催化还原4-硝基苯酚的反应动力学探究"
        )

        self.assertEqual(
            "银纳米粒子的制备及其催化还原对硝基苯酚的反应动力学探究",
            result,
        )

    def test_payload_looks_busy_or_inaccessible_ignores_idle_lease_metadata(self):
        payload = {
            "available": True,
            "occupied": False,
            "active_measurement": False,
            "session_key": "lease-1",
            "lease_idle_timeout_seconds": 14400.0,
        }

        self.assertFalse(intentHandler._payload_looks_busy_or_inaccessible(payload))

    def test_normalize_tts_text_reads_numbered_labels_with_erhao(self):
        result = textUtils.normalize_tts_text(
            "现在做2号样品：在2号烧杯里加入溴化钾0.80毫升，再加入纯水2.10毫升。"
        )

        self.assertIn("二号样品", result)
        self.assertIn("二号烧杯", result)
        self.assertIn("零点八零毫升", result)
        self.assertIn("二点一零毫升", result)

    def test_normalize_tts_text_reads_numbered_label_ranges_with_chinese_digits(self):
        result = textUtils.normalize_tts_text("请把1-5号样品位和参比位都放好。")

        self.assertEqual("请把一到五号样品位和参比位都放好。", result)

    def test_prepare_runtime_spoken_text_strips_meta_scope_clauses(self):
        text = (
            "接下来做这一步：1到5：同时启动搅拌并混匀。"
            "只完成1到5的搅拌统一启动和混匀确认，不要重复共同试剂，也不要讲后续加液，"
            "注意启动搅拌前确认所有烧杯放置平稳。"
            "注意转速不要过高，避免液体飞溅，做好后告诉我。"
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "接下来做这一步：1到5：同时启动搅拌并混匀。"
            "注意启动搅拌前确认所有烧杯放置平稳，注意转速不要过高，避免液体飞溅，做好后告诉我。",
            result,
        )

    def test_prepare_runtime_spoken_text_strips_future_step_transition_clauses(self):
        text = (
            "接下来做这一步：按1到5号顺序统一完成H2O2加入。"
            "完成这一轮后再回到1号样品开始后续步骤。"
            "注意加液后轻轻混匀。"
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "接下来做这一步：按1到5号顺序统一完成H2O2加入。注意加液后轻轻混匀。",
            result,
        )

    def test_prepare_runtime_spoken_text_compacts_long_measured_step(self):
        text = (
            "接下来做这一步：1号样品：加入溴化钾、纯水并加入硼氢化钠。"
            "完成1号样品溴化钾和纯水加入（溴化钾零点零零毫升，纯水二点九零毫升）并混匀后，"
            "快速加入硼氢化钠（二点五零毫升 零点零零五摩尔每升）并保持搅拌，"
            "记录颜色稳定时间和收尾情况。"
            "注意继续保持搅拌，避免液体飞溅，硼氢化钠有腐蚀性，注意防护并避免溅出，做好后告诉我。"
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "接下来做这一步：1号样品：先加入溴化钾零点零零毫升和纯水二点九零毫升并混匀，再快速加入硼氢化钠二点五零毫升，持续搅拌。"
            "避免液体飞溅，硼氢化钠有腐蚀性，注意防护并避免溅出，做好后告诉我。",
            result,
        )

    def test_prepare_runtime_spoken_text_keeps_observation_action_while_dropping_reporting_tail(self):
        text = (
            "接下来做这一步：2号样品：观察颜色变化。"
            "静置1到2分钟后观察并拍照，记录颜色变化时间和结果，做好后告诉我。"
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "接下来做这一步：2号样品：先静置1到2分钟，再观察并拍照，做好后告诉我。",
            result,
        )

    def test_prepare_runtime_spoken_text_drops_recordkeeping_backstage_sentence(self):
        text = "我先记下一号样品的最终颜色和稳定时间。现在可以拍照。"

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual("现在可以拍照。", result)

    def test_current_step_confirmation_fields_accept_all_added_completion_report(self):
        schema_by_name = {
            "h2o2_added_to_all": {
                "type": "bool",
                "description": "已按 1-5 号顺序完成全部 H2O2 加入",
            }
        }

        result = intentHandler._build_experiment_current_step_confirmation_fields(
            "全部加好了",
            schema_by_name,
            ["h2o2_added_to_all"],
            allow_confirmation_autofill=True,
        )

        self.assertEqual({"h2o2_added_to_all": True}, result)

    def test_repeat_reply_is_direct_and_requests_completion(self):
        reply = intentHandler._compose_experiment_step_reply(
            {
                "title": "1-5号样品：统一加入柠檬酸钠",
                "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                "safety": "加液时保持移液操作稳定。",
            },
            mode="repeat",
        )

        self.assertNotIn("我再简短说一遍", reply)
        self.assertIn("当前这一步", reply)
        self.assertIn("做好后告诉我", reply)

    def test_experiment_fast_path_actions_can_be_limited_by_config(self):
        conn = _FakeConn()
        conn.config = {
            "experiment_fast_path_allowed_actions": ["advance", "repeat", "confirm"],
        }

        self.assertTrue(
            intentHandler._is_experiment_fast_path_action_enabled(conn, "advance")
        )
        self.assertTrue(
            intentHandler._is_experiment_fast_path_action_enabled(conn, "confirm")
        )
        self.assertFalse(
            intentHandler._is_experiment_fast_path_action_enabled(conn, "guide")
        )

    def test_neutral_ack_follows_completion_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(role="assistant", content="现在做这一步。做好后告诉我。")
        )

        action = intentHandler._classify_short_experiment_control(conn, "好了")
        self.assertEqual("advance", action)

    def test_continue_follows_start_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做这个实验。你准备好开始了吗？",
            )
        )

        action = intentHandler._classify_short_experiment_control(conn, "继续")
        self.assertEqual("guide", action)

    def test_ready_reply_wins_while_waiting_for_start(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="你准备好后告诉我准备好了，我再带你开始第一步。",
            )
        )

        action = intentHandler._classify_short_experiment_control(conn, "准备好了")
        self.assertEqual("guide", action)

    def test_step_guidance_clears_waiting_for_start_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氳繖涓疄楠屻€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鐜板湪缁欎簲鍙锋牱鍝佸姞鍏ョ〖姘㈠寲閽犮€傚姞瀹屽憡璇夋垜銆?",
            )
        )

        self.assertFalse(intentHandler._assistant_waiting_for_step_start(conn))

    def test_completion_variant_uses_waiting_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="现在给五号样品加入硼氢化钠。加完告诉我。",
            )
        )

        action = intentHandler._classify_short_experiment_control(conn, "已经加好了")
        self.assertEqual("advance", action)

    def test_mixed_completion_report_does_not_shortcut(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="现在给五号样品加入硼氢化钠。加完告诉我。",
            )
        )

        action = intentHandler._classify_short_experiment_control(
            conn,
            "已经加好了是浅黄色一分钟",
        )
        self.assertEqual("", action)

    def test_autofill_allows_simple_manual_bool_fields(self):
        schema = {
            "beakers_labeled": {
                "name": "beakers_labeled",
                "type": "bool",
                "description": "已完成 1-5 号烧杯编号",
            },
            "stir_bars_added_to_all": {
                "name": "stir_bars_added_to_all",
                "type": "bool",
                "description": "已为 1-5 号烧杯全部放入磁转子",
            },
        }

        result = intentHandler._build_experiment_autofill_fields(
            schema,
            ["beakers_labeled", "stir_bars_added_to_all"],
        )

        self.assertEqual({}, result)

    def test_autofill_blocks_photo_confirmation_fields(self):
        schema = {
            "photo_taken": {
                "name": "photo_taken",
                "type": "bool",
                "description": "已通过 MCP 直接拍照记录当前样品颜色",
            },
            "color_confirmed_by_photo": {
                "name": "color_confirmed_by_photo",
                "type": "bool",
                "description": "已基于照片确认当前样品颜色稳定",
            },
        }

        result = intentHandler._build_experiment_autofill_fields(
            schema,
            ["photo_taken", "color_confirmed_by_photo"],
        )

        self.assertEqual({}, result)

    def test_schema_view_falls_back_to_json_schema_properties(self):
        payload = {
            "result": {
                "ok": True,
                "json_schema": {
                    "properties": {
                        "sodium_citrate_added_to_all": {
                            "type": "boolean",
                            "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                        }
                    },
                    "required": ["sodium_citrate_added_to_all"],
                }
            }
        }

        result = intentHandler._extract_experiment_schema_view(payload)

        self.assertIn("sodium_citrate_added_to_all", result)
        self.assertEqual("boolean", result["sodium_citrate_added_to_all"]["type"])
        self.assertFalse(result["sodium_citrate_added_to_all"]["optional"])

    def test_confirmation_step_autofills_missing_bool_confirmation_fields(self):
        schema = {
            "shared_round_confirmed": {
                "name": "shared_round_confirmed",
                "type": "bool",
                "description": "本轮共同操作记录",
            }
        }

        result = intentHandler._build_experiment_autofill_fields(
            schema,
            ["shared_round_confirmed"],
            allow_confirmation_autofill=True,
        )

        self.assertEqual({}, result)

    def test_default_autofill_does_not_false_match_citrate_as_rate(self):
        schema = {
            "sodium_citrate_added_to_all": {
                "name": "sodium_citrate_added_to_all",
                "type": "bool",
                "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
            }
        }

        result = intentHandler._build_experiment_autofill_fields(
            schema,
            ["sodium_citrate_added_to_all"],
        )

        self.assertEqual({}, result)

    def test_photo_confirm_delay_defaults_to_three_seconds(self):
        conn = _FakeConn()
        conn.config = {}

        delay_seconds = intentHandler._resolve_photo_confirm_delay_seconds(conn)

        self.assertEqual(3.0, delay_seconds)

    async def test_server_photo_confirmation_direct_can_be_disabled(self):
        conn = _FakeConn()
        conn.config = {
            "device_mcp_shortcuts": {
                "enable_server_photo_confirmation_direct": False,
            }
        }

        with patch.object(
            intentHandler,
            "_assistant_is_waiting_for_photo_permission_fixed",
            side_effect=AssertionError("should not check confirmation state"),
        ):
            handled = await intentHandler.handle_pending_server_photo_confirmation(
                conn,
                "可以拍照",
                "可以拍照",
            )

        self.assertFalse(handled)

    async def test_handle_user_intent_stages_ready_guard_bypass_when_fast_path_disabled(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.config = {"experiment_fast_path_enabled": False}
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做《银纳米粒子实验》。你准备好开始了吗？",
            )
        )

        handled = await intentHandler.handle_user_intent(conn, "准备好了")

        self.assertFalse(handled)
        self.assertEqual(
            1,
            getattr(conn, "_experiment_ready_guard_bypass_pending_turns", 0),
        )

    async def test_handle_user_intent_routes_photo_command_to_direct_handler(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"

        async def return_false(*args, **kwargs):
            return False

        async def handle_direct_photo(*args, **kwargs):
            return True

        with patch.object(
            intentHandler,
            "handle_pending_direct_photo_confirmation",
            return_false,
        ):
            with patch.object(
                intentHandler,
                "handle_pending_server_photo_confirmation",
                return_false,
            ):
                with patch.object(
                    intentHandler,
                    "handle_direct_photo_navigation_intent",
                    return_false,
                ):
                    with patch.object(
                        intentHandler,
                        "handle_direct_photo_intent",
                        handle_direct_photo,
                    ):
                        with patch.object(
                            intentHandler,
                            "handle_experiment_control_fast_intent",
                            return_false,
                        ):
                            handled = await intentHandler.handle_user_intent(
                                conn,
                                "拍照",
                            )

        self.assertTrue(handled)

    async def test_handle_user_intent_routes_photo_confirmation_reply_directly(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.dialogue.put(Message(role="assistant", content="可以拍照吗？"))
        conn._server_photo_capture_granted = False

        async def return_false(*args, **kwargs):
            return False

        async def handle_pending_server(*args, **kwargs):
            return True

        with patch.object(
            intentHandler,
            "handle_pending_direct_photo_confirmation",
            return_false,
        ):
            with patch.object(
                intentHandler,
                "handle_pending_server_photo_confirmation",
                handle_pending_server,
            ):
                with patch.object(
                    intentHandler,
                    "handle_direct_photo_navigation_intent",
                    return_false,
                ):
                    with patch.object(
                        intentHandler,
                        "handle_direct_photo_intent",
                        return_false,
                    ):
                        handled = await intentHandler.handle_user_intent(conn, "可以拍照")

        self.assertTrue(handled)
        self.assertTrue(getattr(conn, "_server_photo_capture_granted", False))

    async def test_handle_user_intent_refreshes_dirty_experiment_state_before_direct_handlers(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.experiment_session_id = "exp-1"
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._experiment_graph_refresh_required = True

        async def fake_refresh_experiment_foreground_state(*, reason=""):
            conn.experiment_current_step_id = "step_sample1_2_add_kbr_water_nabh4"
            conn._experiment_graph_refresh_required = False
            return {"current_step_id": conn.experiment_current_step_id}

        conn.refresh_experiment_foreground_state = fake_refresh_experiment_foreground_state

        async def handle_pending_direct_photo(*args, **kwargs):
            self.assertEqual(
                "step_sample1_2_add_kbr_water_nabh4",
                conn.experiment_current_step_id,
            )
            self.assertFalse(conn._experiment_graph_refresh_required)
            return True

        async def return_false(*args, **kwargs):
            return False

        with patch.object(
            intentHandler,
            "handle_pending_direct_photo_confirmation",
            handle_pending_direct_photo,
        ):
            with patch.object(
                intentHandler,
                "handle_pending_server_photo_confirmation",
                return_false,
            ):
                with patch.object(
                    intentHandler,
                    "handle_direct_photo_navigation_intent",
                    return_false,
                ):
                    with patch.object(
                        intentHandler,
                        "handle_direct_photo_intent",
                        return_false,
                    ):
                        with patch.object(
                            intentHandler,
                            "handle_direct_uvvis_intent",
                            return_false,
                        ):
                            with patch.object(
                                intentHandler,
                                "handle_experiment_control_fast_intent",
                                return_false,
                            ):
                                handled = await intentHandler.handle_user_intent(
                                    conn,
                                    "可以拍照",
                                )

        self.assertTrue(handled)

    async def test_handle_user_intent_routes_short_experiment_control_to_fast_path(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        seen = []

        async def return_false(*args, **kwargs):
            return False

        async def handle_fast(_conn, original_text, filtered_text):
            seen.append((original_text, filtered_text))
            return True

        with patch.object(
            intentHandler,
            "handle_pending_direct_photo_confirmation",
            return_false,
        ):
            with patch.object(
                intentHandler,
                "handle_pending_server_photo_confirmation",
                return_false,
            ):
                with patch.object(
                    intentHandler,
                    "handle_direct_photo_navigation_intent",
                    return_false,
                ):
                    with patch.object(
                        intentHandler,
                        "handle_direct_photo_intent",
                        return_false,
                    ):
                        with patch.object(
                            intentHandler,
                            "handle_direct_uvvis_intent",
                            return_false,
                        ):
                            with patch.object(
                                intentHandler,
                                "handle_experiment_control_fast_intent",
                                handle_fast,
                            ):
                                handled = await intentHandler.handle_user_intent(
                                    conn,
                                    "鍏ㄩ儴鍔犲ソ浜?",
                                )

        self.assertTrue(handled)
        self.assertEqual(
            [("鍏ㄩ儴鍔犲ソ浜?", "鍏ㄩ儴鍔犲ソ浜")],
            seen,
        )

    async def test_handle_user_intent_routes_short_experiment_control_to_strict_graph_path_when_fast_path_disabled(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.config = {"experiment_fast_path_enabled": False}
        seen = []

        async def return_false(*args, **kwargs):
            return False

        async def handle_strict(_conn, original_text, filtered_text):
            seen.append((original_text, filtered_text))
            return True

        with patch.object(
            intentHandler,
            "handle_pending_direct_photo_confirmation",
            return_false,
        ):
            with patch.object(
                intentHandler,
                "handle_pending_server_photo_confirmation",
                return_false,
            ):
                with patch.object(
                    intentHandler,
                    "handle_direct_photo_navigation_intent",
                    return_false,
                ):
                    with patch.object(
                        intentHandler,
                        "handle_direct_photo_intent",
                        return_false,
                    ):
                        with patch.object(
                            intentHandler,
                            "handle_direct_uvvis_intent",
                            return_false,
                        ):
                            with patch.object(
                                intentHandler,
                                "handle_experiment_control_fast_intent",
                                return_false,
                            ):
                                with patch.object(
                                    intentHandler,
                                    "handle_experiment_control_strict_graph_intent",
                                    handle_strict,
                                ):
                                    handled = await intentHandler.handle_user_intent(
                                        conn,
                                        "继续下一步",
                                    )

        self.assertTrue(handled)
        self.assertEqual([("继续下一步", "继续下一步")], seen)

    async def test_handle_user_intent_routes_imperative_photo_confirmation_to_server_photo(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.device_id = "94:a9:90:27:3c:84"
        conn.config = {
            "device_mcp_shortcuts": {
                "enable_server_photo_confirmation_direct": True,
                "photo_confirm_delay_seconds": 0,
            }
        }
        conn.dialogue.put(
            Message(
                role="assistant",
                content="先把2号样品单独摆好、颜色区域露清楚，再说一声“拍吧”。",
            )
        )

        sent = []
        waits = []
        executed = []

        async def return_false(*args, **kwargs):
            return False

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        async def fake_wait_before_capture(_conn, source):
            waits.append(source)

        async def fake_execute(_conn, arguments):
            executed.append(dict(arguments))
            return True

        with patch.object(
            intentHandler,
            "handle_pending_direct_photo_confirmation",
            return_false,
        ):
            with patch.object(
                intentHandler,
                "handle_direct_photo_navigation_intent",
                return_false,
            ):
                with patch.object(
                    intentHandler,
                    "handle_direct_photo_intent",
                    side_effect=AssertionError("should not fall back to direct photo"),
                ):
                    with patch.object(
                        intentHandler,
                        "send_stt_message",
                        fake_send_stt_message,
                    ):
                        with patch.object(
                            intentHandler,
                            "_maybe_wait_before_photo_capture",
                            fake_wait_before_capture,
                        ):
                            with patch.object(
                                intentHandler,
                                "_execute_server_photo_intent",
                                fake_execute,
                            ):
                                handled = await intentHandler.handle_user_intent(
                                    conn,
                                    "可以拍照",
                                )

        self.assertTrue(handled)
        self.assertEqual(["可以拍照"], sent)
        self.assertEqual(["server_photo_confirmation"], waits)
        self.assertEqual(1, len(executed))
        self.assertEqual("2号样品", executed[0]["photo_name"])
        self.assertEqual(
            "请拍摄2号样品当前状态的照片。",
            executed[0]["question"],
        )

    async def test_server_photo_confirmation_direct_still_works_when_experiment_fast_path_is_disabled(self):
        conn = _FakeConn()
        conn.device_id = "94:a9:90:27:3c:84"
        conn.config = {
            "experiment_fast_path_enabled": False,
            "device_mcp_shortcuts": {
                "enable_server_photo_confirmation_direct": True,
                "photo_confirm_delay_seconds": 0,
            },
        }

        sent = []
        executed = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        async def fake_wait_before_capture(_conn, source):
            executed.append(("delay", source))

        async def fake_execute(_conn, arguments):
            executed.append(("execute", dict(arguments)))
            return True

        with patch.object(
            intentHandler,
            "_assistant_is_waiting_for_photo_permission_fixed",
            return_value=True,
        ):
            with patch.object(
                intentHandler,
                "_is_affirmative_short_reply_fixed",
                return_value=True,
            ):
                with patch.object(
                    intentHandler,
                    "send_stt_message",
                    fake_send_stt_message,
                ):
                    with patch.object(
                        intentHandler,
                        "_maybe_wait_before_photo_capture",
                        fake_wait_before_capture,
                    ):
                        with patch.object(
                            intentHandler,
                            "_build_pending_server_photo_request_fixed",
                            return_value={
                                "device_id": conn.device_id,
                                "question": "请拍摄一号样品当前状态的照片。",
                                "photo_name": "一号样品",
                            },
                        ):
                            with patch.object(
                                intentHandler,
                                "_execute_server_photo_intent",
                                fake_execute,
                            ):
                                handled = await intentHandler.handle_pending_server_photo_confirmation(
                                    conn,
                                    "可以拍照",
                                    "可以拍照",
                                )

        self.assertTrue(handled)
        self.assertEqual(["可以拍照"], sent)
        self.assertEqual(
            [
                ("delay", "server_photo_confirmation"),
                (
                    "execute",
                    {
                        "device_id": conn.device_id,
                        "question": "请拍摄一号样品当前状态的照片。",
                        "photo_name": "一号样品",
                    },
                ),
            ],
            executed,
        )

    async def test_repeat_fast_path_uses_cached_step_context(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和磁转子。"],
                    },
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "没听懂",
                    "没听懂",
                )

        self.assertTrue(handled)
        self.assertEqual(["没听懂"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("当前这一步", spoken[0])
        self.assertIn("烧杯编号和磁转子放置", spoken[0])
        self.assertIn("做好后告诉我", spoken[0])

    async def test_explicit_start_guide_reply_includes_experiment_title(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和洁净磁转子，避免污染。"],
                    },
                }
            }
        }
        conn.experiment_overview = {
            "result": {
                "experiment": {
                    "title": "Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究"
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(
            intentHandler,
            "_reset_experiment_fresh_start_context",
            AsyncMock(),
        ) as reset_mock:
            with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_experiment_control_fast_intent(
                        conn,
                        "\u5f00\u59cb\u4eca\u5929\u7684\u5b9e\u9a8c",
                        "\u5f00\u59cb\u4eca\u5929\u7684\u5b9e\u9a8c",
                    )

        self.assertTrue(handled)
        reset_mock.assert_awaited_once()
        self.assertEqual(["\u5f00\u59cb\u4eca\u5929\u7684\u5b9e\u9a8c"], sent)
        self.assertEqual(1, len(spoken))
        self.assertEqual(
            "今天我们做《Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究》。你准备好开始了吗？",
            spoken[0],
        )


    async def test_reset_fresh_start_context_rotates_session_and_clears_stale_state(self):
        conn = _FakeConn()
        conn.device_id = "94:a9:90:27:3c:84"
        conn.user_id = "test"
        conn.prompt = "system prompt"
        conn.chat_session_id = "chat-old"
        conn.model_session_key = "codex:chat-old"
        conn.dialogue.put(Message(role="assistant", content="old assistant state"))
        conn.experiment_session_id = "exp-old"
        conn.experiment_current_step_id = "step_add_agno3_all"
        conn.experiment_overview = {"result": {"experiment": {"title": "old"}}}
        conn.experiment_current_step = {"result": {"step": {"id": "step_add_agno3_all"}}}
        conn.experiment_progress_summary = {"result": {"summary": {}}}

        stale_session = types.SimpleNamespace(closed=False)

        def _close_stale_session():
            stale_session.closed = True

        stale_session.close = _close_stale_session
        conn.llm = types.SimpleNamespace(
            _sessions={
                "codex:chat-old": stale_session,
                "codex:keep": object(),
            }
        )

        prewarm_calls = []

        async def fake_prewarm_experiment_session(trigger="", force=False):
            prewarm_calls.append((trigger, force))
            conn.experiment_session_id = "exp-new"
            conn.experiment_current_step_id = "step_prepare_setup_all"
            return True

        conn.prewarm_experiment_session = fake_prewarm_experiment_session

        with patch.object(
            intentHandler,
            "rotate_session_binding",
            AsyncMock(
                return_value={
                    "chat_session_id": "chat-new",
                    "model_session_key": "codex:chat-new",
                }
            ),
        ) as rotate_mock:
            await intentHandler._reset_experiment_fresh_start_context(conn)

        rotate_mock.assert_awaited_once()
        self.assertEqual("chat-new", conn.chat_session_id)
        self.assertEqual("codex:chat-new", conn.model_session_key)
        self.assertEqual([], conn.dialogue.dialogue)
        self.assertTrue(stale_session.closed)
        self.assertNotIn("codex:chat-old", conn.llm._sessions)
        self.assertEqual([("explicit_fresh_start", True)], prewarm_calls)
        self.assertEqual("exp-new", conn.experiment_session_id)
        self.assertEqual("step_prepare_setup_all", conn.experiment_current_step_id)
        self.assertIsNone(conn.experiment_overview)

    async def test_explicit_resume_request_without_log_requires_restart(self):
        conn = _FakeConn()
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_reset(_conn):
            _conn.experiment_session_id = "exp-new"
            _conn.experiment_current_step_id = "step_prepare_setup_all"

        def fake_prepare_resume_context(*, previous_session_id="", reason=""):
            conn.experiment_resume_log_path = ""
            conn.experiment_resume_latest_current_step_id = ""

        conn._prepare_experiment_resume_recovery_context = fake_prepare_resume_context

        with patch.object(
            intentHandler,
            "_reset_experiment_fresh_start_context",
            fake_reset,
        ):
            with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_experiment_control_fast_intent(
                        conn,
                        "\u7ee7\u7eed\u4e0a\u6b21\u5b9e\u9a8c",
                        "\u7ee7\u7eed\u4e0a\u6b21\u5b9e\u9a8c",
                    )

        self.assertTrue(handled)
        self.assertEqual(["\u7ee7\u7eed\u4e0a\u6b21\u5b9e\u9a8c"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("\u6ca1\u6709\u627e\u5230", spoken[0])
        self.assertIn("\u91cd\u65b0\u5f00\u59cb", spoken[0])

    async def test_explicit_resume_request_replays_logged_confirmation_steps(self):
        conn = _FakeConn()
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_reset(_conn):
            _conn.experiment_session_id = "exp-new"
            _conn.experiment_current_step_id = "step_prepare_setup_all"
            _conn.experiment_current_step = None
            _conn.experiment_progress_summary = None

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "device.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-05-05T14:35:16.192+08:00] [TRANSCRIPT] [USER] [source=asr] "
                        "[experiment_session_id=exp-old] [current_step_id=step_prepare_setup_all] "
                        "[yaml=C:\\demo\\experiments.yaml] 全部完成。",
                        "[2026-05-05T14:36:16.192+08:00] [TRANSCRIPT] [USER] [source=asr] "
                        "[experiment_session_id=exp-old] [current_step_id=step_add_sodium_citrate_all] "
                        "[yaml=C:\\demo\\experiments.yaml] 柠檬酸钠都加好了。",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_prepare_resume_context(*, previous_session_id="", reason=""):
                conn.experiment_resume_log_path = str(log_path)
                conn.experiment_resume_latest_current_step_id = "step_add_agno3_all"

            conn._prepare_experiment_resume_recovery_context = fake_prepare_resume_context
            conn._experiment_yaml_steps_cache = [
                {"id": "step_prepare_setup_all"},
                {"id": "step_add_sodium_citrate_all"},
                {"id": "step_add_agno3_all"},
            ]

            step_titles = {
                "step_prepare_setup_all": "1-5号样品：准备烧杯与磁转子",
                "step_add_sodium_citrate_all": "1-5号样品：统一加入柠檬酸钠",
                "step_add_agno3_all": "1-5号样品：统一加入AgNO3",
            }
            step_instructions = {
                "step_prepare_setup_all": "完成 1-5 号烧杯编号和磁转子放置。",
                "step_add_sodium_citrate_all": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                "step_add_agno3_all": "按 1 到 5 号顺序统一完成 AgNO3 加入。",
            }
            required_fields = {
                "step_prepare_setup_all": [
                    "magnetic_stirrers_placed",
                    "beakers_labeled",
                ],
                "step_add_sodium_citrate_all": ["sodium_citrate_added"],
                "step_add_agno3_all": [],
            }
            schema_view = {
                "step_prepare_setup_all": [
                    {
                        "name": "magnetic_stirrers_placed",
                        "type": "bool",
                        "description": "为 1-5 号烧杯全部放入磁转子",
                        "required": True,
                    },
                    {
                        "name": "beakers_labeled",
                        "type": "bool",
                        "description": "完成 1-5 号烧杯编号",
                        "required": True,
                    },
                ],
                "step_add_sodium_citrate_all": [
                    {
                        "name": "sodium_citrate_added",
                        "type": "bool",
                        "description": "1-5 号样品统一加入柠檬酸钠",
                        "required": True,
                    }
                ],
                "step_add_agno3_all": [],
            }
            state = {
                "current_index": 0,
                "filled": {
                    "step_prepare_setup_all": set(),
                    "step_add_sodium_citrate_all": set(),
                    "step_add_agno3_all": set(),
                },
                "add_fields_calls": [],
            }

            def current_step_id():
                return conn._experiment_yaml_steps_cache[state["current_index"]]["id"]

            def build_step_payload(step_id):
                interaction = {}
                if step_id != "step_add_agno3_all":
                    interaction = {"fast_path_mode": "confirmation_step"}
                return {
                    "result": {
                        "step": {
                            "id": step_id,
                            "title": step_titles[step_id],
                            "prompts": {
                                "instruction": step_instructions[step_id],
                                "safety": [],
                            },
                            "interaction": interaction,
                        }
                    }
                }

            def build_progress_payload(step_id):
                missing = [
                    field
                    for field in required_fields[step_id]
                    if field not in state["filled"][step_id]
                ]
                return {"result": {"current_progress": {"missing_fields": missing}}}

            def build_summary_payload(step_id):
                return {
                    "result": {
                        "summary": {
                            "current_step": {
                                "step_id": step_id,
                                "title": step_titles[step_id],
                            },
                            "current_step_details": {
                                "title": step_titles[step_id],
                                "instruction": step_instructions[step_id],
                            },
                        }
                    }
                }

            async def fake_graph_tool(_conn, tool_name, payload, priority="foreground"):
                step_id = current_step_id()
                if tool_name == "get_step":
                    return build_step_payload(step_id)
                if tool_name == "get_current_progress":
                    return build_progress_payload(step_id)
                if tool_name == "get_schema":
                    return {"result": {"schema_view": schema_view[step_id]}}
                if tool_name == "add_fields":
                    state["add_fields_calls"].append((step_id, dict(payload["data"])))
                    state["filled"][step_id].update(payload["data"].keys())
                    return build_progress_payload(step_id)
                if tool_name == "finish_trial":
                    return {
                        "result": {
                            "ok": not build_progress_payload(step_id)["result"][
                                "current_progress"
                            ]["missing_fields"]
                        }
                    }
                if tool_name == "can_proceed":
                    return {
                        "result": {
                            "ok": not build_progress_payload(step_id)["result"][
                                "current_progress"
                            ]["missing_fields"]
                        }
                    }
                if tool_name == "proceed_to_next_step":
                    missing = build_progress_payload(step_id)["result"]["current_progress"][
                        "missing_fields"
                    ]
                    if missing or state["current_index"] >= 2:
                        return {"result": {"ok": False, "message": "cannot proceed"}}
                    state["current_index"] += 1
                    conn.experiment_current_step_id = current_step_id()
                    return {"result": {"ok": True}}
                if tool_name == "get_progress_summary":
                    return build_summary_payload(step_id)
                raise AssertionError(f"unexpected tool call: {tool_name}")

            with patch.object(
                intentHandler,
                "_reset_experiment_fresh_start_context",
                fake_reset,
            ):
                with patch.object(
                    intentHandler,
                    "_try_redirect_experiment_step_fast",
                    AsyncMock(
                        side_effect=AssertionError(
                            "explicit resume should rebuild from log instead of redirect"
                        )
                    ),
                ):
                    with patch.object(
                        intentHandler,
                        "_call_experiment_graph_tool_fast",
                        AsyncMock(side_effect=fake_graph_tool),
                    ):
                        with patch.object(
                            intentHandler, "send_stt_message", fake_send_stt_message
                        ):
                            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                                handled = await intentHandler.handle_experiment_control_fast_intent(
                                    conn,
                                    "\u7ee7\u7eed\u4e0a\u6b21\u5b9e\u9a8c",
                                    "\u7ee7\u7eed\u4e0a\u6b21\u5b9e\u9a8c",
                                )

        self.assertTrue(handled)
        self.assertEqual(["\u7ee7\u7eed\u4e0a\u6b21\u5b9e\u9a8c"], sent)
        self.assertEqual(
            [
                (
                    "step_prepare_setup_all",
                    {
                        "magnetic_stirrers_placed": True,
                        "beakers_labeled": True,
                    },
                ),
                (
                    "step_add_sodium_citrate_all",
                    {"sodium_citrate_added": True},
                ),
            ],
            state["add_fields_calls"],
        )
        self.assertEqual("step_add_agno3_all", conn.experiment_current_step_id)
        self.assertEqual(1, len(spoken))
        self.assertIn("AgNO3", spoken[0])

    async def test_ready_reply_after_start_prompt_returns_current_step(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做《Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究》。你准备好开始了吗？",
            )
        )
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和洁净磁转子，避免污染。"],
                    },
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "我准备好了",
                    "我准备好了",
                )

        self.assertTrue(handled)
        self.assertEqual(["我准备好了"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("现在做这一步", spoken[0])
        self.assertIn("烧杯编号和磁转子放置", spoken[0])

    async def test_ready_reply_after_start_prompt_does_not_advance_with_session_id(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做《Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究》。你准备好开始了吗？",
            )
        )
        conn.experiment_session_id = "exp-ready-1"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和洁净磁转子，避免污染。"],
                    },
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(
            intentHandler,
            "_sync_experiment_graph_forward_to_recent_context",
            side_effect=AssertionError("ready reply should not sync graph"),
        ):
            with patch.object(
            intentHandler,
            "_advance_experiment_step_fast",
            side_effect=AssertionError("ready reply should not advance step"),
            ):
                with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_experiment_control_fast_intent(
                            conn,
                            "准备好了",
                            "准备好了",
                        )

        self.assertTrue(handled)
        self.assertEqual(["准备好了"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("现在做这一步", spoken[0])
        self.assertEqual(
            conn.sentence_id,
            getattr(conn, "_experiment_ready_guard_bypass_sentence_id", ""),
        )

    async def test_start_first_step_after_start_prompt_returns_current_step(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做《Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究》。你准备好开始了吗？",
            )
        )
        conn.experiment_session_id = "exp-ready-2"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和洁净磁转子，避免污染。"],
                    },
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(
            intentHandler,
            "_sync_experiment_graph_forward_to_recent_context",
            side_effect=AssertionError("start-first-step reply should not sync graph"),
        ):
            with patch.object(
            intentHandler,
            "_advance_experiment_step_fast",
            side_effect=AssertionError("start-first-step reply should not advance step"),
            ):
                with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_experiment_control_fast_intent(
                            conn,
                            "开始第一步",
                            "开始第一步",
                        )

        self.assertTrue(handled)
        self.assertEqual(["开始第一步"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("现在做这一步", spoken[0])
        self.assertIn("烧杯编号和磁转子放置", spoken[0])
        self.assertEqual(
            conn.sentence_id,
            getattr(conn, "_experiment_ready_guard_bypass_sentence_id", ""),
        )

    async def test_continue_after_start_prompt_returns_current_step(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="今天我们做《Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究》。你准备好开始了吗？",
            )
        )
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和洁净磁转子，避免污染。"],
                    },
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "继续",
                    "继续",
                )

        self.assertTrue(handled)
        self.assertEqual(["继续"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("现在做这一步", spoken[0])
        self.assertIn("烧杯编号和磁转子放置", spoken[0])

    async def test_repeat_current_step_does_not_sync_graph(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="现在做这一步：完成 1-5 号样品的烧杯编号和磁转子放置。做好后告诉我。",
            )
        )
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                        "safety": ["使用洁净烧杯和洁净磁转子，避免污染。"],
                    },
                }
            }
        }
        spoken = []
        sent = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(
            intentHandler,
            "_sync_experiment_graph_forward_to_recent_context",
            side_effect=AssertionError("repeat reply should not sync graph"),
        ):
            with patch.object(
                intentHandler,
                "_advance_experiment_step_fast",
                side_effect=AssertionError("repeat reply should not advance step"),
            ):
                with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_experiment_control_fast_intent(
                            conn,
                            "再说一遍",
                            "再说一遍",
                        )

        self.assertTrue(handled)
        self.assertEqual(["再说一遍"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("当前这一步", spoken[0])
        self.assertIn("烧杯编号", spoken[0])

    async def test_generic_done_after_current_step_advances_one_step_without_context_sync(self):
        conn = _FakeConn()
        conn.experiment_session_id = "exp-generic-done-1"
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5号样品：准备烧杯与磁转子",
                    "prompts": {
                        "instruction": "先完成 1-5 号样品的烧杯编号和磁转子放置，做好后告诉我。",
                    },
                }
            }
        }
        conn.dialogue.put(
            Message(
                role="assistant",
                content="现在做这一步：1-5号样品：准备烧杯与磁转子。先完成 1-5 号样品的烧杯编号和磁转子放置，做好后告诉我。",
            )
        )
        spoken = []
        sent = []
        advanced = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_advance(_conn, session_id):
            advanced.append(session_id)
            return "现在做这一步：1-5号样品：统一加入枸橼酸钠。按 1 到 5 号顺序加入 0.50 mL 0.05 mol/L 枸橼酸钠，做好后告诉我。"

        with patch.object(
            intentHandler,
            "_sync_experiment_graph_forward_to_recent_context",
            side_effect=AssertionError("generic done should not sync future context"),
        ):
            with patch.object(
                intentHandler,
                "_advance_experiment_step_fast",
                fake_advance,
            ):
                with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_experiment_control_fast_intent(
                            conn,
                            "做好了",
                            "做好了",
                        )

        self.assertTrue(handled)
        self.assertEqual(["做好了"], sent)
        self.assertEqual(["exp-generic-done-1"], advanced)
        self.assertEqual(1, len(spoken))
        self.assertIn("统一加入枸橼酸钠", spoken[0])

    async def test_advance_fast_path_records_and_moves_to_next_step(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"can_proceed_calls": 0, "get_step_calls": 0}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                state["get_step_calls"] += 1
                if state["get_step_calls"] == 1:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁转子",
                                "prompts": {
                                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5号样品：统一加入柠檬酸钠",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {"result": {"ok": True, "progress": None}}
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "beakers_labeled",
                                "type": "bool",
                                "description": "已完成 1-5 号烧杯编号",
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "已为 1-5 号烧杯全部放入磁转子",
                            },
                        ],
                    }
                }
            if tool_name == "can_proceed":
                state["can_proceed_calls"] += 1
                return {"result": {"ok": False, "message": "尚未完成"}}
            if tool_name == "start_trial":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": [
                                "beakers_labeled",
                                "stir_bars_added_to_all",
                            ]
                        },
                    }
                }
            if tool_name == "add_fields":
                raise AssertionError("strict mode should not autofill current-step confirmations")
            if tool_name == "finish_trial":
                raise AssertionError("strict mode should not finish while current-step fields are missing")
            if tool_name == "proceed_to_next_step":
                raise AssertionError("strict mode should not advance while current-step fields are missing")
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "当前步骤已完成",
                    "当前步骤已完成",
                )

        self.assertTrue(handled)
        self.assertEqual(["当前步骤已完成"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("这几个确认", spoken[0])
        self.assertIn("烧杯编号", spoken[0])
        self.assertIn("磁转子", spoken[0])
        self.assertTrue(conn.enriched)
        self.assertNotIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertNotIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_confirmation_statement_reports_missing_confirmation_field_instead_of_advancing(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"get_step_calls": 0, "can_proceed_calls": 0, "reported": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                state["get_step_calls"] += 1
                if state["get_step_calls"] <= 2:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                                "interaction": {
                                    "fast_path_mode": "confirmation_step",
                                    "capabilities": ["procedural_guidance", "step_confirmation"],
                                },
                                "prompts": {
                                    "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5号样品：统一加入AgNO3",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": (
                                [] if state["reported"] else ["sodium_citrate_added_to_all"]
                            )
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "sodium_citrate_added_to_all",
                                "type": "bool",
                                "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                            }
                        ],
                    }
                }
            if tool_name == "can_proceed":
                state["can_proceed_calls"] += 1
                return {
                    "result": {
                        "ok": state["reported"],
                        "message": None if state["reported"] else "尚未完成",
                    }
                }
            if tool_name == "add_fields":
                state["reported"] = True
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "proceed_to_next_step":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_agno3_all",
                                "title": "1-5号样品：统一加入AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "按一到五号顺序完成全部柠檬酸钠加入",
                    "按一到五号顺序完成全部柠檬酸钠加入",
                )

        self.assertTrue(handled)
        self.assertEqual(["按一到五号顺序完成全部柠檬酸钠加入"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("一个确认", spoken[0])
        self.assertIn("柠檬酸钠加入", spoken[0])
        self.assertNotIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertNotIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_confirmation_statement_without_interaction_metadata_reports_missing_field(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"get_step_calls": 0, "can_proceed_calls": 0}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                state["get_step_calls"] += 1
                if state["get_step_calls"] <= 2:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                                "prompts": {
                                    "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5号样品：统一加入AgNO3",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": ["sodium_citrate_added_to_all"]
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "sodium_citrate_added_to_all",
                                "type": "bool",
                                "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                            }
                        ],
                    }
                }
            if tool_name == "can_proceed":
                state["can_proceed_calls"] += 1
                return {"result": {"ok": False, "message": "尚未完成"}}
            if tool_name == "add_fields":
                raise AssertionError("missing confirmation field should not be autofilled")
            if tool_name == "finish_trial":
                raise AssertionError("missing confirmation field should block finish_trial")
            if tool_name == "proceed_to_next_step":
                raise AssertionError("missing confirmation field should block next step")
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_agno3_all",
                                "title": "1-5号样品：统一加入AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "按一到五号顺序全部加入柠檬酸钠",
                    "按一到五号顺序全部加入柠檬酸钠",
                )

        self.assertTrue(handled)
        self.assertEqual(["按一到五号顺序全部加入柠檬酸钠"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("一个确认", spoken[0])
        self.assertIn("柠檬酸钠加入", spoken[0])
        self.assertNotIn("add_fields", [name for name, _args, _priority in tool_calls])

    def test_infer_experiment_step_id_from_natural_assistant_reagent_instruction(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_prepare_setup_all",
                "title": "1-5号样品：准备烧杯与磁转子",
                "prompts": {
                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                },
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5号样品：统一加入柠檬酸钠",
                "prompts": {
                    "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                },
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5号样品：统一加入AgNO3",
                "prompts": {
                    "instruction": "按 1 到 5 号顺序统一完成 AgNO3 加入（每个烧杯均为 5.00 mL）。",
                },
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5号样品：统一加入H2O2",
                "prompts": {
                    "instruction": "按 1 到 5 号顺序统一完成 H2O2 加入。",
                },
            },
        ]
        conn.dialogue.put(
            Message(
                role="assistant",
                content="按1到5号顺序，给每个烧杯各加入五点零零毫升硝酸银溶液，注意不要溅出、编号别弄混，全部加完告诉我。",
            )
        )

        inferred = intentHandler._infer_experiment_step_id_from_context(
            conn,
            original_text="全部加好了",
            filtered_text="全部加好了",
        )

        self.assertEqual("step_add_agno3_all", inferred)

    def test_infer_experiment_step_id_ignores_generic_user_advance_text(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_prepare_setup_all",
                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                "prompts": {
                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?",
                },
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?",
                },
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 AgNO3 鍔犲叆銆?",
                },
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆H2O2",
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 H2O2 鍔犲叆銆?",
                },
            },
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘村苟鍔犲叆NaBH4",
                "prompts": {
                    "instruction": "瀹屾垚 1 鍙锋牱鍝?KBr 鍜岀函姘村姞鍏ュ苟娣峰寑鍚庯紝蹇€熷姞鍏?NaBH4銆?",
                },
            },
        ]
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鐜板湪鍋氳繖涓€姝ワ細瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆傚仛濂藉悗鍛婅瘔鎴戙€?",
            )
        )

        inferred = intentHandler._infer_experiment_step_id_from_context(
            conn,
            original_text="缁х画涓嬩竴姝?",
            filtered_text="缁х画涓嬩竴姝?",
        )

        self.assertEqual("", inferred)

    async def test_fast_path_does_not_sync_stale_graph_from_future_context(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {
            "current_idx": 0,
            "ready_to_finish": set(),
            "completed": set(),
        }
        steps = [
            {
                "id": "step_prepare_setup_all",
                "title": "1-5号样品：准备烧杯与磁转子",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "beakers_labeled": {
                        "type": "bool",
                        "description": "已完成 1-5 号烧杯编号",
                    },
                    "stir_bars_added_to_all": {
                        "type": "bool",
                        "description": "已为 1-5 号烧杯全部放入磁转子",
                    },
                },
                "prompts": {
                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                },
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5号样品：统一加入柠檬酸钠",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "sodium_citrate_added_to_all": {
                        "type": "bool",
                        "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                    },
                },
                "prompts": {
                    "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                },
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5号样品：统一加入AgNO3",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "agno3_added_to_all": {
                        "type": "bool",
                        "description": "已按 1-5 号顺序完成全部 AgNO3 加入",
                    },
                },
                "prompts": {
                    "instruction": "按 1 到 5 号顺序统一完成 AgNO3 加入。",
                },
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5号样品：统一加入H2O2",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "h2o2_added_to_all": {
                        "type": "bool",
                        "description": "已按 1-5 号顺序完成全部 H2O2 加入",
                    },
                },
                "prompts": {
                    "instruction": "按 1 到 5 号顺序统一完成 H2O2 加入。",
                },
            },
        ]
        order = [step["id"] for step in steps]
        step_by_id = {step["id"]: step for step in steps}
        conn._experiment_yaml_steps_cache = steps
        conn.dialogue.put(
            Message(
                role="assistant",
                content="按1到5号顺序，给每个烧杯各加入五点零零毫升硝酸银溶液，注意不要溅出、编号别弄混，全部加完告诉我。",
            )
        )

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        def current_step():
            return steps[state["current_idx"]]

        def current_missing_fields():
            step = current_step()
            if step["id"] in state["completed"]:
                return []
            return list(step.get("record_schema", {}).keys())

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            step = current_step()
            if tool_name == "get_step":
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": step["id"],
                            "title": step["title"],
                            "interaction": step.get("interaction", {}),
                            "prompts": step.get("prompts", {}),
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {"result": {"ok": True, "progress": None}}
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {"name": name, **field}
                            for name, field in step.get("record_schema", {}).items()
                        ],
                    }
                }
            if tool_name == "can_proceed":
                return {
                    "result": {
                        "ok": step["id"] in state["completed"],
                        "message": None if step["id"] in state["completed"] else "尚未完成",
                    }
                }
            if tool_name == "start_trial":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": current_missing_fields(),
                        },
                    }
                }
            if tool_name == "add_fields":
                raise AssertionError("strict mode should not autofill stale confirmation steps")
            if tool_name == "finish_trial":
                raise AssertionError("strict mode should not finish unresolved stale steps")
            if tool_name == "proceed_to_next_step":
                raise AssertionError("strict mode should not sync to future steps")
            if tool_name == "get_progress_summary":
                current = current_step()
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": current["id"],
                                "title": current["title"],
                            },
                            "current_step_details": {
                                "instruction": current.get("prompts", {}).get("instruction", ""),
                            },
                        },
                    }
                }
            if tool_name == "redirect_to_step":
                state["current_idx"] = order.index(arguments["step_id"])
                return {"result": {"ok": True}}
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "\u5168\u90e8\u52a0\u597d\u4e86",
                    "\u5168\u90e8\u52a0\u597d\u4e86",
                )

        self.assertTrue(handled)
        self.assertEqual(["\u5168\u90e8\u52a0\u597d\u4e86"], sent)
        self.assertEqual("step_prepare_setup_all", conn.experiment_current_step_id)
        self.assertEqual(1, len(spoken))
        self.assertIn("这几个确认", spoken[0])
        self.assertIn("烧杯编号", spoken[0])
        self.assertIn("磁转子", spoken[0])
        self.assertNotIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_observation_step_reports_missing_fields_without_autofill(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {
            "current_idx": 5,
            "ready_to_finish": set(),
            "completed": set(),
        }
        steps = [
            {
                "id": "step_prepare_setup_all",
                "title": "1-5号样品：准备烧杯与磁转子",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "beakers_labeled": {
                        "type": "bool",
                        "description": "已完成 1-5 号烧杯编号",
                    },
                    "stir_bars_added_to_all": {
                        "type": "bool",
                        "description": "已为 1-5 号烧杯全部放入磁转子",
                    },
                },
                "prompts": {"instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。"},
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5号样品：统一加入柠檬酸钠",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "sodium_citrate_added_to_all": {
                        "type": "bool",
                        "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                    },
                },
                "prompts": {"instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。"},
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5号样品：统一加入AgNO3",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "agno3_added_to_all": {
                        "type": "bool",
                        "description": "已按 1-5 号顺序完成全部 AgNO3 加入",
                    },
                },
                "prompts": {"instruction": "按 1 到 5 号顺序统一完成 AgNO3 加入。"},
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5号样品：统一加入H2O2",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "h2o2_added_to_all": {
                        "type": "bool",
                        "description": "已按 1-5 号顺序完成全部 H2O2 加入",
                    },
                },
                "prompts": {"instruction": "按 1 到 5 号顺序统一完成 H2O2 加入。"},
            },
            {
                "id": "step_stirring_all",
                "title": "1-5号样品：同时启动搅拌并混匀",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "all_samples_stirring": {
                        "type": "bool",
                        "description": "已同时启动 1-5 号样品搅拌并确认混匀",
                    },
                },
                "prompts": {"instruction": "同时启动 1-5 号样品搅拌并确认混匀。"},
            },
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1号样品：加入KBr、纯水并加入NaBH4",
                "interaction": {
                    "fast_path_mode": "observation_record_step",
                    "capabilities": ["step_confirmation", "observation_capture"],
                },
                "record_schema": {
                    "KBr_volume": {
                        "type": "bool",
                        "description": "已按当前样品目标用量加入 KBr",
                    },
                    "H2O_volume": {
                        "type": "bool",
                        "description": "已按当前样品目标用量加入纯水",
                    },
                    "mixed_uniformly": {
                        "type": "bool",
                        "description": "加入 KBr 和纯水后已搅拌均匀",
                    },
                    "nabh4_volume": {
                        "type": "bool",
                        "description": "已准确加入 2.50 mL NaBH4",
                    },
                    "added_quickly": {
                        "type": "bool",
                        "description": "已快速完成 NaBH4 加入",
                    },
                    "color": {
                        "type": "string",
                        "description": "当前样品最终颜色",
                    },
                    "reaction_time": {
                        "type": "float",
                        "description": "当前样品颜色稳定所用时间",
                    },
                    "color_stable": {
                        "type": "bool",
                        "description": "已确认颜色稳定",
                    },
                },
                "prompts": {
                    "instruction": "完成 1 号样品 KBr 和纯水加入并混匀后，快速加入 NaBH4 并保持搅拌，记录颜色稳定时间和收尾情况。",
                },
            },
        ]
        order = [step["id"] for step in steps]
        conn._experiment_yaml_steps_cache = steps
        conn.dialogue.put(
            Message(
                role="assistant",
                content="接着做1号样品：向1号烧杯加入二点五零毫升硼氢化钠，保持搅拌，注意它有腐蚀性、现配后容易分解，尽量快加，做好告诉我已经做好了。",
            )
        )

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        def current_step():
            return steps[state["current_idx"]]

        def current_missing_fields():
            step_id = current_step()["id"]
            if step_id == "step_sample1_2_add_kbr_water_nabh4":
                if step_id in state["ready_to_finish"]:
                    return ["color", "reaction_time", "color_stable"]
                return [
                    "KBr_volume",
                    "H2O_volume",
                    "mixed_uniformly",
                    "nabh4_volume",
                    "added_quickly",
                    "color",
                    "reaction_time",
                    "color_stable",
                ]
            if step_id in state["completed"]:
                return []
            return list(current_step().get("record_schema", {}).keys())

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            step = current_step()
            if tool_name == "get_step":
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": step["id"],
                            "title": step["title"],
                            "interaction": step.get("interaction", {}),
                            "prompts": step.get("prompts", {}),
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {"result": {"ok": True, "progress": None}}
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {"name": name, **field}
                            for name, field in step.get("record_schema", {}).items()
                        ],
                    }
                }
            if tool_name == "can_proceed":
                return {
                    "result": {
                        "ok": step["id"] in state["completed"],
                        "message": None if step["id"] in state["completed"] else "尚未完成",
                    }
                }
            if tool_name == "start_trial":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": current_missing_fields(),
                        },
                    }
                }
            if tool_name == "add_fields":
                raise AssertionError("observation step should not autofill missing fields")
            if tool_name == "finish_trial":
                raise AssertionError("observation step should not finish while fields are missing")
            if tool_name == "proceed_to_next_step":
                raise AssertionError("observation step should not advance while fields are missing")
            if tool_name == "get_progress_summary":
                current = current_step()
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": current["id"],
                                "title": current["title"],
                            },
                            "current_step_details": {
                                "instruction": current.get("prompts", {}).get("instruction", ""),
                            },
                        },
                    }
                }
            if tool_name == "redirect_to_step":
                state["current_idx"] = order.index(arguments["step_id"])
                return {"result": {"ok": True}}
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "已经做好了",
                    "已经做好了",
                )

        self.assertTrue(handled)
        self.assertEqual(["已经做好了"], sent)
        self.assertEqual("step_sample1_2_add_kbr_water_nabh4", conn.experiment_current_step_id)
        self.assertEqual(1, len(spoken))
        self.assertIn("一整组关键记录", spoken[0])
        self.assertNotIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_local_photo_followup_writes_back_and_moves_to_next_step(self):
        conn = _FakeConn()
        conn.experiment_resume_recovery_required = True
        conn.experiment_resume_latest_current_step_id = "step_prepare_setup_all"
        tool_calls = []
        state = {"get_step_calls": 0}

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                state["get_step_calls"] += 1
                if state["get_step_calls"] == 1:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_photo_confirm_sample_5",
                                "title": "五号样品拍照确认",
                                "prompts": {
                                    "instruction": "确认五号样品颜色稳定后拍照。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_tyndall_observation",
                            "title": "丁达尔现象观察",
                            "prompts": {
                                "instruction": "用激光笔从侧面观察 1 到 5 号样品的光路。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": [
                                "photo_taken",
                                "color_confirmed_by_photo",
                                "photo_file_name",
                                "photo_path",
                            ]
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {"name": "photo_taken", "type": "bool"},
                            {"name": "color_confirmed_by_photo", "type": "bool"},
                            {"name": "photo_file_name", "type": "string"},
                            {"name": "photo_path", "type": "string"},
                        ],
                    }
                }
            if tool_name == "add_fields":
                self.assertEqual(
                    {
                        "photo_taken": True,
                        "color_confirmed_by_photo": True,
                        "photo_file_name": "五号样品_20260428_175720.png",
                        "photo_path": "C:/demo/五号样品_20260428_175720.png",
                    },
                    arguments["data"],
                )
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": True}}
            if tool_name == "can_proceed":
                return {"result": {"ok": True}}
            if tool_name == "proceed_to_next_step":
                return {"result": {"ok": True}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_tyndall_observation",
                                "title": "丁达尔现象观察",
                            },
                            "current_step_details": {
                                "instruction": "用激光笔从侧面观察 1 到 5 号样品的光路。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call
        reply = await intentHandler._advance_photo_confirmation_step_locally(
            conn,
            {
                "photo_meta": {
                    "found": True,
                    "file_name": "五号样品_20260428_175720.png",
                    "mirrored_path": "C:/demo/五号样品_20260428_175720.png",
                }
            },
            fallback_reply="拍好了。",
        )

        self.assertIn("拍好了", reply)
        self.assertIn("我接着带你做下一步", reply)
        self.assertIn("接下来做这一步", reply)
        self.assertIn("丁达尔现象观察", reply)
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )
        self.assertEqual("step_tyndall_observation", conn.experiment_current_step_id)
        self.assertEqual(
            "step_tyndall_observation",
            conn.experiment_resume_latest_current_step_id,
        )

    async def test_local_photo_followup_redirects_stale_graph_to_inferred_photo_step(self):
        conn = _FakeConn()
        conn.experiment_resume_recovery_required = True
        conn.experiment_resume_latest_current_step_id = "step_prepare_setup_all"
        tool_calls = []
        state = {"redirected": False, "redirect_step_id": "", "redirect_step_reads": 0}

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                if not state["redirected"]:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁转子",
                                "prompts": {
                                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                                },
                            },
                        }
                    }
                state["redirect_step_reads"] += 1
                if state["redirect_step_reads"] == 1:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_sample1_5_photo_confirm",
                                "title": "1号样品：颜色稳定后拍照记录",
                                "interaction": {
                                    "fast_path_mode": "photo_confirmation_step",
                                },
                                "prompts": {
                                    "instruction": "颜色稳定后拍照记录当前样品颜色，并进入 2 号样品。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_sample2_add_kbr_water_nabh4",
                            "title": "2号样品：加入溴化钾、纯水并加入硼氢化钠",
                            "prompts": {
                                "instruction": "现在做 2 号样品，先加溴化钾和纯水，再加入硼氢化钠。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                if not state["redirected"]:
                    return {
                        "result": {
                            "ok": True,
                            "current_progress": {
                                "missing_fields": ["beakers_labeled"]
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": [
                                "photo_taken",
                                "color_confirmed_by_photo",
                                "photo_file_name",
                                "photo_path",
                            ]
                        },
                    }
                }
            if tool_name == "get_schema":
                if not state["redirected"]:
                    return {
                        "result": {
                            "ok": True,
                            "schema_view": [
                                {"name": "beakers_labeled", "type": "bool"},
                            ],
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {"name": "photo_taken", "type": "bool"},
                            {"name": "color_confirmed_by_photo", "type": "bool"},
                            {"name": "photo_file_name", "type": "string"},
                            {"name": "photo_path", "type": "string"},
                        ],
                    }
                }
            if tool_name == "redirect_to_step":
                state["redirected"] = True
                state["redirect_step_id"] = arguments["step_id"]
                return {"result": {"ok": True}}
            if tool_name == "add_fields":
                self.assertEqual(
                    {
                        "photo_taken": True,
                        "color_confirmed_by_photo": True,
                        "photo_file_name": "1号样品_20260430_195415.png",
                        "photo_path": "C:/demo/1号样品_20260430_195415.png",
                    },
                    arguments["data"],
                )
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": True}}
            if tool_name == "can_proceed":
                return {"result": {"ok": True}}
            if tool_name == "proceed_to_next_step":
                return {"result": {"ok": True}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_sample2_add_kbr_water_nabh4",
                                "title": "2号样品：加入溴化钾、纯水并加入硼氢化钠",
                            },
                            "current_step_details": {
                                "instruction": "现在做 2 号样品，先加溴化钾和纯水，再加入硼氢化钠。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(
            intentHandler,
            "_infer_photo_confirmation_step_id_from_context",
            return_value="step_sample1_5_photo_confirm",
        ):
            reply = await intentHandler._advance_photo_confirmation_step_locally(
                conn,
                {
                    "photo_meta": {
                        "found": True,
                        "file_name": "1号样品_20260430_195415.png",
                        "mirrored_path": "C:/demo/1号样品_20260430_195415.png",
                    }
                },
                fallback_reply="拍好了，已经保存。",
                requested_arguments={"photo_name": "1号样品"},
            )

        self.assertEqual("step_sample1_5_photo_confirm", state["redirect_step_id"])
        self.assertIn("拍好了", reply)
        self.assertIn("我接着带你做下一步", reply)
        self.assertIn("2号样品", reply)
        self.assertIn(
            "redirect_to_step",
            [name for name, _args, _priority in tool_calls],
        )
        self.assertEqual(
            "step_sample2_add_kbr_water_nabh4",
            conn.experiment_current_step_id,
        )
        self.assertEqual(
            "step_sample2_add_kbr_water_nabh4",
            conn.experiment_resume_latest_current_step_id,
        )
        recent_state = getattr(conn, "_recent_server_photo_confirmation", {})
        self.assertTrue(recent_state.get("graph_advanced"))
        self.assertEqual(1, recent_state.get("sample_index"))

    async def test_local_photo_followup_refreshes_current_step_when_graph_does_not_advance(
        self,
    ):
        conn = _FakeConn()
        tool_calls = []
        state = {"redirected": False, "progress_summary_reads": 0}

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_prepare_setup_all",
                            "title": "1-5号样品：准备烧杯与磁子",
                            "prompts": {
                                "instruction": "先完成 1-5 号样品的烧杯编号和磁子放置。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": ["beakers_labeled"]},
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [{"name": "beakers_labeled", "type": "bool"}],
                    }
                }
            if tool_name == "redirect_to_step":
                state["redirected"] = True
                self.assertEqual("step_sample1_5_photo_confirm", arguments["step_id"])
                return {"result": {"ok": True}}
            if tool_name == "get_progress_summary":
                state["progress_summary_reads"] += 1
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁子",
                            },
                            "current_step_details": {
                                "instruction": "先完成 1-5 号样品的烧杯编号和磁子放置。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(
            intentHandler,
            "_infer_photo_confirmation_step_id_from_context",
            return_value="step_sample1_5_photo_confirm",
        ):
            reply = await intentHandler._advance_photo_confirmation_step_locally(
                conn,
                {
                    "photo_meta": {
                        "found": True,
                        "file_name": "1号样品_20260504_145218.png",
                        "mirrored_path": "C:/demo/1号样品_20260504_145218.png",
                    }
                },
                fallback_reply="拍好了，已经保存。",
                requested_arguments={"photo_name": "1号样品"},
            )

        self.assertTrue(state["redirected"])
        self.assertGreaterEqual(state["progress_summary_reads"], 1)
        self.assertIn(
            "redirect_to_step",
            [name for name, _args, _priority in tool_calls],
        )
        self.assertEqual("step_prepare_setup_all", conn.experiment_current_step_id)
        self.assertIn("拍好了", reply)
        self.assertIn("当前实验图谱还停在这一步", reply)
        self.assertIn("烧杯编号和磁子放置", reply)
        recent_state = getattr(conn, "_recent_server_photo_confirmation", {})
        self.assertFalse(recent_state.get("graph_advanced"))
        self.assertEqual(
            "redirect_did_not_land_on_photo_step",
            recent_state.get("graph_status_reason"),
        )
        self.assertEqual(
            "step_prepare_setup_all",
            recent_state.get("current_step_id"),
        )
        self.assertEqual(
            "1-5号样品：准备烧杯与磁子",
            recent_state.get("current_step_title"),
        )
        self.assertGreater(recent_state.get("graph_refresh_checked_at", 0.0), 0.0)

    async def test_local_photo_followup_redirect_rejection_hides_internal_graph_message(
        self,
    ):
        conn = _FakeConn()
        tool_calls = []
        state = {"step_reads": 0}

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                state["step_reads"] += 1
                if state["step_reads"] == 1:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁转子",
                                "prompts": {
                                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_sample1_2_add_kbr_water_nabh4",
                            "title": "1号样品：加入KBr、纯水并加入NaBH4",
                            "prompts": {
                                "instruction": "先加入 KBr 和纯水，再快速加入 NaBH4 并持续搅拌。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": ["beakers_labeled"]},
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [{"name": "beakers_labeled", "type": "bool"}],
                    }
                }
            if tool_name == "redirect_to_step":
                self.assertEqual("step_sample1_5_photo_confirm", arguments["step_id"])
                return {
                    "result": {
                        "ok": False,
                        "message": "无法跳转到 step_sample1_5_photo_confirm：前置步骤未完成: step_sample1_2_add_kbr_water_nabh4",
                    }
                }
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_sample1_2_add_kbr_water_nabh4",
                                "title": "1号样品：加入KBr、纯水并加入NaBH4",
                            },
                            "current_step_details": {
                                "instruction": "先加入 KBr 和纯水，再快速加入 NaBH4 并持续搅拌。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(
            intentHandler,
            "_infer_photo_confirmation_step_id_from_context",
            return_value="step_sample1_5_photo_confirm",
        ):
            reply = await intentHandler._advance_photo_confirmation_step_locally(
                conn,
                {
                    "photo_meta": {
                        "found": True,
                        "file_name": "1号样品_20260505_171900.png",
                        "mirrored_path": "C:/demo/1号样品_20260505_171900.png",
                    }
                },
                fallback_reply="拍好了，已经保存。",
                requested_arguments={"photo_name": "1号样品"},
            )

        self.assertIn("拍好了", reply)
        self.assertIn("当前实验图谱还停在这一步", reply)
        self.assertIn("KBr", reply)
        self.assertIn("NaBH4", reply)
        self.assertNotIn("前置步骤未完成", reply)
        self.assertNotIn("step_sample1_5_photo_confirm", reply)
        self.assertEqual("step_sample1_2_add_kbr_water_nabh4", conn.experiment_current_step_id)
        self.assertIn(
            "redirect_to_step",
            [name for name, _args, _priority in tool_calls],
        )

    def test_infer_photo_confirmation_step_id_ignores_non_photo_step_mentions(self):
        conn = _FakeConn()
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1号样品：加入KBr、纯水并加入NaBH4",
                "interaction": {
                    "fast_path_mode": "observation_record_step",
                },
                "prompts": {
                    "instruction": (
                        "完成 1 号样品 KBr 和纯水加入并混匀后，快速加入 NaBH4；"
                        "完成后进入本样品拍照记录步骤。"
                    ),
                },
                "record_schema": {
                    "color": {"type": "string"},
                },
            },
            {
                "id": "step_sample1_5_photo_confirm",
                "title": "1号样品：颜色稳定后拍照记录",
                "interaction": {
                    "fast_path_mode": "photo_confirmation_step",
                },
                "prompts": {
                    "instruction": "颜色稳定后拍照记录当前样品颜色，并进入 2 号样品。",
                },
                "record_schema": {
                    "photo_taken": {"type": "bool"},
                    "color_confirmed_by_photo": {"type": "bool"},
                },
            },
        ]
        conn.dialogue.put(
            Message(
                role="assistant",
                content="1号样品颜色已经稳定，现在可以拍照吗？",
            )
        )

        inferred = intentHandler._infer_photo_confirmation_step_id_from_context(
            conn,
            {
                "photo_meta": {
                    "file_name": "1号样品_20260504_145218.png",
                }
            },
            requested_arguments={"photo_name": "1号样品照片"},
        )

        self.assertEqual("step_sample1_5_photo_confirm", inferred)

    def test_infer_photo_confirmation_step_id_uses_recent_dialogue_when_request_is_generic(self):
        conn = _FakeConn()
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_5_photo_confirm",
                "title": "1号样品：颜色稳定后拍照记录",
                "interaction": {
                    "fast_path_mode": "photo_confirmation_step",
                },
                "prompts": {
                    "instruction": "颜色稳定后拍照记录当前样品颜色，并进入 2 号样品。",
                },
                "record_schema": {
                    "photo_taken": {"type": "bool"},
                    "color_confirmed_by_photo": {"type": "bool"},
                },
            },
        ]
        conn.dialogue.put(
            Message(role="user", content="1号样品颜色已经稳定了。")
        )
        conn.dialogue.put(
            Message(role="assistant", content="现在可以拍照吗？")
        )

        inferred = intentHandler._infer_photo_confirmation_step_id_from_context(
            conn,
            {
                "photo_meta": {
                    "file_name": "capture.png",
                }
            },
            requested_arguments={"question": "请拍摄当前样品的照片。"},
        )

        self.assertEqual("step_sample1_5_photo_confirm", inferred)

    def test_build_pending_server_photo_request_fixed_uses_recent_dialogue_sample_name(self):
        conn = _FakeConn()
        conn.device_id = "94:a9:90:27:3c:84"
        conn.dialogue.put(
            Message(role="user", content="1号样品颜色已经稳定了。")
        )
        conn.dialogue.put(
            Message(role="assistant", content="现在可以拍照吗？")
        )

        request = intentHandler._build_pending_server_photo_request_fixed(conn)

        self.assertEqual("94:a9:90:27:3c:84", request["device_id"])
        self.assertEqual("1号样品", request.get("photo_name"))

    async def test_server_photo_timeout_recovery_uses_latest_photo_and_continues(self):
        conn = _FakeConn()
        conn.device_id = "94:a9:90:27:3c:84"
        conn.config = {
            "device_mcp_shortcuts": {
                "photo_timeout": 12,
                "server_photo_recovery_window_seconds": 0,
            }
        }
        spoken = []
        tool_calls = []
        latest_photo_calls = {"count": 0}

        async def fake_execute(_conn, tool_name, arguments):
            tool_calls.append((tool_name, dict(arguments)))
            if tool_name == "xiaozhi_get_latest_photo":
                latest_photo_calls["count"] += 1
                if latest_photo_calls["count"] == 1:
                    return {
                        "success": True,
                        "photo_meta": {
                            "file_name": "旧照片.png",
                            "local_path": "C:/demo/old.png",
                            "mtime": 100.0,
                        },
                    }
                return {
                    "success": True,
                    "photo_meta": {
                        "file_name": "一号样品_20260430_105937.png",
                        "local_path": "C:/demo/new.png",
                        "mtime": 101.0,
                    },
                }
            if tool_name == "xiaozhi_take_photo":
                self.assertEqual(12, arguments["timeout"])
                self.assertEqual(60, arguments["request_timeout"])
                return {"success": False, "message": "tool call timeout"}
            raise AssertionError(f"unexpected tool call: {tool_name}")

        async def fake_advance(
            _conn,
            payload,
            fallback_reply="",
            requested_arguments=None,
        ):
            photo_meta = intentHandler._extract_photo_result_meta(payload)
            self.assertTrue(photo_meta["found"])
            self.assertIn("一号样品", photo_meta["file_name"])
            self.assertEqual("一号样品", photo_meta["requested_photo_name"])
            self.assertEqual("一号样品", requested_arguments["photo_name"])
            return "接下来做这一步：观察颜色。做好后告诉我。"

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_get_server_mcp_manager", return_value=object()):
            with patch.object(
                intentHandler,
                "_execute_server_mcp_tool_direct",
                fake_execute,
            ):
                with patch.object(
                    intentHandler,
                    "_advance_photo_confirmation_step_locally",
                    fake_advance,
                ):
                    with patch.object(
                        intentHandler,
                        "sync_server_mcp_payload_state",
                        lambda *args, **kwargs: None,
                    ):
                        with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                            handled = await intentHandler._execute_server_photo_intent(
                                conn,
                                {
                                    "device_id": conn.device_id,
                                    "question": "请拍摄一号样品当前状态的照片。",
                                    "photo_name": "一号样品",
                                },
                            )

        self.assertTrue(handled)
        self.assertEqual(
            ["接下来做这一步：观察颜色。做好后告诉我。"],
            spoken,
        )
        self.assertEqual(
            ["xiaozhi_get_latest_photo", "xiaozhi_take_photo", "xiaozhi_get_latest_photo"],
            [name for name, _arguments in tool_calls],
        )

    async def test_pending_server_photo_confirmation_reuses_recent_success(self):
        conn = _FakeConn()
        conn.device_id = "94:a9:90:27:3c:84"
        conn.dialogue.put(
            Message(role="assistant", content="现在给1号样品拍照确认。现在可以拍照吗？")
        )
        conn._recent_server_photo_confirmation = {
            "captured_at": time.time(),
            "sample_index": 1,
            "sample_name": "1号样品",
            "next_step_reply": "接下来做这一步：2号样品先加入溴化钾和纯水。",
        }
        sent = []
        spoken = []

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(
            intentHandler,
            "_assistant_is_waiting_for_photo_permission_fixed",
            return_value=True,
        ):
            with patch.object(
                intentHandler,
                "_is_affirmative_short_reply_fixed",
                return_value=True,
            ):
                with patch.object(
                    intentHandler,
                    "_build_pending_server_photo_request_fixed",
                    return_value={
                        "device_id": conn.device_id,
                        "question": "请拍摄1号样品当前状态的照片。",
                        "photo_name": "1号样品",
                    },
                ):
                    with patch.object(
                        intentHandler,
                        "send_stt_message",
                        fake_send_stt_message,
                    ):
                        with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                            with patch.object(
                                intentHandler,
                                "_maybe_wait_before_photo_capture",
                                side_effect=AssertionError("should not retake photo"),
                            ):
                                with patch.object(
                                    intentHandler,
                                    "_execute_server_photo_intent",
                                    side_effect=AssertionError("should not retake photo"),
                                ):
                                    handled = await intentHandler.handle_pending_server_photo_confirmation(
                                        conn,
                                        "可以拍照",
                                        "可以拍照",
                                    )

        self.assertTrue(handled)
        self.assertEqual(["可以拍照"], sent)
        self.assertEqual(
            ["接下来做这一步：2号样品先加入溴化钾和纯水。"],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_blank_prep_calls_measurement_and_waits_for_blank(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "pure water blank missing"}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "开始测量",
                        "开始测量",
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": False,
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                "phase": "await_pure_water_blank",
                "session_key": "lease-1",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(
            [
                "先不要放任何液体，我先进行暗电流和空气基线准备。",
                "这一步还缺纯水空白，请先把 1-5 号样品位和参比位都放入纯水比色皿。放好了告诉我。",
            ],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_blank_prep_accepts_start_scan_phrase(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "pure water blank missing"}

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", lambda *_args, **_kwargs: None):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "开始扫描。",
                        "开始扫描。",
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": False,
                    },
                )
            ],
            executed,
        )

    async def test_handle_direct_uvvis_shared_blank_infers_step_from_context_when_graph_stale(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.dialogue.put(
            Message(
                role="assistant",
                content=(
                    "先不要放任何液体，把样品位和参比位都留空，"
                    "准备做暗电流和空气基线。做好了告诉我。"
                ),
            )
        )
        spoken = []
        executed = []
        redirected = []

        async def fake_call(tool_name, arguments, priority="foreground"):
            if tool_name == "redirect_to_step":
                redirected.append((tool_name, dict(arguments), priority))
                return {"result": {"ok": True}}
            raise AssertionError(f"unexpected graph tool call: {tool_name}")

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "pure water blank missing"}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "已经放好了，可以开始了。",
                        "已经放好了，可以开始了。",
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "redirect_to_step",
                    {
                        "session_id": "exp-1",
                        "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                    },
                    "foreground",
                )
            ],
            redirected,
        )
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": False,
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                "phase": "await_pure_water_blank",
                "session_key": "lease-1",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(intentHandler._UVVIS_SHARED_BLANK_STEP_ID, conn.experiment_current_step_id)
        self.assertEqual(
            [
                "先不要放任何液体，我先进行暗电流和空气基线准备。",
                "这一步还缺纯水空白，请先把 1-5 号样品位和参比位都放入纯水比色皿。放好了告诉我。",
            ],
            spoken,
        )

    async def test_handle_direct_uvvis_followup_start_scan_infers_step_from_context_when_graph_stale(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.dialogue.put(
            Message(
                role="assistant",
                content=(
                    "先不要放任何液体，把样品位和参比位都留空，准备做暗电流和空气基线。"
                    "可以开始扫描时直接告诉我开始扫描。"
                ),
            )
        )
        executed = []
        redirected = []

        async def fake_call(tool_name, arguments, priority="foreground"):
            if tool_name == "redirect_to_step":
                redirected.append((tool_name, dict(arguments), priority))
                return {"result": {"ok": True}}
            raise AssertionError(f"unexpected graph tool call: {tool_name}")

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "pure water blank missing"}

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", lambda *_args, **_kwargs: None):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "开始扫描。",
                        "开始扫描。",
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "redirect_to_step",
                    {
                        "session_id": "exp-1",
                        "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                    },
                    "foreground",
                )
            ],
            redirected,
        )
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": False,
                    },
                )
            ],
            executed,
        )

    async def test_handle_direct_uvvis_status_query_reports_idle_shared_blank_waiting(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
            "phase": "await_pure_water_blank",
        }
        spoken = []
        started = []

        async def fake_start(_conn, original_text):
            started.append(original_text)

        async def fake_status(_conn):
            return {
                "ok": True,
                "available": True,
                "occupied": False,
                "active_measurement": False,
                "lease_owner_is_caller": False,
                "session_key": "",
            }

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_start_direct_intent_turn", fake_start):
            with patch.object(intentHandler, "_get_uvvis_session_status", fake_status):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "Uvvis现在正在工作吗？",
                        "Uvvis现在正在工作吗？",
                    )

        self.assertTrue(handled)
        self.assertEqual(["Uvvis现在正在工作吗？"], started)
        self.assertEqual(
            ["UV-Vis 现在没有在工作。暗电流和空气基线已经完成，这一步在等你把一到五号样品位和参比位各放一个纯水比色皿。"],
            spoken,
        )

    async def test_handle_direct_uvvis_intent_stops_when_redirect_is_rejected(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        spoken = []
        graph_calls = []

        async def fake_call(tool_name, arguments, priority="foreground"):
            graph_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "redirect_to_step":
                return {
                    "result": {
                        "ok": False,
                        "message": "无法跳转到 step_3_uv_vis_shared_dark_blank_prep：前置步骤未完成: step_2_tyndall_effect",
                    }
                }
            raise AssertionError(f"unexpected graph tool call: {tool_name}")

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        conn._call_experiment_graph_tool = fake_call

        with patch.object(
            intentHandler,
            "_infer_uvvis_step_id_from_context",
            return_value=intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
        ):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_direct_uvvis_intent(
                    conn,
                    "开始 UV-Vis 前置校正。",
                    "开始 UV-Vis 前置校正。",
                )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "redirect_to_step",
                    {
                        "session_id": "exp-1",
                        "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                    },
                    "foreground",
                )
            ],
            graph_calls,
        )
        self.assertEqual(
            ["当前实验图谱还没推进到 UV-Vis 前置校正，先完成丁达尔现象观察。"],
            spoken,
        )

    async def test_handle_direct_uvvis_intent_ignores_generic_next_with_stale_uvvis_state(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID,
            "run_name": "sample2",
            "sample_position": 1,
            "phase": "done",
            "session_key": "lease-1",
        }
        conn.dialogue.put(
            Message(
                role="assistant",
                content="拍好了，已经保存。当前实验图谱还停在这一步，先按这一步继续。",
            )
        )

        handled = await intentHandler.handle_direct_uvvis_intent(
            conn,
            "继续下一步。",
            "继续下一步。",
        )

        self.assertFalse(handled)
        self.assertEqual(
            intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID,
            getattr(conn, "_uvvis_direct_state", {}).get("step_id"),
        )

    async def test_handle_direct_uvvis_shared_blank_prep_reuses_blank_and_advances(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        spoken = []
        executed = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "shared dark current reused and pure water blank reused"}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "接下来做这一步：装入比色皿。"

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "开始测量",
                            "开始测量",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": False,
                    },
                )
            ],
            executed,
        )
        self.assertEqual(1, len(completed))
        self.assertEqual(
            {
                "shared_dark_current_ready": True,
                "shared_air_baseline_ready": True,
                "pure_water_blank_ready": True,
                "reference_cuvette_ready": True,
                "observations": "共享暗电流、空气基线和纯水空白已完成或可复用",
            },
            completed[0]["fields"],
        )
        self.assertTrue(completed[0]["auto_advance"])
        self.assertEqual(
            [
                "先不要放任何液体，我先进行暗电流和空气基线准备。",
                "接下来做这一步：装入比色皿。",
            ],
            spoken,
        )

    async def test_handle_direct_uvvis_spectra_measurement_records_all_samples(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID
        spoken = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            self.assertEqual("uvvis_measure_spectra", tool_name)
            self.assertEqual(
                {
                    "session_key": "lease-1",
                    "sample_positions": [1, 2, 3, 4, 5],
                    "ready_for_samples": True,
                },
                arguments,
            )
            return {
                "result": {
                    "samples": [
                        {
                            "sample_position": index,
                            "lambda_max_nm": 400.0 + index * 10,
                            "max_absorbance": round(0.1 * index, 3),
                        }
                        for index in range(1, 6)
                    ]
                }
            }

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "接下来做这一步：清洗比色皿。"

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "都放好了",
                            "都放好了",
                        )

        self.assertTrue(handled)
        self.assertEqual(1, len(completed))
        self.assertEqual(
            {
                "sample_1_lambda_max": 410.0,
                "sample_2_lambda_max": 420.0,
                "sample_3_lambda_max": 430.0,
                "sample_4_lambda_max": 440.0,
                "sample_5_lambda_max": 450.0,
                "sample_1_absorbance_max": 0.1,
                "sample_2_absorbance_max": 0.2,
                "sample_3_absorbance_max": 0.3,
                "sample_4_absorbance_max": 0.4,
                "sample_5_absorbance_max": 0.5,
                "spectrum_saved": True,
                "observations": "1-5号样品批量扫描完成，1号样品λmax=410.0nm；2号样品λmax=420.0nm；3号样品λmax=430.0nm；4号样品λmax=440.0nm；5号样品λmax=450.0nm",
            },
            completed[0]["fields"],
        )
        self.assertTrue(completed[0]["auto_advance"])
        self.assertEqual(1, len(spoken))
        self.assertIn("1号410.0纳米", spoken[0])
        self.assertIn("5号450.0纳米", spoken[0])
        self.assertIn("接下来做这一步：清洗比色皿。", spoken[0])

    async def test_handle_direct_uvvis_spectra_measurement_skips_negative_absorbance(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            self.assertEqual("uvvis_measure_spectra", tool_name)
            self.assertTrue(arguments["ready_for_samples"])
            return {
                "result": {
                    "samples": [
                        {"sample_position": 1, "lambda_max_nm": 430.0, "max_absorbance": 0.021523},
                        {"sample_position": 2, "lambda_max_nm": 400.0, "max_absorbance": -0.005743},
                        {"sample_position": 3, "lambda_max_nm": 440.0, "max_absorbance": 0.024618},
                        {"sample_position": 4, "lambda_max_nm": 400.0, "max_absorbance": -0.01182},
                        {"sample_position": 5, "lambda_max_nm": 410.0, "max_absorbance": 1.318574},
                    ]
                }
            }

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(dict(fields))
            return True, "ok"

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda *_args, **_kwargs: None):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "都放好了",
                            "都放好了",
                        )

        self.assertTrue(handled)
        self.assertEqual(1, len(completed))
        self.assertNotIn("sample_2_absorbance_max", completed[0])
        self.assertNotIn("sample_4_absorbance_max", completed[0])
        self.assertEqual(0.021523, completed[0]["sample_1_absorbance_max"])
        self.assertEqual(0.024618, completed[0]["sample_3_absorbance_max"])
        self.assertEqual(1.318574, completed[0]["sample_5_absorbance_max"])

    async def test_handle_direct_uvvis_spectra_measurement_redirects_when_blank_missing(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID
        spoken = []
        graph_calls = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            self.assertEqual("uvvis_measure_spectra", tool_name)
            self.assertTrue(arguments["ready_for_samples"])
            return {"message": "pure water blank missing"}

        async def fake_graph_tool(_conn, tool_name, arguments, priority="foreground"):
            graph_calls.append((tool_name, dict(arguments), priority))
            return {"result": {"ok": True}}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_call_experiment_graph_tool_fast",
                    fake_graph_tool,
                ):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "都放好了",
                            "都放好了",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "redirect_to_step",
                    {
                        "session_id": "exp-1",
                        "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                    },
                    "foreground",
                )
            ],
            graph_calls,
        )
        self.assertEqual(
            ["这一步缺少纯水空白，我先退回前置校正。请先把样品位和参比位都清空，再告诉我开始。"],
            spoken,
        )

    def test_extract_uvvis_measure_spectra_rows_uses_payload_absorbance_csv_paths(self):
        conn = _FakeConn()

        with TemporaryDirectory() as temp_dir:
            csv_paths = []
            for sample_position in range(1, 6):
                csv_path = Path(temp_dir) / f"sample{sample_position}_latest_absorbance.csv"
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["wavelength_nm", "absorbance"],
                    )
                    writer.writeheader()
                    writer.writerow({"wavelength_nm": 400, "absorbance": 0.1})
                    writer.writerow(
                        {
                            "wavelength_nm": 400 + sample_position * 10,
                            "absorbance": 0.2 + sample_position * 0.1,
                        }
                    )
                    writer.writerow({"wavelength_nm": 700, "absorbance": 0.05})
                csv_paths.append(str(csv_path))

            rows = intentHandler._extract_uvvis_measure_spectra_rows(csv_paths, conn)

        self.assertEqual(5, len(rows))
        self.assertEqual(410.0, rows[1]["lambda_max_nm"])
        self.assertEqual(450.0, rows[5]["lambda_max_nm"])
        self.assertAlmostEqual(0.7, rows[5]["max_absorbance"])

    async def test_handle_direct_uvvis_kinetics_start_requests_liquid_blank(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "liquid blank missing for sample2"}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "开始动力学测量",
                        "开始动力学测量",
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_kinetics",
                    {
                        "session_key": "lease-1",
                        "wavelength_nm": 400,
                        "duration_minutes": 34,
                        "interval_seconds": 60,
                        "run_name": "sample2",
                        "ready_for_samples": False,
                        "sample_positions": [1],
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID,
                "run_name": "sample2",
                "sample_position": 1,
                "phase": "await_liquid_blank",
                "session_key": "lease-1",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(
            [
                "先保持样品位为空，我先做暗电流和 400 纳米空气基线准备。",
                "这一步指定的参比液/化学空白液还没放好，请把样品位和参比位同时放入该步骤指定的空白液，不是纯水。放好了告诉我。",
            ],
            spoken,
        )

    async def test_handle_direct_uvvis_kinetics_blank_uses_latest_spoken_sample_position(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_KINETICS_SAMPLE4_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_KINETICS_SAMPLE4_STEP_ID,
            "run_name": "sample4",
            "sample_position": 1,
            "phase": "await_liquid_blank",
        }
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-4", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"message": "liquid blank recorded"}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "2号位放好了",
                        "2号位放好了",
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_kinetics",
                    {
                        "session_key": "lease-4",
                        "wavelength_nm": 400,
                        "duration_minutes": 34,
                        "interval_seconds": 60,
                        "run_name": "sample4",
                        "ready_for_samples": True,
                        "sample_positions": [2],
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_KINETICS_SAMPLE4_STEP_ID,
                "run_name": "sample4",
                "sample_position": 2,
                "phase": "await_reaction_sample",
                "session_key": "lease-4",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(
            ["液体空白已经记录好了。请把参比位保持不变，把2号样品位换成真实反应液，放好了告诉我。"],
            spoken,
        )

    async def test_handle_direct_uvvis_kinetics_measurement_records_fields_and_waits_for_next_turn(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID,
            "run_name": "sample2",
            "sample_position": 3,
            "phase": "await_reaction_sample",
            "session_key": "lease-1",
        }
        spoken = []
        completed = []
        record_fields = {
            f"t{index}_absorbance": round(1.2 - index * 0.01, 4)
            for index in range(35)
        }

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            self.assertEqual("uvvis_measure_kinetics", tool_name)
            self.assertEqual([3], arguments["sample_positions"])
            self.assertTrue(arguments["ready_for_samples"])
            return {
                "record_fields": dict(record_fields),
                "bubble_observed": True,
            }

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, fallback_reply

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "可以开始了",
                            "可以开始了",
                        )

        self.assertTrue(handled)
        self.assertEqual(1, len(completed))
        self.assertFalse(completed[0]["auto_advance"])
        self.assertEqual("我记录好了，可以继续进行下一步了吗？", completed[0]["fallback_reply"])
        for index in range(35):
            self.assertEqual(
                record_fields[f"t{index}_absorbance"],
                completed[0]["fields"][f"t{index}_absorbance"],
            )
        self.assertTrue(completed[0]["fields"]["bubble_observed"])
        self.assertEqual(
            "3号样品400纳米动力学测量完成，共记录35个时间点。",
            completed[0]["fields"]["observations"],
        )
        self.assertEqual("done", getattr(conn, "_uvvis_direct_state", {}).get("phase"))
        self.assertEqual(["我记录好了，可以继续进行下一步了吗？"], spoken)

    async def test_handle_direct_uvvis_kinetics_done_advances_and_releases(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID,
            "run_name": "sample2",
            "sample_position": 1,
            "phase": "done",
            "session_key": "lease-1",
        }
        spoken = []
        released = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_advance(_conn, *, fallback_reply=""):
            conn.experiment_current_step_id = intentHandler._UVVIS_ANALYSIS_STEP_ID
            return True, "接下来做这一步：数据分析。"

        async def fake_release(_conn):
            released.append(True)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_advance_finished_experiment_step", fake_advance):
                with patch.object(intentHandler, "_release_uvvis_session_for_analysis", fake_release):
                    with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "继续下一步",
                            "继续下一步",
                        )

        self.assertTrue(handled)
        self.assertEqual([True], released)
        self.assertEqual(["接下来做这一步：数据分析。"], spoken)
        self.assertEqual({}, getattr(conn, "_uvvis_direct_state", {}))

    async def test_handle_direct_uvvis_analysis_step_releases_session(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_ANALYSIS_STEP_ID
        conn._uvvis_session_key = "lease-9"
        executed = []

        class _FakeManager:
            @staticmethod
            def is_mcp_tool(name):
                return name == "uvvis_session"

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"session_key": ""}

        with patch.object(intentHandler, "_get_server_mcp_manager", return_value=_FakeManager()):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                handled = await intentHandler.handle_direct_uvvis_intent(
                    conn,
                    "开始分析",
                    "开始分析",
                )

        self.assertFalse(handled)
        self.assertEqual(
            [("uvvis_session", {"action": "release", "session_key": "lease-9"})],
            executed,
        )
        self.assertEqual("", getattr(conn, "_uvvis_session_key", ""))

    def test_backstage_filter_drops_new_transition_phrases(self):
        self.assertEqual(
            "",
            textUtils.filter_spoken_backstage_text(
                "我先核对一下五号样品后面的紧接步骤，避免把你带错。"
            ),
        )
        self.assertEqual(
            "",
            textUtils.filter_spoken_backstage_text(
                "我再看一眼这一步要你回报什么。"
            ),
        )
        self.assertEqual(
            "",
            textUtils.filter_spoken_backstage_text(
                "我接着确认一号样品这一小步的记录项，只记你刚才报的颜色和时间。"
            ),
        )


    async def test_confirmation_statement_reports_missing_confirmation_field_instead_of_advancing(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"reported": False, "advanced": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                if not state["advanced"]:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                                "interaction": {
                                    "fast_path_mode": "confirmation_step",
                                    "capabilities": ["procedural_guidance", "step_confirmation"],
                                },
                                "prompts": {
                                    "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5号样品：统一加入AgNO3",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": (
                                [] if state["reported"] else ["sodium_citrate_added_to_all"]
                            )
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "sodium_citrate_added_to_all",
                                "type": "bool",
                                "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                            }
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["reported"] = True
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "can_proceed":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "proceed_to_next_step":
                state["advanced"] = state["reported"]
                conn.experiment_current_step_id = "step_add_agno3_all"
                return {"result": {"ok": state["advanced"]}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_agno3_all",
                                "title": "1-5号样品：统一加入AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "按一到五号顺序完成全部柠檬酸钠加入",
                    "按一到五号顺序完成全部柠檬酸钠加入",
                )

        self.assertTrue(handled)
        self.assertEqual(["按一到五号顺序完成全部柠檬酸钠加入"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("AgNO3", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_advance_strict_graph_path_records_and_moves_to_next_step_when_fast_path_disabled(self):
        conn = _FakeConn()
        conn.config = {"experiment_fast_path_enabled": False}
        spoken = []
        sent = []
        tool_calls = []
        state = {"reported": False, "advanced": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                if not state["advanced"]:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁转子",
                                "prompts": {
                                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5号样品：统一加入柠檬酸钠",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {"result": {"ok": True, "progress": None}}
            if tool_name == "start_trial":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": (
                                []
                                if state["reported"]
                                else ["beakers_labeled", "stir_bars_added_to_all"]
                            )
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "beakers_labeled",
                                "type": "bool",
                                "description": "已完成 1-5 号烧杯编号",
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "已为 1-5 号烧杯全部放入磁转子",
                            },
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["reported"] = True
                self.assertEqual(
                    {"beakers_labeled": True, "stir_bars_added_to_all": True},
                    arguments.get("data"),
                )
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "can_proceed":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "proceed_to_next_step":
                state["advanced"] = state["reported"]
                conn.experiment_current_step_id = "step_add_sodium_citrate_all"
                return {"result": {"ok": state["advanced"]}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_strict_graph_intent(
                    conn,
                    "全部完成",
                    "全部完成",
                )

        self.assertTrue(handled)
        self.assertEqual(["全部完成"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("柠檬酸钠", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_advance_fast_path_records_and_moves_to_next_step(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"reported": False, "advanced": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                if not state["advanced"]:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁转子",
                                "prompts": {
                                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5号样品：统一加入柠檬酸钠",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {"result": {"ok": True, "progress": None}}
            if tool_name == "start_trial":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": (
                                []
                                if state["reported"]
                                else ["beakers_labeled", "stir_bars_added_to_all"]
                            )
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "beakers_labeled",
                                "type": "bool",
                                "description": "已完成 1-5 号烧杯编号",
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "已为 1-5 号烧杯全部放入磁转子",
                            },
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["reported"] = True
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "can_proceed":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "proceed_to_next_step":
                state["advanced"] = state["reported"]
                conn.experiment_current_step_id = "step_add_sodium_citrate_all"
                return {"result": {"ok": state["advanced"]}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "当前步骤已完成",
                    "当前步骤已完成",
                )

        self.assertTrue(handled)
        self.assertEqual(["当前步骤已完成"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("柠檬酸钠", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])

    async def test_confirmation_statement_without_interaction_metadata_reports_missing_field(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"reported": False, "advanced": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                if not state["advanced"]:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                                "prompts": {
                                    "instruction": "按 1 到 5 号顺序统一完成柠檬酸钠加入。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5号样品：统一加入AgNO3",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": (
                                [] if state["reported"] else ["sodium_citrate_added_to_all"]
                            )
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "sodium_citrate_added_to_all",
                                "type": "bool",
                                "description": "已按 1-5 号顺序完成全部柠檬酸钠加入",
                            }
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["reported"] = True
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "can_proceed":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "proceed_to_next_step":
                state["advanced"] = state["reported"]
                conn.experiment_current_step_id = "step_add_agno3_all"
                return {"result": {"ok": state["advanced"]}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_agno3_all",
                                "title": "1-5号样品：统一加入AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 5.00 mL AgNO3。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "按一到五号顺序全部加入柠檬酸钠",
                    "按一到五号顺序全部加入柠檬酸钠",
                )

        self.assertTrue(handled)
        self.assertEqual(["按一到五号顺序全部加入柠檬酸钠"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("AgNO3", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])

    async def test_observation_step_reports_missing_fields_without_autofill(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"bool_fields_written": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_sample1_2_add_kbr_water_nabh4",
                            "title": "1号样品：加入KBr、纯水并加入NaBH4",
                            "interaction": {
                                "fast_path_mode": "observation_record_step",
                                "capabilities": ["step_confirmation", "observation_capture"],
                            },
                            "prompts": {
                                "instruction": "完成 1 号样品 KBr 和纯水加入并混匀后，快速加入 NaBH4 并保持搅拌，记录颜色稳定时间和收尾情况。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {"result": {"ok": True, "progress": None}}
            if tool_name == "start_trial":
                missing = ["color", "reaction_time", "color_stable"]
                if not state["bool_fields_written"]:
                    missing = [
                        "KBr_volume",
                        "H2O_volume",
                        "mixed_uniformly",
                        "nabh4_volume",
                        "added_quickly",
                    ] + missing
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": missing},
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {"name": "KBr_volume", "type": "bool", "description": "已按当前样品目标用量加入 KBr"},
                            {"name": "H2O_volume", "type": "bool", "description": "已按当前样品目标用量加入纯水"},
                            {"name": "mixed_uniformly", "type": "bool", "description": "加入 KBr 和纯水后已搅拌均匀"},
                            {"name": "nabh4_volume", "type": "bool", "description": "已准确加入 2.50 mL NaBH4"},
                            {"name": "added_quickly", "type": "bool", "description": "已快速完成 NaBH4 加入"},
                            {"name": "color", "type": "string", "description": "当前样品最终颜色"},
                            {"name": "reaction_time", "type": "float", "description": "当前样品颜色稳定所用时间"},
                            {"name": "color_stable", "type": "bool", "description": "已确认颜色稳定"},
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["bool_fields_written"] = True
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": ["color", "reaction_time", "color_stable"]
                        },
                    }
                }
            if tool_name == "finish_trial":
                raise AssertionError("observation step should still block finish_trial until data fields are reported")
            if tool_name == "proceed_to_next_step":
                raise AssertionError("observation step should not advance while data fields are missing")
            if tool_name == "can_proceed":
                raise AssertionError("observation step should not call can_proceed while data fields are missing")
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "已经做好了",
                    "已经做好了",
                )

        self.assertTrue(handled)
        self.assertEqual(["已经做好了"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("颜色", spoken[0])
        self.assertIn("时间", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])

    async def test_global_completion_phrase_all_done_advances_after_writeback(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"reported": False, "advanced": False}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                if not state["advanced"]:
                    return {
                        "result": {
                            "ok": True,
                            "step": {
                                "id": "step_prepare_setup_all",
                                "title": "1-5号样品：准备烧杯与磁转子",
                                "prompts": {
                                    "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5号样品：统一加入柠檬酸钠",
                            "prompts": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {
                            "missing_fields": (
                                []
                                if state["reported"]
                                else ["beakers_labeled", "stir_bars_added_to_all"]
                            )
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "beakers_labeled",
                                "type": "bool",
                                "description": "已完成 1-5 号烧杯编号",
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "已为 1-5 号烧杯全部放入磁转子",
                            },
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["reported"] = True
                self.assertEqual(
                    {"beakers_labeled": True, "stir_bars_added_to_all": True},
                    arguments.get("data"),
                )
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": []},
                    }
                }
            if tool_name == "finish_trial":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "can_proceed":
                return {"result": {"ok": state["reported"]}}
            if tool_name == "proceed_to_next_step":
                state["advanced"] = state["reported"]
                conn.experiment_current_step_id = "step_add_sodium_citrate_all"
                return {"result": {"ok": state["advanced"]}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_add_sodium_citrate_all",
                                "title": "1-5号样品：统一加入柠檬酸钠",
                            },
                            "current_step_details": {
                                "instruction": "按 1 到 5 号顺序加入 1.00 mL 柠檬酸钠。",
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "全部完成",
                    "全部完成",
                )

        self.assertTrue(handled)
        self.assertEqual(["全部完成"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("柠檬酸钠", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_specific_confirmation_phrase_marks_only_matching_bool_field(self):
        conn = _FakeConn()
        spoken = []
        sent = []
        tool_calls = []
        state = {"written_fields": set()}

        async def fake_send_stt_message(_conn, text):
            sent.append(text)

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        async def fake_call(tool_name, arguments, priority="foreground"):
            tool_calls.append((tool_name, dict(arguments), priority))
            if tool_name == "get_step":
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_prepare_setup_all",
                            "title": "1-5号样品：准备烧杯与磁转子",
                            "prompts": {
                                "instruction": "完成 1-5 号样品的烧杯编号和磁转子放置。",
                            },
                        },
                    }
                }
            if tool_name == "get_current_progress":
                missing = []
                if "stir_bars_added_to_all" not in state["written_fields"]:
                    missing.append("stir_bars_added_to_all")
                if "beakers_labeled" not in state["written_fields"]:
                    missing.append("beakers_labeled")
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": missing},
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "beakers_labeled",
                                "type": "bool",
                                "description": "已完成 1-5 号烧杯编号",
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "已为 1-5 号烧杯全部放入磁转子",
                            },
                        ],
                    }
                }
            if tool_name == "add_fields":
                state["written_fields"].update(arguments.get("data", {}).keys())
                self.assertEqual(
                    {"stir_bars_added_to_all": True},
                    arguments.get("data"),
                )
                return {
                    "result": {
                        "ok": True,
                        "current_progress": {"missing_fields": ["beakers_labeled"]},
                    }
                }
            if tool_name == "finish_trial":
                raise AssertionError("current step should stay blocked until all bool fields are reported")
            if tool_name == "can_proceed":
                raise AssertionError("current step should not call can_proceed while bool fields are missing")
            if tool_name == "proceed_to_next_step":
                raise AssertionError("current step should not advance while bool fields are missing")
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "一到五号双杯全部放入磁子",
                    "一到五号双杯全部放入磁子",
                )

        self.assertTrue(handled)
        self.assertEqual(["一到五号双杯全部放入磁子"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("烧杯编号", spoken[0])
        self.assertNotIn("磁转子", spoken[0])
        self.assertEqual({"stir_bars_added_to_all"}, state["written_fields"])

if __name__ == "__main__":
    unittest.main()
