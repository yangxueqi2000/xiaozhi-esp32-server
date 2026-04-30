import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


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
        self.experiment_session_id = "exp-1"
        self.experiment_current_step_id = "step_prepare_setup_all"
        self.experiment_current_step = None
        self.experiment_progress_summary = None
        self.intent_type = "function_call"
        self.config = {}
        self.tts = None
        self.enriched = False
        self.cmd_exit = []

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
        self.assertEqual(
            "你准备好后告诉我准备好了，我再带你开始第一步。",
            third,
        )

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

        self.assertEqual(
            {
                "beakers_labeled": True,
                "stir_bars_added_to_all": True,
            },
            result,
        )

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

        self.assertEqual({"shared_round_confirmed": True}, result)

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

        self.assertEqual({"sodium_citrate_added_to_all": True}, result)

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

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_fast_intent(
                    conn,
                    "开始今天的实验",
                    "开始今天的实验",
                )

        self.assertTrue(handled)
        self.assertEqual(["开始今天的实验"], sent)
        self.assertEqual(1, len(spoken))
        self.assertEqual(
            "今天我们做《Ag 纳米粒子的制备及其催化还原 4-硝基苯酚的反应动力学探究》。你准备好开始了吗？",
            spoken[0],
        )

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

    async def test_continue_before_ready_does_not_broadcast_step(self):
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
        self.assertEqual(
            ["你准备好后告诉我准备好了，我再带你开始第一步。"],
            spoken,
        )

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
                if state["can_proceed_calls"] == 1:
                    return {"result": {"ok": False, "message": "尚未完成"}}
                return {"result": {"ok": True, "message": None}}
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
                self.assertEqual(
                    {
                        "beakers_labeled": True,
                        "stir_bars_added_to_all": True,
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
            if tool_name == "proceed_to_next_step":
                return {
                    "result": {
                        "ok": True,
                        "current_step_id": "step_add_sodium_citrate_all",
                    }
                }
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
        self.assertIn("接下来做这一步", spoken[0])
        self.assertIn("柠檬酸钠", spoken[0])
        self.assertTrue(conn.enriched)
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_confirmation_statement_advances_confirmation_step_locally(self):
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
                if state["can_proceed_calls"] == 1:
                    return {"result": {"ok": False, "message": "尚未完成"}}
                return {"result": {"ok": True}}
            if tool_name == "add_fields":
                self.assertEqual(
                    {"sodium_citrate_added_to_all": True},
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
            if tool_name == "proceed_to_next_step":
                return {"result": {"ok": True}}
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
        self.assertIn("接下来做这一步", spoken[0])
        self.assertIn("AgNO3", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )

    async def test_confirmation_statement_advances_without_interaction_metadata(self):
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
                if state["can_proceed_calls"] == 1:
                    return {"result": {"ok": False, "message": "尚未完成"}}
                return {"result": {"ok": True}}
            if tool_name == "add_fields":
                self.assertEqual(
                    {"sodium_citrate_added_to_all": True},
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
            if tool_name == "proceed_to_next_step":
                return {"result": {"ok": True}}
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
        self.assertIn("接下来做这一步", spoken[0])
        self.assertIn("AgNO3", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])

    async def test_local_photo_followup_writes_back_and_moves_to_next_step(self):
        conn = _FakeConn()
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

        self.assertIn("接下来做这一步", reply)
        self.assertIn("丁达尔现象观察", reply)
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn(
            "proceed_to_next_step",
            [name for name, _args, _priority in tool_calls],
        )
        self.assertEqual("step_tyndall_observation", conn.experiment_current_step_id)

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

        async def fake_advance(_conn, payload, fallback_reply=""):
            photo_meta = intentHandler._extract_photo_result_meta(payload)
            self.assertTrue(photo_meta["found"])
            self.assertIn("一号样品", photo_meta["file_name"])
            self.assertEqual("一号样品", photo_meta["requested_photo_name"])
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


if __name__ == "__main__":
    unittest.main()
