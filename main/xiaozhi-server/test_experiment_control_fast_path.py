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
            "浠婂ぉ鎴戜滑鍋氥€婇摱绾崇背绮掑瓙瀹為獙銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
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
                "鍙互鎷嶇収銆?,
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
            self.assertIn("鍙互鎷嶇収銆?, content)

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
                        "[yaml=C:\\demo\\experiments.yaml] 鍏ㄩ儴瀹屾垚銆?,
                        "[2026-05-05T14:34:57.063+08:00] [TRANSCRIPT] [ASSISTANT] [source=speak_txt] "
                        "[experiment_session_id=exp-1] [current_step_id=step_add_sodium_citrate_all] "
                        "[yaml=C:\\demo\\experiments.yaml] 鐜板湪鍋氳繖涓€姝ャ€?,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            entries = experiment_resume.read_transcript_entries(log_path)

            self.assertEqual(2, len(entries))
            self.assertEqual("USER", entries[0]["role"])
            self.assertEqual("step_prepare_setup_all", entries[0]["current_step_id"])
            self.assertEqual("鍏ㄩ儴瀹屾垚銆?, entries[0]["text"])
            self.assertEqual("ASSISTANT", entries[1]["role"])
            self.assertEqual(
                "step_add_sodium_citrate_all",
                entries[1]["current_step_id"],
            )

    def test_runtime_spoken_text_strips_technical_details(self):
        text = (
            "瀹為獙鎶ュ憡宸茬粡鐢熸垚锛宲df_path=C:\\demo\\report.pdf锛?
            "session_id=abc123銆?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual("瀹為獙鎶ュ憡宸茬粡鐢熸垚銆?, result)

    def test_runtime_spoken_text_strips_uvvis_tool_signature_details(self):
        text = (
            "鐜板湪寮€濮?UV-Vis 鍓嶇疆鏍℃銆?
            "鐒跺悗浣跨敤褰撳墠宸叉寔鏈夌殑 session_key 璋冪敤 "
            "`uvvis_measure_spectra(sample_positions=[1,2,3,4,5], ready_for_samples=false)`銆?
            "鍋氬ソ鍚庡憡璇夋垜銆?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertIn("鐜板湪寮€濮?UV-Vis 鍓嶇疆鏍℃", result)
        self.assertNotIn("session_key", result)
        self.assertNotIn("sample_positions", result)
        self.assertNotIn("ready_for_samples", result)
        self.assertNotIn("uvvis_measure_spectra", result)

    def test_runtime_spoken_text_strips_uvvis_internal_confirmation_rules(self):
        text = (
            "璇峰湪 1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁鍏ョ函姘存瘮鑹茬毧锛屽叡 6 涓紝"
            "鍙湁鍦ㄤ富璇磋瘽浜烘槑纭洖鎶モ€滄斁濂戒簡鈥濃€滈兘鏀惧ソ浜嗏€濃€滃凡缁忔斁濂解€濃€滃彲浠ュ紑濮嬩簡鈥濇垨鍚屼箟琛ㄨ揪鍚庯紝"
            "鎵嶈皟鐢ㄨ褰曠函姘寸┖鐧姐€?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertIn("鏀惧叆绾按姣旇壊鐨?, result)
        self.assertNotIn("涓昏璇濅汉", result)
        self.assertNotIn("鍚屼箟琛ㄨ揪", result)
        self.assertNotIn("璁板綍绾按绌虹櫧", result)

    def test_runtime_spoken_text_limits_to_two_sentences(self):
        text = "鐜板湪鍋氳繖涓€姝ャ€傛敞鎰忎笉瑕佹薄鏌撱€傚仛濂藉悗鍛婅瘔鎴戙€?

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual("鐜板湪鍋氳繖涓€姝ャ€傛敞鎰忎笉瑕佹薄鏌擄紝鍋氬ソ鍚庡憡璇夋垜銆?, result)

    def test_runtime_spoken_text_appends_completion_prompt_to_step_guidance(self):
        text = "鐜板湪鍋氳繖涓€姝ワ細鎸夐『搴忓姞鍏?AgNO3 骞惰交杞绘贩鍖€銆傛敞鎰忎笉瑕侀婧呫€?

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "鐜板湪鍋氳繖涓€姝ワ細鎸夐『搴忓姞鍏?AgNO3 骞惰交杞绘贩鍖€銆傛敞鎰忎笉瑕侀婧咃紝鍋氬ソ鍚庡憡璇夋垜銆?,
            result,
        )

    def test_conn_runtime_spoken_text_blocks_false_export_success(self):
        conn = _FakeConn()
        conn._pending_export_report_validation = {
            "active": True,
            "all_expected_outputs_exist": False,
        }

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "瀹為獙鎶ュ憡宸茬粡鐢熸垚瀹屾垚銆?,
        )

        self.assertEqual("瀹為獙鎶ュ憡杩樻病鏈夊畬鏁寸敓鎴愭垚鍔燂紝璇风◢鍚庡啀璇曘€?, result)

    def test_conn_runtime_spoken_text_rewrites_speculative_uvvis_occupation(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-uvvis-1"

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "杩欓噷鏆傛椂杩樻病鑳界洿鎺ュ惎鍔ㄦ牎姝ｏ紝浣犲厛妫€鏌ヤ竴涓嬪厜璋变华鏈夋病鏈夎鍒殑绋嬪簭鍗犵敤锛岀‘璁ゅ悗鍛婅瘔鎴戠户缁€?,
        )

        self.assertEqual("UV-Vis 杩欒竟杩樻病鍑嗗濂斤紝璇风◢鍚庡啀璇曘€?, result)

    def test_conn_runtime_spoken_text_strips_speculative_interface_outage(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-1"

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "瀹為獙鍥捐氨鎺ュ彛杩欒疆娌℃帴閫氥€?,
        )

        self.assertEqual("", result)

    def test_conn_runtime_spoken_text_reanchors_speculative_future_step_to_current_graph_step(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-align-1"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "鍏堝畬鎴?1-5 鍙风儳鏉紪鍙峰拰纾佽浆瀛愭斁缃€?,
                    },
                }
            }
        }
        conn.dialogue.put(Message(role="user", content="缁х画涓嬩竴姝ャ€?))

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "鐜板湪鍋氫竵杈惧皵鐜拌薄瑙傚療锛氭妸鐜璋冩殫锛岀敤婵€鍏夌瑪浠庝晶闈㈢収灏勬牱鍝併€傜湅瀹屽悗鍛婅瘔鎴戙€?,
        )

        self.assertIn("鍑嗗鐑ф澂涓庣杞瓙", result)
        self.assertIn("鐑ф澂缂栧彿", result)
        self.assertNotIn("涓佽揪灏?, result)

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
                    "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆H2O2",
                    "prompts": {
                        "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 H2O2 鍔犲叆銆?,
                    },
                }
            }
        }
        conn.dialogue.put(Message(role="user", content="缁х画涓嬩竴姝ャ€?))

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "鐜板湪鍋氫竵杈惧皵鐜拌薄瑙傚療锛氭妸鐜璋冩殫锛岀敤婵€鍏夌瑪浠庝晶闈㈢収灏勬牱鍝併€傜湅瀹屽悗鍛婅瘔鎴戙€?,
        )

        self.assertIn("缁熶竴鍔犲叆H2O2", result)
        self.assertIn("H2O2", result)
        self.assertNotIn("涓佽揪灏?, result)

    def test_conn_runtime_spoken_text_uses_yaml_step_cache_when_graph_snapshot_missing(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-align-3"
        conn.experiment_current_step_id = "step_sample1_2_add_kbr_water_nabh4"
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                "prompts": {
                    "instruction": (
                        "鍏堝姞鍏ュ苟娣峰寑 KBr 涓庣函姘达紝鍐嶅揩閫熷姞鍏?NaBH4銆?
                        "浠庡姞鍏?NaBH4 鐨勭灛闂村紑濮嬭鏃讹紝鎸佺画鎼呮媽骞惰瀵熼鑹插彉鍖栵紝寰呴鑹茬ǔ瀹氬悗鍐嶆眹鎶ョ粨鏋溿€?
                    )
                },
            }
        ]
        conn.dialogue.put(Message(role="user", content="鍏ㄩ儴纭閮藉仛濂戒簡銆?))

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "鐜板湪鍋氳繖涓€姝ワ細1鍙锋牱鍝侊細鍏堝湪鍔犲叆 NaBH4 鐨勫悓鏃跺紑濮嬭鏃讹紝鎸佺画鎼呮媽锛屾寔缁瀵熼鑹插彉鍖栥€?,
        )

        self.assertIn("KBr", result)
        self.assertIn("绾按", result)
        self.assertIn("NaBH4", result)
        self.assertNotIn("鍙墿鍚庡崐鍙?, result)

    def test_experiment_step_guidance_detector_accepts_observation_tail_prompt(self):
        guidance = (
            "浜屽彿鏍峰搧锛氬厛鍦ㄥ姞鍏?NaBH4 鐨勫悓鏃跺紑濮嬭鏃讹紝鎸佺画鎼呮媽锛屾寔缁瀵熼鑹插彉鍖栵紝"
            "绛夐鑹茬ǔ瀹氾紝鍐嶇洿鎺ュ憡璇夋垜鏈€缁堥鑹插拰浠庡姞鍏?NaBH4 鍒扮ǔ瀹氫竴鍏辩敤浜嗗嚑鍒嗛挓銆?
        )

        self.assertTrue(textUtils._looks_like_experiment_step_or_scan_guidance(guidance))

    def test_conn_runtime_spoken_text_rehydrates_sample2_amounts_for_tail_only_guidance(self):
        conn = _FakeConn()
        conn.sentence_id = "turn-graph-align-4"
        conn.experiment_session_id = "exp-1"
        conn.experiment_current_step_id = "step_sample2_2_add_kbr_water_nabh4"
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample2_2_add_kbr_water_nabh4",
                "title": "2鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                "prompts": {
                    "instruction": (
                        "瀹屾垚 2 鍙锋牱鍝?KBr 鍜岀函姘村姞鍏ワ紙KBr 0.80 mL锛岀函姘?2.10 mL锛夊苟娣峰寑鍚庯紝"
                        "蹇€熷姞鍏?NaBH4锛?.50 mL 0.005 mol/L锛夛紱鍦ㄥ姞鍏?NaBH4 鐨勫悓鏃跺紑濮嬭鏃跺苟淇濇寔鎼呮媽锛?
                        "鎸佺画瑙傚療棰滆壊鍙樺寲锛岀瓑棰滆壊绋冲畾鍚庣洿鎺ュ憡璇夋垜鏈€缁堥鑹插拰浠庡姞鍏?NaBH4 鍒扮ǔ瀹氫竴鍏辩敤浜嗗嚑鍒嗛挓锛?
                        "瀹屾垚鍚庤繘鍏ユ湰鏍峰搧鎷嶇収璁板綍姝ラ銆?
                    )
                },
            }
        ]
        conn.dialogue.put(Message(role="user", content="缁х画涓嬩竴姝?))

        result = textUtils.prepare_runtime_spoken_text_for_conn(
            conn,
            "浜屽彿鏍峰搧锛氬厛鍦ㄥ姞鍏?NaBH4 鐨勫悓鏃跺紑濮嬭鏃讹紝鎸佺画鎼呮媽锛屾寔缁瀵熼鑹插彉鍖栵紝绛夐鑹茬ǔ瀹氾紝鍐嶇洿鎺ュ憡璇夋垜鏈€缁堥鑹插拰浠庡姞鍏?NaBH4 鍒扮ǔ瀹氫竴鍏辩敤浜嗗嚑鍒嗛挓銆?,
        )

        self.assertIn("2鍙锋牱鍝?, result)
        self.assertIn("KBr 0.80 mL", result)
        self.assertIn("绾按 2.10 mL", result)
        self.assertIn("NaBH4锛?.50 mL 0.005 mol/L锛?, result)

    def test_cached_experiment_step_meta_prefers_yaml_instruction_and_photo_prompt(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_sample1_5_photo_confirm",
                    "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                    "prompts": {
                        "instruction": "鎷嶇収璁板綍褰撳墠鏍峰搧棰滆壊銆?,
                    },
                }
            }
        }
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_5_photo_confirm",
                "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                "prompts": {
                    "instruction": "棰滆壊绋冲畾鍚庢媿鐓ц褰曞綋鍓嶆牱鍝侀鑹诧紝骞惰繘鍏?2 鍙锋牱鍝併€?,
                },
            },
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                "prompts": {
                    "instruction": "鍏堝姞鍏ュ苟娣峰寑 KBr 涓庣函姘达紝鍐嶅揩閫熷姞鍏?NaBH4銆?,
                },
            },
        ]

        meta = intentHandler._get_cached_experiment_step_meta(conn)
        reply = intentHandler._compose_experiment_step_reply(meta, mode="next")

        self.assertIn("棰滆壊绋冲畾鍚庢媿鐓?, meta["instruction"])
        self.assertEqual("1鍙锋牱鍝侀鑹插凡缁忕ǔ瀹氾紝鐜板湪鍙互鎷嶇収鍚楋紵", reply)

    def test_cached_experiment_step_meta_prefers_yaml_instruction_for_sample_step(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_sample1_2_add_kbr_water_nabh4",
                    "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                    "prompts": {
                        "instruction": "鍏堝湪鍔犲叆 NaBH4 鐨勫悓鏃跺紑濮嬭鏃讹紝鎸佺画鎼呮媽锛屾寔缁瀵熼鑹插彉鍖栥€?,
                    },
                }
            }
        }
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                "prompts": {
                    "instruction": "鍏堝姞鍏ュ苟娣峰寑 KBr 涓庣函姘达紝鍐嶅揩閫熷姞鍏?NaBH4锛屼粠鍔犲叆 NaBH4 鐨勭灛闂村紑濮嬭鏃跺苟鎸佺画鎼呮媽銆?,
                },
            }
        ]

        meta = intentHandler._get_cached_experiment_step_meta(conn)
        reply = intentHandler._compose_experiment_step_reply(meta, mode="guide")

        self.assertIn("KBr", meta["instruction"])
        self.assertIn("绾按", meta["instruction"])
        self.assertIn("KBr", reply)
        self.assertIn("绾按", reply)

    def test_cached_experiment_step_meta_uses_uvvis_spoken_override_for_shared_blank_step(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_3_uv_vis_shared_dark_blank_prep",
                    "title": "1-5鍙锋牱鍝侊細鏆楃數娴佸拰绾按绌虹櫧鏍℃",
                    "prompts": {
                        "instruction": (
                            "鐜板湪寮€濮?UV-Vis 鍓嶇疆鏍℃銆?
                            "璋冪敤 uvvis_measure_spectra(sample_positions=[1,2,3,4,5], ready_for_samples=false)銆?
                        ),
                    },
                }
            }
        }
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_3_uv_vis_shared_dark_blank_prep",
                "title": "1-5鍙锋牱鍝侊細鏆楃數娴佸拰绾按绌虹櫧鏍℃",
                "prompts": {
                    "instruction": (
                        "鐜板湪寮€濮?UV-Vis 鍓嶇疆鏍℃銆?
                        "璋冪敤 uvvis_measure_spectra(sample_positions=[1,2,3,4,5], ready_for_samples=false)銆?
                    ),
                },
            }
        ]

        meta = intentHandler._get_cached_experiment_step_meta(conn)
        reply = intentHandler._compose_experiment_step_reply(meta, mode="next")

        self.assertIn("绾按绌虹櫧鏍℃", reply)
        self.assertIn("鏀惧叆绾按姣旇壊鐨?, reply)
        self.assertIn("鍙互寮€濮嬫壂鎻?, reply)
        self.assertNotIn("鍏堜笉瑕佹斁浠讳綍娑蹭綋", reply)
        self.assertNotIn("session_key", reply)
        self.assertNotIn("ready_for_samples", reply)
        self.assertNotIn("sample_positions", reply)

    def test_shared_dark_air_step_uses_concise_student_facing_reply(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_3_uv_vis_shared_dark_air_prep",
                    "title": "1-5号样品：共享暗电流和空气能量校正",
                    "prompts": {
                        "instruction": (
                            "先检查 1-5 号样品位都为空，参比位也不要放任何液体。"
                            "确认后先告诉我都空了。"
                        ),
                    },
                }
            }
        }

        meta = intentHandler._get_cached_experiment_step_meta(conn)
        reply = intentHandler._compose_experiment_step_reply(meta, mode="guide")
        spoken = textUtils.prepare_runtime_spoken_text(reply)

        self.assertIn("都空了", spoken)
        self.assertIn("开始扫描", spoken)
        self.assertIn("参比位", spoken)
        self.assertNotIn("session_key", spoken)
        self.assertNotIn("uvvis_prepare_dark_current", spoken)
        self.assertNotIn("现在做这一步", spoken)

    def test_pure_water_blank_step_meta_rewrites_internal_rules_to_student_prompt(self):
        meta = {
            "step_id": "step_uvvis_pure_water_blank_only",
            "title": "1-5鍙锋牱鍝侊細绾按绌虹櫧鏍℃",
            "instruction": (
                "鏆楃數娴佹牎姝ｅ凡缁忓畬鎴愶紝鐜板湪杩涘叆绾按绌虹櫧鏍℃銆?
                "璇峰湪 1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁鍏ョ函姘存瘮鑹茬毧锛屽叡 6 涓紝"
                "鍙湁鍦ㄦ斁濂藉悗鎵嶅紑濮嬭褰曠函姘寸┖鐧姐€?
            ),
        }

        reply = intentHandler._compose_experiment_step_reply(meta, mode="guide")
        spoken = textUtils.prepare_runtime_spoken_text(reply)

        self.assertIn("绾按绌虹櫧鏍℃", spoken)
        self.assertIn("鏀惧叆绾按姣旇壊鐨?, spoken)
        self.assertIn("鍙互寮€濮嬫壂鎻?, spoken)
        self.assertNotIn("pure water", spoken)
        self.assertNotIn("liquid blank", spoken)
        self.assertNotIn("鍙鐢?, spoken)
        self.assertNotIn("璁板綍绾按绌虹櫧", spoken)

    def test_uvvis_step_overrides_keep_internal_control_rules_silent(self):
        cases = [
            {
                "meta": {
                    "step_id": "custom_uvvis_load_step",
                    "title": "1-5鍙锋牱鍝侊細瑁呭叆姣旇壊鐨?,
                    "instruction": (
                        "鍙渶鎻愰啋涓昏璇濅汉鎶?1 鍒?5 鍙风湡瀹炴牱鍝佸垎鍒鍏ユ瘮鑹茬毧锛?
                        "涓嶈灞曞紑鍚庣画鎵归噺娴嬮噺鐨勮鍒欍€?
                    ),
                },
                "expected": ("瑁呭叆姣旇壊鐨?, "鍙傛瘮浣嶄繚鐣欑函姘?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_record_step",
                    "title": "1-5鍙锋牱鍝侊細鎵归噺娴嬪厜璋卞苟璁板綍鏁版嵁",
                    "instruction": (
                        "鍙湁鍦ㄤ富璇磋瘽浜烘槑纭洖鎶モ€滄斁濂戒簡鈥濃€滈兘鏀惧ソ浜嗏€濃€滃凡缁忔斁濂解€濃€滃彲浠ュ紑濮嬩簡鈥?
                        "鎴栧悓涔夎〃杈惧悗锛屾墠璋冪敤 uvvis_measure_spectra銆?
                    ),
                },
                "expected": ("鎵归噺娴嬪厜璋卞苟璁板綍鏁版嵁", "寮€濮嬫祴閲?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_clean_step",
                    "title": "绱-鍙娴嬮噺鍚庯細缁熶竴娓呮礂姣旇壊鐨?,
                    "instruction": "鍙粰涓昏璇濅汉褰撳墠鍔ㄤ綔锛屼笉瑕佽鍚庣画鍔ㄥ姏瀛﹂厤娑层€?,
                },
                "expected": ("缁熶竴娓呮礂姣旇壊鐨?, "鎸夎鑼冨鐞嗘畫娑?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_sample2_reference",
                    "title": "2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫",
                    "instruction": "鍙渶鎻愰啋涓昏璇濅汉鎸夎姹傞厤濂?2 鍙锋牱鍝佸弬姣旀恫锛屼笉瑕佸睍寮€涓嬩竴姝ャ€?,
                },
                "expected": ("2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫", "鏀惧叆鍙傛瘮浣?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_sample2_reaction",
                    "title": "2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弽搴旀恫",
                    "instruction": "鍙彁绀轰富璇磋瘽浜哄綋鍓嶅姩浣滐紝涓嶈璁插悗缁祴閲忋€?,
                },
                "expected": ("2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弽搴旀恫", "鏀惧叆鏍峰搧浣?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_sample2_measurement",
                    "title": "2鍙锋牱鍝佸姩鍔涘锛氬紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴?,
                    "instruction": (
                        "鍙湁鍦ㄤ富璇磋瘽浜烘槑纭洖鎶モ€滃彲浠ュ紑濮嬩簡鈥濇垨鍚屼箟琛ㄨ揪鍚庯紝"
                        "鎵嶈皟鐢?uvvis_measure_kinetics銆?
                    ),
                },
                "expected": ("鍙互寮€濮嬫椂鍛婅瘔鎴?, "400绾崇背鍔ㄥ姏瀛︽祴閲?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_sample4_reference",
                    "title": "4鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫",
                    "instruction": "鍙渶鎻愰啋涓昏璇濅汉鎸夎姹傞厤濂?4 鍙锋牱鍝佸弬姣旀恫锛屼笉瑕佸睍寮€涓嬩竴姝ャ€?,
                },
                "expected": ("4鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫", "鏀惧叆鍙傛瘮浣?),
            },
            {
                "meta": {
                    "step_id": "custom_uvvis_sample4_measurement",
                    "title": "4鍙锋牱鍝佸姩鍔涘锛氬紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴?,
                    "instruction": (
                        "鍙湁鍦ㄤ富璇磋瘽浜烘槑纭洖鎶モ€滃彲浠ュ紑濮嬩簡鈥濇垨鍚屼箟琛ㄨ揪鍚庯紝"
                        "鎵嶈皟鐢?uvvis_measure_kinetics銆?
                    ),
                },
                "expected": ("鍙互寮€濮嬫椂鍛婅瘔鎴?, "400绾崇背鍔ㄥ姏瀛︽祴閲?),
            },
        ]

        for case in cases:
            with self.subTest(title=case["meta"]["title"]):
                reply = intentHandler._compose_experiment_step_reply(case["meta"], mode="guide")
                spoken = textUtils.prepare_runtime_spoken_text(reply)

                for expected in case["expected"]:
                    self.assertIn(expected, spoken)
                self.assertNotIn("涓昏璇濅汉", spoken)
                self.assertNotIn("鍚屼箟琛ㄨ揪", spoken)
                self.assertNotIn("session_key", spoken)
                self.assertNotIn("uvvis_measure", spoken)
                self.assertNotIn("涓嶈璁插悗缁?, spoken)
                self.assertNotIn("涓嶈灞曞紑", spoken)

    def test_conn_runtime_spoken_text_bypasses_ready_guard_for_same_sentence(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氳繖涓疄楠屻€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )
        conn.sentence_id = "turn-1"
        conn._experiment_ready_guard_bypass_sentence_id = "turn-1"

        guidance = "鐜板湪鍋氳繖涓€姝ワ細瀹屾垚 1 鍒?5 鍙风儳鏉紪鍙枫€傚仛濂藉悗鍛婅瘔鎴戙€?
        first = textUtils.prepare_runtime_spoken_text_for_conn(conn, guidance)
        second = textUtils.prepare_runtime_spoken_text_for_conn(conn, guidance)

        self.assertIn("鐜板湪鍋氳繖涓€姝?, first)
        self.assertIn("鐜板湪鍋氳繖涓€姝?, second)
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
        result = textUtils.normalize_tts_text("鍔犲叆1.849mL AgNO3銆?)

        self.assertIn("涓€鐐瑰叓鍥涗節姣崌", result)
        self.assertIn("纭濋吀閾?, result)

    def test_normalize_tts_text_reads_numeric_time_ranges_as_to(self):
        result = textUtils.normalize_tts_text("闈欑疆1-15鍒嗛挓鍚庤瀵熴€?)

        self.assertEqual("闈欑疆1鍒?5鍒嗛挓鍚庤瀵熴€?, result)

    def test_normalize_tts_text_reads_decimal_volume_ranges_as_to(self):
        result = textUtils.normalize_tts_text("鍔犲叆0.5-1.0mL AgNO3銆?)

        self.assertIn("闆剁偣浜斿埌涓€鐐归浂姣崌", result)
        self.assertIn("纭濋吀閾?, result)

    def test_normalize_tts_text_supports_generic_chemistry_pronunciation(self):
        result = textUtils.normalize_tts_text(
            "Ag锛堥摱锛夌撼绫崇矑瀛愮殑鍒跺鍙婂叾鍌寲杩樺師4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒"
        )

        self.assertEqual(
            "閾剁撼绫崇矑瀛愮殑鍒跺鍙婂叾鍌寲杩樺師瀵圭鍩鸿嫰閰氱殑鍙嶅簲鍔ㄥ姏瀛︽帰绌?,
            result,
        )

    def test_normalize_tts_text_reads_borohydride_with_peng_pronunciation(self):
        result = textUtils.normalize_tts_text("鍔犲叆纭兼阿鍖栭挔鍚庯紝鍐嶈ˉ鍔燦aBH4銆?)

        self.assertEqual("鍔犲叆褰阿鍖栭挔鍚庯紝鍐嶈ˉ鍔犲江姘㈠寲閽犮€?, result)

    def test_explicit_completion_report_treats_all_mixed_uniformly_as_done(self):
        self.assertTrue(intentHandler._looks_like_explicit_completion_report("宸茬粡鍏ㄩ儴娣峰寑"))

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
            "鐜板湪鍋?鍙锋牱鍝侊細鍦?鍙风儳鏉噷鍔犲叆婧村寲閽?.80姣崌锛屽啀鍔犲叆绾按2.10姣崌銆?
        )

        self.assertIn("浜屽彿鏍峰搧", result)
        self.assertIn("浜屽彿鐑ф澂", result)
        self.assertIn("闆剁偣鍏浂姣崌", result)
        self.assertIn("浜岀偣涓€闆舵鍗?, result)

    def test_normalize_tts_text_reads_numbered_label_ranges_with_chinese_digits(self):
        result = textUtils.normalize_tts_text("璇锋妸1-5鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁濂姐€?)

        self.assertEqual("璇锋妸涓€鍒颁簲鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁濂姐€?, result)

    def test_prepare_runtime_spoken_text_strips_meta_scope_clauses(self):
        text = (
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細1鍒?锛氬悓鏃跺惎鍔ㄦ悈鎷屽苟娣峰寑銆?
            "鍙畬鎴?鍒?鐨勬悈鎷岀粺涓€鍚姩鍜屾贩鍖€纭锛屼笉瑕侀噸澶嶅叡鍚岃瘯鍓傦紝涔熶笉瑕佽鍚庣画鍔犳恫锛?
            "娉ㄦ剰鍚姩鎼呮媽鍓嶇‘璁ゆ墍鏈夌儳鏉斁缃钩绋炽€?
            "娉ㄦ剰杞€熶笉瑕佽繃楂橈紝閬垮厤娑蹭綋椋炴簠锛屽仛濂藉悗鍛婅瘔鎴戙€?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細1鍒?锛氬悓鏃跺惎鍔ㄦ悈鎷屽苟娣峰寑銆?
            "娉ㄦ剰鍚姩鎼呮媽鍓嶇‘璁ゆ墍鏈夌儳鏉斁缃钩绋筹紝娉ㄦ剰杞€熶笉瑕佽繃楂橈紝閬垮厤娑蹭綋椋炴簠锛屽仛濂藉悗鍛婅瘔鎴戙€?,
            result,
        )

    def test_prepare_runtime_spoken_text_strips_future_step_transition_clauses(self):
        text = (
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細鎸?鍒?鍙烽『搴忕粺涓€瀹屾垚H2O2鍔犲叆銆?
            "瀹屾垚杩欎竴杞悗鍐嶅洖鍒?鍙锋牱鍝佸紑濮嬪悗缁楠ゃ€?
            "娉ㄦ剰鍔犳恫鍚庤交杞绘贩鍖€銆?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細鎸?鍒?鍙烽『搴忕粺涓€瀹屾垚H2O2鍔犲叆銆傛敞鎰忓姞娑插悗杞昏交娣峰寑锛屽仛濂藉悗鍛婅瘔鎴戙€?,
            result,
        )

    def test_prepare_runtime_spoken_text_compacts_long_measured_step(self):
        text = (
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細1鍙锋牱鍝侊細鍔犲叆婧村寲閽俱€佺函姘村苟鍔犲叆纭兼阿鍖栭挔銆?
            "瀹屾垚1鍙锋牱鍝佹捍鍖栭捑鍜岀函姘村姞鍏ワ紙婧村寲閽鹃浂鐐归浂闆舵鍗囷紝绾按浜岀偣涔濋浂姣崌锛夊苟娣峰寑鍚庯紝"
            "蹇€熷姞鍏ョ〖姘㈠寲閽狅紙浜岀偣浜旈浂姣崌 闆剁偣闆堕浂浜旀懇灏旀瘡鍗囷級骞朵繚鎸佹悈鎷岋紝"
            "璁板綍棰滆壊绋冲畾鏃堕棿鍜屾敹灏炬儏鍐点€?
            "娉ㄦ剰缁х画淇濇寔鎼呮媽锛岄伩鍏嶆恫浣撻婧咃紝纭兼阿鍖栭挔鏈夎厫铓€鎬э紝娉ㄦ剰闃叉姢骞堕伩鍏嶆簠鍑猴紝鍋氬ソ鍚庡憡璇夋垜銆?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細1鍙锋牱鍝侊細鍏堝姞鍏ユ捍鍖栭捑闆剁偣闆堕浂姣崌鍜岀函姘翠簩鐐逛節闆舵鍗囧苟娣峰寑锛屽啀蹇€熷姞鍏ョ〖姘㈠寲閽犱簩鐐逛簲闆舵鍗囷紝鎸佺画鎼呮媽銆?
            "閬垮厤娑蹭綋椋炴簠锛岀〖姘㈠寲閽犳湁鑵愯殌鎬э紝娉ㄦ剰闃叉姢骞堕伩鍏嶆簠鍑猴紝鍋氬ソ鍚庡憡璇夋垜銆?,
            result,
        )

    def test_prepare_runtime_spoken_text_keeps_observation_action_while_dropping_reporting_tail(self):
        text = (
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細2鍙锋牱鍝侊細瑙傚療棰滆壊鍙樺寲銆?
            "闈欑疆1鍒?鍒嗛挓鍚庤瀵熷苟鎷嶇収锛岃褰曢鑹插彉鍖栨椂闂村拰缁撴灉锛屽仛濂藉悗鍛婅瘔鎴戙€?
        )

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual(
            "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細2鍙锋牱鍝侊細鍏堥潤缃?鍒?鍒嗛挓锛屽啀瑙傚療骞舵媿鐓э紝鍋氬ソ鍚庡憡璇夋垜銆?,
            result,
        )

    def test_prepare_runtime_spoken_text_drops_recordkeeping_backstage_sentence(self):
        text = "鎴戝厛璁颁笅涓€鍙锋牱鍝佺殑鏈€缁堥鑹插拰绋冲畾鏃堕棿銆傜幇鍦ㄥ彲浠ユ媿鐓с€?

        result = textUtils.prepare_runtime_spoken_text(text)

        self.assertEqual("鐜板湪鍙互鎷嶇収銆?, result)

    def test_current_step_confirmation_fields_accept_all_added_completion_report(self):
        schema_by_name = {
            "h2o2_added_to_all": {
                "type": "bool",
                "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮?H2O2 鍔犲叆",
            }
        }

        result = intentHandler._build_experiment_current_step_confirmation_fields(
            "鍏ㄩ儴鍔犲ソ浜?,
            schema_by_name,
            ["h2o2_added_to_all"],
            allow_confirmation_autofill=True,
        )

        self.assertEqual({"h2o2_added_to_all": True}, result)

    def test_repeat_reply_is_direct_and_requests_completion(self):
        reply = intentHandler._compose_experiment_step_reply(
            {
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                "safety": "鍔犳恫鏃朵繚鎸佺Щ娑叉搷浣滅ǔ瀹氥€?,
            },
            mode="repeat",
        )

        self.assertNotIn("鎴戝啀绠€鐭涓€閬?, reply)
        self.assertIn("褰撳墠杩欎竴姝?, reply)
        self.assertIn("鍋氬ソ鍚庡憡璇夋垜", reply)

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
            Message(role="assistant", content="鐜板湪鍋氳繖涓€姝ャ€傚仛濂藉悗鍛婅瘔鎴戙€?)
        )

        action = intentHandler._classify_short_experiment_control(conn, "濂戒簡")
        self.assertEqual("advance", action)

    def test_continue_follows_start_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氳繖涓疄楠屻€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )

        action = intentHandler._classify_short_experiment_control(conn, "缁х画")
        self.assertEqual("guide", action)

    def test_ready_reply_wins_while_waiting_for_start(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浣犲噯澶囧ソ鍚庡憡璇夋垜鍑嗗濂戒簡锛屾垜鍐嶅甫浣犲紑濮嬬涓€姝ャ€?,
            )
        )

        action = intentHandler._classify_short_experiment_control(conn, "鍑嗗濂戒簡")
        self.assertEqual("guide", action)

    def test_step_guidance_clears_waiting_for_start_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="娴犲﹤銇夐幋鎴滄粦閸嬫俺绻栨稉顏勭杽妤犲被鈧倷缍橀崙鍡楊槵婵傝棄绱戞慨瀣╃啊閸氭绱?,
            )
        )
        conn.dialogue.put(
            Message(
                role="assistant",
                content="閻滄澘婀紒娆庣安閸欓攱鐗遍崫浣稿閸忋儳銆栧銏犲闁界姰鈧倸濮炵€瑰苯鎲＄拠澶嬪灉閵?",
            )
        )

        self.assertFalse(intentHandler._assistant_waiting_for_step_start(conn))

    def test_completion_variant_uses_waiting_context(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鐜板湪缁欎簲鍙锋牱鍝佸姞鍏ョ〖姘㈠寲閽犮€傚姞瀹屽憡璇夋垜銆?,
            )
        )

        action = intentHandler._classify_short_experiment_control(conn, "宸茬粡鍔犲ソ浜?)
        self.assertEqual("advance", action)

    def test_mixed_completion_report_does_not_shortcut(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鐜板湪缁欎簲鍙锋牱鍝佸姞鍏ョ〖姘㈠寲閽犮€傚姞瀹屽憡璇夋垜銆?,
            )
        )

        action = intentHandler._classify_short_experiment_control(
            conn,
            "宸茬粡鍔犲ソ浜嗘槸娴呴粍鑹蹭竴鍒嗛挓",
        )
        self.assertEqual("", action)

    def test_autofill_allows_simple_manual_bool_fields(self):
        schema = {
            "beakers_labeled": {
                "name": "beakers_labeled",
                "type": "bool",
                "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
            },
            "stir_bars_added_to_all": {
                "name": "stir_bars_added_to_all",
                "type": "bool",
                "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
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
                "description": "宸查€氳繃 MCP 鐩存帴鎷嶇収璁板綍褰撳墠鏍峰搧棰滆壊",
            },
            "color_confirmed_by_photo": {
                "name": "color_confirmed_by_photo",
                "type": "bool",
                "description": "宸插熀浜庣収鐗囩‘璁ゅ綋鍓嶆牱鍝侀鑹茬ǔ瀹?,
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
                            "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
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
                "description": "鏈疆鍏卞悓鎿嶄綔璁板綍",
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
                "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
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
                "鍙互鎷嶇収",
                "鍙互鎷嶇収",
            )

        self.assertFalse(handled)

    async def test_handle_user_intent_stages_ready_guard_bypass_when_fast_path_disabled(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.config = {"experiment_fast_path_enabled": False}
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氥€婇摱绾崇背绮掑瓙瀹為獙銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )

        handled = await intentHandler.handle_user_intent(conn, "鍑嗗濂戒簡")

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
                                "鎷嶇収",
                            )

        self.assertTrue(handled)

    async def test_handle_user_intent_routes_photo_confirmation_reply_directly(self):
        conn = _FakeConn()
        conn.intent_type = "function_call"
        conn.dialogue.put(Message(role="assistant", content="鍙互鎷嶇収鍚楋紵"))
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
                        handled = await intentHandler.handle_user_intent(conn, "鍙互鎷嶇収")

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
                                    "鍙互鎷嶇収",
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
                                    "閸忋劑鍎撮崝鐘层偨娴?",
                                )

        self.assertTrue(handled)
        self.assertEqual(
            [("閸忋劑鍎撮崝鐘层偨娴?", "閸忋劑鍎撮崝鐘层偨娴?)],
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
                                        "缁х画涓嬩竴姝?,
                                    )

        self.assertTrue(handled)
        self.assertEqual([("缁х画涓嬩竴姝?, "缁х画涓嬩竴姝?)], seen)

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
                content="鍏堟妸2鍙锋牱鍝佸崟鐙憜濂姐€侀鑹插尯鍩熼湶娓呮锛屽啀璇翠竴澹扳€滄媿鍚р€濄€?,
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
                                    "鍙互鎷嶇収",
                                )

        self.assertTrue(handled)
        self.assertEqual(["鍙互鎷嶇収"], sent)
        self.assertEqual(["server_photo_confirmation"], waits)
        self.assertEqual(1, len(executed))
        self.assertEqual("2鍙锋牱鍝?, executed[0]["photo_name"])
        self.assertEqual(
            "璇锋媿鎽?鍙锋牱鍝佸綋鍓嶇姸鎬佺殑鐓х墖銆?,
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
                                "question": "璇锋媿鎽勪竴鍙锋牱鍝佸綋鍓嶇姸鎬佺殑鐓х墖銆?,
                                "photo_name": "涓€鍙锋牱鍝?,
                            },
                        ):
                            with patch.object(
                                intentHandler,
                                "_execute_server_photo_intent",
                                fake_execute,
                            ):
                                handled = await intentHandler.handle_pending_server_photo_confirmation(
                                    conn,
                                    "鍙互鎷嶇収",
                                    "鍙互鎷嶇収",
                                )

        self.assertTrue(handled)
        self.assertEqual(["鍙互鎷嶇収"], sent)
        self.assertEqual(
            [
                ("delay", "server_photo_confirmation"),
                (
                    "execute",
                    {
                        "device_id": conn.device_id,
                        "question": "璇锋媿鎽勪竴鍙锋牱鍝佸綋鍓嶇姸鎬佺殑鐓х墖銆?,
                        "photo_name": "涓€鍙锋牱鍝?,
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
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜岀杞瓙銆?],
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
                    "娌″惉鎳?,
                    "娌″惉鎳?,
                )

        self.assertTrue(handled)
        self.assertEqual(["娌″惉鎳?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("褰撳墠杩欎竴姝?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆", spoken[0])
        self.assertIn("鍋氬ソ鍚庡憡璇夋垜", spoken[0])

    async def test_explicit_start_guide_reply_includes_experiment_title(self):
        conn = _FakeConn()
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜屾磥鍑€纾佽浆瀛愶紝閬垮厤姹℃煋銆?],
                    },
                }
            }
        }
        conn.experiment_overview = {
            "result": {
                "experiment": {
                    "title": "Ag 绾崇背绮掑瓙鐨勫埗澶囧強鍏跺偓鍖栬繕鍘?4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒"
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
            "浠婂ぉ鎴戜滑鍋氥€夾g 绾崇背绮掑瓙鐨勫埗澶囧強鍏跺偓鍖栬繕鍘?4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
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
                        "[yaml=C:\\demo\\experiments.yaml] 鍏ㄩ儴瀹屾垚銆?,
                        "[2026-05-05T14:36:16.192+08:00] [TRANSCRIPT] [USER] [source=asr] "
                        "[experiment_session_id=exp-old] [current_step_id=step_add_sodium_citrate_all] "
                        "[yaml=C:\\demo\\experiments.yaml] 鏌犳閰搁挔閮藉姞濂戒簡銆?,
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
                "step_prepare_setup_all": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                "step_add_sodium_citrate_all": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                "step_add_agno3_all": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
            }
            step_instructions = {
                "step_prepare_setup_all": "瀹屾垚 1-5 鍙风儳鏉紪鍙峰拰纾佽浆瀛愭斁缃€?,
                "step_add_sodium_citrate_all": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                "step_add_agno3_all": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 AgNO3 鍔犲叆銆?,
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
                        "description": "涓?1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
                        "required": True,
                    },
                    {
                        "name": "beakers_labeled",
                        "type": "bool",
                        "description": "瀹屾垚 1-5 鍙风儳鏉紪鍙?,
                        "required": True,
                    },
                ],
                "step_add_sodium_citrate_all": [
                    {
                        "name": "sodium_citrate_added",
                        "type": "bool",
                        "description": "1-5 鍙锋牱鍝佺粺涓€鍔犲叆鏌犳閰搁挔",
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
                content="浠婂ぉ鎴戜滑鍋氥€夾g 绾崇背绮掑瓙鐨勫埗澶囧強鍏跺偓鍖栬繕鍘?4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜屾磥鍑€纾佽浆瀛愶紝閬垮厤姹℃煋銆?],
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
                    "鎴戝噯澶囧ソ浜?,
                    "鎴戝噯澶囧ソ浜?,
                )

        self.assertTrue(handled)
        self.assertEqual(["鎴戝噯澶囧ソ浜?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鐜板湪鍋氳繖涓€姝?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆", spoken[0])

    async def test_ready_reply_after_start_prompt_does_not_advance_with_session_id(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氥€夾g 绾崇背绮掑瓙鐨勫埗澶囧強鍏跺偓鍖栬繕鍘?4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )
        conn.experiment_session_id = "exp-ready-1"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜屾磥鍑€纾佽浆瀛愶紝閬垮厤姹℃煋銆?],
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
                            "鍑嗗濂戒簡",
                            "鍑嗗濂戒簡",
                        )

        self.assertTrue(handled)
        self.assertEqual(["鍑嗗濂戒簡"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鐜板湪鍋氳繖涓€姝?, spoken[0])
        self.assertEqual(
            conn.sentence_id,
            getattr(conn, "_experiment_ready_guard_bypass_sentence_id", ""),
        )

    async def test_start_first_step_after_start_prompt_returns_current_step(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氥€夾g 绾崇背绮掑瓙鐨勫埗澶囧強鍏跺偓鍖栬繕鍘?4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )
        conn.experiment_session_id = "exp-ready-2"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜屾磥鍑€纾佽浆瀛愶紝閬垮厤姹℃煋銆?],
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
                            "寮€濮嬬涓€姝?,
                            "寮€濮嬬涓€姝?,
                        )

        self.assertTrue(handled)
        self.assertEqual(["寮€濮嬬涓€姝?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鐜板湪鍋氳繖涓€姝?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆", spoken[0])
        self.assertEqual(
            conn.sentence_id,
            getattr(conn, "_experiment_ready_guard_bypass_sentence_id", ""),
        )

    async def test_continue_after_start_prompt_returns_current_step(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="浠婂ぉ鎴戜滑鍋氥€夾g 绾崇背绮掑瓙鐨勫埗澶囧強鍏跺偓鍖栬繕鍘?4-纭濆熀鑻厷鐨勫弽搴斿姩鍔涘鎺㈢┒銆嬨€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵",
            )
        )
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜屾磥鍑€纾佽浆瀛愶紝閬垮厤姹℃煋銆?],
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
                    "缁х画",
                    "缁х画",
                )

        self.assertTrue(handled)
        self.assertEqual(["缁х画"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鐜板湪鍋氳繖涓€姝?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆", spoken[0])

    async def test_repeat_current_step_does_not_sync_graph(self):
        conn = _FakeConn()
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鐜板湪鍋氳繖涓€姝ワ細瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆傚仛濂藉悗鍛婅瘔鎴戙€?,
            )
        )
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                        "safety": ["浣跨敤娲佸噣鐑ф澂鍜屾磥鍑€纾佽浆瀛愶紝閬垮厤姹℃煋銆?],
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
                            "鍐嶈涓€閬?,
                            "鍐嶈涓€閬?,
                        )

        self.assertTrue(handled)
        self.assertEqual(["鍐嶈涓€閬?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("褰撳墠杩欎竴姝?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿", spoken[0])

    async def test_generic_done_after_current_step_advances_one_step_without_context_sync(self):
        conn = _FakeConn()
        conn.experiment_session_id = "exp-generic-done-1"
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.experiment_current_step = {
            "result": {
                "step": {
                    "id": "step_prepare_setup_all",
                    "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                    "prompts": {
                        "instruction": "鍏堝畬鎴?1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆锛屽仛濂藉悗鍛婅瘔鎴戙€?,
                    },
                }
            }
        }
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鐜板湪鍋氳繖涓€姝ワ細1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙銆傚厛瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆锛屽仛濂藉悗鍛婅瘔鎴戙€?,
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
            return "鐜板湪鍋氳繖涓€姝ワ細1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏋告┘閰搁挔銆傛寜 1 鍒?5 鍙烽『搴忓姞鍏?0.50 mL 0.05 mol/L 鏋告┘閰搁挔锛屽仛濂藉悗鍛婅瘔鎴戙€?

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
                            "鍋氬ソ浜?,
                            "鍋氬ソ浜?,
                        )

        self.assertTrue(handled)
        self.assertEqual(["鍋氬ソ浜?], sent)
        self.assertEqual(["exp-generic-done-1"], advanced)
        self.assertEqual(1, len(spoken))
        self.assertIn("缁熶竴鍔犲叆鏋告┘閰搁挔", spoken[0])

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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                                "prompts": {
                                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                                "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
                            },
                        ],
                    }
                }
            if tool_name == "can_proceed":
                state["can_proceed_calls"] += 1
                return {"result": {"ok": False, "message": "灏氭湭瀹屾垚"}}
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                    "褰撳墠姝ラ宸插畬鎴?,
                    "褰撳墠姝ラ宸插畬鎴?,
                )

        self.assertTrue(handled)
        self.assertEqual(["褰撳墠姝ラ宸插畬鎴?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("杩欏嚑涓‘璁?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿", spoken[0])
        self.assertIn("纾佽浆瀛?, spoken[0])
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                                "interaction": {
                                    "fast_path_mode": "confirmation_step",
                                    "capabilities": ["procedural_guidance", "step_confirmation"],
                                },
                                "prompts": {
                                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                                "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                            }
                        ],
                    }
                }
            if tool_name == "can_proceed":
                state["can_proceed_calls"] += 1
                return {
                    "result": {
                        "ok": state["reported"],
                        "message": None if state["reported"] else "灏氭湭瀹屾垚",
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                    "鎸変竴鍒颁簲鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                    "鎸変竴鍒颁簲鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                )

        self.assertTrue(handled)
        self.assertEqual(["鎸変竴鍒颁簲鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("涓€涓‘璁?, spoken[0])
        self.assertIn("鏌犳閰搁挔鍔犲叆", spoken[0])
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                                "prompts": {
                                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                                "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                            }
                        ],
                    }
                }
            if tool_name == "can_proceed":
                state["can_proceed_calls"] += 1
                return {"result": {"ok": False, "message": "灏氭湭瀹屾垚"}}
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                    "鎸変竴鍒颁簲鍙烽『搴忓叏閮ㄥ姞鍏ユ煚妾吀閽?,
                    "鎸変竴鍒颁簲鍙烽『搴忓叏閮ㄥ姞鍏ユ煚妾吀閽?,
                )

        self.assertTrue(handled)
        self.assertEqual(["鎸変竴鍒颁簲鍙烽『搴忓叏閮ㄥ姞鍏ユ煚妾吀閽?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("涓€涓‘璁?, spoken[0])
        self.assertIn("鏌犳閰搁挔鍔犲叆", spoken[0])
        self.assertNotIn("add_fields", [name for name, _args, _priority in tool_calls])

    def test_infer_experiment_step_id_from_natural_assistant_reagent_instruction(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_prepare_setup_all",
                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                "prompts": {
                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                },
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                },
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 AgNO3 鍔犲叆锛堟瘡涓儳鏉潎涓?5.00 mL锛夈€?,
                },
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆H2O2",
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 H2O2 鍔犲叆銆?,
                },
            },
        ]
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鎸?鍒?鍙烽『搴忥紝缁欐瘡涓儳鏉悇鍔犲叆浜旂偣闆堕浂姣崌纭濋吀閾舵憾娑诧紝娉ㄦ剰涓嶈婧呭嚭銆佺紪鍙峰埆寮勬贩锛屽叏閮ㄥ姞瀹屽憡璇夋垜銆?,
            )
        )

        inferred = intentHandler._infer_experiment_step_id_from_context(
            conn,
            original_text="鍏ㄩ儴鍔犲ソ浜?,
            filtered_text="鍏ㄩ儴鍔犲ソ浜?,
        )

        self.assertEqual("step_add_agno3_all", inferred)

    def test_infer_experiment_step_id_ignores_generic_user_advance_text(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_prepare_setup_all",
                "title": "1-5閸欓攱鐗遍崫渚婄窗閸戝棗顦悜褎婢傛稉搴ｎ梿鏉烆剙鐡?,
                "prompts": {
                    "instruction": "鐎瑰本鍨?1-5 閸欓攱鐗遍崫浣烘畱閻懷勬緜缂傛牕褰块崪宀€顥嗘潪顒€鐡欓弨鍓х枂閵?",
                },
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5閸欓攱鐗遍崫渚婄窗缂佺喍绔撮崝鐘插弳閺岀姵顎嬮柊鎼佹寯",
                "prompts": {
                    "instruction": "閹?1 閸?5 閸欑兘銆庢惔蹇曠埠娑撯偓鐎瑰本鍨氶弻鐘愁€嬮柊鎼佹寯閸旂姴鍙嗛妴?",
                },
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5閸欓攱鐗遍崫渚婄窗缂佺喍绔撮崝鐘插弳AgNO3",
                "prompts": {
                    "instruction": "閹?1 閸?5 閸欑兘銆庢惔蹇曠埠娑撯偓鐎瑰本鍨?AgNO3 閸旂姴鍙嗛妴?",
                },
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5閸欓攱鐗遍崫渚婄窗缂佺喍绔撮崝鐘插弳H2O2",
                "prompts": {
                    "instruction": "閹?1 閸?5 閸欑兘銆庢惔蹇曠埠娑撯偓鐎瑰本鍨?H2O2 閸旂姴鍙嗛妴?",
                },
            },
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1閸欓攱鐗遍崫渚婄窗閸旂姴鍙咾Br閵嗕胶鍑藉鏉戣嫙閸旂姴鍙哊aBH4",
                "prompts": {
                    "instruction": "鐎瑰本鍨?1 閸欓攱鐗遍崫?KBr 閸滃瞼鍑藉鏉戝閸忋儱鑻熷ǎ宄板瘧閸氬函绱濊箛顐︹偓鐔峰閸?NaBH4閵?",
                },
            },
        ]
        conn.dialogue.put(
            Message(
                role="assistant",
                content="閻滄澘婀崑姘崇箹娑撯偓濮濄儻绱扮€瑰本鍨?1-5 閸欓攱鐗遍崫浣烘畱閻懷勬緜缂傛牕褰块崪宀€顥嗘潪顒€鐡欓弨鍓х枂閵嗗倸浠涙總钘夋倵閸涘﹨鐦旈幋鎴欌偓?",
            )
        )

        inferred = intentHandler._infer_experiment_step_id_from_context(
            conn,
            original_text="缂佈呯敾娑撳绔村?",
            filtered_text="缂佈呯敾娑撳绔村?",
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
                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "beakers_labeled": {
                        "type": "bool",
                        "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                    },
                    "stir_bars_added_to_all": {
                        "type": "bool",
                        "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
                    },
                },
                "prompts": {
                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                },
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "sodium_citrate_added_to_all": {
                        "type": "bool",
                        "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                    },
                },
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                },
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "agno3_added_to_all": {
                        "type": "bool",
                        "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮?AgNO3 鍔犲叆",
                    },
                },
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 AgNO3 鍔犲叆銆?,
                },
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆H2O2",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "h2o2_added_to_all": {
                        "type": "bool",
                        "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮?H2O2 鍔犲叆",
                    },
                },
                "prompts": {
                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 H2O2 鍔犲叆銆?,
                },
            },
        ]
        order = [step["id"] for step in steps]
        step_by_id = {step["id"]: step for step in steps}
        conn._experiment_yaml_steps_cache = steps
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鎸?鍒?鍙烽『搴忥紝缁欐瘡涓儳鏉悇鍔犲叆浜旂偣闆堕浂姣崌纭濋吀閾舵憾娑诧紝娉ㄦ剰涓嶈婧呭嚭銆佺紪鍙峰埆寮勬贩锛屽叏閮ㄥ姞瀹屽憡璇夋垜銆?,
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
                        "message": None if step["id"] in state["completed"] else "灏氭湭瀹屾垚",
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
        self.assertIn("杩欏嚑涓‘璁?, spoken[0])
        self.assertIn("鐑ф澂缂栧彿", spoken[0])
        self.assertIn("纾佽浆瀛?, spoken[0])
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
                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "beakers_labeled": {
                        "type": "bool",
                        "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                    },
                    "stir_bars_added_to_all": {
                        "type": "bool",
                        "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
                    },
                },
                "prompts": {"instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?},
            },
            {
                "id": "step_add_sodium_citrate_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "sodium_citrate_added_to_all": {
                        "type": "bool",
                        "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                    },
                },
                "prompts": {"instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?},
            },
            {
                "id": "step_add_agno3_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "agno3_added_to_all": {
                        "type": "bool",
                        "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮?AgNO3 鍔犲叆",
                    },
                },
                "prompts": {"instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 AgNO3 鍔犲叆銆?},
            },
            {
                "id": "step_add_h2o2_all",
                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆H2O2",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "h2o2_added_to_all": {
                        "type": "bool",
                        "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮?H2O2 鍔犲叆",
                    },
                },
                "prompts": {"instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚 H2O2 鍔犲叆銆?},
            },
            {
                "id": "step_stirring_all",
                "title": "1-5鍙锋牱鍝侊細鍚屾椂鍚姩鎼呮媽骞舵贩鍖€",
                "interaction": {
                    "fast_path_mode": "confirmation_step",
                    "capabilities": ["step_confirmation"],
                },
                "record_schema": {
                    "all_samples_stirring": {
                        "type": "bool",
                        "description": "宸插悓鏃跺惎鍔?1-5 鍙锋牱鍝佹悈鎷屽苟纭娣峰寑",
                    },
                },
                "prompts": {"instruction": "鍚屾椂鍚姩 1-5 鍙锋牱鍝佹悈鎷屽苟纭娣峰寑銆?},
            },
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘村苟鍔犲叆NaBH4",
                "interaction": {
                    "fast_path_mode": "observation_record_step",
                    "capabilities": ["step_confirmation", "observation_capture"],
                },
                "record_schema": {
                    "KBr_volume": {
                        "type": "bool",
                        "description": "宸叉寜褰撳墠鏍峰搧鐩爣鐢ㄩ噺鍔犲叆 KBr",
                    },
                    "H2O_volume": {
                        "type": "bool",
                        "description": "宸叉寜褰撳墠鏍峰搧鐩爣鐢ㄩ噺鍔犲叆绾按",
                    },
                    "mixed_uniformly": {
                        "type": "bool",
                        "description": "鍔犲叆 KBr 鍜岀函姘村悗宸叉悈鎷屽潎鍖€",
                    },
                    "nabh4_volume": {
                        "type": "bool",
                        "description": "宸插噯纭姞鍏?2.50 mL NaBH4",
                    },
                    "added_quickly": {
                        "type": "bool",
                        "description": "宸插揩閫熷畬鎴?NaBH4 鍔犲叆",
                    },
                    "color": {
                        "type": "string",
                        "description": "褰撳墠鏍峰搧鏈€缁堥鑹?,
                    },
                    "reaction_time": {
                        "type": "float",
                        "description": "褰撳墠鏍峰搧棰滆壊绋冲畾鎵€鐢ㄦ椂闂?,
                    },
                    "color_stable": {
                        "type": "bool",
                        "description": "宸茬‘璁ら鑹茬ǔ瀹?,
                    },
                },
                "prompts": {
                    "instruction": "瀹屾垚 1 鍙锋牱鍝?KBr 鍜岀函姘村姞鍏ュ苟娣峰寑鍚庯紝蹇€熷姞鍏?NaBH4 骞朵繚鎸佹悈鎷岋紝璁板綍棰滆壊绋冲畾鏃堕棿鍜屾敹灏炬儏鍐点€?,
                },
            },
        ]
        order = [step["id"] for step in steps]
        conn._experiment_yaml_steps_cache = steps
        conn.dialogue.put(
            Message(
                role="assistant",
                content="鎺ョ潃鍋?鍙锋牱鍝侊細鍚?鍙风儳鏉姞鍏ヤ簩鐐逛簲闆舵鍗囩〖姘㈠寲閽狅紝淇濇寔鎼呮媽锛屾敞鎰忓畠鏈夎厫铓€鎬с€佺幇閰嶅悗瀹规槗鍒嗚В锛屽敖閲忓揩鍔狅紝鍋氬ソ鍛婅瘔鎴戝凡缁忓仛濂戒簡銆?,
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
                        "message": None if step["id"] in state["completed"] else "灏氭湭瀹屾垚",
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
                    "宸茬粡鍋氬ソ浜?,
                    "宸茬粡鍋氬ソ浜?,
                )

        self.assertTrue(handled)
        self.assertEqual(["宸茬粡鍋氬ソ浜?], sent)
        self.assertEqual("step_sample1_2_add_kbr_water_nabh4", conn.experiment_current_step_id)
        self.assertEqual(1, len(spoken))
        self.assertIn("涓€鏁寸粍鍏抽敭璁板綍", spoken[0])
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
                                "title": "浜斿彿鏍峰搧鎷嶇収纭",
                                "prompts": {
                                    "instruction": "纭浜斿彿鏍峰搧棰滆壊绋冲畾鍚庢媿鐓с€?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_tyndall_observation",
                            "title": "涓佽揪灏旂幇璞¤瀵?,
                            "prompts": {
                                "instruction": "鐢ㄦ縺鍏夌瑪浠庝晶闈㈣瀵?1 鍒?5 鍙锋牱鍝佺殑鍏夎矾銆?,
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
                        "photo_file_name": "浜斿彿鏍峰搧_20260428_175720.png",
                        "photo_path": "C:/demo/浜斿彿鏍峰搧_20260428_175720.png",
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
                                "title": "涓佽揪灏旂幇璞¤瀵?,
                            },
                            "current_step_details": {
                                "instruction": "鐢ㄦ縺鍏夌瑪浠庝晶闈㈣瀵?1 鍒?5 鍙锋牱鍝佺殑鍏夎矾銆?,
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
                    "file_name": "浜斿彿鏍峰搧_20260428_175720.png",
                    "mirrored_path": "C:/demo/浜斿彿鏍峰搧_20260428_175720.png",
                }
            },
            fallback_reply="鎷嶅ソ浜嗐€?,
        )

        self.assertIn("鎷嶅ソ浜?, reply)
        self.assertIn("鎴戞帴鐫€甯︿綘鍋氫笅涓€姝?, reply)
        self.assertIn("鎺ヤ笅鏉ュ仛杩欎竴姝?, reply)
        self.assertIn("涓佽揪灏旂幇璞¤瀵?, reply)
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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                                "prompts": {
                                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
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
                                "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                                "interaction": {
                                    "fast_path_mode": "photo_confirmation_step",
                                },
                                "prompts": {
                                    "instruction": "棰滆壊绋冲畾鍚庢媿鐓ц褰曞綋鍓嶆牱鍝侀鑹诧紝骞惰繘鍏?2 鍙锋牱鍝併€?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_sample2_add_kbr_water_nabh4",
                            "title": "2鍙锋牱鍝侊細鍔犲叆婧村寲閽俱€佺函姘村苟鍔犲叆纭兼阿鍖栭挔",
                            "prompts": {
                                "instruction": "鐜板湪鍋?2 鍙锋牱鍝侊紝鍏堝姞婧村寲閽惧拰绾按锛屽啀鍔犲叆纭兼阿鍖栭挔銆?,
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
                        "photo_file_name": "1鍙锋牱鍝乢20260430_195415.png",
                        "photo_path": "C:/demo/1鍙锋牱鍝乢20260430_195415.png",
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
                                "title": "2鍙锋牱鍝侊細鍔犲叆婧村寲閽俱€佺函姘村苟鍔犲叆纭兼阿鍖栭挔",
                            },
                            "current_step_details": {
                                "instruction": "鐜板湪鍋?2 鍙锋牱鍝侊紝鍏堝姞婧村寲閽惧拰绾按锛屽啀鍔犲叆纭兼阿鍖栭挔銆?,
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
                        "file_name": "1鍙锋牱鍝乢20260430_195415.png",
                        "mirrored_path": "C:/demo/1鍙锋牱鍝乢20260430_195415.png",
                    }
                },
                fallback_reply="鎷嶅ソ浜嗭紝宸茬粡淇濆瓨銆?,
                requested_arguments={"photo_name": "1鍙锋牱鍝?},
            )

        self.assertEqual("step_sample1_5_photo_confirm", state["redirect_step_id"])
        self.assertIn("鎷嶅ソ浜?, reply)
        self.assertIn("鎴戞帴鐫€甯︿綘鍋氫笅涓€姝?, reply)
        self.assertIn("2鍙锋牱鍝?, reply)
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
                            "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣瀛?,
                            "prompts": {
                                "instruction": "鍏堝畬鎴?1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀瀛愭斁缃€?,
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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣瀛?,
                            },
                            "current_step_details": {
                                "instruction": "鍏堝畬鎴?1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀瀛愭斁缃€?,
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
                        "file_name": "1鍙锋牱鍝乢20260504_145218.png",
                        "mirrored_path": "C:/demo/1鍙锋牱鍝乢20260504_145218.png",
                    }
                },
                fallback_reply="鎷嶅ソ浜嗭紝宸茬粡淇濆瓨銆?,
                requested_arguments={"photo_name": "1鍙锋牱鍝?},
            )

        self.assertTrue(state["redirected"])
        self.assertGreaterEqual(state["progress_summary_reads"], 1)
        self.assertIn(
            "redirect_to_step",
            [name for name, _args, _priority in tool_calls],
        )
        self.assertEqual("step_prepare_setup_all", conn.experiment_current_step_id)
        self.assertIn("鎷嶅ソ浜?, reply)
        self.assertIn("褰撳墠瀹為獙鍥捐氨杩樺仠鍦ㄨ繖涓€姝?, reply)
        self.assertIn("鐑ф澂缂栧彿鍜岀瀛愭斁缃?, reply)
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
            "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣瀛?,
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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                                "prompts": {
                                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_sample1_2_add_kbr_water_nabh4",
                            "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘村苟鍔犲叆NaBH4",
                            "prompts": {
                                "instruction": "鍏堝姞鍏?KBr 鍜岀函姘达紝鍐嶅揩閫熷姞鍏?NaBH4 骞舵寔缁悈鎷屻€?,
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
                        "message": "鏃犳硶璺宠浆鍒?step_sample1_5_photo_confirm锛氬墠缃楠ゆ湭瀹屾垚: step_sample1_2_add_kbr_water_nabh4",
                    }
                }
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_sample1_2_add_kbr_water_nabh4",
                                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘村苟鍔犲叆NaBH4",
                            },
                            "current_step_details": {
                                "instruction": "鍏堝姞鍏?KBr 鍜岀函姘达紝鍐嶅揩閫熷姞鍏?NaBH4 骞舵寔缁悈鎷屻€?,
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
                        "file_name": "1鍙锋牱鍝乢20260505_171900.png",
                        "mirrored_path": "C:/demo/1鍙锋牱鍝乢20260505_171900.png",
                    }
                },
                fallback_reply="鎷嶅ソ浜嗭紝宸茬粡淇濆瓨銆?,
                requested_arguments={"photo_name": "1鍙锋牱鍝?},
            )

        self.assertIn("鎷嶅ソ浜?, reply)
        self.assertIn("褰撳墠瀹為獙鍥捐氨杩樺仠鍦ㄨ繖涓€姝?, reply)
        self.assertIn("KBr", reply)
        self.assertIn("NaBH4", reply)
        self.assertNotIn("鍓嶇疆姝ラ鏈畬鎴?, reply)
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
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘村苟鍔犲叆NaBH4",
                "interaction": {
                    "fast_path_mode": "observation_record_step",
                },
                "prompts": {
                    "instruction": (
                        "瀹屾垚 1 鍙锋牱鍝?KBr 鍜岀函姘村姞鍏ュ苟娣峰寑鍚庯紝蹇€熷姞鍏?NaBH4锛?
                        "瀹屾垚鍚庤繘鍏ユ湰鏍峰搧鎷嶇収璁板綍姝ラ銆?
                    ),
                },
                "record_schema": {
                    "color": {"type": "string"},
                },
            },
            {
                "id": "step_sample1_5_photo_confirm",
                "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                "interaction": {
                    "fast_path_mode": "photo_confirmation_step",
                },
                "prompts": {
                    "instruction": "棰滆壊绋冲畾鍚庢媿鐓ц褰曞綋鍓嶆牱鍝侀鑹诧紝骞惰繘鍏?2 鍙锋牱鍝併€?,
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
                content="1鍙锋牱鍝侀鑹插凡缁忕ǔ瀹氾紝鐜板湪鍙互鎷嶇収鍚楋紵",
            )
        )

        inferred = intentHandler._infer_photo_confirmation_step_id_from_context(
            conn,
            {
                "photo_meta": {
                    "file_name": "1鍙锋牱鍝乢20260504_145218.png",
                }
            },
            requested_arguments={"photo_name": "1鍙锋牱鍝佺収鐗?},
        )

        self.assertEqual("step_sample1_5_photo_confirm", inferred)

    def test_infer_photo_confirmation_step_id_uses_recent_dialogue_when_request_is_generic(self):
        conn = _FakeConn()
        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_5_photo_confirm",
                "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                "interaction": {
                    "fast_path_mode": "photo_confirmation_step",
                },
                "prompts": {
                    "instruction": "棰滆壊绋冲畾鍚庢媿鐓ц褰曞綋鍓嶆牱鍝侀鑹诧紝骞惰繘鍏?2 鍙锋牱鍝併€?,
                },
                "record_schema": {
                    "photo_taken": {"type": "bool"},
                    "color_confirmed_by_photo": {"type": "bool"},
                },
            },
        ]
        conn.dialogue.put(
            Message(role="user", content="1鍙锋牱鍝侀鑹插凡缁忕ǔ瀹氫簡銆?)
        )
        conn.dialogue.put(
            Message(role="assistant", content="鐜板湪鍙互鎷嶇収鍚楋紵")
        )

        inferred = intentHandler._infer_photo_confirmation_step_id_from_context(
            conn,
            {
                "photo_meta": {
                    "file_name": "capture.png",
                }
            },
            requested_arguments={"question": "璇锋媿鎽勫綋鍓嶆牱鍝佺殑鐓х墖銆?},
        )

        self.assertEqual("step_sample1_5_photo_confirm", inferred)

    def test_build_pending_server_photo_request_fixed_uses_recent_dialogue_sample_name(self):
        conn = _FakeConn()
        conn.device_id = "94:a9:90:27:3c:84"
        conn.dialogue.put(
            Message(role="user", content="1鍙锋牱鍝侀鑹插凡缁忕ǔ瀹氫簡銆?)
        )
        conn.dialogue.put(
            Message(role="assistant", content="鐜板湪鍙互鎷嶇収鍚楋紵")
        )

        request = intentHandler._build_pending_server_photo_request_fixed(conn)

        self.assertEqual("94:a9:90:27:3c:84", request["device_id"])
        self.assertEqual("1鍙锋牱鍝?, request.get("photo_name"))

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
                            "file_name": "鏃х収鐗?png",
                            "local_path": "C:/demo/old.png",
                            "mtime": 100.0,
                        },
                    }
                return {
                    "success": True,
                    "photo_meta": {
                        "file_name": "涓€鍙锋牱鍝乢20260430_105937.png",
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
            self.assertIn("涓€鍙锋牱鍝?, photo_meta["file_name"])
            self.assertEqual("涓€鍙锋牱鍝?, photo_meta["requested_photo_name"])
            self.assertEqual("涓€鍙锋牱鍝?, requested_arguments["photo_name"])
            return "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑙傚療棰滆壊銆傚仛濂藉悗鍛婅瘔鎴戙€?

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
                                    "question": "璇锋媿鎽勪竴鍙锋牱鍝佸綋鍓嶇姸鎬佺殑鐓х墖銆?,
                                    "photo_name": "涓€鍙锋牱鍝?,
                                },
                            )

        self.assertTrue(handled)
        self.assertEqual(
            ["鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑙傚療棰滆壊銆傚仛濂藉悗鍛婅瘔鎴戙€?],
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
            Message(role="assistant", content="鐜板湪缁?鍙锋牱鍝佹媿鐓х‘璁ゃ€傜幇鍦ㄥ彲浠ユ媿鐓у悧锛?)
        )
        conn._recent_server_photo_confirmation = {
            "captured_at": time.time(),
            "sample_index": 1,
            "sample_name": "1鍙锋牱鍝?,
            "next_step_reply": "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細2鍙锋牱鍝佸厛鍔犲叆婧村寲閽惧拰绾按銆?,
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
                        "question": "璇锋媿鎽?鍙锋牱鍝佸綋鍓嶇姸鎬佺殑鐓х墖銆?,
                        "photo_name": "1鍙锋牱鍝?,
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
                                        "鍙互鎷嶇収",
                                        "鍙互鎷嶇収",
                                    )

        self.assertTrue(handled)
        self.assertEqual(["鍙互鎷嶇収"], sent)
        self.assertEqual(
            ["鎺ヤ笅鏉ュ仛杩欎竴姝ワ細2鍙锋牱鍝佸厛鍔犲叆婧村寲閽惧拰绾按銆?],
            spoken,
        )

    async def legacy_handle_direct_uvvis_shared_blank_prep_calls_measurement_and_waits_for_blank(self):
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
                        "寮€濮嬫祴閲?,
                        "寮€濮嬫祴閲?,
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
                "鍏堜笉瑕佹斁浠讳綍娑蹭綋锛屾垜鍏堣繘琛屾殫鐢垫祦鍜岀┖姘斿熀绾垮噯澶囥€?,
                "杩欎竴姝ヨ繕缂虹函姘寸┖鐧斤紝璇峰厛鎶?1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁鍏ョ函姘存瘮鑹茬毧锛屾斁濂藉悗鍛婅瘔鎴戝彲浠ュ紑濮嬫壂鎻忋€?,
            ],
            spoken,
        )

    async def legacy_handle_direct_uvvis_shared_blank_prep_accepts_start_scan_phrase(self):
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
                        "寮€濮嬫壂鎻忋€?,
                        "寮€濮嬫壂鎻忋€?,
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

    async def legacy_handle_direct_uvvis_shared_blank_followup_ready_reply_calls_measurement(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
            "phase": "await_pure_water_blank",
            "session_key": "lease-1",
        }
        executed = []
        completed = []
        spoken = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"result": {"ok": True}}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細璁板綍 1-5 鍙锋牱鍝佸厜璋便€?

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
                            "閮芥斁濂戒簡",
                            "閮芥斁濂戒簡",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": True,
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
                "observations": "鍏变韩鏆楃數娴併€佺┖姘斿熀绾垮拰绾按绌虹櫧宸插噯澶囧畬鎴?,
            },
            completed[0]["fields"],
        )
        self.assertTrue(completed[0]["auto_advance"])
        self.assertEqual(["鎺ヤ笅鏉ュ仛杩欎竴姝ワ細璁板綍 1-5 鍙锋牱鍝佸厜璋便€?], spoken)

    async def test_handle_direct_uvvis_shared_blank_followup_accepts_start_scan_phrase(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
            "phase": "await_pure_water_blank",
            "session_key": "lease-1",
        }
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
                        "寮€濮嬫壂鎻?,
                        "寮€濮嬫壂鎻?,
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": True,
                    },
                )
            ],
            executed,
        )

    async def legacy_handle_direct_uvvis_shared_blank_infers_step_from_context_when_graph_stale(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.dialogue.put(
            Message(
                role="assistant",
                content=(
                    "鍏堜笉瑕佹斁浠讳綍娑蹭綋锛屾妸鏍峰搧浣嶅拰鍙傛瘮浣嶉兘鐣欑┖锛?
                    "鍑嗗鍋氭殫鐢垫祦鍜岀┖姘斿熀绾裤€傚仛濂戒簡鍛婅瘔鎴戙€?
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
                        "宸茬粡鏀惧ソ浜嗭紝鍙互寮€濮嬩簡銆?,
                        "宸茬粡鏀惧ソ浜嗭紝鍙互寮€濮嬩簡銆?,
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
                "鍏堜笉瑕佹斁浠讳綍娑蹭綋锛屾垜鍏堣繘琛屾殫鐢垫祦鍜岀┖姘斿熀绾垮噯澶囥€?,
                "杩欎竴姝ヨ繕缂虹函姘寸┖鐧斤紝璇峰厛鎶?1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁鍏ョ函姘存瘮鑹茬毧锛屾斁濂藉悗鍛婅瘔鎴戝彲浠ュ紑濮嬫壂鎻忋€?,
            ],
            spoken,
        )

    async def legacy_handle_direct_uvvis_followup_start_scan_infers_step_from_context_when_graph_stale(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.dialogue.put(
            Message(
                role="assistant",
                content=(
                    "鍏堜笉瑕佹斁浠讳綍娑蹭綋锛屾妸鏍峰搧浣嶅拰鍙傛瘮浣嶉兘鐣欑┖锛屽噯澶囧仛鏆楃數娴佸拰绌烘皵鍩虹嚎銆?
                    "鍙互寮€濮嬫壂鎻忔椂鐩存帴鍛婅瘔鎴戝紑濮嬫壂鎻忋€?
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
                        "寮€濮嬫壂鎻忋€?,
                        "寮€濮嬫壂鎻忋€?,
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
                        "Uvvis鐜板湪姝ｅ湪宸ヤ綔鍚楋紵",
                        "Uvvis鐜板湪姝ｅ湪宸ヤ綔鍚楋紵",
                    )

        self.assertTrue(handled)
        self.assertEqual(["Uvvis鐜板湪姝ｅ湪宸ヤ綔鍚楋紵"], started)
        self.assertEqual(
            ["UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傛殫鐢垫祦鏍℃宸茬粡瀹屾垚锛岃繖涓€姝ュ湪绛変綘鎶婁竴鍒颁簲鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁涓€涓函姘存瘮鑹茬毧銆?],
            spoken,
        )

    async def legacy_handle_direct_uvvis_intent_stops_when_redirect_is_rejected(self):
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
                        "message": "鏃犳硶璺宠浆鍒?step_3_uv_vis_shared_dark_blank_prep锛氬墠缃楠ゆ湭瀹屾垚: step_2_tyndall_effect",
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
                    "寮€濮?UV-Vis 鍓嶇疆鏍℃銆?,
                    "寮€濮?UV-Vis 鍓嶇疆鏍℃銆?,
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
            ["褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?UV-Vis 鍓嶇疆鏍℃锛屽厛瀹屾垚涓佽揪灏旂幇璞¤瀵熴€?],
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
                content="鎷嶅ソ浜嗭紝宸茬粡淇濆瓨銆傚綋鍓嶅疄楠屽浘璋辫繕鍋滃湪杩欎竴姝ワ紝鍏堟寜杩欎竴姝ョ户缁€?,
            )
        )

        handled = await intentHandler.handle_direct_uvvis_intent(
            conn,
            "缁х画涓嬩竴姝ャ€?,
            "缁х画涓嬩竴姝ャ€?,
        )

        self.assertFalse(handled)
        self.assertEqual(
            intentHandler._UVVIS_KINETICS_SAMPLE2_STEP_ID,
            getattr(conn, "_uvvis_direct_state", {}).get("step_id"),
        )

    async def legacy_handle_direct_uvvis_shared_blank_prep_reuses_blank_and_advances(self):
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
            return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑁呭叆姣旇壊鐨裤€?

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
                            "寮€濮嬫祴閲?,
                            "寮€濮嬫祴閲?,
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
                "observations": "鍏变韩鏆楃數娴併€佺┖姘斿熀绾垮拰绾按绌虹櫧宸插畬鎴愭垨鍙鐢?,
            },
            completed[0]["fields"],
        )
        self.assertTrue(completed[0]["auto_advance"])
        self.assertEqual(
            [
                "鍏堜笉瑕佹斁浠讳綍娑蹭綋锛屾垜鍏堣繘琛屾殫鐢垫祦鍜岀┖姘斿熀绾垮噯澶囥€?,
                "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑁呭叆姣旇壊鐨裤€?,
            ],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_dark_air_prep_calls_measurement_after_scan_start(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
            "phase": "await_scan_start",
        }
        spoken = []
        executed = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "下一步：装入比色皿。"

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
                            "开始扫描",
                            "开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_prepare_dark_current",
                    {
                        "session_key": "lease-1",
                    },
                )
            ],
            executed,
        )
        self.assertEqual(1, len(completed))
        self.assertEqual(
            {
                "empty_positions_confirmed": True,
                "shared_dark_current_ready": True,
                "shared_air_baseline_ready": True,
                "observations": "共享暗电流和空气能量校正已完成。",
            },
            completed[0]["fields"],
        )
        self.assertTrue(completed[0]["auto_advance"])
        self.assertEqual(["下一步：装入比色皿。"], spoken)

    async def test_handle_direct_uvvis_shared_dark_air_prep_accepts_start_scan_phrase(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
            "phase": "await_empty_positions",
        }
        spoken = []
        executed = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "下一步：装入比色皿。"

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "开始扫描",
                            "开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_prepare_dark_current",
                    {
                        "session_key": "lease-1",
                    },
                ),
            ],
            executed,
        )
        self.assertEqual(1, len(completed))
        self.assertEqual(
            {
                "empty_positions_confirmed": True,
                "shared_dark_current_ready": True,
                "shared_air_baseline_ready": True,
                "observations": "共享暗电流和空气能量校正已完成。",
            },
            completed[0]["fields"],
        )
        self.assertTrue(completed[0]["auto_advance"])
        self.assertEqual(
            ["下一步：装入比色皿。"],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_dark_air_prep_accepts_empty_then_start_in_one_reply(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
            "phase": "await_empty_positions",
        }
        spoken = []
        executed = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "下一步：装入比色皿。"

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "都空了，开始扫描",
                            "都空了，开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                ("uvvis_prepare_dark_current", {"session_key": "lease-1"}),
            ],
            executed,
        )
        self.assertEqual(1, len(completed))
        self.assertEqual(["下一步：装入比色皿。"], spoken)

    async def test_handle_direct_uvvis_shared_dark_air_infers_step_from_context_when_graph_stale(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_prepare_setup_all"
        conn.dialogue.put(
            Message(
                role="assistant",
                content=(
                    "先检查1到5号样品位都为空，参比位也不要放任何液体。"
                    "都空了就告诉我。可以开始时直接说“开始扫描”。"
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

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            return True, ""

        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(intentHandler, "speak_txt", lambda _conn, text: spoken.append(text)):
                        handled = await intentHandler.handle_direct_uvvis_intent(
                            conn,
                            "开始扫描",
                            "开始扫描",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "redirect_to_step",
                    {
                        "session_id": "exp-1",
                        "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
                    },
                    "foreground",
                )
            ],
            redirected,
        )
        self.assertEqual([], executed)
        self.assertEqual(
            {
                "step_id": intentHandler._UVVIS_SHARED_DARK_AIR_STEP_ID,
                "phase": "await_empty_positions",
            },
            getattr(conn, "_uvvis_direct_state", {}),
        )
        self.assertEqual(
            ["先检查1到5号样品位都为空，参比位也不要放任何液体。都空了就告诉我。可以开始时直接说“开始扫描”。"],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_blank_prep_prompts_for_pure_water_when_blank_missing(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        spoken = []
        executed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"success": True, "phase": "shared_prep_ready", "liquid_blank_exists": False}

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(intentHandler, "_execute_uvvis_tool_payload", fake_execute):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "开始扫描",
                        "开始扫描",
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
                "共享前置校正已经准备好。请在 1 到 5 号样品位和参比位各放 1 支纯水比色皿，共 6 支，放好了告诉我可以开始扫描。",
            ],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_blank_followup_ready_reply_calls_measurement(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        conn._uvvis_direct_state = {
            "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
            "phase": "await_pure_water_blank",
            "session_key": "lease-1",
        }
        executed = []
        completed = []
        spoken = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_execute(_conn, tool_name, arguments):
            executed.append((tool_name, dict(arguments)))
            return {"result": {"ok": True}}

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細璁板綍 1-5 鍙锋牱鍝佸厜璋便€?

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
                            "閮芥斁濂戒簡",
                            "閮芥斁濂戒簡",
                        )

        self.assertTrue(handled)
        self.assertEqual(
            [
                (
                    "uvvis_measure_spectra",
                    {
                        "session_key": "lease-1",
                        "sample_positions": [1, 2, 3, 4, 5],
                        "ready_for_samples": True,
                    },
                )
            ],
            executed,
        )
        self.assertEqual(
            [
                {
                    "fields": {
                        "pure_water_blank_ready": True,
                        "reference_cuvette_ready": True,
                        "observations": "褰撳墠鎵规绾按绌虹櫧宸茶褰曞畬鎴愶紝鍙傛瘮浣嶇函姘存瘮鑹茬毧鍙户缁敤浜庡悗缁祴閲忋€?,
                    },
                    "auto_advance": True,
                    "fallback_reply": "绾按绌虹櫧宸茬粡鍑嗗濂戒簡銆?,
                }
            ],
            completed,
        )
        self.assertEqual(["鎺ヤ笅鏉ュ仛杩欎竴姝ワ細璁板綍 1-5 鍙锋牱鍝佸厜璋便€?], spoken)

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
                        "message": "鏃犳硶璺宠浆鍒?step_3_uv_vis_shared_dark_blank_prep锛氬墠缃楠ゆ湭瀹屾垚: step_3_uv_vis_shared_dark_air_prep",
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
                    "寮€濮?UV-Vis 绾按绌虹櫧鏍℃銆?,
                    "寮€濮?UV-Vis 绾按绌虹櫧鏍℃銆?,
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
            ["褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?UV-Vis 鐨勭函姘寸┖鐧芥牎姝ｏ紝鍏堝畬鎴愭殫鐢垫祦鏍℃銆?],
            spoken,
        )

    async def test_handle_direct_uvvis_shared_blank_prep_reuses_blank_and_advances(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        conn._last_uvvis_liquid_blank_state = {
            "liquid_blank_exists": True,
            "liquid_blank_csv": str(Path(__file__).resolve()),
        }
        spoken = []
        completed = []

        async def fake_ensure_session_key(_conn):
            return "lease-1", ""

        async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
            completed.append(
                {
                    "fields": dict(fields),
                    "auto_advance": auto_advance,
                    "fallback_reply": fallback_reply,
                }
            )
            return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑁呭叆姣旇壊鐨裤€?

        def fake_speak_txt(_conn, text):
            spoken.append(text)

        with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
            with patch.object(
                intentHandler,
                "_complete_experiment_step_with_fields",
                fake_complete,
            ):
                with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                    handled = await intentHandler.handle_direct_uvvis_intent(
                        conn,
                        "缁х画涓嬩竴姝?,
                        "缁х画涓嬩竴姝?,
                    )

        self.assertTrue(handled)
        self.assertEqual(
            [
                {
                    "fields": {
                        "pure_water_blank_ready": True,
                        "reference_cuvette_ready": True,
                        "observations": "褰撳墠鎵规绾按绌虹櫧宸茬‘璁ゅ彲澶嶇敤锛屽悗缁祴閲忓皢淇濈暀鎴栭噸鏂版斁濂藉弬姣斾綅绾按姣旇壊鐨裤€?,
                    },
                    "auto_advance": True,
                    "fallback_reply": "褰撳墠鎵规绾按绌虹櫧鍙鐢紝鎺ヤ笅鏉ヨ鍏ユ牱鍝佹瘮鑹茬毧銆?,
                }
            ],
            completed,
        )
        self.assertEqual({}, getattr(conn, "_uvvis_direct_state", {}))
        self.assertEqual(["鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑁呭叆姣旇壊鐨裤€?], spoken)

    async def test_handle_direct_uvvis_shared_blank_prep_reuses_blank_from_shared_dir(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
        spoken = []
        completed = []

        with TemporaryDirectory() as workspace_dir:
            shared_dir = (
                Path(workspace_dir)
                / "lab_runs"
                / "exp1_AgNPs_synthesis"
                / "data"
                / "uv_data_common"
            )
            shared_dir.mkdir(parents=True, exist_ok=True)
            shared_blank_csv = shared_dir / "pure_water_blank_latest.csv"
            shared_blank_csv.write_text("wavelength_nm,absorbance\n", encoding="utf-8")
            conn.config = {
                "LLM": {
                    "codex_app_server": {
                        "workspace": workspace_dir,
                    }
                }
            }

            async def fake_ensure_session_key(_conn):
                return "lease-1", ""

            async def fake_complete(_conn, *, fields, auto_advance, fallback_reply=""):
                completed.append(
                    {
                        "fields": dict(fields),
                        "auto_advance": auto_advance,
                        "fallback_reply": fallback_reply,
                    }
                )
                return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細瑁呭叆姣旇壊鐨裤€?
                return True, "閹恒儰绗呴弶銉ヤ粵鏉╂瑤绔村銉窗鐟佸懎鍙嗗В鏃囧閻ㄨ￥鈧?"

            def fake_speak_txt(_conn, text):
                spoken.append(text)

            with patch.object(
                intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key
            ):
                with patch.object(
                    intentHandler,
                    "_complete_experiment_step_with_fields",
                    fake_complete,
                ):
                    with patch.object(
                        intentHandler,
                        "_execute_uvvis_tool_payload",
                        AsyncMock(),
                    ) as fake_execute:
                        with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                            handled = await intentHandler.handle_direct_uvvis_intent(
                                conn,
                                "缂佈呯敾娑撳绔村?",
                                "缂佈呯敾娑撳绔村?",
                            )

            if not handled:
                with patch.object(intentHandler, "_ensure_uvvis_session_key", fake_ensure_session_key):
                    with patch.object(
                        intentHandler,
                        "_complete_experiment_step_with_fields",
                        fake_complete,
                    ):
                        with patch.object(
                            intentHandler,
                            "_execute_uvvis_tool_payload",
                            AsyncMock(),
                        ) as fallback_execute:
                            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                                handled = await intentHandler._handle_uvvis_shared_blank_prep(
                                    conn,
                                    "开始扫描",
                                    "开始扫描",
                                    {},
                                )
                        fallback_execute.assert_not_awaited()
            self.assertTrue(handled)
            fake_execute.assert_not_awaited()

        self.assertEqual(
            [
                {
                    "fields": {
                        "pure_water_blank_ready": True,
                        "reference_cuvette_ready": True,
                        "observations": "瑜版挸澧犻幍瑙勵偧缁绢垱鎸夌粚铏规瀹歌尙鈥樼拋銈呭讲婢跺秶鏁ら敍灞芥倵缂侇厽绁撮柌蹇撶殺娣囨繄鏆€閹存牠鍣搁弬鐗堟杹婵傝棄寮В鏂剧秴缁绢垱鎸夊В鏃囧閻ㄨ￥鈧?",
                    },
                    "auto_advance": True,
                    "fallback_reply": "瑜版挸澧犻幍瑙勵偧缁绢垱鎸夌粚铏规閸欘垰顦查悽顭掔礉閹恒儰绗呴弶銉棅閸忋儲鐗遍崫浣圭槷閼硅尙姣ч妴?",
                }
            ],
            completed,
        )
        self.assertEqual(
            str(shared_blank_csv.resolve()),
            getattr(conn, "_last_uvvis_liquid_blank_state", {}).get("liquid_blank_csv"),
        )
        self.assertEqual(
            "reused_from_shared_dir",
            getattr(conn, "_last_uvvis_liquid_blank_state", {}).get("liquid_blank_status"),
        )
        self.assertEqual({}, getattr(conn, "_uvvis_direct_state", {}))
        self.assertEqual(["閹恒儰绗呴弶銉ヤ粵鏉╂瑤绔村銉窗鐟佸懎鍙嗗В鏃囧閻ㄨ￥鈧?"], spoken)

    async def test_handle_direct_uvvis_spectra_measurement_records_all_samples(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID
        spoken = []
        completed = []

        with TemporaryDirectory() as temp_dir:
            conn.device_id = "94:a9:90:27:3c:84"
            conn.config = {
                "experiment_fast_path_enabled": False,
                "uvvis_scan_output_root": temp_dir,
            }
            absorbance_paths = {}
            for index in range(1, 6):
                csv_path = (
                    Path(temp_dir)
                    / "94_a9_90_27_3c_84"
                    / f"sample{index}_latest_absorbance.csv"
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["wavelength_nm", "absorbance"],
                    )
                    writer.writeheader()
                    for wavelength_nm in intentHandler._UVVIS_SPECTRA_WAVELENGTH_GRID:
                        writer.writerow(
                            {
                                "wavelength_nm": wavelength_nm,
                                "absorbance": round((index * 0.1) + (wavelength_nm - 400) / 1000, 4),
                            }
                        )
                absorbance_paths[index] = str(csv_path)

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
                                "absorbance_output_csv": absorbance_paths[index],
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
                return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細娓呮礂姣旇壊鐨裤€?

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
                                "閮芥斁濂戒簡",
                                "閮芥斁濂戒簡",
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
                    "observations": "1-5鍙锋牱鍝佹壒閲忔壂鎻忓畬鎴愶紝1鍙锋牱鍝佄籱ax=410.0nm锛?鍙锋牱鍝佄籱ax=420.0nm锛?鍙锋牱鍝佄籱ax=430.0nm锛?鍙锋牱鍝佄籱ax=440.0nm锛?鍙锋牱鍝佄籱ax=450.0nm锛?00-700nm锛?0nm姝ラ暱锛夌殑鏍℃鍚稿厜搴︾粨鏋滃拰鍏夎氨鍥惧凡淇濆瓨",
                },
                completed[0]["fields"],
            )
            self.assertTrue(completed[0]["auto_advance"])
            artifacts = getattr(conn, "_last_uvvis_spectra_artifacts", {})
            self.assertTrue(artifacts.get("all_expected_outputs_exist"))
            self.assertTrue(Path(artifacts["combined_absorbance_csv"]).exists())
            self.assertTrue(Path(artifacts["summary_csv"]).exists())
            self.assertTrue(Path(artifacts["plot_svg"]).exists())
            self.assertTrue(Path(artifacts["manifest_json"]).exists())
            combined_csv_text = Path(artifacts["combined_absorbance_csv"]).read_text(encoding="utf-8-sig")
            self.assertIn("sample_1_corrected_absorbance", combined_csv_text)
            self.assertIn("400", combined_csv_text)
            self.assertIn("700", combined_csv_text)
            self.assertEqual(1, len(spoken))
            self.assertIn("1鍙?10.0绾崇背", spoken[0])
            self.assertIn("5鍙?50.0绾崇背", spoken[0])
            self.assertIn("400鍒?00绾崇背姣忛殧10绾崇背鐨勬牎姝ｅ惛鍏夊害缁撴灉鍜屽厜璋卞浘宸蹭繚瀛?, spoken[0])
            self.assertIn("鎺ヤ笅鏉ュ仛杩欎竴姝ワ細娓呮礂姣旇壊鐨裤€?, spoken[0])

    async def test_handle_direct_uvvis_spectra_measurement_skips_negative_absorbance(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = intentHandler._UVVIS_SAMPLE_RECORD_STEP_ID
        completed = []

        with TemporaryDirectory() as temp_dir:
            conn.device_id = "94:a9:90:27:3c:84"
            conn.config = {
                "experiment_fast_path_enabled": False,
                "uvvis_scan_output_root": temp_dir,
            }
            absorbance_paths = {}
            for index in range(1, 6):
                csv_path = (
                    Path(temp_dir)
                    / "94_a9_90_27_3c_84"
                    / f"sample{index}_latest_absorbance.csv"
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["wavelength_nm", "absorbance"],
                    )
                    writer.writeheader()
                    for wavelength_nm in intentHandler._UVVIS_SPECTRA_WAVELENGTH_GRID:
                        writer.writerow(
                            {
                                "wavelength_nm": wavelength_nm,
                                "absorbance": round((index * 0.05) + (wavelength_nm - 400) / 2000, 6),
                            }
                        )
                absorbance_paths[index] = str(csv_path)

            async def fake_ensure_session_key(_conn):
                return "lease-1", ""

            async def fake_execute(_conn, tool_name, arguments):
                self.assertEqual("uvvis_measure_spectra", tool_name)
                self.assertTrue(arguments["ready_for_samples"])
                return {
                    "result": {
                        "samples": [
                            {
                                "sample_position": 1,
                                "lambda_max_nm": 430.0,
                                "max_absorbance": 0.021523,
                                "absorbance_output_csv": absorbance_paths[1],
                            },
                            {
                                "sample_position": 2,
                                "lambda_max_nm": 400.0,
                                "max_absorbance": -0.005743,
                                "absorbance_output_csv": absorbance_paths[2],
                            },
                            {
                                "sample_position": 3,
                                "lambda_max_nm": 440.0,
                                "max_absorbance": 0.024618,
                                "absorbance_output_csv": absorbance_paths[3],
                            },
                            {
                                "sample_position": 4,
                                "lambda_max_nm": 400.0,
                                "max_absorbance": -0.01182,
                                "absorbance_output_csv": absorbance_paths[4],
                            },
                            {
                                "sample_position": 5,
                                "lambda_max_nm": 410.0,
                                "max_absorbance": 1.318574,
                                "absorbance_output_csv": absorbance_paths[5],
                            },
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
                                "閮芥斁濂戒簡",
                                "閮芥斁濂戒簡",
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
                            "閮芥斁濂戒簡",
                            "閮芥斁濂戒簡",
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
            ["杩欎竴姝ョ己灏戠函姘寸┖鐧斤紝鎴戝厛閫€鍥炲墠缃牎姝ｃ€傝鍏堟妸鏍峰搧浣嶅拰鍙傛瘮浣嶉兘娓呯┖锛屽啀鍛婅瘔鎴戝紑濮嬨€?],
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
                        "寮€濮嬪姩鍔涘娴嬮噺",
                        "寮€濮嬪姩鍔涘娴嬮噺",
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
                "鍏堜繚鎸佹牱鍝佷綅涓虹┖锛屾垜鍏堝仛鏆楃數娴佸拰 400 绾崇背绌烘皵鍩虹嚎鍑嗗銆?,
                "杩欎竴姝ユ寚瀹氱殑鍙傛瘮娑?鍖栧绌虹櫧娑茶繕娌℃斁濂斤紝璇锋妸鏍峰搧浣嶅拰鍙傛瘮浣嶅悓鏃舵斁鍏ヨ姝ラ鎸囧畾鐨勭┖鐧芥恫锛屼笉鏄函姘淬€傛斁濂戒簡鍛婅瘔鎴戙€?,
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
                        "2鍙蜂綅鏀惧ソ浜?,
                        "2鍙蜂綅鏀惧ソ浜?,
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
            ["娑蹭綋绌虹櫧宸茬粡璁板綍濂戒簡銆傝鎶婂弬姣斾綅淇濇寔涓嶅彉锛屾妸2鍙锋牱鍝佷綅鎹㈡垚鐪熷疄鍙嶅簲娑诧紝鏀惧ソ浜嗗憡璇夋垜銆?],
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

        with TemporaryDirectory() as temp_dir:
            conn.device_id = "94:a9:90:27:3c:84"
            conn.config = {
                "experiment_fast_path_enabled": False,
                "uvvis_scan_output_root": temp_dir,
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
                                "鍙互寮€濮嬩簡",
                                "鍙互寮€濮嬩簡",
                            )

            self.assertTrue(handled)
            self.assertEqual(1, len(completed))
            self.assertFalse(completed[0]["auto_advance"])
            self.assertEqual("鎴戣褰曞ソ浜嗭紝鍙互缁х画杩涜涓嬩竴姝ヤ簡鍚楋紵", completed[0]["fallback_reply"])
            for index in range(35):
                self.assertEqual(
                    record_fields[f"t{index}_absorbance"],
                    completed[0]["fields"][f"t{index}_absorbance"],
                )
            self.assertTrue(completed[0]["fields"]["bubble_observed"])
            self.assertEqual(
                "3鍙锋牱鍝?00绾崇背鍔ㄥ姏瀛︽祴閲忓畬鎴愶紝鍏辫褰?5涓椂闂寸偣銆傦紱0-34min 姣忓垎閽熶竴涓偣鐨勬牎姝ｅ惛鍏夊害缁撴灉鍜屽姩鍔涘鏇茬嚎宸蹭繚瀛?,
                completed[0]["fields"]["observations"],
            )
            artifacts = getattr(conn, "_last_uvvis_kinetics_artifacts", {})
            self.assertTrue(artifacts.get("all_expected_outputs_exist"))
            self.assertTrue(Path(artifacts["timeseries_csv"]).exists())
            self.assertTrue(Path(artifacts["plot_svg"]).exists())
            self.assertTrue(Path(artifacts["manifest_json"]).exists())
            kinetics_csv_text = Path(artifacts["timeseries_csv"]).read_text(encoding="utf-8-sig")
            self.assertIn("corrected_absorbance", kinetics_csv_text)
            self.assertIn("0,", kinetics_csv_text)
            self.assertIn("34,", kinetics_csv_text)
            self.assertEqual("done", getattr(conn, "_uvvis_direct_state", {}).get("phase"))
            self.assertEqual(
                ["鎴戣褰曞ソ浜嗭紝鍙互缁х画杩涜涓嬩竴姝ヤ簡鍚楋紵0鍒?4鍒嗛挓姣忓垎閽熶竴涓偣鐨勬牎姝ｅ惛鍏夊害缁撴灉鍜屽姩鍔涘鏇茬嚎涔熷凡缁忎繚瀛樸€?],
                spoken,
            )

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
            return True, "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細鏁版嵁鍒嗘瀽銆?

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
                            "缁х画涓嬩竴姝?,
                            "缁х画涓嬩竴姝?,
                        )

        self.assertTrue(handled)
        self.assertEqual([True], released)
        self.assertEqual(["鎺ヤ笅鏉ュ仛杩欎竴姝ワ細鏁版嵁鍒嗘瀽銆?], spoken)
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
                    "寮€濮嬪垎鏋?,
                    "寮€濮嬪垎鏋?,
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
                "鎴戝厛鏍稿涓€涓嬩簲鍙锋牱鍝佸悗闈㈢殑绱ф帴姝ラ锛岄伩鍏嶆妸浣犲甫閿欍€?
            ),
        )
        self.assertEqual(
            "",
            textUtils.filter_spoken_backstage_text(
                "鎴戝啀鐪嬩竴鐪艰繖涓€姝ヨ浣犲洖鎶ヤ粈涔堛€?
            ),
        )
        self.assertEqual(
            "",
            textUtils.filter_spoken_backstage_text(
                "鎴戞帴鐫€纭涓€鍙锋牱鍝佽繖涓€灏忔鐨勮褰曢」锛屽彧璁颁綘鍒氭墠鎶ョ殑棰滆壊鍜屾椂闂淬€?
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                                "interaction": {
                                    "fast_path_mode": "confirmation_step",
                                    "capabilities": ["procedural_guidance", "step_confirmation"],
                                },
                                "prompts": {
                                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                                "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                    "鎸変竴鍒颁簲鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                    "鎸変竴鍒颁簲鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
                )

        self.assertTrue(handled)
        self.assertEqual(["鎸変竴鍒颁簲鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?], sent)
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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                                "prompts": {
                                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                                "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                    "鍏ㄩ儴瀹屾垚",
                    "鍏ㄩ儴瀹屾垚",
                )

        self.assertTrue(handled)
        self.assertEqual(["鍏ㄩ儴瀹屾垚"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鏌犳閰搁挔", spoken[0])
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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                                "prompts": {
                                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                                "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                    "褰撳墠姝ラ宸插畬鎴?,
                    "褰撳墠姝ラ宸插畬鎴?,
                )

        self.assertTrue(handled)
        self.assertEqual(["褰撳墠姝ラ宸插畬鎴?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鏌犳閰搁挔", spoken[0])
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                                "prompts": {
                                    "instruction": "鎸?1 鍒?5 鍙烽『搴忕粺涓€瀹屾垚鏌犳閰搁挔鍔犲叆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_agno3_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                                "description": "宸叉寜 1-5 鍙烽『搴忓畬鎴愬叏閮ㄦ煚妾吀閽犲姞鍏?,
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆AgNO3",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?5.00 mL AgNO3銆?,
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
                    "鎸変竴鍒颁簲鍙烽『搴忓叏閮ㄥ姞鍏ユ煚妾吀閽?,
                    "鎸変竴鍒颁簲鍙烽『搴忓叏閮ㄥ姞鍏ユ煚妾吀閽?,
                )

        self.assertTrue(handled)
        self.assertEqual(["鎸変竴鍒颁簲鍙烽『搴忓叏閮ㄥ姞鍏ユ煚妾吀閽?], sent)
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
                            "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘村苟鍔犲叆NaBH4",
                            "interaction": {
                                "fast_path_mode": "observation_record_step",
                                "capabilities": ["step_confirmation", "observation_capture"],
                            },
                            "prompts": {
                                "instruction": "瀹屾垚 1 鍙锋牱鍝?KBr 鍜岀函姘村姞鍏ュ苟娣峰寑鍚庯紝蹇€熷姞鍏?NaBH4 骞朵繚鎸佹悈鎷岋紝璁板綍棰滆壊绋冲畾鏃堕棿鍜屾敹灏炬儏鍐点€?,
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
                            {"name": "KBr_volume", "type": "bool", "description": "宸叉寜褰撳墠鏍峰搧鐩爣鐢ㄩ噺鍔犲叆 KBr"},
                            {"name": "H2O_volume", "type": "bool", "description": "宸叉寜褰撳墠鏍峰搧鐩爣鐢ㄩ噺鍔犲叆绾按"},
                            {"name": "mixed_uniformly", "type": "bool", "description": "鍔犲叆 KBr 鍜岀函姘村悗宸叉悈鎷屽潎鍖€"},
                            {"name": "nabh4_volume", "type": "bool", "description": "宸插噯纭姞鍏?2.50 mL NaBH4"},
                            {"name": "added_quickly", "type": "bool", "description": "宸插揩閫熷畬鎴?NaBH4 鍔犲叆"},
                            {"name": "color", "type": "string", "description": "褰撳墠鏍峰搧鏈€缁堥鑹?},
                            {"name": "reaction_time", "type": "float", "description": "褰撳墠鏍峰搧棰滆壊绋冲畾鎵€鐢ㄦ椂闂?},
                            {"name": "color_stable", "type": "bool", "description": "宸茬‘璁ら鑹茬ǔ瀹?},
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
                    "宸茬粡鍋氬ソ浜?,
                    "宸茬粡鍋氬ソ浜?,
                )

        self.assertTrue(handled)
        self.assertEqual(["宸茬粡鍋氬ソ浜?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("棰滆壊", spoken[0])
        self.assertIn("鏃堕棿", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])

    async def test_strict_graph_observation_report_writes_current_step_and_advances_to_photo(self):
        conn = _FakeConn()
        conn.config = {"experiment_fast_path_enabled": False}
        spoken = []
        sent = []
        tool_calls = []
        state = {"advanced": False}

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
                                "id": "step_sample1_2_add_kbr_water_nabh4",
                                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                                "interaction": {
                                    "fast_path_mode": "observation_record_step",
                                    "capabilities": ["step_confirmation", "observation_capture"],
                                },
                                "prompts": {
                                    "instruction": "鍏堝湪鍔犲叆 NaBH4 鐨勫悓鏃跺紑濮嬭鏃讹紝鎸佺画鎼呮媽锛屾寔缁瀵熼鑹插彉鍖栥€?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_sample1_5_photo_confirm",
                            "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                            "interaction": {
                                "fast_path_mode": "photo_confirmation_step",
                            },
                            "prompts": {
                                "instruction": "鎷嶇収璁板綍褰撳墠鏍峰搧棰滆壊銆?,
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
                            "missing_fields": [
                                "KBr_volume",
                                "H2O_volume",
                                "mixed_uniformly",
                                "nabh4_volume",
                                "added_quickly",
                                "color",
                                "reaction_time",
                                "color_stable",
                            ]
                        },
                    }
                }
            if tool_name == "get_schema":
                if not state["advanced"]:
                    return {
                        "result": {
                            "ok": True,
                            "schema_view": [
                                {"name": "KBr_volume", "type": "bool", "description": "宸叉寜褰撳墠鏍峰搧鐩爣鐢ㄩ噺鍔犲叆 KBr"},
                                {"name": "H2O_volume", "type": "bool", "description": "宸叉寜褰撳墠鏍峰搧鐩爣鐢ㄩ噺鍔犲叆绾按"},
                                {"name": "mixed_uniformly", "type": "bool", "description": "鍔犲叆 KBr 鍜岀函姘村悗宸叉悈鎷屽潎鍖€"},
                                {"name": "nabh4_volume", "type": "bool", "description": "宸插噯纭姞鍏?2.50 mL NaBH4"},
                                {"name": "added_quickly", "type": "bool", "description": "宸插揩閫熷畬鎴?NaBH4 鍔犲叆"},
                                {"name": "color", "type": "string", "description": "褰撳墠鏍峰搧鏈€缁堥鑹?},
                                {"name": "reaction_time", "type": "float", "description": "褰撳墠鏍峰搧棰滆壊绋冲畾鎵€鐢ㄦ椂闂?},
                                {"name": "color_stable", "type": "bool", "description": "宸茬‘璁ら鑹茬ǔ瀹?},
                            ],
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {"name": "photo_taken", "type": "bool", "description": "宸插畬鎴愭媿鐓?},
                            {"name": "color_confirmed_by_photo", "type": "bool", "description": "宸插熀浜庣収鐗囩‘璁ゅ綋鍓嶆牱鍝侀鑹茬ǔ瀹?},
                        ],
                    }
                }
            if tool_name == "add_fields":
                self.assertEqual(
                    {
                        "KBr_volume": True,
                        "H2O_volume": True,
                        "mixed_uniformly": True,
                        "nabh4_volume": True,
                        "added_quickly": True,
                        "color": "榛戠伆鑹?,
                        "reaction_time": 1.0,
                        "color_stable": True,
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
                state["advanced"] = True
                conn.experiment_current_step_id = "step_sample1_5_photo_confirm"
                return {"result": {"ok": True}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": "step_sample1_5_photo_confirm",
                                "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                            },
                            "current_step_details": {
                                "instruction": "棰滆壊绋冲畾鍚庢媿鐓ц褰曞綋鍓嶆牱鍝侀鑹诧紝骞惰繘鍏?2 鍙锋牱鍝併€?,
                            },
                        },
                    }
                }
            raise AssertionError(f"unexpected tool call: {tool_name}")

        conn._experiment_yaml_steps_cache = [
            {
                "id": "step_sample1_2_add_kbr_water_nabh4",
                "title": "1鍙锋牱鍝侊細鍔犲叆KBr銆佺函姘淬€丯aBH4骞惰鏃惰瀵熼鑹?,
                "prompts": {
                    "instruction": "鍏堝姞鍏ュ苟娣峰寑 KBr 涓庣函姘达紝鍐嶅揩閫熷姞鍏?NaBH4锛屼粠鍔犲叆 NaBH4 鐨勭灛闂村紑濮嬭鏃跺苟鎸佺画鎼呮媽銆?,
                },
            },
            {
                "id": "step_sample1_5_photo_confirm",
                "title": "1鍙锋牱鍝侊細棰滆壊绋冲畾鍚庢媿鐓ц褰?,
                "interaction": {"fast_path_mode": "photo_confirmation_step"},
                "prompts": {
                    "instruction": "棰滆壊绋冲畾鍚庢媿鐓ц褰曞綋鍓嶆牱鍝侀鑹诧紝骞惰繘鍏?2 鍙锋牱鍝併€?,
                },
            },
        ]
        conn._call_experiment_graph_tool = fake_call

        with patch.object(intentHandler, "send_stt_message", fake_send_stt_message):
            with patch.object(intentHandler, "speak_txt", fake_speak_txt):
                handled = await intentHandler.handle_experiment_control_strict_graph_intent(
                    conn,
                    "榛戠伆鑹蹭竴鍒嗛挓",
                    "榛戠伆鑹蹭竴鍒嗛挓",
                )

        self.assertTrue(handled)
        self.assertEqual(["榛戠伆鑹蹭竴鍒嗛挓"], sent)
        self.assertEqual(["1鍙锋牱鍝侀鑹插凡缁忕ǔ瀹氾紝鐜板湪鍙互鎷嶇収鍚楋紵"], spoken)
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn("finish_trial", [name for name, _args, _priority in tool_calls])
        self.assertIn("proceed_to_next_step", [name for name, _args, _priority in tool_calls])

    async def test_tyndall_observation_report_writes_all_samples_and_advances_to_uvvis(self):
        conn = _FakeConn()
        conn.experiment_current_step_id = "step_2_tyndall_effect"
        spoken = []
        sent = []
        tool_calls = []
        state = {"advanced": False}

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
                                "id": "step_2_tyndall_effect",
                                "title": "涓佽揪灏旂幇璞¤瀵?,
                                "interaction": {
                                    "fast_path_mode": "observation_record_step",
                                    "capabilities": ["observation_capture"],
                                },
                                "prompts": {
                                    "instruction": "鐢ㄦ縺鍏夌瑪鐓у皠姣忎釜鏍峰搧锛岃瀵熷苟璁板綍涓佽揪灏旂幇璞°€?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                            "title": "1-5鍙锋牱鍝侊細鏆楃數娴佸拰绾按绌虹櫧鏍℃",
                            "prompts": {
                                "instruction": (
                                    "鐜板湪寮€濮?UV-Vis 鍓嶇疆鏍℃锛屽厛鎻愮ず涓昏璇濅汉鍏堜笉瑕佹斁浠讳綍娑蹭綋锛?
                                    "鐒跺悗浣跨敤褰撳墠 session_key 璋冪敤 uvvis_measure_spectra銆?
                                ),
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
                            "missing_fields": [
                                "sample_1_tyndall",
                                "sample_2_tyndall",
                                "sample_3_tyndall",
                                "sample_4_tyndall",
                                "sample_5_tyndall",
                            ]
                        },
                    }
                }
            if tool_name == "get_schema":
                return {
                    "result": {
                        "ok": True,
                        "schema_view": [
                            {
                                "name": "sample_1_tyndall",
                                "type": "boolean",
                                "description": "1鍙锋牱鍝佹槸鍚﹁瀵熷埌涓佽揪灏旂幇璞?,
                            },
                            {
                                "name": "sample_2_tyndall",
                                "type": "boolean",
                                "description": "2鍙锋牱鍝佹槸鍚﹁瀵熷埌涓佽揪灏旂幇璞?,
                            },
                            {
                                "name": "sample_3_tyndall",
                                "type": "boolean",
                                "description": "3鍙锋牱鍝佹槸鍚﹁瀵熷埌涓佽揪灏旂幇璞?,
                            },
                            {
                                "name": "sample_4_tyndall",
                                "type": "boolean",
                                "description": "4鍙锋牱鍝佹槸鍚﹁瀵熷埌涓佽揪灏旂幇璞?,
                            },
                            {
                                "name": "sample_5_tyndall",
                                "type": "boolean",
                                "description": "5鍙锋牱鍝佹槸鍚﹁瀵熷埌涓佽揪灏旂幇璞?,
                            },
                            {
                                "name": "tyndall_intensity_comparison",
                                "type": "string",
                                "description": "鍚勬牱鍝佷竵杈惧皵鏁堝簲寮哄害瀵规瘮鎻忚堪",
                            },
                        ],
                    }
                }
            if tool_name == "add_fields":
                self.assertEqual(
                    {
                        "sample_1_tyndall": True,
                        "sample_2_tyndall": True,
                        "sample_3_tyndall": True,
                        "sample_4_tyndall": True,
                        "sample_5_tyndall": True,
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
                state["advanced"] = True
                conn.experiment_current_step_id = intentHandler._UVVIS_SHARED_BLANK_STEP_ID
                return {"result": {"ok": True}}
            if tool_name == "get_progress_summary":
                return {
                    "result": {
                        "ok": True,
                        "summary": {
                            "current_step": {
                                "step_id": intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
                                "title": "1-5鍙锋牱鍝侊細鏆楃數娴佸拰绾按绌虹櫧鏍℃",
                            },
                            "current_step_details": {
                                "instruction": (
                                    "鐜板湪寮€濮?UV-Vis 鍓嶇疆鏍℃锛屽厛鎻愮ず涓昏璇濅汉鍏堜笉瑕佹斁浠讳綍娑蹭綋锛?
                                    "鐒跺悗浣跨敤褰撳墠 session_key 璋冪敤 uvvis_measure_spectra銆?
                                ),
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
                    "鍏ㄩ儴閮芥湁涓佽揪灏旂幇璞★紝缁х画鍋氫笅涓€姝ャ€?,
                    "鍏ㄩ儴閮芥湁涓佽揪灏旂幇璞★紝缁х画鍋氫笅涓€姝ャ€?,
                )

        self.assertTrue(handled)
        self.assertEqual(["鍏ㄩ儴閮芥湁涓佽揪灏旂幇璞★紝缁х画鍋氫笅涓€姝ャ€?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("绾按绌虹櫧鏍℃", spoken[0])
        self.assertIn("鏀惧叆绾按姣旇壊鐨?, spoken[0])
        self.assertIn("鍙互寮€濮嬫壂鎻?, spoken[0])
        self.assertNotIn("鍏堜笉瑕佹斁浠讳綍娑蹭綋", spoken[0])
        self.assertNotIn("session_key", spoken[0])
        self.assertNotIn("ready_for_samples", spoken[0])
        self.assertIn("add_fields", [name for name, _args, _priority in tool_calls])
        self.assertIn("finish_trial", [name for name, _args, _priority in tool_calls])
        self.assertIn("proceed_to_next_step", [name for name, _args, _priority in tool_calls])
        self.assertEqual(
            intentHandler._UVVIS_SHARED_BLANK_STEP_ID,
            conn.experiment_current_step_id,
        )

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
                                "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                                "prompts": {
                                    "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
                                },
                            },
                        }
                    }
                return {
                    "result": {
                        "ok": True,
                        "step": {
                            "id": "step_add_sodium_citrate_all",
                            "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            "prompts": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                                "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
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
                                "title": "1-5鍙锋牱鍝侊細缁熶竴鍔犲叆鏌犳閰搁挔",
                            },
                            "current_step_details": {
                                "instruction": "鎸?1 鍒?5 鍙烽『搴忓姞鍏?1.00 mL 鏌犳閰搁挔銆?,
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
                    "鍏ㄩ儴瀹屾垚",
                    "鍏ㄩ儴瀹屾垚",
                )

        self.assertTrue(handled)
        self.assertEqual(["鍏ㄩ儴瀹屾垚"], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鏌犳閰搁挔", spoken[0])
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
                            "title": "1-5鍙锋牱鍝侊細鍑嗗鐑ф澂涓庣杞瓙",
                            "prompts": {
                                "instruction": "瀹屾垚 1-5 鍙锋牱鍝佺殑鐑ф澂缂栧彿鍜岀杞瓙鏀剧疆銆?,
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
                                "description": "宸插畬鎴?1-5 鍙风儳鏉紪鍙?,
                            },
                            {
                                "name": "stir_bars_added_to_all",
                                "type": "bool",
                                "description": "宸蹭负 1-5 鍙风儳鏉叏閮ㄦ斁鍏ョ杞瓙",
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
                    "涓€鍒颁簲鍙峰弻鏉叏閮ㄦ斁鍏ョ瀛?,
                    "涓€鍒颁簲鍙峰弻鏉叏閮ㄦ斁鍏ョ瀛?,
                )

        self.assertTrue(handled)
        self.assertEqual(["涓€鍒颁簲鍙峰弻鏉叏閮ㄦ斁鍏ョ瀛?], sent)
        self.assertEqual(1, len(spoken))
        self.assertIn("鐑ф澂缂栧彿", spoken[0])
        self.assertNotIn("纾佽浆瀛?, spoken[0])
        self.assertEqual({"stir_bars_added_to_all"}, state["written_fields"])

if __name__ == "__main__":
    unittest.main()
