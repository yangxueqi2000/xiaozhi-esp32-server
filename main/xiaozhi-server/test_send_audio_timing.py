import sys
import types
import unittest
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


fake_util_module = types.ModuleType("core.utils.util")


async def _fake_audio_to_data(*args, **kwargs):
    return []


fake_util_module.audio_to_data = _fake_audio_to_data
fake_util_module.audio_to_data_stream = lambda *args, **kwargs: []
fake_util_module.audio_bytes_to_data_stream = lambda *args, **kwargs: []
sys.modules.setdefault("core.utils.util", fake_util_module)


fake_rate_controller_module = types.ModuleType("core.utils.audioRateController")


class _DummyAudioRateController:
    def __init__(self, *args, **kwargs):
        self.queue = []
        self.queue_empty_event = None


fake_rate_controller_module.AudioRateController = _DummyAudioRateController
sys.modules.setdefault("core.utils.audioRateController", fake_rate_controller_module)


fake_report_module = types.ModuleType("core.handle.reportHandle")
fake_report_module.enqueue_tts_report = lambda *args, **kwargs: None
sys.modules.setdefault("core.handle.reportHandle", fake_report_module)


fake_output_counter_module = types.ModuleType("core.utils.output_counter")
fake_output_counter_module.add_device_output = lambda *args, **kwargs: None
sys.modules.setdefault("core.utils.output_counter", fake_output_counter_module)


fake_logger_module = types.ModuleType("config.logger")


class _DummyLogger:
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


fake_logger_module.setup_logging = lambda: _DummyLogger()
sys.modules.setdefault("config.logger", fake_logger_module)


from core.handle.sendAudioHandle import (
    _estimate_sent_audio_remaining_ms,
    _resolve_tts_sentence_bridge_delay_ms,
    _resolve_tts_stop_buffer_ms,
    _resolve_tts_stop_drain_guard_ms,
    send_tts_message,
    sendAudioMessage,
)
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType


class _DummyTTSProvider(TTSProviderBase):
    async def text_to_speak(self, text, output_file):
        return b""


class _DummyWebSocket:
    def __init__(self):
        self.messages = []

    async def send(self, payload):
        self.messages.append(payload)


class SendAudioTimingTest(unittest.TestCase):
    def test_stop_buffer_keeps_protocol_floor_even_when_config_is_lower(self):
        conn = SimpleNamespace(
            config={
                "tts_stop_extra_buffer_ms": 0,
                "tts_stop_min_buffer_ms": 240,
            }
        )

        requested, minimum, protocol_floor, effective = _resolve_tts_stop_buffer_ms(
            conn, 60
        )

        self.assertEqual(0, requested)
        self.assertEqual(240, minimum)
        self.assertEqual(420, protocol_floor)
        self.assertEqual(420, effective)

    def test_requested_extra_above_protocol_floor_is_preserved(self):
        conn = SimpleNamespace(
            config={
                "tts_stop_extra_buffer_ms": 480,
                "tts_stop_min_buffer_ms": 240,
            }
        )

        requested, minimum, protocol_floor, effective = _resolve_tts_stop_buffer_ms(
            conn, 60
        )

        self.assertEqual(480, requested)
        self.assertEqual(240, minimum)
        self.assertEqual(420, protocol_floor)
        self.assertEqual(480, effective)

    def test_stop_drain_guard_defaults_to_six_hundred_ms(self):
        conn = SimpleNamespace(config={})

        guard_ms = _resolve_tts_stop_drain_guard_ms(conn)

        self.assertEqual(600, guard_ms)

    def test_estimate_sent_audio_remaining_ms_uses_elapsed_playback_time(self):
        flow_control = {
            "packet_count": 4,
            "frame_duration_ms": 60,
            "first_send_monotonic": 100.0,
        }

        remaining_ms = _estimate_sent_audio_remaining_ms(
            flow_control,
            now_monotonic=100.12,
        )

        self.assertAlmostEqual(120.0, remaining_ms, delta=1.0)

    def test_sentence_bridge_delay_waits_for_inflight_audio_when_queue_is_empty(self):
        conn = SimpleNamespace(
            sentence_id="turn-bridge-1",
            config={"tts_sentence_bridge_guard_ms": 0},
            audio_rate_controller=SimpleNamespace(queue=[], frame_duration=60),
            audio_flow_control={
                "sentence_id": "turn-bridge-1",
                "packet_count": 4,
                "frame_duration_ms": 60,
                "first_send_monotonic": 100.0,
            },
        )

        with patch("core.handle.sendAudioHandle.time.monotonic", return_value=100.0):
            delay_ms = _resolve_tts_sentence_bridge_delay_ms(conn)

        self.assertEqual(240, delay_ms)

    def test_sentence_bridge_delay_is_skipped_when_queue_still_has_backlog(self):
        conn = SimpleNamespace(
            sentence_id="turn-bridge-2",
            config={"tts_sentence_bridge_guard_ms": 0},
            audio_rate_controller=SimpleNamespace(
                queue=[("audio", b"one")],
                frame_duration=60,
            ),
            audio_flow_control={
                "sentence_id": "turn-bridge-2",
                "packet_count": 4,
                "frame_duration_ms": 60,
                "first_send_monotonic": time.monotonic(),
            },
        )

        delay_ms = _resolve_tts_sentence_bridge_delay_ms(conn)

        self.assertEqual(0, delay_ms)

    def test_stream_split_holds_terminal_sentence_until_more_context_arrives(self):
        provider = _DummyTTSProvider(
            {"disable_pre_speak_split": True},
            delete_audio_file=True,
        )
        provider.conn = SimpleNamespace(config={})
        provider.tts_text_buff = ["今天我们做《银纳米粒子实验》。"]
        provider.processed_chars = 0
        provider.tts_stop_request = False
        provider.is_first_sentence = True

        segment = provider._get_segment_text()

        self.assertIsNone(segment)
        self.assertEqual(0, provider.processed_chars)

    def test_stream_split_holds_short_followup_after_sentence_boundary(self):
        provider = _DummyTTSProvider(
            {"disable_pre_speak_split": True},
            delete_audio_file=True,
        )
        provider.conn = SimpleNamespace(config={"tts_stream_followup_hold_chars": 8})
        provider.tts_text_buff = ["今天我们做《银纳米粒子实验》。你准备好"]
        provider.processed_chars = 0
        provider.tts_stop_request = False
        provider.is_first_sentence = True

        segment = provider._get_segment_text()

        self.assertIsNone(segment)
        self.assertEqual(0, provider.processed_chars)

    def test_stream_split_flushes_when_followup_is_long_enough(self):
        provider = _DummyTTSProvider(
            {"disable_pre_speak_split": True},
            delete_audio_file=True,
        )
        provider.conn = SimpleNamespace(config={"tts_stream_followup_hold_chars": 6})
        provider.tts_text_buff = ["第一句。后面已经有足够长的补充说明文字"]
        provider.processed_chars = 0
        provider.tts_stop_request = False
        provider.is_first_sentence = True

        segment = provider._get_segment_text()

        self.assertEqual("第一句", segment)
        self.assertEqual(len("第一句。"), provider.processed_chars)

    def test_send_tts_message_start_finishes_deferred_thinking(self):
        called = {"count": 0}
        ws = _DummyWebSocket()
        conn = SimpleNamespace(
            session_id="sess-1",
            websocket=ws,
            config={},
            logger=_DummyLogger(),
            _finish_deferred_thinking_on_tts_start=lambda: called.__setitem__(
                "count", called["count"] + 1
            ),
        )

        asyncio.run(send_tts_message(conn, "start"))

        self.assertEqual(1, called["count"])
        self.assertEqual(1, len(ws.messages))
        payload = json.loads(ws.messages[0])
        self.assertEqual("tts", payload["type"])
        self.assertEqual("start", payload["state"])

    def test_send_tts_message_stop_also_finishes_deferred_thinking(self):
        called = {"count": 0}
        ws = _DummyWebSocket()
        conn = SimpleNamespace(
            session_id="sess-2",
            websocket=ws,
            config={"enable_stop_tts_notify": False},
            logger=_DummyLogger(),
            _finish_deferred_thinking_on_tts_start=lambda: called.__setitem__(
                "count", called["count"] + 1
            ),
            clearSpeakStatus=lambda: None,
            has_external_busy=lambda: False,
        )

        asyncio.run(send_tts_message(conn, "stop"))

        self.assertEqual(1, called["count"])
        self.assertEqual(1, len(ws.messages))
        payload = json.loads(ws.messages[0])
        self.assertEqual("stop", payload["state"])

    def test_first_audio_message_repeats_text_on_start_for_clients_without_sentence_events(self):
        ws = _DummyWebSocket()
        conn = SimpleNamespace(
            session_id="sess-3",
            sentence_id="turn-1",
            websocket=ws,
            config={},
            logger=_DummyLogger(),
            tts=SimpleNamespace(tts_audio_first_sentence=True),
            client_is_speaking=False,
            close_after_chat=False,
        )

        asyncio.run(
            sendAudioMessage(
                conn,
                SentenceType.FIRST,
                [],
                "接下来做这一步",
                sentence_id="turn-1",
            )
        )

        self.assertEqual(2, len(ws.messages))
        start_payload = json.loads(ws.messages[0])
        sentence_payload = json.loads(ws.messages[1])
        self.assertEqual("start", start_payload["state"])
        self.assertEqual("接下来做这一步", start_payload["text"])
        self.assertEqual("sentence_start", sentence_payload["state"])
        self.assertEqual("接下来做这一步", sentence_payload["text"])


    def test_first_audio_message_waits_for_inflight_audio_before_next_sentence(self):
        ws = _DummyWebSocket()
        sleep_calls = []

        async def _fake_sleep(delay_seconds):
            sleep_calls.append(delay_seconds)

        conn = SimpleNamespace(
            session_id="sess-3b",
            sentence_id="turn-bridge-3",
            websocket=ws,
            config={"tts_sentence_bridge_guard_ms": 0},
            logger=_DummyLogger(),
            tts=SimpleNamespace(tts_audio_first_sentence=False),
            client_is_speaking=True,
            close_after_chat=False,
            audio_rate_controller=SimpleNamespace(queue=[], frame_duration=60),
            audio_flow_control={
                "sentence_id": "turn-bridge-3",
                "packet_count": 4,
                "frame_duration_ms": 60,
                "first_send_monotonic": 100.0,
            },
        )

        with patch("core.handle.sendAudioHandle.time.monotonic", return_value=100.0):
            with patch("core.handle.sendAudioHandle.asyncio.sleep", new=_fake_sleep):
                asyncio.run(
                    sendAudioMessage(
                        conn,
                        SentenceType.FIRST,
                        [],
                        "bridged subtitle",
                        sentence_id="turn-bridge-3",
                    )
                )

        self.assertEqual([0.24], sleep_calls)
        self.assertEqual(1, len(ws.messages))
        sentence_payload = json.loads(ws.messages[0])
        self.assertEqual("sentence_start", sentence_payload["state"])
        self.assertEqual("bridged subtitle", sentence_payload["text"])

    def test_middle_audio_message_with_text_still_emits_sentence_start(self):
        ws = _DummyWebSocket()
        conn = SimpleNamespace(
            session_id="sess-4",
            sentence_id="turn-2",
            websocket=ws,
            config={},
            logger=_DummyLogger(),
            tts=SimpleNamespace(tts_audio_first_sentence=True),
            client_is_speaking=False,
            close_after_chat=False,
        )

        asyncio.run(
            sendAudioMessage(
                conn,
                SentenceType.MIDDLE,
                [],
                "fallback subtitle for middle audio",
                sentence_id="turn-2",
            )
        )

        self.assertEqual(2, len(ws.messages))
        start_payload = json.loads(ws.messages[0])
        sentence_payload = json.loads(ws.messages[1])
        self.assertEqual("start", start_payload["state"])
        self.assertNotIn("text", start_payload)
        self.assertEqual("sentence_start", sentence_payload["state"])
        self.assertEqual(
            "fallback subtitle for middle audio", sentence_payload["text"]
        )

    def test_first_audio_message_refreshes_start_when_speaking_state_is_reused(self):
        ws = _DummyWebSocket()
        conn = SimpleNamespace(
            session_id="sess-4b",
            sentence_id="turn-2b",
            websocket=ws,
            config={},
            logger=_DummyLogger(),
            tts=SimpleNamespace(tts_audio_first_sentence=True),
            client_is_speaking=True,
            close_after_chat=False,
        )

        asyncio.run(
            sendAudioMessage(
                conn,
                SentenceType.FIRST,
                [],
                "新的语音字幕",
                sentence_id="turn-2b",
            )
        )

        self.assertEqual(2, len(ws.messages))
        start_payload = json.loads(ws.messages[0])
        sentence_payload = json.loads(ws.messages[1])
        self.assertEqual("start", start_payload["state"])
        self.assertEqual("新的语音字幕", start_payload["text"])
        self.assertEqual("sentence_start", sentence_payload["state"])
        self.assertEqual("新的语音字幕", sentence_payload["text"])

    def test_last_audio_message_with_text_still_emits_sentence_start_before_stop(self):
        ws = _DummyWebSocket()
        conn = SimpleNamespace(
            session_id="sess-5",
            sentence_id="turn-3",
            websocket=ws,
            config={"enable_stop_tts_notify": False},
            logger=_DummyLogger(),
            tts=SimpleNamespace(tts_audio_first_sentence=False),
            client_is_speaking=True,
            close_after_chat=False,
            clearSpeakStatus=lambda: None,
            has_external_busy=lambda: False,
        )

        asyncio.run(
            sendAudioMessage(
                conn,
                SentenceType.LAST,
                [],
                "fallback subtitle for last audio",
                sentence_id="turn-3",
            )
        )

        self.assertEqual(2, len(ws.messages))
        sentence_payload = json.loads(ws.messages[0])
        stop_payload = json.loads(ws.messages[1])
        self.assertEqual("sentence_start", sentence_payload["state"])
        self.assertEqual("fallback subtitle for last audio", sentence_payload["text"])
        self.assertEqual("stop", stop_payload["state"])


if __name__ == "__main__":
    unittest.main()
