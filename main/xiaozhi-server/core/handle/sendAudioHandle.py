import json
import time
import asyncio
from core.utils import textUtils
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.audioRateController import AudioRateController

TAG = __name__
# 音频帧时长（毫秒）
AUDIO_FRAME_DURATION = 60
# 预缓冲包数量，直接发送以减少延迟
PRE_BUFFER_COUNT = 5


def _resolve_tts_stop_protocol_floor_ms(frame_duration_ms):
    return max((PRE_BUFFER_COUNT + 2) * frame_duration_ms, 360)


def _resolve_tts_stop_drain_guard_ms(conn):
    raw_guard_ms = conn.config.get("tts_stop_drain_guard_ms", 600)
    try:
        return max(0, int(raw_guard_ms))
    except (TypeError, ValueError):
        return 600


def _resolve_tts_stop_buffer_ms(conn, frame_duration_ms):
    protocol_floor_ms = _resolve_tts_stop_protocol_floor_ms(frame_duration_ms)

    raw_extra_ms = conn.config.get("tts_stop_extra_buffer_ms", protocol_floor_ms)
    try:
        requested_extra_ms = max(0, int(raw_extra_ms))
    except (TypeError, ValueError):
        requested_extra_ms = protocol_floor_ms

    raw_min_buffer_ms = conn.config.get("tts_stop_min_buffer_ms", 240)
    try:
        min_buffer_ms = max(0, int(raw_min_buffer_ms))
    except (TypeError, ValueError):
        min_buffer_ms = 240

    effective_extra_ms = max(requested_extra_ms, min_buffer_ms, protocol_floor_ms)
    return requested_extra_ms, min_buffer_ms, protocol_floor_ms, effective_extra_ms


def _estimate_sent_audio_remaining_ms(flow_control, *, now_monotonic=None):
    current_flow_control = flow_control or {}
    packet_count = int(current_flow_control.get("packet_count", 0) or 0)
    first_send_monotonic = current_flow_control.get("first_send_monotonic")
    if first_send_monotonic is None or packet_count <= 0:
        return 0

    frame_duration_ms = int(
        current_flow_control.get("frame_duration_ms", AUDIO_FRAME_DURATION)
        or AUDIO_FRAME_DURATION
    )
    if frame_duration_ms <= 0:
        frame_duration_ms = AUDIO_FRAME_DURATION

    current_monotonic = (
        time.monotonic() if now_monotonic is None else float(now_monotonic)
    )
    elapsed_ms = max((current_monotonic - first_send_monotonic) * 1000, 0)
    scheduled_audio_ms = packet_count * frame_duration_ms
    return max(scheduled_audio_ms - elapsed_ms, 0)


def _resolve_tts_sentence_bridge_guard_ms(conn, frame_duration_ms):
    default_guard_ms = max(int(frame_duration_ms or AUDIO_FRAME_DURATION), 40)
    raw_guard_ms = conn.config.get(
        "tts_sentence_bridge_guard_ms",
        default_guard_ms,
    )
    try:
        return max(0, int(raw_guard_ms))
    except (TypeError, ValueError):
        return default_guard_ms


def _resolve_tts_sentence_bridge_delay_ms(conn):
    rate_controller = getattr(conn, "audio_rate_controller", None)
    if rate_controller is None:
        return 0

    flow_control = getattr(conn, "audio_flow_control", {}) or {}
    active_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    flow_sentence_id = str(flow_control.get("sentence_id", "") or "").strip()
    if not active_sentence_id or flow_sentence_id != active_sentence_id:
        return 0

    try:
        queued_items = len(getattr(rate_controller, "queue", ()) or ())
    except Exception:
        queued_items = 1
    if queued_items > 0:
        return 0

    frame_duration_ms = int(
        flow_control.get(
            "frame_duration_ms",
            getattr(rate_controller, "frame_duration", AUDIO_FRAME_DURATION),
        )
        or AUDIO_FRAME_DURATION
    )
    remaining_ms = _estimate_sent_audio_remaining_ms(flow_control)
    if remaining_ms <= 0:
        return 0

    guard_ms = _resolve_tts_sentence_bridge_guard_ms(conn, frame_duration_ms)
    return int(round(remaining_ms + guard_ms))


def _get_open_websocket(conn):
    ws = getattr(conn, "websocket", None)
    if ws is None:
        return None
    try:
        if hasattr(ws, "closed") and ws.closed:
            return None
        if hasattr(ws, "state") and ws.state.name == "CLOSED":
            return None
    except Exception:
        return None
    return ws


def _get_force_independent_tts_sentence_ids(conn):
    sentence_ids = getattr(conn, "_force_independent_tts_sentence_ids", None)
    if sentence_ids is None:
        sentence_ids = set()
        setattr(conn, "_force_independent_tts_sentence_ids", sentence_ids)
    return sentence_ids


def _should_force_independent_tts_cycle(conn, sentence_id=None):
    active_sentence_id = str(sentence_id or "").strip()
    if not active_sentence_id:
        return False
    return active_sentence_id in _get_force_independent_tts_sentence_ids(conn)


def _clear_forced_tts_cycle(conn, sentence_id=None):
    active_sentence_id = str(sentence_id or "").strip()
    if not active_sentence_id:
        return
    _get_force_independent_tts_sentence_ids(conn).discard(active_sentence_id)


def _queue_has_pending_followup_tts(conn, sentence_id=None):
    tts = getattr(conn, "tts", None)
    if not tts:
        return False

    has_queued_followup = False
    try:
        text_queue = getattr(tts, "tts_text_queue", None)
        if text_queue is not None:
            with text_queue.mutex:
                if len(text_queue.queue) > 0:
                    has_queued_followup = True
    except Exception:
        has_queued_followup = False

    try:
        audio_queue = getattr(tts, "tts_audio_queue", None)
        if audio_queue is not None:
            with audio_queue.mutex:
                if len(audio_queue.queue) > 0:
                    has_queued_followup = True
    except Exception:
        pass

    if has_queued_followup:
        return True

    try:
        has_inflight_processing = getattr(
            tts,
            "has_inflight_tts_text_processing",
            None,
        )
        if not callable(has_inflight_processing) or not has_inflight_processing():
            return False

        active_sentence_id = str(sentence_id or "").strip()
        inflight_sentence_id = str(
            getattr(tts, "_current_audio_sentence_id", "") or ""
        ).strip()
        # Ignore the provider's current LAST task when it is just finishing its
        # own turn. Otherwise the final audio item can race ahead of the worker
        # thread teardown and we would suppress the only stop event forever.
        return bool(inflight_sentence_id and inflight_sentence_id != active_sentence_id)
    except Exception:
        return False

    return False


async def sendAudioMessage(conn, sentenceType, audios, text, sentence_id=None):
    active_sentence_id = str(sentence_id or getattr(conn, "sentence_id", "") or "").strip()
    force_independent_cycle = _should_force_independent_tts_cycle(
        conn, active_sentence_id
    )
    sentence_bridge_delay_ms = _resolve_tts_sentence_bridge_delay_ms(conn)
    if sentence_bridge_delay_ms > 0:
        conn.logger.bind(tag=TAG).info(
            "tts sentence bridge wait: "
            f"delay_ms={sentence_bridge_delay_ms}, "
            f"sentence_id={active_sentence_id or 'unknown'}"
        )
        await asyncio.sleep(sentence_bridge_delay_ms / 1000.0)
    if conn.tts.tts_audio_first_sentence:
        conn.logger.bind(tag=TAG).info(f"发送第一段语音: {text}")
        conn.tts.tts_audio_first_sentence = False
        if force_independent_cycle and conn.client_is_speaking:
            conn.logger.bind(tag=TAG).info(
                "force fresh tts start for system prompt: "
                f"sentence_id={active_sentence_id or 'unknown'}"
            )
            conn.clearSpeakStatus()
        should_emit_start = (
            force_independent_cycle
            or not conn.client_is_speaking
            or sentenceType == SentenceType.FIRST
        )
        if should_emit_start:
            # Some devices only refresh their subtitle/status overlay on a
            # fresh TTS start transition. Re-emit start for a new FIRST
            # sentence even if we intentionally reused the speaking state
            # across queued follow-up speech.
            # Some clients only refresh their on-screen subtitle region on the
            # initial speaking transition, so include the first sentence text
            # there as a compatibility fallback in addition to sentence_start.
            start_text = text if sentenceType == SentenceType.FIRST else None
            await send_tts_message(conn, "start", start_text)
            conn.client_is_speaking = True
            conn.logger.bind(tag=TAG).info(
                "tts speaking state entered: "
                f"sentence_id={active_sentence_id or 'unknown'}, "
                f"force_independent={force_independent_cycle}"
            )
        else:
            conn.logger.bind(tag=TAG).debug(
                f"reuse active speaking state for sentence_id={active_sentence_id or 'unknown'}"
            )

    if text is not None and sentenceType == SentenceType.FIRST:
        # 同一句子的后续消息加入流控队列，其他情况立即发送
        if (
            hasattr(conn, "audio_rate_controller")
            and conn.audio_rate_controller
            and getattr(conn, "audio_flow_control", {}).get("sentence_id")
            == conn.sentence_id
            and sentence_bridge_delay_ms <= 0
        ):
            conn.audio_rate_controller.add_message(
                lambda: send_tts_message(conn, "sentence_start", text)
            )
        else:
            # 新句子或流控器未初始化，立即发送
            await send_tts_message(conn, "sentence_start", text)
    elif text is not None and sentenceType in (SentenceType.MIDDLE, SentenceType.LAST):
        # File-based or legacy TTS paths sometimes attach subtitle text to a
        # later audio packet instead of the FIRST packet. Emit the subtitle
        # event here as a compatibility fallback so the device still renders
        # the spoken text on screen.
        await send_tts_message(conn, "sentence_start", text)

    await sendAudio(conn, audios)
    # 发送句子开始消息
    if sentenceType is not SentenceType.MIDDLE:
        conn.logger.bind(tag=TAG).info(f"发送音频消息: {sentenceType}, {text}")

    # 发送结束消息（如果是最后一个文本）
    if sentenceType == SentenceType.LAST:
        if not force_independent_cycle and _queue_has_pending_followup_tts(
            conn, active_sentence_id
        ):
            conn.logger.bind(tag=TAG).info(
                f"skip tts stop because follow-up speech is pending: sentence_id={active_sentence_id or 'unknown'}"
            )
            return
        await send_tts_message(conn, "stop", None)
        conn.client_is_speaking = False
        _clear_forced_tts_cycle(conn, active_sentence_id)
        if conn.close_after_chat:
            await conn.close()


async def _wait_for_audio_completion(conn):
    """
    等待音频队列清空，并根据已发送包时长和设备侧缓冲决定 stop 发送前的等待时间。

    Args:
        conn: 连接对象
    """
    if not hasattr(conn, "audio_rate_controller") or not conn.audio_rate_controller:
        return

    rate_controller = conn.audio_rate_controller
    conn.logger.bind(tag=TAG).debug(
        f"waiting audio completion, queued_packets={len(rate_controller.queue)}"
    )
    await rate_controller.queue_empty_event.wait()

    flow_control = getattr(conn, "audio_flow_control", {}) or {}
    frame_duration_ms = int(
        flow_control.get("frame_duration_ms", rate_controller.frame_duration)
    )
    packet_count = int(flow_control.get("packet_count", 0))
    remaining_ms = _estimate_sent_audio_remaining_ms(
        flow_control,
        now_monotonic=time.monotonic(),
    )

    requested_extra_ms, min_buffer_ms, protocol_floor_ms, effective_extra_ms = (
        _resolve_tts_stop_buffer_ms(conn, frame_duration_ms)
    )
    drain_guard_ms = _resolve_tts_stop_drain_guard_ms(conn)
    total_wait_ms = remaining_ms + effective_extra_ms + drain_guard_ms
    conn.logger.bind(tag=TAG).info(
        "tts stop wait: "
        f"packet_count={packet_count}, "
        f"frame_duration_ms={frame_duration_ms}, "
        f"remaining_ms={remaining_ms:.0f}, "
        f"requested_extra_ms={requested_extra_ms}, "
        f"min_buffer_ms={min_buffer_ms}, "
        f"protocol_floor_ms={protocol_floor_ms}, "
        f"effective_extra_ms={effective_extra_ms}, "
        f"drain_guard_ms={drain_guard_ms}, "
        f"total_wait_ms={total_wait_ms:.0f}"
    )
    if total_wait_ms > 0:
        await asyncio.sleep(total_wait_ms / 1000.0)

    conn.logger.bind(tag=TAG).debug("audio completion wait finished")


async def _send_to_mqtt_gateway(conn, opus_packet, timestamp, sequence):
    """
    发送带16字节头部的opus数据包给mqtt_gateway
    Args:
        conn: 连接对象
        opus_packet: opus数据包
        timestamp: 时间戳
        sequence: 序列号
    """
    # 为opus数据包添加16字节头部
    header = bytearray(16)
    header[0] = 1  # type
    header[2:4] = len(opus_packet).to_bytes(2, "big")  # payload length
    header[4:8] = sequence.to_bytes(4, "big")  # sequence
    header[8:12] = timestamp.to_bytes(4, "big")  # 时间戳
    header[12:16] = len(opus_packet).to_bytes(4, "big")  # opus长度

    # 发送包含头部的完整数据包
    complete_packet = bytes(header) + opus_packet
    ws = _get_open_websocket(conn)
    if ws is None:
        conn.logger.bind(tag=TAG).warning("WebSocket unavailable, skip mqtt audio send")
        return
    await ws.send(complete_packet)


async def sendAudio(conn, audios, frame_duration=AUDIO_FRAME_DURATION):
    """
    发送音频包，使用 AudioRateController 进行精确的流量控制

    Args:
        conn: 连接对象
        audios: 单个opus包(bytes) 或 opus包列表
        frame_duration: 帧时长（毫秒），默认使用全局常量AUDIO_FRAME_DURATION
    """
    if audios is None or len(audios) == 0:
        return

    send_delay = conn.config.get("tts_audio_send_delay", -1) / 1000.0
    is_single_packet = isinstance(audios, bytes)

    # 初始化或获取 RateController
    rate_controller, flow_control = _get_or_create_rate_controller(
        conn, frame_duration, is_single_packet
    )

    # 统一转换为列表处理
    audio_list = [audios] if is_single_packet else audios

    # 发送音频包
    await _send_audio_with_rate_control(
        conn, audio_list, rate_controller, flow_control, send_delay
    )


def _get_or_create_rate_controller(conn, frame_duration, is_single_packet):
    """
    获取或创建 RateController 和 flow_control

    Args:
        conn: 连接对象
        frame_duration: 帧时长
        is_single_packet: 是否单包模式（True: TTS流式单包, False: 批量包）

    Returns:
        (rate_controller, flow_control)
    """
    # 检查是否需要重置控制器
    need_reset = False

    if not hasattr(conn, "audio_rate_controller"):
        # 控制器不存在，需要创建
        need_reset = True
    else:
        rate_controller = conn.audio_rate_controller

        # 后台发送任务已停止, 则需要重置
        if (
            not rate_controller.pending_send_task
            or rate_controller.pending_send_task.done()
        ):
            need_reset = True
        # 当sentence_id 变化，需要重置
        elif (
            getattr(conn, "audio_flow_control", {}).get("sentence_id")
            != conn.sentence_id
        ):
            need_reset = True

    if need_reset:
        # 创建或获取 rate_controller
        if not hasattr(conn, "audio_rate_controller"):
            conn.audio_rate_controller = AudioRateController(frame_duration)
        else:
            conn.audio_rate_controller.reset()

        # 初始化 flow_control
        conn.audio_flow_control = {
            "packet_count": 0,
            "sequence": 0,
            "sentence_id": conn.sentence_id,
            "frame_duration_ms": frame_duration,
            "first_send_monotonic": None,
            "last_send_monotonic": None,
        }

        # 启动后台发送循环
        _start_background_sender(
            conn, conn.audio_rate_controller, conn.audio_flow_control
        )

    return conn.audio_rate_controller, conn.audio_flow_control


def _start_background_sender(conn, rate_controller, flow_control):
    """
    启动后台发送循环任务

    Args:
        conn: 连接对象
        rate_controller: 速率控制器
        flow_control: 流控状态
    """

    async def send_callback(packet):
        # 检查是否应该中止
        if conn.client_abort:
            raise asyncio.CancelledError("客户端已中止")

        conn.last_activity_time = time.time() * 1000
        await _do_send_audio(conn, packet, flow_control)
        conn.client_is_speaking = True

    # 使用 start_sending 启动后台循环
    rate_controller.start_sending(send_callback)


async def _send_audio_with_rate_control(
    conn, audio_list, rate_controller, flow_control, send_delay
):
    """
    使用 rate_controller 发送音频包

    Args:
        conn: 连接对象
        audio_list: 音频包列表
        rate_controller: 速率控制器
        flow_control: 流控状态
        send_delay: 固定延迟（秒），-1表示使用动态流控
    """
    for packet in audio_list:
        if conn.client_abort:
            return

        conn.last_activity_time = time.time() * 1000

        # 预缓冲：前N个包直接发送
        if flow_control["packet_count"] < PRE_BUFFER_COUNT:
            await _do_send_audio(conn, packet, flow_control)
            conn.client_is_speaking = True
        elif send_delay > 0:
            # 固定延迟模式
            await asyncio.sleep(send_delay)
            await _do_send_audio(conn, packet, flow_control)
            conn.client_is_speaking = True
        else:
            # 动态流控模式：仅添加到队列，由后台循环负责发送
            rate_controller.add_audio(packet)


async def _do_send_audio(conn, opus_packet, flow_control):
    """
    执行实际的音频发送
    """
    ws = _get_open_websocket(conn)
    if ws is None:
        conn.logger.bind(tag=TAG).warning("WebSocket unavailable, skip audio packet")
        return

    packet_index = flow_control.get("packet_count", 0)
    sequence = flow_control.get("sequence", 0)
    now_monotonic = time.monotonic()

    if flow_control.get("first_send_monotonic") is None:
        flow_control["first_send_monotonic"] = now_monotonic
    flow_control["last_send_monotonic"] = now_monotonic

    if conn.conn_from_mqtt_gateway:
        # 计算时间戳（基于播放位置）
        start_time = time.time()
        timestamp = int(start_time * 1000) % (2**32)
        await _send_to_mqtt_gateway(conn, opus_packet, timestamp, sequence)
    else:
        # 直接发送opus数据包
        await ws.send(opus_packet)

    # 更新流控状态
    flow_control["packet_count"] = packet_index + 1
    flow_control["sequence"] = sequence + 1


async def send_tts_message(conn, state, text=None):
    """发送 TTS 状态消息"""
    if text is None and state == "sentence_start":
        return
    if state == "start" and hasattr(conn, "_finish_deferred_thinking_on_tts_start"):
        try:
            conn._finish_deferred_thinking_on_tts_start()
        except Exception as exc:
            conn.logger.bind(tag=TAG).debug(
                f"finish deferred thinking on tts start failed: {exc}"
            )
    ws = _get_open_websocket(conn)
    message = {"type": "tts", "state": state, "session_id": conn.session_id}
    if text is not None:
        message["text"] = textUtils.check_emoji(text)

    # TTS播放结束
    if state == "stop":
        if hasattr(conn, "_finish_deferred_thinking_on_tts_start"):
            try:
                conn._finish_deferred_thinking_on_tts_start()
            except Exception as exc:
                conn.logger.bind(tag=TAG).debug(
                    f"finish deferred thinking on tts stop failed: {exc}"
                )
        # 播放提示音
        tts_notify = conn.config.get("enable_stop_tts_notify", False)
        if tts_notify:
            stop_tts_notify_voice = conn.config.get(
                "stop_tts_notify_voice", "config/assets/tts_notify.mp3"
            )
            audios = await audio_to_data(stop_tts_notify_voice, is_opus=True)
            await sendAudio(conn, audios)
        # 等待所有音频包发送完成
        await _wait_for_audio_completion(conn)
        # 清除服务端讲话状态
        conn.clearSpeakStatus()
        if conn.has_external_busy():
            conn._send_llm_event_message("[Thinking]", event="thinking", phase="start")
            conn._start_thinking_pulse()

    # 发送消息到客户端
    if ws is None:
        conn.logger.bind(tag=TAG).warning(
            f"WebSocket unavailable, skip tts state message: {state}"
        )
        return
    await ws.send(json.dumps(message))


async def send_stt_message(conn, text):
    """发送 STT 状态消息"""
    end_prompt_str = conn.config.get("end_prompt", {}).get("prompt")
    if end_prompt_str and end_prompt_str == text:
        await send_tts_message(conn, "start")
        return

    ws = _get_open_websocket(conn)
    if ws is None:
        conn.logger.bind(tag=TAG).warning("WebSocket unavailable, skip stt message")
        return

    # 解析JSON格式，提取实际的用户说话内容
    display_text = text
    try:
        # 尝试解析JSON格式
        if text.strip().startswith("{") and text.strip().endswith("}"):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                # 如果是包含说话人信息的JSON格式，只显示content部分
                display_text = parsed_data["content"]
                # 保存说话人信息到conn对象
                if "speaker" in parsed_data:
                    conn.current_speaker = parsed_data["speaker"]
    except (json.JSONDecodeError, TypeError):
        # 如果不是JSON格式，直接使用原始文本
        display_text = text
    stt_text = textUtils.get_string_no_punctuation_or_emoji(display_text)
    await ws.send(
        json.dumps({"type": "stt", "text": stt_text, "session_id": conn.session_id})
    )
    await send_tts_message(conn, "start")
