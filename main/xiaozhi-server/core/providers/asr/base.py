import os
import io
import wave
import uuid
import json
import time
import queue
import asyncio
import traceback
import threading
import opuslib_next
from abc import ABC, abstractmethod
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.handle.receiveAudioHandle import startToChat
from core.handle.reportHandle import enqueue_asr_report
from core.utils.experiment_resume import (
    build_resume_context,
    should_load_device_log_context,
)
from core.utils import textUtils
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import handleAudioMessage
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType

TAG = __name__
logger = setup_logging()
UNKNOWN_SPEAKER_NAME = "未知说话人"
UNKNOWN_SPEAKER_RETRY_PROMPT = "未知说话人，请重新说"
UNKNOWN_SPEAKER_STATUSES = {"unknown", "rejected"}


def _parse_nonnegative_int(value, default=0):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _selected_vad_config(conn) -> dict:
    config = getattr(conn, "config", {}) or {}
    selected = str((config.get("selected_module") or {}).get("VAD", "") or "").strip()
    vad_config = config.get("VAD") or {}
    if selected and isinstance(vad_config.get(selected), dict):
        return vad_config.get(selected) or {}
    return {}


def _tail_merge_window_ms(conn) -> int:
    return _parse_nonnegative_int(
        _selected_vad_config(conn).get("tail_merge_window_ms"),
        0,
    )


def _min_asr_audio_packets(conn) -> int:
    config = getattr(conn, "config", {}) or {}
    return max(
        1,
        _parse_nonnegative_int(
            config.get("asr_min_audio_packets"),
            15,
        ),
    )


class ASRProviderBase(ABC):
    def __init__(self):
        pass

    # 打开音频通道
    async def open_audio_channels(self, conn):
        conn.asr_priority_thread = threading.Thread(
            target=self.asr_text_priority_thread, args=(conn,), daemon=True
        )
        conn.asr_priority_thread.start()

    # 有序处理 ASR 音频
    def asr_text_priority_thread(self, conn):
        while not conn.stop_event.is_set():
            try:
                message = conn.asr_audio_queue.get(timeout=1)
                future = asyncio.run_coroutine_threadsafe(
                    handleAudioMessage(conn, message),
                    conn.loop,
                )
                future.result()
            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理 ASR 文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )
                continue

    # 接收音频
    async def receive_audio(self, conn, audio, audio_have_voice):
        # Processed PCM (decoded + frontend). Provided by receiveAudioHandle for each packet.
        pcm_packet = getattr(conn, "_pcm_packet_for_asr", None)
        conn._pcm_packet_for_asr = None

        if conn.client_listen_mode == "manual":
            # 手动模式：缓存音频用于 ASR 识别
            conn.asr_audio.append(audio)
            if pcm_packet is not None and hasattr(conn, "asr_pcm_audio"):
                conn.asr_pcm_audio.append(pcm_packet)
        else:
            # 自动/实时模式：使用 VAD 检测
            have_voice = audio_have_voice

            conn.asr_audio.append(audio)
            if pcm_packet is not None and hasattr(conn, "asr_pcm_audio"):
                conn.asr_pcm_audio.append(pcm_packet)
            if not have_voice and not conn.client_have_voice:
                conn.asr_audio = conn.asr_audio[-10:]
                if hasattr(conn, "asr_pcm_audio"):
                    conn.asr_pcm_audio = conn.asr_pcm_audio[-10:]
                return

            # 自动模式下通过 VAD 检测到语音停止时触发识别
            if conn.client_voice_stop:
                if have_voice:
                    conn.client_voice_stop = False
                    conn._asr_voice_stop_deadline_ms = 0.0
                    return

                tail_merge_ms = _tail_merge_window_ms(conn)
                if tail_merge_ms > 0:
                    now_ms = time.time() * 1000
                    deadline_ms = float(
                        getattr(conn, "_asr_voice_stop_deadline_ms", 0.0) or 0.0
                    )
                    if deadline_ms <= 0.0:
                        conn._asr_voice_stop_deadline_ms = now_ms + tail_merge_ms
                        return
                    if now_ms < deadline_ms:
                        return

                asr_audio_task = conn.asr_audio.copy()
                pcm_audio_task = (
                    conn.asr_pcm_audio.copy() if hasattr(conn, "asr_pcm_audio") else None
                )
                conn.asr_audio.clear()
                if hasattr(conn, "asr_pcm_audio"):
                    conn.asr_pcm_audio.clear()
                conn._asr_voice_stop_deadline_ms = 0.0
                conn.reset_vad_states()

                if len(asr_audio_task) >= _min_asr_audio_packets(conn):
                    await self.handle_voice_stop(conn, asr_audio_task, pcm_audio_task)

    # 处理语音停止
    async def handle_voice_stop(
        self,
        conn,
        asr_audio_task: List[bytes],
        pcm_audio_task: Optional[List[bytes]] = None,
    ):
        """并行处理 ASR 与声纹，并在进入 startToChat 前做鉴权闸门。"""
        try:
            total_start_time = time.monotonic()

            # 准备音频数据
            use_pcm_task = bool(pcm_audio_task) and len(pcm_audio_task) == len(asr_audio_task)
            if use_pcm_task:
                pcm_data = pcm_audio_task
                asr_input = pcm_audio_task
                asr_audio_format = "pcm"
            else:
                if conn.audio_format == "pcm":
                    pcm_data = asr_audio_task
                else:
                    pcm_data = self.decode_opus(asr_audio_task)
                asr_input = asr_audio_task
                asr_audio_format = conn.audio_format

            combined_pcm_data = b"".join(pcm_data)

            # Sentence-level frontend hook.
            # Only run it when we DON'T already have per-frame frontend-processed PCM from the realtime pipeline
            # (decode once -> process_frame -> cached as conn.asr_pcm_audio).
            if (
                not use_pcm_task
                and getattr(conn, "audio_frontend", None)
                and combined_pcm_data
            ):
                try:
                    combined_pcm_data = conn.audio_frontend.process_sentence(
                        combined_pcm_data
                    )
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"audio frontend sentence processing failed, bypass: {e}"
                    )

                # If we are sending PCM into ASR, pass the sentence-processed PCM to keep ASR/voiceprint consistent.
                if asr_audio_format == "pcm" and combined_pcm_data:
                    asr_input = [combined_pcm_data]

            # 预先准备 WAV 数据
            wav_data = None
            if conn.voiceprint_provider and combined_pcm_data:
                wav_data = self._pcm_to_wav(combined_pcm_data)

            # 定义 ASR 任务
            asr_task = self.speech_to_text(
                asr_input, conn.session_id, asr_audio_format
            )

            if conn.voiceprint_provider and wav_data:
                if hasattr(conn.voiceprint_provider, "evaluate_voiceprint"):
                    voiceprint_task = conn.voiceprint_provider.evaluate_voiceprint(
                        wav_data, conn.session_id
                    )
                else:
                    voiceprint_task = conn.voiceprint_provider.identify_speaker(
                        wav_data, conn.session_id
                    )

                # 并发等待两个结果
                asr_result, voiceprint_result = await asyncio.gather(
                    asr_task, voiceprint_task, return_exceptions=True
                )
            else:
                asr_result = await asr_task
                voiceprint_result = None

            # 记录 ASR 结果
            if isinstance(asr_result, Exception):
                logger.bind(tag=TAG).error(f"ASR 识别失败: {asr_result}")
                raw_text = ""
            else:
                raw_text, _ = asr_result

            log_context_query = self._extract_query_text(raw_text)
            should_try_log_resume = should_load_device_log_context(log_context_query)
            resume_context = None
            if should_try_log_resume and getattr(conn, "device_id", None):
                resume_context = build_resume_context(conn.config, conn.device_id)
                if resume_context:
                    logger.bind(tag=TAG).info(
                        "device log context loaded for recovery: "
                        f"device_id={conn.device_id}, log_path={resume_context.get('log_path')}"
                    )

            # 处理声纹结果（兼容旧字符串返回 + 新决策字典返回）
            voiceprint_blocked = False
            voiceprint_prompt_text = ""
            voiceprint_status = ""
            if isinstance(voiceprint_result, Exception):
                logger.bind(tag=TAG).error(f"声纹识别失败: {voiceprint_result}")
                speaker_name = ""
            elif isinstance(voiceprint_result, dict):
                speaker_name = (voiceprint_result.get("speaker_name") or "").strip()
                allow_chat = bool(voiceprint_result.get("allow_chat", True))
                status = (voiceprint_result.get("status") or "").strip().lower()
                voiceprint_status = status
                reason = voiceprint_result.get("reason", "")
                score = voiceprint_result.get("score")
                logger.bind(tag=TAG).info(
                    f"声纹决策: status={status}, allow_chat={allow_chat}, "
                    f"speaker={speaker_name}, score={score}, reason={reason}"
                )
                is_unknown_speaker = (
                    status in UNKNOWN_SPEAKER_STATUSES
                    or speaker_name == UNKNOWN_SPEAKER_NAME
                )
                if is_unknown_speaker:
                    voiceprint_blocked = True
                    voiceprint_prompt_text = UNKNOWN_SPEAKER_RETRY_PROMPT
                elif not allow_chat:
                    voiceprint_blocked = True
                    if voiceprint_result.get("need_register_prompt", False):
                        voiceprint_prompt_text = (
                            voiceprint_result.get("register_prompt_text") or ""
                        )
                    else:
                        voiceprint_prompt_text = (
                            voiceprint_result.get("reject_prompt_text") or ""
                        )
            else:
                speaker_name = voiceprint_result
                if (
                    isinstance(speaker_name, str)
                    and speaker_name.strip() == UNKNOWN_SPEAKER_NAME
                ):
                    voiceprint_blocked = True
                    voiceprint_prompt_text = UNKNOWN_SPEAKER_RETRY_PROMPT


            # 判断 ASR 结果类型
            if isinstance(raw_text, dict):
                # FunASR 返回的 dict 格式
                if speaker_name:
                    raw_text["speaker"] = speaker_name

                # 记录识别结果
                if raw_text.get("language"):
                    logger.bind(tag=TAG).info(f"识别语言: {raw_text['language']}")
                if raw_text.get("emotion"):
                    logger.bind(tag=TAG).info(f"识别情绪: {raw_text['emotion']}")
                if raw_text.get("content"):
                    logger.bind(tag=TAG).info(f"识别文本: {raw_text['content']}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}")

                # 转换为 JSON 字符串用于下游
                enhanced_text = json.dumps(raw_text, ensure_ascii=False)
                content_for_length_check = raw_text.get("content", "")
            else:
                # 其他 ASR 返回的纯文本
                if raw_text:
                    logger.bind(tag=TAG).info(f"识别文本: {raw_text}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}")

                # 构建包含说话人信息的 JSON 字符串
                enhanced_text = self._build_enhanced_text(raw_text, speaker_name)
                content_for_length_check = raw_text

            # 性能监控
            total_time = time.monotonic() - total_start_time
            logger.bind(tag=TAG).debug(f"总处理耗时: {total_time:.3f}s")

            # 检查文本长度
            text_len, _ = remove_punctuation_and_length(content_for_length_check)
            self.stop_ws_connection()

            # 声纹未通过：直接拦截，不进入 LLM
            if voiceprint_blocked:
                if voiceprint_prompt_text.strip():
                    self._enqueue_system_tts(conn, voiceprint_prompt_text.strip())
                return

            if text_len > 0:
                if hasattr(conn, "log_clean_user_utterance"):
                    conn.log_clean_user_utterance(
                        content_for_length_check,
                        source="asr",
                        speaker_name=speaker_name,
                        language_tag=(
                            raw_text.get("language", "")
                            if isinstance(raw_text, dict)
                            else ""
                        ),
                    )
                await startToChat(conn, enhanced_text)
                enqueue_asr_report(conn, enhanced_text, asr_audio_task)

        except Exception as e:
            logger.bind(tag=TAG).error(f"处理语音停止失败: {e}")
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")

    def _build_enhanced_text(self, text: str, speaker_name: Optional[str]) -> str:
        """构建包含说话人信息的文本（仅用于纯文本 ASR）。"""
        if speaker_name and speaker_name.strip():
            return json.dumps(
                {
                    "speaker": speaker_name,
                    "content": text,
                },
                ensure_ascii=False,
            )
        return text

    @staticmethod
    def _extract_query_text(raw_text) -> str:
        if isinstance(raw_text, dict):
            content = raw_text.get("content")
            if isinstance(content, str):
                return content.strip()
            if content is not None:
                return str(content).strip()
            return ""
        if isinstance(raw_text, str):
            return raw_text.strip()
        return str(raw_text or "").strip()

    def _enqueue_system_tts(self, conn, text: str):
        """直接下发系统 TTS，不走 LLM 流程。"""
        text = textUtils.prepare_runtime_spoken_text_for_conn(conn, text)
        if not text:
            return
        if not getattr(conn, "tts", None):
            logger.bind(tag=TAG).warning("TTS 未初始化，无法播放系统提示")
            return

        try:
            forced_sentence_ids = getattr(
                conn, "_force_independent_tts_sentence_ids", None
            )
            if forced_sentence_ids is None:
                forced_sentence_ids = set()
                conn._force_independent_tts_sentence_ids = forced_sentence_ids

            conn.sentence_id = str(uuid.uuid4().hex)
            forced_sentence_ids.add(conn.sentence_id)
            conn.tts_MessageText = text
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
            conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )
        except Exception as e:
            logger.bind(tag=TAG).error(f"系统 TTS 下发失败: {e}")

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """将 PCM 数据转换为 WAV 格式。"""
        if len(pcm_data) == 0:
            logger.bind(tag=TAG).warning("PCM 数据为空，无法转换 WAV")
            return b""

        # 确保数据长度是偶数（16位音频）
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(pcm_data)

            wav_buffer.seek(0)
            return wav_buffer.read()
        except Exception as e:
            logger.bind(tag=TAG).error(f"WAV 转换失败: {e}")
            return b""

    def stop_ws_connection(self):
        pass

    def save_audio_to_file(self, pcm_data: List[bytes], session_id: str) -> str:
        """PCM 数据保存为 WAV 文件。"""
        module_name = __name__.split(".")[-1]
        file_name = f"asr_{module_name}_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    @abstractmethod
    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本。"""
        pass

    @staticmethod
    def decode_opus(opus_data: List[bytes]) -> List[bytes]:
        """将 Opus 音频数据解码为 PCM 数据。"""
        decoder = None
        try:
            decoder = opuslib_next.Decoder(16000, 1)
            pcm_data = []
            buffer_size = 960  # 60ms @16kHz

            for i, opus_packet in enumerate(opus_data):
                try:
                    if not opus_packet or len(opus_packet) == 0:
                        continue

                    pcm_frame = decoder.decode(opus_packet, buffer_size)
                    if pcm_frame and len(pcm_frame) > 0:
                        pcm_data.append(pcm_frame)

                except opuslib_next.OpusError as e:
                    logger.bind(tag=TAG).warning(
                        f"Opus 解码错误，跳过数据包 {i}: {e}"
                    )
                except Exception as e:
                    logger.bind(tag=TAG).error(f"音频处理错误，数据包 {i}: {e}")

            return pcm_data

        except Exception as e:
            logger.bind(tag=TAG).error(f"音频解码过程发生错误: {e}")
            return []
        finally:
            if decoder is not None:
                try:
                    del decoder
                except Exception as e:
                    logger.bind(tag=TAG).debug(f"释放 decoder 资源时出错: {e}")
