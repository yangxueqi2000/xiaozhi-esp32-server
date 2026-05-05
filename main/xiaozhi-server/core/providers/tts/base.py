import os
import re
import time
import uuid
import queue
import asyncio
import threading
import traceback
from core.utils import p3
from datetime import datetime
from core.utils import textUtils
from typing import Callable, Any
from abc import ABC, abstractmethod
from config.logger import setup_logging
from core.utils.tts import MarkdownCleaner
from core.utils.output_counter import add_device_output
from core.handle.reportHandle import enqueue_tts_report
from core.handle.sendAudioHandle import sendAudioMessage
from core.utils.util import audio_bytes_to_data_stream, audio_to_data_stream
from core.providers.tts.dto.dto import (
    TTSMessageDTO,
    SentenceType,
    ContentType,
    InterfaceType,
)

TAG = __name__
logger = setup_logging()


class TTSProviderBase(ABC):
    def __init__(self, config, delete_audio_file):
        self.interface_type = InterfaceType.NON_STREAM
        self.conn = None
        self.delete_audio_file = delete_audio_file
        self.audio_file_type = "wav"
        self.output_file = config.get("output_dir", "tmp/")
        self.tts_text_queue = queue.Queue()
        self.tts_audio_queue = queue.Queue()
        self.tts_audio_first_sentence = True
        self.before_stop_play_files = []

        self.tts_text_buff = []
        self.punctuations = (
            "。",
            "？",
            "?",
            "！",
            "!",
            "；",
            ";",
            "：",
        )
        self.first_sentence_punctuations = (
            "，",
            "~",
            "、",
            ",",
            "。",
            "？",
            "?",
            "！",
            "!",
            "；",
            ";",
            "：",
        )
        self.tts_stop_request = False
        self.processed_chars = 0
        self.is_first_sentence = True
        self.disable_pre_speak_split = str(
            config.get("disable_pre_speak_split", False)
        ).lower() in ("1", "true", "yes", "on")
        self.fallback_provider = None
        self.fallback_provider_name = ""
        self._tts_text_processing_lock = threading.Lock()
        self._tts_text_processing_count = 0
        fallback_after_failures = config.get("fallback_after_failures", 1)
        try:
            self.fallback_after_failures = max(1, int(fallback_after_failures))
        except (TypeError, ValueError):
            self.fallback_after_failures = 1

    def generate_filename(self, extension=".wav"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    def _put_audio_queue(self, sentence_type, audio_data, text=None, sentence_id=None):
        resolved_sentence_id = str(
            sentence_id
            or getattr(self, "_current_audio_sentence_id", "")
            or getattr(getattr(self, "conn", None), "sentence_id", "")
            or ""
        ).strip()
        self.tts_audio_queue.put(
            (sentence_type, audio_data, text, resolved_sentence_id)
        )

    def handle_opus(self, opus_data: bytes):
        logger.bind(tag=TAG).debug(f"推送数据到队列里面帧数～～ {len(opus_data)}")
        self._put_audio_queue(SentenceType.MIDDLE, opus_data, None)

    def handle_audio_file(self, file_audio: bytes, text):
        self.before_stop_play_files.append((file_audio, text))

    def _normalize_text_for_tts(self, text: str) -> str:
        aliases = None
        conn = getattr(self, "conn", None)
        if conn is not None:
            aliases = conn.config.get("spoken_aliases")
        return textUtils.normalize_tts_text(text, custom_aliases=aliases)

    def set_fallback_provider(self, provider, provider_name: str = ""):
        self.fallback_provider = provider
        self.fallback_provider_name = str(provider_name or "").strip()

    def _mark_tts_text_processing_start(self):
        with self._tts_text_processing_lock:
            self._tts_text_processing_count += 1

    def _mark_tts_text_processing_end(self):
        with self._tts_text_processing_lock:
            self._tts_text_processing_count = max(
                0, self._tts_text_processing_count - 1
            )

    def has_inflight_tts_text_processing(self) -> bool:
        with self._tts_text_processing_lock:
            return self._tts_text_processing_count > 0

    async def _synthesize_text_once(self, text, output_file, *, attempt_number: int):
        try:
            return await self.text_to_speak(text, output_file)
        except Exception as primary_exc:
            should_fallback = (
                self.fallback_provider is not None
                and attempt_number >= self.fallback_after_failures
            )
            if not should_fallback:
                raise

            fallback_name = self.fallback_provider_name or self.fallback_provider.__class__.__name__
            logger.bind(tag=TAG).warning(
                "主TTS生成失败，切换备用TTS: "
                f"attempt={attempt_number}, primary={self.__class__.__name__}, "
                f"fallback={fallback_name}, error={primary_exc}"
            )
            return await self.fallback_provider.text_to_speak(text, output_file)

    def to_tts_stream(self, text, opus_handler: Callable[[bytes], None] = None) -> None:
        text = MarkdownCleaner.clean_markdown(text)
        text = textUtils.prepare_runtime_spoken_text_for_conn(self.conn, text)
        text = self._normalize_text_for_tts(text)
        if not text:
            return None
        max_repeat_time = 5
        if self.delete_audio_file:
            # 需要删除文件的直接转为音频数据
            while max_repeat_time > 0:
                try:
                    attempt_number = 5 - max_repeat_time + 1
                    audio_bytes = asyncio.run(
                        self._synthesize_text_once(
                            text,
                            None,
                            attempt_number=attempt_number,
                        )
                    )
                    if audio_bytes:
                        self._put_audio_queue(SentenceType.FIRST, None, text)
                        audio_bytes_to_data_stream(
                            audio_bytes,
                            file_type=self.audio_file_type,
                            is_opus=True,
                            callback=opus_handler,
                        )
                        break
                    else:
                        max_repeat_time -= 1
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"语音生成失败{5 - max_repeat_time + 1}次: {text}，错误: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"语音生成成功: {text}，重试{5 - max_repeat_time}次"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"语音生成失败: {text}，请检查网络或服务是否正常"
                )
            return None
        else:
            tmp_file = self.generate_filename()
            try:
                while not os.path.exists(tmp_file) and max_repeat_time > 0:
                    try:
                        attempt_number = 5 - max_repeat_time + 1
                        asyncio.run(
                            self._synthesize_text_once(
                                text,
                                tmp_file,
                                attempt_number=attempt_number,
                            )
                        )
                    except Exception as e:
                        logger.bind(tag=TAG).warning(
                            f"语音生成失败{5 - max_repeat_time + 1}次: {text}，错误: {e}"
                        )
                        # 未执行成功，删除文件
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        max_repeat_time -= 1

                if max_repeat_time > 0:
                    logger.bind(tag=TAG).info(
                        f"语音生成成功: {text}:{tmp_file}，重试{5 - max_repeat_time}次"
                    )
                else:
                    logger.bind(tag=TAG).error(
                        f"语音生成失败: {text}，请检查网络或服务是否正常"
                    )
                    self._put_audio_queue(SentenceType.FIRST, None, text)
                self._process_audio_file_stream(tmp_file, callback=opus_handler)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
                return None
    
    def to_tts(self, text):
        text = MarkdownCleaner.clean_markdown(text)
        text = textUtils.prepare_runtime_spoken_text_for_conn(self.conn, text)
        text = self._normalize_text_for_tts(text)
        if not text:
            return None
        max_repeat_time = 5
        if self.delete_audio_file:
            # 需要删除文件的直接转为音频数据
            while max_repeat_time > 0:
                try:
                    attempt_number = 5 - max_repeat_time + 1
                    audio_bytes = asyncio.run(
                        self._synthesize_text_once(
                            text,
                            None,
                            attempt_number=attempt_number,
                        )
                    )
                    if audio_bytes:
                        audio_datas = []
                        audio_bytes_to_data_stream(
                            audio_bytes,
                            file_type=self.audio_file_type,
                            is_opus=True,
                            callback=lambda data: audio_datas.append(data)
                        )
                        return audio_datas
                    else:
                        max_repeat_time -= 1
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"语音生成失败{5 - max_repeat_time + 1}次: {text}，错误: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"语音生成成功: {text}，重试{5 - max_repeat_time}次"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"语音生成失败: {text}，请检查网络或服务是否正常"
                )
            return None
        else:
            tmp_file = self.generate_filename()
            try:
                while not os.path.exists(tmp_file) and max_repeat_time > 0:
                    try:
                        attempt_number = 5 - max_repeat_time + 1
                        asyncio.run(
                            self._synthesize_text_once(
                                text,
                                tmp_file,
                                attempt_number=attempt_number,
                            )
                        )
                    except Exception as e:
                        logger.bind(tag=TAG).warning(
                            f"语音生成失败{5 - max_repeat_time + 1}次: {text}，错误: {e}"
                        )
                        # 未执行成功，删除文件
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        max_repeat_time -= 1

                if max_repeat_time > 0:
                    logger.bind(tag=TAG).info(
                        f"语音生成成功: {text}:{tmp_file}，重试{5 - max_repeat_time}次"
                    )
                else:
                    logger.bind(tag=TAG).error(
                        f"语音生成失败: {text}，请检查网络或服务是否正常"
                    )

                return tmp_file
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to generate TTS file: {e}")
                return None

    @abstractmethod
    async def text_to_speak(self, text, output_file):
        pass

    def audio_to_pcm_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """音频文件转换为PCM编码"""
        return audio_to_data_stream(audio_file_path, is_opus=False, callback=callback)

    def audio_to_opus_data_stream(
        self, audio_file_path, callback: Callable[[Any], Any] = None
    ):
        """音频文件转换为Opus编码"""
        return audio_to_data_stream(audio_file_path, is_opus=True, callback=callback)

    def tts_one_sentence(
        self,
        conn,
        content_type,
        content_detail=None,
        content_file=None,
        sentence_id=None,
    ):
        """发送一句话"""
        if not sentence_id:
            if conn.sentence_id:
                sentence_id = conn.sentence_id
            else:
                sentence_id = str(uuid.uuid4().hex)
                conn.sentence_id = sentence_id
        if self.disable_pre_speak_split:
            self.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=content_type,
                    content_detail=content_detail,
                    content_file=content_file,
                )
            )
            return
        # 对于单句的文本，进行分段处理
        segments = re.split(r"([。！？!?；;\n])", content_detail)
        for seg in segments:
            self.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=content_type,
                    content_detail=seg,
                    content_file=content_file,
                )
            )

    async def open_audio_channels(self, conn):
        self.conn = conn
        # tts 消化线程
        self.tts_priority_thread = threading.Thread(
            target=self.tts_text_priority_thread, daemon=True
        )
        self.tts_priority_thread.start()

        # 音频播放 消化线程
        self.audio_play_priority_thread = threading.Thread(
            target=self._audio_play_priority_thread, daemon=True
        )
        self.audio_play_priority_thread.start()

    # 这里默认是非流式的处理方式
    # 流式处理方式请在子类中重写
    def tts_text_priority_thread(self):
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                self._mark_tts_text_processing_start()
                try:
                    if message.sentence_type == SentenceType.FIRST:
                        self.conn.client_abort = False
                    if self.conn.client_abort:
                        logger.bind(tag=TAG).info("收到打断信息，终止TTS文本处理线程")
                        continue
                    if message.sentence_type == SentenceType.FIRST:
                        # 初始化参数
                        self.tts_stop_request = False
                        self.processed_chars = 0
                        self.tts_text_buff = []
                        self.is_first_sentence = True
                        self.tts_audio_first_sentence = True
                        self._current_audio_sentence_id = message.sentence_id
                    elif ContentType.TEXT == message.content_type:
                        self.tts_text_buff.append(message.content_detail)
                        segment_text = self._get_segment_text()
                        if segment_text:
                            self.to_tts_stream(segment_text, opus_handler=self.handle_opus)
                    elif ContentType.FILE == message.content_type:
                        self._process_remaining_text_stream(opus_handler=self.handle_opus)
                        tts_file = message.content_file
                        if tts_file and os.path.exists(tts_file):
                            self._process_audio_file_stream(
                                tts_file, callback=self.handle_opus
                            )
                    if message.sentence_type == SentenceType.LAST:
                        self._process_remaining_text_stream(opus_handler=self.handle_opus)
                        self._put_audio_queue(
                            message.sentence_type,
                            [],
                            message.content_detail,
                            sentence_id=message.sentence_id,
                        )
                finally:
                    self._mark_tts_text_processing_end()

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理TTS文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )
                continue

    def _audio_play_priority_thread(self):
        # 需要上报的文本和音频列表
        enqueue_text = None
        enqueue_audio = None
        while not self.conn.stop_event.is_set():
            text = None
            try:
                try:
                    queue_item = self.tts_audio_queue.get(timeout=0.1)
                    if isinstance(queue_item, tuple) and len(queue_item) >= 4:
                        sentence_type, audio_datas, text, sentence_id = queue_item
                    else:
                        sentence_type, audio_datas, text = queue_item
                        sentence_id = ""
                except queue.Empty:
                    if self.conn.stop_event.is_set():
                        break
                    continue

                if self.conn.client_abort:
                    logger.bind(tag=TAG).debug("收到打断信号，跳过当前音频数据")
                    enqueue_text, enqueue_audio = None, []
                    continue

                # 收到下一个文本开始或会话结束时进行上报
                if sentence_type is not SentenceType.MIDDLE:
                    # 上报TTS数据
                    if enqueue_text is not None and enqueue_audio is not None:
                        enqueue_tts_report(self.conn, enqueue_text, enqueue_audio)
                    enqueue_audio = []
                    enqueue_text = text

                # 收集上报音频数据
                if isinstance(audio_datas, bytes) and enqueue_audio is not None:
                    enqueue_audio.append(audio_datas)

                # 发送音频
                future = asyncio.run_coroutine_threadsafe(
                    sendAudioMessage(
                        self.conn,
                        sentence_type,
                        audio_datas,
                        text,
                        sentence_id=sentence_id,
                    ),
                    self.conn.loop,
                )
                future.result()

                # 记录输出和报告
                if self.conn.max_output_size > 0 and text:
                    add_device_output(self.conn.headers.get("device-id"), len(text))

            except Exception as e:
                logger.bind(tag=TAG).error(f"audio_play_priority_thread: {text} {e}")

    async def start_session(self, session_id):
        pass

    async def finish_session(self, session_id):
        pass

    async def close(self):
        """资源清理方法"""
        if hasattr(self, "ws") and self.ws:
            await self.ws.close()

    def _get_segment_text(self):
        # 合并当前全部文本并处理未分割部分
        full_text = "".join(self.tts_text_buff)
        current_text = full_text[self.processed_chars :]  # 从未处理的位置开始
        last_punct_pos = -1

        if self.disable_pre_speak_split:
            # Do not pre-split on commas, but allow TTS to start once a full
            # sentence has completed instead of waiting for the whole turn.
            strong_sentence_endings = ("。", "！", "？", "!", "?", "；", ";", "\n")
            raw_hold_chars = 10
            conn_config = getattr(getattr(self, "conn", None), "config", {}) or {}
            try:
                raw_hold_chars = max(
                    0,
                    int(conn_config.get("tts_stream_followup_hold_chars", 10) or 0),
                )
            except (TypeError, ValueError):
                raw_hold_chars = 10

            trailing_text = current_text.rstrip()
            if (
                trailing_text
                and trailing_text[-1] in strong_sentence_endings
                and not self.tts_stop_request
            ):
                return None

            for punct in strong_sentence_endings:
                pos = current_text.rfind(punct)
                if pos > last_punct_pos:
                    last_punct_pos = pos

            if last_punct_pos != -1:
                followup_text = textUtils.get_string_no_punctuation_or_emoji(
                    current_text[last_punct_pos + 1 :]
                ).strip()
                if (
                    not self.tts_stop_request
                    and raw_hold_chars > 0
                    and len(followup_text) < raw_hold_chars
                ):
                    return None
                segment_text_raw = current_text[: last_punct_pos + 1]
                segment_text = textUtils.get_string_no_punctuation_or_emoji(
                    segment_text_raw
                )
                self.processed_chars += len(segment_text_raw)
                self.is_first_sentence = False
                return segment_text

            if self.tts_stop_request and current_text:
                self.is_first_sentence = True
                return current_text
            return None

        # 根据是否是第一句话选择不同的标点符号集合
        punctuations_to_use = (
            self.first_sentence_punctuations
            if self.is_first_sentence
            else self.punctuations
        )

        for punct in punctuations_to_use:
            pos = current_text.rfind(punct)
            if (pos != -1 and last_punct_pos == -1) or (
                pos != -1 and pos > last_punct_pos
            ):
                last_punct_pos = pos

        if last_punct_pos != -1:
            segment_text_raw = current_text[: last_punct_pos + 1]
            segment_text = textUtils.get_string_no_punctuation_or_emoji(
                segment_text_raw
            )
            self.processed_chars += len(segment_text_raw)  # 更新已处理字符位置

            # 如果是第一句话，在找到第一个逗号后，将标志设置为False
            if self.is_first_sentence:
                self.is_first_sentence = False

            return segment_text
        elif self.tts_stop_request and current_text:
            segment_text = current_text
            self.is_first_sentence = True  # 重置标志
            return segment_text
        else:
            return None

    def _process_audio_file_stream(
        self, tts_file, callback: Callable[[Any], Any]
    ) -> None:
        """处理音频文件并转换为指定格式

        Args:
            tts_file: 音频文件路径
            callback: 文件处理函数
        """
        if tts_file.endswith(".p3"):
            p3.decode_opus_from_file_stream(tts_file, callback=callback)
        elif self.conn.audio_format == "pcm":
            self.audio_to_pcm_data_stream(tts_file, callback=callback)
        else:
            self.audio_to_opus_data_stream(tts_file, callback=callback)

        if (
            self.delete_audio_file
            and tts_file is not None
            and os.path.exists(tts_file)
            and tts_file.startswith(self.output_file)
        ):
            os.remove(tts_file)

    def _process_before_stop_play_files(self):
        for audio_datas, text in self.before_stop_play_files:
            self._put_audio_queue(SentenceType.MIDDLE, audio_datas, text)
        self.before_stop_play_files.clear()
        self._put_audio_queue(SentenceType.LAST, [], None)

    def _process_remaining_text_stream(
        self, opus_handler: Callable[[bytes], None] = None
    ):
        """处理剩余的文本并生成语音

        Returns:
            bool: 是否成功处理了文本
        """
        full_text = "".join(self.tts_text_buff)
        remaining_text = full_text[self.processed_chars :]
        if remaining_text:
            segment_text = textUtils.get_string_no_punctuation_or_emoji(remaining_text)
            if segment_text:
                self.to_tts_stream(segment_text, opus_handler=opus_handler)
                self.processed_chars += len(full_text)
                return True
        return False
