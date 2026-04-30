import asyncio
import base64
import audioop
import io
import os
import queue
import time
import traceback
import wave
from http import HTTPStatus
from pathlib import Path

import dashscope
from dashscope.audio.qwen_tts import SpeechSynthesizer

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import ContentType, InterfaceType, SentenceType
from core.utils import opus_encoder_utils, textUtils
from core.utils.tts import MarkdownCleaner

TAG = __name__
logger = setup_logging()

DEFAULT_MODEL = "qwen-tts"
DEFAULT_VOICE = "Cherry"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_LANGUAGE_TYPE = "Chinese"
SUPPORTED_TTS_MODEL_PREFIXES = ("qwen-tts",)


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.interface_type = InterfaceType.SINGLE_STREAM
        self.before_stop_play_files = []

        self.api_key_env = config.get("api_key_env", "DASHSCOPE_API_KEY")
        self.api_key = str(
            config.get("api_key") or self._read_env_value(self.api_key_env) or ""
        ).strip()
        if not self.api_key:
            raise ValueError(
                f"Missing API key. Set environment variable {self.api_key_env} "
                "or configure api_key."
            )

        configured_base_url = str(
            config.get("base_url", "https://dashscope.aliyuncs.com/api/v1")
        ).strip()
        self.base_url = self._resolve_dashscope_base_url(configured_base_url)

        configured_model = str(config.get("model", DEFAULT_MODEL)).strip()
        self.model = self._resolve_model(configured_model)

        self.voice = str(config.get("voice") or DEFAULT_VOICE).strip() or DEFAULT_VOICE
        self.language_type = str(
            config.get("language_type", DEFAULT_LANGUAGE_TYPE) or DEFAULT_LANGUAGE_TYPE
        ).strip()
        self.speech_rate = self._parse_speech_rate(config.get("speech_rate", 1.0))
        self.timeout = float(config.get("timeout", 60) or 60)
        self.max_retries = max(1, int(config.get("max_retries", 3) or 3))
        self.retry_backoff_seconds = float(
            config.get("retry_backoff_seconds", 0.8) or 0.8
        )
        self.output_audio_format = str(
            config.get("audio_format")
            or config.get("response_format")
            or config.get("format")
            or "wav"
        ).strip().lower()
        self.output_file = config.get("output_dir", "tmp/")
        self.audio_file_type = "pcm"
        self.stream_sample_rate = DEFAULT_SAMPLE_RATE

        self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=self.stream_sample_rate,
            channels=1,
            frame_size_ms=60,
        )

        dashscope.api_key = self.api_key
        dashscope.base_http_api_url = self.base_url

    @staticmethod
    def _parse_speech_rate(value):
        try:
            parsed = float(value or 1.0)
        except (TypeError, ValueError):
            parsed = 1.0
        return min(max(parsed, 0.85), 1.35)

    def _prepare_spoken_text(self, text: str) -> str:
        clean_text = MarkdownCleaner.clean_markdown(text)
        if getattr(self, "conn", None) is not None:
            clean_text = textUtils.prepare_runtime_spoken_text_for_conn(
                self.conn, clean_text
            )
        else:
            clean_text = textUtils.prepare_runtime_spoken_text(clean_text)
        return self._normalize_text_for_tts(clean_text)

    def _apply_speech_rate_to_pcm(self, pcm_bytes: bytes, state=None):
        if not pcm_bytes or abs(self.speech_rate - 1.0) < 1e-3:
            return pcm_bytes, state

        source_rate = max(1, int(round(self.stream_sample_rate * self.speech_rate)))
        converted, next_state = audioop.ratecv(
            pcm_bytes,
            2,
            1,
            source_rate,
            self.stream_sample_rate,
            state,
        )
        return converted, next_state

    def _read_env_value(self, env_name):
        value = os.getenv(env_name)
        if value:
            return value

        if os.name != "nt":
            return None

        try:
            import winreg

            registry_paths = (
                (winreg.HKEY_CURRENT_USER, r"Environment"),
                (
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ),
            )
            for root, subkey in registry_paths:
                try:
                    with winreg.OpenKey(root, subkey) as key:
                        registry_value, _ = winreg.QueryValueEx(key, env_name)
                        if registry_value:
                            return str(registry_value)
                except OSError:
                    continue
        except Exception as exc:
            logger.bind(tag=TAG).warning(
                f"Failed to read Windows environment registry for {env_name}: {exc}"
            )

        return None

    def _resolve_dashscope_base_url(self, base_url):
        url = str(base_url or "").strip().rstrip("/")
        if not url:
            return "https://dashscope.aliyuncs.com/api/v1"
        if url.endswith("/compatible-mode/v1"):
            return url[: -len("/compatible-mode/v1")] + "/api/v1"
        return url

    def _resolve_model(self, configured_model):
        model = configured_model or DEFAULT_MODEL
        if not model:
            return DEFAULT_MODEL
        normalized_model = str(model).strip()
        if normalized_model.lower().startswith(SUPPORTED_TTS_MODEL_PREFIXES):
            return normalized_model
        logger.bind(tag=TAG).warning(
            "Configured Bailian model is incompatible with "
            "dashscope.audio.qwen_tts.SpeechSynthesizer: "
            f"{normalized_model}. Falling back to {DEFAULT_MODEL}."
        )
        return DEFAULT_MODEL

    def tts_text_priority_thread(self):
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                if message.sentence_type == SentenceType.FIRST:
                    self.conn.client_abort = False

                if self.conn.client_abort:
                    logger.bind(tag=TAG).info(
                        "Received client abort, skip current Bailian TTS task."
                    )
                    continue

                if message.sentence_type == SentenceType.FIRST:
                    self.tts_stop_request = False
                    self.processed_chars = 0
                    self.tts_text_buff = []
                    self.is_first_sentence = True
                    # Each new TTS turn must re-enter the frontend/device speaking
                    # state, otherwise later replies may keep the client in
                    # listening mode and the synthesized audio will not play.
                    self.tts_audio_first_sentence = True
                    self.before_stop_play_files.clear()
                    self._current_audio_sentence_id = message.sentence_id
                elif ContentType.TEXT == message.content_type:
                    self.tts_text_buff.append(message.content_detail)
                    segment_text = self._get_segment_text()
                    if segment_text:
                        self.to_tts_single_stream(segment_text)
                elif ContentType.FILE == message.content_type:
                    if message.content_file and os.path.exists(message.content_file):
                        self._process_audio_file_stream(
                            message.content_file,
                            callback=lambda audio_data: self.handle_audio_file(
                                audio_data, message.content_detail
                            ),
                        )

                if message.sentence_type == SentenceType.LAST:
                    self._process_remaining_text_stream(is_last=True)

            except queue.Empty:
                continue
            except Exception as exc:
                logger.bind(tag=TAG).error(
                    "Failed to process Bailian TTS text: "
                    f"{exc}, type={type(exc).__name__}, stack={traceback.format_exc()}"
                )

    def _process_remaining_text_stream(self, is_last=False):
        full_text = "".join(self.tts_text_buff)
        remaining_text = full_text[self.processed_chars :]
        if remaining_text:
            segment_text = textUtils.get_string_no_punctuation_or_emoji(remaining_text)
            if segment_text:
                self.to_tts_single_stream(segment_text, is_last=is_last)
                self.processed_chars += len(full_text)
                return
        self._process_before_stop_play_files()

    def to_tts_single_stream(self, text, is_last=False):
        clean_text = self._prepare_spoken_text(text)
        if not clean_text:
            if is_last:
                self._process_before_stop_play_files()
            return None

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                asyncio.run(self._stream_tts(clean_text, is_last))
                logger.bind(tag=TAG).info(
                    f"Bailian TTS streaming success: {clean_text}, retries={attempt - 1}"
                )
                return None
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    logger.bind(tag=TAG).warning(
                        f"Bailian TTS stream attempt {attempt}/{self.max_retries} "
                        f"failed: {exc}"
                    )
                    time.sleep(self.retry_backoff_seconds * attempt)

        logger.bind(tag=TAG).error(
            f"Bailian TTS streaming failed after {self.max_retries} attempts: "
            f"{clean_text}, error={last_error}"
        )
        if is_last:
            self._process_before_stop_play_files()
        else:
            logger.bind(tag=TAG).warning(
                "Skip premature TTS stop after non-final Bailian segment failure; "
                "waiting for follow-up text or final flush."
            )
        return None

    async def _stream_tts(self, text, is_last):
        await asyncio.to_thread(self._stream_synthesize, text, is_last)

    async def text_to_speak(self, text, output_file):
        return await asyncio.to_thread(self._synthesize_full_audio, text, output_file)

    def _stream_synthesize(self, text, is_last):
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("Bailian TTS text is empty")

        self.opus_encoder.reset_state()
        sent_first_packet = False
        saw_audio = False
        rate_state = None

        for pcm_chunk in self._iter_pcm_chunks(clean_text):
            if self.conn.client_abort:
                return
            if not sent_first_packet:
                self._put_audio_queue(SentenceType.FIRST, [], clean_text)
                sent_first_packet = True
            if not pcm_chunk:
                continue
            pcm_chunk, rate_state = self._apply_speech_rate_to_pcm(
                pcm_chunk, rate_state
            )
            if not pcm_chunk:
                continue
            saw_audio = True
            self.opus_encoder.encode_pcm_to_opus_stream(
                pcm_chunk,
                end_of_stream=False,
                callback=self.handle_opus,
            )

        if not saw_audio:
            raise RuntimeError("Bailian TTS returned no audio data")

        self.opus_encoder.encode_pcm_to_opus_stream(
            b"",
            end_of_stream=True,
            callback=self.handle_opus,
        )

        if is_last:
            self._process_before_stop_play_files()

    def _synthesize_full_audio(self, text, output_file):
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("Bailian TTS text is empty")

        pcm_chunks = list(self._iter_pcm_chunks(clean_text))
        if not pcm_chunks:
            raise RuntimeError("Bailian TTS returned no audio data")

        pcm_bytes = b"".join(pcm_chunks)
        pcm_bytes, _ = self._apply_speech_rate_to_pcm(pcm_bytes, None)
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_audio_format == "pcm":
                output_path.write_bytes(pcm_bytes)
            else:
                output_path.write_bytes(self._wrap_pcm_as_wav(pcm_bytes))
            return None
        return pcm_bytes

    def _iter_pcm_chunks(self, text):
        response = SpeechSynthesizer.call(
            model=self.model,
            text=text,
            voice=self.voice,
            api_key=self.api_key,
            stream=True,
            timeout=self.timeout,
            language_type=self.language_type,
        )

        saw_audio = False
        for chunk in response:
            self._raise_if_failed(chunk)
            pcm_chunk = self._extract_audio_chunk(chunk)
            if not pcm_chunk:
                continue
            saw_audio = True
            yield pcm_chunk

        if not saw_audio:
            raise RuntimeError("Bailian TTS returned no audio data")

    def _extract_audio_chunk(self, chunk):
        output = getattr(chunk, "output", None)
        audio = getattr(output, "audio", None) if output else None
        if isinstance(audio, dict):
            data = audio.get("data")
        else:
            data = getattr(audio, "data", None)
        if not data:
            return b""
        return base64.b64decode(data)

    def _raise_if_failed(self, chunk):
        status_code = getattr(chunk, "status_code", None)
        if status_code in (None, HTTPStatus.OK, 200):
            return
        code = getattr(chunk, "code", "unknown")
        message = getattr(chunk, "message", "unknown error")
        raise RuntimeError(f"Bailian TTS request failed: {code} - {message}")

    def _wrap_pcm_as_wav(self, pcm_bytes):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.stream_sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    def to_tts(self, text):
        clean_text = self._prepare_spoken_text(text)
        if not clean_text:
            return []

        pcm_bytes = asyncio.run(self.text_to_speak(clean_text, None))
        if not pcm_bytes:
            return []

        encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=self.stream_sample_rate,
            channels=1,
            frame_size_ms=60,
        )
        opus_datas = []
        try:
            encoder.encode_pcm_to_opus_stream(
                pcm_bytes,
                end_of_stream=True,
                callback=lambda opus: opus_datas.append(opus),
            )
            return opus_datas
        finally:
            encoder.close()

    async def close(self):
        await super().close()
        if hasattr(self, "opus_encoder"):
            self.opus_encoder.close()
