import os
import uuid
from datetime import datetime

import edge_tts
from edge_tts.exceptions import NoAudioReceived

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase

TAG = __name__
logger = setup_logging()
DEFAULT_RESCUE_VOICES = ["en-US-EmmaMultilingualNeural"]


def _normalize_voice_list(value):
    if not value:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    return []


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.voice = config.get("private_voice") or config.get("voice")
        self.audio_file_type = config.get("format", "mp3")
        self.proxy = config.get("proxy")
        self.connect_timeout = int(
            config.get("connect_timeout", config.get("timeout", 20))
        )
        self.receive_timeout = int(
            config.get("receive_timeout", max(self.connect_timeout, 60))
        )
        self.promote_success_voice = _to_bool(
            config.get("promote_success_voice", True), True
        )
        self.auto_multilingual_rescue = _to_bool(
            config.get("auto_multilingual_rescue", True), True
        )
        self.voice_candidates = self._build_voice_candidates(
            _normalize_voice_list(config.get("voice_candidates"))
        )
        self.active_voice = self.voice_candidates[0]

    def _build_voice_candidates(self, configured_candidates):
        voices = []
        for voice in [self.voice, *configured_candidates]:
            if voice and voice not in voices:
                voices.append(voice)

        if (
            self.auto_multilingual_rescue
            and self.voice
            and self.voice.startswith("zh-")
            and not any("MultilingualNeural" in voice for voice in voices)
        ):
            for voice in DEFAULT_RESCUE_VOICES:
                if voice not in voices:
                    voices.append(voice)

        if not voices:
            raise ValueError("EdgeTTS requires at least one configured voice")

        return voices

    def _ordered_candidate_voices(self):
        if self.active_voice not in self.voice_candidates:
            return list(self.voice_candidates)
        return [self.active_voice] + [
            voice for voice in self.voice_candidates if voice != self.active_voice
        ]

    def _should_retry_with_next_voice(self, error):
        return isinstance(error, NoAudioReceived) or (
            "No audio was received" in str(error)
        )

    async def _synthesize_with_voice(self, text, voice, output_file):
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            proxy=self.proxy,
            connect_timeout=self.connect_timeout,
            receive_timeout=self.receive_timeout,
        )
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "wb") as file_obj:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        file_obj.write(chunk["data"])
            return output_file

        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
        return bytes(audio_bytes)

    def generate_filename(self, extension=".mp3"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    async def text_to_speak(self, text, output_file):
        last_error = None
        attempted_voices = []

        for voice in self._ordered_candidate_voices():
            attempted_voices.append(voice)
            try:
                result = await self._synthesize_with_voice(text, voice, output_file)
                if self.promote_success_voice:
                    if self.active_voice != voice:
                        logger.bind(tag=TAG).warning(
                            f"EdgeTTS switched to working voice: {self.active_voice} -> {voice}"
                        )
                    self.active_voice = voice
                return result
            except Exception as error:
                last_error = error
                if output_file and os.path.exists(output_file):
                    os.remove(output_file)

                if not self._should_retry_with_next_voice(error):
                    break

                logger.bind(tag=TAG).warning(
                    f"EdgeTTS produced no audio with voice={voice}, retrying next Edge voice"
                )

        attempted = ", ".join(attempted_voices)
        raise Exception(
            f"Edge TTS请求失败: {last_error}; attempted_voices=[{attempted}]"
        )
