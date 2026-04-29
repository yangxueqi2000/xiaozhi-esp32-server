import asyncio
import base64
import io
import os
import wave
from pathlib import Path

from openai import OpenAI

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.api_key_env = config.get("api_key_env", "DASHSCOPE_API_KEY")
        self.api_key = (
            str(config.get("api_key") or os.getenv(self.api_key_env) or "").strip()
        )
        if not self.api_key:
            raise ValueError(
                f"Missing API key. Set environment variable {self.api_key_env} "
                "or configure api_key."
            )

        self.base_url = str(
            config.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        ).strip()
        self.model = str(config.get("model", "qwen3.5-omni-plus")).strip()
        self.audio_format = str(
            config.get("audio_format")
            or config.get("response_format")
            or config.get("format")
            or "wav"
        ).strip()
        self.voice = str(config.get("voice") or "").strip() or None
        self.prompt = str(
            config.get("prompt") or "Please answer briefly and naturally in Chinese."
        ).strip()
        self.timeout = float(config.get("timeout", 60) or 60)
        self.max_retries = max(1, int(config.get("max_retries", 2) or 2))
        self.sample_rate = int(config.get("sample_rate", 24000) or 24000)

        self.audio_file_type = self.audio_format
        self.output_file = config.get("output_dir", "tmp/")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    async def text_to_speak(self, text, output_file):
        return await asyncio.to_thread(self._synthesize, text, output_file)

    def _synthesize(self, text, output_file):
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("Bailian TTS text is empty")

        audio_options = {"format": self.audio_format}
        if self.voice:
            audio_options["voice"] = self.voice

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self._build_user_content(clean_text),
                }
            ],
            modalities=["text", "audio"],
            audio=audio_options,
            stream=True,
            stream_options={"include_usage": True},
        )

        audio_parts = []
        for chunk in completion:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue

            audio_data = self._extract_audio_data(getattr(delta, "audio", None))
            if audio_data:
                audio_parts.append(audio_data)

        if not audio_parts:
            raise RuntimeError("Bailian TTS returned no audio data")

        audio_bytes = base64.b64decode("".join(audio_parts))
        audio_bytes = self._normalize_audio_bytes(audio_bytes)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            return None
        return audio_bytes

    def _build_user_content(self, text):
        if not self.prompt:
            return text
        return f"{text}\n\n{self.prompt}"

    def _extract_audio_data(self, audio_delta):
        if not audio_delta:
            return ""
        if isinstance(audio_delta, dict):
            return str(audio_delta.get("data") or "")
        return str(getattr(audio_delta, "data", "") or "")

    def _normalize_audio_bytes(self, audio_bytes):
        if self.audio_format.lower() != "wav":
            return audio_bytes
        if audio_bytes[:4] == b"RIFF":
            return audio_bytes

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)
        return wav_buffer.getvalue()
