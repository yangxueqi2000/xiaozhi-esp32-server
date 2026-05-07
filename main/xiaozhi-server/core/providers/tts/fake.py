from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType


class TTSProvider(TTSProviderBase):
    """Text-only TTS provider for local smoke tests."""

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

    def to_tts_stream(self, text, opus_handler=None) -> None:
        text = self._normalize_text_for_tts(text)
        if not text:
            return None
        self._put_audio_queue(SentenceType.FIRST, [], text)
        return None

    def to_tts(self, text):
        return []

    async def text_to_speak(self, text, output_file):
        return b""
