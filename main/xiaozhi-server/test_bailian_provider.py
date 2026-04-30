import sys
import types
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


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


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: _DummyLogger()
sys.modules.setdefault("config.logger", fake_logger_module)


fake_util_module = types.ModuleType("core.utils.util")
fake_util_module.audio_to_data = lambda *args, **kwargs: []
fake_util_module.audio_to_data_stream = lambda *args, **kwargs: []
fake_util_module.audio_bytes_to_data_stream = lambda *args, **kwargs: []
sys.modules.setdefault("core.utils.util", fake_util_module)


fake_rate_controller_module = types.ModuleType("core.utils.audioRateController")
fake_rate_controller_module.AudioRateController = object
sys.modules.setdefault("core.utils.audioRateController", fake_rate_controller_module)


fake_report_module = types.ModuleType("core.handle.reportHandle")
fake_report_module.enqueue_tts_report = lambda *args, **kwargs: None
sys.modules.setdefault("core.handle.reportHandle", fake_report_module)


fake_output_counter_module = types.ModuleType("core.utils.output_counter")
fake_output_counter_module.add_device_output = lambda *args, **kwargs: None
sys.modules.setdefault("core.utils.output_counter", fake_output_counter_module)


fake_audioop_module = types.ModuleType("audioop")
fake_audioop_module.ratecv = lambda data, width, channels, inrate, outrate, state: (
    data,
    state,
)
sys.modules.setdefault("audioop", fake_audioop_module)


fake_opus_encoder_module = types.ModuleType("core.utils.opus_encoder_utils")
fake_opus_encoder_module.OpusEncoderUtils = object
sys.modules.setdefault("core.utils.opus_encoder_utils", fake_opus_encoder_module)


fake_dashscope_module = types.ModuleType("dashscope")
fake_dashscope_audio_module = types.ModuleType("dashscope.audio")
fake_dashscope_qwen_tts_module = types.ModuleType("dashscope.audio.qwen_tts")


class _DummySpeechSynthesizer:
    @staticmethod
    def call(*args, **kwargs):
        return []


fake_dashscope_qwen_tts_module.SpeechSynthesizer = _DummySpeechSynthesizer
fake_dashscope_audio_module.qwen_tts = fake_dashscope_qwen_tts_module
fake_dashscope_module.audio = fake_dashscope_audio_module
sys.modules.setdefault("dashscope", fake_dashscope_module)
sys.modules.setdefault("dashscope.audio", fake_dashscope_audio_module)
sys.modules.setdefault("dashscope.audio.qwen_tts", fake_dashscope_qwen_tts_module)


from core.providers.tts import bailian


class BailianProviderModelResolutionTest(unittest.TestCase):
    def test_dedicated_qwen_tts_model_is_kept(self):
        provider = object.__new__(bailian.TTSProvider)

        resolved = provider._resolve_model("qwen-tts")

        self.assertEqual("qwen-tts", resolved)

    def test_qwen_tts_compatible_prefix_is_kept(self):
        provider = object.__new__(bailian.TTSProvider)

        resolved = provider._resolve_model("qwen-tts-latest")

        self.assertEqual("qwen-tts-latest", resolved)

    def test_incompatible_omni_model_falls_back_to_dedicated_tts(self):
        provider = object.__new__(bailian.TTSProvider)

        resolved = provider._resolve_model("qwen3.5-omni-plus")

        self.assertEqual(bailian.DEFAULT_MODEL, resolved)


if __name__ == "__main__":
    unittest.main()
