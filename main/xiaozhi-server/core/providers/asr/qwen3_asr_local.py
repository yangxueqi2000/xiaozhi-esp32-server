import os
import time
import asyncio
from typing import Optional, Tuple, List
import numpy as np
import torch
from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None

    value = str(language).strip()
    if not value:
        return None

    aliases = {
        "auto": None,
        "none": None,
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "en": "English",
        "yue": "Cantonese",
    }
    return aliases.get(value.lower(), value)


def _resolve_dtype(dtype_value: str):
    value = str(dtype_value or "auto").strip().lower()
    if value in ("auto", ""):
        return None
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    if value in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_value}")


def _parse_positive_int(value, default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    return default


def _compact_text_for_echo_check(text: str) -> str:
    return "".join(str(text or "").split())


def _looks_like_context_echo(text: str, context: str) -> bool:
    """Drop ASR hallucinations that repeat the recognition context itself."""
    compact_text = _compact_text_for_echo_check(text)
    if len(compact_text) < 16:
        return False

    internal_markers = (
        "当前实验是",
        "常见化学词汇包括",
        "常见缩写和读法包括",
        "实验中会测试",
        "当听到类似",
        "优先识别为这些化学实验词汇",
    )
    if any(marker in compact_text for marker in internal_markers):
        return True

    compact_context = _compact_text_for_echo_check(context)
    if len(compact_context) < 16:
        return False
    if compact_text in compact_context:
        return True

    # Long outputs made mostly of context vocabulary are usually context echo
    # when Qwen3-ASR receives silence/noise or a clipped tail.
    overlap_chars = sum(1 for char in compact_text if char in compact_context)
    return len(compact_text) >= 40 and overlap_chars / max(1, len(compact_text)) > 0.92


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL

        self.model_name = config.get("model_name", "Qwen/Qwen3-ASR-0.6B")
        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file

        self.context = config.get("context", "")
        self.language = _normalize_language(config.get("language", "auto"))
        self.max_inference_batch_size = int(config.get("max_inference_batch_size", 8))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.trust_remote_code = _parse_bool(
            config.get("trust_remote_code"), True
        )
        self.hf_endpoint = str(config.get("hf_endpoint", "")).strip()
        self.http_proxy = str(config.get("http_proxy", "")).strip()
        self.https_proxy = str(config.get("https_proxy", "")).strip()
        self.hf_hub_etag_timeout = _parse_positive_int(
            config.get("hf_hub_etag_timeout"), None
        )
        self.hf_hub_download_timeout = _parse_positive_int(
            config.get("hf_hub_download_timeout"), None
        )
        self.local_files_only = _parse_bool(
            config.get("local_files_only"), False
        )
        self.hf_hub_offline = _parse_bool(
            config.get("hf_hub_offline"), self.local_files_only
        )

        configured_device = str(config.get("device", "auto")).strip().lower()
        if configured_device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = configured_device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.bind(tag=TAG).warning(
                f"CUDA is not available, fallback to CPU. requested_device={self.device}"
            )
            self.device = "cpu"

        self.dtype = _resolve_dtype(config.get("dtype", "auto"))
        if self.device == "cpu" and self.dtype in (torch.float16, torch.bfloat16):
            logger.bind(tag=TAG).warning(
                "CPU mode does not suit fp16/bf16 well, fallback dtype to float32."
            )
            self.dtype = torch.float32

        os.makedirs(self.output_dir, exist_ok=True)

        # Apply Hugging Face environment settings before importing qwen_asr /
        # transformers so mirror or offline mode is respected during init.
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        if self.http_proxy:
            os.environ["HTTP_PROXY"] = self.http_proxy
            os.environ["http_proxy"] = self.http_proxy
        if self.https_proxy:
            os.environ["HTTPS_PROXY"] = self.https_proxy
            os.environ["https_proxy"] = self.https_proxy
        if self.hf_hub_etag_timeout is not None:
            os.environ["HF_HUB_ETAG_TIMEOUT"] = str(self.hf_hub_etag_timeout)
        if self.hf_hub_download_timeout is not None:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(self.hf_hub_download_timeout)
        if self.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from qwen_asr import Qwen3ASRModel
        except Exception as e:
            raise ImportError(
                "qwen-asr is not installed. install it in runtime env first."
            ) from e

        init_kwargs = {
            "device_map": self.device,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dtype is not None:
            init_kwargs["dtype"] = self.dtype
        if self.local_files_only:
            init_kwargs["local_files_only"] = True

        start_time = time.time()
        self.model = Qwen3ASRModel.from_pretrained(self.model_name, **init_kwargs)
        logger.bind(tag=TAG).info(
            f"Qwen3ASR local model loaded: model={self.model_name}, "
            f"device={self.device}, cost={time.time() - start_time:.2f}s"
        )

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        try:
            if audio_format == "pcm":
                pcm_data = opus_data
            else:
                pcm_data = self.decode_opus(opus_data)

            combined_pcm_data = b"".join(pcm_data)
            if not combined_pcm_data:
                return "", file_path

            if not self.delete_audio_file:
                file_path = self.save_audio_to_file(pcm_data, session_id)

            pcm_float = (
                np.frombuffer(combined_pcm_data, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            start_time = time.time()
            results = await asyncio.to_thread(
                self.model.transcribe,
                audio=(pcm_float, 16000),
                context=self.context,
                language=self.language,
            )

            if not results:
                return "", file_path

            best = results[0]
            text = (best.text or "").strip()
            language = (best.language or "").strip()
            if _looks_like_context_echo(text, self.context):
                logger.bind(tag=TAG).warning(
                    f"Qwen3ASR local context echo filtered: text={text[:120]}"
                )
                return "", file_path
            logger.bind(tag=TAG).debug(
                f"Qwen3ASR local recognize cost={time.time() - start_time:.3f}s, "
                f"language={language}, text={text}"
            )

            if not text:
                return "", file_path

            payload = {"content": text}
            if language:
                payload["language"] = language
            return payload, file_path
        except Exception as e:
            logger.bind(tag=TAG).error(f"Qwen3ASR local recognize failed: {e}")
            return "", file_path
        finally:
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"delete temp audio failed: {file_path}, err={e}"
                    )
