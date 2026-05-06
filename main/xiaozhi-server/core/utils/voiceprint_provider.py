import asyncio
import time
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from dataclasses import dataclass
from io import BytesIO
import math
import wave

import aiohttp
import requests

from config.logger import setup_logging
from core.utils.cache.config import CacheType
from core.utils.cache.manager import cache_manager

TAG = __name__
logger = setup_logging()


def is_voiceprint_feature_enabled(config: Optional[dict]) -> bool:
    if not config:
        return False

    enabled = config.get("enabled")
    if enabled is None:
        return True
    if isinstance(enabled, bool):
        return enabled
    if isinstance(enabled, str):
        return enabled.strip().lower() in {"1", "true", "yes", "on"}
    return bool(enabled)


@dataclass
class SpeakerFilterConfig:
    enabled: bool = False
    window_ms: int = 600
    hop_ms: int = 200
    edge_padding_ms: int = 200
    min_match_ratio: float = 0.35
    min_longest_match_ms: int = 800
    max_middle_interference_ratio: float = 0.2
    reject_prompt: str = ""
    debug_log: bool = False


class VoiceprintProvider:
    """声纹识别服务提供者。

    支持两种模式：
    - 静态模式（兼容旧逻辑）：使用 speakers + identify
    - 动态模式（新增）：首次自动注册 master_speaker，后续仅对 master 进行鉴权
    """

    def __init__(self, config: dict, runtime_scope: str = ""):
        self.feature_enabled = is_voiceprint_feature_enabled(config)
        self.original_url = config.get("url", "")
        self.speakers = config.get("speakers", [])
        self.speaker_map = self._parse_speakers()
        self.similarity_threshold = float(config.get("similarity_threshold", 0.25))
        self.runtime_scope = self._sanitize_scope(runtime_scope)

        # Dynamic mode config.
        self.dynamic_mode = self._as_bool(
            config.get("dynamic_mode", config.get("dynamic_master", False))
        )
        self.dynamic_master_speaker_id_base = str(
            config.get("dynamic_master_speaker_id", "master_speaker")
        ).strip() or "master_speaker"
        self.dynamic_master_speaker_id = self._build_dynamic_master_speaker_id(
            self.dynamic_master_speaker_id_base,
            self.runtime_scope,
        )
        self.dynamic_master_name = str(
            config.get("dynamic_master_name", "主说话人")
        )
        self.dynamic_registration_required_samples = max(
            1, int(config.get("dynamic_registration_required_samples", 3))
        )
        self.dynamic_registration_reply = str(
            config.get(
                "dynamic_registration_reply",
                "Hi, your voice has been registered. I will prioritize your voice.",
            )
        )
        self.dynamic_reject_reply = str(config.get("dynamic_reject_reply", ""))
        self.dynamic_fail_open = self._as_bool(config.get("dynamic_fail_open", False))

        # Optional: master speaker filter (window-based gating, dynamic mode only).
        sf = config.get("speaker_filter", {}) or {}
        self.speaker_filter = SpeakerFilterConfig(
            enabled=self._as_bool(sf.get("enabled", False)),
            window_ms=max(200, int(sf.get("window_ms", 600) or 600)),
            hop_ms=max(80, int(sf.get("hop_ms", 200) or 200)),
            edge_padding_ms=max(0, int(sf.get("edge_padding_ms", 200) or 200)),
            min_match_ratio=float(sf.get("min_match_ratio", 0.35) or 0.35),
            min_longest_match_ms=max(
                0, int(sf.get("min_longest_match_ms", 800) or 800)
            ),
            max_middle_interference_ratio=float(
                sf.get("max_middle_interference_ratio", 0.2) or 0.2
            ),
            reject_prompt=str(sf.get("reject_prompt", "") or ""),
            debug_log=self._as_bool(sf.get("debug_log", False)),
        )

        # Runtime state.
        self._dynamic_registered = False
        self._dynamic_registered_samples = 0
        self._dynamic_cleanup_done = False
        self._dynamic_lock = asyncio.Lock()

        # API
        self.base_url: Optional[str] = None
        self.api_url: Optional[str] = None
        self.identify_url: Optional[str] = None
        self.register_url: Optional[str] = None
        self.delete_url_prefix: Optional[str] = None
        self.api_key: Optional[str] = None
        self.speaker_ids = []
        self.enabled = False

        if not self.feature_enabled:
            logger.bind(tag=TAG).info("声纹识别总开关已关闭")
            return
        if not self.original_url:
            logger.bind(tag=TAG).warning("声纹识别URL未配置，声纹识别将被禁用")
            return

        parsed_url = urlparse(self.original_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.identify_url = f"{self.base_url}/voiceprint/identify"
        self.register_url = f"{self.base_url}/voiceprint/register"
        self.delete_url_prefix = f"{self.base_url}/voiceprint"
        self.api_url = self.identify_url

        query_params = parse_qs(parsed_url.query or "")
        self.api_key = query_params.get("key", [""])[0]
        if not self.api_key:
            logger.bind(tag=TAG).error("Missing `key` in voiceprint URL, disable voiceprint.")
            return

        # 静态模式：提取 speaker_ids；动态模式不依赖预配置 speakers
        if not self.dynamic_mode:
            for speaker_str in self.speakers:
                try:
                    parts = speaker_str.split(",", 2)
                    if len(parts) >= 1:
                        speaker_id = parts[0].strip()
                        if speaker_id:
                            self.speaker_ids.append(speaker_id)
                except Exception:
                    continue

            if not self.speaker_ids:
                logger.bind(tag=TAG).warning("未配置有效的说话人，声纹识别将被禁用")
                return

        if self._check_server_health():
            self.enabled = True
            if self.dynamic_mode:
                logger.bind(tag=TAG).info(
                    "声纹识别已启用（动态模式）: "
                    f"register={self.register_url}, identify={self.identify_url}, "
                    f"speaker_id={self.dynamic_master_speaker_id}, "
                    f"required_samples={self.dynamic_registration_required_samples}, "
                    f"threshold={self.similarity_threshold}"
                )
            else:
                logger.bind(tag=TAG).info(
                    "声纹识别已启用（静态模式）: "
                    f"identify={self.identify_url}, "
                    f"speakers={len(self.speaker_ids)}, "
                    f"threshold={self.similarity_threshold}"
                )
        else:
            logger.bind(tag=TAG).warning(
                f"声纹识别服务不可用，声纹识别已禁用: {self.api_url}"
            )

    def restore_dynamic_registration(self, reason: str = "") -> bool:
        """Reuse a previously enrolled master speaker after app restart."""
        if not self.enabled or not self.dynamic_mode:
            return False

        self._dynamic_registered = True
        self._dynamic_registered_samples = max(
            self._dynamic_registered_samples,
            self.dynamic_registration_required_samples,
        )
        logger.bind(tag=TAG).info(
            "dynamic voiceprint registration restored"
            + (f", reason={reason}" if reason else "")
        )
        return True

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sanitize_scope(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        sanitized = []
        for char in text:
            if char.isalnum() or char in {"_", "-"}:
                sanitized.append(char)
            else:
                sanitized.append("_")
        scope = "".join(sanitized).strip("_")
        return scope[:96]

    @classmethod
    def _build_dynamic_master_speaker_id(
        cls, base_speaker_id: Any, runtime_scope: Any
    ) -> str:
        base = cls._sanitize_scope(base_speaker_id) or "master_speaker"
        scope = cls._sanitize_scope(runtime_scope)
        if not scope:
            return base
        return f"{base}__{scope}"

    def _collect_score_candidates(self, body: Any) -> List[float]:
        """Extract score candidates only from whitelisted fields."""
        if not isinstance(body, dict):
            return []

        scores: List[float] = []

        def append_if_number(value: Any):
            num = self._safe_float(value)
            if num is not None:
                scores.append(num)

        def append_from_list(values: Any):
            if not isinstance(values, (list, tuple)):
                return
            for item in values:
                if isinstance(item, dict):
                    append_if_number(item.get("score"))
                    append_if_number(item.get("similarity"))
                else:
                    append_if_number(item)

        append_if_number(body.get("score"))
        append_if_number(body.get("similarity"))

        append_from_list(body.get("scores"))
        append_from_list(body.get("score_list"))
        append_from_list(body.get("similarities"))

        append_from_list(body.get("details"))
        append_from_list(body.get("score_details"))
        append_from_list(body.get("matches"))
        append_from_list(body.get("results"))

        return scores

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _parse_speakers(self) -> Dict[str, Dict[str, str]]:
        """Parse speaker config."""
        speaker_map = {}
        for speaker_str in self.speakers:
            try:
                parts = speaker_str.split(",", 2)
                if len(parts) >= 3:
                    speaker_id, name, description = (
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip(),
                    )
                    speaker_map[speaker_id] = {
                        "name": name,
                        "description": description,
                    }
            except Exception as e:
                logger.bind(tag=TAG).warning(
                    f"解析说话人配置失败: {speaker_str}, 错误: {e}"
                )
        return speaker_map

    def _check_server_health(self) -> bool:
        """Check voiceprint service health."""
        if not self.base_url or not self.api_key:
            return False

        cache_key = f"{self.base_url}:{self.api_key}"
        cached_result = cache_manager.get(CacheType.VOICEPRINT_HEALTH, cache_key)
        if cached_result is not None:
            logger.bind(tag=TAG).debug(f"使用缓存的健康状态: {cached_result}")
            return cached_result

        logger.bind(tag=TAG).info("Running voiceprint health check")
        is_healthy = False
        try:
            health_url = f"{self.base_url}/voiceprint/health?key={self.api_key}"
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                result = response.json()
                is_healthy = result.get("status") == "healthy"
                if is_healthy:
                    logger.bind(tag=TAG).info("声纹识别服务健康检查通过")
                else:
                    logger.bind(tag=TAG).warning(f"声纹服务状态异常: {result}")
            else:
                logger.bind(tag=TAG).warning(
                    f"声纹服务健康检查失败: HTTP {response.status_code}"
                )
        except requests.exceptions.ConnectTimeout:
            logger.bind(tag=TAG).warning("声纹服务连接超时")
        except requests.exceptions.ConnectionError:
            logger.bind(tag=TAG).warning("Voiceprint server connection refused")
        except Exception as e:
            logger.bind(tag=TAG).warning(f"声纹服务健康检查异常: {e}")

        cache_manager.set(CacheType.VOICEPRINT_HEALTH, cache_key, is_healthy)
        logger.bind(tag=TAG).info(f"健康检查结果已缓存: {is_healthy}")
        return is_healthy

    async def evaluate_voiceprint(
        self, audio_data: bytes, session_id: str
    ) -> Dict[str, Any]:
        """Internal method."""
        decision: Dict[str, Any] = {
            "enabled": self.enabled,
            "mode": "dynamic" if self.dynamic_mode else "static",
            "status": "disabled",
            "allow_chat": True,
            "speaker_name": None,
            "score": None,
            "need_register_prompt": False,
            "register_prompt_text": "",
            "reject_prompt_text": "",
            "reason": "",
        }

        if not self.enabled or not self.api_key:
            decision["status"] = "disabled"
            decision["reason"] = "voiceprint disabled or not configured"
            return decision

        if self.dynamic_mode:
            return await self._evaluate_dynamic(audio_data, session_id, decision)
        return await self._evaluate_static(audio_data, session_id, decision)

    async def _evaluate_dynamic(
        self, audio_data: bytes, session_id: str, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal method."""
        async with self._dynamic_lock:
            # Stage 1: auto-registration.
            if not self._dynamic_registered:
                register_ok = await self._register_master(audio_data, session_id)
                if not register_ok:
                    decision["status"] = "register_error"
                    decision["allow_chat"] = self.dynamic_fail_open
                    decision["reason"] = "dynamic register failed"
                    return decision

                self._dynamic_registered_samples += 1
                if self._dynamic_registered_samples < self.dynamic_registration_required_samples:
                    decision["status"] = "registering"
                    decision["allow_chat"] = False
                    decision["reason"] = (
                        f"collecting samples "
                        f"{self._dynamic_registered_samples}/"
                        f"{self.dynamic_registration_required_samples}"
                    )
                    return decision

                self._dynamic_registered = True
                decision["status"] = "registered"
                decision["allow_chat"] = False
                decision["speaker_name"] = self.dynamic_master_name
                decision["need_register_prompt"] = True
                decision["register_prompt_text"] = self.dynamic_registration_reply
                decision["reason"] = "dynamic register completed"
                return decision

            # Stage 2: verify master speaker first, and allow directly when score meets threshold.
            identify = await self._identify_by_speaker_ids(
                audio_data, [self.dynamic_master_speaker_id]
            )
            if not identify["ok"]:
                decision["status"] = "verify_error"
                decision["allow_chat"] = self.dynamic_fail_open
                decision["reason"] = identify["reason"]
                return decision

            score = float(identify["score"] or 0.0)
            speaker_id = identify["speaker_id"]
            decision["score"] = score

            if (
                speaker_id == self.dynamic_master_speaker_id
                and score >= self.similarity_threshold
            ):
                decision["status"] = "accepted"
                decision["allow_chat"] = True
                decision["speaker_name"] = self.dynamic_master_name
                decision["reason"] = "voiceprint matched"
                return decision

            # Stage 2b: low-score/edge case secondary filter (only when not already accepted).
            # Keep the existing filter behavior here to avoid false positives, but it can no longer
            # override a valid threshold match from full utterance.
            if getattr(self, "speaker_filter", None) and self.speaker_filter.enabled:
                filter_result = await self._apply_speaker_filter(
                    audio_data, session_id, self.dynamic_master_speaker_id
                )
                if not filter_result.get("ok", False):
                    decision["status"] = "speaker_filter_reject"
                    decision["allow_chat"] = False
                    decision["speaker_name"] = ""
                    decision["reason"] = filter_result.get(
                        "reason", "speaker filter rejected"
                    )
                    decision["reject_prompt_text"] = (
                        self.speaker_filter.reject_prompt
                        or decision.get("reject_prompt_text")
                        or ""
                    )
                    if self.speaker_filter.debug_log:
                        decision["speaker_filter"] = filter_result
                    return decision

                wav_for_verify = filter_result.get("wav") or audio_data
                if self.speaker_filter.debug_log:
                    decision["speaker_filter"] = filter_result

                # Retry verify with filtered segment for cleaner master speech when needed.
                filtered_identify = await self._identify_by_speaker_ids(
                    wav_for_verify, [self.dynamic_master_speaker_id], log_each=False
                )
                if filtered_identify.get("ok"):
                    filtered_score = float(filtered_identify.get("score") or 0.0)
                    filtered_speaker_id = filtered_identify.get("speaker_id")
                    decision["score"] = max(score, filtered_score)
                    if (
                        filtered_speaker_id == self.dynamic_master_speaker_id
                        and filtered_score >= self.similarity_threshold
                    ):
                        decision["status"] = "accepted"
                        decision["allow_chat"] = True
                        decision["speaker_name"] = self.dynamic_master_name
                        decision["reason"] = "voiceprint matched after speaker_filter"
                        return decision

            decision["status"] = "rejected"
            decision["allow_chat"] = False
            decision["speaker_name"] = "unknown_speaker"
            decision["reject_prompt_text"] = self.dynamic_reject_reply
            decision["reason"] = (
                f"voiceprint mismatch, speaker_id={speaker_id}, score={score:.3f}, "
                f"threshold={self.similarity_threshold}"
            )
            return decision

    async def _evaluate_static(
        self, audio_data: bytes, session_id: str, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal method."""
        identify = await self._identify_by_speaker_ids(audio_data, self.speaker_ids)
        if not identify["ok"]:
            decision["status"] = "identify_error"
            decision["allow_chat"] = True
            decision["reason"] = identify["reason"]
            return decision

        score = float(identify["score"] or 0.0)
        speaker_id = identify["speaker_id"]
        decision["score"] = score

        if score < self.similarity_threshold:
            decision["status"] = "unknown"
            decision["allow_chat"] = True
            decision["speaker_name"] = "unknown_speaker"
            decision["reason"] = (
                f"score below threshold: {score:.3f} < {self.similarity_threshold}"
            )
            return decision

        if speaker_id and speaker_id in self.speaker_map:
            decision["status"] = "accepted"
            decision["allow_chat"] = True
            decision["speaker_name"] = self.speaker_map[speaker_id]["name"]
            decision["reason"] = "speaker recognized"
            return decision

        decision["status"] = "unknown"
        decision["allow_chat"] = True
        decision["speaker_name"] = "unknown_speaker"
        decision["reason"] = f"unknown speaker id: {speaker_id}"
        return decision

    async def _register_master(self, audio_data: bytes, session_id: str) -> bool:
        """Register `master_speaker` in dynamic mode."""
        if not self.register_url:
            logger.bind(tag=TAG).error("register_url not configured")
            return False

        headers = self._build_headers()
        data = aiohttp.FormData()
        data.add_field("speaker_id", self.dynamic_master_speaker_id)
        data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
        timeout = aiohttp.ClientTimeout(total=10)
        begin = time.monotonic()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.register_url, headers=headers, data=data
                ) as response:
                    if response.status in (200, 201):
                        elapsed = time.monotonic() - begin
                        logger.bind(tag=TAG).info(
                            "动态声纹注册成功: "
                            f"speaker_id={self.dynamic_master_speaker_id}, "
                            f"sample={self._dynamic_registered_samples + 1}/"
                            f"{self.dynamic_registration_required_samples}, "
                            f"session={session_id}, elapsed={elapsed:.3f}s"
                        )
                        return True
                    body = await response.text()
                    logger.bind(tag=TAG).error(
                        f"动态声纹注册失败: HTTP {response.status}, body={body[:200]}"
                    )
                    return False
        except asyncio.TimeoutError:
            logger.bind(tag=TAG).error("dynamic voiceprint register timeout")
            return False
        except Exception as e:
            logger.bind(tag=TAG).error(f"动态声纹注册异常: {e}")
            return False

    async def _identify_by_speaker_ids(
        self, audio_data: bytes, speaker_ids: list, *, log_each: bool = True
    ) -> Dict[str, Any]:
        """Internal method."""
        result = {
            "ok": False,
            "speaker_id": None,
            "score": 0.0,
            "reason": "",
        }
        if not self.identify_url:
            result["reason"] = "identify_url not configured"
            return result
        if not speaker_ids:
            result["reason"] = "speaker_ids is empty"
            return result

        headers = self._build_headers()
        data = aiohttp.FormData()
        data.add_field("speaker_ids", ",".join(speaker_ids))
        data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
        timeout = aiohttp.ClientTimeout(total=10)
        begin = time.monotonic()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.identify_url, headers=headers, data=data
                ) as response:
                    if response.status == 200:
                        body = await response.json()
                        result["ok"] = True
                        result["speaker_id"] = body.get("speaker_id")
                        raw_score = self._safe_float(body.get("score"))
                        score_candidates = self._collect_score_candidates(body)
                        if score_candidates:
                            result["score"] = max(score_candidates)
                            if (
                                raw_score is not None
                                and abs(result["score"] - raw_score) > 1e-9
                            ):
                                if log_each:
                                    logger.bind(tag=TAG).debug(
                                        "identify score aggregated by max: "
                                        f"raw={raw_score:.3f}, selected={result['score']:.3f}, "
                                        f"candidates={len(score_candidates)}"
                                    )
                        else:
                            result["score"] = 0.0
                        elapsed = time.monotonic() - begin
                        if log_each:
                            logger.bind(tag=TAG).info(f"声纹识别耗时: {elapsed:.3f}s")
                        return result
                    text = await response.text()
                    result["reason"] = (
                        f"identify http {response.status}, body={text[:200]}"
                    )
                    logger.bind(tag=TAG).error(f"声纹识别API错误: {result['reason']}")
                    return result
        except asyncio.TimeoutError:
            result["reason"] = "identify timeout"
            logger.bind(tag=TAG).error("声纹识别超时")
            return result
        except Exception as e:
            result["reason"] = f"identify exception: {e}"
            logger.bind(tag=TAG).error(f"声纹识别失败: {e}")
            return result

    async def identify_speaker(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """Compatibility method: only return speaker_name."""
        decision = await self.evaluate_voiceprint(audio_data, session_id)
        return decision.get("speaker_name")

    async def cleanup_dynamic_voiceprint(self, session_id: str = "") -> bool:
        """Delete the current dynamic-session voiceprint only.

        Safety boundary:
        - dynamic mode only
        - current connection's derived ``master_speaker_id`` only
        - no-op when this connection has not enrolled any sample
        """
        if not self.enabled or not self.dynamic_mode:
            return False
        if self._dynamic_cleanup_done:
            return True
        if not self.delete_url_prefix or not self.dynamic_master_speaker_id:
            return False
        if self._dynamic_registered_samples <= 0 and not self._dynamic_registered:
            self._dynamic_cleanup_done = True
            return True

        delete_url = f"{self.delete_url_prefix}/{self.dynamic_master_speaker_id}"
        timeout = aiohttp.ClientTimeout(total=10)
        begin = time.monotonic()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.delete(
                    delete_url,
                    headers=self._build_headers(),
                ) as response:
                    elapsed = time.monotonic() - begin
                    if response.status in (200, 204, 404):
                        self._dynamic_cleanup_done = True
                        self._dynamic_registered = False
                        self._dynamic_registered_samples = 0
                        logger.bind(tag=TAG).info(
                            "dynamic voiceprint cleanup finished: "
                            f"speaker_id={self.dynamic_master_speaker_id}, "
                            f"session={session_id}, http={response.status}, "
                            f"elapsed={elapsed:.3f}s"
                        )
                        return True

                    body = await response.text()
                    logger.bind(tag=TAG).error(
                        "dynamic voiceprint cleanup failed: "
                        f"speaker_id={self.dynamic_master_speaker_id}, "
                        f"session={session_id}, http={response.status}, "
                        f"body={body[:200]}"
                    )
                    return False
        except asyncio.TimeoutError:
            logger.bind(tag=TAG).error(
                "dynamic voiceprint cleanup timeout: "
                f"speaker_id={self.dynamic_master_speaker_id}, session={session_id}"
            )
            return False
        except Exception as e:
            logger.bind(tag=TAG).error(
                "dynamic voiceprint cleanup exception: "
                f"speaker_id={self.dynamic_master_speaker_id}, session={session_id}, error={e}"
            )
            return False

    async def _identify_by_speaker_ids_quiet(
        self, audio_data: bytes, speaker_ids: list
    ) -> Dict[str, Any]:
        """
        Same as _identify_by_speaker_ids, but without info/debug logs.
        Used by speaker_filter window scanning to avoid log flooding.
        """
        result = {
            "ok": False,
            "speaker_id": None,
            "score": 0.0,
            "reason": "",
        }
        if not self.identify_url:
            result["reason"] = "identify_url not configured"
            return result
        if not speaker_ids:
            result["reason"] = "speaker_ids is empty"
            return result

        headers = self._build_headers()
        data = aiohttp.FormData()
        data.add_field("speaker_ids", ",".join(speaker_ids))
        data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.identify_url, headers=headers, data=data
                ) as response:
                    if response.status == 200:
                        body = await response.json()
                        result["ok"] = True
                        result["speaker_id"] = body.get("speaker_id")
                        score_candidates = self._collect_score_candidates(body)
                        result["score"] = max(score_candidates) if score_candidates else 0.0
                        return result
                    text = await response.text()
                    result["reason"] = f"identify http {response.status}, body={text[:200]}"
                    return result
        except asyncio.TimeoutError:
            result["reason"] = "identify timeout"
            return result
        except Exception as e:
            result["reason"] = f"identify exception: {e}"
            return result

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> Optional[Dict[str, Any]]:
        try:
            with wave.open(BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.getnframes()
                pcm = wf.readframes(frames)
            return {
                "pcm": pcm,
                "channels": channels,
                "sampwidth": sampwidth,
                "sample_rate": sample_rate,
            }
        except Exception:
            return None

    @staticmethod
    def _pcm_to_wav_bytes(pcm_bytes: bytes, *, sample_rate: int) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    async def _apply_speaker_filter(
        self, wav_bytes: bytes, session_id: str, master_speaker_id: str
    ) -> Dict[str, Any]:
        """
        Window-based gating for dynamic master speaker.

        Returns:
        - ok: bool
        - reason: str
        - wav: cropped wav bytes on success
        - stats: optional diagnostic numbers
        """
        sf = getattr(self, "speaker_filter", None)
        if not sf or not sf.enabled:
            return {"ok": True, "wav": wav_bytes, "reason": ""}

        parsed = self._wav_to_pcm(wav_bytes)
        if not parsed:
            return {"ok": True, "wav": wav_bytes, "reason": "speaker_filter: wav parse failed"}

        if parsed["channels"] != 1 or parsed["sampwidth"] != 2:
            return {
                "ok": True,
                "wav": wav_bytes,
                "reason": "speaker_filter: unsupported wav format",
            }

        pcm = parsed["pcm"] or b""
        sr = int(parsed["sample_rate"] or 16000)
        if not pcm or sr <= 0:
            return {"ok": True, "wav": wav_bytes, "reason": "speaker_filter: empty audio"}

        total_samples = len(pcm) // 2
        total_ms = (total_samples * 1000.0) / float(sr)

        window_samples = max(1, int(sr * sf.window_ms / 1000.0))
        hop_samples = max(1, int(sr * sf.hop_ms / 1000.0))

        if total_samples <= window_samples:
            starts = [0]
        else:
            starts = list(range(0, total_samples - window_samples + 1, hop_samples))

        # Guard API pressure on long utterances.
        max_windows = 30
        if len(starts) > max_windows:
            stride = int(math.ceil(len(starts) / float(max_windows)))
            starts = starts[:: max(1, stride)]

        sem = asyncio.Semaphore(3)

        async def score_window(start_sample: int) -> Dict[str, Any]:
            end = min(total_samples, start_sample + window_samples)
            win_pcm = pcm[start_sample * 2 : end * 2]
            win_wav = self._pcm_to_wav_bytes(win_pcm, sample_rate=sr)
            async with sem:
                return await self._identify_by_speaker_ids_quiet(
                    win_wav, [master_speaker_id]
                )

        results = await asyncio.gather(
            *(score_window(s) for s in starts), return_exceptions=True
        )

        matched = []
        ok_count = 0
        for r in results:
            if isinstance(r, Exception) or not isinstance(r, dict):
                matched.append(False)
                continue
            ok_count += 1 if r.get("ok") else 0
            score = float(r.get("score") or 0.0)
            is_match = (
                bool(r.get("ok"))
                and r.get("speaker_id") == master_speaker_id
                and score >= float(self.similarity_threshold)
            )
            matched.append(bool(is_match))

        n = len(matched) or 1
        match_ratio = float(sum(1 for m in matched if m)) / float(n)

        # Too many identify failures -> bypass to avoid hard false rejects.
        if ok_count < max(1, int(0.5 * len(matched))):
            return {
                "ok": True,
                "wav": wav_bytes,
                "reason": "speaker_filter: bypass due to identify errors",
                "stats": {
                    "windows": len(starts),
                    "ok_windows": ok_count,
                    "match_ratio": match_ratio,
                },
            }

        best_len = 0
        best_start_idx = -1
        cur_len = 0
        cur_start = 0
        for i, m in enumerate(matched):
            if m:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start_idx = cur_start
            else:
                cur_len = 0

        longest_match_ms = 0
        if best_len > 0:
            longest_match_ms = int((best_len - 1) * sf.hop_ms + sf.window_ms)

        first_match = -1
        last_match = -1
        for i, m in enumerate(matched):
            if m:
                first_match = i if first_match < 0 else first_match
                last_match = i
        middle_interference_ratio = 0.0
        if first_match >= 0 and last_match >= first_match:
            span = matched[first_match : last_match + 1]
            if span:
                middle_interference_ratio = float(sum(1 for x in span if not x)) / float(
                    len(span)
                )

        stats = {
            "windows": len(starts),
            "ok_windows": ok_count,
            "match_ratio": match_ratio,
            "longest_match_ms": longest_match_ms,
            "middle_interference_ratio": middle_interference_ratio,
            "threshold": float(self.similarity_threshold),
        }

        if match_ratio < sf.min_match_ratio:
            return {
                "ok": False,
                "reason": f"speaker_filter: match_ratio {match_ratio:.3f} < {sf.min_match_ratio}",
                "stats": stats,
            }
        if longest_match_ms < sf.min_longest_match_ms:
            return {
                "ok": False,
                "reason": f"speaker_filter: longest_match_ms {longest_match_ms} < {sf.min_longest_match_ms}",
                "stats": stats,
            }
        if middle_interference_ratio > sf.max_middle_interference_ratio:
            return {
                "ok": False,
                "reason": (
                    "speaker_filter: middle_interference_ratio "
                    f"{middle_interference_ratio:.3f} > {sf.max_middle_interference_ratio}"
                ),
                "stats": stats,
            }

        if best_start_idx < 0:
            return {
                "ok": False,
                "reason": "speaker_filter: no matched windows",
                "stats": stats,
            }

        run_start_ms = float(best_start_idx) * float(sf.hop_ms)
        run_end_ms = run_start_ms + float(longest_match_ms)
        crop_start_ms = max(0.0, run_start_ms - float(sf.edge_padding_ms))
        crop_end_ms = min(float(total_ms), run_end_ms + float(sf.edge_padding_ms))

        crop_start_sample = int(crop_start_ms * float(sr) / 1000.0)
        crop_end_sample = int(crop_end_ms * float(sr) / 1000.0)
        crop_start_sample = max(0, min(total_samples, crop_start_sample))
        crop_end_sample = max(crop_start_sample, min(total_samples, crop_end_sample))

        cropped_pcm = pcm[crop_start_sample * 2 : crop_end_sample * 2]
        cropped_wav = self._pcm_to_wav_bytes(cropped_pcm, sample_rate=sr)

        stats.update({"crop_start_ms": crop_start_ms, "crop_end_ms": crop_end_ms})

        if sf.debug_log:
            logger.bind(tag=TAG).info(
                "speaker_filter accepted: "
                f"windows={stats['windows']}, match_ratio={match_ratio:.3f}, "
                f"longest_match_ms={longest_match_ms}, "
                f"middle_interference_ratio={middle_interference_ratio:.3f}, "
                f"crop_ms=({crop_start_ms:.0f},{crop_end_ms:.0f}) session={session_id}"
            )

        return {"ok": True, "wav": cropped_wav, "stats": stats}
