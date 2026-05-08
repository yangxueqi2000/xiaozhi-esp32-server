import time
import numpy as np
import torch
import opuslib_next
from config.logger import setup_logging
from core.providers.vad.base import VADProviderBase

TAG = __name__
logger = setup_logging()


def _parse_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _parse_keywords(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip().lower() for part in value if str(part).strip()]
    return list(default)


class VADProvider(VADProviderBase):
    def __init__(self, config):
        logger.bind(tag=TAG).info("SileroVAD", config)
        self.model, _ = torch.hub.load(
            repo_or_dir=config["model_dir"],
            source="local",
            model="silero_vad",
            force_reload=False,
        )

        self.decoder = opuslib_next.Decoder(16000, 1)

        # 处理空字符串的情况
        threshold = config.get("threshold", "0.5")
        threshold_low = config.get("threshold_low", "0.2")
        min_silence_duration_ms = config.get("min_silence_duration_ms", "1000")
        data_report_min_silence_duration_ms = config.get(
            "data_report_min_silence_duration_ms", min_silence_duration_ms
        )

        self.vad_threshold = float(threshold) if threshold else 0.5
        self.vad_threshold_low = float(threshold_low) if threshold_low else 0.2

        self.silence_threshold_ms = _parse_int(min_silence_duration_ms, 1000)
        self.data_report_silence_threshold_ms = _parse_int(
            data_report_min_silence_duration_ms,
            self.silence_threshold_ms,
        )
        self.data_report_step_keywords = _parse_keywords(
            config.get("data_report_step_keywords"),
            (
                "observe",
                "observation",
                "record",
                "measurement",
                "kinetics",
                "tyndall",
                "data",
                "report",
            ),
        )

        # 至少要多少帧才算有语音
        self.frame_window_threshold = _parse_int(config.get("frame_window_threshold"), 3)

    def _step_signature(self, conn) -> str:
        parts = [
            getattr(conn, "experiment_current_step_id", ""),
            getattr(conn, "experiment_resume_latest_current_step_id", ""),
        ]
        for attr_name in ("experiment_current_step", "experiment_progress_summary"):
            value = getattr(conn, attr_name, None)
            if isinstance(value, dict):
                parts.extend(
                    str(value.get(key, "") or "")
                    for key in ("step_id", "id", "title", "description", "instruction")
                )
                step = value.get("step")
                if isinstance(step, dict):
                    parts.extend(
                        str(step.get(key, "") or "")
                        for key in ("id", "title", "description", "instruction")
                    )
                body = value.get("body")
                if isinstance(body, dict):
                    body_step = body.get("step") or body.get("current_step")
                    if isinstance(body_step, dict):
                        parts.extend(
                            str(body_step.get(key, "") or "")
                            for key in (
                                "step_id",
                                "id",
                                "title",
                                "description",
                                "instruction",
                            )
                        )
                    summary = body.get("summary")
                    if isinstance(summary, dict):
                        current_step = summary.get("current_step")
                        if isinstance(current_step, dict):
                            parts.extend(
                                str(current_step.get(key, "") or "")
                                for key in ("step_id", "id", "title")
                            )
            elif value:
                parts.append(str(value))
        return " ".join(str(part) for part in parts if part).lower()

    def _silence_threshold_ms_for_conn(self, conn) -> int:
        signature = self._step_signature(conn)
        if signature and any(
            keyword in signature for keyword in self.data_report_step_keywords
        ):
            return self.data_report_silence_threshold_ms
        return self.silence_threshold_ms

    def __del__(self):
        if hasattr(self, 'decoder') and self.decoder is not None:
            try:
                del self.decoder
            except Exception:
                pass

    def is_vad(self, conn, opus_packet):
        # 手动模式：直接返回True，不进行实时VAD检测，所有音频都缓存
        if conn.client_listen_mode == "manual":
            return True
            
        try:
            pcm_frame = self.decoder.decode(opus_packet, 960)
            return self.is_vad_pcm(conn, pcm_frame)
        except opuslib_next.OpusError as e:
            logger.bind(tag=TAG).info(f"解码错误: {e}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error processing audio packet: {e}")
        return False

    def is_vad_pcm(self, conn, pcm_frame: bytes) -> bool:
        """VAD on PCM frame (16-bit mono @16kHz)."""
        # 手动模式：直接返回True，不进行实时VAD检测
        if conn.client_listen_mode == "manual":
            return True
        if not pcm_frame:
            return False

        try:
            conn.client_audio_buffer.extend(pcm_frame)  # 将新数据加入缓冲区

            # 处理缓冲区中的完整帧（每次处理512采样点）
            client_have_voice = False
            while len(conn.client_audio_buffer) >= 512 * 2:
                # 提取前512个采样点（1024字节）
                chunk = conn.client_audio_buffer[: 512 * 2]
                conn.client_audio_buffer = conn.client_audio_buffer[512 * 2 :]

                # 转换为模型需要的张量格式
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                # 检测语音活动
                with torch.no_grad():
                    speech_prob = self.model(audio_tensor, 16000).item()

                # 双阈值判断
                if speech_prob >= self.vad_threshold:
                    is_voice = True
                elif speech_prob <= self.vad_threshold_low:
                    is_voice = False
                else:
                    is_voice = conn.last_is_voice

                # 声音没低于最低值则延续前一个状态，判断为有声音
                conn.last_is_voice = is_voice

                # 更新滑动窗口
                conn.client_voice_window.append(is_voice)
                client_have_voice = (
                    conn.client_voice_window.count(True) >= self.frame_window_threshold
                )

                # 如果之前有声音，但本次没有声音，且与上次有声音的时间差已经超过了静默阈值，则认为已经说完一句话
                if conn.client_have_voice and not client_have_voice:
                    stop_duration = time.time() * 1000 - conn.last_activity_time
                    if stop_duration >= self._silence_threshold_ms_for_conn(conn):
                        conn.client_voice_stop = True
                if client_have_voice:
                    conn.client_have_voice = True
                    conn.last_activity_time = time.time() * 1000

            return client_have_voice
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error processing PCM frame: {e}")
            return False
