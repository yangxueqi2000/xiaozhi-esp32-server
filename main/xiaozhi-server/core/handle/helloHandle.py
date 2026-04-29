import asyncio
import json
import os
import random
import time
import uuid

from core.handle.sendAudioHandle import send_tts_message, sendAudioMessage
from core.providers.tools.device_mcp import MCPClient, send_mcp_initialize_message
from core.providers.tts.dto.dto import SentenceType
from core.session import resolve_or_create_session_binding
from core.utils.dialogue import Message
from core.utils.util import (
    audio_to_data,
    opus_datas_to_wav_bytes,
    remove_punctuation_and_length,
)
from core.utils.wakeup_word import WakeupWordsConfig

TAG = __name__

WAKEUP_CONFIG = {
    "refresh_time": 10,
    "responses": [
        "我一直都在呢，您请说。",
        "在的呢，请随时吩咐我。",
        "来啦来啦，请告诉我吧。",
        "您请说，我正听着。",
        "请您讲话，我准备好了。",
        "请您说出指令吧。",
        "我认真听着呢，请讲。",
        "请问您需要什么帮助？",
        "我在这里，等候您的指令。",
    ],
}

WAKEUP_FALLBACK_FILE = "config/assets/wakeup_words_short.wav"
WAKEUP_FALLBACK_TEXT = "我在这里哦！"

wakeup_words_config = WakeupWordsConfig()
_wakeup_response_lock = asyncio.Lock()


def _get_current_wakeup_voice(conn) -> str:
    voice = str(getattr(getattr(conn, "tts", None), "voice", "") or "").strip()
    return voice or "default"


def _get_tts_sample_rate(conn) -> int:
    tts = getattr(conn, "tts", None)
    for attr_name in ("stream_sample_rate", "sample_rate"):
        value = getattr(tts, attr_name, None)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 16000


def _build_default_wakeup_response() -> dict:
    return {
        "voice": "default",
        "file_path": WAKEUP_FALLBACK_FILE,
        "time": 0,
        "text": WAKEUP_FALLBACK_TEXT,
    }


async def _wait_for_tts_ready(conn, timeout_seconds: float) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if getattr(conn, "tts", None):
            return True
        await asyncio.sleep(0.1)
    return bool(getattr(conn, "tts", None))


async def _generate_wakeup_response(conn, voice: str) -> bool:
    result = random.choice(WAKEUP_CONFIG["responses"])
    if not result:
        return False

    tts_result = await asyncio.to_thread(conn.tts.to_tts, result)
    if not tts_result:
        return False

    sample_rate = _get_tts_sample_rate(conn)
    wav_bytes = opus_datas_to_wav_bytes(tts_result, sample_rate=sample_rate)
    final_path = wakeup_words_config.generate_file_path(voice)
    temp_path = f"{final_path}.{uuid.uuid4().hex}.tmp"

    try:
        with open(temp_path, "wb") as file_obj:
            file_obj.write(wav_bytes)
        os.replace(temp_path, final_path)
        wakeup_words_config.update_wakeup_response(voice, final_path, result)
        conn.logger.bind(tag=TAG).info(
            "wakeup response cache refreshed: "
            f"voice={voice}, file_path={final_path}, sample_rate={sample_rate}, text={result}"
        )
        return True
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def ensureWakeupWordsResponseReady(
    conn,
    *,
    force: bool = False,
    timeout_seconds: float = 3,
) -> bool:
    if not conn.config.get("enable_wakeup_words_response_cache", False):
        return False

    if not await _wait_for_tts_ready(conn, timeout_seconds):
        return False

    voice = _get_current_wakeup_voice(conn)
    if not force:
        cached_response = wakeup_words_config.get_wakeup_response(voice)
        if cached_response and cached_response.get("file_path"):
            return True

    async with _wakeup_response_lock:
        if not getattr(conn, "tts", None):
            return False

        if not force:
            cached_response = wakeup_words_config.get_wakeup_response(voice)
            if cached_response and cached_response.get("file_path"):
                return True

        try:
            return await _generate_wakeup_response(conn, voice)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"failed to refresh wakeup response cache for voice={voice}: {exc}"
            )
            return False


async def handleHelloMessage(conn, msg_json):
    user_id = str(msg_json.get("user_id", "")).strip()
    if not user_id:
        user_id = str(
            conn.config.get("session_registry", {}).get("default_user_id", "test")
        ).strip()
    if not user_id:
        user_id = "test"
    conn.user_id = user_id

    if conn.device_id:
        try:
            binding = await resolve_or_create_session_binding(
                conn.config, conn.device_id, conn.user_id
            )
            conn.chat_session_id = binding["chat_session_id"]
            conn.model_session_key = binding["model_session_key"]
            conn.logger.bind(tag=TAG).info(
                "resolved chat/model session binding: "
                f"device_id={conn.device_id}, user_id={conn.user_id}, "
                f"chat_session_id={conn.chat_session_id}, "
                f"model_session_key={conn.model_session_key}, "
                f"source={binding.get('source')}, created={binding.get('created')}"
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"resolve chat/model session binding failed, fallback transport session: {exc}"
            )
            conn.chat_session_id = conn.session_id
            conn.model_session_key = conn.session_id
    else:
        conn.logger.bind(tag=TAG).warning(
            "device_id is missing in hello flow, fallback transport session"
        )
        conn.chat_session_id = conn.session_id
        conn.model_session_key = conn.session_id

    audio_params = msg_json.get("audio_params")
    if audio_params:
        audio_format = audio_params.get("format")
        conn.logger.bind(tag=TAG).debug(f"客户端音频格式: {audio_format}")
        conn.audio_format = audio_format
        conn.welcome_msg["audio_params"] = audio_params

    features = msg_json.get("features")
    if features:
        conn.logger.bind(tag=TAG).debug(f"客户端特性: {features}")
        conn.features = features
        if features.get("mcp"):
            conn.logger.bind(tag=TAG).debug("客户端支持MCP")
            conn.mcp_client = MCPClient()
            asyncio.create_task(send_mcp_initialize_message(conn))

    if hasattr(conn, "schedule_experiment_prewarm"):
        scheduled = conn.schedule_experiment_prewarm(trigger="hello")
        if scheduled:
            conn.logger.bind(tag=TAG).info(
                "scheduled experiment prewarm on hello: "
                f"device_id={conn.device_id}, session_id={conn.session_id}"
            )

    await conn.websocket.send(json.dumps(conn.welcome_msg))

    if conn.config.get("enable_wakeup_words_response_cache", False):
        conn.logger.bind(tag=TAG).info(
            "scheduled wakeup response cache warmup: "
            f"device_id={conn.device_id}, session_id={conn.session_id}"
        )
        asyncio.create_task(
            ensureWakeupWordsResponseReady(conn, timeout_seconds=8, force=False)
        )


async def checkWakeupWords(conn, text):
    if not conn.config.get("enable_wakeup_words_response_cache", False):
        return False

    if not await _wait_for_tts_ready(conn, 3):
        return False

    _, filtered_text = remove_punctuation_and_length(text)
    if filtered_text not in conn.config.get("wakeup_words"):
        return False

    conn.just_woken_up = True
    await send_tts_message(conn, "start")

    voice = _get_current_wakeup_voice(conn)
    response = wakeup_words_config.get_wakeup_response(voice)
    response_source = "voice_cache"

    if not response or not response.get("file_path"):
        await ensureWakeupWordsResponseReady(conn, timeout_seconds=0.1, force=False)
        response = wakeup_words_config.get_wakeup_response(voice)
        if response and response.get("file_path"):
            response_source = "voice_cache_generated"

    if not response or not response.get("file_path"):
        response = _build_default_wakeup_response()
        response_source = "default_fallback"
        asyncio.create_task(
            ensureWakeupWordsResponseReady(conn, timeout_seconds=8, force=False)
        )

    opus_packets = await audio_to_data(response.get("file_path"), use_cache=False)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)

    conn.logger.bind(tag=TAG).info(
        "playing wakeup response: "
        f"source={response_source}, voice={response.get('voice')}, "
        f"file_path={response.get('file_path')}, text={response.get('text')}"
    )
    await sendAudioMessage(conn, SentenceType.FIRST, opus_packets, response.get("text"))
    await sendAudioMessage(conn, SentenceType.LAST, [], None)

    conn.dialogue.put(Message(role="assistant", content=response.get("text")))

    if response_source != "default_fallback":
        response_time = float(response.get("time", 0) or 0)
        if time.time() - response_time > WAKEUP_CONFIG["refresh_time"]:
            asyncio.create_task(
                ensureWakeupWordsResponseReady(conn, timeout_seconds=0.1, force=True)
            )
    return True
