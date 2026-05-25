import csv
import json
import math
import re
import uuid
import asyncio
import time
from pathlib import Path

import yaml
from core.utils.dialogue import Message
from core.providers.tts.dto.dto import ContentType
from core.handle.helloHandle import checkWakeupWords
from core.session import rotate_session_binding, save_experiment_session_binding
from plugins_func.register import Action, ActionResponse
from core.handle.sendAudioHandle import send_stt_message
from core.utils import experiment_resume, textUtils
from core.utils.util import remove_punctuation_and_length, sanitize_tool_name
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType
from core.providers.tools.device_mcp import call_mcp_tool
from core.providers.tools.server_mcp.payload_utils import (
    build_server_mcp_spoken_response,
    finalize_server_mcp_payload,
    sync_server_mcp_payload_state,
)

TAG = __name__


async def handle_user_intent(conn, text):
    # 棰勫鐞嗚緭鍏ユ枃鏈紝澶勭悊鍙兘鐨凧SON鏍煎紡
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                text = parsed_data["content"]  # 鎻愬彇content鐢ㄤ簬鎰忓浘鍒嗘瀽
                conn.current_speaker = parsed_data.get("speaker")  # 淇濈暀璇磋瘽浜轰俊鎭?
    except (json.JSONDecodeError, TypeError):
        pass

    # 妫€鏌ユ槸鍚︽湁鏄庣‘鐨勯€€鍑哄懡浠?
    _, filtered_text = remove_punctuation_and_length(text)
    if await check_direct_exit(conn, filtered_text):
        return True

    # 妫€鏌ユ槸鍚︽槸鍞ら啋璇?
    if await checkWakeupWords(conn, filtered_text):
        return True

    # Resume must synchronize the experiment graph before the assistant speaks.
    # Do not let a "continue previous experiment" request fall through to the
    # generic LLM path, where device-log context can make the dialogue continue
    # while the graph is still at the beginning.
    if _is_explicit_experiment_resume_request(filtered_text):
        try:
            return await _handle_explicit_experiment_resume_request(conn, text)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment explicit resume handler failed: {exc}"
            )
            return False

    # Keep photo and UV-Vis direct handlers ahead of the generic experiment fast
    # path so confirmations like "鍙互鎷嶇収" still follow the device shortcut.
    await _maybe_refresh_experiment_state_before_direct_handlers(
        conn,
        reason="before_direct_handlers",
    )
    _update_server_photo_confirmation_state(conn, filtered_text)

    photo_direct_handlers = (
        handle_pending_direct_photo_confirmation,
        handle_pending_server_photo_confirmation,
        handle_direct_photo_navigation_intent,
        handle_direct_photo_intent,
        handle_direct_uvvis_intent,
    )
    for handler in photo_direct_handlers:
        try:
            handled = await handler(conn, text, filtered_text)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"photo direct handler failed: handler={handler.__name__}, error={exc}"
            )
            continue
        if handled:
            return True

    try:
        handled = await handle_experiment_control_fast_intent(
            conn,
            text,
            filtered_text,
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment control fast path failed: {exc}"
        )
    else:
        if handled:
            return True

    try:
        handled = await handle_experiment_control_strict_graph_intent(
            conn,
            text,
            filtered_text,
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment control strict graph path failed: {exc}"
        )
    else:
        if handled:
            return True

    if conn.intent_type == "function_call":
        _maybe_stage_experiment_ready_guard_bypass_for_normal_turn(conn, filtered_text)
        # 浣跨敤鏀寔function calling鐨勮亰澶╂柟娉?涓嶅啀杩涜鎰忓浘鍒嗘瀽
        return False
    # 浣跨敤LLM杩涜鎰忓浘鍒嗘瀽
    intent_result = await analyze_intent_with_llm(conn, text)
    if not intent_result:
        return False
    # 浼氳瘽寮€濮嬫椂鐢熸垚sentence_id
    conn.sentence_id = str(uuid.uuid4().hex)
    if _assistant_waiting_for_step_start(conn) and _is_explicit_ready_to_start_reply(
        filtered_text
    ):
        textUtils.activate_experiment_ready_guard_bypass_for_current_sentence(
            conn,
            force=True,
        )
    # 澶勭悊鍚勭鎰忓浘
    return await process_intent_result(conn, intent_result, text)


async def check_direct_exit(conn, text):
    """妫€鏌ユ槸鍚︽湁鏄庣‘鐨勯€€鍑哄懡浠?""
    _, text = remove_punctuation_and_length(text)
    cmd_exit = conn.cmd_exit
    for cmd in cmd_exit:
        if text == cmd:
            conn.logger.bind(tag=TAG).info(f"璇嗗埆鍒版槑纭殑閫€鍑哄懡浠? {text}")
            await send_stt_message(conn, text)
            await conn.close()
            return True
    return False


async def analyze_intent_with_llm(conn, text):
    """浣跨敤LLM鍒嗘瀽鐢ㄦ埛鎰忓浘"""
    if not hasattr(conn, "intent") or not conn.intent:
        conn.logger.bind(tag=TAG).warning("鎰忓浘璇嗗埆鏈嶅姟鏈垵濮嬪寲")
        return None

    # 瀵硅瘽鍘嗗彶璁板綍
    dialogue = conn.dialogue
    try:
        intent_result = await conn.intent.detect_intent(conn, dialogue.dialogue, text)
        return intent_result
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"鎰忓浘璇嗗埆澶辫触: {str(e)}")

    return None


async def _maybe_refresh_experiment_state_before_direct_handlers(
    conn,
    *,
    reason: str = "",
):
    if not bool(getattr(conn, "_experiment_graph_refresh_required", False)):
        return

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return

    try:
        if hasattr(conn, "refresh_experiment_foreground_state"):
            refreshed = await conn.refresh_experiment_foreground_state(
                reason=reason or "before_direct_handlers"
            )
            if isinstance(refreshed, dict) and str(
                refreshed.get("current_step_id", "") or ""
            ).strip():
                conn._experiment_graph_refresh_required = False
            return

        step_meta = await _safe_refresh_experiment_step_cache(
            conn,
            session_id,
            reason=reason or "before_direct_handlers",
        )
        if str(step_meta.get("step_id", "") or "").strip() or _get_current_experiment_step_id(conn):
            conn._experiment_graph_refresh_required = False
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            "experiment direct-handler refresh bridge failed: "
            f"session_id={session_id}, reason={reason or 'unknown'}, error={exc}"
        )


def _normalize_text_for_match(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "")


def _contains_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _contains_match_token(text: str, words) -> bool:
    normalized = _normalize_text_for_match(text)
    if not normalized:
        return False

    ascii_chunks = re.findall(r"[a-z0-9_]+", normalized)
    ascii_parts = set()
    for chunk in ascii_chunks:
        ascii_parts.update(part for part in chunk.split("_") if part)

    for word in words:
        token = _normalize_text_for_match(str(word or ""))
        if not token:
            continue
        if re.fullmatch(r"[a-z0-9_]+", token):
            if token in ascii_parts:
                return True
            continue
        if token in normalized:
            return True
    return False


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _starts_with_any(text: str, words) -> bool:
    return any(text.startswith(w) for w in words)


def _ends_with_any(text: str, words) -> bool:
    return any(text.endswith(w) for w in words)


def _matches_any_pattern(text: str, patterns) -> bool:
    return any(pattern.fullmatch(text) for pattern in patterns)


def _looks_like_question_reply(text: str) -> bool:
    if not text:
        return False
    if text.endswith(("鍚?, "涔?, "鍢?, "鍛?)):
        return True
    question_tokens = (
        "鍙笉鍙互",
        "鑳戒笉鑳?,
        "琛屼笉琛?,
        "瑕佷笉瑕?,
        "鏄笉鏄?,
        "涓轰粈涔?,
        "鎬庝箞",
        "濡備綍",
    )
    return _contains_any(text, question_tokens)


def _experiment_result_body(payload):
    if isinstance(payload, dict):
        nested = payload.get("result")
        if isinstance(nested, dict):
            return nested
        return payload
    return {}


def _extract_experiment_result_message(payload) -> str:
    body = _experiment_result_body(payload)
    return str(body.get("message", "")).strip()


def _extract_experiment_step_meta(payload) -> dict:
    body = _experiment_result_body(payload)
    meta = {
        "step_id": "",
        "title": "",
        "instruction": "",
        "description": "",
        "safety": "",
        "tip": "",
    }

    step = body.get("step")
    if isinstance(step, dict):
        meta["step_id"] = str(step.get("id", "")).strip()
        meta["title"] = str(step.get("title", "")).strip()
        meta["description"] = str(step.get("description", "")).strip()
        prompts = step.get("prompts")
        if isinstance(prompts, dict):
            meta["instruction"] = str(prompts.get("instruction", "")).strip()
            safety = prompts.get("safety")
            if isinstance(safety, list):
                meta["safety"] = str(safety[0] or "").strip() if safety else ""
            elif isinstance(safety, str):
                meta["safety"] = safety.strip()
            tips = prompts.get("tips")
            if isinstance(tips, list):
                meta["tip"] = str(tips[0] or "").strip() if tips else ""
            elif isinstance(tips, str):
                meta["tip"] = tips.strip()

    summary = body.get("summary")
    if isinstance(summary, dict):
        current_step = summary.get("current_step")
        if isinstance(current_step, dict):
            if not meta["step_id"]:
                meta["step_id"] = str(current_step.get("step_id", "")).strip()
            if not meta["title"]:
                meta["title"] = str(current_step.get("title", "")).strip()
        current_details = summary.get("current_step_details")
        if isinstance(current_details, dict):
            if not meta["title"]:
                meta["title"] = str(current_details.get("title", "")).strip()
            if not meta["description"]:
                meta["description"] = str(
                    current_details.get("description", "")
                ).strip()
            if not meta["instruction"]:
                meta["instruction"] = str(
                    current_details.get("instruction", "")
                ).strip()
            tips = current_details.get("tips")
            if not meta["tip"]:
                if isinstance(tips, list):
                    meta["tip"] = str(tips[0] or "").strip() if tips else ""
                elif isinstance(tips, str):
                    meta["tip"] = tips.strip()
    return meta


def _extract_experiment_step_interaction(payload) -> dict:
    body = _experiment_result_body(payload)
    step = body.get("step")
    if not isinstance(step, dict):
        return {}
    interaction = step.get("interaction")
    if not isinstance(interaction, dict):
        return {}
    return interaction


def _is_current_graph_step_completed(payload) -> bool:
    body = _experiment_result_body(payload)
    summary = body.get("summary")
    if not isinstance(summary, dict):
        return False

    current_step = summary.get("current_step")
    if isinstance(current_step, dict):
        explicit = _normalize_bool(current_step.get("is_completed"))
        if explicit is True:
            return True

    current_details = summary.get("current_step_details")
    if not isinstance(current_details, dict):
        return False

    explicit = _normalize_bool(current_details.get("is_completed"))
    if explicit is True:
        return True

    try:
        valid_record_count = int(current_details.get("valid_record_count"))
    except (TypeError, ValueError):
        valid_record_count = None
    try:
        min_trials = int(current_details.get("min_trials"))
    except (TypeError, ValueError):
        min_trials = None

    if valid_record_count is not None and min_trials is not None:
        return valid_record_count >= max(min_trials, 1)

    return False


def _extract_yaml_step_prompt_value(step: dict, key: str) -> str:
    prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
    value = prompts.get(key)
    if isinstance(value, list):
        return str(value[0] or "").strip() if value else ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _enrich_experiment_step_meta_from_yaml(conn, step_meta: dict) -> dict:
    meta = dict(step_meta or {})
    step_id = str(meta.get("step_id", "") or "").strip()
    if not step_id:
        return meta

    step = _resolve_experiment_step_by_id(conn).get(step_id)
    if not isinstance(step, dict):
        return meta

    yaml_title = str(step.get("title", "") or "").strip()
    yaml_description = str(step.get("description", "") or "").strip()
    yaml_instruction = _extract_yaml_step_prompt_value(step, "instruction")
    yaml_safety = _extract_yaml_step_prompt_value(step, "safety")
    yaml_tip = _extract_yaml_step_prompt_value(step, "tips")

    if yaml_title:
        meta["title"] = yaml_title
    if yaml_description:
        meta["description"] = yaml_description
    if yaml_instruction:
        meta["instruction"] = yaml_instruction
    if yaml_safety:
        meta["safety"] = yaml_safety
    if yaml_tip:
        meta["tip"] = yaml_tip
    return meta


def _merge_experiment_step_meta(primary: dict, fallback: dict) -> dict:
    merged = dict(fallback or {})
    for key, value in (primary or {}).items():
        if value:
            merged[key] = value
    return merged


def _get_cached_experiment_step_meta(conn) -> dict:
    primary = _extract_experiment_step_meta(getattr(conn, "experiment_current_step", None))
    fallback = _extract_experiment_step_meta(
        getattr(conn, "experiment_progress_summary", None)
    )
    return _enrich_experiment_step_meta_from_yaml(
        conn,
        _merge_experiment_step_meta(primary, fallback),
    )


def _extract_experiment_current_progress(payload):
    body = _experiment_result_body(payload)
    progress = body.get("current_progress")
    if isinstance(progress, dict):
        return progress
    progress = body.get("progress")
    if isinstance(progress, dict):
        return progress
    step = body.get("step")
    if isinstance(step, dict):
        nested = step.get("current_progress")
        if isinstance(nested, dict):
            return nested
    return None


def _extract_experiment_schema_view(payload) -> dict:
    body = _experiment_result_body(payload)
    schema_view = body.get("schema_view")
    result = {}
    if isinstance(schema_view, list):
        for item in schema_view:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                result[name] = item
    if result:
        return result

    json_schema = body.get("json_schema")
    if isinstance(json_schema, dict):
        properties = json_schema.get("properties")
        required = set(json_schema.get("required") or [])
        if isinstance(properties, dict):
            for name, item in properties.items():
                if not isinstance(item, dict):
                    continue
                field_name = str(name or "").strip()
                if not field_name:
                    continue
                result[field_name] = {
                    "name": field_name,
                    "type": item.get("type"),
                    "description": item.get("description"),
                    "optional": field_name not in required,
                    "default": item.get("default"),
                    "validation": item,
                }
    return result


def _clean_field_description(text: str) -> str:
    value = " ".join(str(text or "").split()).strip()
    value = re.sub(r"^[宸茶闇€]+", "", value)
    value = value.replace("鏄惁", "")
    return value.strip("锛屻€傦紱;: ")


_CHINESE_DIGIT_MAP = str.maketrans(
    {
        "闆?: "0",
        "涓€": "1",
        "浜?: "2",
        "涓?: "2",
        "涓?: "3",
        "鍥?: "4",
        "浜?: "5",
        "鍏?: "6",
        "涓?: "7",
        "鍏?: "8",
        "涔?: "9",
    }
)


def _normalize_confirmation_signature(text: str) -> str:
    norm = _normalize_text_for_match(text)
    if not norm:
        return ""
    norm = norm.translate(_CHINESE_DIGIT_MAP)
    norm = norm.replace("->", "-")
    norm = norm.replace("鑷?, "鍒?)
    norm = re.sub(r"([0-9]+)鍒?[0-9]+)", r"\1-\2", norm)
    norm = norm.replace("宸叉寜", "鎸?)
    norm = norm.replace("宸茬粡", "宸?)
    norm = norm.replace("瀹屾垚浜?, "瀹屾垚")
    norm = norm.replace("鍔犲叆浜?, "鍔犲叆")
    norm = norm.replace("鍙屾澂", "鐑ф澂")
    norm = norm.replace("纾佸瓙", "纾佽浆瀛?)
    return norm


def _confirmation_char_ngrams(text: str, n: int = 2) -> set:
    clean = re.sub(r"[\s锛屻€傦紱锛氥€?.!?锛焆", "", text)
    if not clean:
        return set()
    if len(clean) < n:
        return {clean}
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def _longest_common_substring_len(left: str, right: str) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    best = 0
    for lch in left:
        curr = [0] * (len(right) + 1)
        for idx, rch in enumerate(right, start=1):
            if lch == rch:
                curr[idx] = prev[idx - 1] + 1
                if curr[idx] > best:
                    best = curr[idx]
        prev = curr
    return best


def _looks_like_confirmation_signature_match(user_text: str, field_text: str) -> bool:
    if not user_text or not field_text:
        return False
    if field_text in user_text or user_text in field_text:
        return True

    user_grams = _confirmation_char_ngrams(user_text)
    field_grams = _confirmation_char_ngrams(field_text)
    if not user_grams or not field_grams:
        return False

    overlap = len(user_grams & field_grams)
    similarity = overlap / max(1, min(len(user_grams), len(field_grams)))
    if similarity < 0.58:
        return False

    return _longest_common_substring_len(user_text, field_text) >= 4


def _format_missing_field_prompts(missing_fields, schema_by_name: dict) -> list:
    prompts = []
    for field_name in missing_fields or []:
        field = schema_by_name.get(field_name, {})
        description = _clean_field_description(field.get("description", ""))
        if not description:
            description = str(field_name or "").replace("_", " ").strip()
        if description:
            prompts.append(description)
    return prompts


def _compose_missing_field_reply(missing_fields, schema_by_name: dict) -> str:
    prompts = _format_missing_field_prompts(missing_fields, schema_by_name)
    if not prompts:
        return "缁х画鍓嶈繕宸繖涓€姝ョ殑鍏抽敭淇℃伅锛屼綘琛ヤ竴鍙ュ綋鍓嶇粨鏋滃氨琛屻€?
    if len(prompts) > 3:
        return "缁х画鍓嶈繕宸繖涓€姝ョ殑涓€鏁寸粍鍏抽敭璁板綍銆備綘鎶婂綋鍓嶈繖涓€姝ヨ璁板綍鐨勬暟鎹寜椤哄簭鍛婅瘔鎴戝氨琛屻€?
    if len(prompts) == 1:
        return f"缁х画鍓嶈繕宸繖涓€姝ョ殑涓€涓‘璁わ細{prompts[0]}銆備綘琛ヤ竴鍙ヨ繖涓氨琛屻€?
    if len(prompts) == 2:
        joined = f"{prompts[0]}锛岃繕鏈?{prompts[1]}"
    else:
        joined = "銆?.join(prompts[:3])
    return f"缁х画鍓嶈繕宸繖鍑犱釜纭锛歿joined}銆備綘琛ヤ竴鍙ヨ繖鍑犱釜缁撴灉灏辫銆?


def _first_nonempty_text(*values) -> str:
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text:
            return text
    return ""


def _compose_uvvis_step_reply(step_meta: dict, mode: str = "guide") -> str:
    step_id = _first_nonempty_text(
        step_meta.get("step_id", ""),
        step_meta.get("id", ""),
    )

    if mode == "repeat":
        prefix = "褰撳墠杩欎竴姝ワ細"
    elif mode == "next":
        prefix = "鎺ヤ笅鏉ュ仛杩欎竴姝ワ細"
    else:
        prefix = "鐜板湪鍋氳繖涓€姝ワ細"

    signature = _normalize_text_for_match(
        " ".join(
            value
            for value in (
                step_id,
                step_meta.get("title", ""),
                step_meta.get("instruction", ""),
                step_meta.get("description", ""),
            )
            if str(value or "").strip()
        )
    )

    def _signature_has_any(*tokens: str) -> bool:
        return any(token in signature for token in tokens if token)

    def _signature_has_all(*tokens: str) -> bool:
        return all(token in signature for token in tokens if token)

    if step_id:
        exact_reply_map = {
            "step_3_uv_vis_shared_dark_air_prep": (
                "鍏堟鏌?鍒?鍙锋牱鍝佷綅閮戒负绌猴紝鍙傛瘮浣嶄篃涓嶈鏀句换浣曟恫浣撱€?
                "閮界┖浜嗗氨鍛婅瘔鎴戯紝鍙互寮€濮嬫椂鐩存帴璇粹€滃紑濮嬫壂鎻忊€濄€?
            ),
            "step_3_uv_vis_shared_dark_blank_prep": (
                f"{prefix}1-5鍙锋牱鍝侊細绾按绌虹櫧鏍℃銆?
                "璇峰湪 1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁鍏ョ函姘存瘮鑹茬毧锛屽叡 6 涓紝鏀惧ソ鍚庡憡璇夋垜鍙互寮€濮嬫壂鎻忋€?
            ),
            "step_3_uv_vis_sample1-5_load_cuvette": (
                "鎶?鍒?鍙风湡瀹炴牱鍝佸垎鍒鍏ユ瘮鑹茬毧锛屾寜缂栧彿鏀惧叆鏍峰搧浣嶏紝浠櫒鍘熺敓鍙傛瘮浣嶆斁绾按锛屾摝鍑€澶栧锛屽仛濂藉憡璇夋垜銆?
            ),
            "step_3_uv_vis_sample1-4_load_cuvette": (
                "鎶?鍒?鍙风湡瀹炴牱鍝佸垎鍒鍏ユ瘮鑹茬毧锛屾寜缂栧彿鏀惧叆鏍峰搧浣嶏紝浠櫒鍘熺敓鍙傛瘮浣嶆斁绾按锛屾摝鍑€澶栧锛屽仛濂藉憡璇夋垜銆?
            ),
            "step_3_uv_vis_sample1-5_record_data": (
                "纭1鍒?鍙锋牱鍝佷綅閮藉凡鏀惧ソ鐪熷疄鏍峰搧锛屼华鍣ㄥ師鐢熷弬姣斾綅鏄函姘淬€傛斁濂戒簡鍛婅瘔鎴戝紑濮嬫壂鎻忋€?
            ),
            "step_3_uv_vis_sample1-4_record_data": (
                "纭1鍒?鍙锋牱鍝佷綅閮藉凡鏀惧ソ鐪熷疄鏍峰搧锛屼华鍣ㄥ師鐢熷弬姣斾綅鏄函姘淬€傛斁濂戒簡鍛婅瘔鎴戝紑濮嬫壂鎻忋€?
            ),
            "step_3_uv_vis_sample5_clean_cuvette": (
                f"{prefix}绱-鍙娴嬮噺鍚庯細缁熶竴娓呮礂姣旇壊鐨裤€?
                "鎸夎鑼冨鐞嗘畫娑插苟娓呮礂姣旇壊鐨匡紝涓哄悗缁姩鍔涘瀹為獙鍋氬噯澶囷紝鍋氬ソ鍛婅瘔鎴戙€?
            ),
        }
        exact_reply = exact_reply_map.get(step_id)
        if exact_reply:
            return exact_reply

    looks_like_pure_water_blank_step = signature and (
        _signature_has_any("绾按绌虹櫧鏍℃", "绾按绌虹櫧")
        and not _signature_has_any(
                "瑁呭叆姣旇壊鐨?,
                "鐪熷疄鏍峰搧",
                "鎵归噺娴嬪厜璋?,
                "鍔ㄥ姏瀛?,
                "鍙嶅簲娑?,
                "鍙傛瘮娑?,
                "sample1-5record",
                "sample2",
                "sample4",
        )
    )
    if looks_like_pure_water_blank_step:
        return (
            f"{prefix}1-5鍙锋牱鍝侊細绾按绌虹櫧鏍℃銆?
            "璇峰湪 1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁鍏ョ函姘存瘮鑹茬毧锛屽叡 6 涓紝鏀惧ソ鍚庡憡璇夋垜鍙互寮€濮嬫壂鎻忋€?
        )

    if signature and _signature_has_any(
            "鏆楃數娴佹牎姝?,
            "鏆楃數娴佸拰绌烘皵鑳介噺鏍℃",
            "鏆楃數娴佸拰绌烘皵鍩虹嚎",
            "鍏变韩鏆楃數娴佸拰绌烘皵鑳介噺鏍℃",
            "鍏变韩鏆楃數娴佸拰绌烘皵鍩虹嚎",
            "shareddarkcurrent",
    ):
        return (
            "鍏堟鏌?鍒?鍙锋牱鍝佷綅閮戒负绌猴紝鍙傛瘮浣嶄篃涓嶈鏀句换浣曟恫浣撱€?
            "閮界┖浜嗗氨鍛婅瘔鎴戙€傚彲浠ュ紑濮嬫椂鐩存帴璇粹€滃紑濮嬫壂鎻忊€濄€?
        )

    if signature and (
        _signature_has_any("瑁呭叆姣旇壊鐨?, "鏍峰搧瑁呮澂")
        or (
            _signature_has_any("鐪熷疄鏍峰搧", "鏍峰搧浣?)
            and _signature_has_any("鍙傛瘮浣?, "绾按")
            and _signature_has_any("姣旇壊鐨?, "瑁呭叆")
        )
    ):
        return (
            "鎶?鍒?鍙风湡瀹炴牱鍝佸垎鍒鍏ユ瘮鑹茬毧锛屾寜缂栧彿鏀惧叆鏍峰搧浣嶏紝鍙傛瘮浣嶄繚鎸佷负绌猴紝鎿﹀噣澶栧锛屽仛濂藉憡璇夋垜銆?
        )

    if signature and (
        _signature_has_any(
            "鎵归噺娴嬪厜璋?,
            "鎵归噺鍏夎氨娴嬮噺",
            "璁板綍鏁版嵁",
            "lambda max",
            "lambdamax",
        )
        or (
            _signature_has_any("1-5鍙锋牱鍝?, "1鍒?鍙锋牱鍝?)
            and _signature_has_any("娴嬮噺", "鍏夎氨")
            and _signature_has_any("鍙傛瘮浣?, "绾按")
        )
    ):
        return (
            "纭1鍒?鍙锋牱鍝佷綅閮藉凡鏀惧ソ鐪熷疄鏍峰搧锛屽弬姣斾綅淇濇寔涓虹┖銆?
            "鏀惧ソ浜嗗憡璇夋垜寮€濮嬫祴閲忋€?
        )

    if signature and _signature_has_any("缁熶竴娓呮礂姣旇壊鐨?, "娓呮礂姣旇壊鐨?, "娴嬮噺鍚庢竻娲?):
        return (
            f"{prefix}绱-鍙娴嬮噺鍚庯細缁熶竴娓呮礂姣旇壊鐨裤€?
            "鎸夎鑼冨鐞嗘畫娑插苟娓呮礂姣旇壊鐨匡紝涓哄悗缁姩鍔涘瀹為獙鍋氬噯澶囷紝鍋氬ソ鍛婅瘔鎴戙€?
        )

    if signature and _signature_has_all("2鍙锋牱鍝?, "鍔ㄥ姏瀛?) and _signature_has_any("鍙傛瘮娑?, "鍙傛瘮浣?) and not _signature_has_any("鍙嶅簲娑?, "400绾崇背", "鍚稿厜搴?):
        return (
            f"{prefix}2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弬姣旀恫锛屾斁鍏ュ弬姣斾綅骞舵鏌ユ瘮鑹茬毧澶栧鍜岄€忓厜闈紝鍋氬ソ鍛婅瘔鎴戙€?
        )

    if signature and _signature_has_all("2鍙锋牱鍝?, "鍔ㄥ姏瀛?) and _signature_has_any("鍙嶅簲娑?, "鏍峰搧浣?) and not _signature_has_any("400绾崇背", "鍚稿厜搴?):
        return (
            f"{prefix}2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弽搴旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弽搴旀恫锛屾斁鍏ユ牱鍝佷綅骞剁‘璁ゅ弬姣斿拰鏍峰搧姣旇壊鐨块兘鏀剧疆姝ｇ‘锛屽仛濂藉憡璇夋垜銆?
        )

    if signature and _signature_has_all("2鍙锋牱鍝?, "鍔ㄥ姏瀛?) and _signature_has_any("400绾崇背", "鍚稿厜搴?, "鍔ㄥ姏瀛︽祴閲?):
        return (
            f"{prefix}2鍙锋牱鍝佸姩鍔涘锛氬紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴︺€?
            "淇濇寔鍙傛瘮娑插拰鍙嶅簲娑叉寜瑕佹眰鏀惧ソ锛屽彲浠ュ紑濮嬫椂鍛婅瘔鎴戯紝鎴戝氨寮€濮?00绾崇背鍔ㄥ姏瀛︽祴閲忋€?
        )

    if signature and _signature_has_all("4鍙锋牱鍝?, "鍔ㄥ姏瀛?) and _signature_has_any("鍙傛瘮娑?, "鍙傛瘮浣?) and not _signature_has_any("鍙嶅簲娑?, "400绾崇背", "鍚稿厜搴?):
        return (
            f"{prefix}4鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弬姣旀恫锛屾斁鍏ュ弬姣斾綅骞舵鏌ユ瘮鑹茬毧澶栧鍜岄€忓厜闈紝鍋氬ソ鍛婅瘔鎴戙€?
        )

    if signature and _signature_has_all("4鍙锋牱鍝?, "鍔ㄥ姏瀛?) and _signature_has_any("鍙嶅簲娑?, "鏍峰搧浣?) and not _signature_has_any("400绾崇背", "鍚稿厜搴?):
        return (
            f"{prefix}4鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弽搴旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弽搴旀恫锛屾斁鍏ユ牱鍝佷綅骞剁‘璁ゅ弬姣斿拰鏍峰搧姣旇壊鐨块兘鏀剧疆姝ｇ‘锛屽仛濂藉憡璇夋垜銆?
        )

    if signature and _signature_has_all("4鍙锋牱鍝?, "鍔ㄥ姏瀛?) and _signature_has_any("400绾崇背", "鍚稿厜搴?, "鍔ㄥ姏瀛︽祴閲?):
        return (
            f"{prefix}4鍙锋牱鍝佸姩鍔涘锛氬紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴︺€?
            "淇濇寔鍙傛瘮娑插拰鍙嶅簲娑叉寜瑕佹眰鏀惧ソ锛屽彲浠ュ紑濮嬫椂鍛婅瘔鎴戯紝鎴戝氨寮€濮?00绾崇背鍔ㄥ姏瀛︽祴閲忋€?
        )

    if signature and _signature_has_any("step_6_data_analysis", "uvvis鏁版嵁鍒嗘瀽", "uv-vis鏁版嵁鍒嗘瀽", "绱鍙鏁版嵁鍒嗘瀽") and _signature_has_any("鍒嗘瀽", "鏁版嵁", "鍥捐氨"):
        return "UV-Vis 娴嬮噺閮ㄥ垎宸茬粡瀹屾垚锛屾帴涓嬫潵鏁寸悊鏁版嵁缁撴灉銆?

    if not step_id:
        return ""

    reply_map = {
        "step_3_uv_vis_shared_dark_air_prep": (
            "鍏堟鏌?鍒?鍙锋牱鍝佷綅閮戒负绌猴紝鍙傛瘮浣嶄篃涓嶈鏀句换浣曟恫浣撱€?
            "閮界┖浜嗗氨鍛婅瘔鎴戙€傚彲浠ュ紑濮嬫椂鐩存帴璇粹€滃紑濮嬫壂鎻忊€濄€?
        ),
        "step_3_uv_vis_shared_dark_blank_prep": (
            f"{prefix}1-5鍙锋牱鍝侊細绾按绌虹櫧鏍℃銆?
            "璇峰湪 1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁鍏ョ函姘存瘮鑹茬毧锛屽叡 6 涓紝鏀惧ソ鍚庡憡璇夋垜鍙互寮€濮嬫壂鎻忋€?
        ),
        "step_3_uv_vis_sample1-5_load_cuvette": (
            "鎶?鍒?鍙风湡瀹炴牱鍝佸垎鍒鍏ユ瘮鑹茬毧锛屾寜缂栧彿鏀惧叆鏍峰搧浣嶏紝鍙傛瘮浣嶄繚鎸佷负绌猴紝鎿﹀噣澶栧锛屽仛濂藉憡璇夋垜銆?
        ),
        "step_3_uv_vis_sample1-5_record_data": (
            "纭1鍒?鍙锋牱鍝佷綅閮藉凡鏀惧ソ鐪熷疄鏍峰搧锛屽弬姣斾綅淇濇寔涓虹┖銆?
            "鏀惧ソ浜嗗憡璇夋垜寮€濮嬫祴閲忋€?
        ),
        "step_3_uv_vis_sample5_clean_cuvette": (
            f"{prefix}绱-鍙娴嬮噺鍚庯細缁熶竴娓呮礂姣旇壊鐨裤€?
            "鎸夎鑼冨鐞嗘畫娑插苟娓呮礂姣旇壊鐨匡紝涓哄悗缁姩鍔涘瀹為獙鍋氬噯澶囷紝鍋氬ソ鍛婅瘔鎴戙€?
        ),
        "step_4_kinetics_sample2_reference_solution_preparation": (
            f"{prefix}2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弬姣旀恫锛屾斁鍏ュ弬姣斾綅骞舵鏌ユ瘮鑹茬毧澶栧鍜岄€忓厜闈紝鍋氬ソ鍛婅瘔鎴戙€?
        ),
        "step_4_kinetics_sample2_reaction_solution_preparation": (
            f"{prefix}2鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弽搴旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弽搴旀恫锛屾斁鍏ユ牱鍝佷綅骞剁‘璁ゅ弬姣斿拰鏍峰搧姣旇壊鐨块兘鏀剧疆姝ｇ‘锛屽仛濂藉憡璇夋垜銆?
        ),
        "step_4_kinetics_sample2_measurement": (
            f"{prefix}2鍙锋牱鍝佸姩鍔涘锛氬紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴︺€?
            "淇濇寔鍙傛瘮娑插拰鍙嶅簲娑叉寜瑕佹眰鏀惧ソ锛屽彲浠ュ紑濮嬫椂鍛婅瘔鎴戯紝鎴戝氨寮€濮?00绾崇背鍔ㄥ姏瀛︽祴閲忋€?
        ),
        "step_5_kinetics_sample4_reference_solution_preparation": (
            f"{prefix}4鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弬姣旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弬姣旀恫锛屾斁鍏ュ弬姣斾綅骞舵鏌ユ瘮鑹茬毧澶栧鍜岄€忓厜闈紝鍋氬ソ鍛婅瘔鎴戙€?
        ),
        "step_5_kinetics_sample4_reaction_solution_preparation": (
            f"{prefix}4鍙锋牱鍝佸姩鍔涘锛氶厤鍒跺弽搴旀恫銆?
            "鎸夎姹傞厤濂?鍙锋牱鍝佸弽搴旀恫锛屾斁鍏ユ牱鍝佷綅骞剁‘璁ゅ弬姣斿拰鏍峰搧姣旇壊鐨块兘鏀剧疆姝ｇ‘锛屽仛濂藉憡璇夋垜銆?
        ),
        "step_5_kinetics_sample4_measurement": (
            f"{prefix}4鍙锋牱鍝佸姩鍔涘锛氬紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴︺€?
            "淇濇寔鍙傛瘮娑插拰鍙嶅簲娑叉寜瑕佹眰鏀惧ソ锛屽彲浠ュ紑濮嬫椂鍛婅瘔鎴戯紝鎴戝氨寮€濮?00绾崇背鍔ㄥ姏瀛︽祴閲忋€?
        ),
        "step_6_data_analysis": "UV-Vis 娴嬮噺閮ㄥ垎宸茬粡瀹屾垚锛屾帴涓嬫潵鏁寸悊鏁版嵁缁撴灉銆?,
        "step_6_kinetics_combined_measurement": (
            f"{prefix}2鍙峰拰4鍙锋牱鍝佸姩鍔涘锛氳仈鍚堝紑濮嬫寜鏃堕棿璁板綍鍚稿厜搴︺€?
            "璇蜂繚鎸佸弬姣斾綅涓虹函姘淬€?鍙蜂綅鐣欑┖锛?鍜?鍙蜂綅鏀?鍙锋牱鍝佺殑鍙嶅簲娑插拰鍙傛瘮娑诧紝4鍜?鍙蜂綅鏀?鍙锋牱鍝佺殑鍙嶅簲娑插拰鍙傛瘮娑层€傚叏閮ㄦ斁濂藉悗鍛婅瘔鎴戯紝鎴戝氨寮€濮?00绾崇背鍔ㄥ姏瀛︽祴閲忋€?
        ),
    }
    return reply_map.get(step_id, "")


def _compose_experiment_step_reply(step_meta: dict, mode: str = "guide") -> str:
    title = _first_nonempty_text(step_meta.get("title", ""))
    instruction = _first_nonempty_text(
        step_meta.get("instruction", ""),
        step_meta.get("description", ""),
        title,
    )
    safety = _first_nonempty_text(step_meta.get("safety", ""))
    tip = _first_nonempty_text(step_meta.get("tip", ""))

    if not instruction:
        return ""

    uvvis_reply = _compose_uvvis_step_reply(step_meta, mode=mode)
    if uvvis_reply:
        return uvvis_reply

    if _step_meta_looks_like_photo_confirmation(step_meta):
        sample_index = _extract_sample_index_from_text(
            " ".join(
                value
                for value in (
                    title,
                    instruction,
                    _first_nonempty_text(step_meta.get("description", "")),
                    str(step_meta.get("step_id", "") or "").strip(),
                )
                if value
            )
        )
        sample_name = _format_sample_name(sample_index, "褰撳墠鏍峰搧")
        return f"{sample_name}棰滆壊宸茬粡绋冲畾锛岀幇鍦ㄥ彲浠ユ媿鐓у悧锛?

    if title and title not in instruction:
        core = f"{title}銆倇instruction}"
    else:
        core = instruction

    if mode == "repeat":
        parts = [f"褰撳墠杩欎竴姝ワ細{core}銆?]
        if safety:
            parts.append(f"娉ㄦ剰{safety}銆?)
        elif tip:
            parts.append(f"{tip}銆?)
        parts.append("鍋氬ソ鍚庡憡璇夋垜銆?)
        return "".join(parts)

    if mode == "next":
        parts = [f"鎺ヤ笅鏉ュ仛杩欎竴姝ワ細{core}銆?]
    else:
        parts = [f"鐜板湪鍋氳繖涓€姝ワ細{core}銆?]

    if safety:
        parts.append(f"娉ㄦ剰{safety}銆?)
    elif tip:
        parts.append(f"{tip}銆?)
    parts.append("鍋氬ソ鍚庡憡璇夋垜銆?)
    return "".join(parts)


def _prepare_fastpath_spoken_reply(
    text: str,
    *,
    fallback_step_meta: dict | None = None,
    fallback_mode: str = "guide",
) -> str:
    prepared = textUtils.prepare_runtime_spoken_text(text)
    if prepared:
        return prepared
    if fallback_step_meta:
        fallback_text = _compose_experiment_step_reply(
            fallback_step_meta,
            mode=fallback_mode,
        )
        return textUtils.prepare_runtime_spoken_text(fallback_text)
    return ""


def _compose_photo_confirmation_advance_reply(
    confirmation_reply: str,
    next_step_reply: str,
) -> str:
    confirmation = textUtils.prepare_runtime_spoken_text(confirmation_reply)
    followup = textUtils.prepare_runtime_spoken_text(next_step_reply)

    if not confirmation:
        return followup
    if not followup:
        return confirmation
    if confirmation == followup:
        return confirmation
    if followup.startswith(confirmation):
        return followup

    confirmation = confirmation.rstrip("銆傦紒锛??锛?锛? ").strip()
    if confirmation:
        confirmation = f"{confirmation}銆?
    return f"{confirmation}鎴戞帴鐫€甯︿綘鍋氫笅涓€姝ャ€倇followup}"


def _extract_experiment_overview_title(payload) -> str:
    body = _experiment_result_body(payload)
    for key in ("title", "experiment_title", "name"):
        value = str(body.get(key, "")).strip()
        if value:
            return value

    experiment = body.get("experiment")
    if isinstance(experiment, dict):
        for key in ("title", "experiment_title", "name"):
            value = str(experiment.get(key, "")).strip()
            if value:
                return value

    overview = body.get("overview")
    if isinstance(overview, dict):
        for key in ("title", "experiment_title", "name"):
            value = str(overview.get(key, "")).strip()
            if value:
                return value

    summary = body.get("summary")
    if isinstance(summary, dict):
        for key in ("title", "experiment_title", "name"):
            value = str(summary.get(key, "")).strip()
            if value:
                return value
    return ""


def _compose_experiment_start_reply(experiment_title: str, step_reply: str) -> str:
    title = " ".join(str(experiment_title or "").split()).strip()
    if title:
        if title.startswith("銆?) and title.endswith("銆?):
            formatted_title = title
        else:
            formatted_title = f"銆妠title.strip('銆娿€?)}銆?
        return f"浠婂ぉ鎴戜滑鍋歿formatted_title}銆備綘鍑嗗濂藉紑濮嬩簡鍚楋紵"
    return "浠婂ぉ鎴戜滑鍋氬綋鍓嶅疄楠屻€備綘鍑嗗濂藉紑濮嬩簡鍚楋紵"


def _is_explicit_experiment_start_request(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    explicit_tokens = (
        "寮€濮嬫祦绋?,
        "寮€濮嬪綋鍓嶅疄楠屾祦绋?,
        "寮€濮嬪綋鍓嶅疄楠?,
        "鍑嗗寮€濮?,
        "鍑嗗寮€濮嬪疄楠?,
        "鍑嗗寮€濮嬫祦绋?,
        "寮€濮嬩粖澶╃殑瀹為獙",
        "寮€濮嬩粖澶╁疄楠?,
        "寮€濮嬫湰娆″疄楠?,
        "寮€濮嬭繖涓疄楠?,
        "寮€濮嬪疄楠?,
        "寮€濮嬪仛瀹為獙",
        "寮€濮嬪仛浠婂ぉ鐨勫疄楠?,
        "寮€濮嬩粖澶╁仛鐨勫疄楠?,
    )
    return _contains_any(norm, explicit_tokens)


def _is_explicit_experiment_resume_request(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    return experiment_resume.is_resume_experiment_request(norm)


def _load_experiment_title_from_yaml_path(yaml_path: str) -> str:
    path_text = str(yaml_path or "").strip()
    if not path_text:
        return ""

    try:
        yaml_file = Path(path_text).expanduser()
        if not yaml_file.is_absolute():
            yaml_file = yaml_file.resolve()
        if not yaml_file.exists():
            return ""
        payload = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""

    candidates = []
    if isinstance(payload, dict):
        candidates.append(payload)
        experiment = payload.get("experiment")
        if isinstance(experiment, dict):
            candidates.append(experiment)

    for candidate in candidates:
        for key in ("title", "experiment_title", "name"):
            value = " ".join(str(candidate.get(key, "")).split()).strip()
            if value:
                return value
    return ""


def _looks_like_experiment_detail_request(norm: str) -> bool:
    if not norm:
        return False
    detail_tokens = (
        "涓轰粈涔?,
        "鍘熺悊",
        "渚濇嵁",
        "璇︾粏",
        "娉ㄦ剰浜嬮」",
        "鏄粈涔?,
        "鍋氫粈涔?,
        "瑕佸仛浠€涔?,
        "闇€瑕佷粈涔?,
        "澶氬皯",
        "娴撳害",
        "浣撶Н",
        "鎬庝箞閰?,
        "鎬庝箞绠?,
        "鍏紡",
        "瀛楁",
        "schema",
        "鍙傝€?,
        "鍚庨潰鎵€鏈?,
        "鍏ㄩ儴姝ラ",
        "鏁翠釜瀹為獙",
        "瀹屾暣娴佺▼",
    )
    return _contains_any(norm, detail_tokens)


def _assistant_waiting_for_step_completion(conn) -> bool:
    last_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not last_text:
        return False
    tokens = (
        "鍋氬ソ鍚庡憡璇夋垜",
        "鍋氬ソ鍛婅瘔鎴?,
        "鍋氬畬鍛婅瘔鎴?,
        "瀹屾垚鍚庡憡璇夋垜",
        "瀹屾垚浜嗗憡璇夋垜",
        "鍋氬畬浜嗗憡璇夋垜",
        "娴嬪畬鍛婅瘔鎴?,
        "鎵畬鍛婅瘔鎴?,
        "缁撴潫鍚庡憡璇夋垜",
        "鍔犲畬鍛婅瘔鎴?,
        "鍔犲ソ浜嗗憡璇夋垜",
        "鎷嶅畬鍛婅瘔鎴?,
        "鎷嶅ソ浜嗗憡璇夋垜",
        "鐪嬪畬鍛婅瘔鎴?,
        "瑙傚療瀹屽憡璇夋垜",
        "璁板綍瀹屽憡璇夋垜",
    )
    completion_markers = (
        "鍋氬ソ",
        "鍋氬畬",
        "瀹屾垚",
        "娴嬪畬",
        "鎵畬",
        "缁撴潫",
        "鍔犲畬",
        "鍔犲ソ",
        "鎷嶅畬",
        "鎷嶅ソ",
        "鐪嬪畬",
        "瑙傚療瀹?,
        "璁板綍瀹?,
    )
    return _contains_any(last_text, tokens) or (
        "鍛婅瘔鎴? in last_text and _contains_any(last_text, completion_markers)
    )


def _assistant_waiting_for_step_start(conn) -> bool:
    last_text = _normalize_text_for_match(_get_last_assistant_text_raw(conn))
    if not last_text:
        return False
    tokens = (
        "鍑嗗濂藉紑濮嬩簡鍚?,
        "鍑嗗濂戒簡鍚?,
        "鍙互寮€濮嬩簡鍚?,
        "鐜板湪寮€濮嬪悧",
        "瑕佸紑濮嬩簡鍚?,
        "瑕佷笉瑕佸紑濮?,
    )
    return _contains_any(last_text, tokens)


def _is_explicit_ready_to_start_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    ready_tokens = (
        "鍑嗗濂戒簡",
        "鎴戝噯澶囧ソ浜?,
        "宸茬粡鍑嗗濂戒簡",
        "鍙互寮€濮?,
        "鍙互寮€濮嬩簡",
        "寮€濮嬪惂",
        "寮€濮嬪仛鍚?,
        "ready",
    )
    if _contains_any(norm, ready_tokens):
        return True

    return bool(
        re.fullmatch(
            r"(?:閭ｅ氨|鐜板湪|鍙互|閭ｆ垜浠瑋鎴戜滑|鎴??寮€濮??:(?:绗?[涓€浜屼笁鍥涗簲鍏竷鍏節鍗?-9]+姝?|(?:(?:杩欎釜|浠婂ぉ鐨剕鏈)?瀹為獙))?(?:鍚鍟浜??",
            norm,
        )
    )


async def _reset_experiment_fresh_start_context(conn) -> None:
    old_chat_session_id = str(getattr(conn, "chat_session_id", "") or "").strip()
    old_model_session_key = str(getattr(conn, "model_session_key", "") or "").strip()
    device_id = str(getattr(conn, "device_id", "") or "").strip()
    user_id = str(getattr(conn, "user_id", "") or "").strip()
    if not user_id:
        user_id = str(
            getattr(conn, "config", {}).get("session_registry", {}).get(
                "default_user_id",
                "test",
            )
        ).strip() or "test"
        conn.user_id = user_id

    binding = None
    if device_id:
        try:
            binding = await rotate_session_binding(
                getattr(conn, "config", {}) or {},
                device_id,
                user_id,
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                "experiment fresh start session rotation failed, falling back to in-memory reset: "
                f"device_id={device_id}, error={exc}"
            )

    if isinstance(binding, dict):
        conn.chat_session_id = str(binding.get("chat_session_id", "")).strip()
        conn.model_session_key = str(binding.get("model_session_key", "")).strip()
    else:
        fresh_chat_session_id = str(uuid.uuid4())
        conn.chat_session_id = fresh_chat_session_id
        conn.model_session_key = f"codex:{fresh_chat_session_id}"

    llm_provider = getattr(conn, "llm", None)
    llm_sessions = getattr(llm_provider, "_sessions", None)
    if old_model_session_key and isinstance(llm_sessions, dict):
        stale_session = llm_sessions.pop(old_model_session_key, None)
        if stale_session is not None and hasattr(stale_session, "close"):
            try:
                stale_session.close()
            except Exception:
                pass

    dialogue = getattr(conn, "dialogue", None)
    if dialogue is not None and hasattr(dialogue, "dialogue"):
        dialogue.dialogue = []
        prompt = str(getattr(conn, "prompt", "") or "").strip()
        if prompt and hasattr(dialogue, "update_system_message"):
            dialogue.update_system_message(prompt)

    for attr_name, reset_value in (
        ("experiment_session_id", ""),
        ("experiment_current_step_id", ""),
        ("experiment_overview", None),
        ("experiment_current_step", None),
        ("experiment_progress_summary", None),
        ("experiment_list_steps", None),
        ("experiment_schema", None),
        ("experiment_reference", None),
        ("experiment_reference_query", ""),
        ("experiment_resume_recovery_required", False),
        ("experiment_resume_latest_current_step_id", ""),
        ("experiment_resume_latest_session_id", ""),
        ("experiment_resume_log_path", ""),
        ("experiment_resume_turn_count", 0),
    ):
        if hasattr(conn, attr_name):
            setattr(conn, attr_name, reset_value)

    if hasattr(conn, "_reset_experiment_resume_recovery_state"):
        try:
            conn._reset_experiment_resume_recovery_state()
        except Exception:
            pass

    if hasattr(conn, "prewarm_experiment_session"):
        await conn.prewarm_experiment_session(
            trigger="explicit_fresh_start",
            force=True,
        )

    conn.logger.bind(tag=TAG).info(
        "experiment fresh start context reset: "
        f"device_id={device_id}, "
        f"old_chat_session_id={old_chat_session_id}, "
        f"new_chat_session_id={getattr(conn, 'chat_session_id', '')}, "
        f"old_model_session_key={old_model_session_key}, "
        f"new_model_session_key={getattr(conn, 'model_session_key', '')}, "
        f"experiment_session_id={getattr(conn, 'experiment_session_id', '')}"
    )


async def _handle_explicit_experiment_resume_request(
    conn,
    original_text: str,
) -> bool:
    def _step_index(step_id: str) -> int:
        normalized_step_id = str(step_id or "").strip()
        if not normalized_step_id:
            return -1
        try:
            return _resolve_experiment_yaml_step_order(conn).index(normalized_step_id)
        except ValueError:
            return -1

    previous_session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    try:
        await _reset_experiment_fresh_start_context(conn)
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment explicit resume reset failed: {exc}"
        )
        return False

    graph_step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
    graph_session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    log_path = str(getattr(conn, "experiment_resume_log_path", "") or "").strip()
    target_step_id = str(
        getattr(conn, "experiment_resume_latest_current_step_id", "") or ""
    ).strip()
    if (not log_path or not target_step_id) and str(
        getattr(conn, "device_id", "") or ""
    ).strip():
        try:
            resume_context = experiment_resume.build_resume_context(
                getattr(conn, "config", {}) or {},
                str(getattr(conn, "device_id", "") or "").strip(),
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment explicit resume build_resume_context failed: {exc}"
            )
            resume_context = None
        if isinstance(resume_context, dict):
            if not log_path:
                log_path = str(resume_context.get("log_path", "") or "").strip()
            if not target_step_id:
                target_step_id = str(
                    resume_context.get("latest_current_step_id", "") or ""
                ).strip()
    report_candidate = _best_resume_report_candidate(conn)
    report_target_step_id = (
        str((report_candidate or {}).get("current_step_id", "") or "").strip()
        if isinstance(report_candidate, dict)
        else ""
    )
    if report_target_step_id:
        step_order = _resolve_experiment_yaml_step_order(conn)
        try:
            report_index = step_order.index(report_target_step_id)
        except ValueError:
            report_index = -1
        try:
            current_target_index = step_order.index(target_step_id) if target_step_id else -1
        except ValueError:
            current_target_index = -1
        if report_index > current_target_index:
            target_step_id = report_target_step_id
            conn.logger.bind(tag=TAG).info(
                "experiment explicit resume target upgraded from exported report: "
                f"target_step_id={target_step_id}"
            )

    graph_completed_steps = 0
    if hasattr(conn, "_extract_experiment_completed_steps_count"):
        try:
            graph_completed_steps = int(
                conn._extract_experiment_completed_steps_count(
                    getattr(conn, "experiment_progress_summary", None)
                )
                or 0
            )
        except Exception:
            graph_completed_steps = 0

    graph_is_behind_resume_target = False
    if graph_step_id and target_step_id:
        graph_index = _step_index(graph_step_id)
        target_index = _step_index(target_step_id)
        graph_is_behind_resume_target = (
            graph_index >= 0 and target_index >= 0 and graph_index < target_index
        )
        if graph_is_behind_resume_target:
            conn.logger.bind(tag=TAG).info(
                "experiment explicit resume prefers log recovery over current graph: "
                f"graph_session_id={graph_session_id}, graph_step_id={graph_step_id}, "
                f"target_step_id={target_step_id}, log_path={log_path or 'missing'}"
            )

    report_session_id = (
        str((report_candidate or {}).get("session_id", "") or "").strip()
        if isinstance(report_candidate, dict)
        else ""
    )
    if (
        graph_is_behind_resume_target
        and report_session_id
        and report_session_id != graph_session_id
    ):
        try:
            report_state_payload, report_progress_payload = await asyncio.gather(
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_state",
                    {"session_id": report_session_id},
                    priority="foreground",
                ),
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_progress_summary",
                    {"session_id": report_session_id},
                    priority="foreground",
                ),
            )
            report_session_step_id = ""
            if hasattr(conn, "_extract_experiment_current_step_id"):
                report_session_step_id = str(
                    conn._extract_experiment_current_step_id(
                        report_state_payload,
                        report_progress_payload,
                    )
                    or ""
                ).strip()
            report_session_completed_steps = 0
            if hasattr(conn, "_extract_experiment_completed_steps_count"):
                try:
                    report_session_completed_steps = int(
                        conn._extract_experiment_completed_steps_count(
                            report_state_payload,
                            report_progress_payload,
                        )
                        or 0
                    )
                except Exception:
                    report_session_completed_steps = 0

            report_session_step_index = _step_index(report_session_step_id)
            target_index = _step_index(target_step_id)
            if (
                report_session_step_index >= 0
                and target_index >= 0
                and report_session_step_index >= target_index
            ):
                conn.experiment_session_id = report_session_id
                conn.experiment_current_step_id = report_session_step_id
                conn.experiment_progress_summary = report_progress_payload
                try:
                    report_group_number = int(
                        (report_candidate or {}).get("current_group_number") or 0
                    )
                except Exception:
                    report_group_number = 0
                if report_group_number >= 1:
                    setattr(conn, "experiment_current_group_number", report_group_number)
                await save_experiment_session_binding(
                    getattr(conn, "config", {}) or {},
                    chat_session_id=str(getattr(conn, "chat_session_id", "") or ""),
                    model_session_key=str(getattr(conn, "model_session_key", "") or ""),
                    device_id=str(getattr(conn, "device_id", "") or ""),
                    user_id=str(getattr(conn, "user_id", "") or ""),
                    yaml_path=str(getattr(conn, "experiment_yaml_path", "") or ""),
                    experiment_session_id=report_session_id,
                    status="active",
                    source="resume_report_session",
                    current_step_id=report_session_step_id,
                    completed_steps_count=report_session_completed_steps,
                    total_steps=getattr(conn, "_extract_experiment_total_steps", lambda *_: 0)(
                        report_progress_payload
                    ),
                )
                graph_session_id = report_session_id
                graph_step_id = report_session_step_id
                graph_completed_steps = report_session_completed_steps
                graph_is_behind_resume_target = False
                conn.logger.bind(tag=TAG).info(
                    "experiment explicit resume rebound to exported-report session: "
                    f"session_id={report_session_id}, step_id={report_session_step_id}, "
                    f"completed_steps={report_session_completed_steps}"
                )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                "experiment explicit resume report-session rebind failed: "
                f"session_id={report_session_id}, error={exc}"
            )

    report_group_number = None
    if isinstance(report_candidate, dict):
        try:
            report_group_number = int(report_candidate.get("current_group_number") or 0)
        except Exception:
            report_group_number = None
        if report_group_number is not None and report_group_number < 1:
            report_group_number = None

    if graph_is_behind_resume_target and graph_session_id and target_step_id:
        redirect_args = {
            "session_id": graph_session_id,
            "step_id": target_step_id,
            "force": True,
        }
        if report_group_number is not None:
            redirect_args["group_number"] = report_group_number
        try:
            redirect_payload = await _call_experiment_graph_tool_fast(
                conn,
                "redirect_to_step",
                redirect_args,
                priority="foreground",
            )
            if bool(_experiment_result_body(redirect_payload).get("ok")):
                rebound_meta = await _safe_refresh_experiment_step_cache(
                    conn,
                    graph_session_id,
                    reason="explicit_resume_redirect_to_report_target",
                )
                rebound_step_id = str(rebound_meta.get("step_id", "") or "").strip()
                if rebound_step_id:
                    conn.experiment_current_step_id = rebound_step_id
                if report_group_number is not None:
                    setattr(conn, "experiment_current_group_number", report_group_number)
                rebound_progress = getattr(conn, "experiment_progress_summary", None)
                rebound_completed_steps = 0
                if hasattr(conn, "_extract_experiment_completed_steps_count"):
                    try:
                        rebound_completed_steps = int(
                            conn._extract_experiment_completed_steps_count(
                                rebound_progress
                            )
                            or 0
                        )
                    except Exception:
                        rebound_completed_steps = 0
                await save_experiment_session_binding(
                    getattr(conn, "config", {}) or {},
                    chat_session_id=str(getattr(conn, "chat_session_id", "") or ""),
                    model_session_key=str(
                        getattr(conn, "model_session_key", "") or ""
                    ),
                    device_id=str(getattr(conn, "device_id", "") or ""),
                    user_id=str(getattr(conn, "user_id", "") or ""),
                    yaml_path=str(getattr(conn, "experiment_yaml_path", "") or ""),
                    experiment_session_id=graph_session_id,
                    status="active",
                    source="resume_redirect_to_report_target",
                    current_step_id=rebound_step_id,
                    completed_steps_count=rebound_completed_steps,
                    total_steps=getattr(
                        conn, "_extract_experiment_total_steps", lambda *_: 0
                    )(rebound_progress),
                )
                graph_step_id = rebound_step_id
                graph_is_behind_resume_target = False
                conn.logger.bind(tag=TAG).info(
                    "experiment explicit resume redirected current graph to report target: "
                    f"session_id={graph_session_id}, target_step_id={target_step_id}, "
                    f"resolved_step_id={rebound_step_id}, group_number={report_group_number or ''}"
                )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                "experiment explicit resume redirect-to-report-target failed: "
                f"session_id={graph_session_id}, target_step_id={target_step_id}, error={exc}"
            )

    if graph_session_id and graph_step_id and (
        graph_completed_steps > 0 or graph_step_id != "step_prepare_setup_all"
    ) and not graph_is_behind_resume_target:
        step_meta = await _safe_refresh_experiment_step_cache(
            conn,
            graph_session_id,
            reason="explicit_resume_from_graph_state",
        )
        reply = _prepare_fastpath_spoken_reply(
            _compose_experiment_step_reply(step_meta, mode="guide"),
            fallback_step_meta=step_meta,
            fallback_mode="guide",
        )
        if not reply:
            reply = "鎴戝凡缁忔帴鍥炲埌涓婃瀹為獙璁板綍瀵瑰簲鐨勬楠や簡锛屼綘璺熺潃杩欎竴姝ョ户缁仛銆?
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    resume_context_prepared = False
    if hasattr(conn, "_prepare_experiment_resume_recovery_context"):
        try:
            conn._prepare_experiment_resume_recovery_context(
                previous_session_id=previous_session_id,
                reason="explicit_resume_request",
            )
            resume_context_prepared = True
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment explicit resume context prepare failed: {exc}"
            )

    if not target_step_id:
        target_step_id = str(
            getattr(conn, "experiment_resume_latest_current_step_id", "") or ""
        ).strip()
    if not log_path:
        log_path = str(getattr(conn, "experiment_resume_log_path", "") or "").strip()
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()

    if not log_path:
        reply = (
            "杩欎釜璁惧褰撳墠娌℃湁鎵惧埌鍙敤浜庣画鎺ョ殑瀹為獙璁板綍锛屽彧鑳介噸鏂板紑濮嬨€?
            "浣犺寮€濮嬩粖澶╃殑瀹為獙锛屾垜灏变粠绗竴姝ュ甫浣犲仛銆?
        )
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    if not target_step_id:
        target_step_id = _infer_experiment_resume_target_step_from_log(conn, log_path)
        if target_step_id:
            conn.logger.bind(tag=TAG).info(
                "experiment explicit resume inferred target step from log: "
                f"device_id={getattr(conn, 'device_id', '')}, "
                f"target_step_id={target_step_id}, log_path={log_path}"
            )
        else:
            reply = (
                "鎴戞壘鍒颁簡杩欎釜璁惧涔嬪墠鐨勫疄楠屾棩蹇楋紝浣嗚繕涓嶈兘鍙潬鍒ゆ柇鐜板湪瀹為檯鍋氬埌鍝竴姝ャ€?
                "璇风洿鎺ュ憡璇夋垜褰撳墠瀹為檯姝ラ锛屾瘮濡傗€滅幇鍦ㄥ仛鍒?鍙锋牱鍝佹媿鐓р€濇垨鈥滅幇鍦ㄥ仛鍒颁竵杈惧皵瑙傚療鈥濄€?
            )
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, reply)
            return True

    recovered, recovery_reply = await _replay_experiment_progress_from_resume_log(
        conn,
        target_step_id,
        log_path,
    )
    if not recovered:
        reply = recovery_reply or (
            "鎴戞壘鍒颁簡涓婃瀹為獙鐨勬棩蹇楋紝浣嗗綋鍓嶅浘璋辨病娉曡嚜鍔ㄨ烦鍒伴偅涓€姝ャ€?
            "浣犲彲浠ヨ鎴戦噸鏂板紑濮嬶紝鎴栬€呮槑纭憡璇夋垜瑕佽烦鍒板摢涓€姝ャ€?
        )
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    step_meta = await _safe_refresh_experiment_step_cache(
        conn,
        session_id,
        reason="explicit_resume_from_device_log",
    )
    reply = _prepare_fastpath_spoken_reply(
        _compose_experiment_step_reply(step_meta, mode="guide"),
        fallback_step_meta=step_meta,
        fallback_mode="guide",
    )
    if not reply:
        if resume_context_prepared:
            reply = "鎴戝凡缁忔寜涓婃瀹為獙鏃ュ織鎺ュ洖褰撳墠姝ラ浜嗭紝浣犺窡鐫€杩欎竴姝ョ户缁仛銆?
        else:
            reply = "鎴戝凡缁忔帴鍥炲埌涓婃瀹為獙璁板綍瀵瑰簲鐨勬楠や簡锛屼綘璺熺潃杩欎竴姝ョ户缁仛銆?

    await _start_direct_intent_turn(conn, original_text)
    speak_txt(conn, reply)
    return True


def _looks_like_generic_experiment_control_text(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    generic_tokens = {
        "缂佈呯敾",
        "缂佈呯敾閸?",
        "缂佈呯敾娑撳绔村?",
        "娑撳绔村?",
        "瀵扳偓娑撳铔?,
        "瀵扳偓閸氬氦铔?,
        "閸嬫艾銈芥禍?",
        "閸嬫艾鐣禍?",
        "鐎瑰本鍨氭禍?",
        "瀹告彃鐣幋?",
        "瀹歌尙绮￠崑姘偨娴?",
        "瀹歌尙绮＄€瑰本鍨氭禍?",
        "闁棄浠涙總鎴掔啊",
        "闁棄浠涚€瑰奔绨?,
        "瑜版挸澧犲銉╊€冨鎻掔暚閹?",
        "鏉╂瑦顒炵€瑰本鍨氭禍?",
        "鏉╂瑤绔村銉ョ暚閹存劒绨?,
        "婵傛垝绨?,
        "閸欘垯浜掓禍?",
        "鐞涘奔绨?,
        "婵傝棄鏆?,
        "ok娴?",
    }
    if norm in generic_tokens:
        return True

    if len(norm) > 18:
        return False

    if _contains_any(norm, ("閺嶅嘲鎼?, "agno3", "h2o2", "nabh4", "kbr", "缁绢垱鎸?)):
        return False

    if re.search(r"[0-9娑撯偓娴滃奔绗侀崶娑楃安閸忣厺绔烽崗顐＄瘈閸?]", norm):
        return False

    generic_fragments = (
        "缂佈呯敾",
        "娑撳绔村?",
        "閸嬫艾銈?,
        "閸嬫艾鐣?,
        "鐎瑰本鍨?,
        "瀹告彃鐣幋?",
        "婵傛垝绨?,
    )
    return _contains_any(norm, generic_fragments)


def _should_attempt_future_step_context_inference(
    original_text: str = "",
    filtered_text: str = "",
) -> bool:
    norm = _normalize_text_for_match(filtered_text or original_text)
    if not norm:
        return False
    if _is_explicit_ready_to_start_reply(filtered_text):
        return False
    if _looks_like_generic_experiment_control_text(filtered_text):
        return False
    if _looks_like_pure_short_completion_control(norm):
        return False
    return True


def _grant_experiment_ready_guard_bypass(conn, count: int = 1) -> None:
    if conn is None:
        return
    if textUtils.activate_experiment_ready_guard_bypass_for_current_sentence(
        conn,
        force=True,
    ):
        return
    textUtils.stage_experiment_ready_guard_bypass_for_next_turn(conn, count=count)


def _maybe_stage_experiment_ready_guard_bypass_for_normal_turn(
    conn,
    filtered_text: str,
) -> None:
    if not _assistant_waiting_for_step_start(conn):
        return
    if not _is_explicit_ready_to_start_reply(filtered_text):
        return
    textUtils.stage_experiment_ready_guard_bypass_for_next_turn(conn)


PURE_SHORT_COMPLETION_PATTERNS = (
    re.compile(
        r"^(?:(?:鎴憒杩欐|杩欎竴姝褰撳墠姝ラ|褰撳墠杩欐|鏈|杩欒疆|宸茬粡|宸瞸閮絴灏眧鐜板湪|鐩墠|鍒氬垰|杩欓噷|杩欒竟|鏍峰搧)){0,3}"
        r"(?:鍔爘瑁厊閰峾鏀緗鍋殀寮剕鎷峾鎵珅鐪媩娴媩閲弢璁皘鍐檤濉珅瑙傚療|纭|鏍稿|澶勭悊|鍑嗗|璋億鎼呮媽|婊村姞|璁板綍|琛ヨ)?"
        r"(?:濂絴瀹寍鎴?(?:浜唡鍟?$"
    ),
    re.compile(
        r"^(?:(?:鎴憒杩欐|杩欎竴姝褰撳墠姝ラ|褰撳墠杩欐|鏈|杩欒疆|宸茬粡|宸瞸閮絴灏眧鐜板湪|鐩墠|鍒氬垰|杩欓噷|杩欒竟|鏍峰搧)){0,3}"
        r"(?:鎼炲畾|缁撴潫|榻愭椿|濡?(?:浜唡鍟??$"
    ),
)


def _looks_like_pure_short_completion_control(norm: str) -> bool:
    if not norm or len(norm) > 18:
        return False
    return _matches_any_pattern(norm, PURE_SHORT_COMPLETION_PATTERNS)


def _assistant_waiting_after_sidetrack_question(conn) -> bool:
    recent = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=2))
    return _contains_any(
        recent,
        (
            "鎴戜滑鐜板湪鑳界户缁仛瀹為獙浜嗗悧",
            "鐜板湪鑳界户缁仛瀹為獙浜嗗悧",
            "鑳界户缁仛瀹為獙浜嗗悧",
            "缁х画鍋氬疄楠屼簡鍚?,
        ),
    )


def _looks_like_sidetrack_continue_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm or len(norm) > 24:
        return False
    return _contains_any(
        norm,
        (
            "鍙互缁х画",
            "缁х画鍚?,
            "缁х画鍋氬疄楠?,
            "缁х画杩涜瀹為獙",
            "鎺ョ潃鍋?,
            "鎺ョ潃瀹為獙",
            "寰€涓嬪仛",
            "鑳界户缁?,
        ),
    )


def _classify_short_experiment_control(conn, filtered_text: str) -> str:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return ""
    if _looks_like_experiment_detail_request(norm):
        return ""

    waiting_for_step_start = _assistant_waiting_for_step_start(conn)
    waiting_for_step_completion = _assistant_waiting_for_step_completion(conn)
    if _assistant_waiting_after_sidetrack_question(
        conn
    ) and _looks_like_sidetrack_continue_reply(filtered_text):
        return "guide"

    if waiting_for_step_completion and (
        _looks_like_explicit_completion_report(filtered_text)
        or _looks_like_explicit_added_completion_report(filtered_text)
    ):
        return "advance"

    if len(norm) > 24:
        return ""

    clarify_tokens = (
        "娌″惉鎳?,
        "娌″惉娓?,
        "鍐嶈涓€閬?,
        "閲嶈涓€閬?,
        "閲嶆柊璇?,
        "閲嶅涓€涓?,
        "鍐嶈涓€閬?,
        "鍐嶈涓?,
        "褰撳墠姝ラ鏄粈涔?,
        "杩欐鏄粈涔?,
        "杩欐鎬庝箞鍋?,
        "浠€涔堟剰鎬?,
    )
    if _contains_any(norm, clarify_tokens):
        return "repeat"

    if waiting_for_step_start and _is_explicit_ready_to_start_reply(filtered_text):
        return "guide"

    advance_tokens = (
        "缁х画涓嬩竴姝?,
        "涓嬩竴姝?,
        "涓嬩竴缁?,
        "涓嬩竴缁勭户缁?,
        "缁х画涓嬩竴缁?,
        "鎹笅涓€缁?,
        "涓嬩竴鎵?,
        "缁х画涓嬩竴鎵?,
        "涓嬩竴缁勫鐢?,
        "涓嬩竴鎵瑰鐢?,
        "寰€涓嬭蛋",
        "寰€鍚庤蛋",
        "鍋氬畬浜?,
        "鍋氬ソ浜?,
        "瀹屾垚浜?,
        "宸插畬鎴?,
        "褰撳墠姝ラ宸插畬鎴?,
        "杩欐瀹屾垚浜?,
        "杩欎竴姝ュ畬鎴愪簡",
        "閮藉仛濂戒簡",
        "閮藉仛瀹屼簡",
    )
    if _contains_any(norm, advance_tokens):
        return "advance"

    if waiting_for_step_completion and _looks_like_pure_short_completion_control(
        norm
    ):
        return "advance"

    ready_tokens = (
        "鍑嗗濂戒簡",
        "鎴戝噯澶囧ソ浜?,
        "鍙互寮€濮?,
        "寮€濮嬪惂",
        "寮€濮?,
        "ready",
    )
    if _contains_any(norm, ready_tokens):
        return "guide"

    neutral_ack_tokens = (
        "濂戒簡",
        "鍙互浜?,
        "琛屼簡",
        "濂藉暒",
        "ok浜?,
    )
    if norm in neutral_ack_tokens or _ends_with_any(norm, neutral_ack_tokens):
        if waiting_for_step_completion:
            return "advance"
        if waiting_for_step_start:
            return "guide"

    if norm in {"缁х画", "缁х画鍚?}:
        return "guide" if waiting_for_step_start else "advance"

    return ""


def _is_experiment_fast_path_available(conn) -> bool:
    raw_enabled = conn.config.get("experiment_fast_path_enabled", True)
    if isinstance(raw_enabled, str):
        enabled = raw_enabled.strip().lower() in ("1", "true", "yes", "on")
    else:
        enabled = bool(raw_enabled)
    if not enabled:
        return False

    if getattr(conn, "experiment_session_id", ""):
        return True
    step_meta = _get_cached_experiment_step_meta(conn)
    return bool(step_meta.get("instruction") or step_meta.get("title"))


def _resolve_experiment_fast_path_allowed_actions(conn) -> set[str] | None:
    raw_value = conn.config.get("experiment_fast_path_allowed_actions")
    if raw_value in (None, "", []):
        return None

    if isinstance(raw_value, str):
        tokens = [
            token.strip().lower()
            for token in re.split(r"[\s,;|]+", raw_value)
            if token.strip()
        ]
    elif isinstance(raw_value, (list, tuple, set)):
        tokens = [str(item or "").strip().lower() for item in raw_value if str(item or "").strip()]
    else:
        token = str(raw_value or "").strip().lower()
        tokens = [token] if token else []

    if not tokens or "all" in tokens or "*" in tokens:
        return None

    alias_map = {
        "next": "advance",
        "continue": "advance",
        "ready": "guide",
        "start_ready": "guide",
        "confirmation": "confirm",
        "semantic_confirmation": "confirm",
    }
    normalized = {
        alias_map.get(token, token)
        for token in tokens
        if alias_map.get(token, token)
    }
    return normalized or None


def _is_experiment_fast_path_action_enabled(conn, action: str) -> bool:
    normalized_action = str(action or "").strip().lower()
    if not normalized_action:
        return True
    allowed_actions = _resolve_experiment_fast_path_allowed_actions(conn)
    if allowed_actions is None:
        return True
    return normalized_action in allowed_actions


def _is_experiment_strict_graph_path_enabled(conn) -> bool:
    raw_enabled = conn.config.get("experiment_strict_graph_path_enabled", True)
    if isinstance(raw_enabled, str):
        return raw_enabled.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw_enabled)


def _is_server_mcp_client_enabled(conn) -> bool:
    raw_enabled = conn.config.get("enable_server_mcp_client", True)
    if isinstance(raw_enabled, str):
        return raw_enabled.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw_enabled)


async def _call_experiment_graph_tool_fast(
    conn,
    tool_name: str,
    arguments: dict,
    *,
    priority: str = "foreground",
):
    if hasattr(conn, "_call_experiment_graph_tool"):
        return await conn._call_experiment_graph_tool(
            tool_name,
            arguments,
            priority=priority,
        )
    raw_result = await _execute_server_mcp_tool_direct(conn, tool_name, arguments)
    payload = finalize_server_mcp_payload(
        raw_result,
        tool_name=tool_name,
        arguments=arguments,
    )
    sync_server_mcp_payload_state(
        conn,
        tool_name=tool_name,
        payload=payload,
        arguments=arguments,
    )
    return payload


async def _load_experiment_step_meta(conn) -> dict:
    step_meta = _get_cached_experiment_step_meta(conn)
    if step_meta.get("instruction") or step_meta.get("title"):
        return step_meta

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return step_meta

    try:
        step_payload, progress_payload = await asyncio.gather(
            _call_experiment_graph_tool_fast(
                conn,
                "get_step",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_progress_summary",
                {"session_id": session_id},
                priority="foreground",
            ),
        )
    except Exception:
        return step_meta

    conn.experiment_current_step = step_payload
    conn.experiment_progress_summary = progress_payload
    loaded_meta = _enrich_experiment_step_meta_from_yaml(
        conn,
        _merge_experiment_step_meta(
            _extract_experiment_step_meta(step_payload),
            _extract_experiment_step_meta(progress_payload),
        ),
    )
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(
            step_payload,
            progress_payload,
        )
        if current_step_id:
            conn.experiment_current_step_id = current_step_id
    return loaded_meta


async def _load_experiment_overview_title(conn) -> str:
    title = _extract_experiment_overview_title(getattr(conn, "experiment_overview", None))
    if title:
        return title

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if session_id:
        try:
            overview_payload = await _call_experiment_graph_tool_fast(
                conn,
                "get_overview",
                {"session_id": session_id},
                priority="foreground",
            )
        except Exception:
            overview_payload = None

        if overview_payload is not None:
            conn.experiment_overview = overview_payload
            title = _extract_experiment_overview_title(overview_payload)
            if title:
                return title

    yaml_path = str(getattr(conn, "experiment_yaml_path", "") or "").strip()
    if not yaml_path and hasattr(conn, "_resolve_experiment_yaml_path"):
        try:
            yaml_path = str(conn._resolve_experiment_yaml_path() or "").strip()
        except Exception:
            yaml_path = ""
    return _load_experiment_title_from_yaml_path(yaml_path)


async def _refresh_experiment_step_cache(conn, session_id: str):
    if not session_id:
        return {}
    step_payload, progress_payload = await asyncio.gather(
        _call_experiment_graph_tool_fast(
            conn,
            "get_step",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_progress_summary",
            {"session_id": session_id},
            priority="foreground",
        ),
    )
    conn.experiment_current_step = step_payload
    conn.experiment_progress_summary = progress_payload
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(
            step_payload,
            progress_payload,
        )
        if current_step_id:
            conn.experiment_current_step_id = current_step_id
            if getattr(conn, "experiment_resume_recovery_required", False):
                conn.experiment_resume_latest_current_step_id = current_step_id
    conn._experiment_graph_refresh_required = False
    return _enrich_experiment_step_meta_from_yaml(
        conn,
        _merge_experiment_step_meta(
            _extract_experiment_step_meta(step_payload),
            _extract_experiment_step_meta(progress_payload),
        ),
    )


def _step_supports_confirmation_autofill(step_payload) -> bool:
    interaction = _extract_experiment_step_interaction(step_payload)
    if not interaction:
        return False

    fast_path_mode = str(interaction.get("fast_path_mode", "")).strip().lower()
    capabilities = {
        str(item or "").strip().lower()
        for item in (interaction.get("capabilities") or [])
    }
    tags = {
        str(item or "").strip().lower()
        for item in (interaction.get("tags") or [])
    }
    return (
        fast_path_mode == "confirmation_step"
        or "step_confirmation" in capabilities
        or "confirmation_step" in tags
    )


def _build_experiment_autofill_fields(
    schema_by_name: dict,
    missing_fields,
    *,
    allow_confirmation_autofill: bool = False,
) -> dict:
    return {}


_RESUME_LOG_NEGATIVE_TOKENS = (
    "娌″仛",
    "杩樻病鍋?,
    "杩樻病鏈夊仛",
    "娌″仛濂?,
    "杩樻病鍋氬ソ",
    "娌″畬鎴?,
    "杩樻病瀹屾垚",
    "鍏堝埆",
    "涓嶈",
    "涓嶈",
    "娌″姞",
    "杩樻病鍔?,
)

_RESUME_LOG_GLOBAL_COMPLETION_TOKENS = (
    "鍏ㄩ儴瀹屾垚",
    "閮藉畬鎴?,
    "鍏ㄩ兘瀹屾垚",
    "鍏ㄩ儴鍋氬ソ",
    "閮藉仛濂?,
    "鍏ㄩ兘鍋氬ソ",
    "鍏ㄩ儴鏀惧ソ",
    "閮芥斁濂?,
    "鍏ㄩ兘鏀惧ソ",
    "鍏ㄩ儴鍔犲ソ",
    "閮藉姞濂戒簡",
    "鍏ㄩ兘鍔犲ソ浜?,
    "宸茬粡鍏ㄩ儴",
    "閮藉凡缁?,
)


def _resume_log_has_global_completion_signal(user_texts) -> bool:
    for text in user_texts or []:
        norm = _normalize_confirmation_signature(text)
        if not norm:
            continue
        if _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
            continue
        if _contains_any(norm, _RESUME_LOG_GLOBAL_COMPLETION_TOKENS):
            return True
    return False


def _resume_log_text_authorizes_photo(user_texts, *, allow_short_reply: bool = False) -> bool:
    positive_tokens = (
        "鍙互鎷嶇収",
        "鍙互鎷?,
        "鑳芥媿鐓?,
        "鑳芥媿",
        "鎷嶅惂",
        "鎷嶇収鍚?,
        "鐜板湪鎷?,
        "寮€濮嬫媿",
        "鎷嶄竴涓?,
        "鎷嶄竴寮?,
        "鍚屾剰鎷嶇収",
        "鎺堟潈鎷嶇収",
    )
    short_positive_tokens = ("鍙互", "濂?, "琛?, "鍚屾剰")
    for text in user_texts or []:
        norm = _normalize_confirmation_signature(text)
        if not norm or _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
            continue
        if _contains_any(norm, positive_tokens):
            return True
        if allow_short_reply and norm in short_positive_tokens:
            return True
    return False


def _step_meta_looks_like_photo_permission(step_meta: dict, schema_by_name: dict) -> bool:
    haystack = _normalize_confirmation_signature(
        " ".join(
            str(value or "")
            for value in (
                step_meta.get("step_id", ""),
                step_meta.get("title", ""),
                step_meta.get("instruction", ""),
                step_meta.get("description", ""),
                " ".join(schema_by_name.keys()),
            )
        )
    )
    if not haystack:
        return False
    return _contains_any(
        haystack,
        (
            "photo_permission",
            "鎷嶇収鏉冮檺",
            "鎺堟潈鎷嶇収",
            "鏄惁鎷嶇収",
            "璇㈤棶鏄惁鎷嶇収",
            "鍚屾剰鎷嶇収",
            "鍙互鎷嶇収",
        ),
    )


def _build_resume_photo_permission_fields(
    user_texts,
    schema_by_name: dict,
    missing_fields,
    step_meta: dict,
) -> dict:
    if not _step_meta_looks_like_photo_permission(step_meta, schema_by_name):
        return {}
    if not _resume_log_text_authorizes_photo(user_texts, allow_short_reply=True):
        return {}

    result = {}
    for field_name in missing_fields or []:
        field = schema_by_name.get(field_name, {})
        type_text = str(field.get("type", "")).strip().lower()
        if type_text not in {"bool", "boolean"}:
            continue
        haystack = _normalize_confirmation_signature(
            f"{field_name} {_clean_field_description(field.get('description', ''))}"
        )
        if _contains_any(
            haystack,
            (
                "photo_permission",
                "鎷嶇収鏉冮檺",
                "鎺堟潈鎷嶇収",
                "鍚屾剰鎷嶇収",
                "鍏佽鎷嶇収",
                "璇锋眰鎷嶇収",
                "璇㈤棶鎷嶇収",
            ),
        ):
            result[field_name] = True
    return result


def _find_resume_photo_meta_for_step(
    log_path: str,
    current_step_id: str,
    step_meta: dict,
    user_texts,
) -> dict:
    data_dir = Path(str(log_path or "")).parent
    if not data_dir.exists() or not data_dir.is_dir():
        return {}

    sample_index = _extract_sample_index_from_text(
        " ".join(
            str(value or "")
            for value in (
                current_step_id,
                step_meta.get("title", ""),
                step_meta.get("instruction", ""),
                step_meta.get("description", ""),
                " ".join(str(text or "") for text in user_texts or []),
            )
        )
    )

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    candidates = [
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes
    ]
    if not candidates:
        return {}

    def _rank(path: Path) -> tuple[int, float]:
        name = _normalize_confirmation_signature(path.stem)
        score = 0
        if sample_index is not None:
            sample_tokens = (
                f"{sample_index}鍙锋牱鍝?,
                f"{sample_index}鍙?,
                f"鏍峰搧{sample_index}",
            )
            if any(token in name for token in sample_tokens):
                score += 100
            else:
                return (-1, 0.0)
        if _contains_any(name, ("鐓х墖", "鎷嶇収", "photo", "image", "img")):
            score += 20
        if "閲嶆媿" in name or "琛ユ媿" in name:
            score += 5
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (score, mtime)

    ranked = sorted(
        ((rank, path) for path in candidates for rank in [_rank(path)] if rank[0] >= 0),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked:
        return {}

    selected = ranked[0][1]
    return {
        "found": True,
        "file_name": selected.name,
        "photo_path": str(selected.resolve()),
    }


def _build_experiment_resume_log_autofill_fields(
    user_texts,
    schema_by_name: dict,
    missing_fields,
    *,
    allow_confirmation_autofill: bool = False,
    allow_observation_autofill: bool = False,
    allow_photo_autofill: bool = False,
    step_payload=None,
    step_meta: dict | None = None,
    current_step_id: str = "",
    log_path: str = "",
) -> dict:
    signatures = []
    for text in user_texts or []:
        norm = _normalize_confirmation_signature(text)
        if not norm or _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
            continue
        signatures.append(norm)

    if not signatures and not allow_photo_autofill:
        return {}

    global_completion = _resume_log_has_global_completion_signal(user_texts)
    result = {}
    if allow_confirmation_autofill:
        for field_name in missing_fields or []:
            field = schema_by_name.get(field_name, {})
            type_text = str(field.get("type", "")).strip().lower()
            if type_text not in {"bool", "boolean"}:
                continue
            description = _normalize_confirmation_signature(
                _clean_field_description(field.get("description", ""))
            )
            if not description:
                continue
            if global_completion or any(
                _looks_like_confirmation_signature_match(signature, description)
                for signature in signatures
            ):
                result[field_name] = True

    remaining_fields = [
        field_name for field_name in (missing_fields or []) if field_name not in result
    ]
    meta = step_meta or _extract_experiment_step_meta(step_payload)
    if remaining_fields:
        result.update(
            _build_resume_photo_permission_fields(
                user_texts,
                schema_by_name,
                remaining_fields,
                meta,
            )
        )

    if allow_observation_autofill and _step_supports_observation_report(
        step_payload,
        schema_by_name,
    ):
        for text in user_texts or []:
            remaining_fields = [
                field_name
                for field_name in (missing_fields or [])
                if field_name not in result
            ]
            if not remaining_fields:
                break
            observation_fields = _build_experiment_current_step_observation_fields(
                text,
                schema_by_name,
                remaining_fields,
                allow_bool_completion=True,
            )
            result.update(
                {
                    key: value
                    for key, value in observation_fields.items()
                    if value not in (None, "")
                }
            )

    remaining_fields = [
        field_name for field_name in (missing_fields or []) if field_name not in result
    ]
    if allow_photo_autofill and remaining_fields and (
        _step_meta_looks_like_photo_confirmation(meta)
        or bool(set(remaining_fields) & _PHOTO_CONFIRMATION_FIELD_NAMES)
    ):
        photo_meta = _find_resume_photo_meta_for_step(
            log_path,
            current_step_id,
            meta,
            user_texts,
        )
        if photo_meta:
            result.update(
                {
                    key: value
                    for key, value in _build_experiment_photo_writeback_fields(
                        schema_by_name,
                        photo_meta,
                    ).items()
                    if key in remaining_fields and value not in (None, "")
                }
            )
    return result


def _looks_like_explicit_completion_report(filtered_text: str) -> bool:
    norm = _normalize_confirmation_signature(filtered_text)
    if not norm:
        return False
    if _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
        return False
    if _looks_like_pure_short_completion_control(norm):
        return True
    completion_tokens = (
        "鍏ㄩ儴瀹屾垚",
        "閮藉畬鎴?,
        "鍏ㄩ兘瀹屾垚",
        "宸茬粡瀹屾垚",
        "宸插畬鎴?,
        "瀹屾垚浜?,
        "鍏ㄩ儴鍋氬ソ",
        "閮藉仛濂?,
        "鍏ㄩ兘鍋氬ソ",
        "鍋氬ソ浜?,
        "鍋氬畬浜?,
        "鍏ㄩ儴娣峰寑",
        "閮芥贩鍖€浜?,
        "鍏ㄩ兘娣峰寑浜?,
        "宸茬粡鍏ㄩ儴娣峰寑",
        "宸插叏閮ㄦ贩鍖€",
        "娣峰悎鍧囧寑",
        "娣峰寑浜?,
        "寮€濮嬫悈鎷?,
        "宸茬粡寮€濮嬫悈鎷?,
        "宸插紑濮嬫悈鎷?,
        "閮藉凡缁忓紑濮嬫悈鎷?,
        "鎼呮媽濂戒簡",
    )
    return _contains_any(norm, completion_tokens)


def _looks_like_explicit_added_completion_report(filtered_text: str) -> bool:
    norm = _normalize_confirmation_signature(filtered_text)
    if not norm:
        return False
    if _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
        return False
    completion_tokens = (
        "鍏ㄩ儴鍔犲ソ",
        "閮藉姞濂戒簡",
        "鍏ㄩ兘鍔犲ソ浜?,
        "宸茬粡鍔犲ソ浜?,
        "宸插姞濂戒簡",
        "鍏ㄩ儴鍔犲畬",
        "閮藉姞瀹屼簡",
        "鍏ㄩ兘鍔犲畬浜?,
        "宸茬粡鍔犲畬浜?,
        "宸插姞瀹屼簡",
        "鍏ㄩ儴鍔犲叆",
        "閮藉姞鍏ヤ簡",
        "鍏ㄩ兘鍔犲叆浜?,
        "宸茬粡鍔犲叆",
        "宸插姞鍏?,
        "閮藉凡缁忓姞鍏?,
        "鍏ㄩ兘宸茬粡鍔犲叆",
        "鎸夐『搴忓姞鍏?,
        "椤哄簭鍔犲叆",
    )
    return _contains_any(norm, completion_tokens)


def _looks_like_addition_confirmation_description(description: str) -> bool:
    norm = _normalize_confirmation_signature(description)
    if not norm:
        return False
    return _contains_any(
        norm,
        (
            "鍔犲叆",
            "鍔犳恫",
            "婊村姞",
            "鍔犲畬",
            "鍔犲ソ",
        ),
    )


def _build_experiment_current_step_confirmation_fields(
    filtered_text: str,
    schema_by_name: dict,
    missing_fields,
    *,
    allow_confirmation_autofill: bool = False,
) -> dict:
    norm = _normalize_confirmation_signature(filtered_text)
    if not norm:
        return {}
    if _looks_like_question_reply(filtered_text):
        return {}
    if _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
        return {}

    global_completion = _looks_like_explicit_completion_report(filtered_text)
    added_completion = _looks_like_explicit_added_completion_report(filtered_text)
    result = {}
    for field_name in missing_fields or []:
        field = schema_by_name.get(field_name, {})
        type_text = str(field.get("type", "")).strip().lower()
        if type_text not in {"bool", "boolean"}:
            continue
        description = _normalize_confirmation_signature(
            _clean_field_description(field.get("description", ""))
        )
        if not description:
            continue
        if (
            global_completion
            or (
                added_completion
                and _looks_like_addition_confirmation_description(description)
            )
            or _looks_like_confirmation_signature_match(norm, description)
        ):
            result[field_name] = True
    return result


_CHINESE_NUMERIC_CHAR_MAP = {
    "闆?: "0",
    "涓€": "1",
    "浜?: "2",
    "涓?: "2",
    "涓?: "3",
    "鍥?: "4",
    "浜?: "5",
    "鍏?: "6",
    "涓?: "7",
    "鍏?: "8",
    "涔?: "9",
}


def _parse_small_chinese_float(token: str) -> float | None:
    text = str(token or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return _extract_float_value(text)

    normalized = text.replace("涓?, "浜?)
    if normalized == "鍗?:
        return 0.5
    if "鐐? in normalized:
        left, right = normalized.split("鐐?, 1)
        if not right:
            return None
        left_value = _parse_small_chinese_integer(left or "闆?)
        if left_value is None:
            return None
        decimal_digits = "".join(
            _CHINESE_NUMERIC_CHAR_MAP.get(char, "")
            for char in right
            if char in _CHINESE_NUMERIC_CHAR_MAP
        )
        if not decimal_digits:
            return None
        return _extract_float_value(f"{left_value}.{decimal_digits}")

    integer_value = _parse_small_chinese_integer(normalized)
    if integer_value is None:
        return None
    return float(integer_value)


def _extract_observation_duration_minutes(filtered_text: str) -> float | None:
    text = textUtils.normalize_spoken_text(filtered_text or "")
    if not text:
        return None

    parsed_durations = textUtils.extract_spoken_duration_expressions(text)
    if parsed_durations:
        return float(parsed_durations[0]["minutes"])

    numeric_match = re.search(r"(\d+(?:\.\d+)?)\s*(鍒嗛挓|鍒唡绉掗挓|绉?", text)
    if numeric_match:
        value = _extract_float_value(numeric_match.group(1))
        if value is None:
            return None
        unit = numeric_match.group(2)
        return round(value / 60.0, 4) if "绉? in unit else value

    chinese_match = re.search(
        r"([闆朵竴浜屼袱涓夊洓浜斿叚涓冨叓涔濆崄鐐瑰崐]+)\s*(鍒嗛挓|鍒唡绉掗挓|绉?",
        text,
    )
    if not chinese_match:
        return None
    value = _parse_small_chinese_float(chinese_match.group(1))
    if value is None:
        return None
    unit = chinese_match.group(2)
    return round(value / 60.0, 4) if "绉? in unit else value


def _field_matches_observation_semantics(field_name: str, field: dict, tokens: tuple[str, ...]) -> bool:
    haystack = _normalize_confirmation_signature(
        f"{field_name} {_clean_field_description(field.get('description', ''))}"
    )
    if not haystack:
        return False
    return _contains_any(haystack, tokens)


def _extract_observation_color_value(filtered_text: str) -> str:
    text = textUtils.normalize_spoken_text(filtered_text or "")
    norm = _normalize_confirmation_signature(filtered_text)
    if (
        not text
        or _looks_like_question_reply(filtered_text)
        or _looks_like_pure_short_completion_control(norm)
        or _looks_like_explicit_completion_report(filtered_text)
        or _looks_like_explicit_added_completion_report(filtered_text)
    ):
        return ""

    cleaned = re.sub(r"\d+(?:\.\d+)?\s*(鍒嗛挓|鍒唡绉掗挓|绉?", "", text)
    cleaned = re.sub(r"[闆朵竴浜屼袱涓夊洓浜斿叚涓冨叓涔濆崄鐐瑰崐]+\s*(鍒嗛挓|鍒唡绉掗挓|绉?", "", cleaned)
    cleaned = re.sub(r"(棰滆壊|鏈€缁坾绋冲畾|鐢ㄤ簡|鐢ㄦ椂|澶х害|澶ф|绾鏄瘄涓簗璁颁綔|宸茬粡|纭|鏍峰搧)", "", cleaned)
    cleaned = re.sub(r"[0-9涓€浜屼笁鍥涗簲鍏竷鍏節鍗乚+\s*鍙?, "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned).strip("锛?銆傦紱;锛? ")
    if not cleaned:
        return ""
    if len(cleaned) > 12:
        return ""
    if _contains_any(cleaned, ("鍙互鎷嶇収", "涓嬩竴姝?, "缁х画", "鐒跺悗鍛?, "骞插槢", "娌￠敊", "瀵?)):
        return ""
    return cleaned


def _build_experiment_current_step_tyndall_fields(
    filtered_text: str,
    schema_by_name: dict,
    missing_fields,
) -> dict:
    norm = _normalize_confirmation_signature(filtered_text)
    if not norm or _looks_like_question_reply(filtered_text):
        return {}
    if not _contains_any(norm, ("涓佽揪灏?, "tyndall")):
        return {}

    tyndall_fields = []
    field_names = list(missing_fields or schema_by_name.keys())
    for field_name in field_names:
        field = schema_by_name.get(field_name, {})
        type_text = str(field.get("type", "")).strip().lower()
        if type_text not in {"bool", "boolean"}:
            continue
        if not _field_matches_observation_semantics(
            field_name,
            field,
            ("涓佽揪灏?, "tyndall"),
        ):
            continue
        tyndall_fields.append(field_name)

    if not tyndall_fields:
        return {}

    collective_positive = (
        _contains_any(
            norm,
            (
                "鍏ㄩ儴閮芥湁",
                "鍏ㄩ兘鏈?,
                "閮芥湁",
                "鍧囨湁",
                "閮借瀵熷埌",
                "閮界湅鍒颁簡",
                "閮借兘鐪嬪埌",
                "閮藉瓨鍦?,
                "閮芥湁鏄庢樉",
            ),
        )
        and _contains_any(norm, ("涓佽揪灏?, "tyndall"))
    )
    collective_negative = (
        _contains_any(
            norm,
            (
                "鍏ㄩ儴閮芥病鏈?,
                "鍏ㄩ兘娌℃湁",
                "閮芥病鏈?,
                "鍧囨棤",
                "閮界湅涓嶅埌",
                "閮芥病鐪嬪埌",
                "閮芥湭瑙傚療鍒?,
                "閮戒笉瀛樺湪",
            ),
        )
        and _contains_any(norm, ("涓佽揪灏?, "tyndall"))
    )

    if collective_positive == collective_negative:
        return {}
    return {field_name: collective_positive for field_name in tyndall_fields}


def _step_supports_observation_report(step_payload, schema_by_name: dict) -> bool:
    interaction = _extract_experiment_step_interaction(step_payload)
    fast_path_mode = str(interaction.get("fast_path_mode", "")).strip().lower()
    capabilities = {
        str(item or "").strip().lower()
        for item in (interaction.get("capabilities") or [])
    }
    if fast_path_mode == "observation_record_step" or "observation_capture" in capabilities:
        return True

    for field_name, field in (schema_by_name or {}).items():
        if _field_matches_observation_semantics(
            field_name,
            field,
            ("鏈€缁堥鑹?, "棰滆壊绋冲畾鎵€鐢ㄦ椂闂?, "鍙嶅簲鏃堕棿", "棰滆壊绋冲畾"),
        ):
            return True
    return False


def _build_experiment_current_step_observation_fields(
    filtered_text: str,
    schema_by_name: dict,
    missing_fields,
    *,
    allow_bool_completion: bool = False,
) -> dict:
    if not allow_bool_completion:
        return {}

    observation_fields = _build_experiment_current_step_tyndall_fields(
        filtered_text,
        schema_by_name,
        missing_fields,
    )
    duration_minutes = _extract_observation_duration_minutes(filtered_text)
    color_value = _extract_observation_color_value(filtered_text)

    color_field_name = ""
    time_field_name = ""
    stable_field_name = ""
    bool_field_names = []
    for field_name in missing_fields or []:
        field = schema_by_name.get(field_name, {})
        type_text = str(field.get("type", "")).strip().lower()
        if type_text in {"bool", "boolean"}:
            bool_field_names.append(field_name)
            if _field_matches_observation_semantics(field_name, field, ("棰滆壊绋冲畾", "纭棰滆壊绋冲畾")):
                stable_field_name = stable_field_name or field_name
            continue
        if type_text in {"string", "str"} and _field_matches_observation_semantics(
            field_name, field, ("鏈€缁堥鑹?, "棰滆壊")
        ):
            color_field_name = color_field_name or field_name
        if type_text in {"float", "number", "int", "integer"} and _field_matches_observation_semantics(
            field_name,
            field,
            ("棰滆壊绋冲畾鎵€鐢ㄦ椂闂?, "鍙嶅簲鏃堕棿", "绋冲畾鏃堕棿", "鎵€鐢ㄦ椂闂?),
        ):
            time_field_name = time_field_name or field_name

    if color_field_name and color_value:
        observation_fields[color_field_name] = color_value
    if time_field_name and duration_minutes is not None:
        observation_fields[time_field_name] = duration_minutes
    if stable_field_name and (
        duration_minutes is not None or "绋冲畾" in textUtils.normalize_spoken_text(filtered_text or "")
    ):
        observation_fields[stable_field_name] = True

    if color_value and duration_minutes is not None:
        for field_name in bool_field_names:
            observation_fields.setdefault(field_name, True)

    return observation_fields


async def _try_apply_current_confirmation_report(
    conn,
    filtered_text: str,
) -> str | None:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return None

    try:
        step_payload, progress_payload, schema_payload = await asyncio.gather(
            _call_experiment_graph_tool_fast(
                conn,
                "get_step",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_current_progress",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_schema",
                {"session_id": session_id},
                priority="foreground",
            ),
        )
    except Exception:
        return None

    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        try:
            start_payload = await _call_experiment_graph_tool_fast(
                conn,
                "start_trial",
                {"session_id": session_id},
                priority="foreground",
            )
        except Exception:
            return None
        current_progress = _extract_experiment_current_progress(start_payload)

    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    if not missing_fields:
        return None

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    write_fields = _build_experiment_current_step_confirmation_fields(
        filtered_text,
        schema_by_name,
        missing_fields,
        allow_confirmation_autofill=True,
    )
    if _step_supports_observation_report(step_payload, schema_by_name):
        observation_fields = _build_experiment_current_step_observation_fields(
            filtered_text,
            schema_by_name,
            missing_fields,
            allow_bool_completion=True,
        )
        write_fields.update(
            {
                key: value
                for key, value in observation_fields.items()
                if value not in (None, "")
            }
        )
    if not write_fields:
        bool_prompts = []
        for field_name in missing_fields or []:
            field = schema_by_name.get(field_name, {})
            type_text = str(field.get("type", "")).strip().lower()
            if type_text not in {"bool", "boolean"}:
                continue
            description = _clean_field_description(field.get("description", ""))
            bool_prompts.append(description or str(field_name or ""))
        conn.logger.bind(tag=TAG).info(
            "experiment confirmation writeback no-match: "
            f"text={filtered_text}, missing_fields={missing_fields}, "
            f"bool_prompts={bool_prompts}"
        )
        return None

    conn.logger.bind(tag=TAG).info(
        "experiment confirmation writeback fast path hit: "
        f"text={filtered_text}, write_fields={sorted(write_fields.keys())}"
    )

    group_number = await _sync_exp2_group_number_from_turn(
        conn,
        "",
        filtered_text,
        preferred_step_id=_get_current_experiment_step_id(conn),
    )
    completed, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=write_fields,
        auto_advance=True,
        fallback_reply="鎴戝厛璁颁笅浜嗗綋鍓嶈繖涓€姝ョ殑纭缁撴灉銆?,
        group_number=group_number,
    )
    if completed or reply:
        return reply
    return None


def _compose_confirmation_step_writeback_block_reply(
    conn,
    step_payload,
    progress_payload,
    schema_payload,
) -> str | None:
    if not _step_supports_confirmation_autofill(step_payload):
        return None

    current_progress = _extract_experiment_current_progress(progress_payload)
    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    if not missing_fields:
        return None

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    step_meta = _merge_experiment_step_meta(
        _extract_experiment_step_meta(step_payload),
        _extract_experiment_step_meta(getattr(conn, "experiment_progress_summary", None)),
    )
    missing_reply = _compose_missing_field_reply(missing_fields, schema_by_name)
    current_step_reply = _compose_experiment_step_reply(step_meta, mode="guide")
    if missing_reply and current_step_reply:
        return f"鎴戣繖杈硅繕娌℃妸褰撳墠杩欎竴姝ユ垚鍔熷啓鍏ュ疄楠岃褰曘€倇missing_reply}"
    if missing_reply:
        return f"鎴戣繖杈硅繕娌℃妸褰撳墠杩欎竴姝ユ垚鍔熷啓鍏ュ疄楠岃褰曘€倇missing_reply}"
    if current_step_reply:
        return current_step_reply
    return "鎴戣繖杈硅繕娌℃妸褰撳墠杩欎竴姝ユ垚鍔熷啓鍏ュ疄楠岃褰曪紝鍏堟寜褰撳墠杩欎竴姝ョ户缁‘璁ゅ悗鍐嶅憡璇夋垜銆?


async def _load_confirmation_step_writeback_block_reply(
    conn,
    session_id: str,
) -> str | None:
    session_id = str(session_id or "").strip()
    if not session_id:
        return None

    try:
        step_payload, progress_payload, schema_payload = await asyncio.gather(
            _call_experiment_graph_tool_fast(
                conn,
                "get_step",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_current_progress",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_schema",
                {"session_id": session_id},
                priority="foreground",
            ),
        )
    except Exception:
        return None

    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        try:
            start_payload = await _call_experiment_graph_tool_fast(
                conn,
                "start_trial",
                {"session_id": session_id},
                priority="foreground",
            )
        except Exception:
            return None
        current_progress = _extract_experiment_current_progress(start_payload)
        if isinstance(current_progress, dict):
            progress_payload = start_payload

    return _compose_confirmation_step_writeback_block_reply(
        conn,
        step_payload,
        progress_payload,
        schema_payload,
    )


def _collect_resume_log_user_texts_by_step(log_path: str) -> dict[str, list[str]]:
    if not log_path:
        return {}

    grouped: dict[str, list[str]] = {}
    for entry in experiment_resume.read_transcript_entries(log_path):
        if str(entry.get("role", "")).strip().upper() != "USER":
            continue
        step_id = str(entry.get("current_step_id", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not step_id or not text:
            continue
        grouped.setdefault(step_id, []).append(text)
    return grouped


def _normalize_device_id_for_report_compare(device_id: str) -> str:
    return str(device_id or "").strip().lower().replace(":", "_").replace("-", "_")


def _normalize_path_text_for_compare(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).resolve()).lower().replace("/", "\\")
    except Exception:
        return raw.lower().replace("/", "\\")


def _infer_resume_target_step_from_exported_reports(conn) -> str:
    report = _best_resume_report_candidate(conn)
    if isinstance(report, dict):
        return str(report.get("current_step_id", "") or "").strip()
    return ""


def _best_resume_report_candidate(conn) -> dict | None:
    yaml_path = str(getattr(conn, "experiment_yaml_path", "") or "").strip()
    normalized_yaml_path = _normalize_path_text_for_compare(yaml_path)
    normalized_device_id = _normalize_device_id_for_report_compare(
        str(getattr(conn, "device_id", "") or "").strip()
    )
    if not normalized_yaml_path or not normalized_device_id:
        return ""

    try:
        data_root = Path(yaml_path).resolve().parent.parent / "data"
    except Exception:
        return ""
    if not data_root.exists():
        return ""

    step_order = _resolve_experiment_yaml_step_order(conn)
    if not step_order:
        return ""
    index_by_id = {step_id: idx for idx, step_id in enumerate(step_order)}

    best_step_id = ""
    best_score = (-1, -1, "")
    best_payload = None
    for report_path in data_root.rglob("experimental_graph_records.yaml"):
        try:
            payload = yaml.safe_load(
                report_path.read_text(encoding="utf-8", errors="ignore")
            )
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        source_yaml_path = _normalize_path_text_for_compare(
            str(payload.get("source_yaml_path", "") or "").strip()
        )
        report_device_id = _normalize_device_id_for_report_compare(
            str(payload.get("device_id", "") or "").strip()
        )
        current_step_id = str(payload.get("current_step_id", "") or "").strip()
        if source_yaml_path != normalized_yaml_path:
            continue
        if report_device_id != normalized_device_id:
            continue
        step_index = index_by_id.get(current_step_id, -1)
        if step_index < 0:
            continue
        completed_count = len(payload.get("completed_step_ids") or [])
        score = (step_index, completed_count, str(report_path))
        if score > best_score:
            best_score = score
            best_step_id = current_step_id
            best_payload = {
                "report_path": str(report_path),
                "session_id": str(payload.get("session_id", "") or "").strip(),
                "current_step_id": current_step_id,
                "current_group_number": payload.get("current_group_number"),
                "completed_steps_count": completed_count,
            }

    if best_payload is None or not best_step_id:
        return None
    return best_payload


def _infer_experiment_resume_target_step_from_log(conn, log_path: str) -> str:
    path_text = str(log_path or "").strip()
    if not path_text:
        return ""
    try:
        log_tail = Path(path_text).read_text(encoding="utf-8", errors="ignore")[-8000:]
    except Exception:
        return ""
    if not log_tail.strip():
        return ""

    order = _resolve_experiment_yaml_step_order(conn)
    if not order:
        return ""
    step_by_id = _resolve_experiment_step_by_id(conn)

    best_step_id = ""
    best_score = 0.0
    for index, step_id in enumerate(order):
        step = step_by_id.get(step_id)
        if not isinstance(step, dict):
            continue
        score = _yaml_step_context_match_score(log_tail, step)
        score += min(0.08, index * 0.002)
        if score > best_score:
            best_step_id = step_id
            best_score = score

    if best_score < 0.46:
        return ""
    return best_step_id


def _resolve_experiment_yaml_step_order(conn) -> list[str]:
    step_ids = []
    for step in _resolve_experiment_yaml_steps(conn):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id", "") or "").strip()
        if step_id:
            step_ids.append(step_id)
    return step_ids


def _compare_experiment_step_order(conn, left_step_id: str, right_step_id: str) -> int | None:
    left = str(left_step_id or "").strip()
    right = str(right_step_id or "").strip()
    if not left or not right:
        return None
    if left == right:
        return 0
    step_order = _resolve_experiment_yaml_step_order(conn)
    if not step_order:
        return None
    try:
        left_index = step_order.index(left)
        right_index = step_order.index(right)
    except ValueError:
        return None
    if left_index < right_index:
        return -1
    if left_index > right_index:
        return 1
    return 0


def _compose_resume_log_recovery_blocked_reply(
    step_meta: dict,
    missing_fields,
    schema_by_name: dict,
    *,
    has_log_evidence: bool,
) -> str:
    title = str(step_meta.get("title", "") or step_meta.get("step_id", "") or "").strip()
    missing_reply = _compose_missing_field_reply(missing_fields, schema_by_name)
    if title:
        if has_log_evidence:
            return (
                f"鎴戞壘鍒颁笂娆″疄楠屾棩蹇椾簡锛屼絾鍦ㄢ€渰title}鈥濊繖涓€姝ワ紝鏃ュ織閲岀殑姹囨姤杩樹笉澶熸垜鑷姩琛ラ綈銆?
                f"{missing_reply}"
            )
        return (
            f"鎴戞壘鍒颁笂娆″疄楠屾棩蹇椾簡锛屼絾鏃ュ織閲岃繕娌℃湁瓒冲鍐呭璇佹槑鈥渰title}鈥濊繖涓€姝ュ凡缁忓仛瀹屻€?
            f"{missing_reply}"
        )
    if has_log_evidence:
        return (
            "鎴戞壘鍒颁笂娆″疄楠屾棩蹇椾簡锛屼絾鏃ュ織閲岀殑姹囨姤杩樹笉澶熸垜鑷姩琛ラ綈褰撳墠杩欎竴姝ャ€?
            f"{missing_reply}"
        )
    return (
        "鎴戞壘鍒颁笂娆″疄楠屾棩蹇椾簡锛屼絾鏃ュ織閲岃繕娌℃湁瓒冲鍐呭璇佹槑褰撳墠杩欎竴姝ュ凡缁忓仛瀹屻€?
        f"{missing_reply}"
    )


async def _replay_experiment_progress_from_resume_log(
    conn,
    target_step_id: str,
    log_path: str,
) -> tuple[bool, str]:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    target_step_id = str(target_step_id or "").strip()
    log_path = str(log_path or "").strip()
    if not session_id or not target_step_id or not log_path:
        return False, ""

    step_order = _resolve_experiment_yaml_step_order(conn)
    if not step_order or target_step_id not in step_order:
        return False, (
            "鎴戞壘鍒颁簡涓婃瀹為獙鏃ュ織锛屼絾褰撳墠瀹為獙 YAML 閲屾病娉曞彲闈犺В鏋愬嚭鎭㈠椤哄簭锛?
            "鐜板湪杩樹笉鑳藉畨鍏ㄥ湴鑷姩缁仛銆?
        )

    current_step_id = _get_current_experiment_step_id(conn)
    if not current_step_id:
        step_meta = await _safe_refresh_experiment_step_cache(
            conn,
            session_id,
            reason="explicit_resume_recovery_bootstrap",
        )
        current_step_id = str(step_meta.get("step_id", "") or "").strip()

    if not current_step_id:
        return False, ""
    if current_step_id not in step_order:
        return False, ""

    target_index = step_order.index(target_step_id)
    current_index = step_order.index(current_step_id)
    if current_index > target_index:
        return False, ""
    if current_step_id == target_step_id:
        return True, ""

    step_user_texts = _collect_resume_log_user_texts_by_step(log_path)

    while current_step_id and current_step_id != target_step_id:
        try:
            step_payload, progress_payload, schema_payload = await asyncio.gather(
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_step",
                    {"session_id": session_id},
                    priority="foreground",
                ),
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_current_progress",
                    {"session_id": session_id},
                    priority="foreground",
                ),
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_schema",
                    {"session_id": session_id},
                    priority="foreground",
                ),
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment explicit resume replay load failed: {exc}"
            )
            return False, ""

        step_meta = _merge_experiment_step_meta(
            _extract_experiment_step_meta(step_payload),
            _extract_experiment_step_meta(getattr(conn, "experiment_progress_summary", None)),
        )
        current_progress = _extract_experiment_current_progress(progress_payload)
        if current_progress is None:
            try:
                start_payload = await _call_experiment_graph_tool_fast(
                    conn,
                    "start_trial",
                    {"session_id": session_id},
                    priority="foreground",
                )
            except Exception as exc:
                conn.logger.bind(tag=TAG).warning(
                    f"experiment explicit resume start_trial failed: {exc}"
                )
                return False, ""
            current_progress = _extract_experiment_current_progress(start_payload)

        schema_by_name = _extract_experiment_schema_view(schema_payload)
        missing_fields = list((current_progress or {}).get("missing_fields") or [])
        current_user_texts = step_user_texts.get(current_step_id, [])

        autofill_fields = _build_experiment_resume_log_autofill_fields(
            current_user_texts,
            schema_by_name,
            missing_fields,
            allow_confirmation_autofill=_step_supports_confirmation_autofill(
                step_payload
            ),
            allow_observation_autofill=True,
            allow_photo_autofill=True,
            step_payload=step_payload,
            step_meta=step_meta,
            current_step_id=current_step_id,
            log_path=log_path,
        )
        if autofill_fields:
            add_fields_payload = await _call_experiment_graph_tool_fast(
                conn,
                "add_fields",
                {"session_id": session_id, "data": autofill_fields},
                priority="foreground",
            )
            updated_progress = _extract_experiment_current_progress(add_fields_payload)
            if isinstance(updated_progress, dict):
                current_progress = updated_progress
            missing_fields = list((current_progress or {}).get("missing_fields") or [])

        if missing_fields:
            return False, _compose_resume_log_recovery_blocked_reply(
                step_meta,
                missing_fields,
                schema_by_name,
                has_log_evidence=bool(current_user_texts),
            )

        finish_payload = await _call_experiment_graph_tool_fast(
            conn,
            "finish_trial",
            {"session_id": session_id, "validate": True},
            priority="foreground",
        )
        if not bool(_experiment_result_body(finish_payload).get("ok")):
            message = _extract_experiment_result_message(finish_payload)
            return False, message or _compose_resume_log_recovery_blocked_reply(
                step_meta,
                [],
                schema_by_name,
                has_log_evidence=bool(current_user_texts),
            )

        can_proceed_payload = await _call_experiment_graph_tool_fast(
            conn,
            "can_proceed",
            {"session_id": session_id},
            priority="foreground",
        )
        if not bool(_experiment_result_body(can_proceed_payload).get("ok")):
            message = _extract_experiment_result_message(can_proceed_payload)
            return False, message or _compose_resume_log_recovery_blocked_reply(
                step_meta,
                [],
                schema_by_name,
                has_log_evidence=bool(current_user_texts),
            )

        proceed_payload = await _call_experiment_graph_tool_fast(
            conn,
            "proceed_to_next_step",
            {"session_id": session_id},
            priority="foreground",
        )
        if not bool(_experiment_result_body(proceed_payload).get("ok")):
            message = _extract_experiment_result_message(proceed_payload)
            return False, message or _compose_resume_log_recovery_blocked_reply(
                step_meta,
                [],
                schema_by_name,
                has_log_evidence=bool(current_user_texts),
            )

        next_meta = await _safe_refresh_experiment_step_cache(
            conn,
            session_id,
            reason="explicit_resume_replay_proceed",
        )
        current_step_id = str(next_meta.get("step_id", "") or "").strip()
        if not current_step_id:
            current_step_id = _get_current_experiment_step_id(conn)
        if not current_step_id:
            return False, ""
        if current_step_id not in step_order:
            return False, ""

    return current_step_id == target_step_id, ""


def _looks_like_confirmation_field_statement(
    filtered_text: str,
    missing_fields,
    schema_by_name: dict,
) -> bool:
    norm = _normalize_confirmation_signature(filtered_text)
    if not norm or len(norm) > 80:
        return False
    if _looks_like_experiment_detail_request(norm):
        return False
    if _looks_like_question_reply(filtered_text):
        return False

    negative_tokens = (
        "娌″仛",
        "杩樻病鍋?,
        "杩樻病鏈夊仛",
        "娌″仛濂?,
        "杩樻病鍋氬ソ",
        "娌″畬鎴?,
        "杩樻病瀹屾垚",
        "鍏堝埆",
        "涓嶈",
        "涓嶈",
        "娌″姞",
        "杩樻病鍔?,
    )
    if _contains_any(norm, negative_tokens):
        return False

    for field_name in missing_fields or []:
        field = schema_by_name.get(field_name, {})
        type_text = str(field.get("type", "")).strip().lower()
        if type_text not in {"bool", "boolean"}:
            continue
        description = _normalize_confirmation_signature(
            _clean_field_description(field.get("description", ""))
        )
        if not description:
            continue
        if _looks_like_confirmation_signature_match(norm, description):
            return True
    return False


async def _handle_confirmation_step_semantic_fast_intent(
    conn,
    original_text: str,
    filtered_text: str,
) -> bool:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False

    try:
        step_payload, progress_payload, schema_payload = await asyncio.gather(
            _call_experiment_graph_tool_fast(
                conn,
                "get_step",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_current_progress",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_schema",
                {"session_id": session_id},
                priority="foreground",
            ),
        )
    except Exception:
        return False

    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        try:
            start_payload = await _call_experiment_graph_tool_fast(
                conn,
                "start_trial",
                {"session_id": session_id},
                priority="foreground",
            )
        except Exception:
            return False
        current_progress = _extract_experiment_current_progress(start_payload)

    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    if not missing_fields:
        return False

    try:
        reply = await _try_apply_current_confirmation_report(conn, filtered_text)
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment confirmation semantic fast writeback failed: {exc}"
        )
        return False

    if not reply:
        reply = _compose_confirmation_step_writeback_block_reply(
            conn,
            step_payload,
            progress_payload,
            schema_payload,
        )
        if not reply:
            return False

    next_step_meta = _get_cached_experiment_step_meta(conn)
    reply = _prepare_fastpath_spoken_reply(
        reply,
        fallback_step_meta=next_step_meta,
        fallback_mode="guide",
    )
    if not reply:
        return False

    await _start_direct_intent_turn(conn, original_text)
    if hasattr(conn, "enrich_latest_clean_user_utterance_snapshot"):
        try:
            conn.enrich_latest_clean_user_utterance_snapshot()
        except Exception:
            pass
    speak_txt(conn, reply)
    return True


def _build_experiment_photo_writeback_fields(
    schema_by_name: dict,
    photo_meta: dict,
    *,
    step_meta: dict | None = None,
    missing_fields=None,
) -> dict:
    result = {}
    normalized_missing = {
        str(item or "").strip() for item in (missing_fields or []) if str(item or "").strip()
    }

    if step_meta and _step_meta_looks_like_photo_permission(step_meta, schema_by_name):
        for field_name, field in (schema_by_name or {}).items():
            if normalized_missing and field_name not in normalized_missing:
                continue
            type_text = str((field or {}).get("type", "")).strip().lower()
            if type_text not in {"bool", "boolean"}:
                continue
            haystack = _normalize_confirmation_signature(
                f"{field_name} {_clean_field_description((field or {}).get('description', ''))}"
            )
            if _contains_any(
                haystack,
                (
                    "photo_permission",
                    "鎷嶇収鏉冮檺",
                    "鎺堟潈鎷嶇収",
                    "鍚屾剰鎷嶇収",
                    "鍏佽鎷嶇収",
                    "璇锋眰鎷嶇収",
                    "璇㈤棶鎷嶇収",
                    "鍙互鎷嶇収",
                ),
            ):
                result[field_name] = True

    if "photo_taken" in schema_by_name:
        result["photo_taken"] = True
    if "color_confirmed_by_photo" in schema_by_name:
        result["color_confirmed_by_photo"] = True

    file_name = str(photo_meta.get("file_name", "") or "").strip()
    photo_path = str(photo_meta.get("photo_path", "") or "").strip()
    if file_name and "photo_file_name" in schema_by_name:
        result["photo_file_name"] = file_name
    if photo_path and "photo_path" in schema_by_name:
        result["photo_path"] = photo_path
    return result


def _step_meta_looks_like_photo_confirmation(step_meta: dict) -> bool:
    title = _normalize_text_for_match(step_meta.get("title", ""))
    if title and _contains_any(title, ("鎷嶇収", "鐓х墖", "鎷嶆憚")):
        return True

    haystack = _normalize_text_for_match(
        " ".join(
            str(step_meta.get(key, "") or "").strip()
            for key in ("instruction", "description", "tip")
        )
    )
    if not haystack:
        return False

    if _contains_any(
        haystack,
        (
            "棰滆壊绋冲畾鍚庢媿鐓?,
            "鎷嶇収纭",
            "璋冪敤mcp宸ュ叿xiaozhi_take_photo",
            "鎷嶇収鍚庡熀浜庣収鐗?,
            "鍩轰簬鐓х墖纭",
            "鎷嶇収鎴愬姛鍚?,
            "鐓х墖棰滆壊",
        ),
    ):
        return True

    # Some non-photo steps mention "瀹屾垚鍚庤繘鍏ユ媿鐓ц褰曟楠?; that should not make
    # the current step itself look like a photo-confirmation step.
    if _contains_any(
        haystack,
        (
            "杩涘叆鏈牱鍝佹媿鐓ц褰曟楠?,
            "杩涘叆鎷嶇収璁板綍姝ラ",
            "杩涘叆涓嬩竴姝ユ媿鐓?,
            "瀹屾垚鍚庤繘鍏ユ媿鐓?,
        ),
    ):
        return False

    return False


def _parse_small_chinese_integer(token: str) -> int | None:
    text = str(token or "").strip()
    if not text:
        return None
    if text.isdigit():
        try:
            value = int(text)
        except ValueError:
            return None
        return value if value > 0 else None

    digit_map = {
        "\u96f6": 0,
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
        "\u4e24": 2,
    }
    normalized = text.replace("\u4e24", "\u4e8c")
    if normalized in digit_map and digit_map[normalized] > 0:
        return digit_map[normalized]

    if "\u5341" in normalized:
        left, right = normalized.split("\u5341", 1)
        tens = digit_map.get(left) if left else 1
        if tens is None:
            return None
        ones = digit_map.get(right) if right else 0
        if ones is None:
            return None
        value = tens * 10 + ones
        return value if value > 0 else None

    return None


def _extract_sample_index_from_text(text: str) -> int | None:
    src = textUtils.normalize_spoken_text(text or "")
    if not src:
        return None

    patterns = (
        r"([0-9]+)\s*\u53f7\u6837\u54c1",
        r"\u6837\u54c1\s*([0-9]+)",
        r"sample[_\\-\\s]*([0-9]+)",
        r"([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]+)\s*\u53f7\u6837\u54c1",
        r"\u6837\u54c1\s*([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, src, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = _parse_small_chinese_integer(match.group(1))
        if parsed is not None:
            return parsed
    return None


def _format_sample_name(sample_index: int | None, fallback: str = "") -> str:
    if sample_index is not None and sample_index > 0:
        return f"{sample_index}\u53f7\u6837\u54c1"
    return str(fallback or "").strip()


def _resolve_experiment_yaml_steps(conn) -> list[dict]:
    cache = getattr(conn, "_experiment_yaml_steps_cache", None)
    if isinstance(cache, list):
        return cache

    yaml_path = str(getattr(conn, "experiment_yaml_path", "") or "").strip()
    if not yaml_path and hasattr(conn, "_resolve_experiment_yaml_path"):
        try:
            yaml_path = str(conn._resolve_experiment_yaml_path() or "").strip()
        except Exception:
            yaml_path = ""
    if not yaml_path:
        return []

    try:
        payload = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return []

    candidate_lists = []
    if isinstance(payload, dict):
        top_steps = payload.get("steps")
        if isinstance(top_steps, list):
            candidate_lists.append(top_steps)
        experiment = payload.get("experiment")
        if isinstance(experiment, dict):
            nested_steps = experiment.get("steps")
            if isinstance(nested_steps, list):
                candidate_lists.append(nested_steps)

    for steps in candidate_lists:
        normalized_steps = [item for item in steps if isinstance(item, dict)]
        if normalized_steps:
            setattr(conn, "_experiment_yaml_steps_cache", normalized_steps)
            return normalized_steps
    return []


def _experiment_has_step_id(conn, step_id: str) -> bool:
    target_step_id = str(step_id or "").strip()
    if not target_step_id:
        return False

    return any(
        str(step.get("id", "") or "").strip() == target_step_id
        for step in _resolve_experiment_yaml_steps(conn)
        if isinstance(step, dict)
    )


def _uvvis_shared_blank_step_enabled(conn) -> bool:
    return _experiment_has_step_id(conn, _UVVIS_SHARED_BLANK_STEP_ID)


_EXPERIMENT_STEP_MATCH_ALIAS_RULES = (
    (re.compile(r"agno3", flags=re.IGNORECASE), ("纭濋吀閾?,)),
    (re.compile(r"纭濋吀閾?), ("agno3",)),
    (re.compile(r"h2o2", flags=re.IGNORECASE), ("杩囨哀鍖栨阿",)),
    (re.compile(r"杩囨哀鍖栨阿"), ("h2o2",)),
    (re.compile(r"nabh4", flags=re.IGNORECASE), ("纭兼阿鍖栭挔",)),
    (re.compile(r"纭兼阿鍖栭挔"), ("nabh4",)),
    (re.compile(r"kbr", flags=re.IGNORECASE), ("婧村寲閽?,)),
    (re.compile(r"婧村寲閽?), ("kbr",)),
    (re.compile(r"uv-?vis", flags=re.IGNORECASE), ("绱鍙",)),
    (re.compile(r"绱[-锛峕?鍙"), ("uvvis",)),
    (re.compile(r"鍘荤瀛愭按"), ("绾按",)),
    (re.compile(r"绾按"), ("鍘荤瀛愭按",)),
)
_EXPERIMENT_STEP_HINT_TOKEN_TEXTS = (
    "鏌犳閰搁挔",
    "agno3",
    "纭濋吀閾?,
    "h2o2",
    "杩囨哀鍖栨阿",
    "kbr",
    "婧村寲閽?,
    "nabh4",
    "纭兼阿鍖栭挔",
    "绾按",
    "鍘荤瀛愭按",
    "鎼呮媽",
    "鎷嶇収",
    "鐓х墖",
    "涓佽揪灏?,
    "姣旇壊鐨?,
    "鍙傛瘮",
    "鍙嶅簲娑?,
    "uvvis",
    "绱鍙",
    "鍚稿厜搴?,
    "鍔ㄥ姏瀛?,
    "400nm",
    "400绾崇背",
)


def _expand_experiment_step_match_aliases(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""

    expanded_parts = [src]
    for pattern, aliases in _EXPERIMENT_STEP_MATCH_ALIAS_RULES:
        if not pattern.search(src):
            continue
        expanded_parts.extend(alias for alias in aliases if alias)
    return " ".join(expanded_parts)


def _normalize_experiment_step_match_text(text: str) -> str:
    src = _expand_experiment_step_match_aliases(text)
    norm = _normalize_confirmation_signature(src)
    norm = re.sub(r"[^\w\u4e00-\u9fff\-]+", "", norm)
    return norm


def _yaml_step_scope_signature(step: dict) -> tuple[str, int | None]:
    prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
    candidate_text = " ".join(
        value
        for value in (
            str(step.get("title", "") or "").strip(),
            str(step.get("description", "") or "").strip(),
            str(prompts.get("instruction", "") or "").strip(),
        )
        if value
    )
    normalized = _normalize_experiment_step_match_text(candidate_text)
    if any(
        token in normalized for token in ("1-5鍙锋牱鍝?, "1鍒?鍙锋牱鍝?, "1鑷?鍙锋牱鍝?, "鍏ㄩ儴鏍峰搧")
    ):
        return "multi_sample", None
    sample_index = _extract_sample_index_from_text(normalized)
    if sample_index is not None:
        return "single_sample", sample_index

    return "", None


def _yaml_step_context_match_score(query_text: str, step: dict) -> float:
    query_norm = _normalize_experiment_step_match_text(query_text)
    if len(query_norm) < 4:
        return 0.0

    prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
    title_text = str(step.get("title", "") or "").strip()
    instruction_text = str(prompts.get("instruction", "") or "").strip()
    description_text = str(step.get("description", "") or "").strip()
    step_text = " ".join(
        value for value in (title_text, instruction_text, description_text) if value
    )
    step_norm = _normalize_experiment_step_match_text(step_text)
    if len(step_norm) < 4:
        return 0.0

    query_has_multi_sample = any(
        token in query_norm for token in ("1-5鍙?, "1鍒?鍙?, "1鑷?鍙?, "姣忎釜鐑ф澂", "鍏ㄩ儴鏍峰搧")
    )
    query_sample_index = None if query_has_multi_sample else _extract_sample_index_from_text(query_norm)
    step_scope, step_sample_index = _yaml_step_scope_signature(step)
    if (
        query_sample_index is not None
        and step_scope == "single_sample"
        and step_sample_index is not None
        and step_sample_index != query_sample_index
    ):
        return 0.0
    if query_sample_index is not None and step_scope == "multi_sample":
        return 0.0
    if query_has_multi_sample and step_scope == "single_sample":
        return 0.0

    title_norm = _normalize_experiment_step_match_text(title_text)
    instruction_norm = _normalize_experiment_step_match_text(instruction_text)
    if title_norm and (title_norm in query_norm or query_norm in title_norm):
        return 1.0
    if instruction_norm and len(instruction_norm) >= 8 and instruction_norm[:8] in query_norm:
        return 0.96

    query_grams = _confirmation_char_ngrams(query_norm)
    step_grams = _confirmation_char_ngrams(step_norm)
    if not query_grams or not step_grams:
        return 0.0

    overlap_ratio = len(query_grams & step_grams) / max(
        1, min(len(query_grams), len(step_grams))
    )
    lcs = _longest_common_substring_len(query_norm, step_norm)
    lcs_ratio = lcs / max(4, min(len(query_norm), len(step_norm)))
    score = overlap_ratio * 0.72 + min(1.0, lcs_ratio) * 0.28

    shared_hint_tokens = []
    for raw_token in _EXPERIMENT_STEP_HINT_TOKEN_TEXTS:
        token = _normalize_experiment_step_match_text(raw_token)
        if token and token in query_norm and token in step_norm:
            shared_hint_tokens.append(token)
    if shared_hint_tokens:
        score += min(0.24, 0.08 * len(shared_hint_tokens))

    if query_sample_index is not None and step_scope == "single_sample":
        score += 0.05
    if query_has_multi_sample and step_scope == "multi_sample":
        score += 0.05
    if title_norm:
        title_lcs = _longest_common_substring_len(query_norm, title_norm)
        if title_lcs >= 4:
            score += 0.08
    return min(1.0, score)


def _resolve_experiment_step_id_order(conn) -> list[str]:
    cache = getattr(conn, "_experiment_yaml_step_id_order_cache", None)
    if isinstance(cache, list):
        return cache

    order = []
    for step in _resolve_experiment_yaml_steps(conn):
        step_id = str(step.get("id", "") or "").strip()
        if step_id:
            order.append(step_id)
    setattr(conn, "_experiment_yaml_step_id_order_cache", order)
    return order


def _resolve_experiment_step_by_id(conn) -> dict[str, dict]:
    cache = getattr(conn, "_experiment_yaml_step_by_id_cache", None)
    if isinstance(cache, dict):
        return cache

    mapping = {}
    for step in _resolve_experiment_yaml_steps(conn):
        step_id = str(step.get("id", "") or "").strip()
        if step_id:
            mapping[step_id] = step
    setattr(conn, "_experiment_yaml_step_by_id_cache", mapping)
    return mapping


def _infer_experiment_step_id_from_context(
    conn,
    original_text: str = "",
    filtered_text: str = "",
) -> str:
    if not _should_attempt_future_step_context_inference(
        original_text,
        filtered_text,
    ):
        return ""

    order = _resolve_experiment_step_id_order(conn)
    if not order:
        return ""

    current_step_id = _get_current_experiment_step_id(conn)
    index_by_id = {step_id: idx for idx, step_id in enumerate(order)}
    current_index = index_by_id.get(current_step_id, -1)
    step_by_id = _resolve_experiment_step_by_id(conn)

    candidates = [
        ("assistant_last", _get_last_assistant_text_raw(conn), 0.08),
        ("assistant_recent", _get_recent_assistant_text(conn, limit=4), 0.06),
        ("user_now", original_text, 0.04),
        ("user_filtered", filtered_text, 0.02),
        ("user_recent", _get_recent_user_text(conn, limit=4), 0.0),
    ]

    best_step_id = ""
    best_score = 0.0
    for _source, text, bonus in candidates:
        normalized = _normalize_experiment_step_match_text(text)
        if len(normalized) < 4:
            continue
        for step_id in order:
            step_index = index_by_id.get(step_id, -1)
            if step_index < 0 or step_index <= current_index:
                continue
            step = step_by_id.get(step_id)
            if not isinstance(step, dict):
                continue
            score = _yaml_step_context_match_score(text, step) + bonus
            if score > best_score:
                best_step_id = step_id
                best_score = score

    if best_score < 0.42:
        return ""
    return best_step_id


def _yaml_step_is_safe_generic_catchup_confirmation(step: dict) -> bool:
    if not isinstance(step, dict):
        return False

    interaction = step.get("interaction") if isinstance(step.get("interaction"), dict) else {}
    fast_path_mode = str(interaction.get("fast_path_mode", "") or "").strip().lower()
    if fast_path_mode != "confirmation_step":
        return False

    record_schema = step.get("record_schema")
    if not isinstance(record_schema, dict) or not record_schema:
        return False

    for field_name, field in record_schema.items():
        if not isinstance(field, dict):
            return False
        if bool(field.get("optional", False)):
            continue
        type_text = str(field.get("type", "") or "").strip().lower()
        if type_text not in {"bool", "boolean"}:
            return False
        haystack = _normalize_experiment_step_match_text(
            f"{field_name} {field.get('description', '')}"
        )
        if _contains_any(
            haystack,
            (
                "鐓х墖",
                "鎷嶇収",
                "photo",
                "棰滆壊",
                "瑙傚療",
                "鐜拌薄",
                "absorbance",
                "鍚稿厜",
                "娉㈤暱",
                "kinetics",
                "csv",
                "璺緞",
                "file",
                "path",
            ),
        ):
            return False
    return True


async def _sync_experiment_graph_forward_to_recent_context(
    conn,
    original_text: str = "",
    filtered_text: str = "",
) -> bool:
    # Strict no-skip mode: normal device-side fast path may not infer or catch up
    # future steps from recent conversational context.
    return False

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False

    order = _resolve_experiment_step_id_order(conn)
    if not order:
        return False

    current_step_id = _get_current_experiment_step_id(conn)
    target_step_id = _infer_experiment_step_id_from_context(
        conn,
        original_text=original_text,
        filtered_text=filtered_text,
    )
    if not current_step_id or not target_step_id or current_step_id == target_step_id:
        return False

    index_by_id = {step_id: idx for idx, step_id in enumerate(order)}
    current_index = index_by_id.get(current_step_id, -1)
    target_index = index_by_id.get(target_step_id, -1)
    if current_index < 0 or target_index < 0 or target_index <= current_index:
        return False

    step_by_id = _resolve_experiment_step_by_id(conn)
    made_progress = False
    for step_id in order[current_index:target_index]:
        current_step_id = _get_current_experiment_step_id(conn)
        if current_step_id == target_step_id:
            break
        if current_step_id != step_id:
            if current_step_id:
                current_idx = index_by_id.get(current_step_id, -1)
                if current_idx >= target_index:
                    break
            redirected = await _try_redirect_experiment_step_fast(
                conn,
                step_id,
                log_reason="experiment fast path syncing stale graph to recent context",
            )
            if not redirected:
                break

        step = step_by_id.get(step_id)
        if not _yaml_step_is_safe_generic_catchup_confirmation(step):
            break

        reply = await _advance_experiment_step_fast(conn, session_id)
        made_progress = True
        next_step_id = _get_current_experiment_step_id(conn)
        if not next_step_id or next_step_id == step_id:
            conn.logger.bind(tag=TAG).info(
                "experiment fast path sync stopped before target: "
                f"current_step_id={step_id}, target_step_id={target_step_id}, reply={reply or ''}"
            )
            break

    return made_progress


_PHOTO_CONFIRMATION_FIELD_NAMES = frozenset(
    {
        "photo_taken",
        "color_confirmed_by_photo",
        "color_mismatch_reason",
        "photo_file_name",
        "photo_path",
    }
)


def _yaml_step_schema_names(step: dict) -> set[str]:
    if not isinstance(step, dict):
        return set()
    record_schema = step.get("record_schema")
    if not isinstance(record_schema, dict):
        return set()
    return {
        str(field_name or "").strip()
        for field_name in record_schema
        if str(field_name or "").strip()
    }


def _yaml_step_looks_like_photo_confirmation(step: dict) -> bool:
    if not isinstance(step, dict):
        return False

    interaction = step.get("interaction") if isinstance(step.get("interaction"), dict) else {}
    fast_path_mode = str(interaction.get("fast_path_mode", "") or "").strip().lower()
    if fast_path_mode == "photo_confirmation_step":
        return True

    if _yaml_step_schema_names(step) & _PHOTO_CONFIRMATION_FIELD_NAMES:
        return True

    prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
    step_meta = {
        "title": str(step.get("title", "") or "").strip(),
        "description": str(step.get("description", "") or "").strip(),
        "instruction": str(prompts.get("instruction", "") or "").strip(),
        "tip": str(prompts.get("tip", "") or "").strip(),
    }
    return _step_meta_looks_like_photo_confirmation(step_meta)


def _infer_photo_confirmation_step_id_from_context(
    conn,
    payload,
    *,
    requested_arguments: dict | None = None,
) -> str:
    sample_index = None
    photo_meta = _extract_photo_result_meta(payload)
    candidate_texts = [
        photo_meta.get("requested_photo_name", ""),
        photo_meta.get("file_name", ""),
        photo_meta.get("photo_path", ""),
        str((requested_arguments or {}).get("photo_name", "") or ""),
        str((requested_arguments or {}).get("question", "") or ""),
        _get_last_assistant_text_raw(conn),
        _get_recent_assistant_text(conn, limit=5),
        _get_recent_user_text(conn, limit=5),
        _get_recent_dialogue_text(conn, roles=("user", "assistant"), limit=8),
    ]
    for text in candidate_texts:
        sample_index = _extract_sample_index_from_text(text)
        if sample_index is not None:
            break
    if sample_index is None:
        return ""

    step_cache = getattr(conn, "_photo_confirmation_step_id_cache", None)
    if not isinstance(step_cache, dict):
        step_cache = {}
    cached_step_id = str(step_cache.get(sample_index, "") or "").strip()
    if cached_step_id:
        return cached_step_id

    for step in _resolve_experiment_yaml_steps(conn):
        step_id = str(step.get("id", "") or "").strip()
        if not step_id:
            continue
        if not _yaml_step_looks_like_photo_confirmation(step):
            continue
        prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
        step_meta = {
            "title": str(step.get("title", "") or "").strip(),
            "description": str(step.get("description", "") or "").strip(),
            "instruction": str(prompts.get("instruction", "") or "").strip(),
            "tip": str(prompts.get("tip", "") or "").strip(),
        }
        step_sample_index = None
        for step_text in (
            step_id,
            step_meta["title"],
            step_meta["description"],
            step_meta["instruction"],
        ):
            step_sample_index = _extract_sample_index_from_text(step_text)
            if step_sample_index is not None:
                break
        if step_sample_index != sample_index:
            continue
        step_cache[sample_index] = step_id
        setattr(conn, "_photo_confirmation_step_id_cache", step_cache)
        return step_id
    return ""


def _append_hidden_assistant_context(conn, text: str, *, source: str = "") -> None:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return
    conn.dialogue.put(Message(role="assistant", content=normalized))
    if hasattr(conn, "append_experiment_interaction_log"):
        try:
            conn.append_experiment_interaction_log(
                "ASSISTANT",
                normalized,
                source=source or "assistant_context",
            )
        except Exception:
            pass


def _remember_recent_server_photo_confirmation(
    conn,
    *,
    payload,
    requested_arguments: dict | None = None,
    graph_advanced: bool = False,
    next_step_id: str = "",
    next_step_title: str = "",
    next_step_reply: str = "",
    graph_status_reason: str = "",
    current_step_id: str = "",
    current_step_title: str = "",
    graph_refresh_checked_at: float | None = None,
) -> dict:
    photo_meta = _extract_photo_result_meta(payload)
    sample_index = None
    for text in (
        photo_meta.get("requested_photo_name", ""),
        str((requested_arguments or {}).get("photo_name", "") or ""),
        str((requested_arguments or {}).get("question", "") or ""),
        photo_meta.get("file_name", ""),
        _get_last_assistant_text_raw(conn),
    ):
        sample_index = _extract_sample_index_from_text(text)
        if sample_index is not None:
            break

    sample_name = str(photo_meta.get("requested_photo_name", "") or "").strip()
    if not sample_name:
        sample_name = str((requested_arguments or {}).get("photo_name", "") or "").strip()
    sample_name = _format_sample_name(sample_index, fallback=sample_name)

    state = {
        "captured_at": time.time(),
        "sample_index": sample_index,
        "sample_name": sample_name,
        "photo_meta": photo_meta,
        "graph_advanced": bool(graph_advanced),
        "next_step_id": str(next_step_id or "").strip(),
        "next_step_title": str(next_step_title or "").strip(),
        "next_step_reply": str(next_step_reply or "").strip(),
        "graph_status_reason": str(graph_status_reason or "").strip(),
        "current_step_id": str(current_step_id or "").strip(),
        "current_step_title": str(current_step_title or "").strip(),
        "graph_refresh_checked_at": (
            float(graph_refresh_checked_at or 0.0)
            if graph_refresh_checked_at not in (None, "")
            else 0.0
        ),
    }
    setattr(conn, "_recent_server_photo_confirmation", state)
    return state


def _compose_photo_confirmation_not_advanced_reply(
    confirmation_reply: str,
    followup_reply: str = "",
) -> str:
    confirmation = textUtils.prepare_runtime_spoken_text(confirmation_reply)
    followup = textUtils.prepare_runtime_spoken_text(followup_reply)
    bridge = "褰撳墠瀹為獙鍥捐氨杩樺仠鍦ㄨ繖涓€姝ワ紝鍏堟寜杩欎竴姝ョ户缁€?

    if confirmation and not followup:
        confirmation = confirmation.rstrip("銆傦紒锛?? ").strip()
        if confirmation:
            return f"{confirmation}銆倇bridge}"
        return bridge

    if followup and not confirmation:
        return f"{bridge}{followup}"

    if not confirmation and not followup:
        return bridge

    if confirmation == followup:
        return f"{bridge}{followup}"

    confirmation = confirmation.rstrip("銆傦紒锛?? ").strip()
    if confirmation:
        confirmation = f"{confirmation}銆?
    return f"{confirmation}{bridge}{followup}"


def _looks_like_internal_experiment_graph_message(message: str) -> bool:
    raw_text = str(message or "").strip()
    if not raw_text:
        return False

    normalized = _normalize_text_for_match(raw_text)
    internal_tokens = (
        "鏃犳硶璺宠浆鍒皊tep",
        "鍓嶇疆姝ラ鏈畬鎴?,
        "redirect_to_step",
        "finish_trial",
        "can_proceed",
        "proceed_to_next_step",
        "session_id",
        "step_",
    )
    if _contains_any(normalized, internal_tokens):
        return True
    return False


def _sanitize_experiment_graph_followup_reply(
    reply: str,
    *,
    fallback_reply: str = "",
) -> str:
    prepared_reply = textUtils.prepare_runtime_spoken_text(reply)
    if prepared_reply and not _looks_like_internal_experiment_graph_message(reply):
        return prepared_reply
    return textUtils.prepare_runtime_spoken_text(fallback_reply)


async def _safe_refresh_experiment_step_cache(
    conn,
    session_id: str,
    *,
    reason: str = "",
) -> dict:
    if not session_id:
        return _get_cached_experiment_step_meta(conn)
    try:
        return await _refresh_experiment_step_cache(conn, session_id)
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            "experiment step cache refresh failed: "
            f"session_id={session_id}, reason={reason or 'unknown'}, error={exc}"
        )
        return _get_cached_experiment_step_meta(conn)


async def _finalize_photo_followup_without_graph_advance(
    conn,
    session_id: str,
    payload,
    *,
    requested_arguments: dict | None = None,
    confirmation_reply: str = "",
    followup_reply: str = "",
    reason: str = "",
    expected_step_id: str = "",
    refresh_state: bool = True,
) -> str:
    step_meta = _get_cached_experiment_step_meta(conn)
    if refresh_state and session_id:
        step_meta = await _safe_refresh_experiment_step_cache(
            conn,
            session_id,
            reason=f"photo_followup_{reason or 'not_advanced'}",
        )

    current_step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
    current_step_title = str(step_meta.get("title", "") or "").strip()
    current_step_reply = _compose_experiment_step_reply(step_meta, mode="guide")
    sanitized_followup_reply = _sanitize_experiment_graph_followup_reply(
        followup_reply,
        fallback_reply=current_step_reply,
    )
    final_reply = _compose_photo_confirmation_not_advanced_reply(
        confirmation_reply,
        sanitized_followup_reply or current_step_reply,
    )

    recent_state = _remember_recent_server_photo_confirmation(
        conn,
        payload=payload,
        requested_arguments=requested_arguments,
        graph_advanced=False,
        next_step_reply=final_reply,
        graph_status_reason=reason,
        current_step_id=current_step_id,
        current_step_title=current_step_title,
        graph_refresh_checked_at=time.time(),
    )
    hidden_note = _build_hidden_photo_confirmation_note(recent_state)
    if hidden_note:
        _append_hidden_assistant_context(
            conn,
            hidden_note,
            source="photo_confirmation_context",
        )

    conn.logger.bind(tag=TAG).warning(
        "local photo follow-up did not advance experiment graph: "
        f"session_id={session_id or ''}, "
        f"reason={reason or 'unknown'}, "
        f"expected_step_id={expected_step_id or ''}, "
        f"current_step_id={current_step_id}, "
        f"current_step_title={current_step_title or ''}"
    )
    return final_reply


def _build_hidden_photo_confirmation_note(state: dict) -> str:
    if not isinstance(state, dict):
        return ""
    sample_name = str(state.get("sample_name", "") or "").strip() or "\u5f53\u524d\u6837\u54c1"
    if state.get("graph_advanced") and str(state.get("next_step_title", "") or "").strip():
        return (
            f"\u62cd\u7167\u5df2\u7ecf\u6210\u529f\uff0c\u6211\u628a{sample_name}\u7684\u62cd\u7167\u786e\u8ba4"
            f"\u8bb0\u5f55\u5199\u56de\u5f53\u524d\u5b9e\u9a8c\u72b6\u6001\uff0c\u5f53\u524d\u5df2\u8fdb\u5165"
            f"{str(state.get('next_step_title', '')).strip()}\u3002"
        )
    return (
        f"\u62cd\u7167\u5df2\u7ecf\u6210\u529f\uff0c{sample_name}\u7167\u7247\u5df2\u4fdd\u5b58\u3002"
        f"\u540e\u7eed\u4e0d\u8981\u518d\u8981\u6c42{sample_name}\u91cd\u590d\u62cd\u7167\uff0c"
        f"\u9664\u975e\u7528\u6237\u660e\u786e\u8981\u6c42\u91cd\u62cd\u3002"
    )


def _recent_server_photo_confirmation_matches_request(conn, request: dict) -> bool:
    state = getattr(conn, "_recent_server_photo_confirmation", None)
    if not isinstance(state, dict):
        return False

    try:
        captured_at = float(state.get("captured_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        captured_at = 0.0
    if captured_at <= 0 or (time.time() - captured_at) > 900:
        return False

    recent_sample_index = state.get("sample_index")
    request_sample_index = None
    for text in (
        str((request or {}).get("photo_name", "") or ""),
        str((request or {}).get("question", "") or ""),
        _get_last_assistant_text_raw(conn),
    ):
        request_sample_index = _extract_sample_index_from_text(text)
        if request_sample_index is not None:
            break

    if recent_sample_index is not None and request_sample_index is not None:
        return int(recent_sample_index) == int(request_sample_index)

    recent_sample_name = str(state.get("sample_name", "") or "").strip()
    request_sample_name = str((request or {}).get("photo_name", "") or "").strip()
    if recent_sample_name and request_sample_name:
        return recent_sample_name == request_sample_name
    return False


async def _advance_photo_confirmation_step_locally(
    conn,
    payload,
    fallback_reply: str = "",
    *,
    requested_arguments: dict | None = None,
) -> str:
    return await _advance_photo_confirmation_step_locally_v2(
        conn,
        payload,
        fallback_reply,
        requested_arguments=requested_arguments,
    )

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return fallback_reply or "鎷嶅ソ浜嗐€?

    step_payload, progress_payload, schema_payload = await asyncio.gather(
        _call_experiment_graph_tool_fast(
            conn,
            "get_step",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_current_progress",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_schema",
            {"session_id": session_id},
            priority="foreground",
        ),
    )

    conn.experiment_current_step = step_payload
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(step_payload)
        if current_step_id:
            conn.experiment_current_step_id = current_step_id

    step_meta = _merge_experiment_step_meta(
        _extract_experiment_step_meta(step_payload),
        _extract_experiment_step_meta(getattr(conn, "experiment_progress_summary", None)),
    )
    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        start_payload = await _call_experiment_graph_tool_fast(
            conn,
            "start_trial",
            {"session_id": session_id},
            priority="foreground",
        )
        current_progress = _extract_experiment_current_progress(start_payload)

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    photo_meta = _extract_photo_result_meta(payload)
    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    photo_fields = _build_experiment_photo_writeback_fields(
        schema_by_name,
        photo_meta,
        step_meta=step_meta,
        missing_fields=missing_fields,
    )
    photo_related_fields = {
        "photo_taken",
        "color_confirmed_by_photo",
        "photo_file_name",
        "photo_path",
    }
    is_photo_confirmation_step = _step_meta_looks_like_photo_confirmation(step_meta) or (
        bool(schema_by_name)
        and any(name in schema_by_name for name in photo_related_fields)
    ) or any(name in photo_related_fields for name in missing_fields)
    if not is_photo_confirmation_step:
        inferred_step_id = _infer_photo_confirmation_step_id_from_context(
            conn,
            payload,
            requested_arguments=requested_arguments,
        )
        if inferred_step_id:
            conn.logger.bind(tag=TAG).info(
                "local photo follow-up redirecting stale graph step: "
                f"session_id={session_id}, inferred_step_id={inferred_step_id}, "
                f"current_step_id={getattr(conn, 'experiment_current_step_id', '')}"
            )
            redirect_payload = await _call_experiment_graph_tool_fast(
                conn,
                "redirect_to_step",
                {"session_id": session_id, "step_id": inferred_step_id},
                priority="foreground",
            )
            if bool(_experiment_result_body(redirect_payload).get("ok")):
                step_payload, progress_payload, schema_payload = await asyncio.gather(
                    _call_experiment_graph_tool_fast(
                        conn,
                        "get_step",
                        {"session_id": session_id},
                        priority="foreground",
                    ),
                    _call_experiment_graph_tool_fast(
                        conn,
                        "get_current_progress",
                        {"session_id": session_id},
                        priority="foreground",
                    ),
                    _call_experiment_graph_tool_fast(
                        conn,
                        "get_schema",
                        {"session_id": session_id},
                        priority="foreground",
                    ),
                )

                conn.experiment_current_step = step_payload
                if hasattr(conn, "_extract_experiment_current_step_id"):
                    current_step_id = conn._extract_experiment_current_step_id(
                        step_payload
                    )
                    if current_step_id:
                        conn.experiment_current_step_id = current_step_id

                step_meta = _merge_experiment_step_meta(
                    _extract_experiment_step_meta(step_payload),
                    _extract_experiment_step_meta(
                        getattr(conn, "experiment_progress_summary", None)
                    ),
                )
                current_progress = _extract_experiment_current_progress(progress_payload)
                if current_progress is None:
                    start_payload = await _call_experiment_graph_tool_fast(
                        conn,
                        "start_trial",
                        {"session_id": session_id},
                        priority="foreground",
                    )
                    current_progress = _extract_experiment_current_progress(
                        start_payload
                    )

                schema_by_name = _extract_experiment_schema_view(schema_payload)
                missing_fields = list((current_progress or {}).get("missing_fields") or [])
                photo_fields = _build_experiment_photo_writeback_fields(
                    schema_by_name,
                    photo_meta,
                    step_meta=step_meta,
                    missing_fields=missing_fields,
                )
                is_photo_confirmation_step = _step_meta_looks_like_photo_confirmation(
                    step_meta
                ) or (
                    bool(schema_by_name)
                    and any(name in schema_by_name for name in photo_related_fields)
                ) or any(name in photo_related_fields for name in missing_fields)

    if not is_photo_confirmation_step:
        _remember_recent_server_photo_confirmation(
            conn,
            payload=payload,
            requested_arguments=requested_arguments,
            graph_advanced=False,
            next_step_reply=fallback_reply,
        )
        return fallback_reply or "鎷嶅ソ浜嗐€?

    if photo_fields:
        add_fields_payload = await _call_experiment_graph_tool_fast(
            conn,
            "add_fields",
            {"session_id": session_id, "data": photo_fields},
            priority="foreground",
        )
        updated_progress = _extract_experiment_current_progress(add_fields_payload)
        if isinstance(updated_progress, dict):
            current_progress = updated_progress
        missing_fields = list((current_progress or {}).get("missing_fields") or [])

    if missing_fields:
        return _compose_missing_field_reply(missing_fields, schema_by_name)

    finish_payload = await _call_experiment_graph_tool_fast(
        conn,
        "finish_trial",
        {"session_id": session_id, "validate": True},
        priority="foreground",
    )
    if not bool(_experiment_result_body(finish_payload).get("ok")):
        message = _extract_experiment_result_message(finish_payload)
        if message:
            return message
        reply = _compose_experiment_step_reply(step_meta, mode="guide")
        return reply or fallback_reply or "鎷嶅ソ浜嗐€?

    can_proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "can_proceed",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(can_proceed_payload).get("ok")):
        message = _extract_experiment_result_message(can_proceed_payload)
        if message:
            return message
        reply = _compose_experiment_step_reply(step_meta, mode="guide")
        return reply or fallback_reply or "鎷嶅ソ浜嗐€?

    proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "proceed_to_next_step",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(proceed_payload).get("ok")):
        message = _extract_experiment_result_message(proceed_payload)
        if message:
            return message
        reply = _compose_experiment_step_reply(step_meta, mode="guide")
        return reply or fallback_reply or "鎷嶅ソ浜嗐€?

    next_meta = await _refresh_experiment_step_cache(conn, session_id)
    reply = _compose_experiment_step_reply(next_meta, mode="next")
    recent_state = _remember_recent_server_photo_confirmation(
        conn,
        payload=payload,
        requested_arguments=requested_arguments,
        graph_advanced=True,
        next_step_id=str(next_meta.get("step_id", "") or "").strip(),
        next_step_title=str(next_meta.get("title", "") or "").strip(),
        next_step_reply=reply or fallback_reply,
    )
    hidden_note = _build_hidden_photo_confirmation_note(recent_state)
    if hidden_note:
        _append_hidden_assistant_context(
            conn,
            hidden_note,
            source="photo_confirmation_context",
        )
    conn.logger.bind(tag=TAG).info(
        "local photo follow-up advanced experiment step: "
        f"session_id={session_id}, next_step_id={str(next_meta.get('step_id', '') or '').strip()}, "
        f"next_step_title={str(next_meta.get('title', '') or '').strip()}"
    )
    spoken_reply = _compose_photo_confirmation_advance_reply(
        fallback_reply,
        reply,
    )
    if spoken_reply:
        return spoken_reply
    if reply:
        return reply
    return fallback_reply or "鎷嶇収宸茬粡瀹屾垚锛岀户缁仛褰撳墠涓嬩竴姝ャ€?


async def _advance_photo_confirmation_step_locally_v2(
    conn,
    payload,
    fallback_reply: str = "",
    *,
    requested_arguments: dict | None = None,
) -> str:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    confirmation_reply = fallback_reply or "鎷嶅ソ浜嗐€?
    if not session_id:
        return await _finalize_photo_followup_without_graph_advance(
            conn,
            session_id,
            payload,
            requested_arguments=requested_arguments,
            confirmation_reply=confirmation_reply,
            reason="missing_experiment_session",
            refresh_state=False,
        )

    photo_meta = _extract_photo_result_meta(payload)

    step_payload, progress_payload, schema_payload = await asyncio.gather(
        _call_experiment_graph_tool_fast(
            conn,
            "get_step",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_current_progress",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_schema",
            {"session_id": session_id},
            priority="foreground",
        ),
    )

    conn.experiment_current_step = step_payload
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(step_payload)
        if current_step_id:
            conn.experiment_current_step_id = current_step_id

    step_meta = _merge_experiment_step_meta(
        _extract_experiment_step_meta(step_payload),
        _extract_experiment_step_meta(getattr(conn, "experiment_progress_summary", None)),
    )
    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        start_payload = await _call_experiment_graph_tool_fast(
            conn,
            "start_trial",
            {"session_id": session_id},
            priority="foreground",
        )
        current_progress = _extract_experiment_current_progress(start_payload)

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    photo_fields = _build_experiment_photo_writeback_fields(
        schema_by_name,
        photo_meta,
        step_meta=step_meta,
        missing_fields=missing_fields,
    )
    photo_related_fields = {
        "photo_taken",
        "color_confirmed_by_photo",
        "photo_file_name",
        "photo_path",
    }
    is_photo_confirmation_step = _step_meta_looks_like_photo_confirmation(step_meta) or (
        bool(schema_by_name)
        and any(name in schema_by_name for name in photo_related_fields)
    ) or any(name in photo_related_fields for name in missing_fields)

    if not is_photo_confirmation_step:
        inferred_step_id = _infer_photo_confirmation_step_id_from_context(
            conn,
            payload,
            requested_arguments=requested_arguments,
        )
        if inferred_step_id:
            conn.logger.bind(tag=TAG).info(
                "local photo follow-up redirecting stale graph step: "
                f"session_id={session_id}, inferred_step_id={inferred_step_id}, "
                f"current_step_id={getattr(conn, 'experiment_current_step_id', '')}"
            )
            redirect_payload = await _call_experiment_graph_tool_fast(
                conn,
                "redirect_to_step",
                {"session_id": session_id, "step_id": inferred_step_id},
                priority="foreground",
            )
            if not bool(_experiment_result_body(redirect_payload).get("ok")):
                message = _extract_experiment_result_message(redirect_payload)
                followup_reply = message or _compose_experiment_step_reply(
                    _get_cached_experiment_step_meta(conn),
                    mode="guide",
                )
                return await _finalize_photo_followup_without_graph_advance(
                    conn,
                    session_id,
                    payload,
                    requested_arguments=requested_arguments,
                    confirmation_reply=confirmation_reply,
                    followup_reply=followup_reply,
                    reason="redirect_to_inferred_photo_step_rejected",
                    expected_step_id=inferred_step_id,
                    refresh_state=True,
                )

            step_meta = await _safe_refresh_experiment_step_cache(
                conn,
                session_id,
                reason="photo_followup_redirect_to_inferred_step",
            )
            step_payload = getattr(conn, "experiment_current_step", step_payload)
            progress_payload, schema_payload = await asyncio.gather(
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_current_progress",
                    {"session_id": session_id},
                    priority="foreground",
                ),
                _call_experiment_graph_tool_fast(
                    conn,
                    "get_schema",
                    {"session_id": session_id},
                    priority="foreground",
                ),
            )

            step_meta = _merge_experiment_step_meta(
                _extract_experiment_step_meta(step_payload),
                step_meta,
            )
            current_progress = _extract_experiment_current_progress(progress_payload)
            if current_progress is None:
                start_payload = await _call_experiment_graph_tool_fast(
                    conn,
                    "start_trial",
                    {"session_id": session_id},
                    priority="foreground",
                )
                current_progress = _extract_experiment_current_progress(start_payload)

            schema_by_name = _extract_experiment_schema_view(schema_payload)
            missing_fields = list((current_progress or {}).get("missing_fields") or [])
            photo_fields = _build_experiment_photo_writeback_fields(
                schema_by_name,
                photo_meta,
                step_meta=step_meta,
                missing_fields=missing_fields,
            )
            is_photo_confirmation_step = _step_meta_looks_like_photo_confirmation(
                step_meta
            ) or (
                bool(schema_by_name)
                and any(name in schema_by_name for name in photo_related_fields)
            ) or any(name in photo_related_fields for name in missing_fields)

        if not is_photo_confirmation_step:
            return await _finalize_photo_followup_without_graph_advance(
                conn,
                session_id,
                payload,
                requested_arguments=requested_arguments,
                confirmation_reply=confirmation_reply,
                reason=(
                    "redirect_did_not_land_on_photo_step"
                    if inferred_step_id
                    else "inferred_photo_step_not_found"
                ),
                expected_step_id=inferred_step_id,
                refresh_state=not bool(inferred_step_id),
            )

    if photo_fields:
        add_fields_payload = await _call_experiment_graph_tool_fast(
            conn,
            "add_fields",
            {"session_id": session_id, "data": photo_fields},
            priority="foreground",
        )
        updated_progress = _extract_experiment_current_progress(add_fields_payload)
        if isinstance(updated_progress, dict):
            current_progress = updated_progress
        missing_fields = list((current_progress or {}).get("missing_fields") or [])

    if missing_fields:
        return await _finalize_photo_followup_without_graph_advance(
            conn,
            session_id,
            payload,
            requested_arguments=requested_arguments,
            confirmation_reply=confirmation_reply,
            followup_reply=_compose_missing_field_reply(missing_fields, schema_by_name),
            reason="photo_confirmation_missing_fields_remaining",
            refresh_state=False,
        )

    finish_payload = await _call_experiment_graph_tool_fast(
        conn,
        "finish_trial",
        {"session_id": session_id, "validate": True},
        priority="foreground",
    )
    if not bool(_experiment_result_body(finish_payload).get("ok")):
        message = _extract_experiment_result_message(finish_payload)
        followup_reply = message or _compose_experiment_step_reply(step_meta, mode="guide")
        return await _finalize_photo_followup_without_graph_advance(
            conn,
            session_id,
            payload,
            requested_arguments=requested_arguments,
            confirmation_reply=confirmation_reply,
            followup_reply=followup_reply,
            reason="finish_trial_rejected_after_photo_confirmation",
            refresh_state=True,
        )

    can_proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "can_proceed",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(can_proceed_payload).get("ok")):
        message = _extract_experiment_result_message(can_proceed_payload)
        followup_reply = message or _compose_experiment_step_reply(step_meta, mode="guide")
        return await _finalize_photo_followup_without_graph_advance(
            conn,
            session_id,
            payload,
            requested_arguments=requested_arguments,
            confirmation_reply=confirmation_reply,
            followup_reply=followup_reply,
            reason="can_proceed_rejected_after_photo_confirmation",
            refresh_state=True,
        )

    proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "proceed_to_next_step",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(proceed_payload).get("ok")):
        message = _extract_experiment_result_message(proceed_payload)
        followup_reply = message or _compose_experiment_step_reply(step_meta, mode="guide")
        return await _finalize_photo_followup_without_graph_advance(
            conn,
            session_id,
            payload,
            requested_arguments=requested_arguments,
            confirmation_reply=confirmation_reply,
            followup_reply=followup_reply,
            reason="proceed_to_next_step_rejected_after_photo_confirmation",
            refresh_state=True,
        )

    next_meta = await _safe_refresh_experiment_step_cache(
        conn,
        session_id,
        reason="photo_followup_proceed_to_next_step",
    )
    reply = _compose_experiment_step_reply(next_meta, mode="next")
    recent_state = _remember_recent_server_photo_confirmation(
        conn,
        payload=payload,
        requested_arguments=requested_arguments,
        graph_advanced=True,
        next_step_id=str(next_meta.get("step_id", "") or "").strip(),
        next_step_title=str(next_meta.get("title", "") or "").strip(),
        next_step_reply=reply or confirmation_reply,
    )
    hidden_note = _build_hidden_photo_confirmation_note(recent_state)
    if hidden_note:
        _append_hidden_assistant_context(
            conn,
            hidden_note,
            source="photo_confirmation_context",
        )
    conn.logger.bind(tag=TAG).info(
        "local photo follow-up advanced experiment step: "
        f"session_id={session_id}, next_step_id={str(next_meta.get('step_id', '') or '').strip()}, "
        f"next_step_title={str(next_meta.get('title', '') or '').strip()}"
    )
    spoken_reply = _compose_photo_confirmation_advance_reply(
        confirmation_reply,
        reply,
    )
    if spoken_reply:
        return spoken_reply
    if reply:
        return reply
    return confirmation_reply or "鐓х墖宸茬粡瀹屾垚锛岀户缁仛褰撳墠涓嬩竴姝ャ€?


async def _start_direct_intent_turn(conn, original_text: str):
    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))


_UVVIS_SHARED_DARK_AIR_STEP_ID = "step_3_uv_vis_shared_dark_air_prep"
_UVVIS_SHARED_BLANK_STEP_ID = "step_3_uv_vis_shared_dark_blank_prep"
_UVVIS_SAMPLE_RECORD_STEP_ID = "step_3_uv_vis_sample1-4_record_data"
_UVVIS_SAMPLE_LOAD_STEP_ID = "step_3_uv_vis_sample1-4_load_cuvette"
_UVVIS_SAMPLE_CLEAN_STEP_ID = "step_3_uv_vis_sample1-4_clean_cuvettes"
_UVVIS_SAMPLE_RECORD_STEP_ID_LEGACY = "step_3_uv_vis_sample1-5_record_data"
_UVVIS_SAMPLE_LOAD_STEP_ID_LEGACY = "step_3_uv_vis_sample1-5_load_cuvette"
_UVVIS_SAMPLE_CLEAN_STEP_ID_LEGACY = "step_3_uv_vis_sample5_clean_cuvette"
_UVVIS_KINETICS_SAMPLE2_STEP_ID = "step_4_kinetics_sample2_measurement"
_UVVIS_KINETICS_SAMPLE4_STEP_ID = "step_5_kinetics_sample4_measurement"
_UVVIS_KINETICS_COMBINED_STEP_ID = "step_6_kinetics_combined_measurement"
_UVVIS_ANALYSIS_STEP_ID = "step_6_data_analysis"
_UVVIS_BUSY_REPLY = "鎴戠幇鍦ㄦ鍦ㄥ伐浣滆浣犺繃5min鍐嶈瘯"
_UVVIS_NOT_READY_REPLY = "UV-Vis 杩欒竟杩樻病鍑嗗濂斤紝璇风◢鍚庡啀璇曘€?
_UVVIS_SAMPLE_POSITIONS = (1, 2, 3, 4, 5)
_UVVIS_GROUPED_KINETICS_POSITIONS = (2, 3, 4, 5)
_UVVIS_SPECTRA_WAVELENGTH_GRID = tuple(range(400, 701, 10))
_UVVIS_KINETICS_TIME_GRID = tuple(range(35))


async def _try_redirect_experiment_step_fast(
    conn,
    step_id: str,
    *,
    log_reason: str,
) -> bool:
    target_step_id = str(step_id or "").strip()
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not target_step_id or not session_id:
        return False

    current_step_id = _get_current_experiment_step_id(conn)
    if current_step_id == target_step_id:
        return True

    conn.logger.bind(tag=TAG).info(
        f"{log_reason}: session_id={session_id}, "
        f"inferred_step_id={target_step_id}, current_step_id={current_step_id}"
    )
    try:
        redirect_payload = await _call_experiment_graph_tool_fast(
            conn,
            "redirect_to_step",
            {"session_id": session_id, "step_id": target_step_id},
            priority="foreground",
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"{log_reason} failed: {exc}"
        )
        return False

    if not bool(_experiment_result_body(redirect_payload).get("ok")):
        message = _extract_experiment_result_message(redirect_payload)
        if message:
            conn.logger.bind(tag=TAG).warning(
                f"{log_reason} rejected: {message}"
            )
        return False

    conn.experiment_current_step_id = target_step_id
    conn.experiment_current_step = {"result": {"step": {"id": target_step_id}}}
    conn.experiment_progress_summary = {
        "result": {"summary": {"current_step": {"step_id": target_step_id}}}
    }
    if getattr(conn, "experiment_resume_recovery_required", False):
        conn.experiment_resume_latest_current_step_id = target_step_id
    return True


def _legacy_compose_uvvis_step_rejection_reply(step_id: str) -> str:
    step_id = str(step_id or "").strip()
    if step_id == _UVVIS_SHARED_BLANK_STEP_ID:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?UV-Vis 鍓嶇疆鏍℃锛屽厛瀹屾垚涓佽揪灏旂幇璞¤瀵熴€?
    if step_id == _UVVIS_SAMPLE_RECORD_STEP_ID:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?1-5 鍙锋牱鍝佺殑鎵归噺鍏夎氨娴嬮噺锛屽厛瀹屾垚鍓嶉潰鐨勬楠ゃ€?
    if step_id in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
    }:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒板搴旂殑 400 绾崇背鍔ㄥ姏瀛︽楠わ紝鍏堝畬鎴愬墠闈㈢殑姝ラ銆?
    return _UVVIS_NOT_READY_REPLY


def _looks_like_explicit_uvvis_turn(
    original_text: str = "",
    filtered_text: str = "",
) -> bool:
    normalized = _normalize_text_for_match(
        " ".join(
            text
            for text in (original_text, filtered_text)
            if str(text or "").strip()
        )
    )
    if not normalized:
        return False

    uvvis_tokens = (
        "uvvis",
        "uv-vis",
        "绱鍙",
        "鍏夎氨",
        "鍚稿厜搴?,
        "lambda max",
        "位max",
        "400绾崇背",
        "鍔ㄥ姏瀛?,
        "鏆楃數娴?,
        "绌烘皵鍩虹嚎",
        "绾按绌虹櫧",
        "鍙傛瘮浣?,
        "鏍峰搧浣?,
        "姣旇壊鐨?,
        "绌虹櫧娑?,
    )
    return _contains_any(normalized, uvvis_tokens)


def _assistant_recently_prompted_uvvis_action(conn) -> bool:
    recent_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not recent_text:
        return False

    uvvis_prompt_tokens = (
        "uvvis",
        "uv-vis",
        "绱鍙",
        "鏆楃數娴?,
        "绌烘皵鍩虹嚎",
        "绾按绌虹櫧",
        "鍙傛瘮浣?,
        "鏍峰搧浣?,
        "姣旇壊鐨?,
        "400绾崇背",
        "鍔ㄥ姏瀛?,
        "鍏夎氨",
        "绌虹櫧娑?,
    )
    return _contains_any(recent_text, uvvis_prompt_tokens)


def _assistant_recently_prompted_pure_water_blank(conn) -> bool:
    recent_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not recent_text:
        return False

    return _contains_any(
        recent_text,
        (
            "绾按绌虹櫧",
            "绾按姣旇壊鐨?,
            "1-5鍙锋牱鍝佷綅",
            "鍙傛瘮浣?,
            "6鏀函姘?,
        ),
    )


def _looks_like_uvvis_followup_reply(conn, filtered_text: str) -> bool:
    if not _assistant_recently_prompted_uvvis_action(conn):
        return False
    return _looks_like_uvvis_ready_reply(filtered_text)


def _looks_like_uvvis_ready_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    if _is_negative_short_reply_fixed(filtered_text):
        return False
    if _is_affirmative_short_reply_fixed(filtered_text):
        return True
    if _looks_like_pure_short_completion_control(norm):
        return True
    ready_tokens = (
        "鏀惧ソ浜?,
        "閮芥斁濂戒簡",
        "宸茬粡鏀惧ソ浜?,
        "宸茬粡鏀惧ソ",
        "鍙互寮€濮嬩簡",
        "寮€濮嬪惂",
        "寮€濮嬫祴閲?,
        "寮€濮嬫壂鎻?,
        "鍙互寮€濮嬫壂鎻?,
        "鎵弿鍚?,
        "寮€濮嬬┖鏋舵壂鎻?,
        "寮€濮嬪姩鍔涘",
        "寮€濮嬭褰?,
        "娴嬪厜璋?,
    )
    return _contains_any(norm, ready_tokens)


def _looks_like_uvvis_empty_positions_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    if _is_negative_short_reply_fixed(filtered_text):
        return False
    if _is_affirmative_short_reply_fixed(filtered_text):
        return True
    ready_tokens = (
        "閮界┖浜?,
        "宸茬粡閮界┖浜?,
        "閮界暀绌轰簡",
        "宸茬粡鐣欑┖浜?,
        "鏍峰搧浣嶉兘绌轰簡",
        "鏍峰搧浣嶅拰鍙傛瘮浣嶉兘绌轰簡",
        "閮藉噯澶囧ソ浜?,
        "鍑嗗濂戒簡",
        "宸茬粡鍑嗗濂戒簡",
        "绌烘灦鍑嗗濂戒簡",
    )
    return _contains_any(norm, ready_tokens)


def _looks_like_uvvis_empty_then_start_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    if _is_negative_short_reply_fixed(filtered_text):
        return False
    empty_tokens = (
        "閮界┖浜?,
        "宸茬粡閮界┖浜?,
        "閮界暀绌轰簡",
        "宸茬粡鐣欑┖浜?,
        "鏍峰搧浣嶉兘绌轰簡",
        "鏍峰搧浣嶅拰鍙傛瘮浣嶉兘绌轰簡",
        "绌烘灦鍑嗗濂戒簡",
    )
    start_tokens = (
        "鍙互寮€濮嬩簡",
        "寮€濮嬪惂",
        "寮€濮嬫祴閲?,
        "寮€濮嬫壂鎻?,
        "鍙互寮€濮嬫壂鎻?,
        "鎵弿鍚?,
        "寮€濮嬬┖鏋舵壂鎻?,
    )
    return _contains_any(norm, empty_tokens) and _contains_any(norm, start_tokens)


def _looks_like_uvvis_start_scan_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    if _is_negative_short_reply_fixed(filtered_text):
        return False
    start_tokens = (
        "鍙互寮€濮嬩簡",
        "寮€濮嬪惂",
        "寮€濮嬫祴閲?,
        "寮€濮嬫壂鎻?,
        "鍙互寮€濮嬫壂鎻?,
        "鎵弿鍚?,
        "寮€濮嬬┖鏋舵壂鎻?,
    )
    return _contains_any(norm, start_tokens)


def _get_current_experiment_step_id(conn) -> str:
    step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
    if step_id:
        return step_id
    step_meta = _get_cached_experiment_step_meta(conn)
    return str(step_meta.get("step_id", "") or "").strip()


def _is_uvvis_step(step_id: str) -> bool:
    return str(step_id or "").strip() in {
        _UVVIS_SHARED_DARK_AIR_STEP_ID,
        _UVVIS_SHARED_BLANK_STEP_ID,
        _UVVIS_SAMPLE_LOAD_STEP_ID,
        _UVVIS_SAMPLE_LOAD_STEP_ID_LEGACY,
        _UVVIS_SAMPLE_RECORD_STEP_ID,
        _UVVIS_SAMPLE_RECORD_STEP_ID_LEGACY,
        _UVVIS_SAMPLE_CLEAN_STEP_ID,
        _UVVIS_SAMPLE_CLEAN_STEP_ID_LEGACY,
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
        _UVVIS_ANALYSIS_STEP_ID,
    }


def _is_uvvis_measurement_step(step_id: str) -> bool:
    return str(step_id or "").strip() in {
        _UVVIS_SHARED_DARK_AIR_STEP_ID,
        _UVVIS_SHARED_BLANK_STEP_ID,
        _UVVIS_SAMPLE_RECORD_STEP_ID,
        _UVVIS_SAMPLE_RECORD_STEP_ID_LEGACY,
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
    }


def _is_uvvis_kinetics_step(step_id: str) -> bool:
    return str(step_id or "").strip() in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
    }


def _is_uvvis_spectra_step(step_id: str) -> bool:
    return str(step_id or "").strip() in {
        _UVVIS_SHARED_DARK_AIR_STEP_ID,
        _UVVIS_SHARED_BLANK_STEP_ID,
        _UVVIS_SAMPLE_RECORD_STEP_ID,
        _UVVIS_SAMPLE_RECORD_STEP_ID_LEGACY,
    }


def _infer_uvvis_step_id_from_context(
    conn,
    original_text: str = "",
    filtered_text: str = "",
) -> str:
    state = getattr(conn, "_uvvis_direct_state", None)
    if isinstance(state, dict):
        state_step_id = str(state.get("step_id", "") or "").strip()
        if _is_uvvis_step(state_step_id):
            return state_step_id

    context_text = _normalize_text_for_match(
        " ".join(
            text
            for text in (
                _get_last_assistant_text_raw(conn),
                _get_recent_assistant_text(conn, limit=4),
                original_text,
                filtered_text,
            )
            if str(text or "").strip()
        )
    )
    if not context_text:
        return ""

    if _uvvis_shared_blank_step_enabled(conn) and _contains_any(
        context_text,
        (
            "2鍙锋牱鍝佸姩鍔涘",
            "sample2",
            "2鍙锋牱鍝佸弽搴旀恫",
            "2鍙锋牱鍝佸弬姣旀恫",
            "2鍙锋牱鍝佷綅",
        ),
    ):
        return _UVVIS_KINETICS_COMBINED_STEP_ID

    if _contains_any(
        context_text,
        (
            "4鍙锋牱鍝佸姩鍔涘",
            "sample4",
            "4鍙锋牱鍝佸弽搴旀恫",
            "4鍙锋牱鍝佸弬姣旀恫",
            "4鍙锋牱鍝佷綅",
        ),
    ):
        return _UVVIS_KINETICS_COMBINED_STEP_ID

    if _contains_any(
        context_text,
        (
            "鏆楃數娴佹牎姝?,
            "鍏变韩鏆楃數娴佹牎姝?,
            "鏆楃數娴佸拰绌烘皵鑳介噺鏍℃",
            "鍏变韩鏆楃數娴佸拰绌烘皵鑳介噺鏍℃",
            "绌烘皵鑳介噺鍑嗗",
            "绌烘皵鑳介噺鏂囦欢",
        ),
    ):
        return _UVVIS_SHARED_DARK_AIR_STEP_ID

    if _contains_any(
        context_text,
        (
            "鏆楃數娴?,
            "绌烘皵鍩虹嚎",
            "绌烘皵鑳介噺",
            "绾按绌虹櫧",
            "绾按姣旇壊鐨?,
            "鏍峰搧浣嶅拰鍙傛瘮浣嶉兘鐣欑┖",
            "鏍峰搧浣嶅拰鍙傛瘮浣嶅悇鏀惧叆绾按姣旇壊鐨?,
            "鍏堜笉瑕佹斁浠讳綍娑蹭綋",
        ),
    ):
        return _UVVIS_SHARED_BLANK_STEP_ID

    if _contains_any(
        context_text,
        (
            "瑁呭叆姣旇壊鐨?,
            "瑁呮牱鍑嗗",
            "鏍峰搧姣旇壊鐨?,
            "鏀惧叆鑷姩浜旇仈鏋?,
            "鍙傛瘮浣嶇函姘存瘮鑹茬毧淇濇寔涓嶅姩",
        ),
    ):
        return _UVVIS_SAMPLE_LOAD_STEP_ID

    if _contains_any(
        context_text,
        (
            "鎵归噺娴嬪厜璋?,
            "鍏夎氨娴嬮噺涓庤褰?,
            "寮€濮?-5鍙锋牱鍝佺殑鍏夎氨娴嬮噺",
            "1-5鍙锋牱鍝佺殑鍏夎氨",
            "寮€濮嬫牱鍝佹祴閲?,
            "位max",
            "lambda max",
        ),
    ):
        return _UVVIS_SAMPLE_RECORD_STEP_ID

    if _contains_any(
        context_text,
        (
            "娓呮礂姣旇壊鐨?,
            "缁熶竴娓呮礂姣旇壊鐨?,
            "娴嬮噺鍚庣殑缁熶竴娓呮礂",
        ),
    ):
        return _UVVIS_SAMPLE_CLEAN_STEP_ID

    if _contains_any(
        context_text,
        (
            "鏁版嵁鍒嗘瀽",
            "缁樺埗ag nps鍚告敹鍏夎氨",
            "lambda max涓巏br鐢ㄩ噺",
        ),
    ):
        return _UVVIS_ANALYSIS_STEP_ID

    return ""


def _normalize_uvvis_device_id(conn) -> str:
    device_id = str(getattr(conn, "device_id", "") or "").strip()
    if not device_id and isinstance(getattr(conn, "headers", None), dict):
        headers = getattr(conn, "headers", {}) or {}
        device_id = str(
            headers.get("device-id")
            or headers.get("Device-Id")
            or headers.get("device_id")
            or ""
        ).strip()
    safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", device_id.lower()).strip("._-")
    return safe or "unknown_device"


def _resolve_uvvis_experiment_data_dirs(conn) -> list[Path]:
    yaml_path = str(getattr(conn, "experiment_yaml_path", "") or "").strip()
    if not yaml_path and hasattr(conn, "_resolve_experiment_yaml_path"):
        try:
            yaml_path = str(conn._resolve_experiment_yaml_path() or "").strip()
        except Exception:
            yaml_path = ""
    if not yaml_path:
        return []

    try:
        yaml_file = Path(yaml_path).expanduser().resolve()
    except Exception:
        return []

    if not yaml_file.name.lower().endswith((".yaml", ".yml")):
        return []

    if yaml_file.parent.name.lower() != "configs":
        return []

    data_dir = (yaml_file.parent.parent / "data").resolve()
    return [data_dir, (data_dir / "uv_data_common").resolve()]


def _resolve_uvvis_native_output_root(conn) -> Path:
    override_root = str(conn.config.get("uvvis_scan_output_root", "") or "").strip()
    if override_root:
        return Path(override_root).expanduser().resolve()

    for candidate in _resolve_uvvis_experiment_data_dirs(conn):
        if candidate.name.lower() == "uv_data_common":
            return candidate

    return (Path("data") / "uv_data_common").resolve()


def _resolve_uvvis_root_candidates(conn) -> list[Path]:
    candidates: list[Path] = []

    override_root = str(conn.config.get("uvvis_scan_output_root", "") or "").strip()
    if override_root:
        root = Path(override_root).expanduser().resolve()
        candidates.append(root)
        candidates.append(root.parent)

    candidates.extend(_resolve_uvvis_experiment_data_dirs(conn))

    candidates.append(Path("data").resolve())
    candidates.append((Path("data") / "uv_data_common").resolve())

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_uvvis_runtime_device_dirs(conn) -> list[Path]:
    device_id = _normalize_uvvis_device_id(conn)
    dirs: list[Path] = []
    for root in _resolve_uvvis_root_candidates(conn):
        dirs.append((root / device_id).resolve())
    return dirs


def _resolve_uvvis_shared_blank_dirs(conn) -> list[Path]:
    candidates: list[Path] = []

    for root in _resolve_uvvis_root_candidates(conn):
        if root.name.lower() == "uv_data_common":
            candidates.append(root)
        else:
            candidates.append((root / "uv_data_common").resolve())

    deduped: list[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _path_exists(path_value) -> bool:
    path_text = str(path_value or "").strip()
    if not path_text:
        return False
    try:
        return Path(path_text).expanduser().exists()
    except Exception:
        return False


def _extract_uvvis_blank_baseline_state(payload) -> dict | None:
    named = _collect_payload_named_values(
        payload,
        (
            "blank_baseline_exists",
            "blank_baseline_status",
            "blank_baseline_csv",
            "blank_baseline_manifest_json",
        ),
    )
    if not named:
        return None

    return {
        "blank_baseline_exists": bool(_normalize_bool(named.get("blank_baseline_exists"))),
        "blank_baseline_status": str(named.get("blank_baseline_status", "") or "").strip(),
        "blank_baseline_csv": str(named.get("blank_baseline_csv", "") or "").strip(),
        "blank_baseline_manifest_json": str(
            named.get("blank_baseline_manifest_json", "") or ""
        ).strip(),
    }


def _blank_baseline_state_has_artifact(state) -> bool:
    if not isinstance(state, dict):
        return False
    return any(
        _path_exists(state.get(key))
        for key in ("blank_baseline_csv", "blank_baseline_manifest_json")
    )


def _looks_like_uvvis_blank_artifact(path: Path) -> bool:
    name = path.name.lower()
    if not name or name.startswith("."):
        return False
    if "connection_state" in name:
        return False
    if path.suffix.lower() not in {".csv", ".json"}:
        return False
    return any(token in name for token in ("blank", "baseline"))


def _stat_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _read_uvvis_blank_baseline_state_from_shared_dir(conn) -> dict | None:
    latest_csv: Path | None = None
    latest_manifest: Path | None = None

    for shared_dir in _resolve_uvvis_shared_blank_dirs(conn):
        try:
            if not shared_dir.exists() or not shared_dir.is_dir():
                continue
        except Exception:
            continue

        try:
            candidates = [path for path in shared_dir.rglob("*") if path.is_file()]
        except Exception:
            continue

        for candidate in candidates:
            if not _looks_like_uvvis_blank_artifact(candidate):
                continue
            suffix = candidate.suffix.lower()
            if suffix == ".csv" and (
                latest_csv is None or _stat_mtime(candidate) > _stat_mtime(latest_csv)
            ):
                latest_csv = candidate
            if suffix == ".json" and (
                latest_manifest is None
                or _stat_mtime(candidate) > _stat_mtime(latest_manifest)
            ):
                latest_manifest = candidate

    if latest_csv is None and latest_manifest is None:
        return None

    return {
        "blank_baseline_exists": True,
        "blank_baseline_status": "reused_from_shared_dir",
        "blank_baseline_csv": str(latest_csv.resolve()) if latest_csv else "",
        "blank_baseline_manifest_json": (
            str(latest_manifest.resolve()) if latest_manifest else ""
        ),
    }


def _looks_like_uvvis_dark_current_artifact(path: Path) -> bool:
    name = path.name.lower()
    if not name or name.startswith("."):
        return False
    if "connection_state" in name:
        return False
    if path.suffix.lower() != ".json":
        return False
    return "dark_current" in name


def _read_uvvis_dark_current_state_from_shared_dir(conn) -> dict | None:
    latest_json: Path | None = None

    for shared_dir in _resolve_uvvis_shared_blank_dirs(conn):
        try:
            if not shared_dir.exists() or not shared_dir.is_dir():
                continue
        except Exception:
            continue

        try:
            candidates = [path for path in shared_dir.rglob("*") if path.is_file()]
        except Exception:
            continue

        for candidate in candidates:
            if not _looks_like_uvvis_dark_current_artifact(candidate):
                continue
            if latest_json is None or _stat_mtime(candidate) > _stat_mtime(latest_json):
                latest_json = candidate

    if latest_json is None:
        return None

    return {
        "dark_current_exists": True,
        "dark_current_status": "reused_from_shared_dir",
        "dark_current_json": str(latest_json.resolve()),
    }


def _extract_uvvis_liquid_blank_state(payload) -> dict | None:
    named = _collect_payload_named_values(
        payload,
        (
            "liquid_blank_exists",
            "liquid_blank_status",
            "liquid_blank_csv",
            "liquid_blank_manifest_json",
        ),
    )
    if not named:
        return None

    return {
        "liquid_blank_exists": bool(_normalize_bool(named.get("liquid_blank_exists"))),
        "liquid_blank_status": str(named.get("liquid_blank_status", "") or "").strip(),
        "liquid_blank_csv": str(named.get("liquid_blank_csv", "") or "").strip(),
        "liquid_blank_manifest_json": str(
            named.get("liquid_blank_manifest_json", "") or ""
        ).strip(),
    }


def _liquid_blank_state_has_artifact(state) -> bool:
    if not isinstance(state, dict):
        return False
    return any(
        _path_exists(state.get(key))
        for key in ("liquid_blank_csv", "liquid_blank_manifest_json")
    )


def _looks_like_uvvis_liquid_blank_artifact(path: Path) -> bool:
    name = path.name.lower()
    if not name or name.startswith("."):
        return False
    if "connection_state" in name or "air_blank" in name:
        return False
    if path.suffix.lower() not in {".csv", ".json"}:
        return False
    return any(token in name for token in ("liquid_blank", "pure_water_blank"))


def _read_uvvis_liquid_blank_state_from_shared_dir(conn) -> dict | None:
    latest_csv: Path | None = None
    latest_manifest: Path | None = None

    for shared_dir in _resolve_uvvis_shared_blank_dirs(conn):
        try:
            if not shared_dir.exists() or not shared_dir.is_dir():
                continue
        except Exception:
            continue

        try:
            candidates = [path for path in shared_dir.rglob("*") if path.is_file()]
        except Exception:
            continue

        for candidate in candidates:
            if not _looks_like_uvvis_liquid_blank_artifact(candidate):
                continue
            suffix = candidate.suffix.lower()
            if suffix == ".csv" and (
                latest_csv is None or _stat_mtime(candidate) > _stat_mtime(latest_csv)
            ):
                latest_csv = candidate
            if suffix == ".json" and (
                latest_manifest is None
                or _stat_mtime(candidate) > _stat_mtime(latest_manifest)
            ):
                latest_manifest = candidate

    if latest_csv is None and latest_manifest is None:
        return None

    return {
        "liquid_blank_exists": True,
        "liquid_blank_status": "reused_from_shared_dir",
        "liquid_blank_csv": str(latest_csv.resolve()) if latest_csv else "",
        "liquid_blank_manifest_json": (
            str(latest_manifest.resolve()) if latest_manifest else ""
        ),
    }


def _collect_payload_strings(payload) -> list[str]:
    strings: list[str] = []

    def _visit(node):
        if isinstance(node, dict):
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)
        elif isinstance(node, str):
            text = node.strip()
            if text:
                strings.append(text)

    _visit(_to_plain_data(payload))
    return strings


def _collect_payload_named_values(payload, target_keys) -> dict:
    remaining_keys = set(target_keys or ())
    found = {}

    def _visit(node):
        nonlocal remaining_keys
        if not remaining_keys:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if key in remaining_keys and value not in (None, ""):
                    found[key] = value
                    remaining_keys.remove(key)
                if isinstance(value, (dict, list)):
                    _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(_to_plain_data(payload))
    return found


def _extract_float_value(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _extract_int_value(value):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed


_CHINESE_GROUP_DIGITS = {
    "闆?: 0,
    "涓€": 1,
    "浜?: 2,
    "涓?: 2,
    "涓?: 3,
    "鍥?: 4,
    "浜?: 5,
    "鍏?: 6,
    "涓?: 7,
    "鍏?: 8,
    "涔?: 9,
}


def _parse_small_chinese_positive_int(text: str) -> int | None:
    token = str(text or "").strip()
    if not token:
        return None
    if token == "鍗?:
        return 10
    if token.startswith("鍗?):
        ones = _CHINESE_GROUP_DIGITS.get(token[1:])
        return 10 + ones if ones is not None else None
    if token.endswith("鍗?):
        tens = _CHINESE_GROUP_DIGITS.get(token[:-1])
        return tens * 10 if tens is not None else None
    if "鍗? in token:
        left, right = token.split("鍗?, 1)
        tens = _CHINESE_GROUP_DIGITS.get(left)
        ones = _CHINESE_GROUP_DIGITS.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return _CHINESE_GROUP_DIGITS.get(token)


def _extract_explicit_group_number_from_text(*texts: str) -> int | None:
    combined = " ".join(str(text or "").strip() for text in texts if str(text or "").strip())
    if not combined:
        return None
    normalized = textUtils.normalize_spoken_text(combined)
    numeric_match = (
        re.search(r"绗琝s*([0-9]{1,2})\s*缁?, normalized)
        or re.search(r"([0-9]{1,2})\s*缁?, normalized)
    )
    if numeric_match:
        value = _extract_int_value(numeric_match.group(1))
        if isinstance(value, int) and value >= 1:
            return value

    chinese_match = (
        re.search(r"绗琝s*([闆朵竴浜屼袱涓夊洓浜斿叚涓冨叓涔濆崄]{1,3})\s*缁?, normalized)
        or re.search(r"([闆朵竴浜屼袱涓夊洓浜斿叚涓冨叓涔濆崄]{1,3})\s*缁?, normalized)
    )
    if chinese_match:
        value = _parse_small_chinese_positive_int(chinese_match.group(1))
        if isinstance(value, int) and value >= 1:
            return value
    return None


def _is_exp2_uvvis_experiment(conn) -> bool:
    yaml_path = str(getattr(conn, "experiment_yaml_path", "") or "").strip().replace("\\", "/").lower()
    return "exp2_uv_vis_analysis" in yaml_path


def _extract_current_progress_group_number(progress_payload) -> int | None:
    progress = _extract_experiment_current_progress(progress_payload)
    if not isinstance(progress, dict):
        return None
    for candidate in (
        progress.get("group_number"),
        (progress.get("current_data") or {}).get("group_number")
        if isinstance(progress.get("current_data"), dict)
        else None,
    ):
        value = _extract_int_value(candidate)
        if isinstance(value, int) and value >= 1:
            return value
    return None


async def _sync_exp2_group_number_from_turn(
    conn,
    original_text: str,
    filtered_text: str,
    *,
    preferred_step_id: str = "",
) -> int | None:
    if not _is_exp2_uvvis_experiment(conn):
        return None
    group_number = _extract_explicit_group_number_from_text(original_text, filtered_text)
    if group_number is None:
        return None

    setattr(conn, "experiment_current_group_number", group_number)
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    step_id = str(preferred_step_id or _get_current_experiment_step_id(conn) or "").strip()
    if not session_id or not step_id:
        return group_number

    try:
        await _call_experiment_graph_tool_fast(
            conn,
            "redirect_to_step",
            {
                "session_id": session_id,
                "step_id": step_id,
                "force": True,
                "group_number": group_number,
            },
            priority="foreground",
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"exp2 group sync failed: group_number={group_number}, step_id={step_id}, error={exc}"
        )
    return group_number


def _resolve_uvvis_group_output_dir(conn, group_number: int | None = None) -> Path:
    output_root = _resolve_uvvis_native_output_root(conn)
    device_id = _normalize_uvvis_device_id(conn)
    target = (output_root / device_id).resolve()
    if isinstance(group_number, int) and group_number >= 1:
        target = (target / str(group_number)).resolve()
    return target


def _extract_uvvis_payload_message(payload) -> str:
    text = _extract_text_from_result_payload(payload)
    if text:
        return text
    data = _to_plain_data(payload)
    if isinstance(data, str):
        return data.strip()
    return ""


def _payload_looks_busy_or_inaccessible(payload) -> bool:
    data = _to_plain_data(payload)
    for candidate in (
        data,
        data.get("result") if isinstance(data, dict) and isinstance(data.get("result"), dict) else None,
    ):
        if not isinstance(candidate, dict):
            continue
        access_state = str(candidate.get("access_state", "") or "").strip().lower()
        if access_state == "inaccessible":
            return True
        occupied = _normalize_bool(candidate.get("occupied"))
        active_measurement = _normalize_bool(candidate.get("active_measurement"))
        available = _normalize_bool(candidate.get("available"))
        if occupied is True or active_measurement is True:
            return True
        if available is False and any(
            key in candidate
            for key in ("available", "occupied", "active_measurement", "access_state")
        ):
            return True

    text = _extract_uvvis_payload_message(payload).lower()
    if not text:
        return False
    busy_tokens = (
        "access_state=inaccessible",
        "inaccessible",
        "busy",
        "occupied",
        "already held",
        "lease already held",
        "lease is held",
        "lease owner",
        "鍗犵敤",
        "蹇?,
        "璇蜂綘杩?min鍐嶈瘯",
    )
    return any(token in text for token in busy_tokens)


def _payload_mentions_missing_blank(payload) -> bool:
    text = _extract_uvvis_payload_message(payload).lower()
    if not text:
        return False
    missing_tokens = (
        "liquid blank",
        "pure water",
        "pure_water",
        "blank",
        "绌虹櫧",
        "绾按",
        "鍙傛瘮娑?,
        "鍖栧绌虹櫧",
    )
    if not any(token in text for token in missing_tokens):
        return False
    return any(
        token in text
        for token in (
            "missing",
            "not found",
            "absent",
            "need",
            "required",
            "涓嶅瓨鍦?,
            "缂哄皯",
            "娌℃湁",
            "鏈壘鍒?,
            "杩樻病鏈?,
        )
    )


def _extract_uvvis_payload_phase(payload) -> str:
    data = _to_plain_data(payload)
    if not isinstance(data, dict):
        return ""

    phase = str(data.get("phase", "") or "").strip().lower()
    if phase:
        return phase

    nested = data.get("result")
    if isinstance(nested, dict):
        return str(nested.get("phase", "") or "").strip().lower()
    return ""


def _payload_mentions_missing_shared_prep(payload) -> bool:
    if _extract_uvvis_payload_phase(payload) == "shared_prep_missing":
        return True

    text = _extract_uvvis_payload_message(payload).lower()
    return (
        "shared dark-current and air baseline data are missing" in text
        or (
            "ready_for_samples=false" in text
            and "sample positions are blank" in text
            and "baseline" in text
        )
    )


def _payload_mentions_shared_prep_saturated(payload) -> bool:
    if _extract_uvvis_payload_phase(payload) == "shared_prep_saturated":
        return True

    text = _extract_uvvis_payload_message(payload).lower()
    return "shared air baseline is saturated" in text


def _payload_indicates_liquid_blank_ready(payload) -> bool:
    if _extract_uvvis_payload_phase(payload) == "liquid_blank_ready":
        return True

    state = _extract_uvvis_liquid_blank_state(payload)
    return _liquid_blank_state_has_artifact(state)


def _payload_mentions_reusable_blank(payload) -> bool:
    named = _collect_payload_named_values(payload, ("blank_baseline_exists",))
    exists = _normalize_bool(named.get("blank_baseline_exists"))
    if exists is True:
        return True

    text = _extract_uvvis_payload_message(payload).lower()
    if not text:
        return False

    if not any(
        token in text
        for token in ("liquid blank", "pure water", "pure_water", "blank", "绌虹櫧", "绾按")
    ):
        return False

    return any(
        token in text
        for token in (
            "reused",
            "reuse",
            "existing",
            "exists",
            "available",
            "already",
            "鍙鐢?,
            "宸插瓨鍦?,
            "宸叉湁",
            "宸茶褰?,
            "澶嶇敤",
        )
    )


def _uvvis_blank_baseline_exists(conn, payload=None) -> bool:
    disk_state = _read_uvvis_blank_baseline_state_from_shared_dir(conn)
    if disk_state is not None:
        setattr(conn, "_last_uvvis_blank_baseline_state", disk_state)
        return True

    payload_state = _extract_uvvis_blank_baseline_state(payload)
    if _blank_baseline_state_has_artifact(payload_state):
        normalized_payload_state = dict(payload_state)
        normalized_payload_state["blank_baseline_exists"] = True
        setattr(conn, "_last_uvvis_blank_baseline_state", normalized_payload_state)
        return True

    blank_state = getattr(conn, "_last_uvvis_blank_baseline_state", None)
    if _blank_baseline_state_has_artifact(blank_state):
        return True

    return False


def _extract_uvvis_dark_current_state(payload) -> dict | None:
    named = _collect_payload_named_values(
        payload,
        (
            "dark_current_exists",
            "dark_current_status",
            "dark_current_json",
        ),
    )
    if not named:
        return None

    return {
        "dark_current_exists": bool(_normalize_bool(named.get("dark_current_exists"))),
        "dark_current_status": str(named.get("dark_current_status", "") or "").strip(),
        "dark_current_json": str(named.get("dark_current_json", "") or "").strip(),
    }


def _dark_current_state_has_artifact(state) -> bool:
    if not isinstance(state, dict):
        return False
    return _path_exists(state.get("dark_current_json"))


def _uvvis_dark_current_exists(conn, payload=None) -> bool:
    disk_state = _read_uvvis_dark_current_state_from_shared_dir(conn)
    if disk_state is not None:
        setattr(conn, "_last_uvvis_dark_current_state", disk_state)
        return True

    payload_state = _extract_uvvis_dark_current_state(payload)
    if _dark_current_state_has_artifact(payload_state):
        normalized_payload_state = dict(payload_state)
        normalized_payload_state["dark_current_exists"] = True
        setattr(conn, "_last_uvvis_dark_current_state", normalized_payload_state)
        return True

    dark_state = getattr(conn, "_last_uvvis_dark_current_state", None)
    if _dark_current_state_has_artifact(dark_state):
        return True

    return False


def _uvvis_shared_dark_air_cache_ready(conn, payload=None) -> bool:
    return _uvvis_dark_current_exists(conn, payload) and _uvvis_blank_baseline_exists(
        conn, payload
    )


def _uvvis_shared_liquid_blank_exists(conn, payload=None) -> bool:
    disk_state = _read_uvvis_liquid_blank_state_from_shared_dir(conn)
    if disk_state is not None:
        setattr(conn, "_last_uvvis_liquid_blank_state", disk_state)
        return True

    payload_state = _extract_uvvis_liquid_blank_state(payload)
    if _liquid_blank_state_has_artifact(payload_state):
        normalized_payload_state = dict(payload_state)
        normalized_payload_state["liquid_blank_exists"] = True
        setattr(conn, "_last_uvvis_liquid_blank_state", normalized_payload_state)
        return True

    blank_state = getattr(conn, "_last_uvvis_liquid_blank_state", None)
    if _liquid_blank_state_has_artifact(blank_state):
        return True

    return False


def _payload_has_success_flag(payload) -> bool | None:
    data = _to_plain_data(payload)
    if isinstance(data, dict):
        success = data.get("success")
        if isinstance(success, bool):
            return success
        result = data.get("result")
        if isinstance(result, dict):
            nested = result.get("success")
            if isinstance(nested, bool):
                return nested
    return None


async def _execute_uvvis_tool_payload(conn, tool_name: str, arguments: dict) -> dict:
    manager = _get_server_mcp_manager(conn)
    if manager is None:
        raise RuntimeError(_UVVIS_NOT_READY_REPLY)
    if str(tool_name or "").startswith("uvvis_"):
        try:
            ready = await manager.ensure_client_initialized("uvvis")
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"uvvis client targeted initialize failed before {tool_name}: {exc}"
            )
            ready = False
        if not ready or not manager.is_mcp_tool(tool_name):
            raise RuntimeError(_UVVIS_NOT_READY_REPLY)

    busy_token = f"uvvis:{tool_name}"
    busy_acquired = False
    if hasattr(conn, "acquire_external_busy"):
        try:
            conn.acquire_external_busy(busy_token)
            busy_acquired = True
        except Exception:
            busy_acquired = False

    try:
        result = await _execute_server_mcp_tool_direct(conn, tool_name, arguments)
    finally:
        if busy_acquired and hasattr(conn, "release_external_busy"):
            try:
                conn.release_external_busy(busy_token)
            except Exception:
                pass

    payload = finalize_server_mcp_payload(
        result,
        tool_name=tool_name,
        arguments=arguments,
    )
    sync_server_mcp_payload_state(
        conn,
        tool_name=tool_name,
        payload=payload,
        arguments=arguments,
    )
    return payload


def _extract_uvvis_status_snapshot(payload) -> dict:
    data = _to_plain_data(payload)
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data.get("result")
    if not isinstance(data, dict):
        data = {}

    return {
        "available": _normalize_bool(data.get("available")),
        "occupied": _normalize_bool(data.get("occupied")),
        "active_measurement": _normalize_bool(data.get("active_measurement")),
        "lease_owner_is_caller": _normalize_bool(data.get("lease_owner_is_caller")),
        "session_key": str(data.get("session_key", "") or "").strip(),
        "lease_owner": str(data.get("lease_owner", "") or "").strip(),
        "last_tool_name": str(data.get("last_tool_name", "") or "").strip(),
        "message": _extract_uvvis_payload_message(payload),
    }


async def _get_uvvis_session_status(conn) -> dict:
    manager = _get_server_mcp_manager(conn)
    if manager is None:
        return {"ok": False, "message": _UVVIS_NOT_READY_REPLY}
    try:
        ready = await manager.ensure_client_initialized("uvvis")
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"uvvis client targeted initialize failed during status check: {exc}"
        )
        ready = False
    if not ready or not manager.is_mcp_tool("uvvis_session"):
        return {"ok": False, "message": _UVVIS_NOT_READY_REPLY}

    try:
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_session",
            {"action": "status"},
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(f"uvvis_session status failed: {exc}")
        message = str(exc or "").strip()
        return {"ok": False, "message": message or _UVVIS_NOT_READY_REPLY}

    status = _extract_uvvis_status_snapshot(payload)
    status["ok"] = True
    return status


def _looks_like_uvvis_status_query(
    conn,
    original_text: str,
    filtered_text: str,
    *,
    inferred_step_id: str = "",
) -> bool:
    normalized = _normalize_text_for_match(
        " ".join(
            text
            for text in (original_text, filtered_text)
            if str(text or "").strip()
        )
    )
    if not normalized:
        return False

    query_tokens = (
        "鐘舵€?,
        "鍦ㄥ伐浣?,
        "姝ｅ湪宸ヤ綔",
        "宸ヤ綔鍚?,
        "蹇欏悧",
        "绌洪棽鍚?,
        "娴嬪畬",
        "缁撴潫浜嗗悧",
        "瀹屾垚浜嗗悧",
        "瀹屾垚浜嗗惂",
    )
    if not any(token in normalized for token in query_tokens):
        return False

    if any(token in normalized for token in ("uvvis", "绱鍙", "鍏夎氨浠?)):
        return True
    if _is_uvvis_step(inferred_step_id):
        return True

    state = getattr(conn, "_uvvis_direct_state", None)
    if isinstance(state, dict) and _is_uvvis_step(str(state.get("step_id", "") or "").strip()):
        return True
    return False


def _compose_uvvis_status_reply(conn, status: dict, *, inferred_step_id: str = "") -> str:
    if not isinstance(status, dict) or not status.get("ok"):
        return str((status or {}).get("message", "") or _UVVIS_NOT_READY_REPLY).strip()

    if status.get("active_measurement") is True:
        return "UV-Vis 鐜板湪姝ｅ湪宸ヤ綔銆?
    if status.get("occupied") is True and not status.get("lease_owner_is_caller"):
        return "UV-Vis 鐜板湪琚埆鐨勪細璇濆崰鐢紝杩樻病绌哄嚭鏉ャ€?

    state = _get_uvvis_direct_state(conn, inferred_step_id if _is_uvvis_step(inferred_step_id) else "")
    phase = str(state.get("phase", "") or "").strip()
    if phase == "blank_reusable":
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傝繖涓€姝ュ凡缁忕‘璁ゅ綋鍓嶆壒娆＄函姘寸┖鐧藉彲澶嶇敤锛岀户缁笅涓€姝ユ椂璁板緱淇濈暀鎴栭噸鏂版斁濂藉弬姣斾綅绾按姣旇壊鐨裤€?
    if phase == "await_empty_positions":
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傝繖涓€姝ュ湪绛変綘纭1鍒?鍙锋牱鍝佷綅鍜屽弬姣斾綅閮藉凡鐣欑┖銆?
    if phase == "await_scan_start":
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傝繖涓€姝ュ湪绛変綘纭鍙互寮€濮嬫壂鎻忋€?
    if phase == "await_pure_water_blank" and _uvvis_shared_blank_step_enabled(conn):
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傛殫鐢垫祦鏍℃宸茬粡瀹屾垚锛岃繖涓€姝ュ湪绛変綘鎶婁竴鍒颁簲鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁涓€涓函姘存瘮鑹茬毧銆?
    if phase == "await_shared_prep_reset":
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傝繖涓€姝ュ湪绛変綘鎶婁竴鍒颁簲鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥竻绌猴紝鎴戝厛琛ュ仛鍏变韩鍓嶇疆鏍℃銆?
    if phase == "await_reaction_sample":
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣滐紝杩欎竴姝ュ湪绛変綘鎶婃牱鍝佸拰鍙傛瘮娑叉斁濂姐€?
    if phase == "await_liquid_blank":
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣滐紝杩欎竴姝ュ湪绛変綘鎶婃寚瀹氱殑绌虹櫧娑叉斁濂姐€?

    if (
        inferred_step_id == _UVVIS_SHARED_BLANK_STEP_ID
        and _uvvis_shared_blank_step_enabled(conn)
        and _uvvis_shared_liquid_blank_exists(conn)
    ):
        return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€傛殫鐢垫祦鏍℃宸茬粡瀹屾垚銆?
    return "UV-Vis 鐜板湪娌℃湁鍦ㄥ伐浣溿€?


async def _ensure_uvvis_session_key(conn) -> tuple[str, str]:
    existing_key = str(getattr(conn, "_uvvis_session_key", "") or "").strip()
    if existing_key:
        return existing_key, ""

    manager = _get_server_mcp_manager(conn)
    if manager is None:
        return "", _UVVIS_NOT_READY_REPLY
    try:
        ready = await manager.ensure_client_initialized("uvvis")
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"uvvis client targeted initialize failed: {exc}"
        )
        ready = False
    if not ready or not manager.is_mcp_tool("uvvis_session"):
        return "", _UVVIS_NOT_READY_REPLY

    try:
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_session",
            {"action": "acquire"},
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(f"uvvis_session acquire failed: {exc}")
        status = await _get_uvvis_session_status(conn)
        status_key = str(status.get("session_key", "") or "").strip()
        if status.get("lease_owner_is_caller") and status_key:
            setattr(conn, "_uvvis_session_key", status_key)
            return status_key, ""
        if status.get("active_measurement") is True or status.get("occupied") is True:
            return "", _UVVIS_BUSY_REPLY
        if _payload_looks_busy_or_inaccessible({"message": str(exc)}):
            return "", _UVVIS_BUSY_REPLY
        return "", _UVVIS_NOT_READY_REPLY

    payload_key = ""
    if isinstance(payload, dict):
        payload_key = str(payload.get("session_key") or "").strip()
        if not payload_key and isinstance(payload.get("result"), dict):
            payload_key = str(payload["result"].get("session_key") or "").strip()

    if payload_key:
        setattr(conn, "_uvvis_session_key", payload_key)
        return payload_key, ""

    status = await _get_uvvis_session_status(conn)
    status_key = str(status.get("session_key", "") or "").strip()
    if status.get("lease_owner_is_caller") and status_key:
        setattr(conn, "_uvvis_session_key", status_key)
        return status_key, ""

    if status.get("active_measurement") is True or status.get("occupied") is True:
        return "", _UVVIS_BUSY_REPLY
    if _payload_looks_busy_or_inaccessible(payload):
        return "", _UVVIS_BUSY_REPLY

    message = _extract_uvvis_payload_message(payload)
    if message:
        return "", message
    return "", _UVVIS_NOT_READY_REPLY


def _clear_uvvis_direct_state(conn) -> None:
    setattr(conn, "_uvvis_direct_state", {})


def _get_uvvis_direct_state(conn, step_id: str = "") -> dict:
    state = getattr(conn, "_uvvis_direct_state", None)
    if not isinstance(state, dict):
        state = {}
    current_step_id = str(step_id or _get_current_experiment_step_id(conn) or "").strip()
    state_step_id = str(state.get("step_id", "") or "").strip()
    if state_step_id and current_step_id and state_step_id != current_step_id:
        state = {}
    if current_step_id and not state_step_id:
        state = {}
    if state != getattr(conn, "_uvvis_direct_state", None):
        setattr(conn, "_uvvis_direct_state", state)
    return state


def _set_uvvis_direct_state(conn, **updates) -> dict:
    state = dict(_get_uvvis_direct_state(conn))
    state.update(updates)
    setattr(conn, "_uvvis_direct_state", state)
    return state


def _extract_uvvis_sample_position_from_text(text: str) -> int | None:
    source = textUtils.normalize_spoken_text(text or "")
    if not source:
        return None

    patterns = (
        r"([1-5])\s*\u53f7\u4f4d",
        r"sample[_\-\s]*([1-5])",
        r"\u653e\u5728\s*([1-5])\s*\u53f7\u4f4d",
        r"\u6837\u54c1\u4f4d\s*([1-5])",
        r"([涓€浜屼笁鍥涗簲])\s*\u53f7\u4f4d",
        r"\u653e\u5728\s*([涓€浜屼笁鍥涗簲])\s*\u53f7\u4f4d",
    )
    chinese_map = {
        "涓€": 1,
        "浜?: 2,
        "涓?: 3,
        "鍥?: 4,
        "浜?: 5,
    }
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if not match:
            continue
        token = match.group(1)
        if token.isdigit():
            value = int(token)
        else:
            value = chinese_map.get(token)
        if value and 1 <= value <= 5:
            return value
    return None


def _extract_uvvis_sample_position_from_path(path_text: str) -> int | None:
    source = str(path_text or "").strip().lower()
    if not source:
        return None

    match = re.search(r"sample(?:_run_)?sample([1-5])", source)
    if not match:
        match = re.search(r"sample([1-5])", source)
    if not match:
        match = re.search(r"(^|[\\/ _-])([1-5])鍙?, source)
        if match:
            return int(match.group(2))
        return None
    return int(match.group(1))


def _extract_uvvis_summary_rows_from_csv(path_text: str) -> dict[int, dict]:
    path = Path(str(path_text or "").strip())
    if not path.exists():
        return {}

    rows: dict[int, dict] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_position = _extract_int_value(row.get("sample_position"))
                lambda_max_nm = _extract_float_value(row.get("lambda_max_nm"))
                if sample_position is None or lambda_max_nm is None:
                    continue
                rows[sample_position] = {
                    "lambda_max_nm": lambda_max_nm,
                    "max_absorbance": _extract_float_value(row.get("max_absorbance")),
                    "summary_csv": str(path.resolve()),
                }
    except Exception:
        return {}
    return rows


def _extract_uvvis_peak_from_absorbance_csv(path_text: str) -> dict | None:
    path = Path(str(path_text or "").strip())
    if not path.exists():
        return None

    max_row = None
    valid_points = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                wavelength_nm = _extract_float_value(row.get("wavelength_nm"))
                absorbance = _extract_float_value(row.get("absorbance"))
                if wavelength_nm is None or absorbance is None:
                    continue
                valid_points += 1
                if max_row is None or absorbance > max_row["max_absorbance"]:
                    max_row = {
                        "lambda_max_nm": wavelength_nm,
                        "max_absorbance": absorbance,
                    }
    except Exception:
        return None

    if max_row is None:
        return None
    max_row["valid_absorbance_points"] = valid_points
    max_row["peak_source_csv"] = str(path.resolve())
    return max_row


def _extract_uvvis_measure_spectra_rows(payload, conn) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    data = _to_plain_data(payload)

    def _visit(node):
        if isinstance(node, dict):
            sample_position = _extract_int_value(
                node.get("sample_position")
                or node.get("position")
                or node.get("sample_index")
            )
            lambda_max_nm = _extract_float_value(
                node.get("lambda_max_nm")
                or node.get("lambda_max")
                or node.get("max_lambda_nm")
            )
            if sample_position is not None and lambda_max_nm is not None:
                rows[sample_position] = {
                    "lambda_max_nm": lambda_max_nm,
                    "max_absorbance": _extract_float_value(
                        node.get("max_absorbance")
                        or node.get("absorbance_max")
                        or node.get("peak_absorbance")
                    ),
                }
            if sample_position is not None and sample_position not in rows:
                absorbance_path = (
                    node.get("absorbance_output_csv")
                    or node.get("absorbance_csv")
                    or node.get("output_csv")
                    or node.get("csv_path")
                )
                peak = _extract_uvvis_peak_from_absorbance_csv(absorbance_path)
                if peak is not None:
                    rows[sample_position] = peak
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(data)

    if len(rows) >= 5:
        return rows

    summary_paths = []
    absorbance_paths = []
    for text in _collect_payload_strings(data):
        lower_text = text.lower()
        if lower_text.endswith("_summary.csv") or "summary.csv" in lower_text:
            summary_paths.append(text)
        if lower_text.endswith("_absorbance.csv") or "absorbance.csv" in lower_text:
            absorbance_paths.append(text)

    for path_text in summary_paths:
        rows.update(_extract_uvvis_summary_rows_from_csv(path_text))
        if len(rows) >= 5:
            return rows

    for path_text in absorbance_paths:
        sample_position = _extract_uvvis_sample_position_from_path(path_text)
        if sample_position is None or sample_position in rows:
            continue
        peak = _extract_uvvis_peak_from_absorbance_csv(path_text)
        if peak is not None:
            rows[sample_position] = peak
    if len(rows) >= 5:
        return rows

    for sample_position in _UVVIS_SAMPLE_POSITIONS:
        if sample_position in rows:
            continue
        found_peak = None
        for device_dir in _resolve_uvvis_runtime_device_dirs(conn):
            candidate_paths = [
                device_dir / f"sample{sample_position}_latest_absorbance.csv",
                device_dir / f"sample_run_sample{sample_position}_latest_absorbance.csv",
            ]
            if (device_dir / f"sample{sample_position}_latest_absorbance.csv").exists():
                candidate_paths.insert(0, device_dir / f"sample{sample_position}_latest_absorbance.csv")
            for candidate in candidate_paths:
                peak = _extract_uvvis_peak_from_absorbance_csv(str(candidate))
                if peak is not None:
                    found_peak = peak
                    break
            if found_peak is not None:
                break
        if found_peak is not None:
            rows[sample_position] = found_peak

    return rows


def _extract_uvvis_measure_kinetics_record_fields(payload, conn, run_name: str) -> dict:
    data = _to_plain_data(payload)
    record_fields = {}
    field_name_pattern = re.compile(r"(?:(sample2|sample4)_)?t(\d+)_absorbance")

    def _visit(node):
        nonlocal record_fields
        if isinstance(node, dict):
            candidate = {
                key: value
                for key, value in node.items()
                if field_name_pattern.fullmatch(str(key or ""))
            }
            if candidate and len(candidate) >= len(record_fields):
                record_fields = candidate
            nested = node.get("record_fields")
            if isinstance(nested, dict):
                nested_candidate = {
                    key: value
                    for key, value in nested.items()
                    if field_name_pattern.fullmatch(str(key or ""))
                }
                if nested_candidate and len(nested_candidate) >= len(record_fields):
                    record_fields = nested_candidate
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(data)
    if len(record_fields) >= 35:
        return record_fields

    absorbance_paths = []
    for text in _collect_payload_strings(data):
        lower_text = text.lower()
        if "absorbance.csv" not in lower_text:
            continue
        if run_name and run_name.lower() not in lower_text and "uvvis_measure_kinetic" not in lower_text:
            continue
        if not run_name and not any(token in lower_text for token in ("sample2", "sample4", "uvvis_measure_kinetic")):
            continue
        absorbance_paths.append(text)

    if not absorbance_paths:
        kinetics_dir_names = ("uvvis_measure_kinetic", "uvvis_measure_kinetics")
        for device_dir in _resolve_uvvis_runtime_device_dirs(conn):
            if run_name:
                for kinetics_dir_name in kinetics_dir_names:
                    kinetics_dir = device_dir / kinetics_dir_name / run_name
                    if not kinetics_dir.exists():
                        continue
                    for candidate in sorted(
                        kinetics_dir.glob("*absorbance*.csv"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    ):
                        absorbance_paths.append(str(candidate))
                        break
            else:
                for kinetics_dir_name in kinetics_dir_names:
                    kinetics_dir = device_dir / kinetics_dir_name
                    if not kinetics_dir.exists():
                        continue
                    for candidate in sorted(
                        kinetics_dir.glob("*/*absorbance*.csv"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    ):
                        absorbance_paths.append(str(candidate))
                    if absorbance_paths:
                        break

    for path_text in absorbance_paths:
        path = Path(str(path_text or "").strip())
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        except Exception:
            continue
        if not rows:
            continue

        lower_name = path.name.lower()
        prefix = ""
        if "sample2" in lower_name:
            prefix = "sample2_"
        elif "sample4" in lower_name:
            prefix = "sample4_"
        elif run_name:
            prefix = ""

        extracted = {}
        for idx, row in enumerate(rows):
            time_index = _extract_int_value(row.get("time_index"))
            if time_index is None:
                time_index = idx
            absorbance = _extract_float_value(row.get("absorbance"))
            if absorbance is None:
                continue
            row_prefix = prefix
            if not row_prefix:
                sample_name = str(row.get("sample_name", "") or "").strip().lower()
                if sample_name in {"sample2", "sample4"}:
                    row_prefix = f"{sample_name}_"
            field_name = f"{row_prefix}t{time_index}_absorbance" if row_prefix else f"t{time_index}_absorbance"
            extracted[field_name] = absorbance
        if len(extracted) > len(record_fields):
            record_fields = extracted
        if len(record_fields) >= 35:
            break

    return record_fields


def _resolve_uvvis_primary_device_dir(conn) -> Path:
    device_dirs = _resolve_uvvis_runtime_device_dirs(conn)
    if device_dirs:
        target_dir = device_dirs[0]
    else:
        target_dir = (Path("data").resolve() / _normalize_uvvis_device_id(conn)).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _write_uvvis_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return True
    except Exception:
        return False


def _write_uvvis_json(path: Path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def _format_uvvis_svg_number(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _write_uvvis_line_plot_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
) -> bool:
    points = []
    for item in series:
        for point in item.get("points", []):
            x = _extract_float_value(point.get("x"))
            y = _extract_float_value(point.get("y"))
            if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                continue
            points.append((x, y))
    if not points:
        return False

    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 0.1 if y_min else 1.0
        y_max += 0.1 if y_max else 1.0

    width = 960
    height = 560
    left = 90
    right = 24
    top = 48
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_span = x_max - x_min
    y_span = y_max - y_min

    def _sx(value: float) -> float:
        return left + ((value - x_min) / x_span) * plot_width

    def _sy(value: float) -> float:
        return top + plot_height - ((value - y_min) / y_span) * plot_height

    colors = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
    )
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
    ]

    for index in range(6):
        y_value = y_min + (y_span * index / 5.0)
        y_pos = _sy(y_value)
        svg_parts.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_width}" y2="{y_pos:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{left - 12}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{_format_uvvis_svg_number(y_value)}</text>'
        )

    for index in range(6):
        x_value = x_min + (x_span * index / 5.0)
        x_pos = _sx(x_value)
        svg_parts.append(
            f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_height}" stroke="#f1f5f9" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x_pos:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="11" font-family="Arial">{_format_uvvis_svg_number(x_value)}</text>'
        )

    svg_parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="13" font-family="Arial">{x_label}</text>'
    )
    svg_parts.append(
        f'<text x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="13" font-family="Arial" transform="rotate(-90 22 {top + plot_height / 2:.1f})">{y_label}</text>'
    )

    legend_x = left + 8
    legend_y = 34
    for index, item in enumerate(series):
        clean_points = []
        for point in item.get("points", []):
            x = _extract_float_value(point.get("x"))
            y = _extract_float_value(point.get("y"))
            if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                continue
            clean_points.append((x, y))
        if not clean_points:
            continue
        color = item.get("color") or colors[index % len(colors)]
        polyline = " ".join(f"{_sx(x):.2f},{_sy(y):.2f}" for x, y in clean_points)
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{polyline}"/>'
        )
        legend_offset = index * 120
        svg_parts.append(
            f'<line x1="{legend_x + legend_offset}" y1="{legend_y}" x2="{legend_x + legend_offset + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + legend_offset + 24}" y="{legend_y + 4}" font-size="12" font-family="Arial">{item.get("label", f"Series {index + 1}")}</text>'
        )

    svg_parts.append("</svg>")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(svg_parts), encoding="utf-8")
        return True
    except Exception:
        return False


def _read_uvvis_absorbance_curve_csv(path_text: str) -> list[dict]:
    path = Path(str(path_text or "").strip())
    if not path.exists():
        return []

    points = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                wavelength_nm = _extract_float_value(
                    row.get("wavelength_nm") or row.get("wavelength")
                )
                absorbance = _extract_float_value(row.get("absorbance"))
                if wavelength_nm is None or absorbance is None:
                    continue
                if not math.isfinite(wavelength_nm) or not math.isfinite(absorbance):
                    continue
                points.append(
                    {
                        "wavelength_nm": wavelength_nm,
                        "corrected_absorbance": absorbance,
                    }
                )
    except Exception:
        return []
    points.sort(key=lambda item: item["wavelength_nm"])
    return points


def _extract_uvvis_measure_spectra_curve_sources(payload, conn) -> dict[int, str]:
    sources: dict[int, str] = {}
    data = _to_plain_data(payload)

    def _visit(node):
        if isinstance(node, dict):
            sample_position = _extract_int_value(
                node.get("sample_position")
                or node.get("position")
                or node.get("sample_index")
            )
            absorbance_path = str(
                node.get("absorbance_output_csv")
                or node.get("absorbance_csv")
                or node.get("output_csv")
                or node.get("csv_path")
                or ""
            ).strip()
            if sample_position is not None and absorbance_path and sample_position not in sources:
                sources[sample_position] = absorbance_path
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(data)

    for text in _collect_payload_strings(data):
        lower_text = text.lower()
        if "absorbance.csv" not in lower_text:
            continue
        sample_position = _extract_uvvis_sample_position_from_path(text)
        if sample_position is not None and sample_position not in sources:
            sources[sample_position] = text

    for sample_position in _UVVIS_SAMPLE_POSITIONS:
        if sample_position in sources:
            continue
        for device_dir in _resolve_uvvis_runtime_device_dirs(conn):
            candidate_paths = [
                device_dir / f"sample{sample_position}_latest_absorbance.csv",
                device_dir / f"sample_run_sample{sample_position}_latest_absorbance.csv",
                device_dir / "uvvis_measure_spectra" / f"sample{sample_position}_latest_absorbance.csv",
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    sources[sample_position] = str(candidate.resolve())
                    break
            if sample_position in sources:
                break

    return sources


def _persist_uvvis_measure_spectra_artifacts(conn, payload, rows: dict[int, dict]) -> dict:
    artifact_dir = _resolve_uvvis_primary_device_dir(conn) / "uvvis_measure_spectra"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    curve_sources = _extract_uvvis_measure_spectra_curve_sources(payload, conn)
    sample_curves = {}

    for sample_position in _UVVIS_SAMPLE_POSITIONS:
        curve_path = str(curve_sources.get(sample_position, "") or "").strip()
        points = _read_uvvis_absorbance_curve_csv(curve_path)
        if not points:
            continue
        sample_curves[sample_position] = {
            "source_csv": curve_path,
            "point_count": len(points),
            "points": points,
        }

    summary_csv_path = artifact_dir / "uvvis_spectra_latest_summary.csv"
    summary_rows = []
    for sample_position in _UVVIS_SAMPLE_POSITIONS:
        row = rows.get(sample_position, {})
        summary_rows.append(
            {
                "sample_position": sample_position,
                "lambda_max_nm": row.get("lambda_max_nm", ""),
                "max_corrected_absorbance": row.get("max_absorbance", ""),
                "source_absorbance_csv": str(curve_sources.get(sample_position, "") or ""),
                "point_count": sample_curves.get(sample_position, {}).get("point_count", 0),
            }
        )
    summary_ok = _write_uvvis_csv_rows(
        summary_csv_path,
        [
            "sample_position",
            "lambda_max_nm",
            "max_corrected_absorbance",
            "source_absorbance_csv",
            "point_count",
        ],
        summary_rows,
    )

    combined_csv_path = artifact_dir / "uvvis_spectra_latest_combined.csv"
    plot_svg_path = artifact_dir / "uvvis_spectra_latest_plot.svg"
    combined_ok = False
    plot_ok = False
    grid_complete = False
    if len(sample_curves) == len(_UVVIS_SAMPLE_POSITIONS):
        per_sample_maps = {
            sample_position: {
                int(round(point["wavelength_nm"])): point["corrected_absorbance"]
                for point in curve["points"]
            }
            for sample_position, curve in sample_curves.items()
        }
        grid_complete = all(
            all(wavelength_nm in per_sample_maps.get(sample_position, {}) for wavelength_nm in _UVVIS_SPECTRA_WAVELENGTH_GRID)
            for sample_position in _UVVIS_SAMPLE_POSITIONS
        )
        combined_rows = []
        for wavelength_nm in _UVVIS_SPECTRA_WAVELENGTH_GRID:
            row = {"wavelength_nm": wavelength_nm}
            for sample_position in _UVVIS_SAMPLE_POSITIONS:
                row[f"sample_{sample_position}_corrected_absorbance"] = per_sample_maps.get(
                    sample_position, {}
                ).get(wavelength_nm, "")
            combined_rows.append(row)
        combined_ok = _write_uvvis_csv_rows(
            combined_csv_path,
            ["wavelength_nm"]
            + [
                f"sample_{sample_position}_corrected_absorbance"
                for sample_position in _UVVIS_SAMPLE_POSITIONS
            ],
            combined_rows,
        )
        plot_ok = _write_uvvis_line_plot_svg(
            plot_svg_path,
            title="UV-Vis Spectra (400-700 nm, corrected)",
            x_label="Wavelength (nm)",
            y_label="Corrected Absorbance",
            series=[
                {
                    "label": f"Sample {sample_position}",
                    "points": [
                        {
                            "x": point["wavelength_nm"],
                            "y": point["corrected_absorbance"],
                        }
                        for point in sample_curves[sample_position]["points"]
                    ],
                }
                for sample_position in _UVVIS_SAMPLE_POSITIONS
                if sample_position in sample_curves
            ],
        )

    manifest = {
        "tool_name": "uvvis_measure_spectra",
        "generated_at_epoch": int(time.time()),
        "device_dir": str(_resolve_uvvis_primary_device_dir(conn)),
        "value_semantics": "corrected absorbance after dark-current and blank/reference subtraction",
        "expected_wavelength_grid_nm": list(_UVVIS_SPECTRA_WAVELENGTH_GRID),
        "summary_csv": str(summary_csv_path.resolve()) if summary_ok else "",
        "combined_absorbance_csv": str(combined_csv_path.resolve()) if combined_ok else "",
        "plot_svg": str(plot_svg_path.resolve()) if plot_ok else "",
        "all_expected_outputs_exist": bool(
            summary_ok
            and combined_ok
            and plot_ok
            and len(sample_curves) == len(_UVVIS_SAMPLE_POSITIONS)
            and grid_complete
        ),
        "sample_curves": {
            str(sample_position): {
                "source_csv": curve.get("source_csv", ""),
                "point_count": curve.get("point_count", 0),
                "lambda_max_nm": rows.get(sample_position, {}).get("lambda_max_nm"),
                "max_corrected_absorbance": rows.get(sample_position, {}).get("max_absorbance"),
            }
            for sample_position, curve in sample_curves.items()
        },
    }
    manifest_path = artifact_dir / "uvvis_spectra_latest_manifest.json"
    manifest_ok = _write_uvvis_json(manifest_path, manifest)
    artifacts = dict(manifest)
    artifacts["manifest_json"] = str(manifest_path.resolve()) if manifest_ok else ""
    return artifacts


def _extract_uvvis_kinetics_series_rows(payload, conn, run_name: str, record_fields: dict) -> list[dict]:
    absorbance_paths = []
    for text in _collect_payload_strings(payload):
        lower_text = text.lower()
        if "absorbance.csv" in lower_text and run_name.lower() in lower_text:
            absorbance_paths.append(text)

    if not absorbance_paths:
        for device_dir in _resolve_uvvis_runtime_device_dirs(conn):
            kinetics_dir = device_dir / "uvvis_measure_kinetics" / run_name
            if not kinetics_dir.exists():
                continue
            for candidate in sorted(
                kinetics_dir.glob("*absorbance*.csv"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                absorbance_paths.append(str(candidate))
                break
            if absorbance_paths:
                break

    for path_text in absorbance_paths:
        path = Path(str(path_text or "").strip())
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = []
                for index, row in enumerate(reader):
                    time_index = _extract_int_value(
                        row.get("time_index")
                        or row.get("time_min")
                        or row.get("minute")
                    )
                    if time_index is None:
                        time_index = index
                    absorbance = _extract_float_value(row.get("absorbance"))
                    if absorbance is None or not math.isfinite(absorbance):
                        continue
                    rows.append(
                        {
                            "time_index": time_index,
                            "time_min": time_index,
                            "corrected_absorbance": absorbance,
                        }
                    )
        except Exception:
            continue
        if rows:
            rows.sort(key=lambda item: item["time_index"])
            return rows

    rows = []
    for key, value in sorted(record_fields.items()):
        match = re.fullmatch(r"t(\d+)_absorbance", str(key or ""))
        if not match:
            continue
        absorbance = _extract_float_value(value)
        if absorbance is None or not math.isfinite(absorbance):
            continue
        time_index = int(match.group(1))
        rows.append(
            {
                "time_index": time_index,
                "time_min": time_index,
                "corrected_absorbance": absorbance,
            }
        )
    rows.sort(key=lambda item: item["time_index"])
    return rows


def _persist_uvvis_measure_kinetics_artifacts(
    conn,
    payload,
    *,
    run_name: str,
    sample_position: int,
    record_fields: dict,
) -> dict:
    artifact_dir = _resolve_uvvis_primary_device_dir(conn) / "uvvis_measure_kinetics" / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = _extract_uvvis_kinetics_series_rows(payload, conn, run_name, record_fields)

    timeseries_csv_path = artifact_dir / f"{run_name}_kinetics_latest_timeseries.csv"
    timeseries_ok = _write_uvvis_csv_rows(
        timeseries_csv_path,
        ["time_index", "time_min", "corrected_absorbance"],
        rows,
    )
    time_grid_complete = {row.get("time_index") for row in rows} >= set(_UVVIS_KINETICS_TIME_GRID)
    plot_svg_path = artifact_dir / f"{run_name}_kinetics_latest_plot.svg"
    plot_ok = _write_uvvis_line_plot_svg(
        plot_svg_path,
        title=f"{run_name} kinetics at 400 nm (corrected)",
        x_label="Time (min)",
        y_label="Corrected Absorbance",
        series=[
            {
                "label": f"Sample position {sample_position}",
                "points": [
                    {"x": row["time_min"], "y": row["corrected_absorbance"]}
                    for row in rows
                ],
            }
        ],
    )
    manifest = {
        "tool_name": "uvvis_measure_kinetics",
        "generated_at_epoch": int(time.time()),
        "device_dir": str(_resolve_uvvis_primary_device_dir(conn)),
        "run_name": run_name,
        "sample_position": sample_position,
        "value_semantics": "corrected absorbance after dark-current and reference subtraction",
        "expected_time_grid_min": list(_UVVIS_KINETICS_TIME_GRID),
        "timeseries_csv": str(timeseries_csv_path.resolve()) if timeseries_ok else "",
        "plot_svg": str(plot_svg_path.resolve()) if plot_ok else "",
        "time_point_count": len(rows),
        "all_expected_outputs_exist": bool(
            timeseries_ok and plot_ok and len(rows) >= 35 and time_grid_complete
        ),
    }
    manifest_path = artifact_dir / f"{run_name}_kinetics_latest_manifest.json"
    manifest_ok = _write_uvvis_json(manifest_path, manifest)
    artifacts = dict(manifest)
    artifacts["manifest_json"] = str(manifest_path.resolve()) if manifest_ok else ""
    return artifacts


def _filter_fields_for_schema(fields: dict, schema_by_name: dict) -> dict:
    if not fields:
        return {}
    if not schema_by_name:
        return dict(fields)
    allowed = set(schema_by_name)
    return {key: value for key, value in fields.items() if key in allowed}


async def _complete_experiment_step_with_fields(
    conn,
    *,
    fields: dict,
    auto_advance: bool,
    fallback_reply: str = "",
    group_number: int | None = None,
) -> tuple[bool, str]:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False, fallback_reply or ""

    normalized_group_number = None
    try:
        if group_number is not None:
            normalized_group_number = int(group_number)
            if normalized_group_number < 1:
                normalized_group_number = None
    except (TypeError, ValueError):
        normalized_group_number = None

    if normalized_group_number is not None:
        setattr(conn, "experiment_current_group_number", normalized_group_number)
        current_step_id_hint = _get_current_experiment_step_id(conn)
        if current_step_id_hint:
            try:
                await _call_experiment_graph_tool_fast(
                    conn,
                    "redirect_to_step",
                    {
                        "session_id": session_id,
                        "step_id": current_step_id_hint,
                        "force": True,
                        "group_number": normalized_group_number,
                    },
                    priority="foreground",
                )
            except Exception as exc:
                conn.logger.bind(tag=TAG).warning(
                    "failed to sync explicit experiment group before writeback: "
                    f"session_id={session_id}, step_id={current_step_id_hint}, "
                    f"group_number={normalized_group_number}, error={exc}"
                )

    step_payload, progress_payload, schema_payload = await asyncio.gather(
        _call_experiment_graph_tool_fast(
            conn,
            "get_step",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_current_progress",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_schema",
            {"session_id": session_id},
            priority="foreground",
        ),
    )

    conn.experiment_current_step = step_payload
    conn.experiment_progress_summary = progress_payload
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(
            step_payload,
            progress_payload,
        )
        if current_step_id:
            conn.experiment_current_step_id = current_step_id

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        start_payload = await _call_experiment_graph_tool_fast(
            conn,
            "start_trial",
            {
                "session_id": session_id,
                **(
                    {"group_number": normalized_group_number}
                    if normalized_group_number is not None
                    else {}
                ),
            },
            priority="foreground",
        )
        current_progress = _extract_experiment_current_progress(start_payload)
    elif normalized_group_number is not None:
        current_progress_group = _extract_current_progress_group_number(progress_payload)
        if current_progress_group != normalized_group_number:
            start_payload = await _call_experiment_graph_tool_fast(
                conn,
                "start_trial",
                {
                    "session_id": session_id,
                    "force": True,
                    "group_number": normalized_group_number,
                },
                priority="foreground",
            )
            current_progress = _extract_experiment_current_progress(start_payload)

    write_fields = _filter_fields_for_schema(fields, schema_by_name)
    if write_fields:
        add_fields_payload = await _call_experiment_graph_tool_fast(
            conn,
            "add_fields",
            {"session_id": session_id, "data": write_fields},
            priority="foreground",
        )
        updated_progress = _extract_experiment_current_progress(add_fields_payload)
        if isinstance(updated_progress, dict):
            current_progress = updated_progress

    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    if missing_fields:
        return False, _compose_missing_field_reply(missing_fields, schema_by_name)

    finish_payload = await _call_experiment_graph_tool_fast(
        conn,
        "finish_trial",
        {"session_id": session_id, "validate": True},
        priority="foreground",
    )
    if not bool(_experiment_result_body(finish_payload).get("ok")):
        message = _extract_experiment_result_message(finish_payload)
        if message:
            return False, message
        return False, fallback_reply or ""

    if not auto_advance:
        return True, fallback_reply or ""

    can_proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "can_proceed",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(can_proceed_payload).get("ok")):
        message = _extract_experiment_result_message(can_proceed_payload)
        if message:
            return False, message
        return False, fallback_reply or ""

    proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "proceed_to_next_step",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(proceed_payload).get("ok")):
        message = _extract_experiment_result_message(proceed_payload)
        if message:
            return False, message
        return False, fallback_reply or ""

    next_meta = await _refresh_experiment_step_cache(conn, session_id)
    reply = _compose_experiment_step_reply(next_meta, mode="next")
    if reply:
        return True, reply
    return True, fallback_reply or ""


async def _advance_finished_experiment_step(
    conn,
    *,
    fallback_reply: str = "",
) -> tuple[bool, str]:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False, fallback_reply or ""

    can_proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "can_proceed",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(can_proceed_payload).get("ok")):
        message = _extract_experiment_result_message(can_proceed_payload)
        if message:
            return False, message
        return False, fallback_reply or ""

    proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "proceed_to_next_step",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(proceed_payload).get("ok")):
        message = _extract_experiment_result_message(proceed_payload)
        if message:
            return False, message
        return False, fallback_reply or ""

    next_meta = await _refresh_experiment_step_cache(conn, session_id)
    reply = _compose_experiment_step_reply(next_meta, mode="next")
    if reply:
        return True, reply
    return True, fallback_reply or ""


async def _release_uvvis_session_for_analysis(conn) -> None:
    step_id = _get_current_experiment_step_id(conn)
    if step_id != _UVVIS_ANALYSIS_STEP_ID:
        return

    session_key = str(getattr(conn, "_uvvis_session_key", "") or "").strip()
    if not session_key:
        return

    released_step_id = str(
        getattr(conn, "_uvvis_analysis_release_done_step_id", "") or ""
    ).strip()
    if released_step_id == step_id:
        return

    manager = _get_server_mcp_manager(conn)
    if manager is None or not manager.is_mcp_tool("uvvis_session"):
        return

    try:
        await _execute_uvvis_tool_payload(
            conn,
            "uvvis_session",
            {"action": "release", "session_key": session_key},
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(f"uvvis_session release failed: {exc}")
        return

    setattr(conn, "_uvvis_session_key", "")
    setattr(conn, "_uvvis_analysis_release_done_step_id", step_id)


async def _run_uvvis_shared_dark_air_scan(conn) -> tuple[bool, str]:
    if _uvvis_shared_dark_air_cache_ready(conn):
        completed, reply = await _complete_experiment_step_with_fields(
            conn,
            fields={
                "empty_positions_confirmed": True,
                "shared_dark_current_ready": True,
                "shared_air_baseline_ready": True,
                "observations": "澶嶇敤宸插瓨鍦ㄧ殑鍏变韩鏆楃數娴佸拰绌烘皵鑳介噺鏍℃缂撳瓨銆?,
            },
            auto_advance=True,
            fallback_reply="宸插鐢ㄥ叡浜殫鐢垫祦鍜岀┖姘旇兘閲忔牎姝ｏ紝鐩存帴杩涘叆涓嬩竴姝ャ€?,
        )
        if completed:
            _clear_uvvis_direct_state(conn)
        return True, reply

    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        return False, busy_reply

    prepare_payload = await _execute_uvvis_tool_payload(
        conn,
        "uvvis_prepare_dark_current",
        {
            "session_key": session_key,
            "output_dir": str(_resolve_uvvis_group_output_dir(conn, None)),
            "shared_output_dir": str(_resolve_uvvis_native_output_root(conn)),
        },
    )
    if _payload_looks_busy_or_inaccessible(prepare_payload):
        return True, _UVVIS_BUSY_REPLY

    completed, reply = await _complete_experiment_step_with_fields(
        conn,
        fields={
            "empty_positions_confirmed": True,
            "shared_dark_current_ready": True,
            "shared_air_baseline_ready": True,
            "observations": "鍏变韩鏆楃數娴佸拰绌烘皵鑳介噺鏍℃宸插畬鎴愩€?,
        },
        auto_advance=True,
        fallback_reply="鍏变韩鏆楃數娴佸拰绌烘皵鑳介噺鏍℃宸茬粡瀹屾垚銆?,
    )
    if completed:
        _clear_uvvis_direct_state(conn)
    return True, reply


async def _handle_uvvis_shared_dark_air_prep(
    conn, original_text: str, filtered_text: str
) -> bool:
    control_action = _classify_short_experiment_control(conn, filtered_text)
    if _uvvis_shared_dark_air_cache_ready(conn):
        if control_action in {"advance", "guide", "repeat"} and _is_current_graph_step_completed(
            getattr(conn, "experiment_progress_summary", None)
        ):
            await _start_direct_intent_turn(conn, original_text)
            advanced, reply = await _advance_finished_experiment_step(
                conn,
                fallback_reply="宸插鐢ㄥ叡浜殫鐢垫祦鍜岀┖姘旇兘閲忔牎姝ｏ紝鐩存帴杩涘叆涓嬩竴姝ャ€?,
            )
            if advanced:
                _clear_uvvis_direct_state(conn)
                if reply:
                    speak_txt(conn, reply)
                return True
            if reply:
                speak_txt(conn, reply)
                return True

        state = _get_uvvis_direct_state(conn, _UVVIS_SHARED_DARK_AIR_STEP_ID)
        phase = str(state.get("phase", "") or "").strip()
        if phase in {"await_empty_positions", "await_scan_start"} or _contains_any(
            _normalize_text_for_match(filtered_text),
            (
                "缁х画鍒氭墠",
                "缁х画瀹為獙",
                "寮€濮嬫壂鎻?,
                "鍙互寮€濮?,
                "閮界┖浜?,
            ),
        ):
            await _start_direct_intent_turn(conn, original_text)
            handled, reply = await _run_uvvis_shared_dark_air_scan(conn)
            if handled:
                if reply:
                    speak_txt(conn, reply)
                return True
            return False

    if _is_explicit_experiment_start_request(filtered_text):
        try:
            await _reset_experiment_fresh_start_context(conn)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"uvvis shared prep explicit start reset failed: {exc}"
            )
        experiment_title = await _load_experiment_overview_title(conn)
        reply = _prepare_fastpath_spoken_reply(
            _compose_experiment_start_reply(experiment_title, "")
        )
        if not reply:
            return False
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    if _assistant_waiting_for_step_start(conn) and _is_explicit_ready_to_start_reply(
        filtered_text
    ):
        await _start_direct_intent_turn(conn, original_text)
        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_DARK_AIR_STEP_ID,
            phase="await_empty_positions",
        )
        speak_txt(
            conn,
            "鍏堟鏌?鍒?鍙锋牱鍝佷綅閮戒负绌猴紝鍙傛瘮浣嶄篃涓嶈鏀句换浣曟恫浣撱€傞兘绌轰簡灏卞憡璇夋垜銆傚彲浠ュ紑濮嬫椂鐩存帴璇粹€滃紑濮嬫壂鎻忊€濄€?,
        )
        return True

    state = _get_uvvis_direct_state(conn, _UVVIS_SHARED_DARK_AIR_STEP_ID)
    phase = str(state.get("phase", "") or "").strip()

    if phase == "await_empty_positions":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘纭鏍峰搧浣嶅拰鍙傛瘮浣嶉兘鐣欑┖鍚庡啀鍛婅瘔鎴戙€?)
            return True
        if (
            _looks_like_uvvis_empty_then_start_reply(filtered_text)
            or _looks_like_uvvis_start_scan_reply(filtered_text)
        ):
            await _start_direct_intent_turn(conn, original_text)
            handled, reply = await _run_uvvis_shared_dark_air_scan(conn)
            if handled:
                if reply:
                    speak_txt(conn, reply)
                return True
            return False
        if not _looks_like_uvvis_empty_positions_reply(filtered_text):
            if _looks_like_uvvis_ready_reply(filtered_text):
                await _start_direct_intent_turn(conn, original_text)
                speak_txt(conn, "鍏堢‘璁?鍒?鍙锋牱鍝佷綅閮戒负绌猴紝鍙傛瘮浣嶄篃涓嶈鏀句换浣曟恫浣撱€傞兘绌轰簡灏卞憡璇夋垜銆傚彲浠ュ紑濮嬫椂鐩存帴璇粹€滃紑濮嬫壂鎻忊€濄€?)
                return True
            return False
        await _start_direct_intent_turn(conn, original_text)
        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_DARK_AIR_STEP_ID,
            phase="await_scan_start",
        )
        speak_txt(conn, "鍙互寮€濮嬫椂鍛婅瘔鎴戔€滃紑濮嬫壂鎻忊€濄€?)
        return True

    if phase == "await_scan_start":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鍙互寮€濮嬫壂鎻忔椂鍐嶅憡璇夋垜銆?)
            return True
        if not (
            _looks_like_uvvis_empty_then_start_reply(filtered_text)
            or _looks_like_uvvis_ready_reply(filtered_text)
        ):
            if _looks_like_uvvis_empty_positions_reply(filtered_text):
                await _start_direct_intent_turn(conn, original_text)
                speak_txt(conn, "鍙互寮€濮嬫椂鍛婅瘔鎴戔€滃紑濮嬫壂鎻忊€濄€?)
                return True
            return False

        await _start_direct_intent_turn(conn, original_text)
        handled, reply = await _run_uvvis_shared_dark_air_scan(conn)
        if not handled:
            return False
        if reply:
            speak_txt(conn, reply)
        return True

    norm = _normalize_text_for_match(filtered_text)
    if not (
        (_assistant_recently_prompted_uvvis_action(conn) and control_action in {"guide", "advance", "repeat"})
        or _contains_any(
            norm,
            (
                "鏆楃數娴佹牎姝?,
                "鏆楃數娴?,
                "绌烘皵鍩虹嚎",
                "绌烘皵鑳介噺",
                "鍏变韩鏆楃數娴?,
                "寮€濮媢vvis",
                "寮€濮嬬传澶栧彲瑙?,
                "寮€濮嬫祴閲?,
                "寮€濮嬫壂鎻?,
                "鍙互寮€濮嬫壂鎻?,
                "绌烘灦鎵弿",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)
    _set_uvvis_direct_state(
        conn,
        step_id=_UVVIS_SHARED_DARK_AIR_STEP_ID,
        phase="await_empty_positions",
    )
    speak_txt(
        conn,
        "鍏堟鏌?鍒?鍙锋牱鍝佷綅閮戒负绌猴紝鍙傛瘮浣嶄篃涓嶈鏀句换浣曟恫浣撱€傞兘绌轰簡灏卞憡璇夋垜銆傚彲浠ュ紑濮嬫椂鐩存帴璇粹€滃紑濮嬫壂鎻忊€濄€?,
    )
    return True


async def _legacy_handle_uvvis_shared_blank_prep_pre_split(
    conn, original_text: str, filtered_text: str, state: dict
) -> bool:
    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    if state.get("phase") == "await_pure_water_blank":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鎶婄函姘存瘮鑹茬毧鏀惧ソ鍐嶅憡璇夋垜銆?)
            return True

        if not _looks_like_uvvis_ready_reply(filtered_text):
            return False

        await _start_direct_intent_turn(conn, original_text)
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_measure_spectra",
            {
                "session_key": session_key,
                "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
                "ready_for_samples": True,
            },
        )
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True
        if _payload_mentions_missing_blank(payload):
            _set_uvvis_direct_state(
                conn,
                step_id=_UVVIS_SHARED_BLANK_STEP_ID,
                phase="await_pure_water_blank",
                session_key=session_key,
            )
            speak_txt(conn, "杩欎竴姝ヨ繕缂虹函姘寸┖鐧斤紝璇峰厛鎶?1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁鍏ョ函姘存瘮鑹茬毧锛屾斁濂藉悗鍛婅瘔鎴戝彲浠ュ紑濮嬫壂鎻忋€?)
            return True

        fields = {
            "shared_dark_current_ready": True,
            "shared_air_baseline_ready": True,
            "pure_water_blank_ready": True,
            "reference_cuvette_ready": True,
            "observations": "鍏变韩鍓嶇疆鏍℃鍜岀函姘寸┖鐧藉凡鍑嗗瀹屾垚",
        }
        auto_advanced, reply = await _complete_experiment_step_with_fields(
            conn,
            fields=fields,
            auto_advance=True,
            fallback_reply="鍏变韩鍓嶇疆鏍℃鍜岀函姘寸┖鐧介兘鍑嗗濂戒簡銆?,
        )
        if auto_advanced and reply:
            speak_txt(conn, reply)
        elif reply:
            speak_txt(conn, reply)
        return True

    if not (
        _is_affirmative_short_reply_fixed(filtered_text)
        or _classify_short_experiment_control(conn, filtered_text) in {"guide", "advance", "repeat"}
        or _contains_any(
            _normalize_text_for_match(filtered_text),
            (
                "鏆楃數娴?,
                "绌烘皵鍩虹嚎",
                "绌烘皵鑳介噺",
                "绾按绌虹櫧",
                "绌虹櫧鏍℃",
                "寮€濮媢vvis",
                "寮€濮嬬传澶栧彲瑙?,
                "寮€濮嬫祴閲?,
                "寮€濮嬫壂鎻?,
                "鍙互寮€濮嬫壂鎻?,
                "绌烘灦鎵弿",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)
    speak_txt(conn, "鍏堜笉瑕佹斁浠讳綍娑蹭綋锛屾垜鍏堣繘琛屽叡浜墠缃牎姝ｃ€?)
    payload = await _execute_uvvis_tool_payload(
        conn,
        "uvvis_measure_spectra",
        {
            "session_key": session_key,
            "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
            "ready_for_samples": False,
        },
    )
    if _payload_looks_busy_or_inaccessible(payload):
        speak_txt(conn, _UVVIS_BUSY_REPLY)
        return True
    if _payload_mentions_missing_blank(payload):
        speak_txt(conn, "杩欎竴姝ヨ繕缂虹函姘寸┖鐧斤紝璇峰厛鎶?1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁鍏ョ函姘存瘮鑹茬毧锛屾斁濂藉悗鍛婅瘔鎴戝彲浠ュ紑濮嬫壂鎻忋€?)
        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_BLANK_STEP_ID,
            phase="await_pure_water_blank",
            session_key=session_key,
        )
        return True

    fields = {
        "shared_dark_current_ready": True,
        "shared_air_baseline_ready": True,
        "pure_water_blank_ready": True,
        "reference_cuvette_ready": True,
        "observations": "鍏变韩鍓嶇疆鏍℃鍜岀函姘寸┖鐧藉凡瀹屾垚鎴栧彲澶嶇敤",
    }
    auto_advanced, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=fields,
        auto_advance=True,
        fallback_reply="鍏变韩鍓嶇疆鏍℃鍜岀函姘寸┖鐧介兘鍑嗗濂戒簡銆?,
    )
    if auto_advanced and reply:
        speak_txt(conn, reply)
    elif reply:
        speak_txt(conn, reply)
    return True


async def _complete_uvvis_shared_blank_step(
    conn,
    *,
    observations: str,
    fallback_reply: str,
) -> tuple[bool, str]:
    return await _complete_experiment_step_with_fields(
        conn,
        fields={
            "pure_water_blank_ready": True,
            "reference_cuvette_ready": True,
            "observations": observations,
        },
        auto_advance=True,
        fallback_reply=fallback_reply,
    )


async def _handle_uvvis_shared_blank_prep(
    conn, original_text: str, filtered_text: str, state: dict
) -> bool:
    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    if state.get("phase") == "await_pure_water_blank":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鎶婄函姘存瘮鑹茬毧鏀惧ソ鍐嶅憡璇夋垜銆?)
            return True

        if not _looks_like_uvvis_ready_reply(filtered_text):
            return False

        await _start_direct_intent_turn(conn, original_text)
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_measure_spectra",
            {
                "session_key": session_key,
                "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
                "ready_for_samples": True,
            },
        )
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True
        if _payload_mentions_missing_blank(payload):
            _set_uvvis_direct_state(
                conn,
                step_id=_UVVIS_SHARED_BLANK_STEP_ID,
                phase="await_pure_water_blank",
                session_key=session_key,
            )
            speak_txt(conn, "杩欎竴姝ヨ繕缂虹函姘寸┖鐧斤紝璇峰厛鎶?1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁鍏ョ函姘存瘮鑹茬毧锛屾斁濂藉悗鍛婅瘔鎴戝彲浠ュ紑濮嬫壂鎻忋€?)
            return True

        auto_advanced, reply = await _complete_experiment_step_with_fields(
            conn,
            fields={
                "pure_water_blank_ready": True,
                "reference_cuvette_ready": True,
                "observations": "褰撳墠鎵规绾按绌虹櫧宸茶褰曞畬鎴愶紝鍙傛瘮浣嶇函姘存瘮鑹茬毧鍙户缁敤浜庡悗缁祴閲忋€?,
            },
            auto_advance=True,
            fallback_reply="绾按绌虹櫧宸茬粡鍑嗗濂戒簡銆?,
        )
        if auto_advanced:
            _clear_uvvis_direct_state(conn)
        if reply:
            speak_txt(conn, reply)
        return True

    if not (
        _is_affirmative_short_reply_fixed(filtered_text)
        or _classify_short_experiment_control(conn, filtered_text) in {"guide", "advance", "repeat"}
        or _contains_any(
            _normalize_text_for_match(filtered_text),
            (
                "绾按绌虹櫧",
                "绾按姣旇壊鐨?,
                "绌虹櫧鏍℃",
                "寮€濮媢vvis",
                "寮€濮嬬传澶栧彲瑙?,
                "寮€濮嬫祴閲?,
                "寮€濮嬫壂鎻?,
                "鍙互寮€濮嬫壂鎻?,
                "绌虹櫧",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)

    if _uvvis_blank_baseline_exists(conn):
        auto_advanced, reply = await _complete_experiment_step_with_fields(
            conn,
            fields={
                "pure_water_blank_ready": True,
                "reference_cuvette_ready": True,
                "observations": "褰撳墠鎵规绾按绌虹櫧宸茬‘璁ゅ彲澶嶇敤锛屽悗缁祴閲忓皢淇濈暀鎴栭噸鏂版斁濂藉弬姣斾綅绾按姣旇壊鐨裤€?,
            },
            auto_advance=True,
            fallback_reply="褰撳墠鎵规绾按绌虹櫧鍙鐢紝鎺ヤ笅鏉ヨ鍏ユ牱鍝佹瘮鑹茬毧銆?,
        )
        _clear_uvvis_direct_state(conn)
        if reply:
            speak_txt(conn, reply)
        return True

    _set_uvvis_direct_state(
        conn,
        step_id=_UVVIS_SHARED_BLANK_STEP_ID,
        phase="await_pure_water_blank",
        session_key=session_key,
    )
    speak_txt(
        conn,
        "鏆楃數娴佹牎姝ｅ凡缁忓畬鎴愩€傝鍦?1-5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁鍏ョ函姘存瘮鑹茬毧锛屽叡 6 涓紝鏀惧ソ鍚庡憡璇夋垜鍙互寮€濮嬫壂鎻忋€?,
    )
    return True


async def _handle_uvvis_shared_blank_prep_v2(
    conn, original_text: str, filtered_text: str, state: dict
) -> bool:
    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    phase = str(state.get("phase", "") or "").strip()
    if (
        not phase
        and _assistant_recently_prompted_pure_water_blank(conn)
        and _looks_like_uvvis_ready_reply(filtered_text)
    ):
        phase = "await_pure_water_blank"

    if phase == "await_shared_prep_reset":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鎶婃牱鍝佷綅鍜屽弬姣斾綅閮芥竻绌哄悗鍐嶅憡璇夋垜銆?)
            return True

        if not _looks_like_uvvis_ready_reply(filtered_text):
            return False

        await _start_direct_intent_turn(conn, original_text)
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_measure_spectra",
            {
                "session_key": session_key,
                "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
                "ready_for_samples": False,
            },
        )
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True
        if _payload_mentions_missing_shared_prep(payload) or _payload_mentions_shared_prep_saturated(payload):
            _set_uvvis_direct_state(
                conn,
                step_id=_UVVIS_SHARED_BLANK_STEP_ID,
                phase="await_shared_prep_reset",
                session_key=session_key,
            )
            speak_txt(conn, "璇峰厛鎶?1 鍒?5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥竻绌猴紝鎴戝厛琛ュ仛鍏变韩鍓嶇疆鏍℃銆傛竻绌哄悗鍛婅瘔鎴戝彲浠ュ紑濮嬨€?)
            return True
        if _uvvis_shared_liquid_blank_exists(conn, payload) or _payload_indicates_liquid_blank_ready(payload):
            auto_advanced, reply = await _complete_uvvis_shared_blank_step(
                conn,
                observations="褰撳墠鎵规绾按绌虹櫧宸茬‘璁ゅ彲澶嶇敤锛屽悗缁祴閲忓皢淇濈暀鎴栭噸鏂版斁濂藉弬姣斾綅绾按姣旇壊鐨裤€?,
                fallback_reply="褰撳墠鎵规绾按绌虹櫧鍙鐢紝鎺ヤ笅鏉ヨ鍏ユ牱鍝佹瘮鑹茬毧銆?,
            )
            _clear_uvvis_direct_state(conn)
            if reply:
                speak_txt(conn, reply)
            return True

        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_BLANK_STEP_ID,
            phase="await_pure_water_blank",
            session_key=session_key,
        )
        speak_txt(
            conn,
            "鍏变韩鍓嶇疆鏍℃宸茬粡鍑嗗濂姐€傝鍦?1 鍒?5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁 1 鏀函姘存瘮鑹茬毧锛屽叡 6 鏀紝鏀惧ソ浜嗗憡璇夋垜鍙互寮€濮嬫壂鎻忋€?,
        )
        return True

    if phase == "await_pure_water_blank":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鎶婄函姘存瘮鑹茬毧鏀惧ソ鍚庡啀鍛婅瘔鎴戙€?)
            return True

        if not _looks_like_uvvis_ready_reply(filtered_text):
            return False

        await _start_direct_intent_turn(conn, original_text)
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_measure_spectra",
            {
                "session_key": session_key,
                "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
                "ready_for_samples": True,
            },
        )
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True
        if _payload_mentions_missing_shared_prep(payload) or _payload_mentions_shared_prep_saturated(payload):
            _set_uvvis_direct_state(
                conn,
                step_id=_UVVIS_SHARED_BLANK_STEP_ID,
                phase="await_shared_prep_reset",
                session_key=session_key,
            )
            speak_txt(conn, "鍏变韩鍓嶇疆鏍℃杩樻病鍑嗗濂斤紝璇峰厛鎶?1 鍒?5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥竻绌猴紝鎴戝厛琛ュ仛鍓嶇疆鏍℃銆傛竻绌哄悗鍛婅瘔鎴戝彲浠ュ紑濮嬨€?)
            return True
        if _payload_mentions_missing_blank(payload):
            _set_uvvis_direct_state(
                conn,
                step_id=_UVVIS_SHARED_BLANK_STEP_ID,
                phase="await_pure_water_blank",
                session_key=session_key,
            )
            speak_txt(conn, "杩欎竴姝ヨ繕缂虹函姘寸┖鐧斤紝璇峰厛鎶?1 鍒?5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮芥斁鍏ョ函姘存瘮鑹茬毧锛屾斁濂藉悗鍛婅瘔鎴戝彲浠ュ紑濮嬫壂鎻忋€?)
            return True

        auto_advanced, reply = await _complete_uvvis_shared_blank_step(
            conn,
            observations="褰撳墠鎵规绾按绌虹櫧宸茶褰曞畬鎴愶紝鍙傛瘮浣嶇函姘存瘮鑹茬毧鍙户缁敤浜庡悗缁祴閲忋€?,
            fallback_reply="绾按绌虹櫧宸茬粡鍑嗗濂戒簡銆?,
        )
        if auto_advanced:
            _clear_uvvis_direct_state(conn)
        if reply:
            speak_txt(conn, reply)
        return True

    if not (
        _is_affirmative_short_reply_fixed(filtered_text)
        or _classify_short_experiment_control(conn, filtered_text) in {"guide", "advance", "repeat"}
        or _contains_any(
            _normalize_text_for_match(filtered_text),
            (
                "绾按绌虹櫧",
                "绾按姣旇壊鐨?,
                "绌虹櫧鏍℃",
                "寮€濮媢vvis",
                "寮€濮嬬传澶栧彲瑙?,
                "寮€濮嬫祴閲?,
                "寮€濮嬫壂鎻?,
                "鍙互寮€濮嬫壂鎻?,
                "绌虹櫧",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)

    if _uvvis_shared_liquid_blank_exists(conn):
        auto_advanced, reply = await _complete_uvvis_shared_blank_step(
            conn,
            observations="褰撳墠鎵规绾按绌虹櫧宸茬‘璁ゅ彲澶嶇敤锛屽悗缁祴閲忓皢淇濈暀鎴栭噸鏂版斁濂藉弬姣斾綅绾按姣旇壊鐨裤€?,
            fallback_reply="褰撳墠鎵规绾按绌虹櫧鍙鐢紝鎺ヤ笅鏉ヨ鍏ユ牱鍝佹瘮鑹茬毧銆?,
        )
        _clear_uvvis_direct_state(conn)
        if reply:
            speak_txt(conn, reply)
        return True

    payload = await _execute_uvvis_tool_payload(
        conn,
        "uvvis_measure_spectra",
        {
            "session_key": session_key,
            "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
            "ready_for_samples": False,
        },
    )
    if _payload_looks_busy_or_inaccessible(payload):
        speak_txt(conn, _UVVIS_BUSY_REPLY)
        return True
    if _uvvis_shared_liquid_blank_exists(conn, payload) or _payload_indicates_liquid_blank_ready(payload):
        auto_advanced, reply = await _complete_uvvis_shared_blank_step(
            conn,
            observations="褰撳墠鎵规绾按绌虹櫧宸茬‘璁ゅ彲澶嶇敤锛屽悗缁祴閲忓皢淇濈暀鎴栭噸鏂版斁濂藉弬姣斾綅绾按姣旇壊鐨裤€?,
            fallback_reply="褰撳墠鎵规绾按绌虹櫧鍙鐢紝鎺ヤ笅鏉ヨ鍏ユ牱鍝佹瘮鑹茬毧銆?,
        )
        _clear_uvvis_direct_state(conn)
        if reply:
            speak_txt(conn, reply)
        return True

    if _payload_mentions_missing_shared_prep(payload) or _payload_mentions_shared_prep_saturated(payload):
        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_BLANK_STEP_ID,
            phase="await_shared_prep_reset",
            session_key=session_key,
        )
        speak_txt(conn, "璇峰厛淇濇寔 1 鍒?5 鍙锋牱鍝佷綅鍜屽弬姣斾綅閮戒负绌猴紝鎴戝厛琛ュ仛鍏变韩鍓嶇疆鏍℃銆傛竻绌哄悗鍛婅瘔鎴戝彲浠ュ紑濮嬨€?)
        return True

    _set_uvvis_direct_state(
        conn,
        step_id=_UVVIS_SHARED_BLANK_STEP_ID,
        phase="await_pure_water_blank",
        session_key=session_key,
    )
    speak_txt(
        conn,
        "鍏变韩鍓嶇疆鏍℃宸茬粡鍑嗗濂姐€傝鍦?1 鍒?5 鍙锋牱鍝佷綅鍜屽弬姣斾綅鍚勬斁 1 鏀函姘存瘮鑹茬毧锛屽叡 6 鏀紝鏀惧ソ浜嗗憡璇夋垜鍙互寮€濮嬫壂鎻忋€?,
    )
    return True


_handle_uvvis_shared_blank_prep = _handle_uvvis_shared_blank_prep_v2


async def _handle_uvvis_spectra_measurement(
    conn, original_text: str, filtered_text: str
) -> bool:
    if not (
        _looks_like_uvvis_ready_reply(filtered_text)
        or _classify_short_experiment_control(conn, filtered_text) in {"guide", "advance", "repeat"}
    ):
        return False

    group_number = None
    if _is_exp2_uvvis_experiment(conn):
        explicit_group_number = _extract_explicit_group_number_from_text(
            original_text,
            filtered_text,
        )
        current_group_number = _extract_int_value(
            getattr(conn, "experiment_current_group_number", None)
        )
        if explicit_group_number is None:
            if isinstance(current_group_number, int) and current_group_number >= 1:
                group_number = current_group_number
            else:
                await _start_direct_intent_turn(conn, original_text)
                speak_txt(conn, "???????????")
                return True
        else:
            group_number = await _sync_exp2_group_number_from_turn(
                conn,
                original_text,
                filtered_text,
                preferred_step_id=_UVVIS_SAMPLE_RECORD_STEP_ID,
            )

    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    await _start_direct_intent_turn(conn, original_text)
    payload = await _execute_uvvis_tool_payload(
        conn,
        "uvvis_measure_spectra",
        {
            "session_key": session_key,
            "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
            "ready_for_samples": True,
            "output_dir": str(_resolve_uvvis_group_output_dir(conn, group_number)),
        },
    )
    if _payload_looks_busy_or_inaccessible(payload):
        speak_txt(conn, _UVVIS_BUSY_REPLY)
        return True
    if _payload_mentions_missing_blank(payload):
        _clear_uvvis_direct_state(conn)
        fallback_step_id = (
            _UVVIS_SHARED_BLANK_STEP_ID
            if _experiment_has_step_id(conn, _UVVIS_SHARED_BLANK_STEP_ID)
            else _UVVIS_SHARED_DARK_AIR_STEP_ID
        )
        await _call_experiment_graph_tool_fast(
            conn,
            "redirect_to_step",
            {
                "session_id": str(getattr(conn, "experiment_session_id", "") or "").strip(),
                "step_id": fallback_step_id,
            },
            priority="foreground",
        )
        if fallback_step_id == _UVVIS_SHARED_BLANK_STEP_ID:
            speak_txt(
                conn,
                "????????????????????????????????????????",
            )
        else:
            speak_txt(
                conn,
                "????????????????????????????????????????????????????",
            )
        return True

    rows = _extract_uvvis_measure_spectra_rows(payload, conn)
    spectra_artifacts = _persist_uvvis_measure_spectra_artifacts(conn, payload, rows)
    setattr(conn, "_last_uvvis_spectra_artifacts", spectra_artifacts)
    if len(rows) < 5 or any(
        sample_position not in rows for sample_position in _UVVIS_SAMPLE_POSITIONS
    ):
        speak_txt(
            conn,
            "???????????????? 1 ? 5 ???????????????",
        )
        return True
    if not spectra_artifacts.get("all_expected_outputs_exist", False):
        speak_txt(
            conn,
            "?????????????? 400 ? 700 ????? 10 ????????????????????????????",
        )
        return True

    fields = {
        f"sample{sample_position}_lambda_max_nm": rows[sample_position]["lambda_max_nm"]
        for sample_position in _UVVIS_SAMPLE_POSITIONS
    }
    for sample_position in _UVVIS_SAMPLE_POSITIONS:
        max_absorbance = rows[sample_position].get("max_absorbance")
        if max_absorbance is not None and math.isfinite(max_absorbance):
            fields[f"sample{sample_position}_max_absorbance"] = max_absorbance
    fields["spectrum_saved"] = True
    fields["observations"] = (
        "1-5??????????"
        + "?".join(
            f"{sample_position}????max={rows[sample_position]['lambda_max_nm']}nm"
            for sample_position in _UVVIS_SAMPLE_POSITIONS
        )
        + "?400-700nm?10nm??????????????????"
    )

    auto_advanced, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=fields,
        auto_advance=True,
        fallback_reply="1-5???????????",
        group_number=group_number,
    )
    summary = "?".join(
        f"{sample_position}?{rows[sample_position]['lambda_max_nm']}??"
        for sample_position in _UVVIS_SAMPLE_POSITIONS
    )
    spoken_reply = (
        f"{summary}?400?700?????10???????????????????"
    )
    if reply:
        spoken_reply = f"{spoken_reply}{reply}"
    if auto_advanced or reply:
        speak_txt(conn, spoken_reply)
    return True


async def _handle_uvvis_sample_load_start_scan(
    conn, original_text: str, filtered_text: str
) -> bool:
    control_action = _classify_short_experiment_control(conn, filtered_text)
    is_ready_reply = _looks_like_uvvis_ready_reply(filtered_text) or control_action in {
        "guide",
        "advance",
        "repeat",
    }
    if not is_ready_reply:
        return False

    immediate_start_scan = _looks_like_uvvis_start_scan_reply(filtered_text)
    group_number = None
    explicit_group_number = None
    if _is_exp2_uvvis_experiment(conn):
        explicit_group_number = _extract_explicit_group_number_from_text(
            original_text,
            filtered_text,
        )
        if explicit_group_number is None:
            current_group_number = _extract_int_value(
                getattr(conn, "experiment_current_group_number", None)
            )
            if isinstance(current_group_number, int) and current_group_number >= 1:
                group_number = current_group_number
        else:
            group_number = await _sync_exp2_group_number_from_turn(
                conn,
                original_text,
                filtered_text,
                preferred_step_id=_UVVIS_SAMPLE_LOAD_STEP_ID,
            )

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False

    try:
        step_payload, progress_payload, schema_payload = await asyncio.gather(
            _call_experiment_graph_tool_fast(
                conn,
                "get_step",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_current_progress",
                {"session_id": session_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_schema",
                {"session_id": session_id},
                priority="foreground",
            ),
        )
    except Exception:
        return False

    if _is_current_graph_step_completed(progress_payload):
        if immediate_start_scan:
            return await _handle_uvvis_spectra_measurement(
                conn,
                original_text,
                filtered_text,
            )
        await _start_direct_intent_turn(conn, original_text)
        if isinstance(group_number, int) and group_number >= 1:
            speak_txt(conn, f"???????{group_number}??????????????")
        else:
            speak_txt(conn, "??????????????????????????")
        return True

    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        try:
            start_payload = await _call_experiment_graph_tool_fast(
                conn,
                "start_trial",
                {
                    "session_id": session_id,
                    **(
                        {"group_number": group_number}
                        if isinstance(group_number, int) and group_number >= 1
                        else {}
                    ),
                },
                priority="foreground",
            )
            current_progress = _extract_experiment_current_progress(start_payload)
            progress_payload = start_payload
        except Exception:
            current_progress = None

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    write_fields = {}
    for field_name in missing_fields:
        field_meta = schema_by_name.get(field_name, {})
        field_type = str(field_meta.get("type", "")).strip().lower()
        if field_type in {"bool", "boolean"}:
            write_fields[field_name] = True

    if missing_fields and not write_fields:
        block_reply = _compose_confirmation_step_writeback_block_reply(
            conn,
            step_payload,
            progress_payload,
            schema_payload,
        )
        if block_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, block_reply)
            return True
        return False

    completed, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=write_fields,
        auto_advance=True,
        fallback_reply="",
        group_number=group_number,
    )
    if not completed:
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply or "???????????????????????")
        return True

    if immediate_start_scan:
        return await _handle_uvvis_spectra_measurement(
            conn,
            original_text,
            filtered_text,
        )

    await _start_direct_intent_turn(conn, original_text)
    if isinstance(group_number, int) and group_number >= 1:
        speak_txt(conn, f"???????{group_number}??????????????")
    else:
        speak_txt(conn, "??????????????????????????")
    return True


async def _handle_uvvis_kinetics_measurement(
    conn, original_text: str, filtered_text: str, step_id: str
) -> bool:
    state = _get_uvvis_direct_state(conn, step_id)
    phase = str(state.get("phase") or "start_pending")
    control_action = _classify_short_experiment_control(conn, filtered_text)
    is_negative = _is_negative_short_reply_fixed(filtered_text)
    is_affirmative = _looks_like_uvvis_ready_reply(filtered_text) or control_action in {
        "guide",
        "advance",
        "repeat",
    }
    explicit_group_number = _extract_explicit_group_number_from_text(
        original_text,
        filtered_text,
    )
    if explicit_group_number is not None:
        await _sync_exp2_group_number_from_turn(
            conn,
            original_text,
            filtered_text,
            preferred_step_id=step_id,
        )
    group_number = explicit_group_number
    if group_number is None:
        current_group_number = _extract_int_value(
            getattr(conn, "experiment_current_group_number", None)
        )
        if isinstance(current_group_number, int) and current_group_number >= 1:
            group_number = current_group_number
    if group_number is None and _is_exp2_uvvis_experiment(conn):
        group_number = 1
        setattr(conn, "experiment_current_group_number", group_number)
    output_dir = str(_resolve_uvvis_group_output_dir(conn, group_number))

    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    if phase == "await_finish_decision":
        if control_action == "repeat":
            await _start_direct_intent_turn(conn, original_text)
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_grouped_samples",
                session_key=session_key,
            )
            speak_txt(
                conn,
                "璇烽噸鏂拌濂藉姩鍔涘鏍峰搧銆傚弬姣斾綅鏀剧函姘达紝1鍙蜂綅鐣欑┖锛?鍜?鍙蜂綅鏀?鍙锋牱鍝佺殑鍙嶅簲娑插拰鍙傛瘮娑诧紝4鍜?鍙蜂綅鏀?鍙锋牱鍝佺殑鍙嶅簲娑插拰鍙傛瘮娑层€傛斁濂藉悗鍛婅瘔鎴戝紑濮嬨€?,
            )
            return True
        if control_action not in {"guide", "advance"}:
            if is_affirmative:
                await _start_direct_intent_turn(conn, original_text)
                speak_txt(
                    conn,
                    "濡傛灉杩樿缁х画閲嶆祴杩欎竴杞紝璇疯鍐嶆祴涓€杞紱濡傛灉杩欎釜鍔ㄥ姏瀛︽楠ゅ凡缁忓叏閮ㄥ畬鎴愶紝璇疯褰撳墠姝ラ瀹屾垚浜嗐€?,
                )
                return True
            return False

        await _start_direct_intent_turn(conn, original_text)
        advanced, reply = await _advance_finished_experiment_step(
            conn,
            fallback_reply="",
        )
        if advanced:
            await _release_uvvis_session_for_analysis(conn)
            _clear_uvvis_direct_state(conn)
        if reply:
            speak_txt(conn, reply)
        return True

    if phase == "await_grouped_samples":
        if is_negative:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鎶?鍒?鍙蜂綅鎸夎姹傛斁濂戒箣鍚庡啀鍛婅瘔鎴戙€?)
            return True
        if not is_affirmative:
            return False

        await _start_direct_intent_turn(conn, original_text)
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_measure_kinetics",
            {
                "session_key": session_key,
                "wavelength_nm": 400,
                "duration_minutes": 34,
                "interval_seconds": 60,
                "ready_for_samples": True,
                "sample_positions": list(_UVVIS_GROUPED_KINETICS_POSITIONS),
                "output_dir": output_dir,
            },
        )
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True

        record_fields = _extract_uvvis_measure_kinetics_record_fields(payload, conn, "")
        sample2_fields = {
            key: value
            for key, value in record_fields.items()
            if re.fullmatch(r"sample2_t\d+_absorbance", str(key or ""))
        }
        sample4_fields = {
            key: value
            for key, value in record_fields.items()
            if re.fullmatch(r"sample4_t\d+_absorbance", str(key or ""))
        }
        if len(sample2_fields) < 35 or len(sample4_fields) < 35:
            speak_txt(conn, "杩欐鍔ㄥ姏瀛︾粨鏋滆繕涓嶅畬鏁达紝鎴戣繕娌℃湁鎷垮埌2鍙峰拰4鍙锋牱鍝佸悇鑷畬鏁寸殑35涓椂闂寸偣锛岃绋嶅悗鍐嶈瘯銆?)
            return True

        payload_data = _to_plain_data(payload)
        round_index = _extract_int_value(payload_data.get("round_index"))
        sample_groups = payload_data.get("sample_groups")
        if not isinstance(sample_groups, list):
            sample_groups = []
        required_paths = [
            payload_data.get("progress_json"),
            payload_data.get("manifest_json"),
            payload_data.get("raw_csv"),
            payload_data.get("absorbance_csv"),
        ]
        for group in sample_groups:
            if not isinstance(group, dict):
                continue
            required_paths.extend([group.get("raw_csv"), group.get("absorbance_csv")])
        artifacts_ok = True
        for path_text in required_paths:
            path = Path(str(path_text or "").strip())
            if not path_text or not path.exists():
                artifacts_ok = False
                break
        setattr(conn, "_last_uvvis_kinetics_artifacts", payload_data)
        if not artifacts_ok:
            speak_txt(conn, "杩欐鍔ㄥ姏瀛︾殑鏁版嵁鏂囦欢杩樻病鏈夊畬鏁磋惤鐩橈紝璇风◢鍚庡啀璇曘€?)
            return True

        fields = dict(sample2_fields)
        fields.update(sample4_fields)
        if round_index is not None:
            fields["kinetics_round_index"] = round_index
        fields["kinetics_round_saved"] = True
        fields["step_finished_confirmed"] = False
        fields["observations"] = (
            f"2鍙峰拰4鍙锋牱鍝?400 nm 鑱斿悎鍔ㄥ姏瀛︽祴閲忓畬鎴愶紝"
            f"2鍙锋牱鍝佷娇鐢?/3鍙蜂綅锛?鍙锋牱鍝佷娇鐢?/5鍙蜂綅锛?
            f"鏈疆鍏辫褰?5涓椂闂寸偣锛屾暟鎹凡鍐欏叆 uvvis_measure_kinetic/{round_index or '?'}銆?
        )

        completed, reply = await _complete_experiment_step_with_fields(
            conn,
            fields=fields,
            auto_advance=False,
            fallback_reply="鎴戣褰曞ソ浜嗐€傚鏋滆繕瑕佺户缁噸娴嬶紝璇疯鍐嶆祴涓€杞紱濡傛灉褰撳墠姝ラ宸茬粡鍏ㄩ儴瀹屾垚锛岃鐩存帴鍛婅瘔鎴戙€?,
        )
        if completed:
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_finish_decision",
                session_key=session_key,
            )
            speak_txt(
                conn,
                "鎴戣褰曞ソ浜嗐€傝繖涓€杞?鍙峰拰4鍙锋牱鍝佺殑鍔ㄥ姏瀛︽暟鎹兘宸茬粡淇濆瓨銆傚鏋滆繕瑕佺户缁噸娴嬶紝璇疯鍐嶆祴涓€杞紱濡傛灉杩欎釜姝ラ宸茬粡鍏ㄩ儴瀹屾垚锛岃鐩存帴鍛婅瘔鎴戙€?,
            )
        elif reply:
            speak_txt(conn, reply)
        else:
            speak_txt(conn, "杩欐鍔ㄥ姏瀛︾粨鏋滆繕娌℃湁瀹屾暣鍐欏洖褰撳墠姝ラ锛岃绋嶅悗鍐嶈瘯銆?)
        return True

    if phase in {"start_pending", ""}:
        if is_negative:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "濂斤紝绛変綘鍑嗗濂藉啀鍛婅瘔鎴戙€?)
            return True

        if not (
            is_affirmative
            or _contains_any(
                _normalize_text_for_match(filtered_text),
                ("鏆楃數娴?, "绌烘皵", "绌虹櫧", "鍔ㄥ姏瀛?, "寮€濮?, "娴嬮噺", "璋冪敤mcp", "鎵氨"),
            )
        ):
            return False

        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, "鍏堜繚鎸?鍒?鍙锋牱鍝佷綅涓虹┖锛屾垜鍏堝仛400绾崇背鍔ㄥ姏瀛︽祴閲忛渶瑕佺殑鍏变韩鍓嶇疆鍑嗗銆?)
        payload = await _execute_uvvis_tool_payload(
            conn,
            "uvvis_measure_kinetics",
            {
                "session_key": session_key,
                "wavelength_nm": 400,
                "duration_minutes": 34,
                "interval_seconds": 60,
                "ready_for_samples": False,
                "sample_positions": list(_UVVIS_GROUPED_KINETICS_POSITIONS),
                "output_dir": output_dir,
            },
        )
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True

        _set_uvvis_direct_state(
            conn,
            step_id=step_id,
            phase="await_grouped_samples",
            session_key=session_key,
        )
        speak_txt(
            conn,
            "鍏变韩鍓嶇疆鍑嗗濂戒簡銆傝淇濇寔鍙傛瘮浣嶆槸绾按锛?鍙蜂綅鐣欑┖锛?鍜?鍙蜂綅鏀?鍙锋牱鍝佺殑鍙嶅簲娑插拰鍙傛瘮娑诧紝4鍜?鍙蜂綅鏀?鍙锋牱鍝佺殑鍙嶅簲娑插拰鍙傛瘮娑层€傚叏閮ㄦ斁濂藉悗鍛婅瘔鎴戝紑濮嬨€?,
        )
        return True

    return False

def _compose_uvvis_step_rejection_reply(step_id: str) -> str:
    step_id = str(step_id or "").strip()
    if step_id == _UVVIS_SHARED_DARK_AIR_STEP_ID:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?UV-Vis 鐨勬殫鐢垫祦鏍℃锛屽厛瀹屾垚涓佽揪灏旂幇璞¤瀵熴€?
    if step_id == _UVVIS_SHARED_BLANK_STEP_ID:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?UV-Vis 鐨勭函姘寸┖鐧芥牎姝ｏ紝鍏堝畬鎴愭殫鐢垫祦鏍℃銆?
    if step_id == _UVVIS_SAMPLE_RECORD_STEP_ID:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒?1-5 鍙锋牱鍝佺殑鎵归噺鍏夎氨娴嬮噺锛屽厛瀹屾垚鍓嶉潰鐨勮鏍峰噯澶囥€?
    if step_id in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
    }:
        return "褰撳墠瀹為獙鍥捐氨杩樻病鎺ㄨ繘鍒板搴旂殑 400 绾崇背鍔ㄥ姏瀛︽祴閲忥紝鍏堝畬鎴愬墠闈㈢殑鍔ㄥ姏瀛﹂厤娑插噯澶囥€?
    return _UVVIS_NOT_READY_REPLY


async def handle_direct_uvvis_intent(conn, original_text: str, filtered_text: str) -> bool:
    if not _is_server_mcp_client_enabled(conn):
        if _looks_like_explicit_uvvis_turn(original_text, filtered_text) or _looks_like_uvvis_followup_reply(
            conn,
            filtered_text,
        ):
            conn.logger.bind(tag=TAG).info(
                "server MCP client disabled; skipping local UV-Vis direct intent so Codex app-server handles it"
            )
        return False

    step_id = _get_current_experiment_step_id(conn)
    status_query = _looks_like_uvvis_status_query(
        conn,
        original_text,
        filtered_text,
        inferred_step_id="",
    )
    if not _is_uvvis_step(step_id):
        if not (
            status_query
            or _looks_like_explicit_uvvis_turn(original_text, filtered_text)
            or _looks_like_uvvis_followup_reply(conn, filtered_text)
        ):
            return False

    inferred_step_id = step_id if _is_uvvis_step(step_id) else _infer_uvvis_step_id_from_context(
        conn,
        original_text,
        filtered_text,
    )
    actual_current_step_id = _get_current_experiment_step_id(conn)
    if (
        inferred_step_id == _UVVIS_SHARED_BLANK_STEP_ID
        and not _uvvis_shared_blank_step_enabled(conn)
    ):
        replacement_step_id = (
            actual_current_step_id
            if _is_uvvis_step(actual_current_step_id)
            and actual_current_step_id != _UVVIS_SHARED_BLANK_STEP_ID
            else _UVVIS_SHARED_DARK_AIR_STEP_ID
        )
        conn.logger.bind(tag=TAG).info(
            "ignoring legacy uvvis shared blank branch for current experiment: "
            f"inferred_step_id={inferred_step_id}, replacement_step_id={replacement_step_id}"
        )
        inferred_step_id = replacement_step_id
    if _is_uvvis_step(actual_current_step_id) and _is_uvvis_step(inferred_step_id):
        step_cmp = _compare_experiment_step_order(
            conn,
            inferred_step_id,
            actual_current_step_id,
        )
        if step_cmp is not None and step_cmp < 0:
            conn.logger.bind(tag=TAG).info(
                "ignoring stale inferred uvvis step behind current graph step: "
                f"inferred_step_id={inferred_step_id}, current_step_id={actual_current_step_id}"
            )
            inferred_step_id = actual_current_step_id
            step_id = actual_current_step_id

    if _looks_like_uvvis_status_query(
        conn,
        original_text,
        filtered_text,
        inferred_step_id=inferred_step_id,
    ):
        await _start_direct_intent_turn(conn, original_text)
        status = await _get_uvvis_session_status(conn)
        speak_txt(
            conn,
            _compose_uvvis_status_reply(
                conn,
                status,
                inferred_step_id=inferred_step_id,
            ),
        )
        return True

    if not _is_uvvis_step(step_id):
        if not inferred_step_id:
            return False
        if inferred_step_id != step_id:
            redirected = await _try_redirect_experiment_step_fast(
                conn,
                inferred_step_id,
                log_reason="direct uvvis redirecting stale graph step",
            )
            if not redirected:
                speak_txt(conn, _compose_uvvis_step_rejection_reply(inferred_step_id))
                return True
        step_id = inferred_step_id

    if step_id == _UVVIS_ANALYSIS_STEP_ID:
        await _release_uvvis_session_for_analysis(conn)
        return False

    state = _get_uvvis_direct_state(conn, step_id)

    if step_id == _UVVIS_SHARED_DARK_AIR_STEP_ID:
        return await _handle_uvvis_shared_dark_air_prep(conn, original_text, filtered_text)

    if step_id == _UVVIS_SHARED_BLANK_STEP_ID and _uvvis_shared_blank_step_enabled(conn):
        return await _handle_uvvis_shared_blank_prep(conn, original_text, filtered_text, state)

    if step_id in {_UVVIS_SAMPLE_RECORD_STEP_ID, _UVVIS_SAMPLE_RECORD_STEP_ID_LEGACY}:
        return await _handle_uvvis_spectra_measurement(conn, original_text, filtered_text)

    if step_id in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
    }:
        return await _handle_uvvis_kinetics_measurement(
            conn,
            original_text,
            filtered_text,
            step_id,
        )

    if step_id in {_UVVIS_SAMPLE_LOAD_STEP_ID, _UVVIS_SAMPLE_LOAD_STEP_ID_LEGACY}:
        return await _handle_uvvis_sample_load_start_scan(
            conn,
            original_text,
            filtered_text,
        )

    if step_id in {_UVVIS_SAMPLE_CLEAN_STEP_ID, _UVVIS_SAMPLE_CLEAN_STEP_ID_LEGACY}:
        return False

    return False


async def _advance_experiment_step_fast(conn, session_id: str) -> str:
    step_payload, progress_payload, schema_payload, can_proceed_payload = await asyncio.gather(
        _call_experiment_graph_tool_fast(
            conn,
            "get_step",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_current_progress",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_schema",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "can_proceed",
            {"session_id": session_id},
            priority="foreground",
        ),
    )

    conn.experiment_current_step = step_payload
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(step_payload)
        if current_step_id:
            conn.experiment_current_step_id = current_step_id

    step_meta = _merge_experiment_step_meta(
        _extract_experiment_step_meta(step_payload),
        _extract_experiment_step_meta(getattr(conn, "experiment_progress_summary", None)),
    )
    schema_by_name = _extract_experiment_schema_view(schema_payload)

    if bool(_experiment_result_body(can_proceed_payload).get("ok")):
        proceed_payload = await _call_experiment_graph_tool_fast(
            conn,
            "proceed_to_next_step",
            {"session_id": session_id},
            priority="foreground",
        )
        if bool(_experiment_result_body(proceed_payload).get("ok")):
            next_meta = await _refresh_experiment_step_cache(conn, session_id)
            reply = _compose_experiment_step_reply(next_meta, mode="next")
            if reply:
                return reply
            return ""

    current_progress = _extract_experiment_current_progress(progress_payload)
    if current_progress is None:
        start_payload = await _call_experiment_graph_tool_fast(
            conn,
            "start_trial",
            {"session_id": session_id},
            priority="foreground",
        )
        current_progress = _extract_experiment_current_progress(start_payload)

    missing_fields = list((current_progress or {}).get("missing_fields") or [])
    if missing_fields:
        return _compose_missing_field_reply(missing_fields, schema_by_name)

    finish_payload = await _call_experiment_graph_tool_fast(
        conn,
        "finish_trial",
        {"session_id": session_id, "validate": True},
        priority="foreground",
    )
    if not bool(_experiment_result_body(finish_payload).get("ok")):
        message = _extract_experiment_result_message(finish_payload)
        if message:
            return message
        return _compose_experiment_step_reply(step_meta, mode="guide")

    can_proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "can_proceed",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(can_proceed_payload).get("ok")):
        message = _extract_experiment_result_message(can_proceed_payload)
        if message:
            return message
        return _compose_experiment_step_reply(step_meta, mode="guide")

    proceed_payload = await _call_experiment_graph_tool_fast(
        conn,
        "proceed_to_next_step",
        {"session_id": session_id},
        priority="foreground",
    )
    if not bool(_experiment_result_body(proceed_payload).get("ok")):
        message = _extract_experiment_result_message(proceed_payload)
        if message:
            return message
        return _compose_experiment_step_reply(step_meta, mode="guide")

    next_meta = await _refresh_experiment_step_cache(conn, session_id)
    reply = _compose_experiment_step_reply(next_meta, mode="next")
    if reply:
        return reply
    return ""


async def handle_experiment_control_fast_intent(
    conn, original_text: str, filtered_text: str
) -> bool:
    if not _is_experiment_fast_path_available(conn):
        return False

    if _is_explicit_experiment_resume_request(filtered_text):
        return await _handle_explicit_experiment_resume_request(
            conn,
            original_text,
        )

    if _is_explicit_experiment_start_request(filtered_text):
        if not _is_experiment_fast_path_action_enabled(conn, "start"):
            return False
        try:
            await _reset_experiment_fresh_start_context(conn)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment fresh start reset failed: {exc}"
            )
        experiment_title = await _load_experiment_overview_title(conn)
        reply = _compose_experiment_start_reply(experiment_title, "")
        reply = _prepare_fastpath_spoken_reply(reply)
        if not reply:
            return False
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    waiting_for_step_start = _assistant_waiting_for_step_start(conn)
    explicitly_ready_to_start = _is_explicit_ready_to_start_reply(filtered_text)
    action = _classify_short_experiment_control(conn, filtered_text)

    if not action:
        if not _is_experiment_fast_path_action_enabled(conn, "confirm"):
            return False
        return await _handle_confirmation_step_semantic_fast_intent(
            conn,
            original_text,
            filtered_text,
        )

    if not _is_experiment_fast_path_action_enabled(conn, action):
        return False

    conn.logger.bind(tag=TAG).info(
        f"experiment control fast path hit: action={action}, text={filtered_text}"
    )

    ready_reply_unlocks_step = (
        waiting_for_step_start
        and explicitly_ready_to_start
        and action in {"guide", "advance"}
    )

    if waiting_for_step_start and action == "repeat":
        experiment_title = await _load_experiment_overview_title(conn)
        reply = _prepare_fastpath_spoken_reply(
            _compose_experiment_start_reply(experiment_title, "")
        )
        if not reply:
            return False
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    if action in {"guide", "repeat"}:
        step_meta = await _load_experiment_step_meta(conn)
        reply = _compose_experiment_step_reply(
            step_meta,
            mode="repeat" if action == "repeat" else "guide",
        )
        if action == "guide" and _is_explicit_experiment_start_request(filtered_text):
            experiment_title = await _load_experiment_overview_title(conn)
            reply = _compose_experiment_start_reply(experiment_title, reply)
            reply = _prepare_fastpath_spoken_reply(reply)
        else:
            reply = _prepare_fastpath_spoken_reply(
                reply,
                fallback_step_meta=step_meta,
                fallback_mode="repeat" if action == "repeat" else "guide",
            )
        if not reply:
            return False
        await _start_direct_intent_turn(conn, original_text)
        if ready_reply_unlocks_step and action == "guide":
            _grant_experiment_ready_guard_bypass(conn)
        speak_txt(conn, reply)
        return True

    if action == "advance":
        session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
        if not session_id:
            step_meta = await _load_experiment_step_meta(conn)
            reply = _compose_experiment_step_reply(step_meta, mode="guide")
            if not reply:
                return False
            await _start_direct_intent_turn(conn, original_text)
            if ready_reply_unlocks_step:
                _grant_experiment_ready_guard_bypass(conn)
            speak_txt(conn, reply)
            return True

        try:
            reply = await _try_apply_current_confirmation_report(conn, filtered_text)
            if not reply:
                block_reply = await _load_confirmation_step_writeback_block_reply(
                    conn, session_id
                )
                if block_reply:
                    conn.logger.bind(tag=TAG).info(
                        "experiment control fast advance blocked until graph writeback completes"
                    )
                    reply = block_reply
                else:
                    reply = await _advance_experiment_step_fast(conn, session_id)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment control fast advance failed: {exc}"
            )
            return False

        next_step_meta = _get_cached_experiment_step_meta(conn)
        reply = _prepare_fastpath_spoken_reply(
            reply,
            fallback_step_meta=next_step_meta,
            fallback_mode="guide",
        )
        if not reply:
            return False

        await _start_direct_intent_turn(conn, original_text)
        if ready_reply_unlocks_step:
            _grant_experiment_ready_guard_bypass(conn)
        if hasattr(conn, "enrich_latest_clean_user_utterance_snapshot"):
            try:
                conn.enrich_latest_clean_user_utterance_snapshot()
            except Exception:
                pass
        speak_txt(conn, reply)
        return True

    return False


async def handle_experiment_control_strict_graph_intent(
    conn,
    original_text: str,
    filtered_text: str,
) -> bool:
    if not _is_experiment_strict_graph_path_enabled(conn):
        return False

    if _is_experiment_fast_path_available(conn):
        return False

    if _is_explicit_experiment_resume_request(filtered_text):
        return False
    if _is_explicit_experiment_start_request(filtered_text):
        return False

    if _assistant_waiting_for_step_start(conn) and _is_explicit_ready_to_start_reply(
        filtered_text
    ):
        return False

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False

    action = _classify_short_experiment_control(conn, filtered_text)
    if not action:
        return await _handle_confirmation_step_semantic_fast_intent(
            conn,
            original_text,
            filtered_text,
        )

    if action != "advance":
        return False

    conn.logger.bind(tag=TAG).info(
        f"experiment control strict graph path hit: action={action}, text={filtered_text}"
    )

    try:
        reply = await _try_apply_current_confirmation_report(conn, filtered_text)
        if not reply:
            block_reply = await _load_confirmation_step_writeback_block_reply(
                conn, session_id
            )
            if block_reply:
                conn.logger.bind(tag=TAG).info(
                    "experiment control strict graph advance blocked until graph writeback completes"
                )
                reply = block_reply
            else:
                reply = await _advance_experiment_step_fast(conn, session_id)
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment control strict graph advance failed: {exc}"
        )
        return False

    next_step_meta = _get_cached_experiment_step_meta(conn)
    reply = _prepare_fastpath_spoken_reply(
        reply,
        fallback_step_meta=next_step_meta,
        fallback_mode="guide",
    )
    if not reply:
        return False

    await _start_direct_intent_turn(conn, original_text)
    if hasattr(conn, "enrich_latest_clean_user_utterance_snapshot"):
        try:
            conn.enrich_latest_clean_user_utterance_snapshot()
        except Exception:
            pass
    speak_txt(conn, reply)
    return True


def _is_direct_photo_command(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    block_keywords = [
        "\u4e3a\u4ec0\u4e48",
        "\u539f\u7406",
        "\u6b65\u9aa4",
        "\u6ce8\u610f\u4e8b\u9879",
        "\u600e\u4e48\u505a",
        "\u5982\u4f55\u505a",
        "\u4ec0\u4e48\u610f\u601d",
    ]
    if _contains_any(norm, block_keywords):
        return False

    trigger_keywords = [
        "\u62cd\u7167",
        "\u62cd\u4e00\u5f20",
        "\u62cd\u4e2a\u7167",
        "\u62cd\u5f20\u7167",
        "\u62cd\u5f20\u7167\u7247",
        "\u62cd\u4e00\u5f20\u7167\u7247",
        "\u91cd\u62cd",
        "\u91cd\u65b0\u62cd",
        "\u518d\u62cd",
        "\u8865\u62cd",
        "\u7167\u4e00\u4e0b",
        "\u770b\u4e00\u4e0b\u524d\u9762",
        "\u770b\u770b\u524d\u9762",
        "\u770b\u770b\u5f53\u524d\u753b\u9762",
        "\u770b\u4e00\u4e0b\u5f53\u524d\u753b\u9762",
        "\u5e2e\u6211\u770b\u4e00\u4e0b",
        "\u5e2e\u6211\u770b\u4e00\u773c",
        "\u770b\u4e00\u773c\u524d\u9762",
    ]
    return _contains_any(norm, trigger_keywords)


_PHOTO_SAMPLE_DIGIT_BY_CN = {
    "\u4e00": "1",
    "\u4e8c": "2",
    "\u4e24": "2",
    "\u4e09": "3",
    "\u56db": "4",
    "\u4e94": "5",
}
_PHOTO_SAMPLE_RE = re.compile(
    r"([1-5\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94])\s*\u53f7?\s*\u6837\u54c1"
)
_PHOTO_RETAKE_RE = re.compile(r"\u91cd\u62cd|\u91cd\u65b0\u62cd|\u518d\u62cd|\u8865\u62cd")


def _extract_sample_photo_stem_from_command(filtered_text: str) -> str:
    match = _PHOTO_SAMPLE_RE.search(_normalize_text_for_match(filtered_text))
    if not match:
        return ""
    sample = _PHOTO_SAMPLE_DIGIT_BY_CN.get(match.group(1), match.group(1))
    return f"{sample}\u53f7\u6837\u54c1\u7167\u7247"


def _is_sample_photo_retake_command(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    return bool(_PHOTO_RETAKE_RE.search(norm) and _PHOTO_SAMPLE_RE.search(norm))


def _classify_photo_nav_command(filtered_text: str) -> str:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return ""

    block_keywords = [
        "\u4e3a\u4ec0\u4e48",
        "\u539f\u7406",
        "\u6b65\u9aa4",
        "\u6ce8\u610f\u4e8b\u9879",
        "\u600e\u4e48\u505a",
        "\u5982\u4f55\u505a",
        "\u4ec0\u4e48\u610f\u601d",
    ]
    if _contains_any(norm, block_keywords):
        return ""

    previous_keywords = [
        "\u4e0a\u4e00\u5f20",
        "\u524d\u4e00\u5f20",
        "\u4e0a\u4e00\u5f20\u7167\u7247",
        "\u524d\u4e00\u5f20\u7167\u7247",
        "\u4e0a\u5f20",
        "\u524d\u5f20",
    ]
    if _contains_any(norm, previous_keywords):
        return "previous"

    latest_keywords = [
        "\u67e5\u770b\u6700\u8fd1\u7167\u7247",
        "\u770b\u6700\u8fd1\u7167\u7247",
        "\u770b\u770b\u6700\u8fd1\u7167\u7247",
        "\u6700\u8fd1\u7167\u7247",
        "\u6700\u65b0\u7167\u7247",
        "\u6700\u8fd1\u4e00\u5f20",
        "\u6700\u65b0\u4e00\u5f20",
    ]
    if _contains_any(norm, latest_keywords):
        return "latest"

    return ""


def _build_direct_photo_question(original_text: str, default_question: str) -> str:
    text = (original_text or "").strip()
    if not text:
        return default_question
    if _contains_any(
        text,
        [
            "\u5206\u6790",
            "\u8bc6\u522b",
            "\u63cf\u8ff0",
            "\u770b\u770b",
            "\u770b\u4e00\u4e0b",
            "\u5e2e\u6211\u770b",
        ],
    ):
        return text
    return default_question


def _to_plain_data(payload):
    if payload is None:
        return None
    if isinstance(payload, (dict, list, str, int, float, bool)):
        return payload
    if hasattr(payload, "model_dump"):
        try:
            return payload.model_dump()
        except Exception:
            pass
    if hasattr(payload, "__dict__"):
        try:
            return dict(payload.__dict__)
        except Exception:
            pass
    return str(payload)


def _try_parse_json_text(text: str):
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    if not (raw.startswith("{") or raw.startswith("[")):
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_server_mcp_payload(raw_result):
    data = _to_plain_data(raw_result)
    if isinstance(data, str):
        parsed = _try_parse_json_text(data)
        return parsed if parsed is not None else data

    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                item_data = _to_plain_data(item)
                if isinstance(item_data, dict):
                    text = item_data.get("text")
                    if isinstance(text, str):
                        parsed_text = _try_parse_json_text(text)
                        if parsed_text is not None:
                            return parsed_text
                        if text.strip():
                            return text.strip()
        return data

    if isinstance(data, list):
        for item in data:
            extracted = _extract_server_mcp_payload(item)
            if extracted is not None:
                return extracted
    return data


def _extract_text_from_result_payload(payload):
    data = _to_plain_data(payload)

    if isinstance(data, str):
        text = data.strip()
        if text and not text.startswith("{") and not text.startswith("["):
            return text
        parsed = _try_parse_json_text(text)
        if parsed is not None:
            return _extract_text_from_result_payload(parsed)
        return ""

    if isinstance(data, dict):
        for key in ["response", "message", "text", "description"]:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        nested = data.get("result")
        if nested is not None:
            nested_text = _extract_text_from_result_payload(nested)
            if nested_text:
                return nested_text

        photo_meta = data.get("photo_meta")
        if isinstance(photo_meta, dict):
            file_name = str(photo_meta.get("file_name", "")).strip()
            if file_name:
                return f"\u62cd\u597d\u4e86\uff0c\u5df2\u4fdd\u5b58\u4e3a {file_name}"
        return ""

    if isinstance(data, list):
        for item in data:
            text = _extract_text_from_result_payload(item)
            if text:
                return text
    return ""


def _extract_direct_photo_reply(raw_result) -> str:
    payload = _extract_server_mcp_payload(raw_result)
    text = _extract_text_from_result_payload(payload)
    if text:
        return text
    return ""


def _extract_photo_result_meta(payload) -> dict:
    data = _to_plain_data(payload)
    if not isinstance(data, dict):
        return {}

    photo_meta = data.get("photo_meta")
    if not isinstance(photo_meta, dict):
        nested = data.get("result")
        nested_data = _to_plain_data(nested)
        if isinstance(nested_data, dict):
            photo_meta = nested_data.get("photo_meta")

    if not isinstance(photo_meta, dict):
        photo_meta = {}

    file_name = str(photo_meta.get("file_name", "")).strip()
    photo_path = str(
        photo_meta.get("mirrored_path")
        or photo_meta.get("local_path")
        or data.get("saved_photo_path")
        or ""
    ).strip()
    if not photo_path:
        nested = _to_plain_data(data.get("result"))
        if isinstance(nested, dict):
            photo_path = str(
                nested.get("saved_photo_path")
                or nested.get("photo_path")
                or nested.get("local_path")
                or ""
            ).strip()
    if not file_name and photo_path:
        try:
            file_name = Path(photo_path).name
        except Exception:
            file_name = str(photo_path).replace("\\", "/").rsplit("/", 1)[-1].strip()
    requested_photo_name = str(
        photo_meta.get("requested_photo_name") or data.get("requested_photo_name") or ""
    ).strip()
    group_number = None
    for candidate in (
        photo_meta.get("group_number"),
        data.get("group_number"),
    ):
        try:
            group_number = int(candidate)
        except (TypeError, ValueError):
            continue
        if group_number >= 1:
            break
        group_number = None
    try:
        mtime = float(photo_meta.get("mtime", 0.0) or 0.0)
    except (TypeError, ValueError):
        mtime = 0.0

    return {
        "found": bool(photo_meta.get("found", False) or file_name or photo_path),
        "file_name": file_name,
        "photo_path": photo_path,
        "requested_photo_name": requested_photo_name,
        "group_number": group_number,
        "mtime": mtime,
    }


def _get_server_mcp_manager(conn):
    func_handler = getattr(conn, "func_handler", None)
    if not func_handler:
        return None
    server_executor = getattr(func_handler, "server_mcp_executor", None)
    if not server_executor:
        return None
    return getattr(server_executor, "mcp_manager", None)


async def _execute_server_mcp_tool_direct(conn, tool_name: str, arguments: dict):
    manager = _get_server_mcp_manager(conn)
    if not manager:
        raise RuntimeError("server mcp manager is not ready")
    return await manager.execute_tool(tool_name, arguments or {})


def _is_affirmative_short_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    negative_tokens = (
        "涓嶅彲浠?,
        "涓嶈",
        "鍒媿",
        "涓嶆媿",
        "鍏堝埆",
        "涓嶈兘鎷?,
        "涓嶈鎷?,
        "鍒幇鍦ㄦ媿",
    )
    if _contains_any(norm, negative_tokens):
        return False

    if norm in {
        "濂?,
        "濂界殑",
        "濂藉晩",
        "濂藉憖",
        "鍙互",
        "鍙互鐨?,
        "鍙互鎷?,
        "鎷嶅惂",
        "鎷?,
        "寮€濮嬫媿",
        "琛?,
        "琛岀殑",
        "琛屽晩",
        "鍡?,
        "鍡棷",
        "鏄?,
        "瀵?,
        "娌￠棶棰?,
        "鍚屾剰",
        "鍏佽",
        "鍑嗗濂戒簡",
        "鎴戝噯澶囧ソ浜?,
    }:
        return True

    if _looks_like_question_reply(norm):
        return False

    affirmative_tokens = (
        "鍙互鎷嶇収",
        "鐜板湪鍙互鎷嶇収",
        "鍙互鎷?,
        "鍙互鎷嶄簡",
        "鐜板湪鍙互鎷?,
        "鐜板湪鍙互浜?,
        "鍙互浜?,
        "鎷嶇収鍚?,
        "鎷嶄竴寮犲惂",
        "鎷嶄竴涓嬪惂",
        "鐩存帴鎷嶅惂",
        "娌￠棶棰樻媿",
        "鍚屾剰鎷?,
    )
    if _contains_any(norm, affirmative_tokens):
        return True

    affirmative_prefixes = (
        "濂?,
        "濂界殑",
        "濂藉晩",
        "濂藉憖",
        "鍙互",
        "鍙互鐨?,
        "鍙互鍟?,
        "鍙互鍛€",
        "琛?,
        "琛岀殑",
        "琛屽晩",
        "琛屽憖",
        "鍡?,
        "鍡棷",
        "瀵?,
        "鏄?,
        "娌￠棶棰?,
        "褰撶劧鍙互",
        "鍚屾剰",
        "鍏佽",
        "鍑嗗濂戒簡",
        "鎴戝噯澶囧ソ浜?,
    )
    return _starts_with_any(norm, affirmative_prefixes)


def _is_negative_short_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if norm in {
        "涓嶈",
        "鍏堝埆",
        "鍒媿",
        "涓嶆媿",
        "杩樻病鍑嗗濂?,
        "娌″噯澶囧ソ",
        "绛夌瓑",
        "绛変竴涓?,
        "鏆傛椂涓嶈",
        "涓嶅彲浠?,
        "鍙栨秷",
    }:
        return True

    if not norm or len(norm) > 12:
        return False

    negative_tokens = (
        "涓嶅彲浠ユ媿鐓?,
        "鐜板湪涓嶅彲浠ユ媿鐓?,
        "杩樹笉鍙互鎷嶇収",
        "杩樹笉鑳芥媿鐓?,
        "鍏堝埆鎷嶇収",
    )
    return _contains_any(norm, negative_tokens)


def _get_last_assistant_text(conn) -> str:
    dialogue_items = getattr(getattr(conn, "dialogue", None), "dialogue", [])
    for item in reversed(dialogue_items):
        if getattr(item, "role", "") != "assistant":
            continue
        content = getattr(item, "content", "")
        if isinstance(content, str) and content.strip():
            return textUtils.normalize_spoken_text(content)
    return ""


def _extract_sample_photo_name(text: str) -> str:
    src = textUtils.normalize_spoken_text(text or "")
    if not src:
        return ""

    patterns = (
        r"([0-9]+鍙锋牱鍝?",
        r"(鏍峰搧[0-9]+)",
        r"([涓€浜屼笁鍥涗簲鍏竷鍏節鍗乚+鍙锋牱鍝?",
    )
    for pattern in patterns:
        match = re.search(pattern, src)
        if match:
            return match.group(1)
    return ""


def _assistant_is_waiting_for_photo_permission(conn) -> bool:
    last_text = _normalize_text_for_match(_get_last_assistant_text(conn))
    if not last_text:
        return False
    photo_tokens = (
        "鎷嶇収",
        "鎷嶄竴寮?,
        "鎷嶄竴涓?,
        "鐓т竴涓?,
        "鐓х墖",
    )
    if not _contains_any(last_text, photo_tokens):
        return False

    explicit_wait_tokens = (
        "寰楀埌鑲畾绛斿鍚庡啀鎷?,
        "纭鍚庡啀鎷?,
        "鍚屾剰鍚庡啀鎷?,
        "鍥炲鍙互鍐嶆媿",
    )
    if _contains_any(last_text, explicit_wait_tokens):
        return True

    prompt_tokens = (
        "鍙互",
        "鑳?,
        "瑕佷笉瑕?,
        "瑕佷笉",
        "瑕佹垜",
        "甯綘",
        "缁欎綘",
        "璁╂垜",
        "鏄惁",
        "纭",
        "鍚屾剰",
    )
    question_tokens = (
        "鍚?,
        "涔?,
        "鍢?,
        "鏄惁",
        "鍙笉鍙互",
        "鑳戒笉鑳?,
        "瑕佷笉瑕?,
    )
    return _contains_any(last_text, prompt_tokens) and _contains_any(
        last_text, question_tokens
    )


def _build_pending_server_photo_request(conn) -> dict:
    last_text = textUtils.normalize_spoken_text(_get_last_assistant_text(conn))
    sample_name = _extract_sample_photo_name(last_text)
    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()

    if sample_name:
        question = f"璇锋媿鎽剓sample_name}褰撳墠鐘舵€佺殑鐓х墖銆?
    else:
        question = "璇锋媿鎽勫綋鍓嶆牱鍝佺殑鐓х墖銆?

    request = {
        "device_id": safe_device_id,
        "question": question,
    }
    if sample_name:
        request["photo_name"] = sample_name
    return request


def _get_last_assistant_text_raw(conn) -> str:
    dialogue_items = getattr(getattr(conn, "dialogue", None), "dialogue", [])
    for item in reversed(dialogue_items):
        if getattr(item, "role", "") != "assistant":
            continue
        content = getattr(item, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _get_recent_dialogue_text(
    conn,
    *,
    roles: tuple[str, ...] = ("assistant",),
    limit: int = 3,
) -> str:
    dialogue_items = getattr(getattr(conn, "dialogue", None), "dialogue", [])
    normalized_roles = {
        str(role or "").strip().lower() for role in roles if str(role or "").strip()
    }
    if not normalized_roles:
        return ""

    texts = []
    for item in reversed(dialogue_items):
        role = str(getattr(item, "role", "") or "").strip().lower()
        if role not in normalized_roles:
            continue
        content = getattr(item, "content", "")
        if not isinstance(content, str):
            continue
        normalized = textUtils.normalize_spoken_text(content)
        if not normalized:
            continue
        texts.append(normalized)
        if len(texts) >= max(1, int(limit or 1)):
            break
    texts.reverse()
    return " ".join(texts).strip()


def _get_recent_assistant_text(conn, limit: int = 3) -> str:
    return _get_recent_dialogue_text(conn, roles=("assistant",), limit=limit)


def _get_recent_user_text(conn, limit: int = 3) -> str:
    return _get_recent_dialogue_text(conn, roles=("user",), limit=limit)


def _extract_sample_photo_name_fixed(text: str) -> str:
    src = text or ""
    if not src:
        return ""

    patterns = (
        r"([0-9]+鍙锋牱鍝?",
        r"(鏍峰搧[0-9]+)",
        r"([涓€浜屼笁鍥涗簲鍏竷鍏節鍗佺櫨涓+鍙锋牱鍝?",
    )
    for pattern in patterns:
        match = re.search(pattern, src)
        if match:
            return match.group(1)
    return ""


def _looks_like_question_reply_fixed(text: str) -> bool:
    if not text:
        return False
    if text.endswith(("鍚?, "涔?, "鍛?, "鍢?)):
        return True
    question_tokens = (
        "鍙笉鍙互",
        "鑳戒笉鑳?,
        "琛屼笉琛?,
        "瑕佷笉瑕?,
        "鏄笉鏄?,
        "涓轰粈涔?,
        "鎬庝箞",
        "濡備綍",
    )
    return _contains_any(text, question_tokens)


def _is_affirmative_short_reply_fixed(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    negative_tokens = (
        "涓嶅彲浠?,
        "涓嶈",
        "鍒媿",
        "涓嶆媿",
        "鍏堝埆",
        "涓嶈兘鎷?,
        "涓嶈鎷?,
        "鍒幇鍦ㄦ媿",
    )
    if _contains_any(norm, negative_tokens):
        return False

    if norm in {
        "濂?,
        "濂界殑",
        "濂藉晩",
        "濂藉憖",
        "鍙互",
        "鍙互鐨?,
        "鍙互鎷?,
        "鎷嶅惂",
        "鎷?,
        "寮€濮嬫媿",
        "琛?,
        "琛岀殑",
        "琛屽晩",
        "鍡?,
        "鍡棷",
        "鏄?,
        "瀵?,
        "娌￠棶棰?,
        "鍚屾剰",
        "鍏佽",
        "鍑嗗濂戒簡",
        "鎴戝噯澶囧ソ浜?,
    }:
        return True

    if _looks_like_question_reply_fixed(norm):
        return False

    affirmative_tokens = (
        "鍙互鎷嶇収",
        "鐜板湪鍙互鎷嶇収",
        "鍙互鎷嶄簡",
        "鐜板湪鍙互鎷?,
        "鐜板湪鍙互浜?,
        "鍙互浜?,
        "鎷嶇収鍚?,
        "鎷嶄竴寮犲惂",
        "鎷嶄竴涓嬪惂",
        "鐩存帴鎷嶅惂",
        "娌￠棶棰樻媿",
        "鍚屾剰鎷?,
    )
    if _contains_any(norm, affirmative_tokens):
        return True

    affirmative_prefixes = (
        "濂?,
        "濂界殑",
        "濂藉晩",
        "濂藉憖",
        "鍙互",
        "鍙互鐨?,
        "鍙互鍛€",
        "鍙互鍠?,
        "琛?,
        "琛岀殑",
        "琛屽晩",
        "琛屽憖",
        "鍡?,
        "鍡棷",
        "瀵?,
        "鏄?,
        "娌￠棶棰?,
        "褰撶劧鍙互",
        "鍚屾剰",
        "鍏佽",
        "鍑嗗濂戒簡",
        "鎴戝噯澶囧ソ浜?,
    )
    return _starts_with_any(norm, affirmative_prefixes)


def _is_negative_short_reply_fixed(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if norm in {
        "涓嶈",
        "鍏堝埆",
        "鍒媿",
        "涓嶆媿",
        "杩樻病鍑嗗濂?,
        "娌″噯澶囧ソ",
        "绛夌瓑",
        "绛変竴涓?,
        "鏆傛椂涓嶈",
        "涓嶅彲浠?,
        "鍙栨秷",
    }:
        return True

    if not norm or len(norm) > 12:
        return False

    negative_tokens = (
        "涓嶅彲浠ユ媿鐓?,
        "鐜板湪涓嶅彲浠ユ媿鐓?,
        "杩樹笉鍙互鎷嶇収",
        "杩樹笉鑳芥媿鐓?,
        "鍏堝埆鎷嶇収",
    )
    return _contains_any(norm, negative_tokens)


def _assistant_is_waiting_for_photo_permission_fixed(conn) -> bool:
    last_text = _normalize_text_for_match(_get_last_assistant_text_raw(conn))
    if not last_text:
        return False

    photo_tokens = ("鎷嶇収", "鎷嶄竴寮?, "鎷嶄竴涓?, "鐓т竴涓?, "鎷嶆憚", "鐓х墖", "鎷嶅惂")
    if not _contains_any(last_text, photo_tokens):
        return False

    negative_prompt_tokens = (
        "涓嶈鎷?,
        "鍒媿",
        "鍏堝埆鎷?,
        "涓嶅彲浠ユ媿",
        "涓嶈兘鎷?,
        "杩樹笉鑳芥媿",
        "涓嶈鎷嶇収",
        "鍒媿鐓?,
        "鍏堝埆鎷嶇収",
        "涓嶅彲浠ユ媿鐓?,
        "涓嶈兘鎷嶇収",
        "杩樹笉鑳芥媿鐓?,
    )
    if _contains_any(last_text, negative_prompt_tokens):
        return False

    explicit_wait_tokens = (
        "寰楀埌鑲畾绛斿鍚庡啀鎷?,
        "纭鍚庡啀鎷?,
        "鍚屾剰鍚庡啀鎷?,
        "鍥炲鍙互鍐嶆媿",
        "鍏佽鎷嶇収鍚庡啀鍛婅瘔鎴?,
        "鍛婅瘔鎴戝彲浠ユ媿鐓?,
        "绛変綘鍏佽鍚庢垜鍐嶆媿",
        "鍐嶈涓€澹版媿鍚?,
        "璇翠竴澹版媿鍚?,
        "鍐嶈涓€閬嶆媿鍚?,
        "璇翠竴閬嶆媿鍚?,
        "鍐嶈涓€澹版媿鐓?,
        "璇翠竴澹版媿鐓?,
        "鍐嶈涓€閬嶆媿鐓?,
        "璇翠竴閬嶆媿鐓?,
        "鎷嶅惂",
        "鎷嶇収鍚?,
        "鎷嶄竴寮犲惂",
        "鎷嶄竴涓嬪惂",
        "鐩存帴鎷嶅惂",
        "寮€濮嬫媿鍚?,
    )
    if _contains_any(last_text, explicit_wait_tokens):
        return True

    if "鍛婅瘔鎴? in last_text and ("鍙互鎷嶇収" in last_text or "鎷嶅惂" in last_text):
        return True

    prompt_tokens = (
        "鍙互",
        "鑳?,
        "瑕佷笉瑕?,
        "瑕佷笉",
        "瑕佹垜",
        "甯綘",
        "缁欎綘",
        "璁╂垜",
        "鏄惁",
        "纭",
        "鍚屾剰",
    )
    question_tokens = ("鍚?, "涔?, "鍛?, "鏄惁", "鍙笉鍙互", "鑳戒笉鑳?, "瑕佷笉瑕?)
    return _contains_any(last_text, prompt_tokens) and _contains_any(
        last_text, question_tokens
    )


def _build_pending_server_photo_request_fixed(conn) -> dict:
    last_text = _get_last_assistant_text_raw(conn)
    sample_name = _extract_sample_photo_name_fixed(last_text)
    if not sample_name:
        sample_name = _extract_sample_photo_name_fixed(
            _get_recent_assistant_text(conn, limit=4)
        )
    if not sample_name:
        sample_name = _extract_sample_photo_name_fixed(
            _get_recent_user_text(conn, limit=4)
        )
    photo_name = sample_name
    if photo_name and not photo_name.endswith("\u7167\u7247"):
        photo_name = f"{photo_name}\u7167\u7247"
    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()

    if sample_name:
        question = f"璇锋媿鎽剓sample_name}褰撳墠鐘舵€佺殑鐓х墖銆?
    else:
        question = "璇锋媿鎽勫綋鍓嶆牱鍝佺殑鐓х墖銆?

    request = {
        "device_id": safe_device_id,
        "question": question,
    }
    if photo_name:
        request["photo_name"] = photo_name
    return request


def _resolve_photo_confirm_delay_seconds(conn) -> float:
    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    raw_value = shortcut_cfg.get("photo_confirm_delay_seconds", 3.0)
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 3.0


async def _maybe_wait_before_photo_capture(conn, source: str) -> None:
    delay_seconds = _resolve_photo_confirm_delay_seconds(conn)
    if delay_seconds <= 0:
        return
    conn.logger.bind(tag=TAG).info(
        "photo capture confirm delay: "
        f"source={source}, delay_seconds={delay_seconds:.2f}"
    )
    await asyncio.sleep(delay_seconds)


def _coerce_positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    if parsed <= 0:
        return int(default)
    return parsed


def _resolve_server_photo_timeout_settings(conn) -> tuple[int, int, float]:
    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    tool_timeout = _coerce_positive_int(
        shortcut_cfg.get("server_photo_timeout", shortcut_cfg.get("photo_timeout", 20)),
        20,
    )
    default_request_timeout = max(tool_timeout + 15, 60)
    request_timeout = _coerce_positive_int(
        shortcut_cfg.get(
            "server_photo_request_timeout",
            shortcut_cfg.get("photo_request_timeout", default_request_timeout),
        ),
        default_request_timeout,
    )
    request_timeout = max(request_timeout, tool_timeout + 5)
    raw_recovery_window = shortcut_cfg.get(
        "server_photo_recovery_window_seconds",
        6.0,
    )
    try:
        recovery_window_seconds = max(0.0, float(raw_recovery_window))
    except (TypeError, ValueError):
        recovery_window_seconds = 6.0
    return tool_timeout, request_timeout, recovery_window_seconds


def _prepare_server_photo_request_arguments(conn, arguments: dict) -> tuple[dict, float]:
    merged = dict(arguments or {})
    default_timeout, default_request_timeout, recovery_window_seconds = (
        _resolve_server_photo_timeout_settings(conn)
    )
    merged["timeout"] = _coerce_positive_int(merged.get("timeout"), default_timeout)
    merged["request_timeout"] = _coerce_positive_int(
        merged.get("request_timeout"),
        max(default_request_timeout, merged["timeout"] + 5),
    )
    merged["request_timeout"] = max(merged["request_timeout"], merged["timeout"] + 5)
    return merged, recovery_window_seconds


async def _fetch_latest_server_photo_meta(conn, device_id: str) -> dict:
    safe_device_id = str(device_id or "").strip()
    if not safe_device_id or _get_server_mcp_manager(conn) is None:
        return {}
    arguments = {"device_id": safe_device_id}
    try:
        result = await _execute_server_mcp_tool_direct(
            conn,
            "xiaozhi_get_latest_photo",
            arguments,
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).debug(
            f"latest server photo lookup failed: {exc}"
        )
        return {}

    payload = finalize_server_mcp_payload(
        result,
        tool_name="xiaozhi_get_latest_photo",
        arguments=arguments,
    )
    if isinstance(payload, dict) and payload.get("success") is False:
        return {}
    photo_meta = _extract_photo_result_meta(payload)
    if not photo_meta.get("found"):
        return {}
    return photo_meta


async def _update_completed_photo_record_after_retake(
    conn,
    payload,
    *,
    requested_arguments: dict | None = None,
) -> bool:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False

    inferred_step_id = _infer_photo_confirmation_step_id_from_context(
        conn,
        payload,
        requested_arguments=requested_arguments,
    )
    if not inferred_step_id:
        return False

    try:
        modifiable_payload, schema_payload = await asyncio.gather(
            _call_experiment_graph_tool_fast(
                conn,
                "get_modifiable_records",
                {"session_id": session_id, "step_id": inferred_step_id},
                priority="foreground",
            ),
            _call_experiment_graph_tool_fast(
                conn,
                "get_schema",
                {"session_id": session_id, "step_id": inferred_step_id},
                priority="foreground",
            ),
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"photo retake record lookup failed: step_id={inferred_step_id}, error={exc}"
        )
        return False

    modifiable_records = []
    body = _experiment_result_body(modifiable_payload)
    if isinstance(body, dict):
        modifiable_records = list(body.get("records") or [])
    completed_records = [
        item
        for item in modifiable_records
        if isinstance(item, dict) and str(item.get("source", "") or "").strip() == "completed"
    ]
    if not completed_records:
        conn.logger.bind(tag=TAG).info(
            f"photo retake record update skipped: no completed record for step_id={inferred_step_id}"
        )
        return False

    target_record = completed_records[-1]
    target_trial_number = target_record.get("trial_number")
    if not isinstance(target_trial_number, int) or target_trial_number < 1:
        target_trial_number = None

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    photo_meta = _extract_photo_result_meta(payload)
    photo_fields = _build_experiment_photo_writeback_fields(
        schema_by_name,
        photo_meta,
        step_meta={},
        missing_fields=[],
    )
    if not photo_fields:
        return False

    updated_any = False
    for field_name, field_value in photo_fields.items():
        try:
            result = await _call_experiment_graph_tool_fast(
                conn,
                "modify_record",
                {
                    "session_id": session_id,
                    "step_id": inferred_step_id,
                    "field_name": field_name,
                    "value": field_value,
                    "trial_number": target_trial_number,
                    "target": "completed",
                    "validate": True,
                },
                priority="foreground",
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"photo retake modify_record failed: step_id={inferred_step_id}, field={field_name}, error={exc}"
            )
            continue

        if bool(_experiment_result_body(result).get("ok")):
            updated_any = True
            continue

        conn.logger.bind(tag=TAG).warning(
            "photo retake modify_record rejected: "
            f"step_id={inferred_step_id}, field={field_name}, "
            f"message={_extract_experiment_result_message(result)}"
        )

    if updated_any:
        conn.logger.bind(tag=TAG).info(
            "photo retake updated completed record to latest photo: "
            f"step_id={inferred_step_id}, trial_number={target_trial_number or 1}, "
            f"file_name={photo_meta.get('file_name', '')}"
        )
    return updated_any


def _is_timeout_like_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    return "timeout" in normalized or "瓒呮椂" in normalized


def _photo_meta_is_new_since_baseline(
    latest_photo: dict,
    baseline_photo: dict,
    capture_started_at: float,
    requested_photo_name: str = "",
) -> bool:
    if not isinstance(latest_photo, dict) or not latest_photo.get("found"):
        return False

    latest_path = str(latest_photo.get("photo_path", "") or "").strip()
    latest_file_name = str(latest_photo.get("file_name", "") or "").strip()
    baseline_path = str((baseline_photo or {}).get("photo_path", "") or "").strip()
    requested_name = str(requested_photo_name or "").strip()

    try:
        latest_mtime = float(latest_photo.get("mtime", 0.0) or 0.0)
    except (TypeError, ValueError):
        latest_mtime = 0.0
    try:
        baseline_mtime = float((baseline_photo or {}).get("mtime", 0.0) or 0.0)
    except (TypeError, ValueError):
        baseline_mtime = 0.0

    if requested_name and latest_file_name and requested_name in latest_file_name:
        return True
    if latest_path and baseline_path:
        if latest_path != baseline_path:
            return True
        return latest_mtime > baseline_mtime + 1e-6
    if latest_path and not baseline_path:
        return latest_mtime >= capture_started_at - 2.0 if latest_mtime > 0 else False
    return latest_mtime > baseline_mtime + 1e-6 and latest_mtime >= capture_started_at - 2.0


async def _recover_server_photo_after_timeout(
    conn,
    baseline_photo: dict,
    arguments: dict,
    capture_started_at: float,
    recovery_window_seconds: float,
):
    safe_device_id = str((arguments or {}).get("device_id", "") or "").strip()
    if not safe_device_id:
        return None

    requested_photo_name = str((arguments or {}).get("photo_name", "") or "").strip()
    deadline = time.time() + max(0.0, float(recovery_window_seconds or 0.0))
    while True:
        latest_photo = await _fetch_latest_server_photo_meta(conn, safe_device_id)
        if _photo_meta_is_new_since_baseline(
            latest_photo,
            baseline_photo,
            capture_started_at,
            requested_photo_name=requested_photo_name,
        ):
            conn.logger.bind(tag=TAG).info(
                "recovered server photo after timeout: "
                f"device_id={safe_device_id}, file_name={latest_photo.get('file_name', '')}"
            )
            return {
                "success": True,
                "requested_photo_name": requested_photo_name,
                "photo_meta": latest_photo,
                "recovered_after_timeout": True,
            }
        if time.time() >= deadline:
            break
        await asyncio.sleep(min(0.5, max(0.0, deadline - time.time())))
    return None


def _compose_server_photo_timeout_reply() -> str:
    return "鎷嶇収杩欒竟瓒呮椂浜嗭紝鎴戣繕娌℃嬁鍒扮粨鏋溿€備綘鍙互绋嶅悗鍐嶈涓€娆℃媿鐓э紝鎴栬€呰鎴戞墦寮€鏈€杩戜竴寮犵収鐗囥€?


def _update_server_photo_confirmation_state(conn, filtered_text: str) -> None:
    if not _assistant_is_waiting_for_photo_permission_fixed(conn):
        return
    if _is_affirmative_short_reply_fixed(filtered_text):
        conn._server_photo_capture_granted = True
    elif _is_negative_short_reply_fixed(filtered_text):
        conn._server_photo_capture_granted = False


async def _execute_direct_photo_intent(
    conn,
    question: str,
    raw_tool_name: str,
    timeout: int,
) -> bool:
    mcp_client = getattr(conn, "mcp_client", None)
    if not mcp_client:
        speak_txt(conn, "\u8bbe\u5907\u8fd8\u6ca1\u51c6\u5907\u597d\u62cd\u7167\u3002")
        return True

    if not await mcp_client.is_ready():
        speak_txt(
            conn,
            "\u8bbe\u5907\u62cd\u7167\u529f\u80fd\u8fd8\u6ca1\u51c6\u5907\u597d\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002",
        )
        return True

    tool_name = sanitize_tool_name(raw_tool_name)
    if timeout <= 0:
        timeout = 45

    try:
        result = await call_mcp_tool(
            conn,
            mcp_client,
            tool_name,
            {"question": question},
            timeout=timeout,
            allow_unlisted=True,
            raw_tool_name=raw_tool_name,
        )
    except TimeoutError:
        speak_txt(conn, "\u62cd\u7167\u8d85\u65f6\u4e86\uff0c\u8bf7\u518d\u8bd5\u4e00\u6b21\u3002")
        return True
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"direct photo mcp failed: {e}")
        speak_txt(conn, f"\u62cd\u7167\u5931\u8d25\uff1a{e}")
        return True

    reply = _extract_direct_photo_reply(result) or "\u62cd\u597d\u4e86\u3002"
    speak_txt(conn, reply)
    return True


async def _execute_server_photo_intent(
    conn,
    arguments: dict,
    *,
    update_experiment_graph: bool = True,
    update_completed_photo_record_on_success: bool = False,
) -> bool:
    safe_device_id = str((arguments or {}).get("device_id", "") or "").strip()
    if not safe_device_id:
        speak_txt(conn, "\u8bbe\u5907\u8fde\u63a5\u4fe1\u606f\u7f3a\u5931\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    if _get_server_mcp_manager(conn) is None:
        speak_txt(conn, "\u62cd\u7167\u529f\u80fd\u8fd8\u6ca1\u51c6\u5907\u597d\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    call_arguments, recovery_window_seconds = _prepare_server_photo_request_arguments(
        conn,
        arguments,
    )
    baseline_photo = await _fetch_latest_server_photo_meta(conn, safe_device_id)
    capture_started_at = time.time()
    conn._server_photo_capture_granted = True
    try:
        result = await _execute_server_mcp_tool_direct(
            conn,
            "xiaozhi_take_photo",
            call_arguments,
        )
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"direct server photo mcp failed: {e}")
        speak_txt(conn, f"\u62cd\u7167\u5931\u8d25\uff1a{e}")
        return True
    finally:
        conn._server_photo_capture_granted = False

    payload = finalize_server_mcp_payload(
        result,
        tool_name="xiaozhi_take_photo",
        arguments=call_arguments,
    )
    sync_server_mcp_payload_state(
        conn,
        tool_name="xiaozhi_take_photo",
        payload=payload,
    )
    if isinstance(payload, dict) and payload.get("success") is False:
        msg = (
            str(payload.get("message", "")).strip()
            or _extract_text_from_result_payload(payload)
            or "\u62cd\u7167\u5931\u8d25\u4e86\u3002"
        )
        if _is_timeout_like_message(msg):
            recovered_payload = await _recover_server_photo_after_timeout(
                conn,
                baseline_photo,
                call_arguments,
                capture_started_at,
                recovery_window_seconds,
            )
            if recovered_payload is not None:
                payload = finalize_server_mcp_payload(
                    recovered_payload,
                    tool_name="xiaozhi_take_photo",
                    arguments=call_arguments,
                )
                sync_server_mcp_payload_state(
                    conn,
                    tool_name="xiaozhi_take_photo",
                    payload=payload,
                )
            else:
                speak_txt(conn, _compose_server_photo_timeout_reply())
                return True
        else:
            speak_txt(conn, msg)
            return True

    reply = build_server_mcp_spoken_response(
        "xiaozhi_take_photo",
        payload,
        default_reply="\u62cd\u597d\u4e86\u3002",
    ) or _extract_text_from_result_payload(payload) or "\u62cd\u597d\u4e86\u3002"
    if update_experiment_graph:
        try:
            local_reply = await _advance_photo_confirmation_step_locally(
                conn,
                payload,
                fallback_reply=reply,
                requested_arguments=call_arguments,
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"local server photo follow-up failed: {exc}"
            )
            local_reply = await _finalize_photo_followup_without_graph_advance(
                conn,
                str(getattr(conn, "experiment_session_id", "") or "").strip(),
                payload,
                requested_arguments=call_arguments,
                confirmation_reply=reply,
                reason="photo_followup_exception",
            )
    else:
        local_reply = reply
        if update_completed_photo_record_on_success:
            try:
                await _update_completed_photo_record_after_retake(
                    conn,
                    payload,
                    requested_arguments=call_arguments,
                )
            except Exception as exc:
                conn.logger.bind(tag=TAG).warning(
                    f"photo retake completed-record update failed: {exc}"
                )

    if hasattr(conn, "enrich_latest_clean_user_utterance_snapshot"):
        try:
            conn.enrich_latest_clean_user_utterance_snapshot()
        except Exception:
            pass

    if local_reply:
        speak_txt(conn, local_reply)
    return True


async def handle_pending_direct_photo_confirmation(
    conn, original_text: str, filtered_text: str
) -> bool:
    pending = getattr(conn, "_pending_direct_photo", None)
    if not isinstance(pending, dict):
        return False

    if _is_negative_short_reply_fixed(filtered_text):
        await send_stt_message(conn, original_text)
        conn.client_abort = False
        conn.sentence_id = str(uuid.uuid4().hex)
        conn.dialogue.put(Message(role="user", content=original_text))
        conn._pending_direct_photo = None
        speak_txt(conn, "\u597d\uff0c\u90a3\u6211\u5148\u4e0d\u62cd\u3002")
        return True

    if not _is_affirmative_short_reply_fixed(filtered_text):
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))
    conn._pending_direct_photo = None
    await _maybe_wait_before_photo_capture(conn, "direct_photo_confirmation")
    return await _execute_direct_photo_intent(
        conn,
        pending.get("question", "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1"),
        pending.get("raw_tool_name", "self.camera.take_photo"),
        int(pending.get("timeout", 45)),
    )


async def handle_pending_server_photo_confirmation(
    conn, original_text: str, filtered_text: str
) -> bool:
    if getattr(conn, "_pending_direct_photo", None):
        return False

    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    if shortcut_cfg.get("enable_server_photo_confirmation_direct", True) is False:
        return False

    if not _assistant_is_waiting_for_photo_permission_fixed(conn):
        return False

    if _is_negative_short_reply_fixed(filtered_text):
        conn._server_photo_capture_granted = False
        await send_stt_message(conn, original_text)
        conn.client_abort = False
        conn.sentence_id = str(uuid.uuid4().hex)
        conn.dialogue.put(Message(role="user", content=original_text))
        speak_txt(conn, "\u597d\uff0c\u90a3\u6211\u5148\u4e0d\u62cd\u3002")
        return True

    if not _is_affirmative_short_reply_fixed(filtered_text):
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))
    pending_request = _build_pending_server_photo_request_fixed(conn)
    if pending_request.get("photo_name"):
        pending_request["append_timestamp"] = True
    conn.logger.bind(tag=TAG).info("confirmed pending server photo capture, executing xiaozhi_take_photo directly")
    await _maybe_wait_before_photo_capture(conn, "server_photo_confirmation")
    return await _execute_server_photo_intent(
        conn,
        pending_request,
    )


async def handle_direct_photo_navigation_intent(
    conn, original_text: str, filtered_text: str
) -> bool:
    nav_type = _classify_photo_nav_command(filtered_text)
    if not nav_type:
        return False

    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    if shortcut_cfg.get("enable_photo_navigation_direct", True) is False:
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))

    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()
    if not safe_device_id:
        speak_txt(conn, "\u8bbe\u5907\u8fde\u63a5\u4fe1\u606f\u7f3a\u5931\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    if _get_server_mcp_manager(conn) is None:
        speak_txt(conn, "\u56fe\u7247\u9884\u89c8\u529f\u80fd\u8fd8\u6ca1\u51c6\u5907\u597d\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    try:
        if nav_type == "previous":
            tool_name = str(
                shortcut_cfg.get(
                    "preview_previous_tool_name",
                    "xiaozhi_preview_previous_photo",
                )
            ).strip() or "xiaozhi_preview_previous_photo"
            result = await _execute_server_mcp_tool_direct(
                conn,
                tool_name,
                {"device_id": safe_device_id},
            )
            default_reply = "\u5df2\u7ecf\u5207\u5230\u4e0a\u4e00\u5f20\u4e86\u3002"
        else:
            tool_name = str(
                shortcut_cfg.get(
                    "preview_latest_tool_name",
                    "xiaozhi_preview_local_file",
                )
            ).strip() or "xiaozhi_preview_local_file"
            result = await _execute_server_mcp_tool_direct(
                conn,
                tool_name,
                {"device_id": safe_device_id, "photo_index": 0},
            )
            default_reply = "\u5df2\u7ecf\u6253\u5f00\u6700\u8fd1\u4e00\u5f20\u7167\u7247\u4e86\u3002"
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"direct photo navigation failed: {e}")
        speak_txt(conn, f"\u6253\u5f00\u7167\u7247\u5931\u8d25\uff1a{e}")
        return True

    payload = finalize_server_mcp_payload(
        result,
        tool_name=tool_name,
        arguments={"device_id": safe_device_id, "photo_index": 0}
        if nav_type != "previous"
        else {"device_id": safe_device_id},
    )
    sync_server_mcp_payload_state(
        conn,
        tool_name=tool_name,
        payload=payload,
    )
    if isinstance(payload, dict) and payload.get("success") is False:
        msg = str(payload.get("message", "")).strip() or "\u6253\u5f00\u7167\u7247\u5931\u8d25\u3002"
        speak_txt(conn, msg)
        return True

    reply = build_server_mcp_spoken_response(
        tool_name,
        payload,
        default_reply=default_reply,
    ) or _extract_text_from_result_payload(payload) or default_reply
    speak_txt(conn, reply)
    return True


async def handle_direct_photo_intent(conn, original_text: str, filtered_text: str) -> bool:
    if not _is_direct_photo_command(filtered_text):
        return False

    if _assistant_is_waiting_for_photo_permission_fixed(conn) and (
        _is_affirmative_short_reply_fixed(filtered_text)
        or _is_negative_short_reply_fixed(filtered_text)
    ):
        return False

    shortcut_cfg = conn.config.get("device_mcp_shortcuts", {}) or {}
    if shortcut_cfg.get("enable_photo_direct", True) is False:
        return False

    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))

    default_question = str(
        shortcut_cfg.get(
            "default_photo_question",
            "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1",
        )
    ).strip() or "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1"
    question = _build_direct_photo_question(original_text, default_question)
    sample_photo_name = _extract_sample_photo_stem_from_command(filtered_text)
    if sample_photo_name:
        question = f"\u8bf7\u62cd\u6444{sample_photo_name}\u3002"

    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()
    if _get_server_mcp_manager(conn) is not None and safe_device_id:
        request = {
            "device_id": safe_device_id,
            "question": question,
        }
        is_sample_retake = _is_sample_photo_retake_command(filtered_text)
        if sample_photo_name:
            request["photo_name"] = sample_photo_name
            request["append_timestamp"] = True
        if is_sample_retake:
            conn.logger.bind(tag=TAG).info(
                "sample photo retake requested; taking a new timestamped photo without graph rollback"
            )
        await _maybe_wait_before_photo_capture(conn, "direct_photo_command")
        return await _execute_server_photo_intent(
            conn,
            request,
            update_experiment_graph=False,
            update_completed_photo_record_on_success=bool(
                is_sample_retake and sample_photo_name
            ),
        )

    raw_tool_name = str(
        shortcut_cfg.get("take_photo_tool_name", "self.camera.take_photo")
    ).strip() or "self.camera.take_photo"
    timeout = int(shortcut_cfg.get("photo_timeout", 45))
    await _maybe_wait_before_photo_capture(conn, "direct_photo_command")
    return await _execute_direct_photo_intent(
        conn,
        question,
        raw_tool_name,
        timeout,
    )


async def process_intent_result(conn, intent_result, original_text):
    """????????"""
    try:
        # ????????JSON
        intent_data = json.loads(intent_result)

        # ?????function_call
        if "function_call" in intent_data:
            # ??????????function_call
            conn.logger.bind(tag=TAG).debug(
                f"???function_call???????: {intent_data['function_call']['name']}"
            )
            function_name = intent_data["function_call"]["name"]
            if function_name == "continue_chat":
                return False

            if function_name == "result_for_context":
                await send_stt_message(conn, original_text)
                conn.client_abort = False

                def process_context_result():
                    conn.dialogue.put(Message(role="user", content=original_text))

                    from core.utils.current_time import get_current_time_info

                    (
                        current_time,
                        today_date,
                        today_weekday,
                        lunar_date,
                    ) = get_current_time_info()

                    # ???????????
                    context_prompt = (
                        f"?????{current_time}\n"
                        f"?????{today_date} ({today_weekday})\n"
                        f"?????{lunar_date}\n\n"
                        f"???????????????{original_text}"
                    )

                    response = conn.intent.replyResult(context_prompt, original_text)
                    speak_txt(conn, response)

                conn.executor.submit(process_context_result)
                return True

            function_args = {}
            if "arguments" in intent_data["function_call"]:
                function_args = intent_data["function_call"]["arguments"]
                if function_args is None:
                    function_args = {}
            # ???????????JSON
            if isinstance(function_args, dict):
                function_args = json.dumps(function_args)

            function_call_data = {
                "name": function_name,
                "id": str(uuid.uuid4().hex),
                "arguments": function_args,
            }

            await send_stt_message(conn, original_text)
            conn.client_abort = False

            # ??executor???????????
            def process_function_call():
                conn.dialogue.put(Message(role="user", content=original_text))

                # ?????????????????
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        conn.func_handler.handle_llm_function_call(
                            conn, function_call_data
                        ),
                        conn.loop,
                    ).result()
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"??????: {e}")
                    result = ActionResponse(
                        action=Action.ERROR, result=str(e), response=str(e)
                    )

                if result:
                    if result.action == Action.RESPONSE:  # ??????
                        text = result.response
                        if text is not None:
                            speak_txt(conn, text)
                    elif result.action == Action.REQLLM:  # ????????llm????
                        text = result.result
                        conn.dialogue.put(Message(role="tool", content=text))
                        llm_result = conn.intent.replyResult(text, original_text)
                        if llm_result is None:
                            llm_result = text
                        speak_txt(conn, llm_result)
                    elif (
                        result.action == Action.NOTFOUND
                        or result.action == Action.ERROR
                    ):
                        text = result.result
                        if text is not None:
                            speak_txt(conn, text)
                    elif function_name != "play_music":
                        # For backward compatibility with original code
                        # ???????????
                        text = result.response
                        if text is None:
                            text = result.result
                        if text is not None:
                            speak_txt(conn, text)

            # ???????????
            conn.executor.submit(process_function_call)
            return True
        return False
    except json.JSONDecodeError as e:
        conn.logger.bind(tag=TAG).error(f"?????????: {e}")
        return False


def speak_txt(conn, text):
    text = textUtils.prepare_runtime_spoken_text_for_conn(conn, text)
    if not text:
        return

    # ????
    conn.tts_MessageText = text

    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.FIRST,
            content_type=ContentType.ACTION,
        )
    )
    conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
    conn.tts.tts_text_queue.put(
        TTSMessageDTO(
            sentence_id=conn.sentence_id,
            sentence_type=SentenceType.LAST,
            content_type=ContentType.ACTION,
        )
    )
    conn.dialogue.put(Message(role="assistant", content=text))
    if hasattr(conn, "append_experiment_interaction_log"):
        try:
            conn.append_experiment_interaction_log(
                "ASSISTANT",
                text,
                source="speak_txt",
            )
        except Exception:
            pass
