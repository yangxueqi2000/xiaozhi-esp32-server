import json
import re
import uuid
import asyncio
from pathlib import Path

import yaml
from core.utils.dialogue import Message
from core.providers.tts.dto.dto import ContentType
from core.handle.helloHandle import checkWakeupWords
from plugins_func.register import Action, ActionResponse
from core.handle.sendAudioHandle import send_stt_message
from core.utils import textUtils
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
    # 预处理输入文本，处理可能的JSON格式
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            parsed_data = json.loads(text)
            if isinstance(parsed_data, dict) and "content" in parsed_data:
                text = parsed_data["content"]  # 提取content用于意图分析
                conn.current_speaker = parsed_data.get("speaker")  # 保留说话人信息
    except (json.JSONDecodeError, TypeError):
        pass

    # 检查是否有明确的退出命令
    _, filtered_text = remove_punctuation_and_length(text)
    if await check_direct_exit(conn, filtered_text):
        return True

    # 检查是否是唤醒词
    if await checkWakeupWords(conn, filtered_text):
        return True

    if await handle_pending_direct_photo_confirmation(conn, text, filtered_text):
        return True

    if await handle_pending_server_photo_confirmation(conn, text, filtered_text):
        return True

    _update_server_photo_confirmation_state(conn, filtered_text)

    # Fast path: photo-navigation commands go directly to MCP (no Codex).
    if await handle_direct_photo_navigation_intent(conn, text, filtered_text):
        return True

    # Fast path: take-photo commands go directly to MCP (no Codex).
    if await handle_direct_photo_intent(conn, text, filtered_text):
        return True

    # Fast path: short experiment control utterances go directly to
    # experiment flow handling instead of a full Codex turn.
    if await handle_experiment_control_fast_intent(conn, text, filtered_text):
        return True

    if conn.intent_type == "function_call":
        # 使用支持function calling的聊天方法,不再进行意图分析
        return False
    # 使用LLM进行意图分析
    intent_result = await analyze_intent_with_llm(conn, text)
    if not intent_result:
        return False
    # 会话开始时生成sentence_id
    conn.sentence_id = str(uuid.uuid4().hex)
    # 处理各种意图
    return await process_intent_result(conn, intent_result, text)


async def check_direct_exit(conn, text):
    """检查是否有明确的退出命令"""
    _, text = remove_punctuation_and_length(text)
    cmd_exit = conn.cmd_exit
    for cmd in cmd_exit:
        if text == cmd:
            conn.logger.bind(tag=TAG).info(f"识别到明确的退出命令: {text}")
            await send_stt_message(conn, text)
            await conn.close()
            return True
    return False


async def analyze_intent_with_llm(conn, text):
    """使用LLM分析用户意图"""
    if not hasattr(conn, "intent") or not conn.intent:
        conn.logger.bind(tag=TAG).warning("意图识别服务未初始化")
        return None

    # 对话历史记录
    dialogue = conn.dialogue
    try:
        intent_result = await conn.intent.detect_intent(conn, dialogue.dialogue, text)
        return intent_result
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"意图识别失败: {str(e)}")

    return None


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


def _starts_with_any(text: str, words) -> bool:
    return any(text.startswith(w) for w in words)


def _ends_with_any(text: str, words) -> bool:
    return any(text.endswith(w) for w in words)


def _matches_any_pattern(text: str, patterns) -> bool:
    return any(pattern.fullmatch(text) for pattern in patterns)


def _looks_like_question_reply(text: str) -> bool:
    if not text:
        return False
    if text.endswith(("吗", "么", "嘛", "呢")):
        return True
    question_tokens = (
        "可不可以",
        "能不能",
        "行不行",
        "要不要",
        "是不是",
        "为什么",
        "怎么",
        "如何",
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
    return _merge_experiment_step_meta(primary, fallback)


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
    value = re.sub(r"^[已请需]+", "", value)
    value = value.replace("是否", "")
    return value.strip("，。；;: ")


_CHINESE_DIGIT_MAP = str.maketrans(
    {
        "零": "0",
        "一": "1",
        "二": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }
)


def _normalize_confirmation_signature(text: str) -> str:
    norm = _normalize_text_for_match(text)
    if not norm:
        return ""
    norm = norm.translate(_CHINESE_DIGIT_MAP)
    norm = norm.replace("->", "-")
    norm = norm.replace("至", "到")
    norm = re.sub(r"([0-9]+)到([0-9]+)", r"\1-\2", norm)
    norm = norm.replace("已按", "按")
    norm = norm.replace("已经", "已")
    norm = norm.replace("完成了", "完成")
    norm = norm.replace("加入了", "加入")
    return norm


def _confirmation_char_ngrams(text: str, n: int = 2) -> set:
    clean = re.sub(r"[\s，。；：、,.!?？]", "", text)
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
        return "继续前还差这一步的关键信息，你补一句当前结果就行。"
    if len(prompts) > 3:
        return "继续前还差这一步的一整组关键记录。你把当前这一步要记录的数据按顺序告诉我就行。"
    if len(prompts) == 1:
        return f"继续前还差这一步的一个确认：{prompts[0]}。你补一句这个就行。"
    if len(prompts) == 2:
        joined = f"{prompts[0]}，还有 {prompts[1]}"
    else:
        joined = "、".join(prompts[:3])
    return f"继续前还差这几个确认：{joined}。你补一句这几个结果就行。"


def _first_nonempty_text(*values) -> str:
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text:
            return text
    return ""


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

    if title and title not in instruction:
        core = f"{title}。{instruction}"
    else:
        core = instruction

    if mode == "repeat":
        parts = [f"当前这一步：{core}。"]
        if safety:
            parts.append(f"注意{safety}。")
        elif tip:
            parts.append(f"{tip}。")
        parts.append("做好后告诉我。")
        return "".join(parts)

    if mode == "next":
        parts = [f"接下来做这一步：{core}。"]
    else:
        parts = [f"现在做这一步：{core}。"]

    if safety:
        parts.append(f"注意{safety}。")
    elif tip:
        parts.append(f"{tip}。")
    parts.append("做好后告诉我。")
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
        if title.startswith("《") and title.endswith("》"):
            formatted_title = title
        else:
            formatted_title = f"《{title.strip('《》')}》"
        return f"今天我们做{formatted_title}。你准备好开始了吗？"
    return "今天我们做当前实验。你准备好开始了吗？"


def _is_explicit_experiment_start_request(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    explicit_tokens = (
        "开始流程",
        "开始当前实验流程",
        "开始当前实验",
        "准备开始",
        "准备开始实验",
        "准备开始流程",
        "开始今天的实验",
        "开始今天实验",
        "开始本次实验",
        "开始这个实验",
        "开始实验",
        "开始做实验",
        "开始做今天的实验",
        "开始今天做的实验",
    )
    return _contains_any(norm, explicit_tokens)


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
        "为什么",
        "原理",
        "依据",
        "详细",
        "注意事项",
        "多少",
        "浓度",
        "体积",
        "怎么配",
        "怎么算",
        "公式",
        "字段",
        "schema",
        "参考",
        "后面所有",
        "全部步骤",
        "整个实验",
        "完整流程",
    )
    return _contains_any(norm, detail_tokens)


def _assistant_waiting_for_step_completion(conn) -> bool:
    last_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not last_text:
        return False
    tokens = (
        "做好后告诉我",
        "做好告诉我",
        "做完告诉我",
        "完成后告诉我",
        "完成了告诉我",
        "做完了告诉我",
        "测完告诉我",
        "扫完告诉我",
        "结束后告诉我",
        "加完告诉我",
        "加好了告诉我",
        "拍完告诉我",
        "拍好了告诉我",
        "看完告诉我",
        "观察完告诉我",
        "记录完告诉我",
    )
    completion_markers = (
        "做好",
        "做完",
        "完成",
        "测完",
        "扫完",
        "结束",
        "加完",
        "加好",
        "拍完",
        "拍好",
        "看完",
        "观察完",
        "记录完",
    )
    return _contains_any(last_text, tokens) or (
        "告诉我" in last_text and _contains_any(last_text, completion_markers)
    )


def _assistant_waiting_for_step_start(conn) -> bool:
    last_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not last_text:
        return False
    tokens = (
        "准备好开始了吗",
        "准备好了吗",
        "可以开始了吗",
        "现在开始吗",
        "要开始了吗",
        "要不要开始",
    )
    return _contains_any(last_text, tokens)


def _is_explicit_ready_to_start_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    ready_tokens = (
        "准备好了",
        "我准备好了",
        "已经准备好了",
        "可以开始",
        "可以开始了",
        "开始吧",
        "开始做吧",
        "ready",
    )
    return _contains_any(norm, ready_tokens)


def _compose_waiting_ready_reply() -> str:
    return "你准备好后告诉我准备好了，我再带你开始第一步。"


PURE_SHORT_COMPLETION_PATTERNS = (
    re.compile(
        r"^(?:(?:我|这步|这一步|当前步骤|当前这步|本步|这轮|已经|已|都|就|现在|目前|刚刚|这里|这边|样品)){0,3}"
        r"(?:加|装|配|放|做|弄|拍|扫|看|测|量|记|写|填|观察|确认|核对|处理|准备|调|搅拌|滴加|记录|补记)?"
        r"(?:好|完|成)(?:了|啦)$"
    ),
    re.compile(
        r"^(?:(?:我|这步|这一步|当前步骤|当前这步|本步|这轮|已经|已|都|就|现在|目前|刚刚|这里|这边|样品)){0,3}"
        r"(?:搞定|结束|齐活|妥)(?:了|啦)?$"
    ),
)


def _looks_like_pure_short_completion_control(norm: str) -> bool:
    if not norm or len(norm) > 18:
        return False
    return _matches_any_pattern(norm, PURE_SHORT_COMPLETION_PATTERNS)


def _classify_short_experiment_control(conn, filtered_text: str) -> str:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return ""
    if len(norm) > 24:
        return ""
    if _looks_like_experiment_detail_request(norm):
        return ""

    clarify_tokens = (
        "没听懂",
        "没听清",
        "再说一遍",
        "重说一遍",
        "重新说",
        "重复一下",
        "再讲一遍",
        "再说下",
        "当前步骤是什么",
        "这步是什么",
        "这步怎么做",
        "什么意思",
    )
    if _contains_any(norm, clarify_tokens):
        return "repeat"

    advance_tokens = (
        "继续下一步",
        "下一步",
        "往下走",
        "往后走",
        "做完了",
        "做好了",
        "完成了",
        "已完成",
        "当前步骤已完成",
        "这步完成了",
        "这一步完成了",
        "都做好了",
        "都做完了",
    )
    if _contains_any(norm, advance_tokens):
        return "advance"

    if _assistant_waiting_for_step_completion(conn) and _looks_like_pure_short_completion_control(
        norm
    ):
        return "advance"

    ready_tokens = (
        "准备好了",
        "我准备好了",
        "可以开始",
        "开始吧",
        "开始",
        "ready",
    )
    if _contains_any(norm, ready_tokens):
        return "guide"

    neutral_ack_tokens = (
        "好了",
        "可以了",
        "行了",
        "好啦",
        "ok了",
    )
    if norm in neutral_ack_tokens or _ends_with_any(norm, neutral_ack_tokens):
        if _assistant_waiting_for_step_completion(conn):
            return "advance"
        if _assistant_waiting_for_step_start(conn):
            return "guide"

    if norm in {"继续", "继续吧"}:
        return "guide" if _assistant_waiting_for_step_start(conn) else "advance"

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
    loaded_meta = _merge_experiment_step_meta(
        _extract_experiment_step_meta(step_payload),
        _extract_experiment_step_meta(progress_payload),
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
    return _merge_experiment_step_meta(
        _extract_experiment_step_meta(step_payload),
        _extract_experiment_step_meta(progress_payload),
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
    autofill = {}
    unsafe_tokens = (
        "photo",
        "图片",
        "照片",
        "拍照",
        "颜色",
        "现象",
        "观察",
        "observed",
        "tyndall",
        "吸光",
        "absorbance",
        "波长",
        "wavelength",
        "lambda",
        "kinetics",
        "rate",
        "constant",
        "csv",
        "file",
        "path",
        "task",
        "导出",
        "报告",
        "pdf",
    )
    safe_name_tokens = (
        "added",
        "loaded",
        "cleaned",
        "prepared",
        "confirmed",
        "mixed",
        "started",
        "ready",
        "placed",
        "labeled",
        "stir",
        "returned",
        "completed",
        "setup",
    )
    safe_desc_tokens = (
        "已",
        "完成",
        "确认",
        "准备好",
        "就位",
        "清洗",
        "混合均匀",
        "加入",
        "启动",
    )

    for field_name in missing_fields or []:
        field = schema_by_name.get(field_name, {})
        type_text = str(field.get("type", "")).strip().lower()
        if type_text not in {"bool", "boolean"}:
            continue

        description = str(field.get("description", "")).strip()
        haystack = f"{field_name} {description}".lower()
        if _contains_match_token(haystack, unsafe_tokens):
            continue

        if allow_confirmation_autofill:
            autofill[field_name] = True
            continue

        if not _contains_match_token(haystack, safe_name_tokens) and not _contains_match_token(
            description,
            safe_desc_tokens,
        ):
            continue

        autofill[field_name] = True
    return autofill


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
        "没做",
        "还没做",
        "还没有做",
        "没做好",
        "还没做好",
        "没完成",
        "还没完成",
        "先别",
        "不要",
        "不行",
        "没加",
        "还没加",
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

    schema_by_name = _extract_experiment_schema_view(schema_payload)
    if not _looks_like_confirmation_field_statement(
        filtered_text,
        missing_fields,
        schema_by_name,
    ):
        return False

    conn.logger.bind(tag=TAG).info(
        "experiment confirmation semantic fast path hit: "
        f"text={filtered_text}, missing_fields={missing_fields}"
    )

    try:
        reply = await _advance_experiment_step_fast(conn, session_id)
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment confirmation semantic fast advance failed: {exc}"
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


def _build_experiment_photo_writeback_fields(schema_by_name: dict, photo_meta: dict) -> dict:
    result = {}
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
    haystack = _normalize_text_for_match(
        " ".join(
            str(
                step_meta.get(key, "")
                or ""
            ).strip()
            for key in ("title", "instruction", "description", "tip")
        )
    )
    if not haystack:
        return False
    return _contains_any(haystack, ("拍照", "照片", "拍一下", "拍一张", "拍摄"))


async def _advance_photo_confirmation_step_locally(
    conn,
    payload,
    fallback_reply: str = "",
) -> str:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return fallback_reply or "拍好了。"

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
    photo_fields = _build_experiment_photo_writeback_fields(schema_by_name, photo_meta)
    missing_fields = list((current_progress or {}).get("missing_fields") or [])
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
        return fallback_reply or "拍好了。"

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
        return reply or fallback_reply or "拍好了。"

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
        return reply or fallback_reply or "拍好了。"

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
        return reply or fallback_reply or "拍好了。"

    next_meta = await _refresh_experiment_step_cache(conn, session_id)
    reply = _compose_experiment_step_reply(next_meta, mode="next")
    if reply:
        return reply
    return fallback_reply or "拍照已经完成，继续做当前下一步。"


async def _start_direct_intent_turn(conn, original_text: str):
    await send_stt_message(conn, original_text)
    conn.client_abort = False
    conn.sentence_id = str(uuid.uuid4().hex)
    conn.dialogue.put(Message(role="user", content=original_text))


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
    allow_confirmation_autofill = _step_supports_confirmation_autofill(step_payload)

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
    autofill_fields = _build_experiment_autofill_fields(
        schema_by_name,
        missing_fields,
        allow_confirmation_autofill=allow_confirmation_autofill,
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

    if _is_explicit_experiment_start_request(filtered_text):
        experiment_title = await _load_experiment_overview_title(conn)
        reply = _compose_experiment_start_reply(experiment_title, "")
        reply = _prepare_fastpath_spoken_reply(reply)
        if not reply:
            return False
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, reply)
        return True

    action = _classify_short_experiment_control(conn, filtered_text)
    if not action:
        return await _handle_confirmation_step_semantic_fast_intent(
            conn,
            original_text,
            filtered_text,
        )

    conn.logger.bind(tag=TAG).info(
        f"experiment control fast path hit: action={action}, text={filtered_text}"
    )

    waiting_for_step_start = _assistant_waiting_for_step_start(conn)
    explicitly_ready_to_start = _is_explicit_ready_to_start_reply(filtered_text)

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

    if waiting_for_step_start and action in {"guide", "advance"} and not explicitly_ready_to_start:
        reply = _prepare_fastpath_spoken_reply(_compose_waiting_ready_reply())
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
            speak_txt(conn, reply)
            return True

        try:
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
        if hasattr(conn, "enrich_latest_clean_user_utterance_snapshot"):
            try:
                conn.enrich_latest_clean_user_utterance_snapshot()
            except Exception:
                pass
        speak_txt(conn, reply)
        return True

    return False


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
        or ""
    ).strip()
    requested_photo_name = str(
        photo_meta.get("requested_photo_name") or data.get("requested_photo_name") or ""
    ).strip()

    return {
        "found": bool(photo_meta.get("found", False)),
        "file_name": file_name,
        "photo_path": photo_path,
        "requested_photo_name": requested_photo_name,
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
        "不可以",
        "不要",
        "别拍",
        "不拍",
        "先别",
        "不能拍",
        "不让拍",
        "别现在拍",
    )
    if _contains_any(norm, negative_tokens):
        return False

    if norm in {
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "可以拍",
        "拍吧",
        "拍",
        "开始拍",
        "行",
        "行的",
        "行啊",
        "嗯",
        "嗯嗯",
        "是",
        "对",
        "没问题",
        "同意",
        "允许",
        "准备好了",
        "我准备好了",
    }:
        return True

    if _looks_like_question_reply(norm):
        return False

    affirmative_tokens = (
        "可以拍照",
        "现在可以拍照",
        "可以拍",
        "可以拍了",
        "现在可以拍",
        "现在可以了",
        "可以了",
        "拍照吧",
        "拍一张吧",
        "拍一下吧",
        "直接拍吧",
        "没问题拍",
        "同意拍",
    )
    if _contains_any(norm, affirmative_tokens):
        return True

    affirmative_prefixes = (
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "可以啊",
        "可以呀",
        "行",
        "行的",
        "行啊",
        "行呀",
        "嗯",
        "嗯嗯",
        "对",
        "是",
        "没问题",
        "当然可以",
        "同意",
        "允许",
        "准备好了",
        "我准备好了",
    )
    return _starts_with_any(norm, affirmative_prefixes)


def _is_negative_short_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if norm in {
        "不要",
        "先别",
        "别拍",
        "不拍",
        "还没准备好",
        "没准备好",
        "等等",
        "等一下",
        "暂时不要",
        "不可以",
        "取消",
    }:
        return True

    if not norm or len(norm) > 12:
        return False

    negative_tokens = (
        "不可以拍照",
        "现在不可以拍照",
        "还不可以拍照",
        "还不能拍照",
        "先别拍照",
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
        r"([0-9]+号样品)",
        r"(样品[0-9]+)",
        r"([一二三四五六七八九十]+号样品)",
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
        "拍照",
        "拍一张",
        "拍一下",
        "照一下",
        "照片",
    )
    if not _contains_any(last_text, photo_tokens):
        return False

    explicit_wait_tokens = (
        "得到肯定答复后再拍",
        "确认后再拍",
        "同意后再拍",
        "回复可以再拍",
    )
    if _contains_any(last_text, explicit_wait_tokens):
        return True

    prompt_tokens = (
        "可以",
        "能",
        "要不要",
        "要不",
        "要我",
        "帮你",
        "给你",
        "让我",
        "是否",
        "确认",
        "同意",
    )
    question_tokens = (
        "吗",
        "么",
        "嘛",
        "是否",
        "可不可以",
        "能不能",
        "要不要",
    )
    return _contains_any(last_text, prompt_tokens) and _contains_any(
        last_text, question_tokens
    )


def _build_pending_server_photo_request(conn) -> dict:
    last_text = textUtils.normalize_spoken_text(_get_last_assistant_text(conn))
    sample_name = _extract_sample_photo_name(last_text)
    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()

    if sample_name:
        question = f"请拍摄{sample_name}当前状态的照片。"
    else:
        question = "请拍摄当前样品的照片。"

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


def _get_recent_assistant_text(conn, limit: int = 3) -> str:
    dialogue_items = getattr(getattr(conn, "dialogue", None), "dialogue", [])
    texts = []
    for item in reversed(dialogue_items):
        if getattr(item, "role", "") != "assistant":
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


def _extract_sample_photo_name_fixed(text: str) -> str:
    src = text or ""
    if not src:
        return ""

    patterns = (
        r"([0-9]+号样品)",
        r"(样品[0-9]+)",
        r"([一二三四五六七八九十百两]+号样品)",
    )
    for pattern in patterns:
        match = re.search(pattern, src)
        if match:
            return match.group(1)
    return ""


def _looks_like_question_reply_fixed(text: str) -> bool:
    if not text:
        return False
    if text.endswith(("吗", "么", "呢", "嘛")):
        return True
    question_tokens = (
        "可不可以",
        "能不能",
        "行不行",
        "要不要",
        "是不是",
        "为什么",
        "怎么",
        "如何",
    )
    return _contains_any(text, question_tokens)


def _is_affirmative_short_reply_fixed(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    negative_tokens = (
        "不可以",
        "不要",
        "别拍",
        "不拍",
        "先别",
        "不能拍",
        "不让拍",
        "别现在拍",
    )
    if _contains_any(norm, negative_tokens):
        return False

    if norm in {
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "可以拍",
        "拍吧",
        "拍",
        "开始拍",
        "行",
        "行的",
        "行啊",
        "嗯",
        "嗯嗯",
        "是",
        "对",
        "没问题",
        "同意",
        "允许",
        "准备好了",
        "我准备好了",
    }:
        return True

    if _looks_like_question_reply_fixed(norm):
        return False

    affirmative_tokens = (
        "可以拍照",
        "现在可以拍照",
        "可以拍了",
        "现在可以拍",
        "现在可以了",
        "可以了",
        "拍照吧",
        "拍一张吧",
        "拍一下吧",
        "直接拍吧",
        "没问题拍",
        "同意拍",
    )
    if _contains_any(norm, affirmative_tokens):
        return True

    affirmative_prefixes = (
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "可以呀",
        "可以喔",
        "行",
        "行的",
        "行啊",
        "行呀",
        "嗯",
        "嗯嗯",
        "对",
        "是",
        "没问题",
        "当然可以",
        "同意",
        "允许",
        "准备好了",
        "我准备好了",
    )
    return _starts_with_any(norm, affirmative_prefixes)


def _is_negative_short_reply_fixed(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if norm in {
        "不要",
        "先别",
        "别拍",
        "不拍",
        "还没准备好",
        "没准备好",
        "等等",
        "等一下",
        "暂时不要",
        "不可以",
        "取消",
    }:
        return True

    if not norm or len(norm) > 12:
        return False

    negative_tokens = (
        "不可以拍照",
        "现在不可以拍照",
        "还不可以拍照",
        "还不能拍照",
        "先别拍照",
    )
    return _contains_any(norm, negative_tokens)


def _assistant_is_waiting_for_photo_permission_fixed(conn) -> bool:
    last_text = _normalize_text_for_match(_get_last_assistant_text_raw(conn))
    if not last_text:
        return False

    photo_tokens = ("拍照", "拍一张", "拍一下", "照一下", "照片")
    if not _contains_any(last_text, photo_tokens):
        return False

    explicit_wait_tokens = (
        "得到肯定答复后再拍",
        "确认后再拍",
        "同意后再拍",
        "回复可以再拍",
        "允许拍照后再告诉我",
        "告诉我可以拍照",
        "等你允许后我再拍",
    )
    if _contains_any(last_text, explicit_wait_tokens):
        return True

    if "告诉我" in last_text and "可以拍照" in last_text:
        return True

    prompt_tokens = (
        "可以",
        "能",
        "要不要",
        "要不",
        "要我",
        "帮你",
        "给你",
        "让我",
        "是否",
        "确认",
        "同意",
    )
    question_tokens = ("吗", "么", "呢", "是否", "可不可以", "能不能", "要不要")
    return _contains_any(last_text, prompt_tokens) and _contains_any(
        last_text, question_tokens
    )


def _build_pending_server_photo_request_fixed(conn) -> dict:
    last_text = _get_last_assistant_text_raw(conn)
    sample_name = _extract_sample_photo_name_fixed(last_text)
    safe_device_id = str(getattr(conn, "device_id", "") or "").strip()

    if sample_name:
        question = f"请拍摄{sample_name}当前状态的照片。"
    else:
        question = "请拍摄当前样品的照片。"

    request = {
        "device_id": safe_device_id,
        "question": question,
    }
    if sample_name:
        request["photo_name"] = sample_name
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


async def _execute_server_photo_intent(conn, arguments: dict) -> bool:
    safe_device_id = str((arguments or {}).get("device_id", "") or "").strip()
    if not safe_device_id:
        speak_txt(conn, "\u8bbe\u5907\u8fde\u63a5\u4fe1\u606f\u7f3a\u5931\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    if _get_server_mcp_manager(conn) is None:
        speak_txt(conn, "\u62cd\u7167\u529f\u80fd\u8fd8\u6ca1\u51c6\u5907\u597d\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
        return True

    conn._server_photo_capture_granted = True
    try:
        result = await _execute_server_mcp_tool_direct(
            conn,
            "xiaozhi_take_photo",
            arguments,
        )
    except Exception as e:
        conn._server_photo_capture_granted = False
        conn.logger.bind(tag=TAG).warning(f"direct server photo mcp failed: {e}")
        speak_txt(conn, f"\u62cd\u7167\u5931\u8d25\uff1a{e}")
        return True

    payload = finalize_server_mcp_payload(
        result,
        tool_name="xiaozhi_take_photo",
        arguments=arguments,
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
        speak_txt(conn, msg)
        return True

    reply = build_server_mcp_spoken_response(
        "xiaozhi_take_photo",
        payload,
        default_reply="\u62cd\u597d\u4e86\u3002",
    ) or _extract_text_from_result_payload(payload) or "\u62cd\u597d\u4e86\u3002"
    try:
        local_reply = await _advance_photo_confirmation_step_locally(
            conn,
            payload,
            fallback_reply=reply,
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"local server photo follow-up failed: {exc}"
        )
        local_reply = reply

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
    conn.logger.bind(tag=TAG).info("confirmed pending server photo capture, executing xiaozhi_take_photo directly")
    await _maybe_wait_before_photo_capture(conn, "server_photo_confirmation")
    return await _execute_server_photo_intent(
        conn,
        _build_pending_server_photo_request_fixed(conn),
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

    raw_tool_name = str(
        shortcut_cfg.get("take_photo_tool_name", "self.camera.take_photo")
    ).strip() or "self.camera.take_photo"
    timeout = int(shortcut_cfg.get("photo_timeout", 45))

    default_question = str(
        shortcut_cfg.get(
            "default_photo_question",
            "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1",
        )
    ).strip() or "\u63cf\u8ff0\u4e00\u4e0b\u770b\u5230\u7684\u7269\u54c1"
    question = _build_direct_photo_question(original_text, default_question)
    conn._pending_direct_photo = {
        "question": question,
        "raw_tool_name": raw_tool_name,
        "timeout": timeout,
    }
    speak_txt(conn, "\u53ef\u4ee5\u62cd\u7167\u5417\uff1f")
    return True


async def process_intent_result(conn, intent_result, original_text):
    """处理意图识别结果"""
    try:
        # 尝试将结果解析为JSON
        intent_data = json.loads(intent_result)

        # 检查是否有function_call
        if "function_call" in intent_data:
            # 直接从意图识别获取了function_call
            conn.logger.bind(tag=TAG).debug(
                f"检测到function_call格式的意图结果: {intent_data['function_call']['name']}"
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

                    current_time, today_date, today_weekday, lunar_date = get_current_time_info()
                    
                    # 构建带上下文的基础提示
                    context_prompt = f"""当前时间：{current_time}
                                        今天日期：{today_date} ({today_weekday})
                                        今天农历：{lunar_date}

                                        请根据以上信息回答用户的问题：{original_text}"""
                    
                    response = conn.intent.replyResult(context_prompt, original_text)
                    speak_txt(conn, response)
                
                conn.executor.submit(process_context_result)
                return True

            function_args = {}
            if "arguments" in intent_data["function_call"]:
                function_args = intent_data["function_call"]["arguments"]
                if function_args is None:
                    function_args = {}
            # 确保参数是字符串格式的JSON
            if isinstance(function_args, dict):
                function_args = json.dumps(function_args)

            function_call_data = {
                "name": function_name,
                "id": str(uuid.uuid4().hex),
                "arguments": function_args,
            }

            await send_stt_message(conn, original_text)
            conn.client_abort = False

            # 使用executor执行函数调用和结果处理
            def process_function_call():
                conn.dialogue.put(Message(role="user", content=original_text))

                # 使用统一工具处理器处理所有工具调用
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        conn.func_handler.handle_llm_function_call(
                            conn, function_call_data
                        ),
                        conn.loop,
                    ).result()
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"工具调用失败: {e}")
                    result = ActionResponse(
                        action=Action.ERROR, result=str(e), response=str(e)
                    )

                if result:
                    if result.action == Action.RESPONSE:  # 直接回复前端
                        text = result.response
                        if text is not None:
                            speak_txt(conn, text)
                    elif result.action == Action.REQLLM:  # 调用函数后再请求llm生成回复
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
                        # 获取当前最新的文本索引
                        text = result.response
                        if text is None:
                            text = result.result
                        if text is not None:
                            speak_txt(conn, text)

            # 将函数执行放在线程池中
            conn.executor.submit(process_function_call)
            return True
        return False
    except json.JSONDecodeError as e:
        conn.logger.bind(tag=TAG).error(f"处理意图结果时出错: {e}")
        return False


def speak_txt(conn, text):
    text = textUtils.prepare_runtime_spoken_text_for_conn(conn, text)
    if not text:
        return

    # 记录文本
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
