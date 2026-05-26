import csv
import json
import math
import re
import uuid
import asyncio
import time
import shutil
from pathlib import Path

import yaml
from core.utils.dialogue import Message
from core.providers.tts.dto.dto import ContentType
from core.handle.helloHandle import checkWakeupWords
from core.session import rotate_session_binding
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

    # Fresh-start semantics must not depend on fast-path availability.
    # When the user explicitly says "start today's experiment", always rotate
    # into a new chat/model session and prewarm a brand-new experiment session
    # instead of letting later handlers fall back to a stale device resume.
    if _is_explicit_experiment_start_request(filtered_text):
        try:
            return await _handle_explicit_experiment_start_request(conn, text)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"experiment explicit start handler failed: {exc}"
            )
            return False

    # Keep photo and UV-Vis direct handlers ahead of the generic experiment fast
    # path so confirmations like "可以拍照" still follow the device shortcut.
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
        # 使用支持function calling的聊天方法,不再进行意图分析
        return False
    # 使用LLM进行意图分析
    intent_result = await analyze_intent_with_llm(conn, text)
    if not intent_result:
        return False
    # 会话开始时生成sentence_id
    conn.sentence_id = str(uuid.uuid4().hex)
    if _assistant_waiting_for_step_start(conn) and _is_explicit_ready_to_start_reply(
        filtered_text
    ):
        textUtils.activate_experiment_ready_guard_bypass_for_current_sentence(
            conn,
            force=True,
        )
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


def _extract_experiment_current_group_number(payload) -> int | None:
    body = _experiment_result_body(payload)
    candidates = [
        body.get("current_group_number"),
        body.get("group_number"),
    ]

    progress = _extract_experiment_current_progress(payload)
    if isinstance(progress, dict):
        candidates.extend(
            [
                progress.get("group_number"),
                (progress.get("current_data") or {}).get("group_number")
                if isinstance(progress.get("current_data"), dict)
                else None,
            ]
        )

    for value in candidates:
        parsed = _extract_int_value(value)
        if parsed is not None and parsed > 0:
            return parsed
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
    norm = norm.replace("双杯", "烧杯")
    norm = norm.replace("磁子", "磁转子")
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
        return "实验图返回了缺失项，但没有给出明确字段名。我需要重新核对当前步骤，先不要补整组数据。"
    if len(prompts) == 1:
        return f"我刚查了实验图，继续前只差这一个确认：{prompts[0]}。"
    if len(prompts) == 2:
        joined = f"{prompts[0]}，还有 {prompts[1]}"
    else:
        joined = "、".join(prompts)
    return f"我刚查了实验图，继续前还差这些记录：{joined}。"


def _first_nonempty_text(*values) -> str:
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text:
            return text
    return ""


_INTERNAL_STEP_SPEECH_MARKERS = (
    "学生说",
    "直接记录",
    "本步全部完成",
    "进入下一步",
    "不要再追问",
    "质量检查",
    "禁止",
)


def _strip_internal_step_speech_text(text: str) -> str:
    """Remove graph-control instructions before composing student-facing speech."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    cleaned_parts = []
    for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", raw):
        if not part.strip():
            continue

        marker_positions = [
            part.find(marker)
            for marker in _INTERNAL_STEP_SPEECH_MARKERS
            if marker in part
        ]
        if not marker_positions:
            cleaned_parts.append(part.strip())
            continue

        keep = part[: min(marker_positions)].strip(" ，,；;。")
        if keep and not any(marker in keep for marker in _INTERNAL_STEP_SPEECH_MARKERS):
            cleaned_parts.append(keep)

    cleaned = "。".join(
        piece.rstrip("。！？!?；;").strip()
        for piece in cleaned_parts
        if piece.strip()
    ).strip()
    return cleaned


def _compose_uvvis_step_reply(step_meta: dict, mode: str = "guide") -> str:
    step_id = _first_nonempty_text(
        step_meta.get("step_id", ""),
        step_meta.get("id", ""),
    )

    if mode == "repeat":
        prefix = "当前这一步："
    elif mode == "next":
        prefix = "接下来做这一步："
    else:
        prefix = "现在做这一步："

    exact_reply_map = {
        "step_3_uv_vis_shared_dark_air_prep": (
            "先检查1到5号样品位都为空，参比位也不要放任何液体。"
            "都空了就告诉我。可以开始时直接说“开始扫描”。"
        ),
        "step_3_uv_vis_shared_dark_blank_prep": (
            f"{prefix}1-5号样品：纯水空白校正。"
            "请在 1-5 号样品位和参比位各放入纯水比色皿，共 6 个，放好后告诉我可以开始扫描。"
        ),
        "step_3_uv_vis_sample1-4_load_cuvette": (
            "把1到5号真实样品分别装入比色皿，按编号放入样品位，参比位放纯水，擦净外壁。"
            "放好后告诉我这是第几组。"
        ),
        "step_3_uv_vis_sample1-4_record_data": (
            "确认1到5号样品位都已放好真实样品，参比位是纯水。"
            "可以开始时直接说开始扫描。"
        ),
        "step_3_uv_vis_sample1-4_clean_cuvettes": (
            f"{prefix}紫外-可见测量后：统一清洗比色皿。"
            "按规范处理残液并清洗比色皿，为后续动力学实验做准备，做好告诉我。"
        ),
        "step_3_uv_vis_sample1-5_load_cuvette": (
            "把1到5号真实样品分别装入比色皿，按编号放入样品位，参比位保持为空，擦净外壁，做好告诉我。"
        ),
        "step_3_uv_vis_sample1-5_record_data": (
            "确认1到5号样品位都已放好真实样品，参比位保持为空。"
            "放好了告诉我开始测量。"
        ),
        "step_3_uv_vis_sample5_clean_cuvette": (
            f"{prefix}紫外-可见测量后：统一清洗比色皿。"
            "按规范处理残液并清洗比色皿，为后续动力学实验做准备，做好告诉我。"
        ),
        "step_6_kinetics_combined_measurement": (
            "请确认这是第几组动力学测量：1号位为空，2号位放2号样品反应液，3号位放2号样品参比液，"
            "4号位放4号样品反应液，5号位放4号样品参比液，仪器参考位为纯水。"
            "确认好后说“第几组，可以开始”。"
        ),
    }
    exact_reply = exact_reply_map.get(step_id)
    if exact_reply:
        return exact_reply

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

    looks_like_pure_water_blank_step = signature and (
        _signature_has_any("纯水空白校正", "纯水空白")
        and not _signature_has_any(
                "装入比色皿",
                "真实样品",
                "批量测光谱",
                "动力学",
                "反应液",
                "参比液",
                "sample1-5record",
                "sample2",
                "sample4",
        )
    )
    if looks_like_pure_water_blank_step:
        return (
            f"{prefix}1-5号样品：纯水空白校正。"
            "请在 1-5 号样品位和参比位各放入纯水比色皿，共 6 个，放好后告诉我可以开始扫描。"
        )

    if step_id == "step_3_uv_vis_shared_dark_air_prep" and signature and _signature_has_any(
            "暗电流校正",
            "暗电流和空气能量校正",
            "暗电流和空气基线",
            "共享暗电流和空气能量校正",
            "共享暗电流和空气基线",
            "shareddarkcurrent",
    ):
        return (
            "先检查1到5号样品位都为空，参比位也不要放任何液体。"
            "都空了就告诉我。可以开始时直接说“开始扫描”。"
        )

    if signature and (
        _signature_has_any("装入比色皿", "样品装杯")
        or (
            _signature_has_any("真实样品", "样品位")
            and _signature_has_any("参比位", "纯水")
            and _signature_has_any("比色皿", "装入")
        )
    ):
        return (
            "把1到5号真实样品分别装入比色皿，按编号放入样品位，参比位保持为空，擦净外壁，做好告诉我。"
        )

    if signature and (
        _signature_has_any(
            "批量测光谱",
            "批量光谱测量",
            "记录数据",
            "lambda max",
            "lambdamax",
        )
        or (
            _signature_has_any("1-5号样品", "1到5号样品")
            and _signature_has_any("测量", "光谱")
            and _signature_has_any("参比位", "纯水")
        )
    ):
        return (
            "确认1到5号样品位都已放好真实样品，参比位保持为空。"
            "放好了告诉我开始测量。"
        )

    if signature and _signature_has_any("统一清洗比色皿", "清洗比色皿", "测量后清洗"):
        return (
            f"{prefix}紫外-可见测量后：统一清洗比色皿。"
            "按规范处理残液并清洗比色皿，为后续动力学实验做准备，做好告诉我。"
        )

    if signature and _signature_has_all("2号样品", "动力学") and _signature_has_any("参比液", "参比位") and not _signature_has_any("反应液", "400纳米", "吸光度"):
        return (
            f"{prefix}2号样品动力学：配制参比液。"
            "按要求配好2号样品参比液，放入参比位并检查比色皿外壁和透光面，做好告诉我。"
        )

    if signature and _signature_has_all("2号样品", "动力学") and _signature_has_any("反应液", "样品位") and not _signature_has_any("400纳米", "吸光度"):
        return (
            f"{prefix}2号样品动力学：配制反应液。"
            "按要求配好2号样品反应液，放入样品位并确认参比和样品比色皿都放置正确，做好告诉我。"
        )

    if signature and _signature_has_all("2号样品", "动力学") and _signature_has_any("400纳米", "吸光度", "动力学测量"):
        return (
            f"{prefix}2号样品动力学：开始按时间记录吸光度。"
            "保持参比液和反应液按要求放好，可以开始时告诉我，我就开始400纳米动力学测量。"
        )

    if signature and _signature_has_all("4号样品", "动力学") and _signature_has_any("参比液", "参比位") and not _signature_has_any("反应液", "400纳米", "吸光度"):
        return (
            f"{prefix}4号样品动力学：配制参比液。"
            "按要求配好4号样品参比液，放入参比位并检查比色皿外壁和透光面，做好告诉我。"
        )

    if signature and _signature_has_all("4号样品", "动力学") and _signature_has_any("反应液", "样品位") and not _signature_has_any("400纳米", "吸光度"):
        return (
            f"{prefix}4号样品动力学：配制反应液。"
            "按要求配好4号样品反应液，放入样品位并确认参比和样品比色皿都放置正确，做好告诉我。"
        )

    if signature and _signature_has_all("4号样品", "动力学") and _signature_has_any("400纳米", "吸光度", "动力学测量"):
        return (
            f"{prefix}4号样品动力学：开始按时间记录吸光度。"
            "保持参比液和反应液按要求放好，可以开始时告诉我，我就开始400纳米动力学测量。"
        )

    if signature and _signature_has_any("step_6_data_analysis", "uvvis数据分析", "uv-vis数据分析", "紫外可见数据分析") and _signature_has_any("分析", "数据", "图谱"):
        return "UV-Vis 测量部分已经完成，接下来整理数据结果。"

    if not step_id:
        return ""

    reply_map = {
        "step_3_uv_vis_shared_dark_air_prep": (
            "先检查1到5号样品位都为空，参比位也不要放任何液体。"
            "都空了就告诉我。可以开始时直接说“开始扫描”。"
        ),
        "step_3_uv_vis_shared_dark_blank_prep": (
            f"{prefix}1-5号样品：纯水空白校正。"
            "请在 1-5 号样品位和参比位各放入纯水比色皿，共 6 个，放好后告诉我可以开始扫描。"
        ),
        "step_3_uv_vis_sample1-4_load_cuvette": (
            "把1到5号真实样品分别装入比色皿，按编号放入样品位，参比位放纯水，擦净外壁。"
            "放好后告诉我是第几组。"
        ),
        "step_3_uv_vis_sample1-4_record_data": (
            "确认1到5号样品位都已放好真实样品，参比位是纯水。"
            "可以开始时直接说开始扫描。"
        ),
        "step_3_uv_vis_sample1-4_clean_cuvettes": (
            f"{prefix}紫外-可见测量后：统一清洗比色皿。"
            "按规范处理残液并清洗比色皿，为后续动力学实验做准备，做好告诉我。"
        ),
        "step_3_uv_vis_sample1-5_load_cuvette": (
            "把1到5号真实样品分别装入比色皿，按编号放入样品位，参比位保持为空，擦净外壁，做好告诉我。"
        ),
        "step_3_uv_vis_sample1-5_record_data": (
            "确认1到5号样品位都已放好真实样品，参比位保持为空。"
            "放好了告诉我开始测量。"
        ),
        "step_3_uv_vis_sample5_clean_cuvette": (
            f"{prefix}紫外-可见测量后：统一清洗比色皿。"
            "按规范处理残液并清洗比色皿，为后续动力学实验做准备，做好告诉我。"
        ),
        "step_4_kinetics_sample2_reference_solution_preparation": (
            f"{prefix}2号样品动力学：配制参比液。"
            "按要求配好2号样品参比液，放入参比位并检查比色皿外壁和透光面，做好告诉我。"
        ),
        "step_4_kinetics_sample2_reaction_solution_preparation": (
            f"{prefix}2号样品动力学：配制反应液。"
            "按要求配好2号样品反应液，放入样品位并确认参比和样品比色皿都放置正确，做好告诉我。"
        ),
        "step_4_kinetics_sample2_measurement": (
            f"{prefix}2号样品动力学：开始按时间记录吸光度。"
            "保持参比液和反应液按要求放好，可以开始时告诉我，我就开始400纳米动力学测量。"
        ),
        "step_5_kinetics_sample4_reference_solution_preparation": (
            f"{prefix}4号样品动力学：配制参比液。"
            "按要求配好4号样品参比液，放入参比位并检查比色皿外壁和透光面，做好告诉我。"
        ),
        "step_5_kinetics_sample4_reaction_solution_preparation": (
            f"{prefix}4号样品动力学：配制反应液。"
            "按要求配好4号样品反应液，放入样品位并确认参比和样品比色皿都放置正确，做好告诉我。"
        ),
        "step_5_kinetics_sample4_measurement": (
            f"{prefix}4号样品动力学：开始按时间记录吸光度。"
            "保持参比液和反应液按要求放好，可以开始时告诉我，我就开始400纳米动力学测量。"
        ),
        "step_6_data_analysis": "UV-Vis 测量部分已经完成，接下来整理数据结果。",
        "step_6_kinetics_combined_measurement": (
            f"{prefix}2号和4号样品动力学：联合开始按时间记录吸光度。"
            "请保持参比位为纯水、1号位留空，2和3号位放2号样品的反应液和参比液，4和5号位放4号样品的反应液和参比液。全部放好后告诉我，我就开始400纳米动力学测量。"
        ),
    }
    return reply_map.get(step_id, "")


def _compose_experiment_step_reply(step_meta: dict, mode: str = "guide") -> str:
    title = _strip_internal_step_speech_text(
        _first_nonempty_text(step_meta.get("title", ""))
    )
    instruction = _first_nonempty_text(
        step_meta.get("instruction", ""),
        step_meta.get("description", ""),
        title,
    )
    instruction = _strip_internal_step_speech_text(instruction)
    safety = _strip_internal_step_speech_text(
        _first_nonempty_text(step_meta.get("safety", ""))
    )
    tip = _strip_internal_step_speech_text(
        _first_nonempty_text(step_meta.get("tip", ""))
    )

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
        sample_name = _format_sample_name(sample_index, "当前样品")
        return f"{sample_name}颜色已经稳定，现在可以拍照吗？"

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
    prepared = textUtils.prepare_runtime_spoken_text(
        _strip_internal_step_speech_text(text)
    )
    if prepared:
        return prepared
    if fallback_step_meta:
        fallback_text = _compose_experiment_step_reply(
            fallback_step_meta,
            mode=fallback_mode,
        )
        return textUtils.prepare_runtime_spoken_text(
            _strip_internal_step_speech_text(fallback_text)
        )
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

    confirmation = confirmation.rstrip("。！？!?；;，, ").strip()
    if confirmation:
        confirmation = f"{confirmation}。"
    return f"{confirmation}我接着带你做下一步。{followup}"


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
        "为什么",
        "原理",
        "依据",
        "详细",
        "注意事项",
        "是什么",
        "做什么",
        "要做什么",
        "需要什么",
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
    last_text = _normalize_text_for_match(_get_last_assistant_text_raw(conn))
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
    if _contains_any(norm, ready_tokens):
        return True

    return bool(
        re.fullmatch(
            r"(?:那就|现在|可以|那我们|我们|我)?开始(?:(?:第?[一二三四五六七八九十0-9]+步)|(?:(?:这个|今天的|本次)?实验))?(?:吧|啦|了)?",
            norm,
        )
    )


async def _reset_experiment_fresh_start_context(
    conn,
    *,
    allow_device_resume: bool = False,
) -> None:
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
            allow_device_resume=allow_device_resume,
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
    previous_session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    try:
        await _reset_experiment_fresh_start_context(
            conn,
            allow_device_resume=True,
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment explicit resume reset failed: {exc}"
        )
        return False

    graph_step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
    graph_session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
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

    if graph_session_id and graph_step_id and (
        graph_completed_steps > 0 or graph_step_id != "step_prepare_setup_all"
    ):
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
            reply = "我已经接回到上次实验记录对应的步骤了，你跟着这一步继续做。"
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

    target_step_id = str(
        getattr(conn, "experiment_resume_latest_current_step_id", "") or ""
    ).strip()
    log_path = str(getattr(conn, "experiment_resume_log_path", "") or "").strip()
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()

    if not log_path:
        reply = (
            "这个设备当前没有找到可用于续接的实验记录，只能重新开始。"
            "你说开始今天的实验，我就从第一步带你做。"
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
                "我找到了这个设备之前的实验日志，但还不能可靠判断现在实际做到哪一步。"
                "请直接告诉我当前实际步骤，比如“现在做到2号样品拍照”或“现在做到丁达尔观察”。"
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
            "我找到了上次实验的日志，但当前图谱没法自动跳到那一步。"
            "你可以让我重新开始，或者明确告诉我要跳到哪一步。"
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
            reply = "我已经按上次实验日志接回当前步骤了，你跟着这一步继续做。"
        else:
            reply = "我已经接回到上次实验记录对应的步骤了，你跟着这一步继续做。"

    await _start_direct_intent_turn(conn, original_text)
    speak_txt(conn, reply)
    return True


async def _handle_explicit_experiment_start_request(
    conn,
    original_text: str,
) -> bool:
    try:
        await _reset_experiment_fresh_start_context(
            conn,
            allow_device_resume=False,
        )
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment explicit start reset failed: {exc}"
        )
        return False

    experiment_title = await _load_experiment_overview_title(conn)
    reply = _compose_experiment_start_reply(experiment_title, "")
    reply = _prepare_fastpath_spoken_reply(reply)
    if not reply:
        return False

    await _start_direct_intent_turn(conn, original_text)
    speak_txt(conn, reply)
    return True


def _looks_like_generic_experiment_control_text(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False

    generic_tokens = {
        "缁х画",
        "缁х画鍚?",
        "缁х画涓嬩竴姝?",
        "涓嬩竴姝?",
        "寰€涓嬭蛋",
        "寰€鍚庤蛋",
        "鍋氬ソ浜?",
        "鍋氬畬浜?",
        "瀹屾垚浜?",
        "宸插畬鎴?",
        "宸茬粡鍋氬ソ浜?",
        "宸茬粡瀹屾垚浜?",
        "閮藉仛濂戒簡",
        "閮藉仛瀹屼簡",
        "褰撳墠姝ラ宸插畬鎴?",
        "杩欐瀹屾垚浜?",
        "杩欎竴姝ュ畬鎴愪簡",
        "濂戒簡",
        "鍙互浜?",
        "琛屼簡",
        "濂藉暒",
        "ok浜?",
    }
    if norm in generic_tokens:
        return True

    if len(norm) > 18:
        return False

    if _contains_any(norm, ("鏍峰搧", "agno3", "h2o2", "nabh4", "kbr", "绾按")):
        return False

    if re.search(r"[0-9涓€浜屼笁鍥涗簲鍏竷鍏節鍗?]", norm):
        return False

    generic_fragments = (
        "缁х画",
        "涓嬩竴姝?",
        "鍋氬ソ",
        "鍋氬畬",
        "瀹屾垚",
        "宸插畬鎴?",
        "濂戒簡",
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


def _assistant_waiting_after_sidetrack_question(conn) -> bool:
    recent = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=2))
    return _contains_any(
        recent,
        (
            "我们现在能继续做实验了吗",
            "现在能继续做实验了吗",
            "能继续做实验了吗",
            "继续做实验了吗",
        ),
    )


def _looks_like_sidetrack_continue_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm or len(norm) > 24:
        return False
    return _contains_any(
        norm,
        (
            "可以继续",
            "继续吧",
            "继续做实验",
            "继续进行实验",
            "接着做",
            "接着实验",
            "往下做",
            "能继续",
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

    repeat_measurement_tokens = (
        "再测一轮",
        "重测一轮",
        "重新测一轮",
        "再测一次",
        "重新测一次",
        "再扫一轮",
        "重新扫描",
        "重扫",
        "重测",
    )
    if _contains_any(norm, repeat_measurement_tokens):
        return "repeat"

    if waiting_for_step_start and _is_explicit_ready_to_start_reply(filtered_text):
        return "guide"

    advance_tokens = (
        "继续下一步",
        "下一步",
        "下一组",
        "下一组继续",
        "继续下一组",
        "换下一组",
        "下一批",
        "继续下一批",
        "下一组学生",
        "下一批学生",
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

    if waiting_for_step_completion and _looks_like_pure_short_completion_control(
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
        if waiting_for_step_completion:
            return "advance"
        if waiting_for_step_start:
            return "guide"

    if norm in {"继续", "继续吧"}:
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
    current_group_number = _extract_experiment_current_group_number(progress_payload)
    if current_group_number is not None:
        _remember_uvvis_group_number(conn, current_group_number)
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
    current_group_number = _extract_experiment_current_group_number(progress_payload)
    if current_group_number is not None:
        _remember_uvvis_group_number(conn, current_group_number)
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

_RESUME_LOG_GLOBAL_COMPLETION_TOKENS = (
    "全部完成",
    "都完成",
    "全都完成",
    "全部做好",
    "都做好",
    "全都做好",
    "全部放好",
    "都放好",
    "全都放好",
    "全部加好",
    "都加好了",
    "全都加好了",
    "已经全部",
    "都已经",
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
        "可以拍照",
        "可以拍",
        "能拍照",
        "能拍",
        "拍吧",
        "拍照吧",
        "现在拍",
        "开始拍",
        "拍一下",
        "拍一张",
        "同意拍照",
        "授权拍照",
    )
    short_positive_tokens = ("可以", "好", "行", "同意")
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
            "拍照权限",
            "授权拍照",
            "是否拍照",
            "询问是否拍照",
            "同意拍照",
            "可以拍照",
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
                "拍照权限",
                "授权拍照",
                "同意拍照",
                "允许拍照",
                "请求拍照",
                "询问拍照",
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
                f"{sample_index}号样品",
                f"{sample_index}号",
                f"样品{sample_index}",
            )
            if any(token in name for token in sample_tokens):
                score += 100
            else:
                return (-1, 0.0)
        if _contains_any(name, ("照片", "拍照", "photo", "image", "img")):
            score += 20
        if "重拍" in name or "补拍" in name:
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
        "全部完成",
        "都完成",
        "全都完成",
        "已经完成",
        "已完成",
        "完成了",
        "全部做好",
        "都做好",
        "全都做好",
        "做好了",
        "做完了",
        "全部混匀",
        "都混匀了",
        "全都混匀了",
        "已经全部混匀",
        "已全部混匀",
        "混合均匀",
        "混匀了",
        "开始搅拌",
        "已经开始搅拌",
        "已开始搅拌",
        "都已经开始搅拌",
        "搅拌好了",
    )
    return _contains_any(norm, completion_tokens)


def _looks_like_explicit_added_completion_report(filtered_text: str) -> bool:
    norm = _normalize_confirmation_signature(filtered_text)
    if not norm:
        return False
    if _contains_any(norm, _RESUME_LOG_NEGATIVE_TOKENS):
        return False
    completion_tokens = (
        "全部加好",
        "都加好了",
        "全都加好了",
        "已经加好了",
        "已加好了",
        "全部加完",
        "都加完了",
        "全都加完了",
        "已经加完了",
        "已加完了",
        "全部加入",
        "都加入了",
        "全都加入了",
        "已经加入",
        "已加入",
        "都已经加入",
        "全都已经加入",
        "按顺序加入",
        "顺序加入",
    )
    return _contains_any(norm, completion_tokens)


def _looks_like_addition_confirmation_description(description: str) -> bool:
    norm = _normalize_confirmation_signature(description)
    if not norm:
        return False
    return _contains_any(
        norm,
        (
            "加入",
            "加液",
            "滴加",
            "加完",
            "加好",
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


def _parse_small_chinese_float(token: str) -> float | None:
    text = str(token or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return _extract_float_value(text)

    normalized = text.replace("两", "二")
    if normalized == "半":
        return 0.5
    if "点" in normalized:
        left, right = normalized.split("点", 1)
        if not right:
            return None
        left_value = _parse_small_chinese_integer(left or "零")
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

    numeric_match = re.search(r"(\d+(?:\.\d+)?)\s*(分钟|分|秒钟|秒)", text)
    if numeric_match:
        value = _extract_float_value(numeric_match.group(1))
        if value is None:
            return None
        unit = numeric_match.group(2)
        return round(value / 60.0, 4) if "秒" in unit else value

    chinese_match = re.search(
        r"([零一二两三四五六七八九十点半]+)\s*(分钟|分|秒钟|秒)",
        text,
    )
    if not chinese_match:
        return None
    value = _parse_small_chinese_float(chinese_match.group(1))
    if value is None:
        return None
    unit = chinese_match.group(2)
    return round(value / 60.0, 4) if "秒" in unit else value


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

    cleaned = re.sub(r"\d+(?:\.\d+)?\s*(分钟|分|秒钟|秒)", "", text)
    cleaned = re.sub(r"[零一二两三四五六七八九十点半]+\s*(分钟|分|秒钟|秒)", "", cleaned)
    cleaned = re.sub(r"(颜色|最终|稳定|用了|用时|大约|大概|约|是|为|记作|已经|确认|样品)", "", cleaned)
    cleaned = re.sub(r"[0-9一二三四五六七八九十]+\s*号", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned).strip("，,。；;：: ")
    if not cleaned:
        return ""
    if len(cleaned) > 12:
        return ""
    if _contains_any(cleaned, ("可以拍照", "下一步", "继续", "然后呢", "干嘛", "没错", "对")):
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
    if not _contains_any(norm, ("丁达尔", "tyndall")):
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
            ("丁达尔", "tyndall"),
        ):
            continue
        tyndall_fields.append(field_name)

    if not tyndall_fields:
        return {}

    collective_positive = (
        _contains_any(
            norm,
            (
                "全部都有",
                "全都有",
                "都有",
                "均有",
                "都观察到",
                "都看到了",
                "都能看到",
                "都存在",
                "都有明显",
            ),
        )
        and _contains_any(norm, ("丁达尔", "tyndall"))
    )
    collective_negative = (
        _contains_any(
            norm,
            (
                "全部都没有",
                "全都没有",
                "都没有",
                "均无",
                "都看不到",
                "都没看到",
                "都未观察到",
                "都不存在",
            ),
        )
        and _contains_any(norm, ("丁达尔", "tyndall"))
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
            ("最终颜色", "颜色稳定所用时间", "反应时间", "颜色稳定"),
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
            if _field_matches_observation_semantics(field_name, field, ("颜色稳定", "确认颜色稳定")):
                stable_field_name = stable_field_name or field_name
            continue
        if type_text in {"string", "str"} and _field_matches_observation_semantics(
            field_name, field, ("最终颜色", "颜色")
        ):
            color_field_name = color_field_name or field_name
        if type_text in {"float", "number", "int", "integer"} and _field_matches_observation_semantics(
            field_name,
            field,
            ("颜色稳定所用时间", "反应时间", "稳定时间", "所用时间"),
        ):
            time_field_name = time_field_name or field_name

    if color_field_name and color_value:
        observation_fields[color_field_name] = color_value
    if time_field_name and duration_minutes is not None:
        observation_fields[time_field_name] = duration_minutes
    if stable_field_name and (
        duration_minutes is not None or "稳定" in textUtils.normalize_spoken_text(filtered_text or "")
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
    current_step_id = ""
    if hasattr(conn, "_extract_experiment_current_step_id"):
        current_step_id = conn._extract_experiment_current_step_id(
            step_payload,
            progress_payload,
        )
    if (
        current_step_id in _UVVIS_SAMPLE_LOAD_STEP_IDS
        and "native_reference_water_loaded" in missing_fields
        and (
            _looks_like_uvvis_ready_reply(filtered_text)
            or _looks_like_uvvis_start_scan_reply(filtered_text)
            or _classify_short_experiment_control(conn, filtered_text)
            in {"guide", "advance", "repeat"}
        )
    ):
        current_data = (current_progress or {}).get("current_data") or {}
        if current_data.get("all_samples_loaded_into_cuvettes") and current_data.get(
            "all_cuvettes_ready_for_measurement"
        ):
            group_number = current_data.get("group_number") or _get_current_uvvis_group_number(conn)
            fields = {
                "native_reference_water_loaded": True,
                "all_samples_loaded_into_cuvettes": True,
                "all_cuvettes_ready_for_measurement": True,
                "observations": current_data.get("observations")
                or "1-5号样品已装入比色皿并放入样品位，参比位为纯水。",
            }
            if group_number is not None:
                fields["group_number"] = group_number
                _remember_uvvis_group_number(conn, int(group_number))
            conn.logger.bind(tag=TAG).info(
                "uvvis sample load completion writeback recovered missing native reference: "
                f"text={filtered_text}, fields={sorted(fields.keys())}"
            )
            completed, reply = await _complete_experiment_step_with_fields(
                conn,
                fields=fields,
                auto_advance=True,
                fallback_reply="确认1到5号样品位都已放好真实样品，参比位是纯水。可以开始时直接说开始扫描。",
            )
            if completed or reply:
                return reply

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

    completed, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=write_fields,
        auto_advance=True,
        fallback_reply="我先记下了当前这一步的确认结果。",
    )
    if completed or reply:
        return reply
    return None


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
                f"我找到上次实验日志了，但在“{title}”这一步，日志里的汇报还不够我自动补齐。"
                f"{missing_reply}"
            )
        return (
            f"我找到上次实验日志了，但日志里还没有足够内容证明“{title}”这一步已经做完。"
            f"{missing_reply}"
        )
    if has_log_evidence:
        return (
            "我找到上次实验日志了，但日志里的汇报还不够我自动补齐当前这一步。"
            f"{missing_reply}"
        )
    return (
        "我找到上次实验日志了，但日志里还没有足够内容证明当前这一步已经做完。"
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
            "我找到了上次实验日志，但当前实验 YAML 里没法可靠解析出恢复顺序，"
            "现在还不能安全地自动续做。"
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

    try:
        reply = await _try_apply_current_confirmation_report(conn, filtered_text)
    except Exception as exc:
        conn.logger.bind(tag=TAG).warning(
            f"experiment confirmation semantic fast writeback failed: {exc}"
        )
        return False

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
    title = _normalize_text_for_match(step_meta.get("title", ""))
    if title and _contains_any(title, ("拍照", "照片", "拍摄")):
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
            "颜色稳定后拍照",
            "拍照确认",
            "调用mcp工具xiaozhi_take_photo",
            "拍照后基于照片",
            "基于照片确认",
            "拍照成功后",
            "照片颜色",
        ),
    ):
        return True

    # Some non-photo steps mention "完成后进入拍照记录步骤"; that should not make
    # the current step itself look like a photo-confirmation step.
    if _contains_any(
        haystack,
        (
            "进入本样品拍照记录步骤",
            "进入拍照记录步骤",
            "进入下一步拍照",
            "完成后进入拍照",
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


def _extract_uvvis_group_number_from_text(text: str) -> int | None:
    src = textUtils.normalize_spoken_text(text or "")
    if not src:
        return None

    patterns = (
        r"\u7b2c\s*([0-9]+)\s*\u7ec4",
        r"([0-9]+)\s*\u7ec4",
        r"\u7b2c\s*([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]+)\s*\u7ec4",
        r"([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]+)\s*\u7ec4",
    )
    for pattern in patterns:
        match = re.search(pattern, src)
        if not match:
            continue
        parsed = _parse_small_chinese_integer(match.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _get_current_uvvis_group_number(conn) -> int | None:
    for value in (
        getattr(conn, "experiment_current_group_number", None),
        getattr(conn, "_uvvis_current_group_number", None),
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _remember_uvvis_group_number(conn, group_number: int | None) -> int | None:
    if group_number is None:
        return _get_current_uvvis_group_number(conn)
    try:
        parsed = int(group_number)
    except (TypeError, ValueError):
        return _get_current_uvvis_group_number(conn)
    if parsed <= 0:
        return _get_current_uvvis_group_number(conn)
    setattr(conn, "experiment_current_group_number", parsed)
    setattr(conn, "_uvvis_current_group_number", parsed)
    return parsed


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


_EXPERIMENT_STEP_MATCH_ALIAS_RULES = (
    (re.compile(r"agno3", flags=re.IGNORECASE), ("硝酸银",)),
    (re.compile(r"硝酸银"), ("agno3",)),
    (re.compile(r"h2o2", flags=re.IGNORECASE), ("过氧化氢",)),
    (re.compile(r"过氧化氢"), ("h2o2",)),
    (re.compile(r"nabh4", flags=re.IGNORECASE), ("硼氢化钠",)),
    (re.compile(r"硼氢化钠"), ("nabh4",)),
    (re.compile(r"kbr", flags=re.IGNORECASE), ("溴化钾",)),
    (re.compile(r"溴化钾"), ("kbr",)),
    (re.compile(r"uv-?vis", flags=re.IGNORECASE), ("紫外可见",)),
    (re.compile(r"紫外[-－]?可见"), ("uvvis",)),
    (re.compile(r"去离子水"), ("纯水",)),
    (re.compile(r"纯水"), ("去离子水",)),
)
_EXPERIMENT_STEP_HINT_TOKEN_TEXTS = (
    "柠檬酸钠",
    "agno3",
    "硝酸银",
    "h2o2",
    "过氧化氢",
    "kbr",
    "溴化钾",
    "nabh4",
    "硼氢化钠",
    "纯水",
    "去离子水",
    "搅拌",
    "拍照",
    "照片",
    "丁达尔",
    "比色皿",
    "参比",
    "反应液",
    "uvvis",
    "紫外可见",
    "吸光度",
    "动力学",
    "400nm",
    "400纳米",
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
        token in normalized for token in ("1-5号样品", "1到5号样品", "1至5号样品", "全部样品")
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
        token in query_norm for token in ("1-5号", "1到5号", "1至5号", "每个烧杯", "全部样品")
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
                "照片",
                "拍照",
                "photo",
                "颜色",
                "观察",
                "现象",
                "absorbance",
                "吸光",
                "波长",
                "kinetics",
                "csv",
                "路径",
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
    bridge = "当前实验图谱还停在这一步，先按这一步继续。"

    if confirmation and not followup:
        confirmation = confirmation.rstrip("。！？!? ").strip()
        if confirmation:
            return f"{confirmation}。{bridge}"
        return bridge

    if followup and not confirmation:
        return f"{bridge}{followup}"

    if not confirmation and not followup:
        return bridge

    if confirmation == followup:
        return f"{bridge}{followup}"

    confirmation = confirmation.rstrip("。！？!? ").strip()
    if confirmation:
        confirmation = f"{confirmation}。"
    return f"{confirmation}{bridge}{followup}"


def _looks_like_internal_experiment_graph_message(message: str) -> bool:
    raw_text = str(message or "").strip()
    if not raw_text:
        return False

    normalized = _normalize_text_for_match(raw_text)
    internal_tokens = (
        "无法跳转到step",
        "前置步骤未完成",
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
                photo_fields = _build_experiment_photo_writeback_fields(
                    schema_by_name, photo_meta
                )
                missing_fields = list((current_progress or {}).get("missing_fields") or [])
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
    return fallback_reply or "拍照已经完成，继续做当前下一步。"


async def _advance_photo_confirmation_step_locally_v2(
    conn,
    payload,
    fallback_reply: str = "",
    *,
    requested_arguments: dict | None = None,
) -> str:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    confirmation_reply = fallback_reply or "拍好了。"
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
            photo_fields = _build_experiment_photo_writeback_fields(
                schema_by_name, photo_meta
            )
            missing_fields = list((current_progress or {}).get("missing_fields") or [])
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
    return confirmation_reply or "照片已经完成，继续做当前下一步。"


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
_UVVIS_LEGACY_SAMPLE_RECORD_STEP_ID = "step_3_uv_vis_sample1-5_record_data"
_UVVIS_LEGACY_SAMPLE_LOAD_STEP_ID = "step_3_uv_vis_sample1-5_load_cuvette"
_UVVIS_LEGACY_SAMPLE_CLEAN_STEP_ID = "step_3_uv_vis_sample5_clean_cuvette"
_UVVIS_SAMPLE_RECORD_STEP_IDS = {
    _UVVIS_SAMPLE_RECORD_STEP_ID,
    _UVVIS_LEGACY_SAMPLE_RECORD_STEP_ID,
}
_UVVIS_SAMPLE_LOAD_STEP_IDS = {
    _UVVIS_SAMPLE_LOAD_STEP_ID,
    _UVVIS_LEGACY_SAMPLE_LOAD_STEP_ID,
}
_UVVIS_SAMPLE_CLEAN_STEP_IDS = {
    _UVVIS_SAMPLE_CLEAN_STEP_ID,
    _UVVIS_LEGACY_SAMPLE_CLEAN_STEP_ID,
}
_UVVIS_KINETICS_SAMPLE2_STEP_ID = "step_4_kinetics_sample2_measurement"
_UVVIS_KINETICS_SAMPLE4_STEP_ID = "step_5_kinetics_sample4_measurement"
_UVVIS_KINETICS_COMBINED_STEP_ID = "step_6_kinetics_combined_measurement"
_UVVIS_ANALYSIS_STEP_ID = "step_6_data_analysis"
_UVVIS_BUSY_REPLY = "我现在正在工作请你过5min再试"
_UVVIS_NOT_READY_REPLY = "UV-Vis 这边还没准备好，请稍后再试。"
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
        return "当前实验图谱还没推进到 UV-Vis 前置校正，先完成丁达尔现象观察。"
    if step_id in _UVVIS_SAMPLE_RECORD_STEP_IDS:
        return "当前实验图谱还没推进到 1-5 号样品的批量光谱测量，先完成前面的步骤。"
    if step_id in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
    }:
        return "当前实验图谱还没推进到对应的 400 纳米动力学步骤，先完成前面的步骤。"
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
        "紫外可见",
        "光谱",
        "吸光度",
        "lambda max",
        "λmax",
        "400纳米",
        "动力学",
        "暗电流",
        "空气基线",
        "纯水空白",
        "参比位",
        "样品位",
        "比色皿",
        "空白液",
    )
    return _contains_any(normalized, uvvis_tokens)


def _assistant_recently_prompted_uvvis_action(conn) -> bool:
    recent_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not recent_text:
        return False

    uvvis_prompt_tokens = (
        "uvvis",
        "uv-vis",
        "紫外可见",
        "暗电流",
        "空气基线",
        "纯水空白",
        "参比位",
        "样品位",
        "比色皿",
        "400纳米",
        "动力学",
        "光谱",
        "空白液",
    )
    return _contains_any(recent_text, uvvis_prompt_tokens)


def _assistant_recently_prompted_pure_water_blank(conn) -> bool:
    recent_text = _normalize_text_for_match(_get_recent_assistant_text(conn, limit=3))
    if not recent_text:
        return False

    return _contains_any(
        recent_text,
        (
            "纯水空白",
            "纯水比色皿",
            "1-5号样品位",
            "参比位",
            "6支纯水",
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
        "放好了",
        "都放好了",
        "已经放好了",
        "已经放好",
        "可以开始了",
        "开始吧",
        "开始测量",
        "开始扫描",
        "可以开始扫描",
        "扫描吧",
        "开始空架扫描",
        "开始动力学",
        "开始记录",
        "测光谱",
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
        "都空了",
        "已经都空了",
        "都留空了",
        "已经留空了",
        "样品位都空了",
        "样品位和参比位都空了",
        "都准备好了",
        "准备好了",
        "已经准备好了",
        "空架准备好了",
    )
    return _contains_any(norm, ready_tokens)


def _looks_like_uvvis_empty_then_start_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    if _is_negative_short_reply_fixed(filtered_text):
        return False
    empty_tokens = (
        "都空了",
        "已经都空了",
        "都留空了",
        "已经留空了",
        "样品位都空了",
        "样品位和参比位都空了",
        "空架准备好了",
    )
    start_tokens = (
        "可以开始了",
        "开始吧",
        "开始测量",
        "开始扫描",
        "可以开始扫描",
        "扫描吧",
        "开始空架扫描",
    )
    return _contains_any(norm, empty_tokens) and _contains_any(norm, start_tokens)


def _looks_like_uvvis_start_scan_reply(filtered_text: str) -> bool:
    norm = _normalize_text_for_match(filtered_text)
    if not norm:
        return False
    if _is_negative_short_reply_fixed(filtered_text):
        return False
    start_tokens = (
        "可以开始了",
        "开始吧",
        "开始测量",
        "开始扫描",
        "可以开始扫描",
        "扫描吧",
        "开始空架扫描",
    )
    return _contains_any(norm, start_tokens)


def _looks_like_kinetics_rerun_request(
    original_text: str = "",
    filtered_text: str = "",
) -> bool:
    norm = _normalize_text_for_match(
        " ".join(
            text
            for text in (original_text, filtered_text)
            if str(text or "").strip()
        )
    )
    if not norm:
        return False
    kinetics_tokens = (
        "动力学",
        "400纳米",
        "2号和4号",
        "二号和四号",
        "2号与4号",
        "二号与四号",
    )
    rerun_tokens = (
        "重跑",
        "重新跑",
        "重测",
        "重新测",
        "重新做",
        "重新扫描",
        "重新扫",
        "重扫",
        "再测一轮",
        "再扫一轮",
    )
    return _contains_any(norm, kinetics_tokens) and _contains_any(norm, rerun_tokens)


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
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
        _UVVIS_ANALYSIS_STEP_ID,
    } | _UVVIS_SAMPLE_LOAD_STEP_IDS | _UVVIS_SAMPLE_RECORD_STEP_IDS | _UVVIS_SAMPLE_CLEAN_STEP_IDS


def _is_uvvis_measurement_step(step_id: str) -> bool:
    return str(step_id or "").strip() in {
        _UVVIS_SHARED_DARK_AIR_STEP_ID,
        _UVVIS_SHARED_BLANK_STEP_ID,
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
    } | _UVVIS_SAMPLE_RECORD_STEP_IDS


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
    } | _UVVIS_SAMPLE_RECORD_STEP_IDS


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

    if _contains_any(
        context_text,
        (
            "2号样品动力学",
            "sample2",
            "2号样品反应液",
            "2号样品参比液",
            "2号样品位",
        ),
    ):
        return _UVVIS_KINETICS_COMBINED_STEP_ID

    if _contains_any(
        context_text,
        (
            "4号样品动力学",
            "sample4",
            "4号样品反应液",
            "4号样品参比液",
            "4号样品位",
        ),
    ):
        return _UVVIS_KINETICS_COMBINED_STEP_ID

    if _contains_any(
        context_text,
        (
            "暗电流校正",
            "共享暗电流校正",
            "暗电流和空气能量校正",
            "共享暗电流和空气能量校正",
            "空气能量准备",
            "空气能量文件",
        ),
    ):
        return _UVVIS_SHARED_DARK_AIR_STEP_ID

    if _contains_any(
        context_text,
        (
            "暗电流",
            "空气基线",
            "空气能量",
            "纯水空白",
            "纯水比色皿",
            "样品位和参比位都留空",
            "样品位和参比位各放入纯水比色皿",
            "先不要放任何液体",
        ),
    ):
        return _UVVIS_SHARED_BLANK_STEP_ID

    if _contains_any(
        context_text,
        (
            "装入比色皿",
            "装样准备",
            "样品比色皿",
            "放入自动五联架",
            "参比位纯水比色皿保持不动",
        ),
    ):
        return _UVVIS_SAMPLE_LOAD_STEP_ID

    if _contains_any(
        context_text,
        (
            "批量测光谱",
            "光谱测量与记录",
            "开始1-5号样品的光谱测量",
            "1-5号样品的光谱",
            "开始样品测量",
            "λmax",
            "lambda max",
        ),
    ):
        return _UVVIS_SAMPLE_RECORD_STEP_ID

    if _contains_any(
        context_text,
        (
            "清洗比色皿",
            "统一清洗比色皿",
            "测量后的统一清洗",
        ),
    ):
        return _UVVIS_SAMPLE_CLEAN_STEP_ID

    if _contains_any(
        context_text,
        (
            "数据分析",
            "绘制ag nps吸收光谱",
            "lambda max与kbr用量",
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


def _resolve_experiment_report_output_root(conn) -> Path:
    override_root = str(conn.config.get("experiment_data_root", "") or "").strip()
    if override_root:
        return Path(override_root).expanduser().resolve()

    for candidate in _resolve_uvvis_experiment_data_dirs(conn):
        if candidate.name.lower() != "uv_data_common":
            return candidate

    return Path("data").resolve()


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


def _uvvis_dark_json_artifact_ready(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 2:
            return False
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("sample_dark_signal") is not None
            and payload.get("reference_dark_signal") is not None
        )
    except Exception:
        return False


def _uvvis_csv_artifact_ready(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required_columns = {"wavelength_nm"}
            for sample_position in _UVVIS_SAMPLE_POSITIONS:
                required_columns.add(f"sample{sample_position}_blank_signal")
                required_columns.add(f"sample{sample_position}_reference_blank_signal")
            if not required_columns.issubset(fieldnames):
                return False
            wavelengths = set()
            for row in reader:
                try:
                    wavelengths.add(round(float(row.get("wavelength_nm", ""))))
                except Exception:
                    continue
        expected_wavelengths = set(_UVVIS_SPECTRA_WAVELENGTH_GRID)
        return expected_wavelengths.issubset(wavelengths)
    except Exception:
        return False


def _uvvis_air_manifest_artifact_ready(path: Path, shared_dir: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 2:
            return False
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return False
        positions = {int(value) for value in payload.get("sample_positions", [])}
        if not set(_UVVIS_SAMPLE_POSITIONS).issubset(positions):
            return False
        wavelengths = {round(float(value)) for value in payload.get("wavelengths_nm", [])}
        if not set(_UVVIS_SPECTRA_WAVELENGTH_GRID).issubset(wavelengths):
            return False
        raw_csv = str(payload.get("raw_csv", "") or "").strip()
        if raw_csv:
            raw_path = Path(raw_csv)
            if not raw_path.is_absolute():
                raw_path = shared_dir / raw_path
            return _uvvis_csv_artifact_ready(raw_path)
        return True
    except Exception:
        return False


def _format_uvvis_shared_dark_air_cache_observation(cache_details: dict) -> str:
    shared_dir = str(cache_details.get("shared_dir", "") or "").strip()
    dark_current = str(cache_details.get("dark_current", "") or "").strip()
    air_manifest = str(cache_details.get("air_manifest", "") or "").strip()
    air_csv = str(cache_details.get("air_csv", "") or "").strip()
    parts = ["共享暗电流和空气能量校正已复用"]
    if shared_dir:
        parts.append(f"数据目录={shared_dir}")
    if dark_current:
        parts.append(f"暗电流={dark_current}")
    if air_manifest:
        parts.append(f"空气校正manifest={air_manifest}")
    if air_csv:
        parts.append(f"空气校正CSV={air_csv}")
    return "；".join(parts) + "。"


def _uvvis_shared_dark_air_cache_ready(conn) -> tuple[bool, dict]:
    for shared_dir in _resolve_uvvis_shared_blank_dirs(conn):
        dark_candidates = [
            shared_dir / "latest_dark_current.json",
            *sorted(shared_dir.glob("dark_current_*.json")),
        ]
        air_manifest_candidates = [
            shared_dir / "latest_air_blank_manifest.json",
            *sorted(shared_dir.glob("air_baseline_*_manifest.json")),
        ]
        air_csv_candidates = [
            shared_dir / "air_blank_latest.csv",
            *sorted(shared_dir.glob("air_baseline_*_raw.csv")),
        ]

        dark_path = next((path for path in dark_candidates if _uvvis_dark_json_artifact_ready(path)), None)
        air_manifest_path = next(
            (
                path
                for path in air_manifest_candidates
                if _uvvis_air_manifest_artifact_ready(path, shared_dir)
            ),
            None,
        )
        air_csv_path = next(
            (path for path in air_csv_candidates if _uvvis_csv_artifact_ready(path)),
            None,
        )

        if dark_path is not None and (air_manifest_path is not None or air_csv_path is not None):
            return True, {
                "shared_dir": str(shared_dir),
                "dark_current": str(dark_path),
                "air_manifest": str(air_manifest_path or ""),
                "air_csv": str(air_csv_path or ""),
            }

    return False, {}


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
        "占用",
        "忙",
        "请你过5min再试",
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
        "空白",
        "纯水",
        "参比液",
        "化学空白",
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
            "不存在",
            "缺少",
            "没有",
            "未找到",
            "还没有",
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
        for token in ("liquid blank", "pure water", "pure_water", "blank", "空白", "纯水")
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
            "可复用",
            "已存在",
            "已有",
            "已记录",
            "复用",
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
        "状态",
        "在工作",
        "正在工作",
        "工作吗",
        "忙吗",
        "空闲吗",
        "测完",
        "结束了吗",
        "完成了吗",
        "完成了吧",
    )
    if not any(token in normalized for token in query_tokens):
        return False

    if any(token in normalized for token in ("uvvis", "紫外可见", "光谱仪")):
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
        return "UV-Vis 现在正在工作。"
    if status.get("occupied") is True and not status.get("lease_owner_is_caller"):
        return "UV-Vis 现在被别的会话占用，还没空出来。"

    state = _get_uvvis_direct_state(conn, inferred_step_id if _is_uvvis_step(inferred_step_id) else "")
    phase = str(state.get("phase", "") or "").strip()
    if phase == "blank_reusable":
        return "UV-Vis 现在没有在工作。这一步已经确认当前批次纯水空白可复用，继续下一步时记得保留或重新放好参比位纯水比色皿。"
    if phase == "await_empty_positions":
        return "UV-Vis 现在没有在工作。这一步在等你确认1到5号样品位和参比位都已留空。"
    if phase == "await_scan_start":
        return "UV-Vis 现在没有在工作。这一步在等你确认可以开始扫描。"
    if phase == "await_pure_water_blank":
        return "UV-Vis 现在没有在工作。暗电流校正已经完成，这一步在等你把一到五号样品位和参比位各放一个纯水比色皿。"
    if phase == "await_shared_prep_reset":
        return "UV-Vis 现在没有在工作。这一步在等你把一到五号样品位和参比位都清空，我先补做共享前置校正。"
    if phase == "await_reaction_sample":
        return "UV-Vis 现在没有在工作，这一步在等你把样品和参比液放好。"
    if phase == "await_liquid_blank":
        return "UV-Vis 现在没有在工作，这一步在等你把指定的空白液放好。"

    if (
        inferred_step_id == _UVVIS_SHARED_BLANK_STEP_ID
        and _uvvis_shared_liquid_blank_exists(conn)
    ):
        return "UV-Vis 现在没有在工作。暗电流校正已经完成。"
    return "UV-Vis 现在没有在工作。"


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
        r"([一二三四五])\s*\u53f7\u4f4d",
        r"\u653e\u5728\s*([一二三四五])\s*\u53f7\u4f4d",
    )
    chinese_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
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
        match = re.search(r"(^|[\\/ _-])([1-5])号", source)
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


def _resolve_uvvis_primary_group_dir(conn) -> Path:
    device_dir = _resolve_uvvis_primary_device_dir(conn)
    group_number = _get_current_uvvis_group_number(conn)
    if group_number is None:
        return device_dir
    target_dir = (device_dir / str(group_number)).resolve()
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
        candidate_dirs = [_resolve_uvvis_primary_group_dir(conn)]
        candidate_dirs.extend(_resolve_uvvis_runtime_device_dirs(conn))
        for device_dir in candidate_dirs:
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
    artifact_dir = _resolve_uvvis_primary_group_dir(conn) / "uvvis_measure_spectra"
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
        "device_dir": str(_resolve_uvvis_primary_group_dir(conn)),
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
) -> tuple[bool, str]:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False, fallback_reply or ""

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
        start_args = {"session_id": session_id}
        group_number = _get_current_uvvis_group_number(conn)
        if group_number is not None:
            start_args["group_number"] = group_number
        start_payload = await _call_experiment_graph_tool_fast(
            conn,
            "start_trial",
            start_args,
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
        advanced, advanced_reply = await _advance_if_current_step_already_completed(
            conn,
            fallback_reply=fallback_reply,
        )
        if advanced:
            return True, advanced_reply
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


def _current_step_is_completed_from_payload(payload) -> bool:
    body = _experiment_result_body(payload)
    if not isinstance(body, dict):
        return False

    summary = body.get("summary")
    if isinstance(summary, dict):
        current_step = summary.get("current_step")
        if isinstance(current_step, dict):
            value = current_step.get("is_completed")
            if isinstance(value, bool):
                return value

    state = body.get("state")
    if isinstance(state, dict):
        value = state.get("current_step_is_completed")
        if isinstance(value, bool):
            return value

    return False


def _uvvis_sample_record_progress_has_saved_measurement(progress: dict | None) -> bool:
    if not isinstance(progress, dict):
        return False

    current_data = progress.get("current_data")
    if not isinstance(current_data, dict):
        current_data = progress

    has_saved_flag = bool(
        current_data.get("spectrum_saved")
        or current_data.get("measurement_round_saved")
    )
    has_lambda_fields = all(
        current_data.get(f"sample{sample_position}_lambda_max_nm") is not None
        for sample_position in _UVVIS_SAMPLE_POSITIONS
    )
    if not (has_saved_flag and has_lambda_fields):
        return False

    missing_fields = {
        str(field or "").strip()
        for field in (progress.get("missing_fields") or [])
        if str(field or "").strip()
    }
    return missing_fields.issubset({"step_finished_confirmed"})


async def _finish_saved_uvvis_spectra_step_if_ready(
    conn,
    *,
    fallback_reply: str = "当前批量光谱测量步骤已经完成。",
) -> tuple[bool, str]:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False, ""

    state_payload, progress_payload = await asyncio.gather(
        _call_experiment_graph_tool_fast(
            conn,
            "get_state",
            {"session_id": session_id},
            priority="foreground",
        ),
        _call_experiment_graph_tool_fast(
            conn,
            "get_current_progress",
            {"session_id": session_id},
            priority="foreground",
        ),
    )

    body = _experiment_result_body(state_payload)
    state = body.get("state") if isinstance(body, dict) else {}
    current_step_id = ""
    if isinstance(state, dict):
        current_step_id = str(state.get("current_step_id", "") or "").strip()
    if not current_step_id:
        current_step_id = _get_current_experiment_step_id(conn)
    if current_step_id not in _UVVIS_SAMPLE_RECORD_STEP_IDS:
        return False, ""

    current_progress = _extract_experiment_current_progress(progress_payload)
    if not _uvvis_sample_record_progress_has_saved_measurement(current_progress):
        return False, ""

    completed, reply = await _complete_experiment_step_with_fields(
        conn,
        fields={"step_finished_confirmed": True},
        auto_advance=True,
        fallback_reply=fallback_reply,
    )
    if not completed:
        return True, reply or fallback_reply

    _clear_uvvis_direct_state(conn)
    await _export_current_experiment_report(conn)
    return True, reply or fallback_reply


async def _advance_if_current_step_already_completed(
    conn,
    *,
    fallback_reply: str = "",
) -> tuple[bool, str]:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return False, fallback_reply or ""

    progress_payload = await _call_experiment_graph_tool_fast(
        conn,
        "get_current_progress",
        {"session_id": session_id},
        priority="foreground",
    )
    if not _current_step_is_completed_from_payload(progress_payload):
        return False, fallback_reply or ""

    conn.logger.bind(tag=TAG).info(
        "experiment current step already completed; forcing advance: "
        f"session_id={session_id}, current_step_id={getattr(conn, 'experiment_current_step_id', '')}"
    )
    return await _advance_finished_experiment_step(
        conn,
        fallback_reply=fallback_reply,
    )


async def _export_current_experiment_report(conn) -> dict:
    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    if not session_id:
        return {}

    output_root = _resolve_experiment_report_output_root(conn)
    output_path = (
        output_root
        / _normalize_uvvis_device_id(conn)
        / "experimental_graph_records.yaml"
    )
    payload = await _call_experiment_graph_tool_fast(
        conn,
        "export_records_to_yaml",
        {
            "session_id": session_id,
            "device_id": _normalize_uvvis_device_id(conn),
            "output_path": str(output_path),
            "output_root": str(output_root),
            "split_groups": False,
        },
        priority="foreground",
    )
    body = _experiment_result_body(payload)
    if bool(body.get("ok")):
        setattr(conn, "_last_experiment_report_export", body)
        conn.logger.bind(tag=TAG).info(
            "experiment report exported: "
            f"session_id={session_id}, "
            f"yaml_path={body.get('yaml_path', '')}, pdf_path={body.get('pdf_path', '')}"
        )
    else:
        conn.logger.bind(tag=TAG).warning(
            "experiment report export failed: "
            f"session_id={session_id}, "
            f"message={_extract_experiment_result_message(payload)}"
        )
    return body


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
    cache_ready, cache_details = _uvvis_shared_dark_air_cache_ready(conn)
    if cache_ready:
        conn.logger.bind(tag=TAG).info(
            "uvvis shared dark/air cache hit; skip uvvis_prepare_dark_current: "
            f"{cache_details}"
        )
        cache_notice = "已找到可复用的共享暗电流和空气能量校正数据。"
        not_advanced_reply = (
            f"{cache_notice}我还没有把实验图推进到下一步，请再说一次继续下一步。"
        )
        completed, reply = await _complete_experiment_step_with_fields(
            conn,
            fields={
                "empty_positions_confirmed": True,
                "shared_dark_current_ready": True,
                "shared_air_baseline_ready": True,
                "observations": _format_uvvis_shared_dark_air_cache_observation(cache_details),
            },
            auto_advance=True,
            fallback_reply=not_advanced_reply,
        )
        if completed:
            _clear_uvvis_direct_state(conn)
            if reply and reply != not_advanced_reply:
                reply = f"{cache_notice}{reply}"
            elif not reply:
                reply = cache_notice
        return True, reply

    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        return False, busy_reply

    prepare_payload = await _execute_uvvis_tool_payload(
        conn,
        "uvvis_prepare_dark_current",
        {"session_key": session_key},
    )
    if _payload_looks_busy_or_inaccessible(prepare_payload):
        return True, _UVVIS_BUSY_REPLY

    completed, reply = await _complete_experiment_step_with_fields(
        conn,
        fields={
            "empty_positions_confirmed": True,
            "shared_dark_current_ready": True,
            "shared_air_baseline_ready": True,
            "observations": "共享暗电流和空气能量校正已完成。",
        },
        auto_advance=True,
        fallback_reply="共享暗电流和空气能量校正已经完成。",
    )
    if completed:
        _clear_uvvis_direct_state(conn)
    return True, reply


async def _handle_uvvis_shared_dark_air_prep(
    conn, original_text: str, filtered_text: str
) -> bool:
    if _is_explicit_experiment_start_request(filtered_text):
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
        cache_ready, _cache_details = _uvvis_shared_dark_air_cache_ready(conn)
        if cache_ready:
            handled, reply = await _run_uvvis_shared_dark_air_scan(conn)
            if handled:
                if reply:
                    speak_txt(conn, reply)
                return True
            return False
        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_DARK_AIR_STEP_ID,
            phase="await_empty_positions",
        )
        speak_txt(
            conn,
            "先检查1到5号样品位都为空，参比位也不要放任何液体。都空了就告诉我。可以开始时直接说“开始扫描”。",
        )
        return True

    state = _get_uvvis_direct_state(conn, _UVVIS_SHARED_DARK_AIR_STEP_ID)
    phase = str(state.get("phase", "") or "").strip()
    control_action = _classify_short_experiment_control(conn, filtered_text)

    if control_action in {"guide", "advance", "repeat"}:
        advanced, reply = await _advance_if_current_step_already_completed(
            conn,
            fallback_reply="",
        )
        if advanced:
            await _start_direct_intent_turn(conn, original_text)
            _clear_uvvis_direct_state(conn)
            if reply:
                speak_txt(conn, reply)
            return True

    if phase == "await_empty_positions":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "好，等你确认样品位和参比位都留空后再告诉我。")
            return True
        cache_ready, _cache_details = _uvvis_shared_dark_air_cache_ready(conn)
        if cache_ready and (
            _looks_like_uvvis_ready_reply(filtered_text)
            or _looks_like_uvvis_empty_positions_reply(filtered_text)
            or _looks_like_uvvis_empty_then_start_reply(filtered_text)
            or _looks_like_uvvis_start_scan_reply(filtered_text)
            or _classify_short_experiment_control(conn, filtered_text)
            in {"guide", "advance", "repeat"}
        ):
            await _start_direct_intent_turn(conn, original_text)
            handled, reply = await _run_uvvis_shared_dark_air_scan(conn)
            if handled:
                if reply:
                    speak_txt(conn, reply)
                return True
            return False
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
                speak_txt(conn, "先确认1到5号样品位都为空，参比位也不要放任何液体。都空了就告诉我。可以开始时直接说“开始扫描”。")
                return True
            return False
        await _start_direct_intent_turn(conn, original_text)
        _set_uvvis_direct_state(
            conn,
            step_id=_UVVIS_SHARED_DARK_AIR_STEP_ID,
            phase="await_scan_start",
        )
        speak_txt(conn, "可以开始时告诉我“开始扫描”。")
        return True

    if phase == "await_scan_start":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "好，等你可以开始扫描时再告诉我。")
            return True
        if not (
            _looks_like_uvvis_empty_then_start_reply(filtered_text)
            or _looks_like_uvvis_ready_reply(filtered_text)
        ):
            if _looks_like_uvvis_empty_positions_reply(filtered_text):
                await _start_direct_intent_turn(conn, original_text)
                speak_txt(conn, "可以开始时告诉我“开始扫描”。")
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
                "暗电流校正",
                "暗电流",
                "空气基线",
                "空气能量",
                "共享暗电流",
                "开始uvvis",
                "开始紫外可见",
                "开始测量",
                "开始扫描",
                "可以开始扫描",
                "空架扫描",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)
    cache_ready, _cache_details = _uvvis_shared_dark_air_cache_ready(conn)
    if cache_ready:
        handled, reply = await _run_uvvis_shared_dark_air_scan(conn)
        if handled:
            if reply:
                speak_txt(conn, reply)
            return True
        return False
    _set_uvvis_direct_state(
        conn,
        step_id=_UVVIS_SHARED_DARK_AIR_STEP_ID,
        phase="await_empty_positions",
    )
    speak_txt(
        conn,
        "先检查1到5号样品位都为空，参比位也不要放任何液体。都空了就告诉我。可以开始时直接说“开始扫描”。",
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
            speak_txt(conn, "好，等你把纯水比色皿放好再告诉我。")
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
            speak_txt(conn, "这一步还缺纯水空白，请先把 1-5 号样品位和参比位都放入纯水比色皿，放好后告诉我可以开始扫描。")
            return True

        fields = {
            "shared_dark_current_ready": True,
            "shared_air_baseline_ready": True,
            "pure_water_blank_ready": True,
            "reference_cuvette_ready": True,
            "observations": "共享前置校正和纯水空白已准备完成",
        }
        auto_advanced, reply = await _complete_experiment_step_with_fields(
            conn,
            fields=fields,
            auto_advance=True,
            fallback_reply="共享前置校正和纯水空白都准备好了。",
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
                "暗电流",
                "空气基线",
                "空气能量",
                "纯水空白",
                "空白校正",
                "开始uvvis",
                "开始紫外可见",
                "开始测量",
                "开始扫描",
                "可以开始扫描",
                "空架扫描",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)
    speak_txt(conn, "先不要放任何液体，我先进行共享前置校正。")
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
        speak_txt(conn, "这一步还缺纯水空白，请先把 1-5 号样品位和参比位都放入纯水比色皿，放好后告诉我可以开始扫描。")
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
        "observations": "共享前置校正和纯水空白已完成或可复用",
    }
    auto_advanced, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=fields,
        auto_advance=True,
        fallback_reply="共享前置校正和纯水空白都准备好了。",
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
            speak_txt(conn, "好，等你把纯水比色皿放好再告诉我。")
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
            speak_txt(conn, "这一步还缺纯水空白，请先把 1-5 号样品位和参比位都放入纯水比色皿，放好后告诉我可以开始扫描。")
            return True

        auto_advanced, reply = await _complete_experiment_step_with_fields(
            conn,
            fields={
                "pure_water_blank_ready": True,
                "reference_cuvette_ready": True,
                "observations": "当前批次纯水空白已记录完成，参比位纯水比色皿可继续用于后续测量。",
            },
            auto_advance=True,
            fallback_reply="纯水空白已经准备好了。",
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
                "纯水空白",
                "纯水比色皿",
                "空白校正",
                "开始uvvis",
                "开始紫外可见",
                "开始测量",
                "开始扫描",
                "可以开始扫描",
                "空白",
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
                "observations": "当前批次纯水空白已确认可复用，后续测量将保留或重新放好参比位纯水比色皿。",
            },
            auto_advance=True,
            fallback_reply="当前批次纯水空白可复用，接下来装入样品比色皿。",
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
        "暗电流校正已经完成。请在 1-5 号样品位和参比位各放入纯水比色皿，共 6 个，放好后告诉我可以开始扫描。",
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
            speak_txt(conn, "好，等你把样品位和参比位都清空后再告诉我。")
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
            speak_txt(conn, "请先把 1 到 5 号样品位和参比位都清空，我先补做共享前置校正。清空后告诉我可以开始。")
            return True
        if _uvvis_shared_liquid_blank_exists(conn, payload) or _payload_indicates_liquid_blank_ready(payload):
            auto_advanced, reply = await _complete_uvvis_shared_blank_step(
                conn,
                observations="当前批次纯水空白已确认可复用，后续测量将保留或重新放好参比位纯水比色皿。",
                fallback_reply="当前批次纯水空白可复用，接下来装入样品比色皿。",
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
            "共享前置校正已经准备好。请在 1 到 5 号样品位和参比位各放 1 支纯水比色皿，共 6 支，放好了告诉我可以开始扫描。",
        )
        return True

    if phase == "await_pure_water_blank":
        if _is_negative_short_reply_fixed(filtered_text):
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "好，等你把纯水比色皿放好后再告诉我。")
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
            speak_txt(conn, "共享前置校正还没准备好，请先把 1 到 5 号样品位和参比位都清空，我先补做前置校正。清空后告诉我可以开始。")
            return True
        if _payload_mentions_missing_blank(payload):
            _set_uvvis_direct_state(
                conn,
                step_id=_UVVIS_SHARED_BLANK_STEP_ID,
                phase="await_pure_water_blank",
                session_key=session_key,
            )
            speak_txt(conn, "这一步还缺纯水空白，请先把 1 到 5 号样品位和参比位都放入纯水比色皿，放好后告诉我可以开始扫描。")
            return True

        auto_advanced, reply = await _complete_uvvis_shared_blank_step(
            conn,
            observations="当前批次纯水空白已记录完成，参比位纯水比色皿可继续用于后续测量。",
            fallback_reply="纯水空白已经准备好了。",
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
                "纯水空白",
                "纯水比色皿",
                "空白校正",
                "开始uvvis",
                "开始紫外可见",
                "开始测量",
                "开始扫描",
                "可以开始扫描",
                "空白",
            ),
        )
    ):
        return False

    await _start_direct_intent_turn(conn, original_text)

    if _uvvis_shared_liquid_blank_exists(conn):
        auto_advanced, reply = await _complete_uvvis_shared_blank_step(
            conn,
            observations="当前批次纯水空白已确认可复用，后续测量将保留或重新放好参比位纯水比色皿。",
            fallback_reply="当前批次纯水空白可复用，接下来装入样品比色皿。",
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
            observations="当前批次纯水空白已确认可复用，后续测量将保留或重新放好参比位纯水比色皿。",
            fallback_reply="当前批次纯水空白可复用，接下来装入样品比色皿。",
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
        speak_txt(conn, "请先保持 1 到 5 号样品位和参比位都为空，我先补做共享前置校正。清空后告诉我可以开始。")
        return True

    _set_uvvis_direct_state(
        conn,
        step_id=_UVVIS_SHARED_BLANK_STEP_ID,
        phase="await_pure_water_blank",
        session_key=session_key,
    )
    speak_txt(
        conn,
        "共享前置校正已经准备好。请在 1 到 5 号样品位和参比位各放 1 支纯水比色皿，共 6 支，放好了告诉我可以开始扫描。",
    )
    return True


_handle_uvvis_shared_blank_prep = _handle_uvvis_shared_blank_prep_v2


async def _handle_uvvis_sample_load_completion(
    conn, original_text: str, filtered_text: str
) -> bool:
    if not _looks_like_uvvis_ready_reply(filtered_text):
        return False

    explicit_group_number = _extract_uvvis_group_number_from_text(
        f"{original_text} {filtered_text}"
    )
    group_number = explicit_group_number
    if group_number is None:
        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, "这是第几组的样品？")
        return True
    group_number = _remember_uvvis_group_number(conn, group_number)

    await _start_direct_intent_turn(conn, original_text)
    fields = {
        "group_number": group_number,
        "all_samples_loaded_into_cuvettes": True,
        "all_cuvettes_ready_for_measurement": True,
        "native_reference_water_loaded": True,
        "reference_cuvette_ready": True,
        "observations": f"第{group_number}组1-5号样品已装入比色皿并放入样品位，参比位为纯水。",
    }
    advanced, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=fields,
        auto_advance=True,
        fallback_reply=f"好，已经记成第{group_number}组。现在可以直接说开始扫描。",
    )
    if not advanced:
        speak_txt(conn, reply or f"这是第{group_number}组。请确认1到5号样品都已经放好。")
        return True

    if _looks_like_uvvis_start_scan_reply(filtered_text):
        return await _handle_uvvis_spectra_measurement(
            conn,
            original_text,
            filtered_text,
            start_turn=False,
        )

    speak_txt(conn, f"好，已经记成第{group_number}组。现在可以直接说开始扫描。")
    return True


async def _handle_uvvis_spectra_measurement(
    conn, original_text: str, filtered_text: str, *, start_turn: bool = True
) -> bool:
    control_action = _classify_short_experiment_control(conn, filtered_text)
    if not (
        _looks_like_uvvis_ready_reply(filtered_text)
        or _is_explicit_experiment_resume_request(filtered_text)
        or control_action in {"guide", "advance", "repeat"}
    ):
        return False

    state = _get_uvvis_direct_state(conn, _get_current_experiment_step_id(conn))
    phase = str(state.get("phase") or "").strip()
    if phase == "await_finish_decision":
        if control_action == "repeat":
            await _start_direct_intent_turn(conn, original_text)
            _set_uvvis_direct_state(
                conn,
                step_id=_get_current_experiment_step_id(conn),
                phase="",
            )
            speak_txt(conn, "可以，再测一轮。请确认这是第几组，然后说开始扫描。")
            return True

        if control_action == "advance":
            await _start_direct_intent_turn(conn, original_text)
            handled, reply = await _finish_saved_uvvis_spectra_step_if_ready(conn)
            speak_txt(conn, reply or "当前批量光谱测量步骤已经完成。")
            return True

        await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, "你现在可以说“再测一轮”，或说“当前步骤全部完成”再进入下一步。")
        return True

    if control_action == "advance":
        handled, reply = await _finish_saved_uvvis_spectra_step_if_ready(conn)
        if handled:
            if start_turn:
                await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, reply or "当前批量光谱测量步骤已经完成。")
            return True

    explicit_group_number = _extract_uvvis_group_number_from_text(
        f"{original_text} {filtered_text}"
    )
    group_number = explicit_group_number
    if group_number is None:
        if start_turn:
            await _start_direct_intent_turn(conn, original_text)
        speak_txt(conn, "这是第几组的样品扫描？")
        return True
    group_number = _remember_uvvis_group_number(conn, group_number)

    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            if start_turn:
                await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    if start_turn:
        await _start_direct_intent_turn(conn, original_text)
    output_dir = str(_resolve_uvvis_primary_group_dir(conn))
    payload = await _execute_uvvis_tool_payload(
        conn,
        "uvvis_measure_spectra",
        {
            "session_key": session_key,
            "sample_positions": list(_UVVIS_SAMPLE_POSITIONS),
            "ready_for_samples": True,
            "output_dir": output_dir,
        },
    )
    if _payload_looks_busy_or_inaccessible(payload):
        speak_txt(conn, _UVVIS_BUSY_REPLY)
        return True
    if _payload_mentions_missing_blank(payload):
        cache_ready, cache_details = _uvvis_shared_dark_air_cache_ready(conn)
        if cache_ready:
            conn.logger.bind(tag=TAG).warning(
                "uvvis_measure_spectra reported missing shared prep, but reusable "
                f"shared dark/air cache exists; preserving current graph step: {cache_details}"
            )
            speak_txt(
                conn,
                "共享暗电流和空气能量校正数据是完整的，我没有回退实验步骤。请再说一次开始扫描。",
            )
            return True
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
            speak_txt(conn, "这一步缺少纯水空白，我先退回前置校正。请先把样品位和参比位都清空，再告诉我开始。")
        else:
            speak_txt(conn, "这一步缺少共享前置校正，我先退回共享暗电流和空气能量校正。请先把样品位和参比位都清空，再告诉我开始扫描。")
        return True

    rows = _extract_uvvis_measure_spectra_rows(payload, conn)
    spectra_artifacts = _persist_uvvis_measure_spectra_artifacts(conn, payload, rows)
    setattr(conn, "_last_uvvis_spectra_artifacts", spectra_artifacts)
    if len(rows) < 5 or any(sample_position not in rows for sample_position in _UVVIS_SAMPLE_POSITIONS):
        speak_txt(conn, "这次光谱结果还不完整，我还没拿到 1 到 5 号样品的完整结果，请稍后再试。")
        return True
    if not spectra_artifacts.get("all_expected_outputs_exist", False):
        speak_txt(
            conn,
            "这次光谱峰位我已经拿到了，但 400 到 700 纳米每隔 10 纳米的完整校正吸光度结果或绘图还没保存成功，请稍后重试。",
        )
        return True

    round_index = len(
        list(
            (
                _resolve_uvvis_primary_group_dir(conn) / "uvvis_measure_spectra"
            ).glob("sample_batch_*_manifest.json")
        )
    )
    if round_index < 1:
        round_index = 1

    fields = {
        f"sample{sample_position}_lambda_max_nm": rows[sample_position]["lambda_max_nm"]
        for sample_position in _UVVIS_SAMPLE_POSITIONS
    }
    for sample_position in _UVVIS_SAMPLE_POSITIONS:
        max_absorbance = rows[sample_position].get("max_absorbance")
        if max_absorbance is not None:
            fields[f"sample{sample_position}_max_absorbance"] = max_absorbance
    fields["spectrum_saved"] = True
    fields["measurement_round_index"] = round_index
    fields["measurement_round_saved"] = True
    fields["step_finished_confirmed"] = False
    fields["observations"] = (
        "1-5号样品批量扫描完成，"
        + "；".join(
            f"{sample_position}号样品λmax={rows[sample_position]['lambda_max_nm']}nm"
            for sample_position in _UVVIS_SAMPLE_POSITIONS
        )
        + "；400-700nm（10nm步长）的校正吸光度结果和光谱图已保存"
    )

    auto_advanced, reply = await _complete_experiment_step_with_fields(
        conn,
        fields=fields,
        auto_advance=False,
        fallback_reply="1-5号样品的光谱都测好了。",
    )
    report_payload = {}
    if auto_advanced:
        report_payload = await _export_current_experiment_report(conn)
    summary = "，".join(
        f"{sample_position}号{rows[sample_position]['lambda_max_nm']}纳米"
        for sample_position in _UVVIS_SAMPLE_POSITIONS
    )
    spoken_reply = f"{summary}。400到700纳米每隔10纳米的校正吸光度结果和光谱图已保存。"
    if bool(report_payload.get("ok")) and report_payload.get("pdf_generated"):
        spoken_reply += "实验总报告PDF和YAML也已更新。"
    elif auto_advanced:
        spoken_reply += "实验总报告还没有完整生成成功，我已保留当前步骤。"
    elif reply:
        spoken_reply += reply
    spoken_reply += "你现在可以说“再测一轮”，或说“当前步骤全部完成”再进入下一步。"
    _set_uvvis_direct_state(
        conn,
        step_id=_get_current_experiment_step_id(conn),
        phase="await_finish_decision",
    )
    speak_txt(conn, spoken_reply)
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
    explicit_group_number = _extract_uvvis_group_number_from_text(
        f"{original_text} {filtered_text}"
    )
    is_explicit_rerun = control_action == "repeat" or _looks_like_kinetics_rerun_request(
        original_text,
        filtered_text,
    )

    def _select_kinetics_output_dir(group_number: int | None) -> str:
        if group_number is not None:
            _remember_uvvis_group_number(conn, group_number)
        selected_group_number = _get_current_uvvis_group_number(conn)
        if selected_group_number is None:
            return ""
        target_dir = (
            _resolve_uvvis_native_output_root(conn)
            / _normalize_uvvis_device_id(conn)
            / str(selected_group_number)
        ).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir)

    def _extract_grouped_task_id(payload: dict) -> str:
        values = _collect_payload_named_values(payload, {"task_id"})
        return str(values.get("task_id") or "").strip()

    def _grouped_kinetics_counts(record_fields: dict) -> tuple[int, int]:
        sample2_count = sum(
            1
            for key in record_fields
            if re.fullmatch(r"sample2_t\d+_absorbance", str(key or ""))
        )
        sample4_count = sum(
            1
            for key in record_fields
            if re.fullmatch(r"sample4_t\d+_absorbance", str(key or ""))
        )
        return sample2_count, sample4_count

    def _grouped_kinetics_progress_text(payload: dict, group_number: int | None) -> str:
        data = _to_plain_data(payload)
        point_count = None
        completed_points = None
        if isinstance(data, dict):
            point_count = _extract_int_value(data.get("point_count"))
            completed_points = _extract_int_value(data.get("completed_points"))
            sample_groups = data.get("sample_groups")
            if completed_points is None and isinstance(sample_groups, list):
                counts = [
                    _extract_int_value(item.get("completed_points"))
                    for item in sample_groups
                    if isinstance(item, dict)
                ]
                counts = [value for value in counts if value is not None]
                if counts:
                    completed_points = min(counts)
            if point_count is None and isinstance(sample_groups, list):
                totals = [
                    _extract_int_value(item.get("point_count"))
                    for item in sample_groups
                    if isinstance(item, dict)
                ]
                totals = [value for value in totals if value is not None]
                if totals:
                    point_count = max(totals)
        group_label = f"第{group_number}组" if group_number else "当前组"
        if completed_points is not None and point_count:
            return (
                f"{group_label}动力学测量正在记录，当前已完成 {completed_points}/{point_count} 个时间点。"
                "请不要移动2、3、4、5号位比色皿。"
            )
        return f"{group_label}动力学测量正在记录。请不要移动2、3、4、5号位比色皿。"

    def _grouped_kinetics_complete_fields(payload: dict, group_number: int | None) -> tuple[dict, int | None]:
        record_fields = _extract_uvvis_measure_kinetics_record_fields(payload, conn, "")
        sample2_count, sample4_count = _grouped_kinetics_counts(record_fields)
        if sample2_count < 35 or sample4_count < 35:
            return {}, None
        data = _to_plain_data(payload)
        round_index = None
        if isinstance(data, dict):
            round_index = _extract_int_value(data.get("round_index"))
        fields = {
            key: value
            for key, value in record_fields.items()
            if re.fullmatch(r"(sample2|sample4)_t\d+_absorbance", str(key or ""))
        }
        if group_number is not None:
            fields["group_number"] = group_number
        if round_index is not None:
            fields["kinetics_round_index"] = round_index
        fields["kinetics_round_saved"] = True
        fields["step_finished_confirmed"] = False
        fields["observations"] = (
            f"第{group_number}组2号和4号样品 400 nm 联合动力学测量完成，"
            "2号样品使用2/3号位，4号样品使用4/5号位；"
            "本轮共记录35个时间点。"
        )
        return fields, round_index

    async def _cancel_open_kinetics_trial_if_only_metadata(group_number: int | None) -> None:
        session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
        if not session_id:
            return
        try:
            progress_payload = await _call_experiment_graph_tool_fast(
                conn,
                "get_current_progress",
                {"session_id": session_id},
                priority="foreground",
            )
            progress = _extract_experiment_current_progress(progress_payload)
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"kinetics progress check before async start failed: {exc}"
            )
            return
        if not isinstance(progress, dict):
            return
        current_data = progress.get("current_data")
        if not isinstance(current_data, dict):
            current_data = {}
        has_absorbance = any(
            re.fullmatch(r"(sample2|sample4)_t\d+_absorbance", str(key or ""))
            for key in current_data
        )
        if has_absorbance:
            return
        progress_group = _extract_int_value(progress.get("group_number")) or _extract_int_value(
            current_data.get("group_number")
        )
        if group_number is not None and progress_group not in (None, group_number):
            should_cancel = True
        else:
            collected = _extract_int_value(progress.get("collected_required_fields"))
            should_cancel = collected is None or collected <= 4
        if not should_cancel:
            return
        try:
            await _call_experiment_graph_tool_fast(
                conn,
                "cancel_trial",
                {"session_id": session_id},
                priority="foreground",
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"kinetics stale trial cancel failed before async start: {exc}"
            )

    def _clear_existing_kinetics_output(output_dir: str, group_number: int | None) -> bool:
        output_path = Path(str(output_dir or "")).resolve()
        if not output_path:
            return False
        expected_group = str(group_number or "").strip()
        if expected_group and output_path.name != expected_group:
            conn.logger.bind(tag=TAG).warning(
                "skip clearing kinetics output because output_dir does not end with "
                f"group number: output_dir={output_path}, group={expected_group}"
            )
            return False
        kinetics_dir = (output_path / "uvvis_measure_kinetic").resolve()
        if kinetics_dir.parent != output_path:
            conn.logger.bind(tag=TAG).warning(
                f"skip clearing unexpected kinetics output path: {kinetics_dir}"
            )
            return False
        if not kinetics_dir.exists():
            return True
        try:
            shutil.rmtree(kinetics_dir)
            conn.logger.bind(tag=TAG).info(
                f"cleared existing grouped kinetics output before rerun: {kinetics_dir}"
            )
            return True
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"failed to clear existing grouped kinetics output before rerun: {kinetics_dir}, error={exc}"
            )
            return False

    def _kinetics_csv_row_count(csv_path: Path) -> int:
        if not csv_path.exists() or not csv_path.is_file():
            return 0
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return sum(1 for _row in csv.DictReader(handle))
        except Exception:
            return 0

    def _existing_kinetics_data_summary(output_dir: str, group_number: int | None) -> dict:
        output_path = Path(str(output_dir or "")).resolve()
        expected_group = str(group_number or "").strip()
        if expected_group and output_path.name != expected_group:
            return {"has_data": False}

        kinetics_dir = (output_path / "uvvis_measure_kinetic").resolve()
        if kinetics_dir.parent != output_path or not kinetics_dir.exists():
            return {"has_data": False, "kinetics_dir": str(kinetics_dir)}

        positions_count = _kinetics_csv_row_count(kinetics_dir / "positions_absorbance.csv")
        sample2_count = _kinetics_csv_row_count(
            kinetics_dir / "2号样品" / "absorbance.csv"
        )
        sample4_count = _kinetics_csv_row_count(
            kinetics_dir / "4号样品" / "absorbance.csv"
        )
        point_count = None
        phase_text = ""
        progress_has_data = False
        progress_path = kinetics_dir / "kinetics_progress.json"
        if progress_path.exists() and progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except Exception:
                progress = {}
            if isinstance(progress, dict):
                point_count = _extract_int_value(progress.get("point_count"))
                phase_text = str(progress.get("phase") or "").strip()
                record_fields = progress.get("record_fields")
                if isinstance(record_fields, dict):
                    progress_has_data = any(
                        re.fullmatch(r"(sample2|sample4)_t\d+_absorbance", str(key or ""))
                        for key in record_fields
                    )
                progress_has_data = progress_has_data or bool(point_count and point_count > 0)

        has_data = any(
            count > 0 for count in (positions_count, sample2_count, sample4_count)
        ) or progress_has_data
        complete = (
            point_count is not None
            and point_count >= 35
            and sample2_count >= 35
            and sample4_count >= 35
        ) or (
            phase_text == "grouped_kinetics_complete"
            and sample2_count >= 35
            and sample4_count >= 35
        )
        return {
            "has_data": bool(has_data),
            "complete": bool(complete),
            "kinetics_dir": str(kinetics_dir),
            "positions_count": positions_count,
            "sample2_count": sample2_count,
            "sample4_count": sample4_count,
            "point_count": point_count,
            "phase": phase_text,
        }

    def _pending_overwrite_confirmation_matches(group_number: int, output_dir: str) -> bool:
        if phase != "await_kinetics_overwrite_confirmation":
            return False
        state_group = _extract_int_value(state.get("group_number"))
        state_output_dir = str(state.get("output_dir") or "").strip()
        return state_group == group_number and state_output_dir == str(output_dir or "").strip()

    async def _finish_grouped_kinetics_payload(
        payload: dict,
        group_number: int | None,
        *,
        running_reply: str = "",
    ) -> bool:
        fields, _round_index = _grouped_kinetics_complete_fields(payload, group_number)
        if not fields:
            speak_txt(conn, running_reply or _grouped_kinetics_progress_text(payload, group_number))
            return True
        completed, reply = await _complete_experiment_step_with_fields(
            conn,
            fields=fields,
            auto_advance=False,
            fallback_reply=(
                "我记录好了。如果还要继续重测，请说再测一轮；"
                "如果当前步骤已经全部完成，请直接告诉我。"
            ),
        )
        if completed:
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_finish_decision",
                session_key=session_key,
                group_number=group_number,
                output_dir=_select_kinetics_output_dir(group_number),
            )
            speak_txt(
                conn,
                f"我记录好了。第{group_number}组2号和4号样品的动力学数据已经保存。"
                "如果还要继续重测，请说再测一轮；如果这个步骤已经全部完成，请直接告诉我。",
            )
        elif reply:
            speak_txt(conn, reply)
        else:
            speak_txt(conn, "这次动力学结果还没有完整写回当前步骤，请稍后再试。")
        return True

    async def _check_grouped_kinetics_status(group_number: int | None) -> bool:
        task_id = str(state.get("task_id") or "").strip()
        status_args = {"task_id": task_id} if task_id else {}
        try:
            payload = await _execute_uvvis_tool_payload(
                conn,
                "uvvis_grouped_kinetics_status",
                status_args,
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"uvvis_grouped_kinetics_status failed: {exc}"
            )
            speak_txt(conn, _UVVIS_NOT_READY_REPLY)
            return True
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True
        fields, _round_index = _grouped_kinetics_complete_fields(payload, group_number)
        if fields:
            return await _finish_grouped_kinetics_payload(payload, group_number)
        data = _to_plain_data(payload)
        state_text = ""
        if isinstance(data, dict):
            state_text = str(data.get("state") or data.get("phase") or "").strip().lower()
        if any(token in state_text for token in ("succeed", "complete", "done", "finished")):
            try:
                result_payload = await _execute_uvvis_tool_payload(
                    conn,
                    "uvvis_grouped_kinetics_result",
                    status_args,
                )
            except Exception as exc:
                conn.logger.bind(tag=TAG).warning(
                    f"uvvis_grouped_kinetics_result failed: {exc}"
                )
                speak_txt(conn, "动力学测量已结束，但结果还没有取回成功，请稍后再查一次。")
                return True
            return await _finish_grouped_kinetics_payload(result_payload, group_number)
        speak_txt(conn, _grouped_kinetics_progress_text(payload, group_number))
        return True

    async def _start_grouped_kinetics(group_number: int, *, clear_existing: bool = False) -> bool:
        output_dir = _select_kinetics_output_dir(group_number)
        if not output_dir:
            speak_txt(conn, "这是第几组的动力学测量？请说第几组可以开始。")
            return True
        existing_summary = _existing_kinetics_data_summary(output_dir, group_number)
        if (
            existing_summary.get("has_data")
            and not _pending_overwrite_confirmation_matches(group_number, output_dir)
        ):
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_kinetics_overwrite_confirmation",
                session_key=session_key,
                group_number=group_number,
                output_dir=output_dir,
                existing_kinetics_summary=existing_summary,
            )
            if existing_summary.get("complete"):
                detail = "已经有完整的35个时间点"
            else:
                detail = (
                    "已经有部分数据，"
                    f"2号样品{existing_summary.get('sample2_count', 0)}个点，"
                    f"4号样品{existing_summary.get('sample4_count', 0)}个点"
                )
            speak_txt(
                conn,
                f"我检查到第{group_number}组动力学目录里{detail}。是否需要重测并覆盖这些数据？",
            )
            return True
        if clear_existing and not _clear_existing_kinetics_output(output_dir, group_number):
            speak_txt(
                conn,
                f"第{group_number}组旧的动力学结果目录没有清理成功，我先不启动重测，避免新旧数据混在一起。",
            )
            return True
        await _cancel_open_kinetics_trial_if_only_metadata(group_number)
        try:
            payload = await _execute_uvvis_tool_payload(
                conn,
                "uvvis_grouped_kinetics_start",
                {
                    "session_key": session_key,
                    "wavelength_nm": 400,
                    "duration_minutes": 34,
                    "interval_seconds": 60,
                    "readings_per_point": 1,
                    "settle_time": 0.01,
                    "sample_positions": list(_UVVIS_GROUPED_KINETICS_POSITIONS),
                    "output_dir": output_dir,
                    "force_restart": bool(clear_existing),
                },
            )
        except Exception as exc:
            conn.logger.bind(tag=TAG).warning(
                f"uvvis_grouped_kinetics_start failed: {exc}"
            )
            speak_txt(conn, _UVVIS_NOT_READY_REPLY)
            return True
        if _payload_looks_busy_or_inaccessible(payload):
            speak_txt(conn, _UVVIS_BUSY_REPLY)
            return True
        task_id = _extract_grouped_task_id(payload)
        setattr(conn, "_last_uvvis_kinetics_artifacts", _to_plain_data(payload))
        _set_uvvis_direct_state(
            conn,
            step_id=step_id,
            phase="kinetics_running",
            session_key=session_key,
            group_number=group_number,
            output_dir=output_dir,
            task_id=task_id,
        )
        speak_txt(
            conn,
            f"第{group_number}组2号和4号样品动力学测量已经开始，仪器会连续记录35个时间点。"
            "请不要移动2、3、4、5号位比色皿。",
        )
        return True

    session_key, busy_reply = await _ensure_uvvis_session_key(conn)
    if not session_key:
        if busy_reply:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, busy_reply)
            return True
        return False

    if phase == "await_kinetics_overwrite_confirmation":
        await _start_direct_intent_turn(conn, original_text)
        group_number = explicit_group_number or _extract_int_value(state.get("group_number"))
        output_dir = str(state.get("output_dir") or "").strip()
        if group_number is None:
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_grouped_samples",
                session_key=session_key,
                group_number=_get_current_uvvis_group_number(conn),
            )
            speak_txt(conn, "这是第几组的动力学测量？请说第几组可以开始。")
            return True
        if is_negative:
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_grouped_samples",
                session_key=session_key,
                group_number=group_number,
                output_dir=output_dir,
            )
            speak_txt(conn, f"好，我不会覆盖第{group_number}组已有动力学数据。")
            return True
        if is_affirmative or is_explicit_rerun:
            return await _start_grouped_kinetics(group_number, clear_existing=True)
        speak_txt(
            conn,
            f"第{group_number}组动力学目录里已经有数据。请确认是否重测并覆盖这些数据。",
        )
        return True

    if phase in {"kinetics_running", "await_kinetics_result"}:
        if not (
            _looks_like_uvvis_status_query(
                conn,
                original_text,
                filtered_text,
                inferred_step_id=step_id,
            )
            or is_affirmative
            or is_explicit_rerun
            or control_action in {"guide", "advance"}
        ):
            return False
        await _start_direct_intent_turn(conn, original_text)
        group_number = explicit_group_number or _extract_int_value(state.get("group_number"))
        if is_explicit_rerun:
            if group_number is None:
                _set_uvvis_direct_state(
                    conn,
                    step_id=step_id,
                    phase="await_grouped_samples",
                    session_key=session_key,
                    group_number=_get_current_uvvis_group_number(conn),
                    clear_existing_output=True,
                )
                speak_txt(conn, "这是第几组的动力学测量？请说第几组可以开始。")
                return True
            return await _start_grouped_kinetics(group_number, clear_existing=True)
        return await _check_grouped_kinetics_status(group_number)

    if phase == "await_finish_decision":
        if control_action == "repeat":
            await _start_direct_intent_turn(conn, original_text)
            _set_uvvis_direct_state(
                conn,
                step_id=step_id,
                phase="await_grouped_samples",
                session_key=session_key,
                group_number=_get_current_uvvis_group_number(conn),
                clear_existing_output=True,
            )
            speak_txt(
                conn,
                "请重新装好动力学样品。参比位放纯水，1号位留空，2和3号位放2号样品的反应液和参比液，4和5号位放4号样品的反应液和参比液。放好后告诉我是第几组并说开始。",
            )
            return True
        if control_action not in {"guide", "advance"}:
            if is_affirmative:
                await _start_direct_intent_turn(conn, original_text)
                speak_txt(
                    conn,
                    "如果还要继续重测这一轮，请说再测一轮；如果这个动力学步骤已经全部完成，请说当前步骤完成了。",
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
            speak_txt(conn, "好，等你把2到5号位按要求放好之后再告诉我。")
            return True
        if not is_affirmative:
            return False

        group_number = explicit_group_number or _extract_int_value(state.get("group_number"))
        if group_number is None:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "这是第几组的动力学测量？请说第几组可以开始。")
            return True

        await _start_direct_intent_turn(conn, original_text)
        return await _start_grouped_kinetics(
            group_number,
            clear_existing=is_explicit_rerun or bool(state.get("clear_existing_output")),
        )

    if phase in {"start_pending", ""}:
        if is_negative:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "好，等你准备好再告诉我。")
            return True

        if not (
            is_affirmative
            or _contains_any(
                _normalize_text_for_match(filtered_text),
                ("暗电流", "空气", "空白", "动力学", "开始", "测量", "调用mcp", "扫谱"),
            )
        ):
            return False

        if explicit_group_number is not None and (
            _looks_like_uvvis_start_scan_reply(filtered_text)
            or _contains_any(_normalize_text_for_match(filtered_text), ("开始", "测量", "扫描", "扫谱"))
        ):
            await _start_direct_intent_turn(conn, original_text)
            return await _start_grouped_kinetics(
                explicit_group_number,
                clear_existing=is_explicit_rerun,
            )

        if explicit_group_number is None:
            await _start_direct_intent_turn(conn, original_text)
            if is_explicit_rerun:
                _set_uvvis_direct_state(
                    conn,
                    step_id=step_id,
                    phase="await_grouped_samples",
                    session_key=session_key,
                    group_number=_get_current_uvvis_group_number(conn),
                    clear_existing_output=True,
                )
            speak_txt(conn, "这是第几组的动力学测量？请说第几组可以开始。")
            return True

        await _start_direct_intent_turn(conn, original_text)
        _set_uvvis_direct_state(
            conn,
            step_id=step_id,
            phase="await_grouped_samples",
            session_key=session_key,
            group_number=explicit_group_number,
            clear_existing_output=is_explicit_rerun,
        )
        speak_txt(
            conn,
            f"好，已记为第{explicit_group_number}组。请确认1号位为空，2/3号位为2号样品反应液/参比液，4/5号位为4号样品反应液/参比液，仪器参考位为纯水。放好后说可以开始。",
        )
        return True

    return False

def _compose_uvvis_step_rejection_reply(step_id: str) -> str:
    step_id = str(step_id or "").strip()
    if step_id == _UVVIS_SHARED_DARK_AIR_STEP_ID:
        return "当前实验图谱还没推进到 UV-Vis 的暗电流校正，先完成丁达尔现象观察。"
    if step_id == _UVVIS_SHARED_BLANK_STEP_ID:
        return "当前实验图谱还没推进到 UV-Vis 的纯水空白校正，先完成暗电流校正。"
    if step_id in _UVVIS_SAMPLE_RECORD_STEP_IDS:
        return "当前实验图谱还没推进到 1-5 号样品的批量光谱测量，先完成前面的装样准备。"
    if step_id in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
    }:
        return "当前实验图谱还没推进到对应的 400 纳米动力学测量，先完成前面的动力学配液准备。"
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
        ):
            return False

    inferred_step_id = step_id if _is_uvvis_step(step_id) else _infer_uvvis_step_id_from_context(
        conn,
        original_text,
        filtered_text,
    )
    if _looks_like_kinetics_rerun_request(original_text, filtered_text):
        inferred_step_id = _UVVIS_KINETICS_COMBINED_STEP_ID

    if inferred_step_id in {
        _UVVIS_KINETICS_SAMPLE2_STEP_ID,
        _UVVIS_KINETICS_SAMPLE4_STEP_ID,
        _UVVIS_KINETICS_COMBINED_STEP_ID,
    }:
        kinetics_state = _get_uvvis_direct_state(conn, inferred_step_id)
        if str(kinetics_state.get("phase") or "").strip() in {
            "kinetics_running",
            "await_kinetics_result",
        } and _looks_like_uvvis_status_query(
            conn,
            original_text,
            filtered_text,
            inferred_step_id=inferred_step_id,
        ):
            return await _handle_uvvis_kinetics_measurement(
                conn,
                original_text,
                filtered_text,
                inferred_step_id,
            )

    if (
        inferred_step_id == _UVVIS_KINETICS_COMBINED_STEP_ID
        and step_id != _UVVIS_KINETICS_COMBINED_STEP_ID
        and _looks_like_kinetics_rerun_request(original_text, filtered_text)
    ):
        redirected = await _try_redirect_experiment_step_fast(
            conn,
            _UVVIS_KINETICS_COMBINED_STEP_ID,
            log_reason="direct uvvis redirecting kinetics rerun request",
        )
        if not redirected:
            await _start_direct_intent_turn(conn, original_text)
            speak_txt(conn, "我还没有把实验图谱切回动力学测量步骤，先不启动重测，避免写错步骤。")
            return True
        step_id = _UVVIS_KINETICS_COMBINED_STEP_ID

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

    if step_id == _UVVIS_SHARED_BLANK_STEP_ID:
        return await _handle_uvvis_shared_blank_prep(conn, original_text, filtered_text, state)

    if step_id in _UVVIS_SAMPLE_RECORD_STEP_IDS:
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

    if step_id in _UVVIS_SAMPLE_LOAD_STEP_IDS:
        return await _handle_uvvis_sample_load_completion(conn, original_text, filtered_text)

    if step_id in _UVVIS_SAMPLE_CLEAN_STEP_IDS:
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
            await _reset_experiment_fresh_start_context(
                conn,
                allow_device_resume=False,
            )
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
        or ""
    ).strip()
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

    photo_tokens = ("拍照", "拍一张", "拍一下", "照一下", "拍摄", "照片", "拍吧")
    if not _contains_any(last_text, photo_tokens):
        return False

    negative_prompt_tokens = (
        "不要拍",
        "别拍",
        "先别拍",
        "不可以拍",
        "不能拍",
        "还不能拍",
        "不要拍照",
        "别拍照",
        "先别拍照",
        "不可以拍照",
        "不能拍照",
        "还不能拍照",
    )
    if _contains_any(last_text, negative_prompt_tokens):
        return False

    explicit_wait_tokens = (
        "得到肯定答复后再拍",
        "确认后再拍",
        "同意后再拍",
        "回复可以再拍",
        "允许拍照后再告诉我",
        "告诉我可以拍照",
        "等你允许后我再拍",
        "再说一声拍吧",
        "说一声拍吧",
        "再说一遍拍吧",
        "说一遍拍吧",
        "再说一声拍照",
        "说一声拍照",
        "再说一遍拍照",
        "说一遍拍照",
        "拍吧",
        "拍照吧",
        "拍一张吧",
        "拍一下吧",
        "直接拍吧",
        "开始拍吧",
    )
    if _contains_any(last_text, explicit_wait_tokens):
        return True

    if "告诉我" in last_text and ("可以拍照" in last_text or "拍吧" in last_text):
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
        question = f"请拍摄{sample_name}当前状态的照片。"
    else:
        question = "请拍摄当前样品的照片。"

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


def _is_timeout_like_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    return "timeout" in normalized or "超时" in normalized


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
    return "拍照这边超时了，我还没拿到结果。你可以稍后再说一次拍照，或者让我打开最近一张照片。"


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
        if sample_photo_name:
            request["photo_name"] = sample_photo_name
            request["append_timestamp"] = True
        if _is_sample_photo_retake_command(filtered_text):
            conn.logger.bind(tag=TAG).info(
                "sample photo retake requested; taking a new timestamped photo without graph rollback"
            )
        await _maybe_wait_before_photo_capture(conn, "direct_photo_command")
        return await _execute_server_photo_intent(
            conn,
            request,
            update_experiment_graph=False,
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
    if hasattr(conn, "append_experiment_interaction_log"):
        try:
            conn.append_experiment_interaction_log(
                "ASSISTANT",
                text,
                source="speak_txt",
            )
        except Exception:
            pass
