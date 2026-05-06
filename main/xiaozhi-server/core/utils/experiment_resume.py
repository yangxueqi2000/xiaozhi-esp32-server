import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logger import setup_logging
from core.utils import textUtils

TAG = __name__
logger = setup_logging()

RESUME_INTENT_PATTERNS = (
    re.compile(
        r"(继续|接着|恢复)(刚才|之前|上次)?(没做完|未完成)?的?(实验|实验流程|当前实验|当前流程|步骤)"
    ),
    re.compile(r"(继续|接着)(刚才|之前|上次)?没做完"),
)

EXPORT_RECORD_PATTERNS = (
    re.compile(r"(实验结束|结束实验|结束当前实验)"),
    re.compile(
        r"(生成|导出|写出|保存)(实验记录|实验报告|报告|记录yaml|记录YAML|yaml|YAML|pdf|PDF)"
    ),
)

TURN_SPLIT_RE = re.compile(r"(?=^\[[^\]]+\] \[TURN_START\])", re.MULTILINE)
TURN_END_RE = re.compile(r"^\[[^\]]+\] \[TURN_END\] chars=.*$", re.MULTILINE)
USER_LINE_RE = re.compile(r"^\[[^\]]+\] \[USER\] (?P<user>.*)$", re.MULTILINE)
TRANSCRIPT_STEP_RE = re.compile(r"\[current_step_id=(?P<step_id>[^\]]+)\]")
TRANSCRIPT_SESSION_RE = re.compile(
    r"\[experiment_session_id=(?P<session_id>[^\]]+)\]"
)
TRANSCRIPT_YAML_RE = re.compile(r"\[yaml=(?P<yaml_path>[^\]]+)\]")
TRANSCRIPT_ROLE_RE = re.compile(r"\[TRANSCRIPT\] \[(?P<role>USER|ASSISTANT)\]")
TRANSCRIPT_SOURCE_RE = re.compile(r"\[source=(?P<source>[^\]]+)\]")


class _PathFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _safe_filename(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text[:120] if len(text) > 120 else text


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def is_resume_experiment_request(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in RESUME_INTENT_PATTERNS)


def is_experiment_record_request(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in EXPORT_RECORD_PATTERNS)


def should_load_device_log_context(text: Any) -> bool:
    return is_resume_experiment_request(text) or is_experiment_record_request(text)


def _iter_codex_llm_configs(config: Dict[str, Any]):
    llm_map = config.get("LLM", {}) or {}
    if not isinstance(llm_map, dict):
        return

    preferred = str(config.get("codex_app", {}).get("llm_name", "")).strip()
    seen = set()

    if preferred:
        preferred_cfg = llm_map.get(preferred)
        if (
            isinstance(preferred_cfg, dict)
            and str(preferred_cfg.get("type", "")).strip() == "codex"
        ):
            seen.add(preferred)
            yield preferred, preferred_cfg

    for name, llm_cfg in llm_map.items():
        if name in seen:
            continue
        if isinstance(llm_cfg, dict) and str(llm_cfg.get("type", "")).strip() == "codex":
            yield str(name), llm_cfg


def _format_template_path(template: str, replacements: Dict[str, str]) -> str:
    text = str(template or "").strip()
    if not text:
        return ""
    try:
        return text.format_map(_PathFormatDict(replacements))
    except Exception:
        return text


def _iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _normalize_utterance_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _user_utterance_log_template(llm_cfg: Dict[str, Any]) -> str:
    explicit = str(llm_cfg.get("user_utterance_log_path", "")).strip()
    if explicit:
        return explicit

    stream_template = str(llm_cfg.get("stream_log_path", "")).strip()
    if not stream_template:
        return ""
    if stream_template.lower().endswith(".log"):
        return stream_template[:-4] + "_utterances.jsonl"
    return stream_template + ".utterances.jsonl"


def resolve_experiment_user_utterance_log_paths(
    config: Dict[str, Any], device_id: str
) -> List[Path]:
    normalized_device_id = str(device_id or "").strip()
    if not normalized_device_id:
        return []

    safe_device_id = _safe_filename(normalized_device_id)
    paths: List[Path] = []
    seen = set()

    for _, llm_cfg in _iter_codex_llm_configs(config):
        template = _user_utterance_log_template(llm_cfg)
        if not template:
            continue
        for candidate_device_id in (safe_device_id, normalized_device_id):
            path_text = _format_template_path(
                template,
                {
                    "device_id": candidate_device_id,
                    "session_key": safe_device_id or "resume",
                },
            )
            if not path_text:
                continue
            try:
                path = Path(path_text)
            except Exception:
                continue
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)

    return paths


def _select_preferred_user_utterance_log_path(
    config: Dict[str, Any], device_id: str
) -> Optional[Path]:
    candidates = resolve_experiment_user_utterance_log_paths(config, device_id)
    if not candidates:
        return None
    return candidates[0]


def append_user_utterance_log(
    config: Dict[str, Any],
    device_id: str,
    text: Any,
    *,
    source: str = "",
    speaker: str = "",
    language: str = "",
    chat_session_id: str = "",
    model_session_key: str = "",
    connection_session_id: str = "",
    experiment_session_id: str = "",
    current_step_id: str = "",
    experiment_yaml_path: str = "",
) -> Optional[str]:
    normalized_text = _normalize_utterance_text(text)
    normalized_device_id = str(device_id or "").strip()
    if not normalized_device_id or not normalized_text:
        return None

    target_path = _select_preferred_user_utterance_log_path(config, normalized_device_id)
    if target_path is None:
        return None

    entry = {
        "ts": _iso_timestamp(),
        "device_id": normalized_device_id,
        "source": str(source or "").strip(),
        "text": normalized_text,
        "speaker": str(speaker or "").strip(),
        "language": str(language or "").strip(),
        "chat_session_id": str(chat_session_id or "").strip(),
        "model_session_key": str(model_session_key or "").strip(),
        "connection_session_id": str(connection_session_id or "").strip(),
        "experiment_session_id": str(experiment_session_id or "").strip(),
        "current_step_id": str(current_step_id or "").strip(),
        "experiment_yaml_path": str(experiment_yaml_path or "").strip(),
    }

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"user utterance log write failed: {target_path} ({exc})"
        )
        return None

    return str(target_path)


def enrich_latest_user_utterance_log(
    config: Dict[str, Any],
    device_id: str,
    *,
    connection_session_id: str = "",
    experiment_session_id: str = "",
    current_step_id: str = "",
    experiment_yaml_path: str = "",
) -> Optional[str]:
    normalized_device_id = str(device_id or "").strip()
    if not normalized_device_id:
        return None

    normalized_connection_session_id = str(connection_session_id or "").strip()
    normalized_experiment_session_id = str(experiment_session_id or "").strip()
    normalized_current_step_id = str(current_step_id or "").strip()
    normalized_experiment_yaml_path = str(experiment_yaml_path or "").strip()

    if not (
        normalized_experiment_session_id
        or normalized_current_step_id
        or normalized_experiment_yaml_path
    ):
        return None

    target_path = _select_preferred_user_utterance_log_path(config, normalized_device_id)
    if target_path is None or not target_path.exists() or not target_path.is_file():
        return None

    try:
        lines = target_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"user utterance log update read failed: {target_path} ({exc})"
        )
        return None

    changed = False
    for index in range(len(lines) - 1, -1, -1):
        payload = str(lines[index] or "").strip()
        if not payload:
            continue
        try:
            item = json.loads(payload)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue

        entry_connection_session_id = str(
            item.get("connection_session_id", "")
        ).strip()
        if (
            normalized_connection_session_id
            and entry_connection_session_id != normalized_connection_session_id
        ):
            continue

        if (
            normalized_experiment_session_id
            and str(item.get("experiment_session_id", "")).strip()
            != normalized_experiment_session_id
        ):
            item["experiment_session_id"] = normalized_experiment_session_id
            changed = True
        if (
            normalized_current_step_id
            and str(item.get("current_step_id", "")).strip()
            != normalized_current_step_id
        ):
            item["current_step_id"] = normalized_current_step_id
            changed = True
        if (
            normalized_experiment_yaml_path
            and str(item.get("experiment_yaml_path", "")).strip()
            != normalized_experiment_yaml_path
        ):
            item["experiment_yaml_path"] = normalized_experiment_yaml_path
            changed = True

        if changed:
            lines[index] = json.dumps(item, ensure_ascii=False)
            try:
                text = "\n".join(lines)
                if text:
                    text += "\n"
                target_path.write_text(text, encoding="utf-8")
            except Exception as exc:
                logger.bind(tag=TAG).warning(
                    f"user utterance log update write failed: {target_path} ({exc})"
                )
                return None
        return str(target_path)

    return None


def resolve_experiment_log_paths(config: Dict[str, Any], device_id: str) -> List[Path]:
    normalized_device_id = str(device_id or "").strip()
    if not normalized_device_id:
        return []

    safe_device_id = _safe_filename(normalized_device_id)
    paths: List[Path] = []
    seen = set()

    for _, llm_cfg in _iter_codex_llm_configs(config):
        template = str(llm_cfg.get("stream_log_path", "")).strip()
        if not template:
            continue
        for candidate_device_id in (safe_device_id, normalized_device_id):
            path_text = _format_template_path(
                template,
                {
                    "device_id": candidate_device_id,
                    "session_key": safe_device_id or "resume",
                },
            )
            if not path_text:
                continue
            try:
                path = Path(path_text)
            except Exception:
                continue
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)

    return paths


def append_experiment_interaction_log(
    config: Dict[str, Any],
    device_id: str,
    text: Any,
    *,
    role: str,
    source: str = "",
    experiment_session_id: str = "",
    current_step_id: str = "",
    experiment_yaml_path: str = "",
) -> Optional[str]:
    normalized_device_id = str(device_id or "").strip()
    normalized_text = _normalize_utterance_text(text)
    normalized_role = str(role or "").strip().upper()
    if not normalized_device_id or not normalized_text or not normalized_role:
        return None

    paths = resolve_experiment_log_paths(config, normalized_device_id)
    if not paths:
        return None
    target_path = paths[0]

    source_text = str(source or "").strip()
    parts = [f"[{_iso_timestamp()}]", "[TRANSCRIPT]", f"[{normalized_role}]"]
    if source_text:
        parts.append(f"[source={source_text}]")
    if experiment_session_id:
        parts.append(f"[experiment_session_id={str(experiment_session_id).strip()}]")
    if current_step_id:
        parts.append(f"[current_step_id={str(current_step_id).strip()}]")
    if experiment_yaml_path:
        parts.append(f"[yaml={str(experiment_yaml_path).strip()}]")
    line = " ".join(parts) + f" {normalized_text}\n"

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"experiment interaction log write failed: {target_path} ({exc})"
        )
        return None
    return str(target_path)


def _select_existing_log_path(paths: List[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda item: item.stat().st_mtime)


def _select_existing_user_utterance_log_path(
    config: Dict[str, Any], device_id: str
) -> Optional[Path]:
    return _select_existing_log_path(
        resolve_experiment_user_utterance_log_paths(config, device_id)
    )


def _read_user_utterance_entries(
    log_path: Path,
    *,
    max_entries: int = 8,
) -> List[Dict[str, str]]:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"user utterance log read failed: {log_path} ({exc})"
        )
        return []

    entries: List[Dict[str, str]] = []
    for line in reversed(lines):
        payload = str(line or "").strip()
        if not payload:
            continue
        try:
            item = json.loads(payload)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        text = _normalize_utterance_text(item.get("text", ""))
        if not text:
            continue
        entries.append(
            {
                "text": text,
                "source": str(item.get("source", "")).strip(),
                "speaker": str(item.get("speaker", "")).strip(),
                "language": str(item.get("language", "")).strip(),
                "ts": str(item.get("ts", "")).strip(),
                "experiment_session_id": str(
                    item.get("experiment_session_id", "")
                ).strip(),
                "current_step_id": str(item.get("current_step_id", "")).strip(),
                "experiment_yaml_path": str(
                    item.get("experiment_yaml_path", "")
                ).strip(),
            }
        )
        if len(entries) >= max(1, int(max_entries)):
            break

    entries.reverse()
    return entries


def _extract_latest_user_utterance_snapshot(
    entries: List[Dict[str, str]],
) -> Dict[str, str]:
    latest_experiment_session_id = ""
    latest_current_step_id = ""
    latest_experiment_yaml_path = ""

    for entry in entries:
        experiment_session_id = str(entry.get("experiment_session_id", "")).strip()
        current_step_id = str(entry.get("current_step_id", "")).strip()
        experiment_yaml_path = str(entry.get("experiment_yaml_path", "")).strip()
        if experiment_session_id:
            latest_experiment_session_id = experiment_session_id
        if current_step_id:
            latest_current_step_id = current_step_id
        if experiment_yaml_path:
            latest_experiment_yaml_path = experiment_yaml_path

    return {
        "latest_experiment_session_id": latest_experiment_session_id,
        "latest_current_step_id": latest_current_step_id,
        "latest_experiment_yaml_path": latest_experiment_yaml_path,
    }


def _extract_latest_transcript_snapshot(log_text: str) -> Dict[str, str]:
    latest_experiment_session_id = ""
    latest_current_step_id = ""
    latest_experiment_yaml_path = ""

    for raw_line in reversed((log_text or "").splitlines()):
        line = str(raw_line or "").strip()
        if "[TRANSCRIPT]" not in line:
            continue
        if not latest_current_step_id:
            step_match = TRANSCRIPT_STEP_RE.search(line)
            if step_match:
                latest_current_step_id = str(step_match.group("step_id") or "").strip()
        if not latest_experiment_session_id:
            session_match = TRANSCRIPT_SESSION_RE.search(line)
            if session_match:
                latest_experiment_session_id = str(
                    session_match.group("session_id") or ""
                ).strip()
        if not latest_experiment_yaml_path:
            yaml_match = TRANSCRIPT_YAML_RE.search(line)
            if yaml_match:
                latest_experiment_yaml_path = str(
                    yaml_match.group("yaml_path") or ""
                ).strip()
        if (
            latest_current_step_id
            and latest_experiment_session_id
            and latest_experiment_yaml_path
        ):
            break

    return {
        "latest_experiment_session_id": latest_experiment_session_id,
        "latest_current_step_id": latest_current_step_id,
        "latest_experiment_yaml_path": latest_experiment_yaml_path,
    }


def read_transcript_entries(
    log_path: str | Path,
    *,
    max_entries: int = 0,
) -> List[Dict[str, str]]:
    path = Path(log_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"transcript log read failed: {path} ({exc})"
        )
        return []

    entries: List[Dict[str, str]] = []
    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line or "[TRANSCRIPT]" not in line:
            continue

        role_match = TRANSCRIPT_ROLE_RE.search(line)
        if not role_match:
            continue

        source_match = TRANSCRIPT_SOURCE_RE.search(line)
        session_match = TRANSCRIPT_SESSION_RE.search(line)
        step_match = TRANSCRIPT_STEP_RE.search(line)
        yaml_match = TRANSCRIPT_YAML_RE.search(line)
        text = line.rsplit("] ", 1)[-1].strip()
        if not text or text == line:
            continue

        entries.append(
            {
                "role": str(role_match.group("role") or "").strip(),
                "source": str(
                    source_match.group("source") if source_match else ""
                ).strip(),
                "experiment_session_id": str(
                    session_match.group("session_id") if session_match else ""
                ).strip(),
                "current_step_id": str(
                    step_match.group("step_id") if step_match else ""
                ).strip(),
                "experiment_yaml_path": str(
                    yaml_match.group("yaml_path") if yaml_match else ""
                ).strip(),
                "text": text,
            }
        )

    limit = max(0, int(max_entries or 0))
    if limit > 0:
        return entries[-limit:]
    return entries


def _parse_turns(log_text: str) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for chunk in TURN_SPLIT_RE.split(log_text or ""):
        if "[TURN_START]" not in chunk:
            continue
        end_match = TURN_END_RE.search(chunk)
        if not end_match:
            continue
        body = chunk[: end_match.start()]
        user_match = USER_LINE_RE.search(body)
        if not user_match:
            continue

        user_text = (user_match.group("user") or "").strip()
        if not user_text:
            user_text = _extract_wrapped_user_text(body)
        assistant_text = body[user_match.end() :].strip()
        if assistant_text:
            assistant_text = assistant_text.lstrip("\r\n").strip()

        if not user_text:
            continue
        turns.append(
            {
                "user": user_text,
                "assistant": assistant_text,
            }
        )
    return turns


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _extract_wrapped_user_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n")
    marker = "USER CHAT CONTENT"
    index = text.find(marker)
    if index < 0:
        return ""

    tail = text[index + len(marker) :].lstrip()
    header_prefix = (
        "The following additional message is the user's chat message, and should "
        "be followed to the best of your ability without interfering with the "
        "TOOL USE guidelines."
    )
    if tail.startswith(header_prefix):
        tail = tail[len(header_prefix) :].lstrip()

    for line in tail.split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        if candidate in {"====", "TOOL USE"}:
            continue
        return candidate
    return ""


def _strip_codex_prompt_noise(value: str) -> str:
    text = str(value or "")
    for marker in (
        "====\n\nTOOL USE",
        "\n====\n\nTOOL USE",
        "TOOL USE\n\nYou have access to a set of tools",
        "USER CHAT CONTENT",
        "<tool_call>",
    ):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def _clean_assistant_text(value: str) -> str:
    compact = _compact_text(_strip_codex_prompt_noise(value))
    if not compact:
        return ""
    if compact in {"====", "---", "-----"}:
        return ""
    cleaned = textUtils.filter_spoken_backstage_text(compact).strip()
    if cleaned:
        if cleaned in {"====", "---", "-----"}:
            return ""
        return cleaned
    return compact


def _parse_latest_partial_turn(log_text: str) -> Optional[Dict[str, str]]:
    chunks = TURN_SPLIT_RE.split(log_text or "")
    for chunk in reversed(chunks):
        if "[TURN_START]" not in chunk:
            continue
        user_match = USER_LINE_RE.search(chunk)
        if not user_match:
            continue

        user_text = (user_match.group("user") or "").strip()
        if not user_text:
            user_text = _extract_wrapped_user_text(chunk)
        assistant_text = chunk[user_match.end() :].strip()
        if assistant_text:
            assistant_text = assistant_text.lstrip("\r\n").strip()

        if not user_text:
            continue
        return {
            "user": user_text,
            "assistant": assistant_text,
        }
    return None


def _format_turn_block(turns: List[Dict[str, str]]) -> str:
    lines = [
        "恢复上下文（来自当前设备日志，可信）：",
        "这是同一设备在服务重启后的继续未完成实验请求。",
        "不要重新开始实验，也不要要求重新注册声纹；直接根据最近进度继续当前未完成步骤。",
        "最近实验记录摘录：",
    ]

    for turn in turns:
        user_text = _compact_text(turn.get("user", ""))
        assistant_text = _clean_assistant_text(turn.get("assistant", ""))
        if user_text:
            lines.append(f"学生：{user_text}")
        if assistant_text:
            lines.append(f"上一轮指导：{assistant_text}")

    return "\n".join(lines).strip()


def _format_user_utterance_block(entries: List[Dict[str, str]]) -> str:
    lines = [
        "恢复上下文（来自当前设备原始用户话语日志，可信）：",
        "这是同一设备在服务重启后的继续未完成实验请求。",
        "不要重新开始实验，也不要要求重新注册声纹；直接根据最近进度继续当前未完成步骤。",
        "最近用户原始话语摘录：",
    ]

    latest_snapshot = _extract_latest_user_utterance_snapshot(entries)
    latest_experiment_session_id = latest_snapshot["latest_experiment_session_id"]
    latest_current_step_id = latest_snapshot["latest_current_step_id"]

    if latest_experiment_session_id or latest_current_step_id:
        snapshot_parts: List[str] = []
        if latest_experiment_session_id:
            snapshot_parts.append(
                f"experiment_session_id={latest_experiment_session_id}"
            )
        if latest_current_step_id:
            snapshot_parts.append(f"current_step_id={latest_current_step_id}")
        lines.append("最近一次已知实验快照：" + "，".join(snapshot_parts))

    for entry in entries:
        text = _normalize_utterance_text(entry.get("text", ""))
        if text:
            lines.append(f"学生：{text}")

    return "\n".join(lines).strip()


def _format_partial_turn_block(turn: Dict[str, str]) -> str:
    lines = [
        "恢复上下文（来自当前设备日志，可信）：",
        "这是同一设备在服务重启后的继续未完成实验请求。",
        "最近一次未完成轮次摘录：",
    ]

    user_text = _compact_text(turn.get("user", ""))
    assistant_text = _clean_assistant_text(turn.get("assistant", ""))
    if user_text:
        lines.append(f"学生：{user_text}")
    if assistant_text:
        lines.append(f"上一轮未完成输出：{assistant_text}")

    return "\n".join(lines).strip()


def build_resume_context(
    config: Dict[str, Any],
    device_id: str,
    *,
    max_turns: int = 4,
    max_chars: int = 2800,
) -> Optional[Dict[str, str]]:
    user_log_path = _select_existing_user_utterance_log_path(config, device_id)
    if user_log_path is not None:
        user_entries = _read_user_utterance_entries(
            user_log_path,
            max_entries=max(1, int(max_turns)),
        )
        if user_entries:
            while user_entries:
                context_text = _format_user_utterance_block(user_entries)
                if len(context_text) <= max_chars or len(user_entries) == 1:
                    latest_snapshot = _extract_latest_user_utterance_snapshot(
                        user_entries
                    )
                    return {
                        "log_path": str(user_log_path),
                        "context_text": context_text[:max_chars],
                        "turn_count": str(len(user_entries)),
                        "latest_experiment_session_id": latest_snapshot.get(
                            "latest_experiment_session_id", ""
                        ),
                        "latest_current_step_id": latest_snapshot.get(
                            "latest_current_step_id", ""
                        ),
                        "latest_experiment_yaml_path": latest_snapshot.get(
                            "latest_experiment_yaml_path", ""
                        ),
                    }
                user_entries = user_entries[1:]

    candidates = resolve_experiment_log_paths(config, device_id)
    log_path = _select_existing_log_path(candidates)
    if log_path is None:
        logger.bind(tag=TAG).info(
            f"resume context skipped: no device log found for device_id={device_id}"
        )
        return None

    try:
        log_text = log_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.bind(tag=TAG).warning(f"resume log read failed: {log_path} ({exc})")
        return None

    transcript_snapshot = _extract_latest_transcript_snapshot(log_text)
    turns = _parse_turns(log_text)
    if not turns:
        partial_turn = _parse_latest_partial_turn(log_text)
        if partial_turn:
            context_text = _format_partial_turn_block(partial_turn)
            if context_text:
                return {
                    "log_path": str(log_path),
                    "context_text": context_text[:max_chars],
                    "turn_count": "0",
                    "latest_experiment_session_id": transcript_snapshot.get(
                        "latest_experiment_session_id", ""
                    ),
                    "latest_current_step_id": transcript_snapshot.get(
                        "latest_current_step_id", ""
                    ),
                    "latest_experiment_yaml_path": transcript_snapshot.get(
                        "latest_experiment_yaml_path", ""
                    ),
                }

        tail = _compact_text(_strip_codex_prompt_noise(log_text[-max_chars:]))
        if tail:
            context_text = (
                "恢复上下文（来自当前设备日志，可信）：\n"
                "这是同一设备在服务重启后的继续未完成实验请求。\n"
                "最近日志尾部：\n"
                f"{tail}"
            )
            return {
                "log_path": str(log_path),
                "context_text": context_text[:max_chars],
                "turn_count": "0",
                "latest_experiment_session_id": transcript_snapshot.get(
                    "latest_experiment_session_id", ""
                ),
                "latest_current_step_id": transcript_snapshot.get(
                    "latest_current_step_id", ""
                ),
                "latest_experiment_yaml_path": transcript_snapshot.get(
                    "latest_experiment_yaml_path", ""
                ),
            }
        return None

    selected = turns[-max(1, int(max_turns)) :]
    while selected:
        context_text = _format_turn_block(selected)
        if len(context_text) <= max_chars or len(selected) == 1:
            return {
                "log_path": str(log_path),
                "context_text": context_text[:max_chars],
                "turn_count": str(len(selected)),
                "latest_experiment_session_id": transcript_snapshot.get(
                    "latest_experiment_session_id", ""
                ),
                "latest_current_step_id": transcript_snapshot.get(
                    "latest_current_step_id", ""
                ),
                "latest_experiment_yaml_path": transcript_snapshot.get(
                    "latest_experiment_yaml_path", ""
                ),
            }
        selected = selected[1:]

    return None


def build_resume_tool_message(
    config: Dict[str, Any],
    device_id: str,
    query: str,
) -> Optional[Dict[str, str]]:
    if not should_load_device_log_context(query):
        return None

    is_record_request = is_experiment_record_request(query)
    resume_context = build_resume_context(
        config,
        device_id,
        max_turns=8 if is_record_request else 4,
        max_chars=4200 if is_record_request else 2800,
    )
    if not resume_context:
        return None

    content_lines = [resume_context["context_text"]]
    content_lines.append(f"当前设备日志文件：{resume_context['log_path']}")
    if is_record_request:
        content_lines.append(
            "当前请求涉及实验记录导出；先复用最近一次已知 experiment_session_id 直接尝试导出。"
            "只有当工具明确返回 session_id not found、记录缺失或导出失败时，才继续读取这个设备日志文件的更早内容后再补记录。"
        )
    content_lines.append("当前用户消息：" + _compact_text(query))
    content = "\n".join(content_lines).strip()
    return {
        "role": "tool",
        "tool_call_id": "resume_context",
        "content": content,
    }


def dump_resume_context(config: Dict[str, Any], device_id: str) -> str:
    context = build_resume_context(config, device_id)
    if not context:
        return "{}"
    return json.dumps(context, ensure_ascii=False, indent=2)
