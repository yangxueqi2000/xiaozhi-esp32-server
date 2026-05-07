from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .uvvis_spoken import build_uvvis_spoken_response, extract_blank_baseline_state


_ARTIFACT_VALIDATION_SPECS = {
    "export_records_to_yaml": {
        "file_fields": ("yaml_path", "pdf_path"),
        "bool_fields": ("pdf_generated",),
        "directory_argument_fields": ("output_path",),
    },
    "xiaozhi_take_photo": {
        "file_fields": ("mirrored_path", "photo_path", "saved_photo_path", "local_path"),
        "directory_argument_fields": (),
    },
    "uvvis_scan_result": {
        "file_fields": ("output_csv", "absorbance_output_csv"),
        "directory_argument_fields": (),
    },
    "uvvis_scan_to_csv": {
        "file_fields": ("output_csv", "absorbance_output_csv"),
        "directory_argument_fields": (),
    },
    "xiaozhi_save_latest_photo_as": {
        "file_fields": ("local_path", "photo_path"),
        "directory_argument_fields": (),
    },
}

_EXPERIMENT_GRAPH_TOOLS = {
    "create_session",
    "close_session",
    "get_state",
    "get_overview",
    "list_steps",
    "get_step",
    "get_schema",
    "get_progress_summary",
    "get_current_progress",
    "get_modifiable_records",
    "export_records",
    "export_records_to_yaml",
    "start_trial",
    "cancel_trial",
    "add_field",
    "add_fields",
    "finish_trial",
    "can_proceed",
    "proceed_to_next_step",
    "redirect_to_step",
    "redo_trial",
    "modify_record",
}

_EXPERIMENT_GRAPH_MUTATING_TOOLS = {
    "create_session",
    "start_trial",
    "cancel_trial",
    "add_field",
    "add_fields",
    "finish_trial",
    "proceed_to_next_step",
    "redirect_to_step",
    "redo_trial",
    "modify_record",
}


def to_plain_data(payload):
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


def try_parse_json_text(text: str):
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


def extract_server_mcp_payload(raw_result):
    data = to_plain_data(raw_result)
    if isinstance(data, str):
        parsed = try_parse_json_text(data)
        return parsed if parsed is not None else data

    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                item_data = to_plain_data(item)
                if isinstance(item_data, dict):
                    text = item_data.get("text")
                    if isinstance(text, str):
                        parsed_text = try_parse_json_text(text)
                        if parsed_text is not None:
                            return parsed_text
                        if text.strip():
                            return text.strip()
        return data

    if isinstance(data, list):
        for item in data:
            extracted = extract_server_mcp_payload(item)
            if extracted is not None:
                return extracted
    return data


def serialize_result_for_llm(payload) -> str:
    data = extract_server_mcp_payload(payload)
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    if data is None:
        return ""
    return str(data)


def _collect_named_values(payload, target_keys):
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

    _visit(payload)
    return found


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _experiment_result_body(payload):
    if isinstance(payload, dict):
        nested = payload.get("result")
        if isinstance(nested, dict):
            return nested
        return payload
    return {}


def extract_experiment_session_id(payload, arguments: dict | None = None) -> str:
    body = _experiment_result_body(payload)
    for key in ("session_id", "sessionId"):
        value = str(body.get(key, "") or "").strip()
        if value:
            return value
    state = body.get("state")
    if isinstance(state, dict):
        for key in ("session_id", "sessionId"):
            value = str(state.get(key, "") or "").strip()
            if value:
                return value
    return str((arguments or {}).get("session_id", "") or "").strip()


def _coerce_positive_int(value) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def extract_experiment_group_number(payload, arguments: dict | None = None) -> int | None:
    body = _experiment_result_body(payload)

    def from_mapping(mapping) -> int | None:
        if not isinstance(mapping, dict):
            return None
        for key in ("current_group_number", "group_number", "experiment_group_number"):
            number = _coerce_positive_int(mapping.get(key))
            if number is not None:
                return number
        return None

    for mapping in (body, body.get("state"), body.get("current_progress"), body.get("progress")):
        number = from_mapping(mapping)
        if number is not None:
            return number
        if isinstance(mapping, dict):
            number = from_mapping(mapping.get("current_data"))
            if number is not None:
                return number

    summary = body.get("summary")
    if isinstance(summary, dict):
        for mapping in (summary, summary.get("current_progress"), summary.get("current_step")):
            number = from_mapping(mapping)
            if number is not None:
                return number

    return from_mapping(arguments or {})


def extract_experiment_message(payload) -> str:
    body = _experiment_result_body(payload)
    if not isinstance(body, dict):
        return ""

    for key in ("message", "error", "detail"):
        value = str(body.get(key, "") or "").strip()
        if value:
            return value

    state = body.get("state")
    if isinstance(state, dict):
        for key in ("message", "error", "detail"):
            value = str(state.get(key, "") or "").strip()
            if value:
                return value
    return ""


def _extract_experiment_current_step_id(payload) -> str:
    body = _experiment_result_body(payload)
    state = body.get("state")
    if isinstance(state, dict):
        value = str(state.get("current_step_id", "") or "").strip()
        if value:
            return value
    step = body.get("step")
    if isinstance(step, dict):
        value = str(step.get("id", "") or "").strip()
        if value:
            return value
    summary = body.get("summary")
    if isinstance(summary, dict):
        current_step = summary.get("current_step")
        if isinstance(current_step, dict):
            value = str(current_step.get("step_id", "") or "").strip()
            if value:
                return value
    return ""


def _payload_contains_experiment_step(payload) -> bool:
    body = _experiment_result_body(payload)
    return isinstance(body.get("step"), dict)


def _payload_contains_experiment_progress(payload) -> bool:
    body = _experiment_result_body(payload)
    return any(
        isinstance(body.get(key), dict)
        for key in ("summary", "current_progress", "progress", "state")
    )


def annotate_artifact_validation(
    payload,
    *,
    tool_name: str = "",
    arguments: dict | None = None,
):
    if not isinstance(payload, dict):
        return payload

    spec = _ARTIFACT_VALIDATION_SPECS.get(str(tool_name or "").strip())
    if not spec:
        return payload

    file_fields = tuple(spec.get("file_fields", ()))
    bool_fields = tuple(spec.get("bool_fields", ()))
    directory_argument_fields = tuple(spec.get("directory_argument_fields", ()))

    found_values = _collect_named_values(payload, file_fields + bool_fields)
    checked_files = []
    for field in file_fields:
        path_text = str(found_values.get(field, "") or "").strip()
        if not path_text:
            continue
        path = Path(path_text)
        checked_files.append(
            {
                "field": field,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    bool_status = {}
    for field in bool_fields:
        normalized = _normalize_bool(found_values.get(field))
        if normalized is not None:
            bool_status[field] = normalized

    checked_directories = []
    for field in directory_argument_fields:
        dir_text = str((arguments or {}).get(field, "") or "").strip()
        if not dir_text:
            continue
        path = Path(dir_text)
        checked_directories.append(
            {
                "field": field,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    all_reported_files_exist = bool(checked_files) and all(
        item["exists"] for item in checked_files
    )
    all_reported_directories_exist = (
        True
        if not checked_directories
        else all(item["exists"] for item in checked_directories)
    )

    expected_ok = all_reported_files_exist and all_reported_directories_exist
    pdf_generated = bool_status.get("pdf_generated")
    if pdf_generated is False:
        expected_ok = False

    annotated = dict(payload)
    annotated["artifact_validation"] = {
        "tool_name": str(tool_name or "").strip(),
        "checked_files": checked_files,
        "checked_directories": checked_directories,
        "reported_flags": bool_status,
        "all_reported_files_exist": all_reported_files_exist,
        "all_reported_directories_exist": all_reported_directories_exist,
        "all_expected_outputs_exist": expected_ok,
    }
    return annotated


def sync_server_mcp_payload_state(conn, *, tool_name: str = "", payload=None, arguments: dict | None = None):
    if conn is None:
        return

    actual_tool_name = str(tool_name or "").strip()
    setattr(conn, "_last_server_mcp_tool_name", actual_tool_name)
    setattr(conn, "_last_server_mcp_payload", payload)
    setattr(
        conn,
        "_last_server_mcp_sentence_id",
        str(getattr(conn, "sentence_id", "") or "").strip(),
    )
    current_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    tracked_sentence_id = str(
        getattr(conn, "_current_turn_server_mcp_sentence_id", "") or ""
    ).strip()
    if current_sentence_id and current_sentence_id == tracked_sentence_id:
        current_turn_tools = list(
            getattr(conn, "_current_turn_server_mcp_tool_names", []) or []
        )
    else:
        current_turn_tools = []
        setattr(conn, "_current_turn_server_mcp_sentence_id", current_sentence_id)
    if actual_tool_name:
        current_turn_tools.append(actual_tool_name)
        setattr(conn, "_current_turn_server_mcp_tool_names", current_turn_tools)

    if actual_tool_name in _EXPERIMENT_GRAPH_TOOLS:
        session_id = extract_experiment_session_id(payload, arguments=arguments)
        if session_id:
            setattr(conn, "experiment_session_id", session_id)

        group_number = extract_experiment_group_number(payload, arguments=arguments)
        if group_number is not None:
            setattr(conn, "experiment_current_group_number", group_number)

        current_step_id = _extract_experiment_current_step_id(payload)
        if current_step_id:
            setattr(conn, "experiment_current_step_id", current_step_id)
            if getattr(conn, "experiment_resume_recovery_required", False):
                setattr(conn, "experiment_resume_latest_current_step_id", current_step_id)

        if actual_tool_name == "get_step" or _payload_contains_experiment_step(payload):
            setattr(conn, "experiment_current_step", payload)

        if actual_tool_name in {"get_progress_summary", "get_current_progress", "get_state"} or _payload_contains_experiment_progress(payload):
            setattr(conn, "experiment_progress_summary", payload)

        if actual_tool_name in _EXPERIMENT_GRAPH_MUTATING_TOOLS:
            setattr(
                conn,
                "_experiment_graph_refresh_required",
                not bool(current_step_id),
            )
            setattr(conn, "_last_experiment_graph_mutation_tool", actual_tool_name)
        elif current_step_id:
            setattr(conn, "_experiment_graph_refresh_required", False)

    if actual_tool_name.startswith("uvvis_"):
        blank_baseline_state = extract_blank_baseline_state(payload)
        if blank_baseline_state is not None:
            setattr(conn, "_last_uvvis_blank_baseline_state", blank_baseline_state)
        if actual_tool_name == "uvvis_session":
            action = str((arguments or {}).get("action", "") or "").strip().lower()
            payload_session_key = ""
            if isinstance(payload, dict):
                payload_session_key = str(payload.get("session_key") or "").strip()
            if payload_session_key:
                setattr(conn, "_uvvis_session_key", payload_session_key)
            elif action == "release":
                setattr(conn, "_uvvis_session_key", "")

    if actual_tool_name != "export_records_to_yaml":
        return

    validation = payload.get("artifact_validation", {}) if isinstance(payload, dict) else {}
    setattr(
        conn,
        "_pending_export_report_validation",
        {
            "active": True,
            "tool_name": "export_records_to_yaml",
            "all_expected_outputs_exist": bool(
                validation.get("all_expected_outputs_exist", False)
            ),
        },
    )


def build_server_mcp_spoken_response(
    tool_name: str,
    payload,
    *,
    default_reply: str = "",
):
    actual_tool_name = str(tool_name or "").strip()
    if not actual_tool_name:
        return str(default_reply or "").strip()

    data = extract_server_mcp_payload(payload)
    if not isinstance(data, dict):
        return str(default_reply or "").strip()

    validation = data.get("artifact_validation")
    if not isinstance(validation, dict):
        validation = {}

    if actual_tool_name == "export_records_to_yaml":
        if not bool(validation.get("all_expected_outputs_exist", False)):
            return "实验报告还没有完整生成成功，请稍后再试。"
        return "实验报告已经生成好了。"

    if actual_tool_name == "xiaozhi_take_photo":
        if data.get("success") is False:
            return ""
        checked_files = validation.get("checked_files", [])
        if any(item.get("exists") for item in checked_files):
            return "拍好了，已经保存。"
        return "拍好了。"

    if actual_tool_name == "xiaozhi_save_latest_photo_as":
        if data.get("success") is False:
            return ""
        checked_files = validation.get("checked_files", [])
        if any(item.get("exists") for item in checked_files):
            return "已经保存好了。"
        return "照片还没有成功保存，请再试一次。"

    if actual_tool_name == "xiaozhi_preview_previous_photo":
        if data.get("success") is False:
            return ""
        return "已经切到上一张了。"

    if actual_tool_name == "xiaozhi_preview_local_file":
        if data.get("success") is False:
            return ""
        return str(default_reply or "已经打开照片了。").strip()

    uvvis_reply = build_uvvis_spoken_response(
        actual_tool_name,
        data,
        validation=validation,
        default_reply=default_reply,
    )
    if uvvis_reply is not None:
        return uvvis_reply

    return str(default_reply or "").strip()


def finalize_server_mcp_payload(
    raw_result,
    *,
    tool_name: str = "",
    arguments: dict | None = None,
):
    payload = extract_server_mcp_payload(raw_result)
    return annotate_artifact_validation(
        payload,
        tool_name=tool_name,
        arguments=arguments,
    )


def pick_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
