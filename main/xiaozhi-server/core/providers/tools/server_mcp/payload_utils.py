from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ARTIFACT_VALIDATION_SPECS = {
    "export_records_to_yaml": {
        "file_fields": ("yaml_path", "pdf_path"),
        "bool_fields": ("pdf_generated",),
        "directory_argument_fields": ("output_path",),
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


def sync_server_mcp_payload_state(conn, *, tool_name: str = "", payload=None):
    if conn is None:
        return

    setattr(conn, "_last_server_mcp_tool_name", str(tool_name or "").strip())
    setattr(conn, "_last_server_mcp_payload", payload)

    if str(tool_name or "").strip() != "export_records_to_yaml":
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

    if actual_tool_name == "uvvis_scan_result":
        result = data.get("result")
        if not isinstance(result, dict):
            result = {}
        lambda_max_nm = result.get("lambda_max_nm")
        max_absorbance = result.get("max_absorbance")
        if lambda_max_nm is not None and max_absorbance is not None:
            return (
                f"最大吸收波长在{lambda_max_nm}纳米，"
                f"最大吸光度是{max_absorbance}。"
            )
        if validation and not bool(validation.get("all_expected_outputs_exist", True)):
            return "这次扫描结果还没有完整取到，请稍后再试。"
        return str(default_reply or "").strip()

    if actual_tool_name == "uvvis_scan_status":
        state = pick_text(data.get("state"))
        result = data.get("result")
        if not state and isinstance(result, dict):
            state = pick_text(result.get("state"))
        state = state.lower()
        if state == "running":
            return "扫描还在进行中。"
        if state == "queued":
            return "扫描请求已经发出，正在等待真正开始。"
        if state in {"failed", "error", "cancelled", "canceled"}:
            return "这次扫描失败了，请再试一次。"
        if state == "succeeded":
            return "扫描已经结束了。"
        return str(default_reply or "").strip()

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
