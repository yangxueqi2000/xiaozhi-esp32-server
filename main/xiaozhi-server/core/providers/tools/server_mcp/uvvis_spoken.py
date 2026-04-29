from __future__ import annotations

from pathlib import Path
from typing import Any


_FAILED_SCAN_STATES = {"failed", "error", "cancelled", "canceled"}
_UVVIS_MEASUREMENT_TOOL_NAMES = {"uvvis_measure_spectra", "uvvis_measure_kinetics"}


def _pick_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


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


def extract_blank_baseline_state(payload):
    if not isinstance(payload, dict):
        return None

    found = _collect_named_values(
        payload,
        (
            "blank_baseline_exists",
            "blank_baseline_status",
            "blank_baseline_csv",
            "blank_baseline_manifest_json",
        ),
    )
    if not found:
        return None

    exists = _normalize_bool(found.get("blank_baseline_exists"))
    if exists is None:
        csv_path = str(found.get("blank_baseline_csv", "") or "").strip()
        exists = bool(csv_path) and Path(csv_path).exists()

    return {
        "blank_baseline_exists": bool(exists),
        "blank_baseline_status": str(found.get("blank_baseline_status", "") or "").strip(),
        "blank_baseline_csv": str(found.get("blank_baseline_csv", "") or "").strip(),
        "blank_baseline_manifest_json": str(
            found.get("blank_baseline_manifest_json", "") or ""
        ).strip(),
    }


def _extract_scan_state(payload) -> str:
    if not isinstance(payload, dict):
        return ""

    state = _pick_text(payload.get("state")).lower()
    if state:
        return state

    result = payload.get("result")
    if isinstance(result, dict):
        return _pick_text(result.get("state")).lower()
    return ""


def _has_saved_artifact(validation) -> bool:
    if not isinstance(validation, dict):
        return False

    checked_files = validation.get("checked_files", [])
    if not isinstance(checked_files, list):
        return False

    return any(
        isinstance(item, dict) and item.get("exists")
        for item in checked_files
    )


def build_uvvis_spoken_response(
    tool_name: str,
    payload,
    *,
    validation=None,
    default_reply: str = "",
):
    actual_tool_name = str(tool_name or "").strip()
    if not actual_tool_name.startswith("uvvis_"):
        return None

    data = payload if isinstance(payload, dict) else {}
    validation = validation if isinstance(validation, dict) else {}

    if actual_tool_name == "uvvis_scan_result":
        result = data.get("result")
        if not isinstance(result, dict):
            result = {}
        if validation and not _has_saved_artifact(validation):
            return "这次扫描结果还没有保存到指定位置，请稍后再试。"
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
        state = _extract_scan_state(data)
        if state == "running":
            return "扫描还在进行中。"
        if state == "queued":
            return "扫描请求已经发出，正在等待真正开始。"
        if state in _FAILED_SCAN_STATES:
            return "这次扫描失败了，请再试一次。"
        if state == "succeeded":
            return "扫描已经结束了。"
        return str(default_reply or "").strip()

    if actual_tool_name in _UVVIS_MEASUREMENT_TOOL_NAMES:
        baseline_state = extract_blank_baseline_state(data)
        if baseline_state is not None and not baseline_state.get("blank_baseline_exists", True):
            return "还没有空白基线，请先确认空白基线已经准备好，再继续。"
        return str(default_reply or "").strip()

    return None
