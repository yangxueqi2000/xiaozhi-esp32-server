from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict

from plugins_func.register import Action, ActionResponse

from .payload_utils import build_server_mcp_spoken_response, pick_text
from .uvvis_spoken import (
    build_uvvis_scan_start_failed_response,
    build_uvvis_scan_status_followup_response,
    get_uvvis_scan_context_required_response,
    get_uvvis_scan_result_incomplete_response,
    get_uvvis_scan_result_not_saved_response,
    get_uvvis_scan_start_pending_response,
)


FAILED_SCAN_STATES = {"failed", "error", "cancelled", "canceled"}
RUNNING_SCAN_STATE = "running"
QUEUED_SCAN_STATE = "queued"
_UVVIS_SHARED_SPECTRA_STEP_IDS = {
    "step_3_uv_vis_shared_dark_air_prep",
    "step_3_uv_vis_shared_dark_blank_prep",
}
_UVVIS_SHARED_BLANK_PHASES = {"await_pure_water_blank"}


def _resolve_experiment_uvvis_output_root(conn) -> Path | None:
    yaml_path = str(getattr(conn, "experiment_yaml_path", "") or "").strip()
    if not yaml_path and hasattr(conn, "_resolve_experiment_yaml_path"):
        try:
            yaml_path = str(conn._resolve_experiment_yaml_path() or "").strip()
        except Exception:
            yaml_path = ""
    if not yaml_path:
        return None

    try:
        yaml_file = Path(yaml_path).expanduser().resolve()
    except Exception:
        return None

    if yaml_file.name.lower().endswith((".yaml", ".yml")) and yaml_file.parent.name.lower() == "configs":
        return (yaml_file.parent.parent / "data" / "uv_data_common").resolve()
    return None


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _extract_uvvis_scan_context(
    payload, *, fallback_task_id: str = ""
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    task_id = pick_text(payload.get("task_id"), fallback_task_id)
    if not task_id:
        return None

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    result = payload.get("result")
    if not isinstance(result, dict):
        result = {}

    return {
        "task_id": task_id,
        "state": pick_text(payload.get("state")),
        "output_csv": pick_text(result.get("output_csv"), parameters.get("output_csv")),
        "absorbance_output_csv": pick_text(
            result.get("absorbance_output_csv"),
            parameters.get("absorbance_output_csv"),
        ),
        "sample_name": pick_text(parameters.get("sample_name")),
        "payload": payload,
    }


def _extract_uvvis_scan_state(payload) -> str:
    if not isinstance(payload, dict):
        return ""

    state = pick_text(payload.get("state")).lower()
    if state:
        return state

    result = payload.get("result")
    if isinstance(result, dict):
        return pick_text(result.get("state")).lower()
    return ""


def _read_absorbance_peak(absorbance_csv: str) -> dict[str, Any] | None:
    path_text = str(absorbance_csv or "").strip()
    if not path_text:
        return None

    path = Path(path_text)
    if not path.exists():
        return None

    max_row = None
    valid_points = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    wavelength_nm = float(row.get("wavelength_nm", "nan"))
                    absorbance = float(row.get("absorbance", "nan"))
                except Exception:
                    continue
                if not math.isfinite(wavelength_nm) or not math.isfinite(absorbance):
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


def _extract_scan_peak(payload) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    result = payload.get("result")
    if isinstance(result, dict):
        lambda_max_nm = result.get("lambda_max_nm")
        max_absorbance = result.get("max_absorbance")
        if lambda_max_nm is not None and max_absorbance is not None:
            try:
                return {
                    "lambda_max_nm": float(lambda_max_nm),
                    "max_absorbance": float(max_absorbance),
                }
            except Exception:
                pass
        absorbance_path = pick_text(result.get("absorbance_output_csv"))
        fallback = _read_absorbance_peak(absorbance_path)
        if fallback is not None:
            return fallback

    context = _extract_uvvis_scan_context(payload)
    if context is not None:
        fallback = _read_absorbance_peak(context.get("absorbance_output_csv", ""))
        if fallback is not None:
            return fallback
    return None


def _has_saved_scan_artifact(payload) -> bool:
    if not isinstance(payload, dict):
        return False

    validation = payload.get("artifact_validation")
    if not isinstance(validation, dict):
        return False

    checked_files = validation.get("checked_files", [])
    if not isinstance(checked_files, list):
        return False

    return any(
        isinstance(item, dict) and item.get("exists")
        for item in checked_files
    )


class UVVisScanRule:
    def __init__(
        self,
        conn,
        manager_getter: Callable[[], Any],
    ) -> None:
        self.conn = conn
        self._manager_getter = manager_getter

    def before_execute(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
    ) -> ActionResponse | None:
        if actual_tool_name == "uvvis_connect":
            self._inject_force_reconnect_flag(arguments)
            self.conn.logger.info(
                "uvvis_connect prepared: force_reconnect=%s pending=%s has_succeeded=%s arguments=%s",
                arguments.get("force_reconnect"),
                getattr(self.conn, "_uvvis_connect_force_reconnect_pending", False),
                getattr(self.conn, "_uvvis_connect_has_succeeded", False),
                arguments,
            )
            return None

        if actual_tool_name != "uvvis_scan_batch":
            if actual_tool_name in {"uvvis_scan_status", "uvvis_scan_result"}:
                if not pick_text(arguments.get("task_id")):
                    return ActionResponse(
                        action=Action.RESPONSE,
                        response=get_uvvis_scan_context_required_response(),
                    )
            return None

        return ActionResponse(
            action=Action.REQLLM,
            result=(
                '{"success": false, '
                '"error": "uvvis_scan_batch is disabled for this experiment to avoid long blocking scans and request timeouts.", '
                '"recommended_action": "Use uvvis_scan_start for each sample, wait for the student to report that the scan has finished, then call uvvis_scan_status and uvvis_scan_result with the saved task_id."}'
            ),
        )

    def prepare_arguments(self, actual_tool_name: str, arguments: Dict[str, Any]) -> None:
        self._inject_uvvis_native_output_dir(actual_tool_name, arguments)
        self._inject_uvvis_scan_output_paths(actual_tool_name, arguments)
        self._inject_saved_uvvis_task_id(actual_tool_name, arguments)

    async def after_execute(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
        payload,
    ) -> ActionResponse | None:
        if actual_tool_name == "uvvis_connect":
            self.conn._uvvis_connect_force_reconnect_pending = False
            self.conn._uvvis_connect_has_succeeded = True
            self.conn.logger.info("uvvis_connect completed successfully: payload=%s", payload)
            return None

        if actual_tool_name == "uvvis_scan_start":
            start_error = self._extract_uvvis_scan_start_error(payload)
            if start_error:
                self.conn.logger.warning(
                    f"uvvis scan start rejected: {start_error}, payload={payload}"
                )
                return ActionResponse(
                    action=Action.ERROR,
                    response=build_uvvis_scan_start_failed_response(start_error),
                )
            return await self._handle_uvvis_scan_start(payload)

        if actual_tool_name == "uvvis_scan_status":
            self._refresh_uvvis_scan_context(
                payload,
                fallback_task_id=arguments.get("task_id", ""),
            )
            return self._handle_uvvis_scan_status(payload)

        if actual_tool_name == "uvvis_scan_result":
            self._refresh_uvvis_scan_context(
                payload,
                fallback_task_id=arguments.get("task_id", ""),
            )
            context = _extract_uvvis_scan_context(
                payload,
                fallback_task_id=arguments.get("task_id", ""),
            )
            if context is None:
                return ActionResponse(
                    action=Action.RESPONSE,
                    response=get_uvvis_scan_context_required_response(),
                )
            self._clear_pending_scan_start_confirmation(context.get("task_id", ""))
            state = _extract_uvvis_scan_state(payload)
            if state and state != "succeeded":
                reply = build_uvvis_scan_status_followup_response(payload)
                if reply:
                    return ActionResponse(
                        action=Action.RESPONSE,
                        response=reply,
                    )
                return None
            peak = _extract_scan_peak(payload)
            if peak is not None:
                result = payload.get("result")
                if not isinstance(result, dict):
                    result = {}
                    payload["result"] = result
                result.setdefault("lambda_max_nm", peak.get("lambda_max_nm"))
                result.setdefault("max_absorbance", peak.get("max_absorbance"))
                if peak.get("peak_source_csv"):
                    result.setdefault("peak_source_csv", peak.get("peak_source_csv"))
            has_saved_artifact = _has_saved_scan_artifact(payload)
            if not has_saved_artifact:
                return ActionResponse(
                    action=Action.RESPONSE,
                    response=get_uvvis_scan_result_not_saved_response(),
                )
            peak = _extract_scan_peak(payload)
            if peak is None:
                return ActionResponse(
                    action=Action.RESPONSE,
                    response=get_uvvis_scan_result_incomplete_response(),
                )
            reply = build_server_mcp_spoken_response("uvvis_scan_result", payload)
            if reply:
                return ActionResponse(
                    action=Action.RESPONSE,
                    response=reply,
                )

        if actual_tool_name in {"uvvis_measure_spectra", "uvvis_measure_kinetics"}:
            reply = build_server_mcp_spoken_response(actual_tool_name, payload)
            if reply:
                return ActionResponse(
                    action=Action.RESPONSE,
                    response=reply,
                )

        return None

    async def cleanup(self) -> None:
        return None

    def handle_execute_error(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
        error: Exception,
    ) -> None:
        if actual_tool_name != "uvvis_connect":
            return
        self.conn._uvvis_connect_force_reconnect_pending = True
        self.conn.logger.warning("uvvis_connect failed: error=%s arguments=%s", error, arguments)

    def _inject_force_reconnect_flag(self, arguments: Dict[str, Any]) -> None:
        if "force_reconnect" in arguments:
            return
        if getattr(self.conn, "_uvvis_connect_force_reconnect_pending", False):
            arguments["force_reconnect"] = True
            return
        if not getattr(self.conn, "_uvvis_connect_has_succeeded", False):
            arguments["force_reconnect"] = True

    def _inject_saved_uvvis_task_id(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        if actual_tool_name not in {"uvvis_scan_status", "uvvis_scan_result"}:
            return
        if str(arguments.get("task_id", "")).strip():
            return
        context = getattr(self.conn, "_uvvis_scan_context", None)
        if isinstance(context, dict):
            task_id = str(context.get("task_id", "")).strip()
            if task_id:
                arguments["task_id"] = task_id

    @staticmethod
    def _normalize_sample_folder_name(sample_name: str) -> str:
        raw = str(sample_name or "").strip()
        if not raw:
            return "sample_unknown"

        lower = raw.lower()
        if re.fullmatch(r"sample[_-]?\d+", lower):
            return lower.replace("_", "").replace("-", "")

        num_match = re.search(r"(\d+)", raw)
        if num_match and ("sample" in lower or "样品" in raw or "号" in raw):
            return f"sample{num_match.group(1)}"

        safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", raw).strip("._-")
        return safe or "sample_unknown"

    @staticmethod
    def _normalize_device_id(device_id: str) -> str:
        raw = str(device_id or "").strip().lower()
        safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", raw).strip("._-")
        return safe or "unknown_device"

    @staticmethod
    def _normalize_group_number(group_number) -> int | None:
        if isinstance(group_number, bool) or group_number in (None, ""):
            return None
        try:
            normalized = int(group_number)
        except (TypeError, ValueError):
            return None
        return normalized if normalized >= 1 else None

    @staticmethod
    def _format_group_dir_name(group_number: int) -> str:
        return f"group_{int(group_number):02d}"

    def _resolve_uvvis_output_root_dir(self) -> Path:
        override_root = str(self.conn.config.get("uvvis_scan_output_root", "")).strip()
        if override_root:
            return Path(override_root).resolve()

        experiment_root = _resolve_experiment_uvvis_output_root(self.conn)
        if experiment_root is not None:
            return experiment_root

        return (Path("data") / "uv_data_common").resolve()

    def _resolve_runtime_group_number(self) -> int | None:
        return self._normalize_group_number(
            getattr(self.conn, "experiment_current_group_number", None)
        )

    def _resolve_runtime_device_dir(self, *, include_group: bool = False) -> Path:
        device_id = str(getattr(self.conn, "device_id", "") or "").strip()
        if not device_id and isinstance(getattr(self.conn, "headers", None), dict):
            headers = getattr(self.conn, "headers", {}) or {}
            device_id = str(
                headers.get("device-id")
                or headers.get("Device-Id")
                or headers.get("device_id")
                or ""
            ).strip()

        normalized_device_id = self._normalize_device_id(device_id)
        device_dir = (self._resolve_uvvis_output_root_dir() / normalized_device_id).resolve()
        if include_group:
            group_number = self._resolve_runtime_group_number()
            if group_number is not None:
                device_dir = (device_dir / self._format_group_dir_name(group_number)).resolve()
        return device_dir

    def _resolve_uvvis_shared_output_dir(self) -> Path:
        return self._resolve_uvvis_output_root_dir()

    def _should_use_shared_uvvis_output_dir(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        ready_for_samples = _normalize_bool(arguments.get("ready_for_samples"))
        if actual_tool_name == "uvvis_measure_kinetics":
            return ready_for_samples is False

        if actual_tool_name != "uvvis_measure_spectra":
            return False

        if ready_for_samples is False:
            return True

        current_step_id = str(getattr(self.conn, "experiment_current_step_id", "") or "").strip()
        if current_step_id in _UVVIS_SHARED_SPECTRA_STEP_IDS:
            return True

        direct_state = getattr(self.conn, "_uvvis_direct_state", None)
        if not isinstance(direct_state, dict):
            return False

        direct_step_id = str(direct_state.get("step_id", "") or "").strip()
        if direct_step_id in _UVVIS_SHARED_SPECTRA_STEP_IDS:
            return True

        direct_phase = str(direct_state.get("phase", "") or "").strip()
        return direct_phase in _UVVIS_SHARED_BLANK_PHASES

    def _inject_uvvis_native_output_dir(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        if actual_tool_name not in {"uvvis_measure_spectra", "uvvis_measure_kinetics"}:
            return
        if pick_text(arguments.get("output_dir")):
            return

        if self._should_use_shared_uvvis_output_dir(actual_tool_name, arguments):
            target_dir = self._resolve_uvvis_shared_output_dir()
        else:
            target_dir = self._resolve_runtime_device_dir(include_group=True)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.conn.logger.warning(
                f"failed to create uvvis native output dir {target_dir}: {exc}"
            )
        arguments["output_dir"] = str(target_dir)

    def _inject_uvvis_scan_output_paths(
        self,
        actual_tool_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        if actual_tool_name not in {"uvvis_scan_start", "uvvis_scan_to_csv"}:
            return

        sample_name = pick_text(arguments.get("sample_name"))
        if not sample_name:
            context = getattr(self.conn, "_uvvis_scan_context", None)
            if isinstance(context, dict):
                sample_name = pick_text(context.get("sample_name"))

        sample_folder = self._normalize_sample_folder_name(sample_name)
        device_dir = self._resolve_runtime_device_dir(include_group=True)
        target_dir = device_dir / sample_folder

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.conn.logger.warning(
                f"failed to create uvvis output dir {target_dir}: {exc}"
            )

        arguments["sample_name"] = sample_name or sample_folder
        arguments["output_csv"] = str((target_dir / "uvvis_scan_latest_sample.csv").resolve())
        arguments["absorbance_output_csv"] = str(
            (target_dir / "uvvis_scan_latest_absorbance.csv").resolve()
        )

    @staticmethod
    def _extract_uvvis_scan_start_error(payload) -> str:
        if isinstance(payload, dict):
            if payload.get("success") is False:
                return pick_text(payload.get("message"), payload.get("error")) or (
                    "返回 success=false。"
                )

            state = str(payload.get("state", "")).strip().lower()
            if state in FAILED_SCAN_STATES:
                return pick_text(payload.get("error"), payload.get("message")) or (
                    f"state={state}"
                )

            task_id = pick_text(payload.get("task_id"))
            if task_id:
                return ""

            nested = payload.get("result")
            if isinstance(nested, dict):
                if nested.get("success") is False:
                    return pick_text(
                        nested.get("message"),
                        nested.get("error"),
                    ) or "result.success=false。"
                nested_state = str(nested.get("state", "")).strip().lower()
                if nested_state in FAILED_SCAN_STATES:
                    return pick_text(
                        nested.get("error"),
                        nested.get("message"),
                    ) or f"result.state={nested_state}"
                if pick_text(nested.get("task_id")):
                    return ""

            return "工具没有返回 task_id。"

        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return "工具返回了空结果。"
            lower = text.lower()
            if "failed" in lower or "error" in lower or "失败" in text:
                return text
            return "工具没有返回 task_id。"

        if payload is None:
            return "工具没有返回结果。"

        return "返回结果格式异常。"

    def _refresh_uvvis_scan_context(
        self,
        payload,
        *,
        fallback_task_id: str = "",
    ) -> None:
        context = _extract_uvvis_scan_context(payload, fallback_task_id=fallback_task_id)
        if context is not None:
            self.conn._uvvis_scan_context = context

    async def _handle_uvvis_scan_start(self, payload) -> ActionResponse | None:
        context = _extract_uvvis_scan_context(payload)
        if context is None:
            return None

        self.conn._uvvis_scan_context = context
        self._mark_pending_scan_start_confirmation(context.get("task_id", ""))
        return ActionResponse(
            action=Action.RESPONSE,
            response=get_uvvis_scan_start_pending_response(),
        )

    def _handle_uvvis_scan_status(self, payload) -> ActionResponse | None:
        context = _extract_uvvis_scan_context(payload)
        if context is None:
            return None

        pending_task_id = self._get_pending_scan_start_task_id()
        task_id = context.get("task_id", "")
        if not pending_task_id or task_id != pending_task_id:
            reply = build_uvvis_scan_status_followup_response(payload)
            if reply:
                return ActionResponse(
                    action=Action.RESPONSE,
                    response=reply,
                )
            return None

        state = _extract_uvvis_scan_state(payload)
        if state == RUNNING_SCAN_STATE:
            self._clear_pending_scan_start_confirmation(task_id)
            return ActionResponse(
                action=Action.RESPONSE,
                response=build_uvvis_scan_status_followup_response(
                    payload,
                    pending_confirmation=True,
                ),
            )

        if state == QUEUED_SCAN_STATE or not state:
            return ActionResponse(
                action=Action.RESPONSE,
                response=build_uvvis_scan_status_followup_response(
                    payload,
                    pending_confirmation=True,
                ),
            )

        if state in FAILED_SCAN_STATES or state == "succeeded":
            self._clear_pending_scan_start_confirmation(task_id)

        reply = build_uvvis_scan_status_followup_response(
            payload,
            pending_confirmation=True,
        )
        if reply:
            return ActionResponse(
                action=Action.RESPONSE,
                response=reply,
            )
        return None

    def _get_pending_scan_start_task_id(self) -> str:
        return str(getattr(self.conn, "_uvvis_scan_start_pending_task_id", "")).strip()

    def _mark_pending_scan_start_confirmation(self, task_id: str) -> None:
        self.conn._uvvis_scan_start_pending_task_id = str(task_id or "").strip()

    def _clear_pending_scan_start_confirmation(self, task_id: str = "") -> None:
        pending_task_id = self._get_pending_scan_start_task_id()
        target_task_id = str(task_id or "").strip()
        if target_task_id and pending_task_id and pending_task_id != target_task_id:
            return
        self.conn._uvvis_scan_start_pending_task_id = ""
