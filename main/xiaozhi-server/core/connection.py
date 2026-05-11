import os
import sys
import copy
import json
from urllib import response
import uuid
import time
import queue
import asyncio
import threading
import traceback
import subprocess
import websockets
from pathlib import Path

from core.utils.util import (
    extract_json_from_string,
    check_vad_update,
    check_asr_update,
    filter_sensitive_info,
)
from typing import Dict, Any
from collections import deque
from core.utils.modules_initialize import (
    initialize_modules,
    initialize_tts,
    initialize_asr,
)
from core.handle.reportHandle import report
from core.providers.tts.default import DefaultTTS
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Message, Dialogue
from core.providers.asr.dto.dto import InterfaceType
from core.handle.textHandle import handleTextMessage
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from plugins_func.loadplugins import auto_import_modules
from plugins_func.register import Action
from core.auth import AuthenticationError
from config.config_loader import get_private_config_from_api
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from config.logger import setup_logging, build_module_string, create_connection_logger
from config.manage_api_client import DeviceNotFoundException, DeviceBindException
from core.utils.prompt_manager import PromptManager
from core.utils.voiceprint_provider import (
    VoiceprintProvider,
    is_voiceprint_feature_enabled,
)
from core.utils.audio_frontend import AudioFrontend
from core.utils import textUtils
from core.utils.experiment_resume import (
    append_user_utterance_log,
    append_experiment_interaction_log,
    build_resume_context,
    build_resume_tool_message,
    enrich_latest_user_utterance_log,
)
from core.providers.tools.server_mcp.payload_utils import (
    extract_experiment_message,
    extract_experiment_session_id,
    finalize_server_mcp_payload,
    sync_server_mcp_payload_state,
)
from core.session import (
    load_experiment_session_binding,
    load_experiment_session_bindings_for_device,
    save_experiment_session_binding,
    delete_experiment_session_binding,
)

TAG = __name__

auto_import_modules("plugins_func.functions")


class TTSException(RuntimeError):
    pass


class ConnectionHandler:
    def __init__(
        self,
        config: Dict[str, Any],
        _vad,
        _asr,
        _llm,
        _memory,
        _intent,
        server=None,
    ):
        self.common_config = config
        self.config = copy.deepcopy(config)
        self.session_id = str(uuid.uuid4())
        self.transport_session_id = self.session_id
        self.chat_session_id = None
        self.model_session_key = None
        self.user_id = None
        self.logger = setup_logging()
        self.server = server  # 保存server实例的引用

        self.need_bind = False  # 是否需要绑定设备
        self.bind_completed_event = asyncio.Event()
        self.bind_code = None  # 绑定设备的验证码
        self.last_bind_prompt_time = 0  # 上次播放绑定提示的时间戳(秒)
        self.bind_prompt_interval = 60  # 绑定提示播放间隔(秒)

        self.read_config_from_api = self.config.get("read_config_from_api", False)

        self.websocket = None
        self.headers = None
        self.device_id = None
        self.client_ip = None
        self.prompt = None
        self.welcome_msg = None
        self.max_output_size = 0
        self.chat_history_conf = 0
        self.audio_format = "opus"

        # 客户端状态相关
        self.client_abort = False
        self.client_is_speaking = False
        self.client_listen_mode = "auto"

        # 线程任务相关
        self.loop = None  # 在 handle_connection 中获取运行中的事件循环
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.thinking_pulse_thread = None
        self.thinking_pulse_stop = threading.Event()
        self.action_pulse_thread = None
        self.action_pulse_stop = threading.Event()
        self._thinking_event_active = False
        self._thinking_finish_on_tts_start_pending = False
        self._thinking_event_lock = threading.RLock()

        # 添加上报线程池
        self.report_queue = queue.Queue()
        self.report_thread = None
        # 未来可以通过修改此处，调节asr的上报和tts的上报，目前默认都开启
        self.report_asr_enable = self.read_config_from_api
        self.report_tts_enable = self.read_config_from_api

        # 依赖的组件
        self.vad = None
        self.asr = None
        self.tts = None
        self._asr = _asr
        self._vad = _vad
        self.llm = _llm
        self.memory = _memory
        self.intent = _intent

        # 为每个连接单独管理声纹识别
        self.voiceprint_provider = None
        # Unified audio frontend (AEC/NS/etc). Per-connection instance.
        self.audio_frontend = None

        # vad相关变量
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_window = deque(maxlen=5)
        self.first_activity_time = 0.0  # 记录首次活动的时间（毫秒）
        self.last_activity_time = 0.0  # 统一的活动时间戳（毫秒）
        self.client_voice_stop = False
        self.last_is_voice = False
        self._asr_voice_stop_deadline_ms = 0.0

        # asr相关变量
        # 因为实际部署时可能会用到公共的本地ASR，不能把变量暴露给公共ASR
        # 所以涉及到ASR的变量，需要在这里定义，属于connection的私有变量
        self.asr_audio = []
        # PCM audio buffer after frontend processing (shared by VAD/ASR/voiceprint)
        self.asr_pcm_audio = []
        self.asr_audio_queue = queue.Queue()
        self.current_speaker = None  # 存储当前说话人
        self.current_language_tag = None  # 存储当前ASR识别的语言标签

        # llm相关变量
        self.llm_finish_task = True
        self.dialogue = Dialogue()
        self._llm_turn_started = False
        self._external_busy_tokens = set()
        self._external_busy_lock = threading.RLock()

        # tts相关变量
        self.sentence_id = None
        # 处理TTS响应没有文本返回
        self.tts_MessageText = ""

        # iot相关变量
        self.iot_descriptors = {}
        self.func_handler = None

        self.cmd_exit = self.config["exit_commands"]

        # 是否在聊天结束后关闭连接
        self.close_after_chat = False
        self.load_function_plugin = False
        self.intent_type = "nointent"

        close_connection_no_voice_time = int(
            self.config.get("close_connection_no_voice_time", 120)
        )
        self.timeout_seconds = (
            close_connection_no_voice_time + 60
            if close_connection_no_voice_time > 0
            else 0
        )  # <= 0 表示禁用无语音自动断开；否则在第一道关闭基础上加 60 秒做二道关闭
        self.timeout_task = None

        # {"mcp":true} 表示启用MCP功能
        self.features = None

        # 标记连接是否来自MQTT
        self.conn_from_mqtt_gateway = False

        # 初始化提示词管理器
        self.prompt_manager = PromptManager(self.config, self.logger)

        # Experiment prewarm state. This is intentionally server-side only:
        # it should prepare enough graph context for the first real user turn
        # without sending any audio/text back to the device.
        self.experiment_prewarm_task = None
        self.experiment_prewarm_lock = None
        self.experiment_prewarm_minimal_ready_event = None
        self.experiment_prewarm_status = "idle"
        self.experiment_prewarm_ready_level = "none"
        self.experiment_prewarm_trigger = ""
        self.experiment_prewarm_error = ""
        self.experiment_prewarm_started_at = 0.0
        self.experiment_prewarm_minimal_ready_at = 0.0
        self.experiment_prewarm_completed_at = 0.0
        self.experiment_deep_prefetch_task = None
        self.experiment_deep_prefetch_lock = None
        self.experiment_deep_prefetch_status = "idle"
        self.experiment_deep_prefetch_error = ""
        self.experiment_deep_prefetch_focus = ""
        self.experiment_deep_prefetch_query = ""
        self.experiment_deep_prefetch_started_at = 0.0
        self.experiment_deep_prefetch_completed_at = 0.0
        self.experiment_graph_priority_lock = None
        self.experiment_graph_background_resume_event = None
        self.experiment_graph_foreground_active = 0
        self.experiment_yaml_path = ""
        self.experiment_session_id = ""
        self.experiment_current_step_id = ""
        self.experiment_overview = None
        self.experiment_current_step = None
        self.experiment_progress_summary = None
        self.experiment_list_steps = None
        self.experiment_schema = None
        self.experiment_reference = None
        self.experiment_reference_query = ""
        self.experiment_resume_recovery_required = False
        self.experiment_resume_recovery_source = ""
        self.experiment_resume_previous_session_id = ""
        self.experiment_resume_reason = ""
        self.experiment_resume_log_path = ""
        self.experiment_resume_turn_count = ""
        self.experiment_resume_context_excerpt = ""
        self.experiment_resume_latest_session_id = ""
        self.experiment_resume_latest_current_step_id = ""
        self.experiment_prewarm_session_adopted = False
        self.experiment_first_real_user_turn_pending = True
        self.experiment_first_real_user_turn_lock = threading.Lock()

        # 连接生命周期状态
        self.keep_resources_on_transport_disconnect = bool(
            self.config.get("keep_resources_on_transport_disconnect", True)
        )
        self._connection_started = False
        self._allow_transport_reconnect = False
        self._final_close_requested = False
        self._closed = False
        self._transport_detached_event = asyncio.Event()
        self._transport_detached_event.set()

    def _format_ws_state(self, ws):
        if ws is None:
            return "ws=None"
        items = [f"type={type(ws).__name__}"]
        try:
            state = getattr(ws, "state", None)
            if state is not None:
                state_name = getattr(state, "name", str(state))
                items.append(f"state={state_name}")
            if hasattr(ws, "closed"):
                items.append(f"closed={ws.closed}")
            if hasattr(ws, "close_code"):
                items.append(f"close_code={getattr(ws, 'close_code', None)}")
            if hasattr(ws, "close_reason"):
                items.append(
                    f"close_reason={repr(getattr(ws, 'close_reason', None))}"
                )
        except Exception as e:
            items.append(f"state_read_error={e}")
        return ", ".join(items)

    @staticmethod
    def _format_close_frame(close_frame):
        if close_frame is None:
            return "None"
        code = getattr(close_frame, "code", None)
        reason = getattr(close_frame, "reason", None)
        return f"code={code}, reason={repr(reason)}"

    def _describe_connection_closed(self, exc):
        rcvd = getattr(exc, "rcvd", None)
        sent = getattr(exc, "sent", None)
        rcvd_then_sent = getattr(exc, "rcvd_then_sent", None)

        initiator = "unknown"
        if rcvd is not None and sent is None:
            initiator = "client"
        elif sent is not None and rcvd is None:
            initiator = "server"
        elif rcvd is not None and sent is not None:
            if rcvd_then_sent is True:
                initiator = "client"
            elif rcvd_then_sent is False:
                initiator = "server"

        code = getattr(exc, "code", None)
        reason = getattr(exc, "reason", None)
        return (
            f"initiator_guess={initiator}, "
            f"code={code}, reason={repr(reason)}, "
            f"rcvd=({self._format_close_frame(rcvd)}), "
            f"sent=({self._format_close_frame(sent)}), "
            f"rcvd_then_sent={rcvd_then_sent}"
        )

    @staticmethod
    def _close_call_trace():
        stack = traceback.extract_stack(limit=8)
        stack = stack[:-1]
        last = stack[-4:]
        return " <- ".join(
            f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
            for frame in last
        )

    def _llm_session_key(self) -> str:
        return self.model_session_key if self.model_session_key else self.session_id

    def _memory_session_key(self) -> str:
        return self.chat_session_id if self.chat_session_id else self.session_id

    def _llm_route_context_kwargs(self) -> Dict[str, str]:
        context: Dict[str, str] = {}
        for key, value in (
            ("device_id", self.device_id),
            ("chat_session_id", self.chat_session_id),
            ("model_session_key", self.model_session_key),
            ("connection_session_id", self.session_id),
            ("transport_session_id", self.transport_session_id),
            ("user_id", self.user_id),
        ):
            text = str(value or "").strip()
            if text:
                context[key] = text
        context.update(self._recent_server_photo_confirmation_route_context())
        return context

    def _recent_server_photo_confirmation_state(self) -> Dict[str, Any] | None:
        state = getattr(self, "_recent_server_photo_confirmation", None)
        if not isinstance(state, dict):
            return None

        try:
            captured_at = float(state.get("captured_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if captured_at <= 0:
            return None
        if (time.time() - captured_at) > 900:
            return None
        return state

    def _recent_server_photo_confirmation_route_context(self) -> Dict[str, str]:
        state = self._recent_server_photo_confirmation_state()
        if not isinstance(state, dict):
            return {}

        try:
            captured_at = float(state.get("captured_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            return {}

        age_seconds = max(0, int(time.time() - captured_at))

        sample_name = str(state.get("sample_name", "") or "").strip()
        graph_advanced = bool(state.get("graph_advanced"))
        next_step_id = str(state.get("next_step_id", "") or "").strip()
        next_step_title = str(state.get("next_step_title", "") or "").strip()
        current_step_id = str(state.get("current_step_id", "") or "").strip()
        current_step_title = str(state.get("current_step_title", "") or "").strip()
        graph_status_reason = str(state.get("graph_status_reason", "") or "").strip()
        photo_meta = (
            state.get("photo_meta") if isinstance(state.get("photo_meta"), dict) else {}
        )
        photo_file_name = str(photo_meta.get("file_name", "") or "").strip()

        sample_label = sample_name or "the current sample"
        summary_parts = [
            f"Recent server photo confirmation succeeded about {age_seconds} seconds ago for {sample_label}.",
        ]
        if photo_file_name:
            summary_parts.append(f"The saved file name was {photo_file_name}.")
        if graph_advanced:
            if next_step_title:
                summary_parts.append(
                    f"The experiment was already advanced to the next step: {next_step_title}."
                )
            elif next_step_id:
                summary_parts.append(
                    f"The experiment was already advanced to next_step_id={next_step_id}."
                )
        else:
            if current_step_title:
                summary_parts.append(
                    "The experiment graph was not confirmed as advanced after that photo "
                    f"and the best-known current step is {current_step_title}."
                )
            elif current_step_id:
                summary_parts.append(
                    "The experiment graph was not confirmed as advanced after that photo "
                    f"and the best-known current_step_id is {current_step_id}."
                )
            else:
                summary_parts.append(
                    "The experiment graph was not confirmed as advanced after that photo, "
                    "so do not assume the next step has changed."
                )
            if graph_status_reason:
                summary_parts.append(f"Follow-up status: {graph_status_reason}.")
        summary_parts.append(
            "Unless the user explicitly wants a retake, do not ask to retake the same sample photo again."
        )

        context = {
            "experiment_recent_photo_confirmation_summary": " ".join(summary_parts),
            "experiment_recent_photo_graph_advanced": "true" if graph_advanced else "false",
        }
        if sample_name:
            context["experiment_recent_photo_sample_name"] = sample_name
        if next_step_id:
            context["experiment_recent_photo_next_step_id"] = next_step_id
        if next_step_title:
            context["experiment_recent_photo_next_step_title"] = next_step_title
        if current_step_id:
            context["experiment_recent_photo_current_step_id"] = current_step_id
        if current_step_title:
            context["experiment_recent_photo_current_step_title"] = current_step_title
        return context

    def _experiment_prewarm_enabled(self) -> bool:
        if self.config.get("enable_server_mcp_client") is False:
            return False
        return bool(self.config.get("codex_app", {}).get("prewarm_on_hello", False))

    def _experiment_prewarm_wait_seconds(self) -> float:
        raw_value = self.config.get("codex_app", {}).get("prewarm_wait_seconds", 0)
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 0.0

    def _experiment_prewarm_timing_log_suffix(self) -> str:
        def _as_float(value) -> float:
            try:
                return max(0.0, float(value or 0.0))
            except (TypeError, ValueError):
                return 0.0

        started_at = _as_float(getattr(self, "experiment_prewarm_started_at", 0.0))
        if started_at <= 0:
            return ""

        now = time.time()
        parts = [f"elapsed_ms={int(max(0.0, now - started_at) * 1000)}"]

        minimal_ready_at = _as_float(
            getattr(self, "experiment_prewarm_minimal_ready_at", 0.0)
        )
        if minimal_ready_at > 0:
            parts.append(
                f"minimal_ready_ms={int(max(0.0, minimal_ready_at - started_at) * 1000)}"
            )

        completed_at = _as_float(
            getattr(self, "experiment_prewarm_completed_at", 0.0)
        )
        if completed_at > 0:
            parts.append(
                f"completed_ms={int(max(0.0, completed_at - started_at) * 1000)}"
            )

        return f", {', '.join(parts)}" if parts else ""

    def _experiment_prewarm_debug_delay_seconds(self) -> float:
        raw_value = (
            self.config.get("codex_app", {}).get("prewarm_debug_delay_seconds", 0)
        )
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 0.0

    def _experiment_deep_prefetch_micro_wait_seconds(self) -> float:
        raw_value = (
            self.config.get("codex_app", {}).get("deep_prefetch_micro_wait_seconds", 0.05)
        )
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 0.05

    async def _get_experiment_prewarm_minimal_ready_event(self):
        event = self.experiment_prewarm_minimal_ready_event
        if event is None:
            event = asyncio.Event()
            if self._experiment_prewarm_is_minimal_ready():
                event.set()
            self.experiment_prewarm_minimal_ready_event = event
        return event

    def _experiment_prewarm_is_minimal_ready(self) -> bool:
        return bool(
            self.experiment_session_id
            and self.experiment_prewarm_ready_level in {"minimal_ready", "completed"}
        )

    def _experiment_prewarm_is_completed(self) -> bool:
        return bool(
            self.experiment_session_id
            and self.experiment_prewarm_ready_level == "completed"
        )

    @staticmethod
    def _experiment_deep_prefetch_focuses(query: Any) -> list[str]:
        text = " ".join(str(query or "").strip().lower().split())
        if not text:
            return []

        focuses: list[str] = []
        theory_keywords = (
            "原理",
            "机理",
            "为什么",
            "背景",
            "注意事项",
            "依据",
        )
        workflow_keywords = (
            "后续流程",
            "后续步骤",
            "后面步骤",
            "完整流程",
            "完整步骤",
            "所有步骤",
            "全部步骤",
            "全流程",
            "整个实验",
            "后面都",
        )
        schema_keywords = (
            "字段定义",
            "字段",
            "schema",
            "记录项",
            "记录字段",
            "填什么",
            "记录什么",
            "单位",
        )

        if any(keyword in text for keyword in theory_keywords):
            focuses.append("theory")
        if any(keyword in text for keyword in workflow_keywords):
            focuses.append("workflow")
        if any(keyword in text for keyword in schema_keywords):
            focuses.append("schema")
        return focuses

    @staticmethod
    def _experiment_reference_sidecar_plan(query: Any) -> Dict[str, Any]:
        text = " ".join(str(query or "").strip().split())
        has_theory = any(
            keyword in text for keyword in ("原理", "机理", "为什么", "背景", "依据")
        )
        has_safety = any(keyword in text for keyword in ("注意事项", "安全", "风险", "小心"))
        has_materials = any(
            keyword in text for keyword in ("试剂", "材料", "药品", "仪器")
        )
        has_data_processing = any(
            keyword in text
            for keyword in ("数据处理", "作图", "速率常数", "拟合", "计算", "excel", "origin")
        )
        has_extraction = any(
            keyword in text for keyword in ("讲义", "原文", "整理", "提取")
        )

        only_safety = has_safety and not any(
            (has_theory, has_materials, has_data_processing, has_extraction)
        )
        only_materials = has_materials and not any(
            (has_theory, has_safety, has_data_processing, has_extraction)
        )
        only_data_processing = has_data_processing and not any(
            (has_theory, has_safety, has_materials, has_extraction)
        )
        only_extraction = has_extraction and not any(
            (has_theory, has_safety, has_materials, has_data_processing)
        )

        sections: list[str] = []
        if only_safety:
            sections = ["safety_notes"]
            mode = "safety_only"
            reference_query = "注意事项"
            include_description = False
            max_items_per_section = 3
        elif only_materials:
            sections = ["materials_notes"]
            mode = "materials_only"
            reference_query = "试剂与仪器"
            include_description = False
            max_items_per_section = 4
        elif only_data_processing:
            sections = ["data_processing_notes"]
            mode = "data_processing_only"
            reference_query = "数据处理"
            include_description = False
            max_items_per_section = 4
        elif only_extraction:
            sections = ["extraction_notes"]
            mode = "extraction_only"
            reference_query = "讲义提取"
            include_description = False
            max_items_per_section = 4
        else:
            if has_theory:
                sections.append("principle_notes")
            if has_safety:
                sections.append("safety_notes")
            if has_materials:
                sections.append("materials_notes")
            if has_data_processing:
                sections.append("data_processing_notes")
            if has_extraction:
                sections.append("extraction_notes")
            if not sections:
                sections.append("principle_notes")

            if has_theory and has_safety and not any(
                (has_materials, has_data_processing, has_extraction)
            ):
                reference_query = "原理+注意事项"
            elif has_theory and not any(
                (has_safety, has_materials, has_data_processing, has_extraction)
            ):
                reference_query = "原理"
            elif not text:
                reference_query = "原理"
            else:
                reference_query = text[:24]

            include_description = sections == ["principle_notes"]
            max_items_per_section = 4
            mode = "focused_multi" if len(sections) > 1 else "theory_default"

        deduped: list[str] = []
        for section in sections:
            if section not in deduped:
                deduped.append(section)
        return {
            "query_text": text,
            "reference_query": reference_query,
            "sections": deduped,
            "include_description": include_description,
            "max_items_per_section": max_items_per_section,
            "mode": mode,
        }

    @classmethod
    def _experiment_deep_prefetch_reference_query(cls, query: Any) -> str:
        plan = cls._experiment_reference_sidecar_plan(query)
        return str(plan.get("reference_query", "")).strip() or "原理"

    @classmethod
    def _experiment_reference_sidecar_sections(cls, query: Any) -> list[str]:
        plan = cls._experiment_reference_sidecar_plan(query)
        sections = plan.get("sections", [])
        if not isinstance(sections, list):
            return ["principle_notes"]
        return [str(section).strip() for section in sections if str(section).strip()]

    @staticmethod
    def _trim_experiment_reference_sidecar_text(
        value: Any, max_chars: int = 220
    ) -> str:
        text = " ".join(str(value or "").split()).strip()
        if max_chars > 0 and len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    def _normalize_experiment_reference_sidecar_notes(
        self,
        value: Any,
        *,
        max_items: int = 4,
        max_item_chars: int = 180,
    ) -> list[str]:
        if isinstance(value, list):
            raw_items = value
        elif value is None:
            raw_items = []
        else:
            raw_items = [value]

        notes: list[str] = []
        for item in raw_items:
            text = self._trim_experiment_reference_sidecar_text(
                item,
                max_chars=max_item_chars,
            )
            if text:
                notes.append(text)
            if max_items > 0 and len(notes) >= max_items:
                break
        return notes

    def _resolve_experiment_reference_sidecar_path(self) -> str:
        yaml_path = str(
            self.experiment_yaml_path or self._resolve_experiment_yaml_path() or ""
        ).strip()
        if not yaml_path:
            return ""
        return str(Path(yaml_path).with_suffix(".json"))

    def _load_experiment_reference_sidecar(self, query: Any):
        sidecar_path = self._resolve_experiment_reference_sidecar_path()
        if not sidecar_path:
            return None

        path = Path(sidecar_path)
        if not path.is_file():
            return None

        raw_text = None
        last_decode_error = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                raw_text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_decode_error = exc
            except OSError as exc:
                self.logger.bind(tag=TAG).warning(
                    "experiment reference sidecar read failed: "
                    f"device_id={self.device_id}, path={path}, error={exc}"
                )
                return None

        if raw_text is None:
            if last_decode_error is not None:
                self.logger.bind(tag=TAG).warning(
                    "experiment reference sidecar decode failed: "
                    f"device_id={self.device_id}, path={path}, error={last_decode_error}"
                )
            return None

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.logger.bind(tag=TAG).warning(
                "experiment reference sidecar json parse failed: "
                f"device_id={self.device_id}, path={path}, error={exc}"
            )
            return None

        if not isinstance(payload, dict):
            self.logger.bind(tag=TAG).warning(
                "experiment reference sidecar ignored non-dict payload: "
                f"device_id={self.device_id}, path={path}, type={type(payload).__name__}"
            )
            return None

        plan = self._experiment_reference_sidecar_plan(query)
        reference_query = str(plan.get("reference_query", "")).strip() or "原理"
        section_names = self._experiment_reference_sidecar_sections(query)
        include_description = bool(plan.get("include_description", False))
        max_items_per_section = int(plan.get("max_items_per_section", 4) or 4)
        mode = str(plan.get("mode", "")).strip() or "default"
        reference_payload: Dict[str, Any] = {
            "source": "local_static_sidecar",
            "query": reference_query,
        }

        title = self._trim_experiment_reference_sidecar_text(
            payload.get("title", ""),
            max_chars=160,
        )
        if title:
            reference_payload["title"] = title

        if include_description:
            description = self._trim_experiment_reference_sidecar_text(
                payload.get("description", ""),
                max_chars=260,
            )
            if description:
                reference_payload["description"] = description

        matched_sections: list[str] = []
        for section_name in section_names:
            notes = self._normalize_experiment_reference_sidecar_notes(
                payload.get(section_name),
                max_items=max_items_per_section,
                max_item_chars=180,
            )
            if not notes:
                continue
            reference_payload[section_name] = notes
            matched_sections.append(section_name)

        if matched_sections:
            reference_payload["matched_sections"] = matched_sections

        if len(reference_payload) <= 3:
            return None

        self.logger.bind(tag=TAG).info(
            "experiment reference sidecar loaded: "
            f"device_id={self.device_id}, path={path}, "
            f"query={reference_query}, mode={mode}, "
            f"sections={','.join(matched_sections) or 'none'}"
        )
        return reference_payload

    @staticmethod
    def _compact_experiment_payload(payload: Any, max_chars: int = 480) -> str:
        if payload is None:
            return ""

        if isinstance(payload, (dict, list)):
            try:
                text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                text = str(payload)
        else:
            text = str(payload)

        compact = " ".join(text.split()).strip()
        if max_chars > 0 and len(compact) > max_chars:
            return compact[: max_chars - 3].rstrip() + "..."
        return compact

    def _experiment_context_presence_flags(self) -> Dict[str, bool]:
        return {
            "has_overview_summary": bool(
                self._compact_experiment_payload(self.experiment_overview)
            ),
            "has_current_step_summary": bool(
                self._compact_experiment_payload(self.experiment_current_step)
            ),
            "has_list_steps_summary": bool(
                self._compact_experiment_payload(self.experiment_list_steps)
            ),
            "has_schema_summary": bool(
                self._compact_experiment_payload(self.experiment_schema)
            ),
            "has_reference_summary": bool(
                self._compact_experiment_payload(self.experiment_reference)
            ),
        }

    def _experiment_context_presence_log_fields(self) -> str:
        return ", ".join(
            f"{key}={str(value).lower()}"
            for key, value in self._experiment_context_presence_flags().items()
        )

    def _experiment_deep_prefetch_missing_focuses(
        self, query: Any, focuses: list[str]
    ) -> list[str]:
        missing: list[str] = []
        reference_query = self._experiment_deep_prefetch_reference_query(query)
        for focus in focuses:
            if focus == "workflow":
                if self.experiment_list_steps is None:
                    missing.append(focus)
            elif focus == "schema":
                if self.experiment_schema is None:
                    missing.append(focus)
            elif focus == "theory":
                if (
                    self.experiment_reference is None
                    or self.experiment_reference_query != reference_query
                ):
                    missing.append(focus)
        return missing

    async def _get_experiment_deep_prefetch_lock(self):
        if self.experiment_deep_prefetch_lock is None:
            self.experiment_deep_prefetch_lock = asyncio.Lock()
        return self.experiment_deep_prefetch_lock

    async def _get_experiment_graph_priority_lock(self):
        if self.experiment_graph_priority_lock is None:
            self.experiment_graph_priority_lock = asyncio.Lock()
        return self.experiment_graph_priority_lock

    async def _get_experiment_graph_background_resume_event(self):
        event = self.experiment_graph_background_resume_event
        if event is None:
            event = asyncio.Event()
            event.set()
            self.experiment_graph_background_resume_event = event
        return event

    async def _cancel_experiment_deep_prefetch_task(self):
        task = self.experiment_deep_prefetch_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def _reset_experiment_deep_prefetch_state(self):
        self.experiment_deep_prefetch_task = None
        self.experiment_deep_prefetch_status = "idle"
        self.experiment_deep_prefetch_error = ""
        self.experiment_deep_prefetch_focus = ""
        self.experiment_deep_prefetch_query = ""
        self.experiment_deep_prefetch_started_at = 0.0
        self.experiment_deep_prefetch_completed_at = 0.0
        self.experiment_list_steps = None
        self.experiment_schema = None
        self.experiment_reference = None
        self.experiment_reference_query = ""

    def _reset_experiment_resume_recovery_state(self):
        self.experiment_resume_recovery_required = False
        self.experiment_resume_recovery_source = ""
        self.experiment_resume_previous_session_id = ""
        self.experiment_resume_reason = ""
        self.experiment_resume_log_path = ""
        self.experiment_resume_turn_count = ""
        self.experiment_resume_context_excerpt = ""
        self.experiment_resume_latest_session_id = ""
        self.experiment_resume_latest_current_step_id = ""

    async def _before_experiment_graph_tool_call(
        self,
        tool_name: str,
        *,
        priority: str = "foreground",
    ):
        priority_text = str(priority or "foreground").strip() or "foreground"
        if priority_text == "background_common":
            await self._wait_for_experiment_graph_background_slot(
                tool_name=tool_name,
                stage="common",
            )
            return
        if not priority_text.startswith("foreground"):
            return

        event = await self._get_experiment_graph_background_resume_event()
        lock = await self._get_experiment_graph_priority_lock()
        async with lock:
            self.experiment_graph_foreground_active += 1
            event.clear()

    async def _after_experiment_graph_tool_call(
        self,
        tool_name: str,
        *,
        priority: str = "foreground",
    ):
        priority_text = str(priority or "foreground").strip() or "foreground"
        if not priority_text.startswith("foreground"):
            return

        event = await self._get_experiment_graph_background_resume_event()
        lock = await self._get_experiment_graph_priority_lock()
        async with lock:
            if self.experiment_graph_foreground_active > 0:
                self.experiment_graph_foreground_active -= 1
            if self.experiment_graph_foreground_active <= 0:
                self.experiment_graph_foreground_active = 0
                event.set()

    async def _wait_for_experiment_graph_background_slot(
        self,
        *,
        tool_name: str,
        stage: str,
    ):
        event = await self._get_experiment_graph_background_resume_event()
        if event.is_set():
            return

        self.logger.bind(tag=TAG).info(
            "experiment graph background warming yielding to foreground call: "
            f"device_id={self.device_id}, tool_name={tool_name}, stage={stage}"
        )

        await event.wait()

    @staticmethod
    def _trim_experiment_resume_excerpt(value: Any, max_chars: int = 1600) -> str:
        text = str(value or "").replace("\r\n", "\n").strip()
        if max_chars > 0 and len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    @staticmethod
    def _normalize_user_utterance_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _experiment_user_utterance_snapshot(self) -> Dict[str, str]:
        experiment_yaml_path = str(
            self.experiment_yaml_path or self._resolve_experiment_yaml_path() or ""
        ).strip()
        experiment_session_id = str(self.experiment_session_id or "").strip()
        current_step_id = str(self.experiment_current_step_id or "").strip()
        if not current_step_id:
            current_step_id = self._extract_experiment_current_step_id(
                self.experiment_current_step,
                self.experiment_progress_summary,
            )
        return {
            "experiment_yaml_path": experiment_yaml_path,
            "experiment_session_id": experiment_session_id,
            "current_step_id": current_step_id,
        }

    def log_clean_user_utterance(
        self,
        text: Any,
        *,
        source: str = "",
        speaker_name: str = "",
        language_tag: str = "",
    ) -> str:
        normalized_text = self._normalize_user_utterance_text(text)
        if not self.device_id or not normalized_text:
            return ""

        snapshot = self._experiment_user_utterance_snapshot()
        log_path = append_user_utterance_log(
            self.config,
            self.device_id,
            normalized_text,
            source=source,
            speaker=speaker_name,
            language=language_tag,
            chat_session_id=self.chat_session_id or "",
            model_session_key=self.model_session_key or "",
            connection_session_id=self.session_id or "",
            experiment_session_id=snapshot.get("experiment_session_id", ""),
            current_step_id=snapshot.get("current_step_id", ""),
            experiment_yaml_path=snapshot.get("experiment_yaml_path", ""),
        )
        if not log_path:
            return ""

        preview = normalized_text
        if len(preview) > 120:
            preview = preview[:117].rstrip() + "..."
        self.logger.bind(tag=TAG).info(
            "clean user utterance logged: "
            f"device_id={self.device_id}, "
            f"source={source or 'unknown'}, "
            f"log_path={log_path}, "
            f"text={preview}"
        )
        self.append_experiment_interaction_log(
            "USER",
            normalized_text,
            source=source or "unknown",
        )
        return log_path

    def append_experiment_interaction_log(
        self,
        role: str,
        text: Any,
        *,
        source: str = "",
    ) -> str:
        if not self.device_id:
            return ""
        snapshot = self._experiment_user_utterance_snapshot()
        log_path = append_experiment_interaction_log(
            self.config,
            self.device_id,
            text,
            role=role,
            source=source,
            experiment_session_id=snapshot.get("experiment_session_id", ""),
            current_step_id=snapshot.get("current_step_id", ""),
            experiment_yaml_path=snapshot.get("experiment_yaml_path", ""),
        )
        return str(log_path or "")

    def enrich_latest_clean_user_utterance_snapshot(self) -> str:
        if not self.device_id:
            return ""

        snapshot = self._experiment_user_utterance_snapshot()
        log_path = enrich_latest_user_utterance_log(
            self.config,
            self.device_id,
            connection_session_id=self.session_id or "",
            experiment_session_id=snapshot.get("experiment_session_id", ""),
            current_step_id=snapshot.get("current_step_id", ""),
            experiment_yaml_path=snapshot.get("experiment_yaml_path", ""),
        )
        if not log_path:
            return ""

        self.logger.bind(tag=TAG).info(
            "clean user utterance snapshot enriched: "
            f"device_id={self.device_id}, "
            f"log_path={log_path}, "
            f"experiment_session_id={snapshot.get('experiment_session_id', '')}, "
            f"current_step_id={snapshot.get('current_step_id', '')}"
        )
        return log_path

    def _prepare_experiment_resume_recovery_context(
        self,
        *,
        previous_session_id: str,
        reason: str,
    ):
        self._reset_experiment_resume_recovery_state()
        self.experiment_resume_recovery_required = True
        self.experiment_resume_recovery_source = "device_log"
        self.experiment_resume_previous_session_id = str(
            previous_session_id or ""
        ).strip()
        self.experiment_resume_reason = str(reason or "").strip()

        resume_context = None
        if self.device_id:
            try:
                resume_context = build_resume_context(
                    self.config,
                    self.device_id,
                    max_turns=4,
                    max_chars=1600,
                )
            except Exception as exc:
                self.logger.bind(tag=TAG).warning(
                    "experiment session recovery context load failed: "
                    f"device_id={self.device_id}, error={exc}"
                )

        if isinstance(resume_context, dict):
            self.experiment_resume_log_path = str(
                resume_context.get("log_path", "")
            ).strip()
            self.experiment_resume_turn_count = str(
                resume_context.get("turn_count", "")
            ).strip()
            self.experiment_resume_context_excerpt = (
                self._trim_experiment_resume_excerpt(
                    resume_context.get("context_text", ""),
                    max_chars=1600,
                )
            )
            self.experiment_resume_latest_session_id = str(
                resume_context.get("latest_experiment_session_id", "")
            ).strip()
            self.experiment_resume_latest_current_step_id = str(
                resume_context.get("latest_current_step_id", "")
            ).strip()

        self.logger.bind(tag=TAG).info(
            "experiment session recovery context prepared: "
            f"device_id={self.device_id}, "
            f"previous_session_id={self.experiment_resume_previous_session_id}, "
            f"source={self.experiment_resume_recovery_source}, "
            f"log_path={self.experiment_resume_log_path or 'missing'}, "
            f"log_turns={self.experiment_resume_turn_count or '0'}, "
            f"latest_session_id={self.experiment_resume_latest_session_id or ''}, "
            f"latest_current_step_id={self.experiment_resume_latest_current_step_id or ''}, "
            f"reason={self.experiment_resume_reason or 'unknown'}"
        )

    def _experiment_deep_prefetch_route_context(
        self, wait_result: str = ""
    ) -> Dict[str, str]:
        context: Dict[str, str] = {}
        if wait_result:
            context["experiment_deep_prefetch_wait_result"] = str(wait_result).strip()
        if self.experiment_deep_prefetch_status != "idle":
            context["experiment_deep_prefetch_status"] = str(
                self.experiment_deep_prefetch_status
            ).strip()
        if self.experiment_deep_prefetch_focus:
            context["experiment_deep_prefetch_focus"] = str(
                self.experiment_deep_prefetch_focus
            ).strip()
        if self.experiment_deep_prefetch_query:
            context["experiment_deep_prefetch_query"] = str(
                self.experiment_deep_prefetch_query
            ).strip()
        if self.experiment_list_steps is not None:
            context["experiment_list_steps_summary"] = self._compact_experiment_payload(
                self.experiment_list_steps,
                max_chars=720,
            )
        if self.experiment_schema is not None:
            context["experiment_schema_summary"] = self._compact_experiment_payload(
                self.experiment_schema,
                max_chars=720,
            )
        if self.experiment_reference is not None:
            context["experiment_reference_summary"] = self._compact_experiment_payload(
                self.experiment_reference,
                max_chars=960,
            )
        if self.experiment_deep_prefetch_error:
            context["experiment_deep_prefetch_error"] = str(
                self.experiment_deep_prefetch_error
            ).strip()
        return context

    def _experiment_prewarm_route_context(
        self, wait_result: str = "", deep_wait_result: str = ""
    ) -> Dict[str, str]:
        include_session_context = bool(wait_result) or bool(
            self.experiment_prewarm_session_adopted
        )
        if not include_session_context and self.experiment_resume_recovery_required:
            include_session_context = True
        if not include_session_context and self._experiment_prewarm_is_minimal_ready():
            # After a timeout first turn, prewarm may finish in the background a
            # moment later. Subsequent turns should keep reusing that trusted
            # session/current-step snapshot instead of dropping back to no
            # experiment context just because the first wait already happened.
            include_session_context = True
        if not include_session_context:
            return {}

        context: Dict[str, str] = {}
        if wait_result:
            context["experiment_prewarm_wait_result"] = str(wait_result).strip()
        if self.experiment_prewarm_status:
            context["experiment_prewarm_status"] = str(
                self.experiment_prewarm_status
            ).strip()
        if self.experiment_prewarm_ready_level:
            context["experiment_prewarm_ready_level"] = str(
                self.experiment_prewarm_ready_level
            ).strip()
        if self.experiment_prewarm_trigger:
            context["experiment_prewarm_trigger"] = str(
                self.experiment_prewarm_trigger
            ).strip()
        if include_session_context and self.experiment_yaml_path:
            context["experiment_yaml_path"] = str(self.experiment_yaml_path).strip()
        if include_session_context and self.experiment_session_id:
            context["experiment_session_id"] = str(
                self.experiment_session_id
            ).strip()
        if include_session_context and self.experiment_current_step_id:
            context["experiment_current_step_id"] = str(
                self.experiment_current_step_id
            ).strip()
        current_group_number = str(
            getattr(self, "experiment_current_group_number", "") or ""
        ).strip()
        if include_session_context and current_group_number:
            context["experiment_current_group_number"] = current_group_number

        overview_summary = ""
        if include_session_context:
            overview_summary = self._compact_experiment_payload(self.experiment_overview)
        if overview_summary:
            context["experiment_overview_summary"] = overview_summary

        step_summary = ""
        if include_session_context:
            step_summary = self._compact_experiment_payload(self.experiment_current_step)
        if step_summary:
            context["experiment_current_step_summary"] = step_summary

        if include_session_context:
            context.update(
                self._experiment_deep_prefetch_route_context(
                    wait_result=deep_wait_result
                )
            )

        if include_session_context and self.experiment_resume_recovery_required:
            context["experiment_resume_recovery_required"] = "true"
            if self.experiment_resume_recovery_source:
                context["experiment_resume_recovery_source"] = str(
                    self.experiment_resume_recovery_source
                ).strip()
            if self.experiment_resume_previous_session_id:
                context["experiment_resume_previous_session_id"] = str(
                    self.experiment_resume_previous_session_id
                ).strip()
            if self.experiment_resume_reason:
                context["experiment_resume_reason"] = str(
                    self.experiment_resume_reason
                ).strip()
            if self.experiment_resume_log_path:
                context["experiment_resume_log_path"] = str(
                    self.experiment_resume_log_path
                ).strip()
            if self.experiment_resume_turn_count:
                context["experiment_resume_turn_count"] = str(
                    self.experiment_resume_turn_count
                ).strip()
            if self.experiment_resume_latest_session_id:
                context["experiment_resume_latest_session_id"] = str(
                    self.experiment_resume_latest_session_id
                ).strip()
            if self.experiment_resume_latest_current_step_id:
                context["experiment_resume_latest_current_step_id"] = str(
                    self.experiment_resume_latest_current_step_id
                ).strip()
            if self.experiment_resume_context_excerpt:
                context["experiment_resume_context_excerpt"] = str(
                    self.experiment_resume_context_excerpt
                ).strip()

        if self.experiment_prewarm_error:
            context["experiment_prewarm_error"] = str(
                self.experiment_prewarm_error
            ).strip()
        return context

    def _current_experiment_step_snapshot_id(self) -> str:
        current_step_id = str(self.experiment_current_step_id or "").strip()
        if current_step_id:
            return current_step_id
        return self._extract_experiment_current_step_id(
            self.experiment_current_step,
            self.experiment_progress_summary,
        )

    def _should_refresh_experiment_state_before_llm(self) -> bool:
        session_id = str(self.experiment_session_id or "").strip()
        if not session_id:
            return False

        if bool(
            self.config.get("codex_app", {}).get(
                "refresh_experiment_state_before_each_turn", False
            )
        ):
            return True

        if bool(getattr(self, "_experiment_graph_refresh_required", False)):
            return True

        recent_photo_state = self._recent_server_photo_confirmation_state()
        if isinstance(recent_photo_state, dict) and not bool(
            recent_photo_state.get("graph_advanced")
        ):
            try:
                captured_at = float(recent_photo_state.get("captured_at", 0.0) or 0.0)
            except (TypeError, ValueError):
                captured_at = 0.0
            try:
                refresh_checked_at = float(
                    recent_photo_state.get("graph_refresh_checked_at", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                refresh_checked_at = 0.0
            if captured_at > 0 and refresh_checked_at < captured_at:
                return True

        current_step_id = self._current_experiment_step_snapshot_id()
        payload_step_id = self._extract_experiment_current_step_id(
            self.experiment_current_step,
            self.experiment_progress_summary,
        )
        recovery_step_id = str(
            self.experiment_resume_latest_current_step_id or ""
        ).strip()

        if not current_step_id:
            return True
        if payload_step_id and payload_step_id != current_step_id:
            return True
        if self.experiment_resume_recovery_required and recovery_step_id:
            return recovery_step_id != current_step_id
        return False

    async def refresh_experiment_foreground_state(
        self,
        *,
        reason: str = "",
    ) -> Dict[str, Any]:
        session_id = str(self.experiment_session_id or "").strip()
        if not session_id:
            return {}

        self.logger.bind(tag=TAG).info(
            "experiment foreground state refresh begin: "
            f"device_id={self.device_id}, session_id={session_id}, "
            f"reason={reason or 'unknown'}, "
            f"cached_step_id={self._current_experiment_step_snapshot_id()}, "
            f"recovery_step_id={self.experiment_resume_latest_current_step_id or ''}"
        )

        step_payload, progress_payload = await asyncio.gather(
            self._call_experiment_graph_tool(
                "get_step",
                {"session_id": session_id},
                priority="foreground",
            ),
            self._call_experiment_graph_tool(
                "get_progress_summary",
                {"session_id": session_id},
                priority="foreground",
            ),
        )

        current_step_id = self._extract_experiment_current_step_id(
            step_payload,
            progress_payload,
        )
        self.experiment_current_step = step_payload
        self.experiment_progress_summary = progress_payload
        if current_step_id:
            self.experiment_current_step_id = current_step_id
            if self.experiment_resume_recovery_required:
                self.experiment_resume_latest_current_step_id = current_step_id
        self._experiment_graph_refresh_required = False

        recent_photo_state = self._recent_server_photo_confirmation_state()
        if isinstance(recent_photo_state, dict) and not bool(
            recent_photo_state.get("graph_advanced")
        ):
            recent_photo_state["graph_refresh_checked_at"] = time.time()
            if current_step_id:
                recent_photo_state["current_step_id"] = current_step_id
            current_step_title = ""
            step_body = step_payload.get("result", step_payload) if isinstance(step_payload, dict) else {}
            if isinstance(step_body, dict):
                step_node = step_body.get("step")
                if isinstance(step_node, dict):
                    current_step_title = str(step_node.get("title", "") or "").strip()
            if not current_step_title:
                progress_body = (
                    progress_payload.get("result", progress_payload)
                    if isinstance(progress_payload, dict)
                    else {}
                )
                if isinstance(progress_body, dict):
                    summary = progress_body.get("summary")
                    if isinstance(summary, dict):
                        current_step = summary.get("current_step")
                        if isinstance(current_step, dict):
                            current_step_title = str(
                                current_step.get("title", "")
                                or current_step.get("step_title", "")
                                or ""
                            ).strip()
            if current_step_title:
                recent_photo_state["current_step_title"] = current_step_title

        self.logger.bind(tag=TAG).info(
            "experiment foreground state refresh done: "
            f"device_id={self.device_id}, session_id={session_id}, "
            f"reason={reason or 'unknown'}, current_step_id={current_step_id}, "
            f"{self._experiment_context_presence_log_fields()}"
        )
        return {
            "step_payload": step_payload,
            "progress_payload": progress_payload,
            "current_step_id": current_step_id,
        }

    async def maybe_refresh_experiment_state_before_llm(
        self,
        *,
        reason: str = "",
    ) -> Dict[str, str]:
        if not self._should_refresh_experiment_state_before_llm():
            return {}

        self.logger.bind(tag=TAG).info(
            "experiment state stale before llm, forcing foreground refresh: "
            f"device_id={self.device_id}, session_id={self.experiment_session_id}, "
            f"reason={reason or 'unknown'}, "
            f"cached_step_id={self._current_experiment_step_snapshot_id()}, "
            f"recovery_step_id={self.experiment_resume_latest_current_step_id or ''}"
        )

        await self.refresh_experiment_foreground_state(reason=reason or "before_llm")
        return self._experiment_prewarm_route_context()

    def _consume_experiment_first_real_user_turn_gate(self) -> bool:
        if not self._experiment_prewarm_enabled():
            return False

        with self.experiment_first_real_user_turn_lock:
            if not self.experiment_first_real_user_turn_pending:
                return False
            self.experiment_first_real_user_turn_pending = False
            return True

    def _resolve_experiment_yaml_path(self) -> str:
        llm_map = self.config.get("LLM", {}) or {}
        if not isinstance(llm_map, dict):
            return ""

        preferred = str(self.config.get("codex_app", {}).get("llm_name", "")).strip()
        llm_cfg = None
        if preferred:
            llm_cfg = llm_map.get(preferred)

        if not isinstance(llm_cfg, dict):
            selected_name = str(
                self.config.get("selected_module", {}).get("LLM", "")
            ).strip()
            selected_cfg = llm_map.get(selected_name)
            if (
                isinstance(selected_cfg, dict)
                and str(selected_cfg.get("type", "")).strip() == "codex"
            ):
                llm_cfg = selected_cfg

        if not isinstance(llm_cfg, dict):
            for candidate in llm_map.values():
                if (
                    isinstance(candidate, dict)
                    and str(candidate.get("type", "")).strip() == "codex"
                ):
                    llm_cfg = candidate
                    break

        if not isinstance(llm_cfg, dict):
            return ""

        workspace = str(llm_cfg.get("workspace", "")).strip()
        yaml_path = str(llm_cfg.get("yaml_path", "")).strip()
        if not yaml_path:
            return ""
        if os.path.isabs(yaml_path):
            return str(Path(yaml_path))
        if workspace:
            return str(Path(workspace) / yaml_path)
        return str(Path(yaml_path).resolve())

    @staticmethod
    def _extract_experiment_session_id(payload: Any) -> str:
        return extract_experiment_session_id(payload)

    @staticmethod
    def _experiment_result_body(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            nested = payload.get("result")
            if isinstance(nested, dict):
                return nested
            return payload
        return {}

    @staticmethod
    def _coerce_nonnegative_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _extract_experiment_result_ok(cls, payload: Any):
        body = cls._experiment_result_body(payload)
        value = body.get("ok")
        return value if isinstance(value, bool) else None

    @classmethod
    def _extract_experiment_result_message(cls, payload: Any) -> str:
        body = cls._experiment_result_body(payload)
        return str(body.get("message", "")).strip()

    @classmethod
    def _extract_experiment_current_step_id(cls, *payloads: Any) -> str:
        for payload in payloads:
            body = cls._experiment_result_body(payload)
            state = body.get("state")
            if isinstance(state, dict):
                text = str(state.get("current_step_id", "")).strip()
                if text:
                    return text
            step = body.get("step")
            if isinstance(step, dict):
                text = str(step.get("id", "")).strip()
                if text:
                    return text
            summary = body.get("summary")
            if isinstance(summary, dict):
                current_step = summary.get("current_step")
                if isinstance(current_step, dict):
                    text = str(current_step.get("step_id", "")).strip()
                    if text:
                        return text
        return ""

    @classmethod
    def _extract_experiment_total_steps(cls, *payloads: Any) -> int:
        for payload in payloads:
            body = cls._experiment_result_body(payload)
            total_steps = cls._coerce_nonnegative_int(body.get("steps_count"))
            if total_steps > 0:
                return total_steps
            summary = body.get("summary")
            if isinstance(summary, dict):
                progress = summary.get("progress")
                if isinstance(progress, dict):
                    total_steps = cls._coerce_nonnegative_int(
                        progress.get("total_steps")
                    )
                    if total_steps > 0:
                        return total_steps
        return 0

    @classmethod
    def _extract_experiment_completed_steps_count(cls, *payloads: Any) -> int:
        for payload in payloads:
            body = cls._experiment_result_body(payload)
            state = body.get("state")
            if isinstance(state, dict):
                completed_steps = state.get("completed_steps")
                if isinstance(completed_steps, list):
                    return len(completed_steps)
            summary = body.get("summary")
            if isinstance(summary, dict):
                progress = summary.get("progress")
                if isinstance(progress, dict):
                    completed_count = cls._coerce_nonnegative_int(
                        progress.get("completed_steps")
                    )
                    if completed_count > 0 or "completed_steps" in progress:
                        return completed_count
        return 0

    @classmethod
    def _classify_experiment_resume_candidate(
        cls,
        state_payload: Any,
        overview_payload: Any = None,
        progress_summary_payload: Any = None,
    ) -> str:
        current_step_id = cls._extract_experiment_current_step_id(
            state_payload, progress_summary_payload
        )
        total_steps = cls._extract_experiment_total_steps(
            overview_payload, progress_summary_payload
        )
        completed_steps_count = cls._extract_experiment_completed_steps_count(
            state_payload, progress_summary_payload
        )

        if total_steps > 0 and completed_steps_count >= total_steps:
            return "completed"
        if current_step_id:
            return "active"
        return "invalid"

    async def _get_experiment_prewarm_lock(self):
        if self.experiment_prewarm_lock is None:
            self.experiment_prewarm_lock = asyncio.Lock()
        return self.experiment_prewarm_lock

    async def _run_experiment_deep_prefetch(
        self,
        *,
        session_id: str,
        query: str,
        focuses: list[str],
    ):
        current_task = asyncio.current_task()
        focus_text = ",".join(focuses)
        query_text = " ".join(str(query or "").strip().split())
        self.experiment_deep_prefetch_status = "running"
        self.experiment_deep_prefetch_error = ""
        self.experiment_deep_prefetch_focus = focus_text
        self.experiment_deep_prefetch_query = query_text
        self.experiment_deep_prefetch_started_at = time.time()
        self.logger.bind(tag=TAG).info(
            "experiment deep prefetch begin: "
            f"device_id={self.device_id}, session_id={session_id}, "
            f"focus={focus_text}, query={query_text[:120]}"
        )

        tool_tasks: dict[str, asyncio.Task] = {}
        try:
            if "workflow" in focuses and self.experiment_list_steps is None:
                tool_tasks["workflow"] = asyncio.create_task(
                    self._call_experiment_graph_tool(
                        "list_steps",
                        {"session_id": session_id},
                        priority="foreground_detail",
                    )
                )
            if "schema" in focuses and self.experiment_schema is None:
                tool_tasks["schema"] = asyncio.create_task(
                    self._call_experiment_graph_tool(
                        "get_schema",
                        {"session_id": session_id},
                        priority="foreground_detail",
                    )
                )
            if "theory" in focuses:
                reference_query = self._experiment_deep_prefetch_reference_query(
                    query_text
                )
                if (
                    self.experiment_reference is None
                    or self.experiment_reference_query != reference_query
                ):
                    sidecar_payload = self._load_experiment_reference_sidecar(query_text)
                    if sidecar_payload is not None:
                        self.experiment_reference = sidecar_payload
                        self.experiment_reference_query = reference_query
                    else:
                        self.logger.bind(tag=TAG).info(
                            "experiment reference sidecar unavailable, falling back to MCP: "
                            f"device_id={self.device_id}, session_id={session_id}, "
                            f"query={reference_query}"
                        )
                        tool_tasks["theory"] = asyncio.create_task(
                            self._call_experiment_graph_tool(
                                "search_experiment_reference",
                                {
                                    "session_id": session_id,
                                    "query": reference_query,
                                    "max_hits": 3,
                                    "context_chars": 220,
                                },
                                priority="foreground_detail",
                            )
                        )

            if not tool_tasks:
                self.experiment_deep_prefetch_status = "ready"
                self.experiment_deep_prefetch_completed_at = time.time()
                return

            results = await asyncio.gather(
                *tool_tasks.values(),
                return_exceptions=True,
            )
            errors: list[str] = []
            success = False
            for label, result in zip(tool_tasks.keys(), results):
                if isinstance(result, Exception):
                    errors.append(f"{label}: {result}")
                    continue
                success = True
                if label == "workflow":
                    self.experiment_list_steps = result
                elif label == "schema":
                    self.experiment_schema = result
                elif label == "theory":
                    reference_query = self._experiment_deep_prefetch_reference_query(
                        query_text
                    )
                    self.experiment_reference = {
                        "query": reference_query,
                        "result": result,
                    }
                    self.experiment_reference_query = reference_query

            if success:
                self.experiment_deep_prefetch_status = "ready"
                if errors:
                    self.experiment_deep_prefetch_error = "; ".join(errors)
            else:
                self.experiment_deep_prefetch_status = "failed"
                self.experiment_deep_prefetch_error = (
                    "; ".join(errors) or "deep prefetch returned no usable payload"
                )
            self.experiment_deep_prefetch_completed_at = time.time()
            self.logger.bind(tag=TAG).info(
                "experiment deep prefetch finished: "
                f"device_id={self.device_id}, session_id={session_id}, "
                f"focus={focus_text}, status={self.experiment_deep_prefetch_status}"
            )
        except asyncio.CancelledError:
            self.experiment_deep_prefetch_status = "cancelled"
            self.experiment_deep_prefetch_error = "deep prefetch cancelled"
            raise
        except Exception as exc:
            self.experiment_deep_prefetch_status = "failed"
            self.experiment_deep_prefetch_error = str(exc)
            self.experiment_deep_prefetch_completed_at = time.time()
            self.logger.bind(tag=TAG).warning(
                "experiment deep prefetch failed: "
                f"device_id={self.device_id}, session_id={session_id}, "
                f"focus={focus_text}, error={exc}"
            )
        finally:
            if self.experiment_deep_prefetch_task is current_task:
                self.experiment_deep_prefetch_task = None

    async def _ensure_experiment_deep_prefetch_task(
        self, query: str, focuses: list[str]
    ):
        lock = await self._get_experiment_deep_prefetch_lock()
        async with lock:
            task = self.experiment_deep_prefetch_task
            if task is not None and task.done():
                self.experiment_deep_prefetch_task = None
                task = None

            missing_focuses = self._experiment_deep_prefetch_missing_focuses(
                query, focuses
            )
            if not missing_focuses:
                self.experiment_deep_prefetch_status = "ready"
                self.experiment_deep_prefetch_focus = ",".join(focuses)
                self.experiment_deep_prefetch_query = " ".join(
                    str(query or "").strip().split()
                )
                self.experiment_deep_prefetch_completed_at = time.time()
                return None

            if task is not None and not task.done():
                return task

            session_id = str(self.experiment_session_id or "").strip()
            if not session_id:
                return None

            task = asyncio.create_task(
                self._run_experiment_deep_prefetch(
                    session_id=session_id,
                    query=query,
                    focuses=missing_focuses,
                )
            )
            self.experiment_deep_prefetch_task = task
            return task

    async def wait_for_experiment_deep_prefetch(
        self,
        query: str,
        timeout_seconds: float,
    ) -> Dict[str, str]:
        timeout_seconds = max(0.0, float(timeout_seconds or 0.0))
        if not self._experiment_prewarm_is_minimal_ready():
            return self._experiment_prewarm_route_context()

        focuses = self._experiment_deep_prefetch_focuses(query)
        if not focuses:
            return self._experiment_prewarm_route_context()

        self.logger.bind(tag=TAG).info(
            "experiment deep prefetch requested: "
            f"device_id={self.device_id}, session_id={self.experiment_session_id}, "
            f"focus={','.join(focuses)}, timeout_seconds={timeout_seconds}"
        )
        task = await self._ensure_experiment_deep_prefetch_task(query, focuses)
        missing_focuses = self._experiment_deep_prefetch_missing_focuses(
            query, focuses
        )
        if task is None:
            wait_result = "ready" if not missing_focuses else "skipped"
            return self._experiment_prewarm_route_context(
                deep_wait_result=wait_result
            )

        if task.done():
            wait_result = (
                "ready"
                if not self._experiment_deep_prefetch_missing_focuses(query, focuses)
                else (self.experiment_deep_prefetch_status or "done")
            )
            return self._experiment_prewarm_route_context(deep_wait_result=wait_result)

        if timeout_seconds <= 0:
            return self._experiment_prewarm_route_context(
                deep_wait_result="skipped"
            )

        start = time.perf_counter()
        done, _pending = await asyncio.wait(
            {task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if task in done:
            wait_result = (
                "ready"
                if not self._experiment_deep_prefetch_missing_focuses(query, focuses)
                else (self.experiment_deep_prefetch_status or "done")
            )
        else:
            wait_result = "timeout"
        self.logger.bind(tag=TAG).info(
            "experiment deep prefetch wait result: "
            f"device_id={self.device_id}, session_id={self.experiment_session_id}, "
            f"focus={','.join(focuses)}, wait_result={wait_result}, "
            f"elapsed_ms={elapsed_ms:.1f}, status={self.experiment_deep_prefetch_status}, "
            f"{self._experiment_context_presence_log_fields()}"
        )
        return self._experiment_prewarm_route_context(deep_wait_result=wait_result)

    async def _wait_for_server_mcp_ready(self, timeout_seconds: float = 12.0) -> bool:
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        while time.monotonic() < deadline:
            func_handler = getattr(self, "func_handler", None)
            server_executor = getattr(func_handler, "server_mcp_executor", None)
            if server_executor is not None:
                if not getattr(server_executor, "_initialized", False):
                    try:
                        await server_executor.initialize()
                    except Exception as exc:
                        self.logger.bind(tag=TAG).debug(
                            f"server MCP initialize not ready yet: {exc}"
                        )
                manager = getattr(server_executor, "mcp_manager", None)
                if manager is not None:
                    try:
                        if manager.is_mcp_tool("get_state") and manager.is_mcp_tool(
                            "create_session"
                        ):
                            return True
                    except Exception:
                        pass
                if manager is not None and getattr(func_handler, "finish_init", False):
                    return True
            await asyncio.sleep(0.1)
        return False

    async def _call_experiment_graph_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        priority: str = "foreground",
    ):
        func_handler = getattr(self, "func_handler", None)
        server_executor = getattr(func_handler, "server_mcp_executor", None)
        manager = getattr(server_executor, "mcp_manager", None)
        if manager is None:
            raise RuntimeError("server MCP manager is not ready")
        raw_result = await manager.execute_tool(
            tool_name,
            arguments,
            priority=priority,
        )
        payload = finalize_server_mcp_payload(
            raw_result,
            tool_name=tool_name,
            arguments=arguments,
        )
        sync_server_mcp_payload_state(
            self,
            tool_name=tool_name,
            payload=payload,
            arguments=arguments,
        )
        return payload

    async def _run_experiment_prewarm_deep_stage(
        self,
        *,
        session_id: str,
        progress_summary_payload: Any,
    ):
        # This stage is unconditional background warming that starts
        # immediately after minimal_ready. It is not tied to user silence.
        self.experiment_prewarm_status = "deep_warming"
        self.logger.bind(tag=TAG).info(
            "experiment prewarm deep warming begin: "
            f"device_id={self.device_id}, trigger={self.experiment_prewarm_trigger}, "
            f"session_id={session_id}, current_step_id={self.experiment_current_step_id}"
            f"{self._experiment_prewarm_timing_log_suffix()}"
        )

        overview_payload = await self._call_experiment_graph_tool(
            "get_overview",
            {"session_id": session_id},
            priority="background_common",
        )
        step_payload = await self._call_experiment_graph_tool(
            "get_step",
            {"session_id": session_id},
            priority="background_common",
        )
        list_steps_payload = await self._call_experiment_graph_tool(
            "list_steps",
            {"session_id": session_id},
            priority="background_common",
        )
        schema_payload = await self._call_experiment_graph_tool(
            "get_schema",
            {"session_id": session_id},
            priority="background_common",
        )

        refreshed_current_step_id = self._extract_experiment_current_step_id(
            step_payload, progress_summary_payload
        )
        if refreshed_current_step_id:
            self.experiment_current_step_id = refreshed_current_step_id

        self.experiment_progress_summary = progress_summary_payload
        self.experiment_overview = overview_payload
        self.experiment_current_step = step_payload
        self.experiment_list_steps = list_steps_payload
        self.experiment_schema = schema_payload
        self.experiment_prewarm_status = "completed"
        self.experiment_prewarm_ready_level = "completed"
        self.experiment_prewarm_completed_at = time.time()
        self.logger.bind(tag=TAG).info(
            "experiment prewarm completed: "
            f"device_id={self.device_id}, trigger={self.experiment_prewarm_trigger}, "
            f"session_id={self.experiment_session_id}, "
            f"current_step_id={self.experiment_current_step_id}, "
            f"{self._experiment_context_presence_log_fields()}"
            f"{self._experiment_prewarm_timing_log_suffix()}"
        )

    async def prewarm_experiment_session(self, trigger: str = "", force: bool = False) -> bool:
        if not self.device_id:
            return False

        lock = await self._get_experiment_prewarm_lock()
        async with lock:
            if not force and self._experiment_prewarm_is_minimal_ready():
                return True

            await self._cancel_experiment_deep_prefetch_task()
            self._reset_experiment_deep_prefetch_state()
            minimal_ready_event = await self._get_experiment_prewarm_minimal_ready_event()
            minimal_ready_event.clear()

            self.experiment_prewarm_status = "minimal_warming"
            self.experiment_prewarm_ready_level = "none"
            self.experiment_prewarm_trigger = str(trigger or "").strip()
            self.experiment_prewarm_error = ""
            self.experiment_prewarm_started_at = time.time()
            self.experiment_prewarm_minimal_ready_at = 0.0
            self.experiment_prewarm_completed_at = 0.0
            self.experiment_session_id = ""
            self.experiment_current_step_id = ""
            self.experiment_overview = None
            self.experiment_current_step = None
            self.experiment_progress_summary = None
            self.experiment_list_steps = None
            self.experiment_schema = None
            self.experiment_reference = None
            self.experiment_reference_query = ""

            yaml_path = self._resolve_experiment_yaml_path()
            self.experiment_yaml_path = yaml_path
            if not yaml_path:
                self.experiment_prewarm_status = "failed"
                self.experiment_prewarm_error = "yaml_path is empty"
                self.logger.bind(tag=TAG).warning(
                    "experiment prewarm skipped: yaml_path is empty"
                )
                return False

            ready = await self._wait_for_server_mcp_ready()
            if not ready:
                self.experiment_prewarm_status = "failed"
                self.experiment_prewarm_error = "server MCP not ready"
                self.logger.bind(tag=TAG).warning(
                    "experiment prewarm skipped: server MCP not ready"
                )
                return False

            minimal_ready_reached = False
            try:
                self._reset_experiment_resume_recovery_state()
                session_id = ""
                session_source = ""
                progress_summary_payload = None
                current_step_id = ""
                total_steps = 0
                completed_steps_count = 0
                resume_recovery_needed = False
                resume_recovery_previous_session_id = ""
                resume_recovery_reason = ""

                resume_binding = None
                if self.chat_session_id:
                    resume_binding = await load_experiment_session_binding(
                        self.config,
                        self.chat_session_id,
                        yaml_path,
                    )

                if resume_binding and resume_binding.get("experiment_session_id"):
                    candidate_session_id = str(
                        resume_binding.get("experiment_session_id", "")
                    ).strip()
                    self.logger.bind(tag=TAG).info(
                        "experiment session resume hit: "
                        f"device_id={self.device_id}, "
                        f"chat_session_id={self.chat_session_id}, "
                        f"experiment_session_id={candidate_session_id}"
                    )
                    try:
                        state_payload = await self._call_experiment_graph_tool(
                            "get_state",
                            {"session_id": candidate_session_id},
                            priority="prewarm_minimal",
                        )
                        progress_summary_payload = await self._call_experiment_graph_tool(
                            "get_progress_summary",
                            {"session_id": candidate_session_id},
                            priority="prewarm_minimal",
                        )

                        resume_reason = ""
                        state_ok = self._extract_experiment_result_ok(state_payload)
                        progress_ok = self._extract_experiment_result_ok(
                            progress_summary_payload
                        )
                        if state_ok is False:
                            resume_status = "invalid"
                            resume_reason = (
                                self._extract_experiment_result_message(state_payload)
                                or "get_state returned not ok"
                            )
                        elif progress_ok is False:
                            resume_status = "invalid"
                            resume_reason = (
                                self._extract_experiment_result_message(
                                    progress_summary_payload
                                )
                                or "get_progress_summary returned not ok"
                            )
                        else:
                            resume_status = self._classify_experiment_resume_candidate(
                                state_payload,
                                None,
                                progress_summary_payload,
                            )
                        current_step_id = self._extract_experiment_current_step_id(
                            state_payload, progress_summary_payload
                        )
                        total_steps = self._extract_experiment_total_steps(
                            progress_summary_payload
                        )
                        completed_steps_count = (
                            self._extract_experiment_completed_steps_count(
                                state_payload, progress_summary_payload
                            )
                        )

                        if resume_status == "active":
                            session_id = candidate_session_id
                            session_source = "resume"
                            self.logger.bind(tag=TAG).info(
                                "experiment session resume ready: "
                                f"device_id={self.device_id}, "
                                f"chat_session_id={self.chat_session_id}, "
                                f"experiment_session_id={session_id}, "
                                f"current_step_id={current_step_id}, "
                                f"completed_steps={completed_steps_count}/{total_steps or '?'}"
                            )
                        else:
                            await delete_experiment_session_binding(
                                self.config,
                                self.chat_session_id,
                                yaml_path,
                            )
                            if resume_status == "completed":
                                self.logger.bind(tag=TAG).info(
                                    "experiment session resume completed_fallback_create: "
                                    f"device_id={self.device_id}, "
                                    f"chat_session_id={self.chat_session_id}, "
                                    f"experiment_session_id={candidate_session_id}, "
                                    f"completed_steps={completed_steps_count}/{total_steps or '?'}"
                                )
                            else:
                                resume_recovery_needed = True
                                resume_recovery_previous_session_id = (
                                    candidate_session_id
                                )
                                resume_recovery_reason = (
                                    resume_reason or "session payload invalid"
                                )
                                self.logger.bind(tag=TAG).info(
                                    "experiment session resume invalid_fallback_create: "
                                    f"device_id={self.device_id}, "
                                    f"chat_session_id={self.chat_session_id}, "
                                    f"experiment_session_id={candidate_session_id}, "
                                    f"reason={resume_reason or 'session payload invalid'}"
                                )
                    except Exception as exc:
                        await delete_experiment_session_binding(
                            self.config,
                            self.chat_session_id,
                            yaml_path,
                        )
                        resume_recovery_needed = True
                        resume_recovery_previous_session_id = candidate_session_id
                        resume_recovery_reason = str(exc)
                        self.logger.bind(tag=TAG).warning(
                            "experiment session resume invalid_fallback_create: "
                            f"device_id={self.device_id}, "
                            f"chat_session_id={self.chat_session_id}, "
                            f"experiment_session_id={candidate_session_id}, "
                            f"error={exc}"
                        )
                else:
                    self.logger.bind(tag=TAG).info(
                        "experiment session resume miss: "
                        f"device_id={self.device_id}, "
                        f"chat_session_id={self.chat_session_id}, "
                        f"yaml_path={yaml_path}"
                    )
                    device_resume_candidates = []
                    try:
                        device_resume_candidates = (
                            await load_experiment_session_bindings_for_device(
                                self.config,
                                device_id=self.device_id or "",
                                user_id=self.user_id or "",
                                yaml_path=yaml_path,
                            )
                        )
                    except Exception as exc:
                        self.logger.bind(tag=TAG).warning(
                            "experiment device resume lookup failed: "
                            f"device_id={self.device_id}, yaml_path={yaml_path}, "
                            f"error={exc}"
                        )

                    best_device_resume = None
                    for candidate_binding in device_resume_candidates:
                        candidate_session_id = str(
                            candidate_binding.get("experiment_session_id", "")
                        ).strip()
                        if not candidate_session_id:
                            continue
                        try:
                            candidate_state_payload = (
                                await self._call_experiment_graph_tool(
                                    "get_state",
                                    {"session_id": candidate_session_id},
                                    priority="prewarm_minimal",
                                )
                            )
                            candidate_progress_payload = (
                                await self._call_experiment_graph_tool(
                                    "get_progress_summary",
                                    {"session_id": candidate_session_id},
                                    priority="prewarm_minimal",
                                )
                            )
                        except Exception as exc:
                            self.logger.bind(tag=TAG).warning(
                                "experiment device resume candidate failed: "
                                f"device_id={self.device_id}, "
                                f"experiment_session_id={candidate_session_id}, "
                                f"error={exc}"
                            )
                            continue

                        candidate_status = self._classify_experiment_resume_candidate(
                            candidate_state_payload,
                            None,
                            candidate_progress_payload,
                        )
                        if candidate_status != "active":
                            continue

                        candidate_completed = (
                            self._extract_experiment_completed_steps_count(
                                candidate_state_payload,
                                candidate_progress_payload,
                            )
                        )
                        candidate_step_id = self._extract_experiment_current_step_id(
                            candidate_state_payload,
                            candidate_progress_payload,
                        )
                        candidate_total_steps = self._extract_experiment_total_steps(
                            candidate_progress_payload
                        )
                        candidate_score = (
                            int(candidate_completed or 0),
                            str(candidate_binding.get("updated_at", "")),
                        )
                        if (
                            best_device_resume is None
                            or candidate_score > best_device_resume["score"]
                        ):
                            best_device_resume = {
                                "score": candidate_score,
                                "session_id": candidate_session_id,
                                "state_payload": candidate_state_payload,
                                "progress_payload": candidate_progress_payload,
                                "current_step_id": candidate_step_id,
                                "completed_steps_count": candidate_completed,
                                "total_steps": candidate_total_steps,
                            }

                    if best_device_resume is not None:
                        session_id = best_device_resume["session_id"]
                        session_source = "resume_device"
                        state_payload = best_device_resume["state_payload"]
                        progress_summary_payload = best_device_resume[
                            "progress_payload"
                        ]
                        current_step_id = best_device_resume["current_step_id"]
                        completed_steps_count = best_device_resume[
                            "completed_steps_count"
                        ]
                        total_steps = best_device_resume["total_steps"]
                        self.logger.bind(tag=TAG).info(
                            "experiment session device resume ready: "
                            f"device_id={self.device_id}, "
                            f"chat_session_id={self.chat_session_id}, "
                            f"experiment_session_id={session_id}, "
                            f"current_step_id={current_step_id}, "
                            f"completed_steps={completed_steps_count}/{total_steps or '?'}"
                        )

                if not session_id:
                    create_payload = await self._call_experiment_graph_tool(
                        "create_session",
                        {
                            "yaml_path": yaml_path,
                            "device_id": self.device_id or "",
                        },
                        priority="prewarm_minimal",
                    )
                    session_id = self._extract_experiment_session_id(create_payload)
                    if not session_id:
                        create_message = extract_experiment_message(create_payload)
                        if create_message:
                            raise RuntimeError(
                                f"create_session failed: {create_message}"
                            )
                        raise RuntimeError("create_session returned empty session_id")

                    progress_summary_payload = await self._call_experiment_graph_tool(
                        "get_progress_summary",
                        {"session_id": session_id},
                        priority="prewarm_minimal",
                    )
                    current_step_id = self._extract_experiment_current_step_id(
                        progress_summary_payload
                    )
                    total_steps = self._extract_experiment_total_steps(
                        progress_summary_payload
                    )
                    completed_steps_count = (
                        self._extract_experiment_completed_steps_count(
                            progress_summary_payload
                        )
                    )
                    session_source = (
                        "recovery_create" if resume_recovery_needed else "create"
                    )
                    if resume_recovery_needed:
                        self._prepare_experiment_resume_recovery_context(
                            previous_session_id=resume_recovery_previous_session_id,
                            reason=resume_recovery_reason,
                        )
                        self.logger.bind(tag=TAG).info(
                            "experiment session recovery armed after fallback create: "
                            f"device_id={self.device_id}, "
                            f"previous_session_id={resume_recovery_previous_session_id}, "
                            f"new_session_id={session_id}, "
                            f"log_path={self.experiment_resume_log_path or 'missing'}, "
                            f"log_turns={self.experiment_resume_turn_count or '0'}"
                        )

                self.experiment_session_id = session_id
                self.experiment_current_step_id = current_step_id
                self.experiment_progress_summary = progress_summary_payload
                if self.experiment_session_id and self.device_id:
                    try:
                        await self._call_experiment_graph_tool(
                            "configure_auto_export",
                            {
                                "session_id": self.experiment_session_id,
                                "device_id": self.device_id or "",
                            },
                            priority="prewarm_minimal",
                        )
                    except Exception as exc:
                        self.logger.bind(tag=TAG).warning(
                            "experiment auto export configuration failed: "
                            f"device_id={self.device_id}, "
                            f"session_id={self.experiment_session_id}, error={exc}"
                        )
                if self.chat_session_id and self.experiment_session_id:
                    await save_experiment_session_binding(
                        self.config,
                        chat_session_id=self.chat_session_id,
                        model_session_key=self.model_session_key or "",
                        device_id=self.device_id or "",
                        user_id=self.user_id or "",
                        yaml_path=yaml_path,
                        experiment_session_id=self.experiment_session_id,
                        status="active",
                        source=session_source or "create",
                        current_step_id=current_step_id,
                        completed_steps_count=completed_steps_count,
                        total_steps=total_steps,
                    )

                debug_delay_seconds = self._experiment_prewarm_debug_delay_seconds()
                if debug_delay_seconds > 0:
                    self.logger.bind(tag=TAG).info(
                        "experiment prewarm debug delay before minimal ready: "
                        f"device_id={self.device_id}, "
                        f"trigger={self.experiment_prewarm_trigger}, "
                        f"delay_seconds={debug_delay_seconds}"
                    )
                    await asyncio.sleep(debug_delay_seconds)

                self.experiment_prewarm_ready_level = "minimal_ready"
                self.experiment_prewarm_minimal_ready_at = time.time()
                self.experiment_prewarm_status = "deep_warming"
                minimal_ready_reached = True
                minimal_ready_event.set()
                self.logger.bind(tag=TAG).info(
                    "experiment prewarm minimal ready: "
                    f"device_id={self.device_id}, trigger={self.experiment_prewarm_trigger}, "
                    f"session_id={self.experiment_session_id}, "
                    f"current_step_id={self.experiment_current_step_id}, "
                    f"completed_steps={completed_steps_count}/{total_steps or '?'}, "
                    f"{self._experiment_context_presence_log_fields()}"
                    f"{self._experiment_prewarm_timing_log_suffix()}"
                )

                await self._run_experiment_prewarm_deep_stage(
                    session_id=session_id,
                    progress_summary_payload=progress_summary_payload,
                )
                return True
            except asyncio.CancelledError:
                self.experiment_prewarm_status = "cancelled"
                self.experiment_prewarm_error = "prewarm cancelled"
                raise
            except Exception as exc:
                self.experiment_prewarm_error = str(exc)
                if minimal_ready_reached:
                    self.experiment_prewarm_status = "failed"
                    self.logger.bind(tag=TAG).warning(
                        "experiment prewarm deep warming failed after minimal ready: "
                        f"device_id={self.device_id}, trigger={self.experiment_prewarm_trigger}, "
                        f"session_id={self.experiment_session_id}, error={exc}"
                    )
                    return True

                self.experiment_prewarm_status = "failed"
                self.logger.bind(tag=TAG).warning(
                    "experiment prewarm failed before minimal ready: "
                    f"device_id={self.device_id}, trigger={self.experiment_prewarm_trigger}, "
                    f"error={exc}"
                )
                return False

    def schedule_experiment_prewarm(self, trigger: str = "") -> bool:
        if not self._experiment_prewarm_enabled():
            return False
        if not self.loop:
            return False
        if self._experiment_prewarm_is_minimal_ready():
            return False
        existing_task = self.experiment_prewarm_task
        if existing_task is not None and not existing_task.done():
            return False

        async def _runner():
            await self.prewarm_experiment_session(trigger=trigger)

        self.experiment_prewarm_task = asyncio.create_task(_runner())
        return True

    async def wait_for_experiment_prewarm_for_real_user_turn(
        self,
        timeout_seconds: float,
        trigger: str = "first_real_user_turn",
    ) -> Dict[str, str]:
        wait_result = "disabled"
        timeout_seconds = max(0.0, float(timeout_seconds or 0.0))

        if not self._experiment_prewarm_enabled():
            return self._experiment_prewarm_route_context(wait_result=wait_result)

        if self._experiment_prewarm_is_minimal_ready():
            self.logger.bind(tag=TAG).info(
                "experiment prewarm wait immediate-ready: "
                f"device_id={self.device_id}, trigger={trigger}, "
                f"session_id={self.experiment_session_id}, "
                f"ready_level={self.experiment_prewarm_ready_level}, "
                f"status={self.experiment_prewarm_status}"
                f"{self._experiment_prewarm_timing_log_suffix()}"
            )
            return self._experiment_prewarm_route_context(wait_result="ready")

        if self.experiment_prewarm_status == "idle":
            self.schedule_experiment_prewarm(trigger=trigger)

        task = self.experiment_prewarm_task
        if task is None:
            wait_result = (
                "ready"
                if self._experiment_prewarm_is_minimal_ready()
                else (self.experiment_prewarm_status or "idle")
            )
            return self._experiment_prewarm_route_context(wait_result=wait_result)

        if task.done():
            wait_result = (
                "ready"
                if self._experiment_prewarm_is_minimal_ready()
                else (self.experiment_prewarm_status or "done")
            )
            self.logger.bind(tag=TAG).info(
                "experiment prewarm wait finished-before-block: "
                f"device_id={self.device_id}, trigger={trigger}, result={wait_result}, "
                f"session_id={self.experiment_session_id}, "
                f"ready_level={self.experiment_prewarm_ready_level}, "
                f"status={self.experiment_prewarm_status}"
                f"{self._experiment_prewarm_timing_log_suffix()}"
            )
            return self._experiment_prewarm_route_context(wait_result=wait_result)

        if timeout_seconds <= 0:
            self.logger.bind(tag=TAG).info(
                "experiment prewarm wait skipped: "
                f"device_id={self.device_id}, trigger={trigger}, timeout_seconds={timeout_seconds}"
            )
            return self._experiment_prewarm_route_context(wait_result="skipped")

        self.logger.bind(tag=TAG).info(
            "experiment prewarm wait begin: "
            f"device_id={self.device_id}, trigger={trigger}, timeout_seconds={timeout_seconds}, "
            f"status={self.experiment_prewarm_status}, "
            f"ready_level={self.experiment_prewarm_ready_level}"
            f"{self._experiment_prewarm_timing_log_suffix()}"
        )
        minimal_ready_event = await self._get_experiment_prewarm_minimal_ready_event()
        wait_task = asyncio.create_task(minimal_ready_event.wait())
        try:
            done, pending = await asyncio.wait(
                {wait_task, task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if wait_task in done and minimal_ready_event.is_set():
                wait_result = (
                    "ready"
                    if self._experiment_prewarm_is_minimal_ready()
                    else (self.experiment_prewarm_status or "done")
                )
                self.logger.bind(tag=TAG).info(
                    "experiment prewarm wait minimal-ready: "
                    f"device_id={self.device_id}, trigger={trigger}, result={wait_result}, "
                    f"session_id={self.experiment_session_id}, "
                    f"ready_level={self.experiment_prewarm_ready_level}, "
                    f"status={self.experiment_prewarm_status}, "
                    f"current_step_id={self.experiment_current_step_id}, "
                    f"{self._experiment_context_presence_log_fields()}"
                    f"{self._experiment_prewarm_timing_log_suffix()}"
                )
            elif task in done:
                wait_result = (
                    "ready"
                    if self._experiment_prewarm_is_minimal_ready()
                    else (self.experiment_prewarm_status or "done")
                )
                self.logger.bind(tag=TAG).info(
                    "experiment prewarm wait finished: "
                    f"device_id={self.device_id}, trigger={trigger}, result={wait_result}, "
                    f"session_id={self.experiment_session_id}, "
                    f"ready_level={self.experiment_prewarm_ready_level}, "
                    f"status={self.experiment_prewarm_status}, "
                    f"{self._experiment_context_presence_log_fields()}"
                    f"{self._experiment_prewarm_timing_log_suffix()}"
                )
            else:
                wait_result = "timeout"
                self.logger.bind(tag=TAG).info(
                    "experiment prewarm wait timeout: "
                    f"device_id={self.device_id}, trigger={trigger}, timeout_seconds={timeout_seconds}, "
                    f"status={self.experiment_prewarm_status}, "
                    f"ready_level={self.experiment_prewarm_ready_level}"
                    f"{self._experiment_prewarm_timing_log_suffix()}"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            wait_result = "failed"
            self.logger.bind(tag=TAG).warning(
                "experiment prewarm wait failed: "
                f"device_id={self.device_id}, trigger={trigger}, error={exc}"
                f"{self._experiment_prewarm_timing_log_suffix()}"
            )
        finally:
            if not wait_task.done():
                wait_task.cancel()
            try:
                await wait_task
            except asyncio.CancelledError:
                pass
        return self._experiment_prewarm_route_context(wait_result=wait_result)

    @staticmethod
    def _ws_is_open(ws) -> bool:
        if ws is None:
            return False
        try:
            if hasattr(ws, "closed"):
                return not ws.closed
            state = getattr(ws, "state", None)
            if state is not None:
                return getattr(state, "name", str(state)) != "CLOSED"
        except Exception:
            return False
        return True

    def can_accept_reconnect(self) -> bool:
        return bool(
            self.device_id
            and self._connection_started
            and not self._closed
            and not self._final_close_requested
            and self._allow_transport_reconnect
            and not self._ws_is_open(self.websocket)
        )

    async def wait_until_transport_detached(self, timeout: float = 2.0) -> bool:
        if self.can_accept_reconnect():
            return True
        if timeout is not None and timeout > 0:
            try:
                await asyncio.wait_for(
                    self._transport_detached_event.wait(), timeout=timeout
                )
            except asyncio.TimeoutError:
                return self.can_accept_reconnect()
        return self.can_accept_reconnect()

    def request_final_close(self, reason: str = ""):
        self._final_close_requested = True
        if reason:
            self.logger.bind(tag=TAG).info(
                f"连接已标记为最终关闭: session_id={self.session_id}, "
                f"device_id={self.device_id}, reason={reason}"
            )

    @staticmethod
    def _drain_queue(q):
        if not q:
            return
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

    def _reset_transport_state(self):
        self.client_abort = False
        self.client_is_speaking = False
        self.client_have_voice = False
        self.client_voice_stop = False
        self.last_is_voice = False
        self.current_speaker = None
        self.current_language_tag = None
        self.tts_MessageText = ""

        self.client_audio_buffer = bytearray()
        self.client_voice_window.clear()
        self.asr_audio.clear()
        if hasattr(self, "asr_pcm_audio"):
            self.asr_pcm_audio.clear()
        self._pcm_packet_for_asr = None

        if hasattr(self, "audio_timestamp_buffer"):
            self.audio_timestamp_buffer.clear()
            self.last_processed_timestamp = 0

        self._drain_queue(getattr(self, "asr_audio_queue", None))

        if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
            self.audio_rate_controller.reset()

        if self.tts:
            self._drain_queue(getattr(self.tts, "tts_text_queue", None))
            self._drain_queue(getattr(self.tts, "tts_audio_queue", None))

        try:
            self.reset_vad_states()
        except Exception:
            pass

        try:
            if getattr(self, "audio_frontend", None):
                self.audio_frontend.reset()
        except Exception:
            pass

    def _should_preserve_transport_disconnect(self, disconnect_code) -> bool:
        if not self.keep_resources_on_transport_disconnect:
            return False
        if self._final_close_requested or self.close_after_chat or self._closed:
            return False
        if not self.device_id:
            return False
        return disconnect_code == 1006

    async def _detach_transport(self, ws=None, disconnect_code=None):
        target_ws = ws if ws else self.websocket
        self.logger.bind(tag=TAG).info(
            "[debug-close] transport detached, preserving resources: "
            f"session_id={self.session_id}, device_id={self.device_id}, "
            f"disconnect_code={disconnect_code}, "
            f"target_ws_state={self._format_ws_state(target_ws)}, "
            f"self_ws_state={self._format_ws_state(self.websocket)}"
        )

        self._reset_transport_state()
        now_ms = time.time() * 1000
        self.last_activity_time = now_ms
        self.first_activity_time = now_ms

        try:
            if target_ws and self._ws_is_open(target_ws):
                await target_ws.close()
        except Exception:
            pass
        finally:
            if target_ws is None or self.websocket is target_ws:
                self.websocket = None
            self._allow_transport_reconnect = True
            self._transport_detached_event.set()
            self.logger.bind(tag=TAG).info(
                "连接进入等待重连状态，资源未释放: "
                f"session_id={self.session_id}, device_id={self.device_id}, "
                f"disconnect_code={disconnect_code}"
            )

    def _initialize_audio_frontend(self):
        """为当前连接初始化音频前处理（AEC/NS等）。"""
        try:
            self.audio_frontend = AudioFrontend(self.config.get("audio_frontend", {}))
            if getattr(getattr(self.audio_frontend, "config", None), "enabled", False):
                self.logger.bind(tag=TAG).info("音频前处理已启用")
            else:
                self.logger.bind(tag=TAG).info("音频前处理未启用")
        except Exception as e:
            # Fail safe: keep disabled on errors.
            self.audio_frontend = AudioFrontend({"enabled": False})
            self.logger.bind(tag=TAG).warning(f"音频前处理初始化失败，已降级为关闭: {e}")

    async def handle_connection(self, ws):
        disconnect_code = None
        preserve_transport = False
        try:
            # 获取运行中的事件循环（必须在异步上下文中）
            self.loop = asyncio.get_running_loop()

            # 获取并验证headers
            self.headers = dict(ws.request.headers)
            real_ip = self.headers.get("x-real-ip") or self.headers.get(
                "x-forwarded-for"
            )
            if real_ip:
                self.client_ip = real_ip.split(",")[0].strip()
            else:
                self.client_ip = ws.remote_address[0]
            self.logger.bind(tag=TAG).info(
                f"{self.client_ip} conn - Headers: {self.headers}"
            )

            self.device_id = self.headers.get("device-id", None)

            # 认证通过,继续处理
            self.websocket = ws
            self._allow_transport_reconnect = False
            self._transport_detached_event.clear()
            if self.server and hasattr(self.server, "register_connection"):
                try:
                    await self.server.register_connection(self)
                except Exception as register_error:
                    self.logger.bind(tag=TAG).error(
                        f"register connection failed: {register_error}"
                    )

            # 检查是否来自MQTT连接
            request_path = ws.request.path
            self.conn_from_mqtt_gateway = request_path.endswith("?from=mqtt_gateway")
            if self.conn_from_mqtt_gateway:
                self.logger.bind(tag=TAG).info("连接来自:MQTT网关")

            now_ms = time.time() * 1000
            self.last_activity_time = now_ms

            if not self._connection_started:
                # 初始化活动时间戳
                self.first_activity_time = now_ms

                # 启动超时检查任务；当 close_connection_no_voice_time <= 0 时禁用自动断开
                if self.timeout_seconds > 0:
                    self.timeout_task = asyncio.create_task(self._check_timeout())

                self.welcome_msg = self.config["xiaozhi"]
                self.welcome_msg["session_id"] = self.session_id

                # 在后台初始化配置和组件（完全不阻塞主循环）
                asyncio.create_task(self._background_initialize())
                self._connection_started = True
            else:
                self.close_after_chat = False
                self.client_abort = False
                self.client_is_speaking = False
                self.logger.bind(tag=TAG).info(
                    "同设备重连接管现有连接资源: "
                    f"session_id={self.session_id}, device_id={self.device_id}, "
                    f"ws_state={self._format_ws_state(self.websocket)}"
                )

            try:
                async for message in self.websocket:
                    await self._route_message(message)
            except websockets.exceptions.ConnectionClosed as e:
                disconnect_code = getattr(e, "code", None)
                self.logger.bind(tag=TAG).info(
                    f"客户端断开连接: {self._describe_connection_closed(e)}"
                )

        except AuthenticationError as e:
            self.logger.bind(tag=TAG).error(f"Authentication failed: {str(e)}")
            return
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.bind(tag=TAG).error(f"Connection error: {str(e)}-{stack_trace}")
            return
        finally:
            try:
                if disconnect_code is None:
                    disconnect_code = getattr(ws, "close_code", None)
                preserve_transport = self._should_preserve_transport_disconnect(
                    disconnect_code
                )
                if preserve_transport:
                    await self._detach_transport(ws, disconnect_code)
                else:
                    await self._save_and_close(ws)
            except Exception as final_error:
                self.logger.bind(tag=TAG).error(f"最终清理时出错: {final_error}")
                # 确保即使保存记忆失败，也要关闭连接
                try:
                    self.request_final_close("cleanup fallback")
                    await self.close(ws)
                except Exception as close_error:
                    self.logger.bind(tag=TAG).error(
                        f"强制关闭连接时出错: {close_error}"
                    )
            if (
                not preserve_transport
                and self.server
                and hasattr(self.server, "unregister_connection")
            ):
                try:
                    await self.server.unregister_connection(self)
                except Exception as unregister_error:
                    self.logger.bind(tag=TAG).error(
                        f"unregister connection failed: {unregister_error}"
                    )

    async def _save_and_close(self, ws):
        """保存记忆并关闭连接"""
        try:
            self.logger.bind(tag=TAG).info(
                "[debug-close] _save_and_close begin: "
                f"session_id={self.session_id}, device_id={self.device_id}, "
                f"ws_state={self._format_ws_state(ws if ws else self.websocket)}, "
                f"memory_enabled={self.memory is not None}"
            )
            if self.memory:
                # 使用线程池异步保存记忆
                def save_memory_task():
                    try:
                        # 创建新事件循环（避免与主循环冲突）
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.memory.save_memory(
                                self.dialogue.dialogue, self._memory_session_key()
                            )
                        )
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"保存记忆失败: {e}")
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                # 启动线程保存记忆，不等待完成
                threading.Thread(target=save_memory_task, daemon=True).start()
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"保存记忆失败: {e}")
        finally:
            # 立即关闭连接，不等待记忆保存完成
            try:
                await self.close(ws)
            except Exception as close_error:
                self.logger.bind(tag=TAG).error(
                    f"保存记忆后关闭连接失败: {close_error}"
                )

    async def _discard_message_with_bind_prompt(self):
        """丢弃消息并检查是否需要播放绑定提示"""
        current_time = time.time()
        # 检查是否需要播放绑定提示
        if current_time - self.last_bind_prompt_time >= self.bind_prompt_interval:
            self.last_bind_prompt_time = current_time
            # 复用现有的绑定提示逻辑
            from core.handle.receiveAudioHandle import check_bind_device

            asyncio.create_task(check_bind_device(self))

    async def _route_message(self, message):
        """消息路由"""
        # 检查是否已经获取到真实的绑定状态
        if not self.bind_completed_event.is_set():
            # 还没有获取到真实状态，等待直到获取到真实状态或超时
            try:
                await asyncio.wait_for(self.bind_completed_event.wait(), timeout=1)
            except asyncio.TimeoutError:
                # 超时仍未获取到真实状态，丢弃消息
                await self._discard_message_with_bind_prompt()
                return

        # 已经获取到真实状态，检查是否需要绑定
        if self.need_bind:
            # 需要绑定，丢弃消息
            await self._discard_message_with_bind_prompt()
            return

        # 不需要绑定，继续处理消息

        if isinstance(message, str):
            await handleTextMessage(self, message)
        elif isinstance(message, bytes):
            if self.vad is None or self.asr is None:
                return

            # 处理来自MQTT网关的音频包
            if self.conn_from_mqtt_gateway and len(message) >= 16:
                handled = await self._process_mqtt_audio_message(message)
                if handled:
                    return

            # 不需要头部处理或没有头部时，直接处理原始消息
            self.asr_audio_queue.put(message)

    async def _process_mqtt_audio_message(self, message):
        """
        处理来自MQTT网关的音频消息，解析16字节头部并提取音频数据

        Args:
            message: 包含头部的音频消息

        Returns:
            bool: 是否成功处理了消息
        """
        try:
            # 提取头部信息
            timestamp = int.from_bytes(message[8:12], "big")
            audio_length = int.from_bytes(message[12:16], "big")

            # 提取音频数据
            if audio_length > 0 and len(message) >= 16 + audio_length:
                # 有指定长度，提取精确的音频数据
                audio_data = message[16 : 16 + audio_length]
                # 基于时间戳进行排序处理
                self._process_websocket_audio(audio_data, timestamp)
                return True
            elif len(message) > 16:
                # 没有指定长度或长度无效，去掉头部后处理剩余数据
                audio_data = message[16:]
                self.asr_audio_queue.put(audio_data)
                return True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"解析WebSocket音频包失败: {e}")

        # 处理失败，返回False表示需要继续处理
        return False

    def _process_websocket_audio(self, audio_data, timestamp):
        """处理WebSocket格式的音频包"""
        # 初始化时间戳序列管理
        if not hasattr(self, "audio_timestamp_buffer"):
            self.audio_timestamp_buffer = {}
            self.last_processed_timestamp = 0
            self.max_timestamp_buffer_size = 20

        # 如果时间戳是递增的，直接处理
        if timestamp >= self.last_processed_timestamp:
            self.asr_audio_queue.put(audio_data)
            self.last_processed_timestamp = timestamp

            # 处理缓冲区中的后续包
            processed_any = True
            while processed_any:
                processed_any = False
                for ts in sorted(self.audio_timestamp_buffer.keys()):
                    if ts > self.last_processed_timestamp:
                        buffered_audio = self.audio_timestamp_buffer.pop(ts)
                        self.asr_audio_queue.put(buffered_audio)
                        self.last_processed_timestamp = ts
                        processed_any = True
                        break
        else:
            # 乱序包，暂存
            if len(self.audio_timestamp_buffer) < self.max_timestamp_buffer_size:
                self.audio_timestamp_buffer[timestamp] = audio_data
            else:
                self.asr_audio_queue.put(audio_data)

    async def handle_restart(self, message):
        """处理服务器重启请求"""
        try:

            self.logger.bind(tag=TAG).info("收到服务器重启指令，准备执行...")

            # 发送确认响应
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "success",
                        "message": "服务器重启中...",
                        "content": {"action": "restart"},
                    }
                )
            )

            # 异步执行重启操作
            def restart_server():
                """实际执行重启的方法"""
                time.sleep(1)
                self.logger.bind(tag=TAG).info("执行服务器重启...")
                subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    start_new_session=True,
                )
                os._exit(0)

            # 使用线程执行重启避免阻塞事件循环
            threading.Thread(target=restart_server, daemon=True).start()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"重启失败: {str(e)}")
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "error",
                        "message": f"Restart failed: {str(e)}",
                        "content": {"action": "restart"},
                    }
                )
            )

    def _initialize_components(self):
        try:
            if self.tts is None:
                self.tts = self._initialize_tts()
            # 打开语音合成通道
            asyncio.run_coroutine_threadsafe(
                self.tts.open_audio_channels(self), self.loop
            )
            if self.need_bind:
                self.bind_completed_event.set()
                return
            self.selected_module_str = build_module_string(
                self.config.get("selected_module", {})
            )
            self.logger = create_connection_logger(self.selected_module_str)

            """初始化组件"""
            if self.config.get("prompt") is not None:
                user_prompt = self.config["prompt"]
                # 使用快速提示词进行初始化
                self.prompt_manager.update_context_info(self, self.client_ip)
                prompt = self.prompt_manager.build_enhanced_prompt(
                    user_prompt, self.device_id, self.client_ip
                )
                if not prompt:
                    prompt = self.prompt_manager.get_quick_prompt(
                        user_prompt, self.device_id
                    )
                self.change_system_prompt(prompt)
                self.logger.bind(tag=TAG).info(
                    f"快速初始化组件: prompt成功 {prompt[:50]}..."
                )

            """初始化本地组件"""
            if self.vad is None:
                self.vad = self._vad
            if self.asr is None:
                self.asr = self._initialize_asr()

            # 初始化音频前处理（AEC/NS等）
            self._initialize_audio_frontend()

            # 初始化声纹识别
            self._initialize_voiceprint()
            # 打开语音识别通道
            asyncio.run_coroutine_threadsafe(
                self.asr.open_audio_channels(self), self.loop
            )

            """加载记忆"""
            self._initialize_memory()
            """加载意图识别"""
            self._initialize_intent()
            """初始化上报线程"""
            self._init_report_threads()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"实例化组件失败: {e}")

    

    def _init_report_threads(self):
        """初始化ASR和TTS上报线程"""
        if not self.read_config_from_api or self.need_bind:
            return
        if self.chat_history_conf == 0:
            return
        if self.report_thread is None or not self.report_thread.is_alive():
            self.report_thread = threading.Thread(
                target=self._report_worker, daemon=True
            )
            self.report_thread.start()
            self.logger.bind(tag=TAG).info("TTS上报线程已启动")

    def _initialize_tts(self):
        """初始化TTS"""
        tts = None
        if not self.need_bind:
            tts = initialize_tts(self.config)

        if tts is None:
            tts = DefaultTTS(self.config, delete_audio_file=True)

        return tts

    def _initialize_asr(self):
        """初始化ASR"""
        if (
            self._asr is not None
            and hasattr(self._asr, "interface_type")
            and self._asr.interface_type == InterfaceType.LOCAL
        ):
            # 如果公共ASR是本地服务，则直接返回
            # 因为本地一个实例ASR，可以被多个连接共享
            asr = self._asr
        else:
            # 如果公共ASR是远程服务，则初始化一个新实例
            # 因为远程ASR，涉及到websocket连接和接收线程，需要每个连接一个实例
            asr = initialize_asr(self.config)

        return asr

    def _initialize_voiceprint(self):
        """为当前连接初始化声纹识别"""
        try:
            voiceprint_config = self.config.get("voiceprint", {})
            if not voiceprint_config:
                self.logger.bind(tag=TAG).info("声纹识别功能未启用")
                return

            if not is_voiceprint_feature_enabled(voiceprint_config):
                self.logger.bind(tag=TAG).info("声纹识别总开关已关闭")
                return

            runtime_device_id = str(self.device_id or "").strip()
            if not runtime_device_id and isinstance(self.headers, dict):
                runtime_device_id = str(
                    self.headers.get("device-id", self.headers.get("client-id", ""))
                ).strip()
            runtime_transport_id = str(
                self.transport_session_id or self.session_id
            ).strip()
            if runtime_device_id:
                runtime_scope = f"{runtime_device_id}__{runtime_transport_id}"
            else:
                runtime_scope = runtime_transport_id

            voiceprint_provider = VoiceprintProvider(
                voiceprint_config,
                runtime_scope=runtime_scope,
            )
            if voiceprint_provider is not None and voiceprint_provider.enabled:
                self.voiceprint_provider = voiceprint_provider
                self.logger.bind(tag=TAG).info(
                    "声纹识别功能已在连接时动态启用: "
                    f"runtime_scope={runtime_scope}, "
                    f"master_speaker_id={voiceprint_provider.dynamic_master_speaker_id}"
                )
            else:
                self.logger.bind(tag=TAG).warning("声纹识别功能已开启，但配置不完整或服务不可用")
        except Exception as e:
            self.logger.bind(tag=TAG).warning(f"声纹识别初始化失败: {str(e)}")

    async def _background_initialize(self):
        """在后台初始化配置和组件（完全不阻塞主循环）"""
        try:
            # 异步获取差异化配置
            await self._initialize_private_config_async()
            # 在线程池中初始化组件
            self.executor.submit(self._initialize_components)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"后台初始化失败: {e}")

    async def _initialize_private_config_async(self):
        """从接口异步获取差异化配置（异步版本，不阻塞主循环）"""
        if not self.read_config_from_api:
            self.need_bind = False
            self.bind_completed_event.set()
            return
        try:
            begin_time = time.time()
            private_config = await get_private_config_from_api(
                self.config,
                self.headers.get("device-id"),
                self.headers.get("client-id", self.headers.get("device-id")),
            )
            private_config["delete_audio"] = bool(self.config.get("delete_audio", True))
            self.logger.bind(tag=TAG).info(
                f"{time.time() - begin_time} 秒，异步获取差异化配置成功: {json.dumps(filter_sensitive_info(private_config), ensure_ascii=False)}"
            )
            self.need_bind = False
            self.bind_completed_event.set()
        except DeviceNotFoundException as e:
            self.need_bind = True
            private_config = {}
        except DeviceBindException as e:
            self.need_bind = True
            self.bind_code = e.bind_code
            private_config = {}
        except Exception as e:
            self.need_bind = True
            self.logger.bind(tag=TAG).error(f"异步获取差异化配置失败: {e}")
            private_config = {}

        init_llm, init_tts, init_memory, init_intent = (
            False,
            False,
            False,
            False,
        )

        init_vad = check_vad_update(self.common_config, private_config)
        init_asr = check_asr_update(self.common_config, private_config)

        if init_vad:
            self.config["VAD"] = private_config["VAD"]
            self.config["selected_module"]["VAD"] = private_config["selected_module"][
                "VAD"
            ]
        if init_asr:
            self.config["ASR"] = private_config["ASR"]
            self.config["selected_module"]["ASR"] = private_config["selected_module"][
                "ASR"
            ]
        if private_config.get("TTS", None) is not None:
            init_tts = True
            self.config["TTS"] = private_config["TTS"]
            self.config["selected_module"]["TTS"] = private_config["selected_module"][
                "TTS"
            ]
        if private_config.get("LLM", None) is not None:
            init_llm = True
            self.config["LLM"] = private_config["LLM"]
            self.config["selected_module"]["LLM"] = private_config["selected_module"][
                "LLM"
            ]
        if private_config.get("VLLM", None) is not None:
            self.config["VLLM"] = private_config["VLLM"]
            self.config["selected_module"]["VLLM"] = private_config["selected_module"][
                "VLLM"
            ]
        if private_config.get("Memory", None) is not None:
            init_memory = True
            self.config["Memory"] = private_config["Memory"]
            self.config["selected_module"]["Memory"] = private_config[
                "selected_module"
            ]["Memory"]
        if private_config.get("Intent", None) is not None:
            init_intent = True
            self.config["Intent"] = private_config["Intent"]
            model_intent = private_config.get("selected_module", {}).get("Intent", {})
            self.config["selected_module"]["Intent"] = model_intent
            # 加载插件配置
            if model_intent != "Intent_nointent":
                plugin_from_server = private_config.get("plugins", {})
                for plugin, config_str in plugin_from_server.items():
                    plugin_from_server[plugin] = json.loads(config_str)
                self.config["plugins"] = plugin_from_server
                self.config["Intent"][self.config["selected_module"]["Intent"]][
                    "functions"
                ] = plugin_from_server.keys()
        if private_config.get("prompt", None) is not None:
            self.config["prompt"] = private_config["prompt"]
        # 获取声纹信息
        if private_config.get("voiceprint", None) is not None:
            self.config["voiceprint"] = private_config["voiceprint"]
        if private_config.get("summaryMemory", None) is not None:
            self.config["summaryMemory"] = private_config["summaryMemory"]
        if private_config.get("device_max_output_size", None) is not None:
            self.max_output_size = int(private_config["device_max_output_size"])
        if private_config.get("chat_history_conf", None) is not None:
            self.chat_history_conf = int(private_config["chat_history_conf"])
        if private_config.get("mcp_endpoint", None) is not None:
            self.config["mcp_endpoint"] = private_config["mcp_endpoint"]
        if private_config.get("context_providers", None) is not None:
            self.config["context_providers"] = private_config["context_providers"]

        # 使用 run_in_executor 在线程池中执行 initialize_modules，避免阻塞主循环
        try:
            modules = await self.loop.run_in_executor(
                None,  # 使用默认线程池
                initialize_modules,
                self.logger,
                private_config,
                init_vad,
                init_asr,
                init_llm,
                init_tts,
                init_memory,
                init_intent,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"初始化组件失败: {e}")
            modules = {}
        if modules.get("tts", None) is not None:
            self.tts = modules["tts"]
        if modules.get("vad", None) is not None:
            self.vad = modules["vad"]
        if modules.get("asr", None) is not None:
            self.asr = modules["asr"]
        if modules.get("llm", None) is not None:
            self.llm = modules["llm"]
        if modules.get("intent", None) is not None:
            self.intent = modules["intent"]
        if modules.get("memory", None) is not None:
            self.memory = modules["memory"]

    def _initialize_memory(self):
        if self.memory is None:
            return
        """初始化记忆模块"""
        self.memory.init_memory(
            role_id=self.device_id,
            llm=self.llm,
            summary_memory=self.config.get("summaryMemory", None),
            save_to_file=not self.read_config_from_api,
        )

        # 获取记忆总结配置
        memory_config = self.config["Memory"]
        memory_type = self.config["Memory"][self.config["selected_module"]["Memory"]][
            "type"
        ]
        # 如果使用 nomen，直接返回
        if memory_type == "nomem":
            return
        # 使用 mem_local_short 模式
        elif memory_type == "mem_local_short":
            memory_llm_name = memory_config[self.config["selected_module"]["Memory"]][
                "llm"
            ]
            if memory_llm_name and memory_llm_name in self.config["LLM"]:
                # 如果配置了专用LLM，则创建独立的LLM实例
                from core.utils import llm as llm_utils

                memory_llm_config = self.config["LLM"][memory_llm_name]
                memory_llm_type = memory_llm_config.get("type", memory_llm_name)
                memory_llm = llm_utils.create_instance(
                    memory_llm_type, memory_llm_config
                )
                self.logger.bind(tag=TAG).info(
                    f"为记忆总结创建了专用LLM: {memory_llm_name}, 类型: {memory_llm_type}"
                )
                self.memory.set_llm(memory_llm)
            else:
                # 否则使用主LLM
                self.memory.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("使用主LLM作为意图识别模型")

    def _initialize_intent(self):
        if self.intent is None:
            return
        self.intent_type = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ]["type"]
        if self.intent_type == "function_call" or self.intent_type == "intent_llm":
            self.load_function_plugin = True
        """初始化意图识别模块"""
        # 获取意图识别配置
        intent_config = self.config["Intent"]
        intent_type = self.config["Intent"][self.config["selected_module"]["Intent"]][
            "type"
        ]

        # 如果使用 nointent，直接返回
        if intent_type == "nointent":
            return
        # 使用 intent_llm 模式
        elif intent_type == "intent_llm":
            intent_llm_name = intent_config[self.config["selected_module"]["Intent"]][
                "llm"
            ]

            if intent_llm_name and intent_llm_name in self.config["LLM"]:
                # 如果配置了专用LLM，则创建独立的LLM实例
                from core.utils import llm as llm_utils

                intent_llm_config = self.config["LLM"][intent_llm_name]
                intent_llm_type = intent_llm_config.get("type", intent_llm_name)
                intent_llm = llm_utils.create_instance(
                    intent_llm_type, intent_llm_config
                )
                self.logger.bind(tag=TAG).info(
                    f"为意图识别创建了专用LLM: {intent_llm_name}, 类型: {intent_llm_type}"
                )
                self.intent.set_llm(intent_llm)
            else:
                # 否则使用主LLM
                self.intent.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("使用主LLM作为意图识别模型")

        """加载统一工具处理器"""
        self.func_handler = UnifiedToolHandler(self)

        # 异步初始化工具处理器
        if hasattr(self, "loop") and self.loop:
            asyncio.run_coroutine_threadsafe(self.func_handler._initialize(), self.loop)

    def _append_resume_tool_context(self, llm_dialogue, query, depth):
        if depth != 0:
            return llm_dialogue

        resume_tool_message = build_resume_tool_message(
            self.config,
            self.device_id,
            query,
        )
        if not resume_tool_message:
            return llm_dialogue

        enriched_dialogue = list(llm_dialogue)
        enriched_dialogue.append(resume_tool_message)
        self.logger.bind(tag=TAG).info(
            "appended device log recovery context to llm dialogue: "
            f"device_id={self.device_id}, log_turns={resume_tool_message['content'].count('学生：')}"
        )
        return enriched_dialogue

    def change_system_prompt(self, prompt):
        if prompt == self.prompt:
            return False
        self.prompt = prompt
        # 更新系统prompt至上下文
        self.dialogue.update_system_message(self.prompt)
        return True

    def _send_llm_event_message(self, text, event=None, phase=None):
        if not self.websocket or not self.loop:
            return
        payload = {"type": "stt", "text": text, "session_id": self.session_id}
        if event is not None:
            payload["event"] = event
        if phase is not None:
            payload["phase"] = phase
        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps(payload)),
                self.loop,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).warning(f"send llm event failed: {e}")

    def _start_thinking_pulse(self):
        if self.thinking_pulse_thread and self.thinking_pulse_thread.is_alive():
            return
        self.thinking_pulse_stop.clear()

        def _worker():
            dots = 1
            while not self.thinking_pulse_stop.is_set():
                self._send_llm_event_message(
                    f"[Thinking{'.' * dots}]",
                    event="thinking",
                    phase="progress",
                )
                dots = 1 if dots >= 3 else dots + 1
                self.thinking_pulse_stop.wait(0.6)

        self.thinking_pulse_thread = threading.Thread(target=_worker, daemon=True)
        self.thinking_pulse_thread.start()

    def _stop_thinking_pulse(self):
        if self.thinking_pulse_stop:
            self.thinking_pulse_stop.set()

    def _mark_thinking_event_started(self):
        with self._thinking_event_lock:
            self._thinking_event_active = True
            self._thinking_finish_on_tts_start_pending = False

    def _defer_thinking_finish_until_tts_start(self):
        with self._thinking_event_lock:
            if self._thinking_event_active:
                self._thinking_finish_on_tts_start_pending = True

    def _finish_thinking_event_if_active(self):
        should_send = False
        with self._thinking_event_lock:
            if self._thinking_event_active:
                self._thinking_event_active = False
                self._thinking_finish_on_tts_start_pending = False
                should_send = True
            else:
                self._thinking_finish_on_tts_start_pending = False
        if not should_send:
            return False
        self._stop_thinking_pulse()
        self._send_llm_event_message("[Thinking Finished]", event="thinking", phase="done")
        return True

    def _finish_deferred_thinking_on_tts_start(self):
        with self._thinking_event_lock:
            pending = self._thinking_finish_on_tts_start_pending
        if pending:
            return self._finish_thinking_event_if_active()
        return False

    def _start_action_pulse(self):
        if self.action_pulse_thread and self.action_pulse_thread.is_alive():
            return
        self.action_pulse_stop.clear()

        def _worker():
            dots = 1
            while not self.action_pulse_stop.is_set():
                self._send_llm_event_message(
                    f"[Action{'.' * dots}]",
                    event="action",
                    phase="progress",
                )
                dots = 1 if dots >= 3 else dots + 1
                self.action_pulse_stop.wait(0.6)

        self.action_pulse_thread = threading.Thread(target=_worker, daemon=True)
        self.action_pulse_thread.start()

    def _stop_action_pulse(self):
        if self.action_pulse_stop:
            self.action_pulse_stop.set()

    def acquire_external_busy(self, token: str):
        token = str(token or "").strip()
        if not token:
            return
        should_start = False
        with self._external_busy_lock:
            if token not in self._external_busy_tokens:
                self._external_busy_tokens.add(token)
                should_start = (
                    len(self._external_busy_tokens) == 1
                    and self.llm_finish_task
                    and not self.client_is_speaking
                )
        if should_start:
            self._send_llm_event_message("[Thinking]", event="thinking", phase="start")
            self._mark_thinking_event_started()
            self._start_thinking_pulse()

    def release_external_busy(self, token: str):
        token = str(token or "").strip()
        if not token:
            return
        should_stop = False
        with self._external_busy_lock:
            self._external_busy_tokens.discard(token)
            should_stop = not self._external_busy_tokens
        if should_stop:
            self._finish_thinking_event_if_active()

    def has_external_busy(self) -> bool:
        with self._external_busy_lock:
            return bool(self._external_busy_tokens)

    def chat(self, query, depth=0, is_real_user_turn=False):
        if query is not None:
            self.logger.bind(tag=TAG).info(f"大模型收到用户消息: {query}")

        # 为最顶层时新建会话ID和发送FIRST请求
        if depth == 0:
            self._llm_turn_started = True
            self.llm_finish_task = False
            with self._thinking_event_lock:
                self._thinking_event_active = False
                self._thinking_finish_on_tts_start_pending = False
            self.sentence_id = str(uuid.uuid4().hex)
            textUtils.activate_experiment_ready_guard_bypass_for_current_sentence(self)
            self.dialogue.put(Message(role="user", content=query))
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )

        # 设置最大递归深度，避免无限循环，可根据实际需求调整
        MAX_DEPTH = 5
        force_final_answer = False  # 标记是否强制最终回答

        if depth >= MAX_DEPTH:
            self.logger.bind(tag=TAG).debug(
                f"已达到最大工具调用深度 {MAX_DEPTH}，将强制基于现有信息回答"
            )
            force_final_answer = True
            # 添加系统指令，要求 LLM 基于现有信息回答
            self.dialogue.put(
                Message(
                    role="user",
                    content="[系统提示] 已达到最大工具调用次数限制，请你基于目前已经获取的所有信息，直接给出最终答案。不要再尝试调用任何工具。",
                )
            )

        # Define intent functions
        functions = None
        # 达到最大深度时，禁用工具调用，强制 LLM 直接回答
        if (
            self.intent_type == "function_call"
            and hasattr(self, "func_handler")
            and not force_final_answer
        ):
            functions = self.func_handler.get_functions()
        response_message = []

        try:
            # 使用带记忆的对话
            memory_str = None
            llm_route_kwargs = {}
            if depth == 0:
                llm_route_kwargs = self._llm_route_context_kwargs()
                llm_route_kwargs["state_conn"] = self
                should_wait_for_prewarm = (
                    is_real_user_turn
                    and query is not None
                    and self._consume_experiment_first_real_user_turn_gate()
                )
                if should_wait_for_prewarm and self.loop:
                    prewarm_wait_seconds = self._experiment_prewarm_wait_seconds()
                    self.logger.bind(tag=TAG).info(
                        "first real user turn entering prewarm gate: "
                        f"device_id={self.device_id}, wait_seconds={prewarm_wait_seconds}, "
                        f"query={str(query)[:120]}"
                    )
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self.wait_for_experiment_prewarm_for_real_user_turn(
                                prewarm_wait_seconds
                            ),
                            self.loop,
                        )
                        wait_timeout = max(prewarm_wait_seconds + 2.0, 2.0)
                        prewarm_route_context = future.result(timeout=wait_timeout)
                        if (
                            prewarm_route_context.get("experiment_prewarm_wait_result")
                            == "ready"
                            and prewarm_route_context.get("experiment_session_id")
                        ):
                            self.experiment_prewarm_session_adopted = True
                        self.logger.bind(tag=TAG).info(
                            "first real user turn prewarm gate result: "
                            f"device_id={self.device_id}, "
                            f"wait_result={prewarm_route_context.get('experiment_prewarm_wait_result', '')}, "
                            f"ready_level={prewarm_route_context.get('experiment_prewarm_ready_level', '')}, "
                            f"status={prewarm_route_context.get('experiment_prewarm_status', '')}, "
                            f"adopted={self.experiment_prewarm_session_adopted}, "
                            f"experiment_session_id={prewarm_route_context.get('experiment_session_id', '')}, "
                            f"experiment_current_step_id={prewarm_route_context.get('experiment_current_step_id', '')}, "
                            f"recovery_latest_current_step_id={prewarm_route_context.get('experiment_resume_latest_current_step_id', '')}, "
                            f"{self._experiment_context_presence_log_fields()}"
                            f"{self._experiment_prewarm_timing_log_suffix()}"
                        )
                        llm_route_kwargs.update(prewarm_route_context)
                    except Exception as exc:
                        self.logger.bind(tag=TAG).warning(
                            "experiment prewarm wait bridge failed: "
                            f"device_id={self.device_id}, error={exc}"
                        )
                        llm_route_kwargs.update(
                            self._experiment_prewarm_route_context(
                                wait_result="bridge_error"
                            )
                        )
                else:
                    llm_route_kwargs.update(self._experiment_prewarm_route_context())

                if is_real_user_turn and query is not None:
                    self.enrich_latest_clean_user_utterance_snapshot()

                if is_real_user_turn and query is not None and self.loop:
                    deep_prefetch_wait_seconds = (
                        self._experiment_deep_prefetch_micro_wait_seconds()
                    )
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self.wait_for_experiment_deep_prefetch(
                                query,
                                deep_prefetch_wait_seconds,
                            ),
                            self.loop,
                        )
                        deep_wait_timeout = max(deep_prefetch_wait_seconds + 2.0, 2.0)
                        deep_prefetch_context = future.result(
                            timeout=deep_wait_timeout
                        )
                        if deep_prefetch_context:
                            llm_route_kwargs.update(deep_prefetch_context)
                    except Exception as exc:
                        self.logger.bind(tag=TAG).warning(
                            "experiment deep prefetch wait bridge failed: "
                            f"device_id={self.device_id}, error={exc}"
                        )

                if is_real_user_turn and query is not None and self.loop:
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self.maybe_refresh_experiment_state_before_llm(
                                reason="real_user_turn"
                            ),
                            self.loop,
                        )
                        refreshed_route_context = future.result(timeout=3.0)
                        if refreshed_route_context:
                            llm_route_kwargs.update(refreshed_route_context)
                    except Exception as exc:
                        self.logger.bind(tag=TAG).warning(
                            "experiment foreground refresh bridge failed: "
                            f"device_id={self.device_id}, error={exc}"
                        )

                self.logger.bind(tag=TAG).info(
                    "llm turn route ready: "
                    f"device_id={self.device_id}, "
                    f"chat_session_id={llm_route_kwargs.get('chat_session_id', '')}, "
                    f"model_session_key={llm_route_kwargs.get('model_session_key', '')}, "
                    f"wait_result={llm_route_kwargs.get('experiment_prewarm_wait_result', '')}, "
                    f"experiment_status={llm_route_kwargs.get('experiment_prewarm_status', '')}, "
                    f"experiment_ready_level={llm_route_kwargs.get('experiment_prewarm_ready_level', '')}, "
                    f"experiment_session_id={llm_route_kwargs.get('experiment_session_id', '')}, "
                    f"experiment_current_step_id={llm_route_kwargs.get('experiment_current_step_id', '')}, "
                    f"deep_wait_result={llm_route_kwargs.get('experiment_deep_prefetch_wait_result', '')}, "
                    f"deep_status={llm_route_kwargs.get('experiment_deep_prefetch_status', '')}, "
                    f"deep_focus={llm_route_kwargs.get('experiment_deep_prefetch_focus', '')}, "
                    f"{self._experiment_context_presence_log_fields()}, "
                    f"recovery_required={llm_route_kwargs.get('experiment_resume_recovery_required', '')}, "
                    f"recovery_log_turns={llm_route_kwargs.get('experiment_resume_turn_count', '')}, "
                    f"recovery_latest_session_id={llm_route_kwargs.get('experiment_resume_latest_session_id', '')}, "
                    f"recovery_latest_current_step_id={llm_route_kwargs.get('experiment_resume_latest_current_step_id', '')}"
                )

            if self.memory is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self.memory.query_memory(query), self.loop
                )
                memory_str = future.result()

            llm_dialogue = self.dialogue.get_llm_dialogue_with_memory(
                memory_str, self.config.get("voiceprint", {})
            )
            llm_dialogue = self._append_resume_tool_context(llm_dialogue, query, depth)
            if depth > 0:
                # Internal recursive turn: avoid resending full system prompt.
                llm_dialogue = [
                    msg for msg in llm_dialogue if msg.get("role") != "system"
                ]

            if self.intent_type == "function_call" and functions is not None:
                # 使用支持functions的streaming接口
                llm_responses = self.llm.response_with_functions(
                    self._llm_session_key(),
                    llm_dialogue,
                    functions=functions,
                    **llm_route_kwargs,
                )
            else:
                llm_responses = self.llm.response(
                    self._llm_session_key(),
                    llm_dialogue,
                    **llm_route_kwargs,
                )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM 处理出错 {query}: {e}")
            self.llm_finish_task = True
            return None

        # 处理流式响应（满足：非 action、未产 content、0.5s 无信息增加 => thinking）
        tool_call_flag = False
        tool_calls_list = []  # 格式: [{"id": "", "name": "", "arguments": ""}]
        content_arguments = ""
        self.client_abort = False
        emotion_flag = True
        stream_tts_from_llm = bool(self.config.get("stream_tts_from_llm", True))

        thinking_event_sent = False
        thinking_event_done = False

        # --- idle -> thinking 状态变量 ---
        IDLE_TO_THINKING_SEC = 0.5
        last_progress_t = [time.monotonic()]  # “信息增加”的最后时间（只在增量有意义时更新）
        action_in_progress = threading.Event()   # True: action(start) 后未 done
        content_started = threading.Event()      # True: 真实 content 已开始输出
        watchdog_stop = threading.Event()

        def mark_progress():
            last_progress_t[0] = time.monotonic()

        def ensure_thinking_running():
            """只有在：非 action、未产 content、且 thinking 未结束 时，启动/维持 thinking pulse"""
            nonlocal thinking_event_sent, thinking_event_done
            if thinking_event_done:
                return
            if action_in_progress.is_set() or content_started.is_set():
                return
            if not thinking_event_sent:
                self._send_llm_event_message("[Thinking]", event="thinking", phase="start")
                thinking_event_sent = True
                self._mark_thinking_event_started()
            self._start_thinking_pulse()

        def finish_thinking_once():
            """第一次拿到真实 content 时，结束 thinking"""
            nonlocal thinking_event_done
            if thinking_event_sent and not thinking_event_done:
                if stream_tts_from_llm and bool(
                    self.config.get("thinking_finish_on_tts_start", True)
                ):
                    pause_thinking_pulse_only()
                    self._defer_thinking_finish_until_tts_start()
                else:
                    self._finish_thinking_event_if_active()
                thinking_event_done = True

        def pause_thinking_pulse_only():
            """action 期间暂停 thinking pulse，但不发 Thinking Finished"""
            if thinking_event_sent and not thinking_event_done:
                self._stop_thinking_pulse()

        def emit_action_event(phase):
            """action 事件出现时：暂停 thinking、跑 action pulse，并维护 action_in_progress 状态"""
            if phase == "start":
                action_in_progress.set()
                pause_thinking_pulse_only()
                self._start_action_pulse()
                label = "[Action Started]"
            elif phase == "done":
                action_in_progress.clear()
                self._stop_action_pulse()
                label = "[Action Finished]"
            else:
                label = "[Action]"
            self._send_llm_event_message(label, event="action", phase=phase)

        def is_toolcall_payload(prefix: str) -> bool:
            s = (prefix or "").lstrip()
            return s.startswith("<tool_call>")

        def _watchdog():
            # 规则：非 action、未产 content、且 0.5s 无信息增加 => thinking
            while not watchdog_stop.is_set():
                if (not action_in_progress.is_set()) and (not content_started.is_set()) and (not thinking_event_done):
                    idle = time.monotonic() - last_progress_t[0]
                    if idle >= IDLE_TO_THINKING_SEC:
                        ensure_thinking_running()
                watchdog_stop.wait(0.05)  # 50ms tick

        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            for response in llm_responses:
                if self.client_abort:
                    break

                # ---------- 1) 归一化：拆出 event / text_delta / tools_call ----------
                event = None
                text_delta = None
                tools_call = None

                if isinstance(response, dict):
                    # 注意：有些 provider 用 dict 承载 {"content":..., "tools_call":...}
                    if "content" in response:
                        text_delta = response.get("content")
                        tools_call = response.get("tools_call")
                    else:
                        event = response
                else:
                    if self.intent_type == "function_call" and functions is not None:
                        if isinstance(response, tuple) and len(response) == 2:
                            text_delta, tools_call = response
                        else:
                            text_delta = response
                    else:
                        text_delta = response

                # ---------- 2) 处理 event(dict) ----------
                if event is not None:
                    # event 算“信息增加”（你能看见/能用于状态判断）
                    mark_progress()

                    kind = event.get("kind")
                    if kind == "action":
                        emit_action_event(event.get("phase"))
                    elif kind == "thinking":
                        # 模型显式发 thinking：立即进入 thinking（不等 0.5s）
                        ensure_thinking_running()
                    # 其它 event：不立刻进入 thinking，交给 watchdog（看 idle）
                    continue

                # text_delta 也可能是 dict(kind=...)
                if isinstance(text_delta, dict):
                    mark_progress()
                    kind = text_delta.get("kind")
                    if kind == "action":
                        emit_action_event(text_delta.get("phase"))
                    elif kind == "thinking":
                        ensure_thinking_running()
                    continue

                # ---------- 3) tools_call（算信息增加） ----------
                if tools_call is not None and len(tools_call) > 0:
                    mark_progress()
                    tool_call_flag = True
                    self._merge_tool_calls(tool_calls_list, tools_call)

                # ---------- 4) 处理文本增量 ----------
                if text_delta is None:
                    # 没信息增加 -> 不 mark_progress，watchdog 会在 idle 后触发 thinking
                    continue
                if isinstance(text_delta, str) and text_delta == "":
                    # 空串不算信息增加 -> watchdog 触发 thinking
                    continue

                content = str(text_delta)
                if content:
                    mark_progress()

                # 情绪表情：一轮只做一次
                if emotion_flag and content.strip():
                    asyncio.run_coroutine_threadsafe(
                        textUtils.get_emotion(self, content),
                        self.loop,
                    )
                    emotion_flag = False

                # 累积用于 tool_call 判断与后续解析
                content_arguments += content

                if not tool_call_flag and is_toolcall_payload(content_arguments):
                    tool_call_flag = True

                if tool_call_flag:
                    # tool_call 阶段不算“真实 content”，由 watchdog 负责 idle->thinking
                    continue

                # 真正用户可见 content：开始输出后结束 thinking
                if content.strip():
                    content_started.set()
                    finish_thinking_once()

                response_message.append(content)
                if stream_tts_from_llm:
                    self.tts.tts_text_queue.put(
                        TTSMessageDTO(
                            sentence_id=self.sentence_id,
                            sentence_type=SentenceType.MIDDLE,
                            content_type=ContentType.TEXT,
                            content_detail=content,
                        )
                    )

        finally:
            watchdog_stop.set()

        # 流结束：如果曾经进入 thinking 但没结束，收尾
        if thinking_event_sent and not thinking_event_done:
            self._finish_thinking_event_if_active()
            thinking_event_done = True


        if tool_call_flag:
            bHasError = False
            # 处理基于文本的工具调用格式
            if len(tool_calls_list) == 0 and content_arguments:
                a = extract_json_from_string(content_arguments)
                if a is not None:
                    try:
                        content_arguments_json = json.loads(a)
                        tool_calls_list.append(
                            {
                                "id": str(uuid.uuid4().hex),
                                "name": content_arguments_json["name"],
                                "arguments": json.dumps(
                                    content_arguments_json["arguments"],
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    except Exception as e:
                        bHasError = True
                        response_message.append(a)
                else:
                    bHasError = True
                    response_message.append(content_arguments)
                if bHasError:
                    self.logger.bind(tag=TAG).error(
                        f"function call error: {content_arguments}"
                    )

            if not bHasError and len(tool_calls_list) > 0:
                # 如需要大模型先处理一轮，添加相关处理后的日志情况
                if len(response_message) > 0:
                    text_buff = textUtils.normalize_spoken_text(
                        "".join(response_message)
                    )
                    self.tts_MessageText = text_buff
                    if text_buff:
                        self.dialogue.put(Message(role="assistant", content=text_buff))
                response_message.clear()

                self.logger.bind(tag=TAG).debug(
                    f"检测到 {len(tool_calls_list)} 个工具调用"
                )

                # 收集所有工具调用的 Future
                futures_with_data = []
                for tool_call_data in tool_calls_list:
                    self.logger.bind(tag=TAG).debug(
                        f"function_name={tool_call_data['name']}, function_id={tool_call_data['id']}, function_arguments={tool_call_data['arguments']}"
                    )

                    future = asyncio.run_coroutine_threadsafe(
                        self.func_handler.handle_llm_function_call(
                            self, tool_call_data
                        ),
                        self.loop,
                    )
                    futures_with_data.append((future, tool_call_data))

                # 等待协程结束（实际等待时长为最慢的那个）
                tool_results = []
                for future, tool_call_data in futures_with_data:
                    result = future.result()
                    tool_results.append((result, tool_call_data))

                # 统一处理所有工具调用结果
                if tool_results:
                    self._handle_function_result(tool_results, depth=depth)

        # 存储对话内容
        if len(response_message) > 0:
            text_buff = textUtils.prepare_runtime_spoken_text_for_conn(
                self,
                "".join(response_message)
            )
            self.tts_MessageText = text_buff
            if text_buff:
                self.dialogue.put(Message(role="assistant", content=text_buff))
                if not stream_tts_from_llm:
                    self.tts.tts_one_sentence(
                        self, ContentType.TEXT, content_detail=text_buff
                    )
        if depth == 0:
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )
            self.llm_finish_task = True
            # 使用lambda延迟计算，只有在DEBUG级别时才执行get_llm_dialogue()
            self.logger.bind(tag=TAG).debug(
                lambda: json.dumps(
                    self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False
                )
            )
        
        return True

    def _handle_function_result(self, tool_results, depth):
        need_llm_tools = []

        for result, tool_call_data in tool_results:
            if result.action in [
                Action.RESPONSE,
                Action.NOTFOUND,
                Action.ERROR,
            ]:  # 直接回复前端
                text = result.response if result.response else result.result
                text = textUtils.prepare_runtime_spoken_text_for_conn(self, text)
                if text:
                    self.tts.tts_one_sentence(
                        self, ContentType.TEXT, content_detail=text
                    )
                    self.dialogue.put(Message(role="assistant", content=text))
            elif result.action == Action.REQLLM:
                # 收集需要 LLM 处理的工具
                need_llm_tools.append((result, tool_call_data))
            else:
                pass

        if need_llm_tools:
            all_tool_calls = [
                {
                    "id": tool_call_data["id"],
                    "function": {
                        "arguments": (
                            "{}"
                            if tool_call_data["arguments"] == ""
                            else tool_call_data["arguments"]
                        ),
                        "name": tool_call_data["name"],
                    },
                    "type": "function",
                    "index": idx,
                }
                for idx, (_, tool_call_data) in enumerate(need_llm_tools)
            ]
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            for result, tool_call_data in need_llm_tools:
                text = result.result
                if text is not None and len(text) > 0:
                    self.dialogue.put(
                        Message(
                            role="tool",
                            tool_call_id=(
                                str(uuid.uuid4())
                                if tool_call_data["id"] is None
                                else tool_call_data["id"]
                            ),
                            content=text,
                        )
                    )

            self.chat(None, depth=depth + 1)

    def _report_worker(self):
        """聊天记录上报工作线程"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取数据，设置超时以便定期检查停止事件
                item = self.report_queue.get(timeout=1)
                if item is None:  # 检测毒丸对象
                    break
                try:
                    # 检查线程池状态
                    if self.executor is None:
                        continue
                    # 提交任务到线程池
                    self.executor.submit(self._process_report, *item)
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"聊天记录上报线程异常: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"聊天记录上报工作线程异常: {e}")

        self.logger.bind(tag=TAG).info("聊天记录上报线程已退出")

    def _process_report(self, type, text, audio_data, report_time):
        """处理上报任务"""
        try:
            # 执行异步上报（在事件循环中运行）
            asyncio.run(report(self, type, text, audio_data, report_time))
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"上报处理异常: {e}")
        finally:
            # 标记任务完成
            self.report_queue.task_done()

    def clearSpeakStatus(self):
        self.client_is_speaking = False
        self.logger.bind(tag=TAG).debug(f"清除服务端讲话状态")

    async def close(self, ws=None):
        """资源清理方法"""
        try:
            self.request_final_close("close() called")
            target_ws = ws if ws else self.websocket
            self.logger.bind(tag=TAG).info(
                "[debug-close] close() called: "
                f"session_id={self.session_id}, device_id={self.device_id}, "
                f"close_after_chat={self.close_after_chat}, "
                f"stop_event={self.stop_event.is_set() if self.stop_event else None}, "
                f"target_ws_state={self._format_ws_state(target_ws)}, "
                f"self_ws_state={self._format_ws_state(self.websocket)}, "
                f"call_trace={self._close_call_trace()}"
            )

            # 清理音频缓冲区
            if hasattr(self, "audio_buffer"):
                self.audio_buffer.clear()

            # Clear frontend-related per-connection state (avoid leaking state on long-lived server).
            try:
                if getattr(self, "audio_frontend", None):
                    self.audio_frontend.reset()
            except Exception:
                pass
            self.audio_frontend = None

            if hasattr(self, "_frontend_opus_decoder"):
                try:
                    self._frontend_opus_decoder = None
                    delattr(self, "_frontend_opus_decoder")
                except Exception:
                    self._frontend_opus_decoder = None

            if hasattr(self, "_pcm_packet_for_asr"):
                self._pcm_packet_for_asr = None

            if hasattr(self, "asr_pcm_audio"):
                try:
                    self.asr_pcm_audio.clear()
                except Exception:
                    self.asr_pcm_audio = []

            # 取消超时任务
            if self.timeout_task and not self.timeout_task.done():
                self.timeout_task.cancel()
                try:
                    await self.timeout_task
                except asyncio.CancelledError:
                    pass
                self.timeout_task = None

            # 清理工具处理器资源
            if hasattr(self, "func_handler") and self.func_handler:
                try:
                    await self.func_handler.cleanup()
                except Exception as cleanup_error:
                    self.logger.bind(tag=TAG).error(
                        f"清理工具处理器时出错: {cleanup_error}"
                    )

            # 只清理当前连接自己创建的动态声纹，避免误删其他设备或仍在使用中的会话。
            if getattr(self, "voiceprint_provider", None):
                try:
                    await self.voiceprint_provider.cleanup_dynamic_voiceprint(
                        self.session_id
                    )
                except Exception as voiceprint_cleanup_error:
                    self.logger.bind(tag=TAG).error(
                        f"清理当前连接声纹时出错: {voiceprint_cleanup_error}"
                    )

            # 触发停止事件
            if self.stop_event:
                self.stop_event.set()
            prewarm_task = getattr(self, "experiment_prewarm_task", None)
            if prewarm_task is not None and not prewarm_task.done():
                prewarm_task.cancel()
            deep_prefetch_task = getattr(self, "experiment_deep_prefetch_task", None)
            if deep_prefetch_task is not None and not deep_prefetch_task.done():
                deep_prefetch_task.cancel()
            with self._external_busy_lock:
                self._external_busy_tokens.clear()
            self._stop_thinking_pulse()
            self._stop_action_pulse()

            # 清空任务队列
            self.clear_queues()

            # 关闭WebSocket连接
            try:
                if ws:
                    self.logger.bind(tag=TAG).info(
                        "[debug-close] close() using param ws: "
                        f"{self._format_ws_state(ws)}"
                    )
                    # 安全地检查WebSocket状态并关闭
                    try:
                        if hasattr(ws, "closed") and not ws.closed:
                            await ws.close()
                        elif hasattr(ws, "state") and ws.state.name != "CLOSED":
                            await ws.close()
                        else:
                            # 如果没有closed属性，直接尝试关闭
                            await ws.close()
                    except Exception:
                        # 如果关闭失败，忽略错误
                        pass
                elif self.websocket:
                    self.logger.bind(tag=TAG).info(
                        "[debug-close] close() using self.websocket: "
                        f"{self._format_ws_state(self.websocket)}"
                    )
                    try:
                        if (
                            hasattr(self.websocket, "closed")
                            and not self.websocket.closed
                        ):
                            await self.websocket.close()
                        elif (
                            hasattr(self.websocket, "state")
                            and self.websocket.state.name != "CLOSED"
                        ):
                            await self.websocket.close()
                        else:
                            # 如果没有closed属性，直接尝试关闭
                            await self.websocket.close()
                    except Exception:
                        # 如果关闭失败，忽略错误
                        pass
            except Exception as ws_error:
                self.logger.bind(tag=TAG).error(f"关闭WebSocket连接时出错: {ws_error}")
            finally:
                if target_ws is None or self.websocket is target_ws:
                    self.websocket = None
                self.logger.bind(tag=TAG).info(
                    "[debug-close] close() websocket close phase done: "
                    f"param_ws_state={self._format_ws_state(ws)}, "
                    f"self_ws_state={self._format_ws_state(self.websocket)}"
                )

            if self.tts:
                await self.tts.close()

            # 最后关闭线程池（避免阻塞）
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(
                        f"关闭线程池时出错: {executor_error}"
                    )
                self.executor = None
            self._allow_transport_reconnect = False
            self._transport_detached_event.set()
            self._closed = True
            self.logger.bind(tag=TAG).info("连接资源已释放")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"关闭连接时出错: {e}")
        finally:
            # 确保停止事件被设置
            if self.stop_event:
                self.stop_event.set()
            self._stop_thinking_pulse()
            self._stop_action_pulse()

    def clear_queues(self):
        """清空所有任务队列"""
        if self.tts:
            self.logger.bind(tag=TAG).debug(
                f"开始清理: TTS队列大小={self.tts.tts_text_queue.qsize()}, 音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )

            # 使用非阻塞方式清空队列
            for q in [
                self.tts.tts_text_queue,
                self.tts.tts_audio_queue,
                self.report_queue,
            ]:
                if not q:
                    continue
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            # 重置音频流控器（取消后台任务并清空队列）
            if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
                self.audio_rate_controller.reset()
                self.logger.bind(tag=TAG).debug("已重置音频流控器")

            self.logger.bind(tag=TAG).debug(
                f"清理结束: TTS队列大小={self.tts.tts_text_queue.qsize()}, 音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )

    def reset_vad_states(self):
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_stop = False
        self._asr_voice_stop_deadline_ms = 0.0
        self.logger.bind(tag=TAG).debug("VAD states reset.")

    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            # Use the existing chat method
            self.chat(text)

            # After chat is complete, close the connection
            self.close_after_chat = True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")

    async def _check_timeout(self):
        """检查连接超时"""
        try:
            while not self.stop_event.is_set():
                last_activity_time = self.last_activity_time
                if self.need_bind:
                    last_activity_time = self.first_activity_time

                # 检查是否超时（只有在时间戳已初始化的情况下）
                if last_activity_time > 0.0:
                    current_time = time.time() * 1000
                    elapsed_ms = current_time - last_activity_time
                    timeout_ms = self.timeout_seconds * 1000
                    if elapsed_ms > timeout_ms:
                        if not self.stop_event.is_set():
                            self.logger.bind(tag=TAG).info(
                                "连接超时，准备关闭: "
                                f"elapsed_ms={int(elapsed_ms)}, "
                                f"timeout_ms={int(timeout_ms)}, "
                                f"session_id={self.session_id}, "
                                f"device_id={self.device_id}, "
                                f"ws_state={self._format_ws_state(self.websocket)}"
                            )
                            # 设置停止事件，防止重复处理
                            self.stop_event.set()
                            # 使用 try-except 包装关闭操作，确保不会因为异常而阻塞
                            try:
                                await self.close(self.websocket)
                            except Exception as close_error:
                                self.logger.bind(tag=TAG).error(
                                    f"超时关闭连接时出错: {close_error}"
                                )
                        break
                # 每10秒检查一次，避免过于频繁
                await asyncio.sleep(10)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"超时检查任务出错: {e}")
        finally:
            self.logger.bind(tag=TAG).info(
                "超时检查任务已退出: "
                f"session_id={self.session_id}, "
                f"device_id={self.device_id}, "
                f"stop_event={self.stop_event.is_set() if self.stop_event else None}, "
                f"ws_state={self._format_ws_state(self.websocket)}"
            )

    def _merge_tool_calls(self, tool_calls_list, tools_call):
        """合并工具调用列表

        Args:
            tool_calls_list: 已收集的工具调用列表
            tools_call: 新的工具调用
        """
        for tool_call in tools_call:
            tool_index = getattr(tool_call, "index", None)
            if tool_index is None:
                if tool_call.function.name:
                    # 有 function_name，说明是新的工具调用
                    tool_index = len(tool_calls_list)
                else:
                    tool_index = len(tool_calls_list) - 1 if tool_calls_list else 0

            # 确保列表有足够的位置
            if tool_index >= len(tool_calls_list):
                tool_calls_list.append({"id": "", "name": "", "arguments": ""})

            # 更新工具调用信息
            if tool_call.id:
                tool_calls_list[tool_index]["id"] = tool_call.id
            if tool_call.function.name:
                tool_calls_list[tool_index]["name"] = tool_call.function.name
            if tool_call.function.arguments:
                tool_calls_list[tool_index]["arguments"] += tool_call.function.arguments
