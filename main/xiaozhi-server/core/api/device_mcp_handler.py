import json
import mimetypes
import os
import subprocess
import uuid
import asyncio
import time
from aiohttp import web

from core.api.base_handler import BaseHandler
from core.handle.sendAudioHandle import sendAudio, send_tts_message
from core.providers.tools.device_mcp import call_mcp_tool, send_mcp_initialize_message
from core.utils.util import (
    audio_to_data,
    sanitize_tool_name,
    get_local_ip,
    is_valid_image_file,
)

TAG = __name__


class DeviceMCPHandler(BaseHandler):
    def __init__(self, config: dict, ws_server):
        super().__init__(config)
        self.ws_server = ws_server
        data_dir = self.config.get("log", {}).get("data_dir", "data")
        self.preview_dir = os.path.join(data_dir, "preview_files")
        os.makedirs(self.preview_dir, exist_ok=True)
        self._mcp_refresh_lock = asyncio.Lock()

    def _project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _coerce_bool(self, value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in ("1", "true", "yes", "on", "y"):
            return True
        if normalized in ("0", "false", "no", "off", "n"):
            return False
        return default

    def _coerce_nonnegative_int(self, value, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(default))

    def _resolve_photo_capture_notify_config(self) -> dict:
        raw = self.config.get("photo_capture_notify", {})
        if not isinstance(raw, dict):
            raw = {}
        audio_file = str(raw.get("audio_file", "config/assets/tts_notify.mp3")).strip()
        if audio_file and not os.path.isabs(audio_file):
            audio_file = os.path.abspath(os.path.join(self._project_root(), audio_file))
        return {
            "enabled": self._coerce_bool(raw.get("enabled", False), False),
            "audio_file": audio_file,
            "lead_time_ms": self._coerce_nonnegative_int(raw.get("lead_time_ms", 700), 700),
            "send_tts_state": self._coerce_bool(raw.get("send_tts_state", True), True),
        }

    async def _play_photo_capture_notify(self, conn) -> None:
        notify_cfg = self._resolve_photo_capture_notify_config()
        if not notify_cfg.get("enabled"):
            return

        audio_file = str(notify_cfg.get("audio_file", "") or "").strip()
        if not audio_file:
            return

        try:
            audios = await audio_to_data(audio_file, is_opus=True)
            send_tts_state = bool(notify_cfg.get("send_tts_state", True))
            if send_tts_state:
                await send_tts_message(conn, "start", None)
                conn.client_is_speaking = True
            await sendAudio(conn, audios)
            if send_tts_state:
                old_stop_notify = conn.config.get("enable_stop_tts_notify", False)
                conn.config["enable_stop_tts_notify"] = False
                try:
                    await send_tts_message(conn, "stop", None)
                finally:
                    conn.config["enable_stop_tts_notify"] = old_stop_notify
                    conn.client_is_speaking = False
            lead_time_ms = int(notify_cfg.get("lead_time_ms", 0) or 0)
            if lead_time_ms > 0:
                await asyncio.sleep(lead_time_ms / 1000.0)
        except Exception as exc:
            self.logger.bind(tag=TAG).warning(
                f"photo capture notify failed, continue taking photo: {exc}"
            )

    def _sanitize_device_for_path(self, device_id: str) -> str:
        value = str(device_id or "").strip()
        if not value:
            return "unknown"
        chars = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_", "."):
                chars.append(ch)
                continue
            if ch == ":":
                chars.append("_")
                continue
            chars.append("_")
        safe = "".join(chars).strip("._-")
        return safe or "unknown"

    def _normalize_group_number(self, value):
        if isinstance(value, bool) or value in (None, ""):
            return None
        try:
            group_number = int(value)
        except (TypeError, ValueError):
            return None
        return group_number if group_number >= 1 else None

    def _format_group_dir_name(self, group_number: int) -> str:
        return f"group_{int(group_number):02d}"

    def _derive_experiment_data_root(self) -> str:
        cfg = self.config or {}
        llm_map = cfg.get("LLM") or {}
        selected = str((cfg.get("selected_module") or {}).get("LLM", "")).strip()
        candidates = []

        if isinstance(llm_map, dict):
            if selected:
                selected_cfg = llm_map.get(selected) or {}
                if isinstance(selected_cfg, dict):
                    candidates.append(
                        (
                            str(selected_cfg.get("workspace", "")).strip(),
                            str(selected_cfg.get("yaml_path", "")).strip(),
                        )
                    )
            for llm_cfg in llm_map.values():
                if not isinstance(llm_cfg, dict):
                    continue
                candidates.append(
                    (
                        str(llm_cfg.get("workspace", "")).strip(),
                        str(llm_cfg.get("yaml_path", "")).strip(),
                    )
                )

        prompt_template = str(cfg.get("prompt_template", "")).strip()
        if prompt_template:
            candidates.append(("", prompt_template))

        server_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )

        for workspace, path_text in candidates:
            if not path_text:
                continue
            if os.path.isabs(path_text):
                abs_path = os.path.abspath(path_text)
            else:
                base = workspace if workspace else server_root
                abs_path = os.path.abspath(os.path.join(base, path_text))
            cfg_dir = os.path.dirname(abs_path)
            if os.path.basename(cfg_dir).lower() == "configs":
                exp_root = os.path.dirname(cfg_dir)
            else:
                exp_root = cfg_dir
            if exp_root:
                return os.path.abspath(os.path.join(exp_root, "data"))
        return ""

    def _resolve_take_photo_recovery_timeout_seconds(self) -> float:
        shortcut_cfg = self.config.get("device_mcp_shortcuts", {}) or {}
        raw_value = shortcut_cfg.get("server_photo_recovery_window_seconds", 8.0)
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 8.0

    def _resolve_take_photo_timeout_seconds(self, provided_timeout=None) -> int:
        shortcut_cfg = self.config.get("device_mcp_shortcuts", {}) or {}

        def _coerce_positive_int(value, fallback: int) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = int(fallback)
            return parsed if parsed > 0 else int(fallback)

        configured_default = _coerce_positive_int(
            shortcut_cfg.get("server_photo_timeout", shortcut_cfg.get("photo_timeout", 20)),
            20,
        )
        return _coerce_positive_int(provided_timeout, configured_default)

    def _list_saved_device_photo_items(self, device_id: str, group_number=None) -> list[dict]:
        data_root = self._derive_experiment_data_root()
        if not data_root:
            return []

        safe_device = self._sanitize_device_for_path(device_id)
        device_dir = os.path.join(data_root, safe_device)
        normalized_group = self._normalize_group_number(group_number)
        if normalized_group is not None:
            device_dir = os.path.join(device_dir, self._format_group_dir_name(normalized_group))
        if not os.path.isdir(device_dir):
            return []

        allowed_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }
        items = []
        try:
            with os.scandir(device_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in allowed_exts:
                        continue
                    try:
                        stat = entry.stat()
                    except FileNotFoundError:
                        continue
                    items.append(
                        {
                            "device_id": str(device_id or "").strip(),
                            "local_path": os.path.abspath(entry.path),
                            "file_name": entry.name,
                            "mtime": float(stat.st_mtime),
                            "size": int(stat.st_size),
                        }
                    )
        except FileNotFoundError:
            return []
        return items

    def _find_latest_saved_photo(self, device_id: str, group_number=None) -> dict:
        items = self._list_saved_device_photo_items(device_id, group_number=group_number)
        if not items:
            return {}
        latest = max(items, key=lambda item: float(item.get("mtime", 0.0)))
        latest["found"] = True
        return latest

    def _saved_photo_matches_request(
        self,
        file_name: str,
        requested_photo_name: str,
    ) -> bool:
        target = str(requested_photo_name or "").strip()
        candidate = str(file_name or "").strip()
        if not target or not candidate:
            return False
        return os.path.splitext(candidate)[0] == target

    def _is_new_saved_photo(
        self,
        latest_photo: dict,
        baseline_photo: dict,
        capture_started_at: float,
        requested_photo_name: str = "",
    ) -> bool:
        if not isinstance(latest_photo, dict) or not latest_photo.get("found"):
            return False

        latest_path = str(latest_photo.get("local_path", "") or "").strip()
        latest_file_name = str(latest_photo.get("file_name", "") or "").strip()
        baseline_path = str((baseline_photo or {}).get("local_path", "") or "").strip()

        try:
            latest_mtime = float(latest_photo.get("mtime", 0.0) or 0.0)
        except (TypeError, ValueError):
            latest_mtime = 0.0
        try:
            baseline_mtime = float((baseline_photo or {}).get("mtime", 0.0) or 0.0)
        except (TypeError, ValueError):
            baseline_mtime = 0.0

        if self._saved_photo_matches_request(latest_file_name, requested_photo_name):
            if not baseline_path:
                return latest_mtime >= capture_started_at - 2.0 if latest_mtime > 0 else True
            if latest_path != baseline_path:
                return True
            return latest_mtime > baseline_mtime + 1e-6

        if latest_path and baseline_path:
            if latest_path != baseline_path:
                return True
            return latest_mtime > baseline_mtime + 1e-6
        if latest_path and not baseline_path:
            return latest_mtime >= capture_started_at - 2.0 if latest_mtime > 0 else False
        return latest_mtime > baseline_mtime + 1e-6

    async def _recover_saved_photo_after_timeout(
        self,
        *,
        device_id: str,
        baseline_photo: dict,
        requested_photo_name: str,
        capture_started_at: float,
        max_wait_seconds: float,
        group_number=None,
    ) -> dict | None:
        safe_wait = max(0.0, float(max_wait_seconds or 0.0))
        deadline = time.time() + safe_wait
        while True:
            latest_photo = self._find_latest_saved_photo(device_id, group_number=group_number)
            if self._is_new_saved_photo(
                latest_photo,
                baseline_photo,
                capture_started_at,
                requested_photo_name=requested_photo_name,
            ):
                return latest_photo
            if time.time() >= deadline:
                break
            await asyncio.sleep(min(0.5, max(0.0, deadline - time.time())))
        return None

    def _json_response(self, body: dict, status: int = 200):
        return web.Response(
            text=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            content_type="application/json",
            status=status,
        )

    def _resolve_target_params(self, body: dict):
        session_id = str(body.get("session_id", "")).strip()
        device_id = str(body.get("device_id", "")).strip()
        return session_id, device_id

    @staticmethod
    def _is_websocket_alive(conn) -> bool:
        ws = getattr(conn, "websocket", None)
        if ws is None:
            return False
        if hasattr(ws, "state"):
            try:
                return ws.state.name != "CLOSED"
            except Exception:
                return False
        if hasattr(ws, "closed"):
            try:
                return not ws.closed
            except Exception:
                return False
        return True

    async def _resolve_target_conn(self, session_id: str, device_id: str):
        if not session_id and not device_id:
            return None, None, self._json_response(
                {"success": False, "message": "session_id or device_id is required"},
                status=400,
            )

        conn = await self.ws_server.get_connection(
            session_id=session_id if session_id else None,
            device_id=device_id if device_id else None,
        )
        if not conn:
            return None, None, self._json_response(
                {"success": False, "message": "connection not found"},
                status=404,
            )

        if not self._is_websocket_alive(conn):
            return None, None, self._json_response(
                {
                    "success": False,
                    "message": "device websocket is offline and waiting for reconnect",
                },
                status=409,
            )

        mcp_client = getattr(conn, "mcp_client", None)
        if not mcp_client:
            return None, None, self._json_response(
                {"success": False, "message": "mcp client is not initialized"},
                status=409,
            )

        if not await mcp_client.is_ready():
            return None, None, self._json_response(
                {"success": False, "message": "mcp client is not ready"},
                status=409,
            )

        return conn, mcp_client, None

    def _read_json_body(self, body: dict):
        if not isinstance(body, dict):
            raise ValueError("request body must be json object")
        return body

    def _save_preview_file(self, source_file_path: str):
        source_file_path = os.path.abspath(source_file_path)
        if not os.path.exists(source_file_path):
            raise ValueError(f"file not found: {source_file_path}")
        if not os.path.isfile(source_file_path):
            raise ValueError(f"not a file: {source_file_path}")

        with open(source_file_path, "rb") as f:
            image_data = f.read()
        if not image_data:
            raise ValueError("file is empty")
        if not is_valid_image_file(image_data):
            raise ValueError("file is not a supported image")

        png_signature = b"\x89PNG\r\n\x1a\n"
        file_name = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join(self.preview_dir, file_name)

        if image_data.startswith(png_signature):
            with open(save_path, "wb") as f:
                f.write(image_data)
            return file_name, save_path, len(image_data)

        # Firmware image decoder supports PNG in this project config.
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    source_file_path,
                    "-frames:v",
                    "1",
                    save_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            raise ValueError(f"failed to convert image to png: {err}") from e
        except FileNotFoundError as e:
            raise ValueError(
                "failed to convert image to png: ffmpeg is not installed or not in PATH"
            ) from e

        if not os.path.exists(save_path):
            raise ValueError("failed to convert image to png: output not generated")
        with open(save_path, "rb") as f:
            png_data = f.read()
        if not png_data.startswith(png_signature):
            raise ValueError("failed to convert image to png: invalid output format")

        return file_name, save_path, len(png_data)

    def _build_preview_url(self, file_name: str):
        port = int(self.config.get("server", {}).get("http_port", 8003))
        return f"http://{get_local_ip()}:{port}/mcp/device/local_files/{file_name}"

    def _build_preview_tool_candidates(self, tool_name_raw: str):
        candidates = []
        raw = (tool_name_raw or "").strip()
        if raw:
            candidates.append(raw)

        sanitized = sanitize_tool_name(raw) if raw else ""
        if sanitized and sanitized not in candidates:
            candidates.append(sanitized)

        dotted = raw.replace("_", ".") if raw else ""
        if dotted and dotted not in candidates:
            candidates.append(dotted)

        underscored = raw.replace(".", "_") if raw else ""
        if underscored and underscored not in candidates:
            candidates.append(underscored)

        for fixed_name in [
            "self.screen.preview_image",
            "self_screen_preview_image",
            "self.screen.preview_screen_shot",
            "self_screen_preview_screen_shot",
            "preview_screen_shot",
            "self.screen.preview_screenshot",
            "self_screen_preview_screenshot",
            "preview_screenshot",
        ]:
            if fixed_name not in candidates:
                candidates.append(fixed_name)

        return candidates

    @staticmethod
    def _is_photo_upload_failed_error(exc: Exception) -> bool:
        msg = str(exc or "")
        return "Failed to upload photo" in msg or "上传照片失败" in msg

    async def _refresh_device_mcp(self, conn, mcp_client, timeout_sec: float = 8.0) -> bool:
        async with self._mcp_refresh_lock:
            previous_ready = await mcp_client.is_ready()
            try:
                await mcp_client.set_ready(False)
                await send_mcp_initialize_message(conn)

                loop = asyncio.get_running_loop()
                deadline = loop.time() + max(float(timeout_sec), 1.0)
                while loop.time() < deadline:
                    if await mcp_client.is_ready():
                        return True
                    await asyncio.sleep(0.2)
            except Exception as e:
                self.logger.bind(tag=TAG).warning(f"refresh mcp session failed: {e}")
            finally:
                # 避免刷新失败后将连接永久卡在 not ready
                if previous_ready and not await mcp_client.is_ready():
                    await mcp_client.set_ready(True)
        return False

    async def handle_get(self, request):
        response = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
            else:
                connections = await self.ws_server.list_connections()
                response = self._json_response(
                    {"success": True, "connections": connections}
                )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"list sessions failed: {e}")
            response = self._json_response(
                {"success": False, "message": "internal server error"},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_post(self, request):
        response = None
        capture_device_id = ""
        photo_name = ""
        baseline_photo = {}
        capture_started_at = 0.0
        group_number = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
                return response

            try:
                body = await request.json()
            except Exception:
                response = self._json_response(
                    {"success": False, "message": "request body must be json"},
                    status=400,
                )
                return response

            self._read_json_body(body)
            session_id, device_id = self._resolve_target_params(body)
            question = str(body.get("question", "Please take a photo.")).strip()
            photo_name = str(body.get("photo_name", "")).strip()
            group_number = self._normalize_group_number(body.get("group_number"))
            tool_name_raw = str(
                body.get("tool_name", "self.camera.take_photo")
            ).strip()
            tool_name = sanitize_tool_name(tool_name_raw)
            timeout = self._resolve_take_photo_timeout_seconds(body.get("timeout"))

            conn, mcp_client, error_resp = await self._resolve_target_conn(
                session_id, device_id
            )
            if error_resp:
                response = error_resp
                return response

            capture_device_id = str(conn.device_id or device_id or "").strip()
            baseline_photo = self._find_latest_saved_photo(
                capture_device_id,
                group_number=group_number,
            )
            capture_started_at = time.time()

            # The device-side camera tool only declares `question`. Keep any
            # save/mirroring metadata on the server side instead of passing
            # undeclared arguments through the MCP tool call.
            if "[XIAOZHI_META]" not in question and (photo_name or group_number is not None):
                meta = {}
                if photo_name:
                    meta["photo_name"] = photo_name
                if group_number is not None:
                    meta["group_number"] = group_number
                    meta["group_dir_name"] = self._format_group_dir_name(group_number)
                question = (
                    f"{question}\n[XIAOZHI_META]"
                    f"{json.dumps(meta, ensure_ascii=False, separators=(',', ':'))}"
                )
            tool_args = {"question": question}

            try:
                await self._play_photo_capture_notify(conn)
                result = await call_mcp_tool(
                    conn,
                    mcp_client,
                    tool_name,
                    tool_args,
                    timeout=timeout,
                )
            except Exception as first_err:
                if self._is_photo_upload_failed_error(first_err):
                    self.logger.bind(tag=TAG).warning(
                        "take_photo upload failed, try refresh MCP session once"
                    )
                    refreshed = await self._refresh_device_mcp(conn, mcp_client)
                    if refreshed:
                        result = await call_mcp_tool(
                            conn,
                            mcp_client,
                            tool_name,
                            tool_args,
                            timeout=timeout,
                        )
                    else:
                        raise first_err
                else:
                    raise

            saved_photo = self._find_latest_saved_photo(
                capture_device_id,
                group_number=group_number,
            )
            response = self._json_response(
                {
                    "success": True,
                    "tool": tool_name_raw,
                    "tool_sanitized": tool_name,
                    "session_id": conn.session_id,
                    "device_id": conn.device_id,
                    "group_number": group_number,
                    "requested_photo_name": photo_name,
                    "result": result,
                    "saved_photo_path": str(
                        saved_photo.get("local_path", "") or ""
                    ).strip(),
                    "photo_meta": saved_photo,
                }
            )
        except ValueError as e:
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=400,
            )
        except TimeoutError:
            recovered_photo = await self._recover_saved_photo_after_timeout(
                device_id=capture_device_id,
                baseline_photo=baseline_photo,
                requested_photo_name=photo_name,
                capture_started_at=capture_started_at,
                max_wait_seconds=self._resolve_take_photo_recovery_timeout_seconds(),
                group_number=group_number,
            )
            if recovered_photo:
                self.logger.bind(tag=TAG).info(
                    "take_photo recovered from saved artifact after timeout: "
                    f"device_id={capture_device_id}, file_name={recovered_photo.get('file_name', '')}"
                )
                response = self._json_response(
                    {
                        "success": True,
                        "message": "photo recovered after timeout",
                        "device_id": capture_device_id,
                        "group_number": group_number,
                        "requested_photo_name": photo_name,
                        "saved_photo_path": str(
                            recovered_photo.get("local_path", "") or ""
                        ).strip(),
                        "photo_meta": recovered_photo,
                        "recovered_after_timeout": True,
                    }
                )
            else:
                response = self._json_response(
                    {"success": False, "message": "tool call timeout"},
                    status=504,
                )
        except Exception as e:
            recovered_photo = await self._recover_saved_photo_after_timeout(
                device_id=capture_device_id,
                baseline_photo=baseline_photo,
                requested_photo_name=photo_name,
                capture_started_at=capture_started_at,
                max_wait_seconds=self._resolve_take_photo_recovery_timeout_seconds(),
                group_number=group_number,
            )
            if recovered_photo:
                self.logger.bind(tag=TAG).warning(
                    "take_photo recovered from saved artifact after error: "
                    f"device_id={capture_device_id}, file_name={recovered_photo.get('file_name', '')}, error={e}"
                )
                response = self._json_response(
                    {
                        "success": True,
                        "message": "photo recovered after tool error",
                        "device_id": capture_device_id,
                        "group_number": group_number,
                        "requested_photo_name": photo_name,
                        "saved_photo_path": str(
                            recovered_photo.get("local_path", "") or ""
                        ).strip(),
                        "photo_meta": recovered_photo,
                        "recovered_after_error": True,
                    }
                )
            else:
                self.logger.bind(tag=TAG).error(f"take_photo failed: {e}")
                response = self._json_response(
                    {"success": False, "message": str(e)},
                    status=500,
                )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_preview_local_file_post(self, request):
        response = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
                return response

            try:
                body = await request.json()
            except Exception:
                response = self._json_response(
                    {"success": False, "message": "request body must be json"},
                    status=400,
                )
                return response

            self._read_json_body(body)
            session_id, device_id = self._resolve_target_params(body)
            file_path = str(body.get("file_path", "")).strip()
            if not file_path:
                response = self._json_response(
                    {"success": False, "message": "file_path is required"},
                    status=400,
                )
                return response

            timeout = int(body.get("timeout", 90))
            if timeout <= 0:
                timeout = 90

            tool_name_raw = str(
                body.get("tool_name", "self.screen.preview_image")
            ).strip()
            tool_name = sanitize_tool_name(tool_name_raw)

            conn, mcp_client, error_resp = await self._resolve_target_conn(
                session_id, device_id
            )
            if error_resp:
                response = error_resp
                return response

            file_name, save_path, size = self._save_preview_file(file_path)
            file_url = self._build_preview_url(file_name)
            self.logger.bind(tag=TAG).info(
                f"prepared preview file: src={file_path} saved={save_path} bytes={size}"
            )

            result = None
            used_tool = ""
            tried_tools = self._build_preview_tool_candidates(tool_name_raw)
            last_unknown_error = None
            for candidate_tool in tried_tools:
                try:
                    result = await call_mcp_tool(
                        conn,
                        mcp_client,
                        sanitize_tool_name(candidate_tool),
                        {"url": file_url},
                        timeout=timeout,
                        allow_unlisted=True,
                        raw_tool_name=candidate_tool,
                    )
                    used_tool = candidate_tool
                    break
                except Exception as e:
                    err_msg = str(e)
                    unknown_tool = (
                        "Unknown tool" in err_msg
                        or "工具" in err_msg and "不存在" in err_msg
                        or "tool" in err_msg and "not found" in err_msg
                    )
                    if unknown_tool:
                        last_unknown_error = e
                        self.logger.bind(tag=TAG).warning(
                            f"preview tool not found: {candidate_tool}, trying next"
                        )
                        continue
                    raise e

            if result is None:
                if last_unknown_error:
                    raise ValueError(
                        "no preview tool found on device, tried: "
                        + ", ".join(tried_tools)
                    )
                raise RuntimeError("no preview tool succeeded")

            response = self._json_response(
                {
                    "success": True,
                    "tool": used_tool if used_tool else tool_name_raw,
                    "tool_sanitized": sanitize_tool_name(
                        used_tool if used_tool else tool_name_raw
                    ),
                    "tried_tools": tried_tools,
                    "session_id": conn.session_id,
                    "device_id": conn.device_id,
                    "file_path": file_path,
                    "served_file": file_name,
                    "url": file_url,
                    "result": result,
                }
            )
        except ValueError as e:
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=400,
            )
        except TimeoutError:
            response = self._json_response(
                {"success": False, "message": "tool call timeout"},
                status=504,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"preview_local_file failed: {e}")
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_call_tool_post(self, request):
        response = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
                return response

            try:
                body = await request.json()
            except Exception:
                response = self._json_response(
                    {"success": False, "message": "request body must be json"},
                    status=400,
                )
                return response

            self._read_json_body(body)
            session_id, device_id = self._resolve_target_params(body)
            tool_name_raw = str(body.get("tool_name", "")).strip()
            if not tool_name_raw:
                response = self._json_response(
                    {"success": False, "message": "tool_name is required"},
                    status=400,
                )
                return response

            args = body.get("args", {})
            if args is None:
                args = {}
            if not isinstance(args, (dict, str)):
                response = self._json_response(
                    {"success": False, "message": "args must be object or json string"},
                    status=400,
                )
                return response

            timeout = int(body.get("timeout", 30) or 30)
            if timeout <= 0:
                timeout = 30
            allow_unlisted = bool(body.get("allow_unlisted", False))
            tool_name = sanitize_tool_name(tool_name_raw)

            conn, mcp_client, error_resp = await self._resolve_target_conn(
                session_id, device_id
            )
            if error_resp:
                response = error_resp
                return response

            result = await call_mcp_tool(
                conn,
                mcp_client,
                tool_name,
                args,
                timeout=timeout,
                allow_unlisted=allow_unlisted,
                raw_tool_name=tool_name_raw,
            )
            response = self._json_response(
                {
                    "success": True,
                    "tool": tool_name_raw,
                    "tool_sanitized": tool_name,
                    "session_id": conn.session_id,
                    "device_id": conn.device_id,
                    "result": result,
                }
            )
        except ValueError as e:
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=400,
            )
        except TimeoutError:
            response = self._json_response(
                {"success": False, "message": "tool call timeout"},
                status=504,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"call_tool failed: {e}")
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_local_file_get(self, request):
        response = None
        try:
            file_name = request.match_info.get("file_name", "").strip()
            if not file_name:
                response = self._json_response(
                    {"success": False, "message": "missing file_name"},
                    status=400,
                )
                return response
            if "/" in file_name or "\\" in file_name or ".." in file_name:
                response = self._json_response(
                    {"success": False, "message": "invalid file_name"},
                    status=400,
                )
                return response

            file_path = os.path.abspath(os.path.join(self.preview_dir, file_name))
            preview_dir_abs = os.path.abspath(self.preview_dir)
            if not file_path.startswith(preview_dir_abs + os.sep):
                response = self._json_response(
                    {"success": False, "message": "invalid file path"},
                    status=400,
                )
                return response
            if not os.path.exists(file_path):
                response = self._json_response(
                    {"success": False, "message": "file not found"},
                    status=404,
                )
                return response

            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            response = web.FileResponse(path=file_path)
            response.content_type = mime_type
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"serve local preview file failed: {e}")
            response = self._json_response(
                {"success": False, "message": "internal server error"},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response
