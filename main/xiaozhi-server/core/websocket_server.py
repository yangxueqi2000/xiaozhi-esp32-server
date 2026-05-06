import asyncio
import logging
from typing import Dict, Optional

import websockets

from config.config_loader import get_config_from_api_async
from config.logger import setup_logging
from core.auth import AuthManager, AuthenticationError
from core.connection import ConnectionHandler
from core.providers.tools.server_mcp.mcp_manager import ServerMCPManager
from core.utils.modules_initialize import initialize_modules
from core.utils.util import check_asr_update, check_vad_update


class SuppressInvalidHandshakeFilter(logging.Filter):
    """过滤无效握手噪音日志，例如直接用 HTTP 访问 WebSocket 端口。"""

    def filter(self, record):
        msg = record.getMessage()
        suppress_keywords = [
            "opening handshake failed",
            "did not receive a valid HTTP request",
            "connection closed while reading HTTP request",
            "line without CRLF",
        ]
        return not any(keyword in msg for keyword in suppress_keywords)


def _setup_websockets_logger():
    """给 websockets 相关 logger 挂过滤器，压掉无效握手噪音。"""
    filter_instance = SuppressInvalidHandshakeFilter()
    for logger_name in ["websockets", "websockets.server", "websockets.client"]:
        logger = logging.getLogger(logger_name)
        logger.addFilter(filter_instance)


_setup_websockets_logger()

TAG = __name__


def _coerce_positive_float(value, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if numeric > 0 else float(default)


class WebSocketServer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.config_lock = asyncio.Lock()
        modules = initialize_modules(
            self.logger,
            self.config,
            "VAD" in self.config["selected_module"],
            "ASR" in self.config["selected_module"],
            "LLM" in self.config["selected_module"],
            False,
            "Memory" in self.config["selected_module"],
            "Intent" in self.config["selected_module"],
        )
        self._vad = modules["vad"] if "vad" in modules else None
        self._asr = modules["asr"] if "asr" in modules else None
        self._llm = modules["llm"] if "llm" in modules else None
        self._intent = modules["intent"] if "intent" in modules else None
        self._memory = modules["memory"] if "memory" in modules else None

        auth_config = self.config["server"].get("auth", {})
        self.auth_enable = auth_config.get("enabled", False)
        self.allowed_devices = set(auth_config.get("allowed_devices", []))
        secret_key = self.config["server"]["auth_key"]
        expire_seconds = auth_config.get("expire_seconds", None)
        self.auth = AuthManager(secret_key=secret_key, expire_seconds=expire_seconds)
        self.connections_lock = asyncio.Lock()
        self.connections_by_session: Dict[str, ConnectionHandler] = {}
        self.connections_by_device: Dict[str, ConnectionHandler] = {}

    async def register_connection(self, handler: ConnectionHandler):
        session_id = getattr(handler, "session_id", "")
        device_id = getattr(handler, "device_id", "")
        async with self.connections_lock:
            if session_id:
                self.connections_by_session[session_id] = handler
            if device_id:
                self.connections_by_device[device_id] = handler

    async def unregister_connection(self, handler: ConnectionHandler):
        session_id = getattr(handler, "session_id", "")
        device_id = getattr(handler, "device_id", "")
        async with self.connections_lock:
            if (
                session_id
                and session_id in self.connections_by_session
                and self.connections_by_session[session_id] is handler
            ):
                self.connections_by_session.pop(session_id, None)
            if (
                device_id
                and device_id in self.connections_by_device
                and self.connections_by_device[device_id] is handler
            ):
                self.connections_by_device.pop(device_id, None)

    async def get_connection(
        self, session_id: Optional[str] = None, device_id: Optional[str] = None
    ) -> Optional[ConnectionHandler]:
        async with self.connections_lock:
            if session_id:
                return self.connections_by_session.get(session_id)
            if device_id:
                return self.connections_by_device.get(device_id)
            return None

    async def _resolve_existing_device_handler(self, websocket, device_id: str):
        normalized_device_id = str(device_id or "").strip()
        if not normalized_device_id:
            return None, False

        existing_handler = await self.get_connection(device_id=normalized_device_id)
        if existing_handler is None:
            return None, False

        ready_for_reconnect = await existing_handler.wait_until_transport_detached(
            timeout=2.0
        )
        if ready_for_reconnect:
            self.logger.bind(tag=TAG).info(
                "同设备重连，复用现有连接资源: "
                f"device_id={normalized_device_id}, session_id={existing_handler.session_id}"
            )
            return existing_handler, False

        self.logger.bind(tag=TAG).warning(
            "duplicate live device connection rejected: "
            f"device_id={normalized_device_id}, session_id={existing_handler.session_id}"
        )
        try:
            await websocket.send("same device-id is already connected")
        except Exception:
            pass
        try:
            await websocket.close(code=1008, reason="device already connected")
        except Exception:
            pass
        return None, True

    async def list_connections(self) -> list:
        async with self.connections_lock:
            items = []
            for session_id, conn in self.connections_by_session.items():
                websocket_alive = False
                ws = getattr(conn, "websocket", None)
                if ws is not None:
                    if hasattr(ws, "state"):
                        websocket_alive = ws.state.name != "CLOSED"
                    elif hasattr(ws, "closed"):
                        websocket_alive = not ws.closed

                mcp_ready = False
                mcp_tools = []
                mcp_client = getattr(conn, "mcp_client", None)
                if mcp_client is not None:
                    mcp_ready = await mcp_client.is_ready()
                    mcp_tools = sorted(list(getattr(mcp_client, "tools", {}).keys()))

                items.append(
                    {
                        "session_id": session_id,
                        "transport_session_id": getattr(
                            conn, "transport_session_id", ""
                        ),
                        "device_id": getattr(conn, "device_id", ""),
                        "user_id": getattr(conn, "user_id", ""),
                        "chat_session_id": getattr(conn, "chat_session_id", ""),
                        "model_session_key": getattr(conn, "model_session_key", ""),
                        "client_ip": getattr(conn, "client_ip", ""),
                        "mcp_ready": mcp_ready,
                        "mcp_tools": mcp_tools,
                        "websocket_alive": websocket_alive,
                    }
                )
            return items

    async def start(self):
        server_config = self.config["server"]
        host = server_config.get("ip", "0.0.0.0")
        port = int(server_config.get("port", 8000))
        ping_interval = _coerce_positive_float(
            server_config.get("websocket_ping_interval_seconds", 20),
            20.0,
        )
        ping_timeout = _coerce_positive_float(
            server_config.get("websocket_ping_timeout_seconds", 90),
            90.0,
        )
        self.logger.bind(tag=TAG).info(
            f"starting websocket server with ping_interval={ping_interval}s, "
            f"ping_timeout={ping_timeout}s"
        )

        async with websockets.serve(
            self._handle_connection,
            host,
            port,
            process_request=self._http_response,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        ):
            await asyncio.Future()

    async def stop(self):
        """Best-effort shutdown for active connections and shared MCP clients."""
        async with self.connections_lock:
            handlers = []
            seen = set()
            for handler in list(self.connections_by_session.values()) + list(
                self.connections_by_device.values()
            ):
                ident = id(handler)
                if ident in seen:
                    continue
                seen.add(ident)
                handlers.append(handler)
            self.connections_by_session = {}
            self.connections_by_device = {}

        for handler in handlers:
            try:
                if hasattr(handler, "request_final_close"):
                    handler.request_final_close("websocket server shutdown")
            except Exception as exc:
                self.logger.bind(tag=TAG).warning(
                    f"request_final_close failed during websocket shutdown: {exc}"
                )

        closable_handlers = [handler for handler in handlers if hasattr(handler, "close")]
        if closable_handlers:
            try:
                close_results = await asyncio.wait_for(
                    asyncio.gather(
                        *[
                            handler.close(getattr(handler, "websocket", None))
                            for handler in closable_handlers
                        ],
                        return_exceptions=True,
                    ),
                    timeout=15.0,
                )
                for handler, result in zip(closable_handlers, close_results):
                    if isinstance(result, Exception):
                        self.logger.bind(tag=TAG).error(
                            "websocket server shutdown close failed: "
                            f"session_id={getattr(handler, 'session_id', '')}, "
                            f"device_id={getattr(handler, 'device_id', '')}, error={result}"
                        )
            except asyncio.TimeoutError:
                self.logger.bind(tag=TAG).warning(
                    "websocket server shutdown timed out while closing active connections"
                )

        await ServerMCPManager.force_cleanup_shared_pool()

    async def _handle_connection(self, websocket):
        headers = dict(websocket.request.headers)
        if headers.get("device-id", None) is None:
            from urllib.parse import parse_qs, urlparse

            request_path = websocket.request.path
            if not request_path:
                self.logger.bind(tag=TAG).error("无法获取请求路径")
                await websocket.close()
                return

            parsed_url = urlparse(request_path)
            query_params = parse_qs(parsed_url.query)
            if "device-id" not in query_params:
                await websocket.send("端口正常，如需测试连接，请使用test_page.html")
                await websocket.close()
                return

            websocket.request.headers["device-id"] = query_params["device-id"][0]
            if "client-id" in query_params:
                websocket.request.headers["client-id"] = query_params["client-id"][0]
            if "authorization" in query_params:
                websocket.request.headers["authorization"] = query_params[
                    "authorization"
                ][0]

        try:
            await self._handle_auth(websocket)
        except AuthenticationError:
            await websocket.send("认证失败")
            await websocket.close()
            return

        device_id = websocket.request.headers.get("device-id", None)
        handler = None
        if device_id:
            handler, rejected = await self._resolve_existing_device_handler(
                websocket,
                device_id,
            )
            if rejected:
                return
        if device_id and handler is None:
            existing_handler = await self.get_connection(device_id=device_id)
            if existing_handler is not None:
                ready_for_reconnect = await existing_handler.wait_until_transport_detached(
                    timeout=2.0
                )
                if ready_for_reconnect:
                    handler = existing_handler
                    self.logger.bind(tag=TAG).info(
                        "同设备重连，复用现有连接资源: "
                        f"device_id={device_id}, session_id={handler.session_id}"
                    )

        if handler is None:
            handler = ConnectionHandler(
                self.config,
                self._vad,
                self._asr,
                self._llm,
                self._memory,
                self._intent,
                self,
            )

        try:
            await handler.handle_connection(websocket)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"处理连接时出错: {e}")
        finally:
            try:
                if hasattr(websocket, "closed") and not websocket.closed:
                    await websocket.close()
                elif hasattr(websocket, "state") and websocket.state.name != "CLOSED":
                    await websocket.close()
                else:
                    await websocket.close()
            except Exception as close_error:
                self.logger.bind(tag=TAG).error(
                    f"服务端强制关闭连接时出错: {close_error}"
                )

    async def _http_response(self, websocket, request_headers):
        if request_headers.headers.get("connection", "").lower() == "upgrade":
            return None
        return websocket.respond(200, "Server is running\n")

    async def update_config(self) -> bool:
        """更新服务器配置并重新初始化组件。"""
        try:
            async with self.config_lock:
                new_config = await get_config_from_api_async(self.config)
                if new_config is None:
                    self.logger.bind(tag=TAG).error("获取新配置失败")
                    return False

                self.logger.bind(tag=TAG).info("获取新配置成功")
                update_vad = check_vad_update(self.config, new_config)
                update_asr = check_asr_update(self.config, new_config)
                self.logger.bind(tag=TAG).info(
                    f"检查VAD和ASR类型是否需要更新: {update_vad} {update_asr}"
                )

                self.config = new_config
                modules = initialize_modules(
                    self.logger,
                    new_config,
                    update_vad,
                    update_asr,
                    "LLM" in new_config["selected_module"],
                    False,
                    "Memory" in new_config["selected_module"],
                    "Intent" in new_config["selected_module"],
                )

                if "vad" in modules:
                    self._vad = modules["vad"]
                if "asr" in modules:
                    self._asr = modules["asr"]
                if "llm" in modules:
                    self._llm = modules["llm"]
                if "intent" in modules:
                    self._intent = modules["intent"]
                if "memory" in modules:
                    self._memory = modules["memory"]

                self.logger.bind(tag=TAG).info("更新配置任务执行完毕")
                return True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"更新服务器配置失败: {str(e)}")
            return False

    async def _handle_auth(self, websocket):
        if not self.auth_enable:
            return

        headers = dict(websocket.request.headers)
        device_id = headers.get("device-id", None)
        client_id = headers.get("client-id", None)

        if self.allowed_devices and device_id in self.allowed_devices:
            return

        token = headers.get("authorization", "")
        if token.startswith("Bearer "):
            token = token[7:]
        else:
            raise AuthenticationError("Missing or invalid Authorization header")

        auth_success = self.auth.verify_token(
            token, client_id=client_id, username=device_id
        )
        if not auth_success:
            raise AuthenticationError("Invalid token")
