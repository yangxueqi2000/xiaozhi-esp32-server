"""Shared server-side MCP manager."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from mcp.types import LoggingMessageNotificationParams

from config.config_loader import get_project_dir
from config.logger import setup_logging
from .mcp_client import ServerMCPClient, _coerce_timeout_value

TAG = __name__
logger = setup_logging()

_REQUIRED_CLIENT_TOOLS: Dict[str, frozenset[str]] = {
    "uvvis": frozenset(
        {
            "uvvis_session",
            "uvvis_prepare_dark_current",
            "uvvis_measure_spectra",
            "uvvis_measure_kinetics",
        }
    )
}


class ServerMCPManager:
    """Shared MCP client pool for all device connections."""

    _shared_clients: Dict[str, ServerMCPClient] = {}
    _shared_client_tools: Dict[str, List[Dict[str, Any]]] = {}
    _shared_tool_to_client: Dict[str, str] = {}
    _shared_initialized = False
    _shared_ref_count = 0
    _shared_init_lock: Optional[asyncio.Lock] = None
    _shared_reconnect_locks: Dict[str, asyncio.Lock] = {}

    def __init__(self, conn) -> None:
        self.conn = conn
        self.config_path = get_project_dir() + "data/.mcp_server_settings.json"
        if not os.path.exists(self.config_path):
            self.config_path = ""
            logger.bind(tag=TAG).warning(
                "MCP config file is missing: data/.mcp_server_settings.json"
            )
        self._acquired = False

    @classmethod
    def _take_shared_clients_snapshot(
        cls,
    ) -> list[tuple[str, ServerMCPClient]]:
        clients = list(cls._shared_clients.items())
        cls._shared_clients = {}
        cls._shared_client_tools = {}
        cls._shared_tool_to_client = {}
        cls._shared_reconnect_locks = {}
        cls._shared_initialized = False
        cls._shared_ref_count = 0
        return clients

    @classmethod
    async def force_cleanup_shared_pool(cls) -> None:
        """Force-close the shared MCP pool regardless of connection ref count."""
        init_lock = cls._get_init_lock()
        async with init_lock:
            clients = cls._take_shared_clients_snapshot()

        for name, client in clients:
            try:
                await asyncio.wait_for(client.cleanup(), timeout=20)
                logger.bind(tag=TAG).info(
                    f"Force-closed shared server MCP client: {name}"
                )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.bind(tag=TAG).error(
                    f"Error force-closing server MCP client {name}: {exc}"
                )

    @classmethod
    def _get_init_lock(cls) -> asyncio.Lock:
        if cls._shared_init_lock is None:
            cls._shared_init_lock = asyncio.Lock()
        return cls._shared_init_lock

    @classmethod
    def _get_reconnect_lock(cls, client_name: str) -> asyncio.Lock:
        lock = cls._shared_reconnect_locks.get(client_name)
        if lock is None:
            lock = asyncio.Lock()
            cls._shared_reconnect_locks[client_name] = lock
        return lock

    def load_config(self) -> Dict[str, Any]:
        """Load MCP server config."""
        if not self.config_path:
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            return config.get("mcpServers", {})
        except Exception as exc:
            logger.bind(tag=TAG).error(
                f"Error loading MCP config from {self.config_path}: {exc}"
            )
            return {}

    @staticmethod
    def _resolve_client_initialize_timeout(srv_config: Dict[str, Any]) -> float:
        transport_type = str(srv_config.get("transport", "") or "").strip().lower()
        default_timeout = 20.0 if transport_type in {"streamable-http", "http", "sse"} else 10.0
        raw_timeout = srv_config.get(
            "initialize_timeout",
            srv_config.get("init_timeout", srv_config.get("timeout", default_timeout)),
        )
        timeout_value = _coerce_timeout_value(
            raw_timeout,
            default=default_timeout,
            allow_unbounded=False,
        )
        try:
            resolved = float(timeout_value or default_timeout)
        except (TypeError, ValueError):
            resolved = default_timeout
        return max(5.0, resolved)

    async def _build_client(
        self, name: str, srv_config: Dict[str, Any]
    ) -> Optional[tuple[str, ServerMCPClient, List[Dict[str, Any]]]]:
        client = None
        try:
            logger.bind(tag=TAG).info(f"Initializing server MCP client: {name}")
            client = ServerMCPClient(srv_config)
            init_timeout = self._resolve_client_initialize_timeout(srv_config)
            await asyncio.wait_for(
                client.initialize(logging_callback=self.logging_callback),
                timeout=init_timeout,
            )
            return name, client, client.get_available_tools()
        except asyncio.TimeoutError:
            logger.bind(tag=TAG).error(
                f"Failed to initialize MCP server {name}: timeout after {init_timeout:.1f}s"
            )
            if client:
                await client.cleanup()
        except Exception as exc:
            logger.bind(tag=TAG).error(
                f"Failed to initialize MCP server {name}: {exc}"
            )
            if client:
                await client.cleanup()
        return None

    @classmethod
    def _register_client_tools(
        cls,
        client_name: str,
        client: ServerMCPClient,
        client_tools: List[Dict[str, Any]],
    ) -> None:
        cls._shared_clients[client_name] = client
        cls._shared_client_tools[client_name] = list(client_tools)

        stale_tool_names = [
            tool_name
            for tool_name, owner in list(cls._shared_tool_to_client.items())
            if owner == client_name
        ]
        for tool_name in stale_tool_names:
            cls._shared_tool_to_client.pop(tool_name, None)

        for tool in client_tools:
            function = tool.get("function") or {}
            tool_name = function.get("name")
            if tool_name:
                cls._shared_tool_to_client[tool_name] = client_name

    async def _initialize_shared_pool(self) -> None:
        config = self.load_config()
        tasks = []
        for name, srv_config in config.items():
            if not srv_config.get("command") and not srv_config.get("url"):
                logger.bind(tag=TAG).warning(
                    f"Skipping server {name}: neither command nor url specified"
                )
                continue
            tasks.append(self._build_client(name, srv_config))

        results: List[Optional[tuple[str, ServerMCPClient, List[Dict[str, Any]]]]] = []
        if tasks:
            results = await asyncio.gather(*tasks)

        type(self)._shared_clients.clear()
        type(self)._shared_client_tools.clear()
        type(self)._shared_tool_to_client.clear()

        for result in results:
            if not result:
                continue
            name, client, client_tools = result
            type(self)._register_client_tools(name, client, client_tools)

        type(self)._shared_initialized = True

    def _refresh_conn_tool_cache(self) -> None:
        func_handler = getattr(self.conn, "func_handler", None)
        if not func_handler:
            return
        tool_manager = getattr(func_handler, "tool_manager", None)
        if tool_manager:
            tool_manager.refresh_tools()
        func_handler.current_support_functions()

    async def initialize_servers(self) -> None:
        """Acquire the shared MCP pool for this connection."""
        init_lock = self._get_init_lock()
        async with init_lock:
            if not type(self)._shared_initialized:
                await self._initialize_shared_pool()

            if not self._acquired:
                type(self)._shared_ref_count += 1
                self._acquired = True

        if "uvvis" in self.load_config():
            await self.ensure_client_initialized("uvvis")
        self._refresh_conn_tool_cache()

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Return all shared MCP tool definitions."""
        tools: List[Dict[str, Any]] = []
        for client_tools in type(self)._shared_client_tools.values():
            tools.extend(client_tools)
        return tools

    def _build_tool_call_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}

        device_id = str(getattr(self.conn, "device_id", "") or "").strip()
        if not device_id and isinstance(getattr(self.conn, "headers", None), dict):
            headers = getattr(self.conn, "headers", {}) or {}
            device_id = str(
                headers.get("device-id")
                or headers.get("Device-Id")
                or headers.get("device_id")
                or ""
            ).strip()
        if device_id:
            meta["device_id"] = device_id
            meta["deviceId"] = device_id
            meta["xiaozhi_device_id"] = device_id
            meta["target_device_id"] = device_id

        session_id = str(getattr(self.conn, "session_id", "") or "").strip()
        if session_id:
            meta["session_id"] = session_id
            meta["sessionId"] = session_id

        experiment_session_id = str(
            getattr(self.conn, "experiment_session_id", "") or ""
        ).strip()
        if experiment_session_id:
            meta["experiment_session_id"] = experiment_session_id

        return meta

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check whether a tool exists in the shared pool."""
        return tool_name in type(self)._shared_tool_to_client

    @classmethod
    def _missing_required_tools(cls, client_name: str) -> set[str]:
        required = _REQUIRED_CLIENT_TOOLS.get(client_name, frozenset())
        if not required:
            return set()

        available = {
            (tool.get("function") or {}).get("name")
            for tool in cls._shared_client_tools.get(client_name, [])
            if isinstance(tool, dict)
        }
        return {tool_name for tool_name in required if tool_name not in available}

    async def ensure_client_initialized(self, client_name: str) -> bool:
        """Best-effort targeted recovery for a missing or disconnected shared client."""
        current = type(self)._shared_clients.get(client_name)
        missing_required_tools = self._missing_required_tools(client_name)
        if current is not None and current.is_connected() and not missing_required_tools:
            return True
        if missing_required_tools:
            logger.bind(tag=TAG).warning(
                "MCP client %s is missing required tools %s; forcing a refresh",
                client_name,
                sorted(missing_required_tools),
            )

        reconnect_lock = self._get_reconnect_lock(client_name)
        async with reconnect_lock:
            current = type(self)._shared_clients.get(client_name)
            missing_required_tools = self._missing_required_tools(client_name)
            if current is not None and current.is_connected() and not missing_required_tools:
                return True

            if current is not None:
                try:
                    await current.cleanup()
                except Exception as exc:
                    logger.bind(tag=TAG).warning(
                        f"Error cleaning stale MCP client {client_name}: {exc}"
                    )

            config = self.load_config()
            srv_config = config.get(client_name)
            if not isinstance(srv_config, dict):
                logger.bind(tag=TAG).warning(
                    f"Cannot initialize MCP client {client_name}: config not found"
                )
                return False

            rebuilt = await self._build_client(client_name, srv_config)
            if not rebuilt:
                return False

            _, client, client_tools = rebuilt
            type(self)._register_client_tools(client_name, client, client_tools)
            logger.bind(tag=TAG).info(
                f"Initialized missing shared MCP client successfully: {client_name}"
            )

        self._refresh_conn_tool_cache()
        missing_required_tools = self._missing_required_tools(client_name)
        if missing_required_tools:
            logger.bind(tag=TAG).warning(
                "MCP client %s is still missing required tools after refresh: %s",
                client_name,
                sorted(missing_required_tools),
            )
            return False
        return True

    async def _reconnect_client(
        self, client_name: str, failed_client: ServerMCPClient
    ) -> ServerMCPClient:
        reconnect_lock = self._get_reconnect_lock(client_name)
        async with reconnect_lock:
            current = type(self)._shared_clients.get(client_name)
            if current is not None and current is not failed_client and current.is_connected():
                return current

            if current is not None:
                try:
                    await current.cleanup()
                except Exception as exc:
                    logger.bind(tag=TAG).warning(
                        f"Error cleaning stale MCP client {client_name}: {exc}"
                    )

            config = self.load_config()
            srv_config = config.get(client_name)
            if not isinstance(srv_config, dict):
                raise RuntimeError(
                    f"Cannot reconnect MCP client {client_name}: config not found"
                )

            rebuilt = await self._build_client(client_name, srv_config)
            if not rebuilt:
                raise RuntimeError(f"Failed to reconnect MCP client {client_name}")

            _, client, client_tools = rebuilt
            type(self)._register_client_tools(client_name, client, client_tools)
            logger.bind(tag=TAG).info(
                f"Reconnected shared MCP client successfully: {client_name}"
            )
            return client

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        priority: str = "foreground",
    ) -> Any:
        """Execute a shared MCP tool call with retry and reconnect."""
        logger.bind(tag=TAG).info(
            f"Executing server MCP tool {tool_name}, priority={priority}, arguments: {arguments}"
        )

        max_retries = 3
        retry_interval = 2

        client_name = type(self)._shared_tool_to_client.get(tool_name)
        if not client_name:
            raise ValueError(f"Tool {tool_name} was not found in any MCP server")

        target_client = type(self)._shared_clients.get(client_name)
        if not target_client:
            raise RuntimeError(f"MCP client {client_name} is not initialized")

        call_hook_entered = False
        tool_call_meta = self._build_tool_call_meta()
        try:
            if (
                client_name == "experiment-graph"
                and hasattr(self.conn, "_before_experiment_graph_tool_call")
            ):
                await self.conn._before_experiment_graph_tool_call(
                    tool_name,
                    priority=priority,
                )
                call_hook_entered = True

            for attempt in range(max_retries):
                try:
                    return await target_client.call_tool(
                        tool_name,
                        arguments,
                        progress_callback=self.progress_callback,
                        meta=tool_call_meta or None,
                    )
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise

                    logger.bind(tag=TAG).warning(
                        f"Tool {tool_name} failed (attempt {attempt + 1}/{max_retries}): {exc}"
                    )
                    logger.bind(tag=TAG).info(
                        f"Trying to reconnect shared MCP client before retry: {client_name}"
                    )

                    try:
                        target_client = await self._reconnect_client(
                            client_name, target_client
                        )
                    except Exception as reconnect_error:
                        logger.bind(tag=TAG).error(
                            f"Failed to reconnect MCP client {client_name}: {reconnect_error}"
                        )

                    await asyncio.sleep(retry_interval)
        finally:
            if (
                call_hook_entered
                and client_name == "experiment-graph"
                and hasattr(self.conn, "_after_experiment_graph_tool_call")
            ):
                await self.conn._after_experiment_graph_tool_call(
                    tool_name,
                    priority=priority,
                )

    async def cleanup_all(self) -> None:
        """Release this connection's reference to the shared MCP pool."""
        if not self._acquired:
            return

        init_lock = self._get_init_lock()
        async with init_lock:
            if self._acquired and type(self)._shared_ref_count > 0:
                type(self)._shared_ref_count -= 1
            self._acquired = False

            if type(self)._shared_ref_count > 0:
                return

            clients = type(self)._take_shared_clients_snapshot()

        for name, client in clients:
            try:
                await asyncio.wait_for(client.cleanup(), timeout=20)
                logger.bind(tag=TAG).info(f"Server MCP client closed: {name}")
            except (asyncio.TimeoutError, Exception) as exc:
                logger.bind(tag=TAG).error(
                    f"Error closing server MCP client {name}: {exc}"
                )

    async def logging_callback(self, params: LoggingMessageNotificationParams):
        logger.bind(tag=TAG).info(
            f"[Server Log - {params.level.upper()}] {params.data}"
        )

    async def progress_callback(
        self, progress: float, total: float | None, message: str | None
    ) -> None:
        logger.bind(tag=TAG).info(f"[Progress {progress}/{total}]: {message}")
