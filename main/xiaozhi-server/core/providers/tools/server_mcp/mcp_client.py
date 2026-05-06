"""服务端MCP客户端"""

from __future__ import annotations

from datetime import timedelta
import asyncio
import os
import shutil
import concurrent.futures
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any

from mcp import ClientSession, StdioServerParameters, Implementation
from mcp.client.session import SamplingFnT, ElicitationFnT, ListRootsFnT, LoggingFnT, MessageHandlerFnT
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.session import ProgressFnT

from config.logger import setup_logging
from core.utils.util import sanitize_tool_name

TAG = __name__


def _coerce_timeout_value(
    raw_value: Any,
    *,
    default: float | None,
    allow_unbounded: bool = False,
) -> float | None:
    """Normalize timeout config values for MCP transports.

    When allow_unbounded=True, explicit null / empty / non-positive values disable
    the read timeout so long-running tools can stream until completion.
    """
    if isinstance(raw_value, timedelta):
        numeric_value = raw_value.total_seconds()
        if allow_unbounded and numeric_value <= 0:
            return None
        return numeric_value

    if raw_value is None:
        return None if allow_unbounded else default

    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if not normalized:
            return None if allow_unbounded else default
        if normalized in {"none", "null", "inf", "infinite", "unlimited", "disable", "disabled"}:
            return None if allow_unbounded else default
        try:
            numeric_value = float(normalized)
        except ValueError:
            return default
    else:
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            return default

    if allow_unbounded and numeric_value <= 0:
        return None
    return numeric_value


class ServerMCPClient:
    """服务端MCP客户端，用于连接和管理MCP服务"""

    def __init__(self, config: Dict[str, Any]):
        """初始化服务端MCP客户端

        Args:
            config: MCP服务配置字典
        """
        self.logger = setup_logging()
        self.config = config

        self._worker_task: Optional[asyncio.Task] = None
        self._ready_evt = asyncio.Event()
        self._shutdown_evt = asyncio.Event()
        self._call_lock: Optional[asyncio.Lock] = None
        self._worker_error: Optional[BaseException] = None

        self.session: Optional[ClientSession] = None
        self.tools: List = []  # 原始工具对象
        self.tools_dict: Dict[str, Any] = {}
        self.name_mapping: Dict[str, str] = {}

    def _reset_runtime_state(self) -> None:
        self.session = None
        self.tools = []
        self.tools_dict = {}
        self.name_mapping = {}
        self._call_lock = None

    async def initialize(self, read_timeout_seconds: timedelta | None = None,
             sampling_callback: SamplingFnT | None = None,
             elicitation_callback: ElicitationFnT | None = None,
             list_roots_callback: ListRootsFnT | None = None,
             logging_callback: LoggingFnT | None = None,
             message_handler: MessageHandlerFnT | None = None,
             client_info: Implementation | None = None):
        """初始化MCP客户端连接"""
        if self._worker_task:
            if self._worker_error is not None:
                raise RuntimeError(
                    f"服务端MCP客户端初始化失败: {self._worker_error}"
                ) from self._worker_error
            return

        self._ready_evt = asyncio.Event()
        self._shutdown_evt = asyncio.Event()
        self._worker_error = None
        self._reset_runtime_state()

        self._worker_task = asyncio.create_task(
            self._worker(read_timeout_seconds=read_timeout_seconds,
                        sampling_callback=sampling_callback,
                        elicitation_callback=elicitation_callback,
                        list_roots_callback=list_roots_callback,
                        logging_callback=logging_callback,
                        message_handler=message_handler,
                        client_info=client_info), name="ServerMCPClientWorker"
        )
        await self._ready_evt.wait()

        if self._worker_error is not None:
            worker_error = self._worker_error
            await self.cleanup()
            raise RuntimeError(
                f"服务端MCP客户端初始化失败: {worker_error}"
            ) from worker_error

        if not self.session:
            await self.cleanup()
            raise RuntimeError("服务端MCP客户端初始化失败: MCP session unavailable")

        self.logger.bind(tag=TAG).info(
            f"服务端MCP客户端已连接，可用工具: {[name for name in self.name_mapping.values()]}"
        )

    async def cleanup(self):
        """清理MCP客户端资源"""
        if not self._worker_task:
            self._reset_runtime_state()
            self._ready_evt = asyncio.Event()
            self._shutdown_evt = asyncio.Event()
            self._worker_error = None
            return

        worker_task = self._worker_task
        self._shutdown_evt.set()
        if self._worker_error is not None:
            try:
                await asyncio.gather(worker_task, return_exceptions=True)
            except Exception:
                pass
            self._worker_task = None
            self._reset_runtime_state()
            self._ready_evt = asyncio.Event()
            self._shutdown_evt = asyncio.Event()
            self._worker_error = None
            return
        try:
            await asyncio.wait_for(asyncio.shield(worker_task), timeout=20)
        except (asyncio.TimeoutError, Exception) as e:
            self.logger.bind(tag=TAG).error(f"服务端MCP客户端关闭错误: {e}")
        finally:
            self._worker_task = None
            self._reset_runtime_state()
            self._ready_evt = asyncio.Event()
            self._shutdown_evt = asyncio.Event()
            self._worker_error = None

    def has_tool(self, name: str) -> bool:
        """检查是否包含指定工具

        Args:
            name: 工具名称

        Returns:
            bool: 是否包含该工具
        """
        return name in self.tools_dict

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取所有可用工具的定义

        Returns:
            List[Dict[str, Any]]: 工具定义列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for name, tool in self.tools_dict.items()
        ]

    async def call_tool(self, name: str, arguments: dict, read_timeout_seconds: timedelta | None = None, progress_callback: ProgressFnT | None = None, *, meta: dict[str, Any] | None = None) -> Any:
        """调用指定工具

        Args:
            name: 工具名称
            arguments: 工具参数
            read_timeout_seconds:
            progress_callback: 进度回调函数
            meta:

        Returns:
            Any: 工具执行结果

        Raises:
            RuntimeError: 客户端未初始化时抛出
        """
        if not self.session:
            raise RuntimeError("服务端MCP客户端未初始化")

        real_name = self.name_mapping.get(name, name)
        loop = self._worker_task.get_loop()
        coro = self._call_tool_serialized(
            real_name,
            arguments=arguments,
            read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
            meta=meta,
        )

        if loop is asyncio.get_running_loop():
            return await coro

        fut: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wrap_future(fut)

    async def _call_tool_serialized(
        self,
        real_name: str,
        *,
        arguments: dict,
        read_timeout_seconds: timedelta | None,
        progress_callback: ProgressFnT | None,
        meta: dict[str, Any] | None,
    ) -> Any:
        if not self.session:
            raise RuntimeError("MCP session is not initialized")
        if self._call_lock is None:
            self._call_lock = asyncio.Lock()

        async with self._call_lock:
            return await self.session.call_tool(
                real_name,
                arguments=arguments,
                read_timeout_seconds=read_timeout_seconds,
                progress_callback=progress_callback,
                meta=meta,
            )

    def is_connected(self) -> bool:
        """检查MCP客户端是否连接正常

        Returns:
            bool: 如果客户端已连接并正常工作，返回True，否则返回False
        """
        # 检查工作任务是否存在
        if self._worker_task is None:
            return False

        # 检查工作任务是否已经完成或取消
        if self._worker_task.done():
            return False

        # 检查会话是否存在
        if self.session is None:
            return False

        # 所有检查都通过，连接正常
        return True

    async def _worker(self, read_timeout_seconds: timedelta | None = None,
             sampling_callback: SamplingFnT | None = None,
             elicitation_callback: ElicitationFnT | None = None,
             list_roots_callback: ListRootsFnT | None = None,
             logging_callback: LoggingFnT | None = None,
             message_handler: MessageHandlerFnT | None = None,
             client_info: Implementation | None = None):
        """MCP客户端工作协程"""
        async with AsyncExitStack() as stack:
            try:
                # 建立 StdioClient
                if "command" in self.config:
                    cmd = (
                        shutil.which("npx")
                        if self.config["command"] == "npx"
                        else self.config["command"]
                    )
                    env = {**os.environ, **self.config.get("env", {})}
                    params = StdioServerParameters(
                        command=cmd,
                        args=self.config.get("args", []),
                        env=env,
                    )
                    stdio_r, stdio_w = await stack.enter_async_context(
                        stdio_client(params)
                    )
                    read_stream, write_stream = stdio_r, stdio_w

                # 建立SSEClient
                elif "url" in self.config:
                    headers = dict(self.config.get("headers", {}))
                    # TODO 兼容旧版本
                    if "API_ACCESS_TOKEN" in self.config:
                        headers["Authorization"] = f"Bearer {self.config['API_ACCESS_TOKEN']}"
                        self.logger.bind(tag=TAG).warning(f"你正在使用旧过时的配置 API_ACCESS_TOKEN ，请在.mcp_server_settings.json中将API_ACCESS_TOKEN直接设置在headers中，例如 'Authorization': 'Bearer API_ACCESS_TOKEN'")
                   
                    # 根据transport类型选择不同的客户端，默认为SSE
                    transport_type = self.config.get("transport", "sse")
                    timeout_value = _coerce_timeout_value(
                        self.config.get("timeout", 30 if transport_type in {"streamable-http", "http"} else 5),
                        default=30 if transport_type in {"streamable-http", "http"} else 5,
                        allow_unbounded=False,
                    )
                    sse_read_timeout_value = _coerce_timeout_value(
                        self.config.get("sse_read_timeout", 60 * 5),
                        default=60 * 5,
                        allow_unbounded=True,
                    )
                    if sse_read_timeout_value is None:
                        self.logger.bind(tag=TAG).info(
                            f"server MCP read timeout is disabled for {self.config.get('url', '')}; "
                            "long-running tools may wait until completion"
                        )

                    if transport_type == "streamable-http" or transport_type == "http":
                        # 使用 Streamable HTTP 传输
                        http_r, http_w, get_session_id = await stack.enter_async_context(
                            streamablehttp_client(
                                url=self.config["url"],
                                headers=headers,
                                timeout=timeout_value,
                                sse_read_timeout=sse_read_timeout_value,
                                terminate_on_close=self.config.get("terminate_on_close", True)
                            )
                        )
                        read_stream, write_stream = http_r, http_w
                    else:
                        # 使用传统的 SSE 传输
                        sse_r, sse_w = await stack.enter_async_context(
                            sse_client(
                                url=self.config["url"],
                                headers=headers,
                                timeout=timeout_value,
                                sse_read_timeout=sse_read_timeout_value
                            )
                        )
                        read_stream, write_stream = sse_r, sse_w

                else:
                    raise ValueError("MCP客户端配置必须包含'command'或'url'")

                self.session = await stack.enter_async_context(
                    ClientSession(
                        read_stream=read_stream,
                        write_stream=write_stream,
                        read_timeout_seconds=read_timeout_seconds,
                        sampling_callback=sampling_callback,
                        elicitation_callback=elicitation_callback,
                        list_roots_callback=list_roots_callback,
                        logging_callback=logging_callback,
                        message_handler=message_handler,
                        client_info=client_info
                    )
                )
                await self.session.initialize()

                # 获取工具
                self.tools = (await self.session.list_tools()).tools
                self.tools_dict = {}
                self.name_mapping = {}
                for t in self.tools:
                    sanitized = sanitize_tool_name(t.name)
                    self.tools_dict[sanitized] = t
                    self.name_mapping[sanitized] = t.name

                self._ready_evt.set()

                # 挂起等待关闭
                await self._shutdown_evt.wait()

            except Exception as e:
                self._worker_error = e
                self.logger.bind(tag=TAG).error(f"服务端MCP客户端工作协程错误: {e}")
                self._ready_evt.set()
                raise
            finally:
                self._reset_runtime_state()
