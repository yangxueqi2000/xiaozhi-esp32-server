from typing import Any, Dict, Optional

from plugins_func.register import Action, ActionResponse

from ..base import ToolDefinition, ToolExecutor, ToolType
from .mcp_manager import ServerMCPManager
from .payload_utils import (
    finalize_server_mcp_payload,
    serialize_result_for_llm,
    sync_server_mcp_payload_state,
)
from .photo_capture_rule import ServerPhotoCaptureRule
from .uvvis_scan_rule import UVVisScanRule


class ServerMCPExecutor(ToolExecutor):
    """Server MCP tool executor."""

    def __init__(self, conn):
        self.conn = conn
        self.mcp_manager: Optional[ServerMCPManager] = None
        self._initialized = False
        self.photo_capture_rule = ServerPhotoCaptureRule(conn)
        self.uvvis_scan_rule = UVVisScanRule(conn, self._get_mcp_manager)

    def _get_mcp_manager(self) -> Optional[ServerMCPManager]:
        return self.mcp_manager

    async def initialize(self):
        if not self._initialized:
            self.mcp_manager = ServerMCPManager(self.conn)
            self._initialized = True
            await self.mcp_manager.initialize_servers()

    async def execute(
        self, conn, tool_name: str, arguments: Dict[str, Any]
    ) -> ActionResponse:
        if not self._initialized or not self.mcp_manager:
            return ActionResponse(
                action=Action.ERROR,
                response="MCP管理器未初始化",
            )

        actual_tool_name = tool_name[4:] if tool_name.startswith("mcp_") else tool_name
        call_args = dict(arguments or {})
        self.uvvis_scan_rule.prepare_arguments(actual_tool_name, call_args)
        uvvis_intercept = self.uvvis_scan_rule.before_execute(actual_tool_name, call_args)
        if uvvis_intercept is not None:
            return uvvis_intercept

        intercept_response, restore_photo_grant = self.photo_capture_rule.before_execute(
            actual_tool_name
        )
        if intercept_response is not None:
            return intercept_response

        try:
            result = await self.mcp_manager.execute_tool(actual_tool_name, call_args)
            payload = finalize_server_mcp_payload(
                result,
                tool_name=actual_tool_name,
                arguments=call_args,
            )
            sync_server_mcp_payload_state(
                self.conn,
                tool_name=actual_tool_name,
                payload=payload,
            )
            follow_up_response = await self.uvvis_scan_rule.after_execute(
                actual_tool_name,
                call_args,
                payload,
            )
            if follow_up_response is not None:
                return follow_up_response

            return ActionResponse(
                action=Action.REQLLM,
                result=serialize_result_for_llm(payload if payload is not None else result),
            )
        except ValueError as e:
            self.uvvis_scan_rule.handle_execute_error(
                actual_tool_name,
                call_args,
                e,
            )
            self.photo_capture_rule.restore_after_failure(
                actual_tool_name,
                restore_photo_grant,
            )
            return ActionResponse(action=Action.NOTFOUND, response=str(e))
        except Exception as e:
            self.uvvis_scan_rule.handle_execute_error(
                actual_tool_name,
                call_args,
                e,
            )
            self.photo_capture_rule.restore_after_failure(
                actual_tool_name,
                restore_photo_grant,
            )
            return ActionResponse(action=Action.ERROR, response=str(e))

    def get_tools(self) -> Dict[str, ToolDefinition]:
        if not self._initialized or not self.mcp_manager:
            return {}

        tools = {}
        mcp_tools = self.mcp_manager.get_all_tools()
        for tool in mcp_tools:
            func_def = tool.get("function", {})
            tool_name = func_def.get("name", "")
            if tool_name:
                tools[tool_name] = ToolDefinition(
                    name=tool_name,
                    description=tool,
                    tool_type=ToolType.SERVER_MCP,
                )
        return tools

    def has_tool(self, tool_name: str) -> bool:
        if not self._initialized or not self.mcp_manager:
            return False
        actual_tool_name = tool_name[4:] if tool_name.startswith("mcp_") else tool_name
        return self.mcp_manager.is_mcp_tool(actual_tool_name)

    async def cleanup(self):
        await self.uvvis_scan_rule.cleanup()
        if self.mcp_manager:
            await self.mcp_manager.cleanup_all()
