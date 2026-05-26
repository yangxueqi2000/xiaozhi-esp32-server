from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

mcp = FastMCP("fake-xiaozhi-device-trigger")
LATEST: Dict[str, Dict[str, Any]] = {}


def _device_id(device_id: Optional[str]) -> str:
    return str(device_id or os.getenv("FAKE_XIAOZHI_DEVICE_ID") or "codex-exp1-tutorial-smoke").strip()


def _data_root() -> Path:
    root = str(os.getenv("FAKE_XIAOZHI_DATA_ROOT") or "").strip()
    return Path(root).resolve() if root else (Path.cwd() / "data").resolve()


def _safe_file_name(photo_name: str, append_timestamp: bool) -> str:
    base = str(photo_name or "fake_photo").strip()
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", base).strip(" ._") or "fake_photo"
    if append_timestamp:
        base = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not base.lower().endswith((".png", ".jpg", ".jpeg")):
        base = f"{base}.png"
    return base


def _group_dir_name(group_number: Optional[int]) -> str:
    try:
        n = int(group_number) if group_number is not None else 0
    except (TypeError, ValueError):
        n = 0
    return str(n) if n > 0 else ""


def _photo_dir(device_id: str, group_number: Optional[int]) -> Path:
    root = _data_root() / device_id
    group_dir = _group_dir_name(group_number)
    return root / group_dir if group_dir else root


@mcp.tool(name="xiaozhi_list_sessions", description="List fake online xiaozhi sessions for text smoke tests.")
def xiaozhi_list_sessions() -> Dict[str, Any]:
    device = _device_id(None)
    return {
        "success": True,
        "connections": [
            {
                "device_id": device,
                "session_id": "fake-session",
                "chat_session_id": "fake-chat-session",
                "model_session_key": "fake-model-session",
                "websocket_alive": True,
                "mcp_ready": True,
            }
        ],
    }


@mcp.tool(name="xiaozhi_debug_route_context", description="Return fake route context for text smoke tests.")
def xiaozhi_debug_route_context() -> Dict[str, Any]:
    return {"success": True, "mode": "fake", "device_id": _device_id(None)}


@mcp.tool(name="xiaozhi_take_photo", description="Fake device photo capture for text smoke tests.")
def xiaozhi_take_photo(
    question: str = "Please take a photo.",
    photo_name: str = "",
    append_timestamp: bool = True,
    time_format: str = "",
    device_id: Optional[str] = None,
    group_number: Optional[int] = None,
    tool_name: str = "",
    timeout: int = 20,
    request_timeout: int = 30,
) -> Dict[str, Any]:
    device = _device_id(device_id)
    file_name = _safe_file_name(photo_name, append_timestamp)
    out_dir = _photo_dir(device, group_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / file_name
    path.write_bytes(PNG_1X1)
    meta = {
        "file_name": file_name,
        "local_path": str(path),
        "mirrored_path": str(path),
        "path": str(path),
        "is_new_photo": True,
        "fake": True,
        "question": question,
    }
    LATEST[device] = meta
    return {
        "success": True,
        "recovered_after_error": False,
        "requested_photo_name": photo_name,
        "group_number": group_number,
        "photo_meta": meta,
        "result": {
            "success": True,
            "message": "fake photo captured",
            "device_id": device,
        },
    }


@mcp.tool(name="xiaozhi_get_latest_photo", description="Get latest fake local photo metadata.")
def xiaozhi_get_latest_photo(
    device_id: Optional[str] = None,
    group_number: Optional[int] = None,
) -> Dict[str, Any]:
    device = _device_id(device_id)
    meta = LATEST.get(device)
    if not meta:
        return {"success": False, "message": "no fake photo captured", "device_id": device}
    return {"success": True, "device_id": device, "group_number": group_number, "photo_meta": meta}


@mcp.tool(name="xiaozhi_list_recent_photos", description="List recent fake photos.")
def xiaozhi_list_recent_photos(
    limit: int = 5,
    device_id: Optional[str] = None,
    group_number: Optional[int] = None,
) -> Dict[str, Any]:
    device = _device_id(device_id)
    photos = [LATEST[device]] if device in LATEST else []
    return {"success": True, "device_id": device, "group_number": group_number, "photos": photos[:limit]}


@mcp.tool(name="xiaozhi_preview_local_file", description="Pretend to preview a local file.")
def xiaozhi_preview_local_file(
    file_name: str = "",
    photo_index: int = 0,
    device_id: Optional[str] = None,
    group_number: Optional[int] = None,
) -> Dict[str, Any]:
    return {"success": True, "message": "fake preview accepted", "file_name": file_name}


@mcp.tool(name="xiaozhi_preview_previous_photo", description="Pretend to preview the previous photo.")
def xiaozhi_preview_previous_photo(
    device_id: Optional[str] = None,
    group_number: Optional[int] = None,
) -> Dict[str, Any]:
    return {"success": True, "message": "fake previous photo preview accepted"}


@mcp.tool(name="xiaozhi_save_latest_photo_as", description="Pretend to save latest fake photo under another name.")
def xiaozhi_save_latest_photo_as(
    photo_name: str,
    append_timestamp: bool = False,
    device_id: Optional[str] = None,
    group_number: Optional[int] = None,
) -> Dict[str, Any]:
    return xiaozhi_take_photo(
        question="save latest fake photo",
        photo_name=photo_name,
        append_timestamp=append_timestamp,
        device_id=device_id,
        group_number=group_number,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
