#!/usr/bin/env python3
"""
Temporary fake device driver for end-to-end experiment flow validation.

This client connects to the local xiaozhi websocket server, answers via
listen-text messages, and emulates the device-side MCP camera/screen tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import websockets
from PIL import Image, ImageDraw


SERVER_WS = "ws://127.0.0.1:8000/xiaozhi/v1/"
DATA_ROOT = Path(
    r"C:\\Users\\11979\\Documents\\GitHub\\codex_edu\\lab_runs\\exp1_AgNPs_synthesis\\data"
)
DEVICE_ID = "e2e_uvvis_autodrive_20260504"
CLIENT_ID = "codex_e2e"
INITIAL_TURN_SENT = False


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_text(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _device_dir() -> Path:
    path = DATA_ROOT / DEVICE_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def _user_utterance_log_path() -> Path:
    return _device_dir() / f"{DEVICE_ID}_user_utterances.jsonl"


def _has_existing_user_turns() -> bool:
    path = _user_utterance_log_path()
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _pick_color_from_question(question: str):
    q = question or ""
    if "1号" in q or "一号" in q:
        return ("#0e1b52", "1号样品深蓝")
    if "2号" in q or "二号" in q:
        return ("#5a4db7", "2号样品蓝紫")
    if "3号" in q or "三号" in q:
        return ("#c9332c", "3号样品红色")
    if "4号" in q or "四号" in q:
        return ("#e08a1d", "4号样品橙色")
    if "5号" in q or "五号" in q:
        return ("#f0c419", "5号样品明黄")
    return ("#b5b5b5", "模拟拍照")


def _create_photo(question: str) -> Path:
    color, label = _pick_color_from_question(question)
    out_dir = _device_dir()
    filename = f"{_now_stamp()}_{uuid.uuid4().hex[:8]}.png"
    out_path = out_dir / filename

    img = Image.new("RGB", (960, 720), "#e9ecef")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((120, 70, 840, 650), radius=28, fill=color, outline="#222222", width=6)
    draw.rounded_rectangle((350, 120, 610, 600), radius=20, fill="#ffffff", outline="#1f1f1f", width=4)
    draw.rectangle((380, 150, 580, 570), fill=color)
    draw.text((140, 20), label, fill="#111111")
    draw.text((140, 665), question[:80] if question else "模拟照片", fill="#111111")
    img.save(out_path)
    return out_path


async def _send_json(ws, payload: dict):
    raw = json.dumps(payload, ensure_ascii=False)
    await ws.send(raw)
    print(f"[send] {raw}")


async def _send_listen_text(ws, text: str):
    payload = {
        "type": "listen",
        "state": "detect",
        "text": text,
        "mode": "manual",
    }
    await _send_json(ws, payload)


def _extract_step_hint(text: str) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if "丁达尔" in compact:
        return "tyndall"
    if "准备好开始" in compact or "准备好了" in compact or "开始了吗" in compact:
        return "start"
    if "可以拍照吗" in compact or "现在可以拍照" in compact or "能拍照吗" in compact:
        return "photo_confirm"
    if "装入比色皿" in compact or "样品位" in compact or "参比位" in compact:
        return "uvvis_load"
    if "纯水" in compact or "空白" in compact:
        return "blank"
    if "拍照" in compact or "颜色" in compact:
        if "1号" in compact or "一号" in compact:
            return "sample1_color"
        if "2号" in compact or "二号" in compact:
            return "sample2_color"
        if "3号" in compact or "三号" in compact:
            return "sample3_color"
        if "4号" in compact or "四号" in compact:
            return "sample4_color"
        if "5号" in compact or "五号" in compact:
            return "sample5_color"
        return "photo"
    if "暗电流" in compact or "空气能量" in compact:
        return "uvvis_prewarm"
    if "λmax" in compact or "lambda" in compact or "最大吸收峰" in compact:
        return "uvvis_result"
    if "光谱" in compact or "测量" in compact or "波长" in compact:
        return "uvvis_measure"
    if "加入" in compact or "加液" in compact or "混匀" in compact or "搅拌" in compact:
        return "prep"
    return "generic"


def _reply_for_assistant_text(text: str) -> Optional[str]:
    hint = _extract_step_hint(text)
    if hint == "start":
        return "准备好了。"
    if hint == "tyndall":
        return "看到了明显丁达尔现象。"
    if hint == "photo_confirm":
        return "可以拍照了。"
    if hint == "blank":
        return "已经放好了。"
    if hint == "sample1_color":
        return "看到了，1号是深蓝色。"
    if hint == "sample2_color":
        return "看到了，2号是蓝紫色。"
    if hint == "sample3_color":
        return "看到了，3号是红色。"
    if hint == "sample4_color":
        return "看到了，4号是橙色。"
    if hint == "sample5_color":
        return "看到了，5号是明黄色。"
    if hint == "photo":
        return "已经拍好了。"
    if hint == "uvvis_prewarm":
        return None
    if hint == "uvvis_load":
        return "已经放好了，可以开始了。"
    if hint == "uvvis_measure":
        return "可以开始测量了。"
    if hint == "uvvis_result":
        return "已经记下来了。"
    if hint == "prep":
        return "已经做好了。"
    return "已经做好了。"


def _should_early_reply(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    markers = (
        "准备好开始了吗",
        "做好后告诉我",
        "做好了告诉我",
        "加完后告诉我",
        "完成后告诉我",
        "放好了告诉我",
        "可以开始了",
    )
    return any(marker in compact for marker in markers)


async def _respond_to_turn(ws, assistant_text: str):
    reply = _reply_for_assistant_text(assistant_text)
    if not reply:
        return
    await asyncio.sleep(0.8)
    print(f"[auto-reply] {reply}")
    await _send_listen_text(ws, reply)


async def _send_initial_user_turn(ws):
    await asyncio.sleep(1.5)
    print("[auto-reply] 开始今天的实验。")
    await _send_listen_text(ws, "开始今天的实验。")


async def _handle_mcp_message(ws, payload: dict):
    global INITIAL_TURN_SENT
    msg_id = int(payload.get("id", 0) or 0)
    method = payload.get("method", "")
    params = payload.get("params") or {}

    if method == "initialize":
        result = {
            "serverInfo": {"name": "TmpUvvisAutoDriver", "version": "1.0.0"},
            "capabilities": {},
        }
        await _send_json(
            ws,
            {
                "type": "mcp",
                "payload": {"jsonrpc": "2.0", "id": msg_id, "result": result},
            },
        )
        return

    if method == "tools/list":
        tools = [
            {
                "name": "self.camera.take_photo",
                "description": "Take a simulated photo.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
            {
                "name": "self.screen.preview_image",
                "description": "Preview a simulated image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        ]
        await _send_json(
            ws,
            {
                "type": "mcp",
                "payload": {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": tools},
                },
            },
        )
        if not INITIAL_TURN_SENT and not _has_existing_user_turns():
            INITIAL_TURN_SENT = True
            asyncio.create_task(_send_initial_user_turn(ws))
        return

    if method == "tools/call":
        name = str(params.get("name", "")).strip()
        arguments = params.get("arguments") or {}
        if name == "self.camera.take_photo":
            question = _safe_text(arguments.get("question", ""))
            photo_path = _create_photo(question)
            text = json.dumps(
                {
                    "success": True,
                    "photo_path": str(photo_path),
                    "device_id": DEVICE_ID,
                },
                ensure_ascii=False,
            )
        elif name == "self.screen.preview_image":
            text = json.dumps(
                {
                    "success": True,
                    "previewed": True,
                    "device_id": DEVICE_ID,
                    "url": _safe_text(arguments.get("url", "")),
                },
                ensure_ascii=False,
            )
        else:
            text = json.dumps(
                {"success": False, "error": f"unsupported tool: {name}"},
                ensure_ascii=False,
            )

        await _send_json(
            ws,
            {
                "type": "mcp",
                "payload": {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                },
            },
        )
        return


async def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    uri = f"{SERVER_WS}?device-id={DEVICE_ID}&client-id={CLIENT_ID}"
    print(f"[info] connecting {uri}")

    async with websockets.connect(uri, ping_interval=20, ping_timeout=180) as ws:
        session_id = str(uuid.uuid4())
        hello = {
            "type": "hello",
            "version": 1,
            "transport": "websocket",
            "session_id": session_id,
            "user_id": "test",
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
            "features": {"mcp": True},
        }
        await _send_json(ws, hello)

        assistant_parts: list[str] = []
        pending_task: Optional[asyncio.Task] = None
        responded_this_turn = False

        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                print(f"[recv][binary] len={len(raw)}")
                continue

            print(f"[recv] {raw}")
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            msg_type = obj.get("type")
            if msg_type == "mcp":
                await _handle_mcp_message(ws, obj.get("payload") or {})
                continue

            if msg_type == "tts":
                state = obj.get("state")
                text = _safe_text(obj.get("text"))
                if text and state in {"start", "sentence_start"}:
                    assistant_parts.append(text)
                    if (
                        not responded_this_turn
                        and _should_early_reply(" ".join(assistant_parts))
                    ):
                        if pending_task and not pending_task.done():
                            pending_task.cancel()
                        pending_task = asyncio.create_task(
                            _respond_to_turn(ws, " ".join(assistant_parts).strip())
                        )
                        responded_this_turn = True
                elif state == "stop":
                    assistant_text = " ".join(assistant_parts).strip()
                    assistant_parts.clear()
                    if not responded_this_turn:
                        if pending_task and not pending_task.done():
                            pending_task.cancel()
                        pending_task = asyncio.create_task(
                            _respond_to_turn(ws, assistant_text)
                        )
                    responded_this_turn = False

        if pending_task:
            with contextlib.suppress(Exception):
                await pending_task


if __name__ == "__main__":
    asyncio.run(main())
