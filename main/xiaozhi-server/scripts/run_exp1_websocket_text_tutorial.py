from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import websockets


SERVER_ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _add_server_to_path() -> None:
    server_text = str(SERVER_ROOT)
    if server_text not in sys.path:
        sys.path.insert(0, server_text)


def _load_exp1_config() -> Dict[str, Any]:
    _add_server_to_path()
    from config.config_loader import (
        apply_config_path_templates,
        get_default_config_path,
        merge_configs,
        read_config,
    )

    default_config = read_config(get_default_config_path(), required=False)
    custom_config = read_config(str(SERVER_ROOT / "data" / ".config_exp1.yaml"), required=True)
    return apply_config_path_templates(merge_configs(default_config, custom_config))


def _safe_device_id(device_id: str) -> str:
    chars = []
    for ch in str(device_id or "").strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        elif ch == ":":
            chars.append("_")
        else:
            chars.append("_")
    return "".join(chars).strip("._-") or "unknown"


def _turns() -> List[str]:
    return [
        "开始实验。",
        "准备好了。",
        "1到5号烧杯都已经编号好了，每个烧杯里都放入了磁转子。",
        "1到5号样品都已经按顺序加入了1.00毫升柠檬酸钠。",
        "1到5号样品都已经按顺序加入了5.00毫升硝酸银。",
        "1到5号样品都已经按顺序加入了5.00毫升过氧化氢。",
        "1到5号样品都已经开始搅拌，并且混合均匀。",
        "1号样品已经加入KBr 0.00毫升和纯水2.90毫升，并混匀。",
        "1号样品已经快速加入硼氢化钠2.50毫升，颜色稳定为深蓝偏黑，用时3分钟。",
        "可以拍照。",
        "2号样品已经加入KBr 0.80毫升和纯水2.10毫升，并混匀。",
        "2号样品已经快速加入硼氢化钠2.50毫升，颜色稳定为蓝紫色，用时4分钟。",
        "可以拍照。",
        "3号样品已经加入KBr 1.20毫升和纯水1.70毫升，并混匀。",
        "3号样品已经快速加入硼氢化钠2.50毫升，颜色稳定为红色，用时5分钟。",
        "可以拍照。",
        "4号样品已经加入KBr 1.50毫升和纯水1.40毫升，并混匀。",
        "4号样品已经快速加入硼氢化钠2.50毫升，颜色稳定为橙色，用时6分钟。",
        "可以拍照。",
        "5号样品已经加入KBr 1.80毫升和纯水1.10毫升，并混匀。",
        "5号样品已经快速加入硼氢化钠2.50毫升，颜色稳定为明黄色，用时7分钟。",
        "可以拍照。",
        "5个样品都观察到了丁达尔现象，1号最强，5号最弱，强度从1号到5号逐渐减弱。",
        "实验结束，生成实验报告。",
    ]


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _extract_meta(question: str) -> Dict[str, Any]:
    marker = "[XIAOZHI_META]"
    if marker not in question:
        return {}
    text = question.split(marker, 1)[1].strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class WebsocketTutorialSmoke:
    def __init__(
        self,
        *,
        ws_url: str,
        device_id: str,
        user_id: str,
        data_root: Path,
        log_dir: Path,
        turn_timeout: float,
        stop_grace: float,
    ) -> None:
        self.ws_url = ws_url
        self.device_id = device_id
        self.user_id = user_id
        self.data_root = data_root
        self.log_dir = log_dir
        self.turn_timeout = turn_timeout
        self.stop_grace = stop_grace
        self.messages_path = log_dir / "messages.jsonl"
        self.fake_mcp_path = log_dir / "fake_device_mcp.jsonl"
        self.session_id = ""
        self._pending_turn: Optional[Dict[str, Any]] = None
        self._turn_done = asyncio.Event()
        self._turns: List[Dict[str, Any]] = []

    def _url(self) -> str:
        sep = "&" if "?" in self.ws_url else "?"
        query = urlencode({"device-id": self.device_id, "client-id": "codex-text-smoke"})
        return f"{self.ws_url}{sep}{query}"

    def _device_dir(self, group_number: Optional[int] = None) -> Path:
        path = self.data_root / _safe_device_id(self.device_id)
        if group_number:
            path = path / f"group_{int(group_number):02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_fake_photo(self, arguments: Dict[str, Any]) -> Path:
        question = str(arguments.get("question", "") or "")
        meta = _extract_meta(question)
        group_number = meta.get("group_number")
        try:
            group_number = int(group_number) if group_number else None
        except (TypeError, ValueError):
            group_number = None
        photo_name = str(meta.get("photo_name") or "").strip()
        if not photo_name:
            photo_name = "fake_photo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        photo_name = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", photo_name).strip("._-")
        if not photo_name:
            photo_name = "fake_photo"
        path = self._device_dir(group_number) / f"{photo_name}.png"
        path.write_bytes(PNG_1X1)
        return path

    async def _handle_mcp(self, ws, payload: Dict[str, Any]) -> None:
        method = payload.get("method")
        msg_id = payload.get("id")
        _append_jsonl(self.fake_mcp_path, {"direction": "server_to_fake_device", "payload": payload})

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "FakeXiaozhiDevice", "version": "1.0.0"},
                },
            }
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "self.camera.take_photo",
                            "description": "Capture a fake local photo for text-only smoke tests.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"question": {"type": "string"}},
                                "required": ["question"],
                            },
                        }
                    ]
                },
            }
        elif method == "tools/call":
            params = payload.get("params") or {}
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            if name == "self.camera.take_photo":
                photo_path = self._write_fake_photo(arguments if isinstance(arguments, dict) else {})
                text = json.dumps(
                    {
                        "action": "REQLLM",
                        "response": "fake photo captured",
                        "photo_path": str(photo_path),
                    },
                    ensure_ascii=False,
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"isError": True, "error": f"unknown fake tool: {name}"},
                }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"unknown method: {method}"},
            }

        _append_jsonl(self.fake_mcp_path, {"direction": "fake_device_to_server", "payload": response})
        await ws.send(json.dumps({"type": "mcp", "payload": response}, ensure_ascii=False))

    def _record_message(self, message: Any) -> None:
        _append_jsonl(
            self.messages_path,
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "message": message,
            },
        )

    async def _receiver(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                self._record_message({"type": "binary", "bytes": len(raw)})
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                self._record_message({"type": "text", "raw": raw})
                continue

            self._record_message(message)
            msg_type = message.get("type")
            if msg_type == "hello":
                self.session_id = str(message.get("session_id") or "")
                continue
            if msg_type == "mcp":
                payload = message.get("payload")
                if isinstance(payload, dict):
                    await self._handle_mcp(ws, payload)
                continue

            turn = self._pending_turn
            if turn is None:
                continue
            if msg_type == "stt":
                turn["stt"].append(str(message.get("text") or ""))
            elif msg_type == "tts":
                state = message.get("state")
                text = str(message.get("text") or "").strip()
                if text and state in {"start", "sentence_start"}:
                    if not turn["assistant_chunks"] or turn["assistant_chunks"][-1] != text:
                        turn["assistant_chunks"].append(text)
                if state == "stop":
                    turn["stop_seen"] = True
                    await asyncio.sleep(self.stop_grace)
                    self._turn_done.set()
            elif msg_type == "llm":
                text = str(message.get("text") or "").strip()
                if text:
                    turn["assistant_chunks"].append(text)

    async def run(self, turns: List[str]) -> Dict[str, Any]:
        uri = self._url()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = self.log_dir / "transcript.md"

        async with websockets.connect(uri, ping_interval=20, ping_timeout=180, max_size=None) as ws:
            receiver_task = asyncio.create_task(self._receiver(ws))
            hello = {
                "type": "hello",
                "device_id": self.device_id,
                "device_name": "Fake text smoke device",
                "device_mac": self.device_id,
                "user_id": self.user_id,
                "features": {"mcp": True},
            }
            await ws.send(json.dumps(hello, ensure_ascii=False))
            started_wait = time.time()
            while not self.session_id and time.time() - started_wait < 20:
                await asyncio.sleep(0.1)
            if not self.session_id:
                raise TimeoutError("hello response timed out")

            with transcript_path.open("w", encoding="utf-8") as transcript:
                transcript.write("# Exp1 WebSocket Text Tutorial Transcript\n\n")
                transcript.write(f"- device_id: `{self.device_id}`\n")
                transcript.write(f"- session_id: `{self.session_id}`\n\n")
                for idx, user_text in enumerate(turns, start=1):
                    turn = {
                        "turn": idx,
                        "user": user_text,
                        "stt": [],
                        "assistant_chunks": [],
                        "stop_seen": False,
                        "elapsed_seconds": None,
                        "error": "",
                    }
                    self._pending_turn = turn
                    self._turn_done.clear()
                    print(f"\n===== TURN {idx} USER =====\n{user_text}\n", flush=True)
                    transcript.write(f"## Turn {idx}\n\n**User:** {user_text}\n\n")
                    started = time.time()
                    await ws.send(
                        json.dumps(
                            {"type": "listen", "state": "detect", "text": user_text},
                            ensure_ascii=False,
                        )
                    )

                    try:
                        await asyncio.wait_for(self._turn_done.wait(), timeout=self.turn_timeout)
                    except asyncio.TimeoutError:
                        turn["error"] = "turn timeout waiting for tts stop"

                    turn["elapsed_seconds"] = round(time.time() - started, 3)
                    assistant_text = "\n".join(turn["assistant_chunks"]).strip()
                    print(f"===== TURN {idx} ASSISTANT =====\n{assistant_text}\n", flush=True)
                    transcript.write(f"**Assistant:** {assistant_text}\n\n")
                    self._turns.append(turn)
                    if turn["error"]:
                        transcript.write(f"**Error:** {turn['error']}\n\n")
                        break

            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

        return {
            "session_id": self.session_id,
            "completed_turns": len(self._turns),
            "turns": self._turns,
            "transcript_path": str(transcript_path),
        }


def _find_recent_outputs(device_dir: Path, start_time: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"device_dir": str(device_dir), "photos": [], "reports": []}
    if not device_dir.exists():
        return result
    for path in device_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < start_time - 2:
                continue
        except OSError:
            continue
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            result["photos"].append(str(path))
        elif suffix in {".yaml", ".yml", ".pdf"}:
            result["reports"].append(str(path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exp1 through the real Xiaozhi WebSocket text path.")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/xiaozhi/v1/")
    parser.add_argument("--device-id", default="codex-exp1-websocket-tutorial-smoke")
    parser.add_argument("--user-id", default="tutorial-test")
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--turn-timeout", type=float, default=240.0)
    parser.add_argument("--stop-grace", type=float, default=0.5)
    args = parser.parse_args()

    start_time = time.time()
    config = _load_exp1_config()
    data_root = Path(config["experiment_paths"]["experiment_data_root"]).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = data_root / args.device_id / f"tutorial_ws_smoke_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    turns = _turns()
    if args.max_turns and args.max_turns > 0:
        turns = turns[: args.max_turns]

    smoke = WebsocketTutorialSmoke(
        ws_url=args.ws_url,
        device_id=args.device_id,
        user_id=args.user_id,
        data_root=data_root,
        log_dir=log_dir,
        turn_timeout=args.turn_timeout,
        stop_grace=args.stop_grace,
    )
    summary: Dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device_id": args.device_id,
        "log_dir": str(log_dir),
        "turn_count_requested": len(turns),
    }
    try:
        summary.update(asyncio.run(smoke.run(turns)))
        status = 0
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        status = 1

    summary["outputs"] = _find_recent_outputs(data_root / _safe_device_id(args.device_id), start_time)
    summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (log_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
