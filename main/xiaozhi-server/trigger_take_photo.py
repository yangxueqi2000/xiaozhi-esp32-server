#!/usr/bin/env python3
"""
Trigger one device camera capture through app.py HTTP endpoints.

Functions:
  - list_sessions(base_url)
  - trigger_take_photo(base_url, ...)

CLI examples:
  python trigger_take_photo.py --server http://127.0.0.1:8003 --list-sessions
  python trigger_take_photo.py --server http://127.0.0.1:8003 --device-id 94:a9:90:28:e8:ec --question "请拍照" --photo-name room_a --time-format "%Y-%m-%d_%H-%M-%S"
"""

import argparse
import json
from datetime import datetime
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT = 20
DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT = 35


def _request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except Exception:
            return {"success": False, "message": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def list_sessions(base_url: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/mcp/device/sessions"
    return _request_json("GET", url, timeout=timeout)


def build_photo_name(base_name: str, time_format: str = "%Y%m%d_%H%M%S") -> str:
    name = str(base_name or "").strip()
    if not name:
        return ""
    try:
        timestamp = datetime.now().strftime(time_format)
    except Exception as e:
        raise ValueError(f"invalid --time-format: {time_format}, error: {e}")
    return f"{name}_{timestamp}" if timestamp else name


def build_question_with_photo_name(question: str, photo_name: str) -> str:
    q = str(question or "").strip()
    if not photo_name:
        return q
    meta = json.dumps({"photo_name": photo_name}, ensure_ascii=False, separators=(",", ":"))
    if q:
        return f"{q}\n[XIAOZHI_META]{meta}"
    return f"[XIAOZHI_META]{meta}"


def trigger_take_photo(
    base_url: str,
    *,
    session_id: str = "",
    device_id: str = "",
    question: str = "Please take a photo.",
    photo_name: str = "",
    tool_name: str = "self.camera.take_photo",
    tool_timeout: int = DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT,
    request_timeout: int = DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    if not session_id and not device_id:
        raise ValueError("session_id or device_id is required")

    safe_tool_timeout = int(tool_timeout)
    if safe_tool_timeout <= 0:
        safe_tool_timeout = DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT

    safe_request_timeout = int(request_timeout)
    if safe_request_timeout <= 0:
        safe_request_timeout = DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT
    safe_request_timeout = max(
        safe_request_timeout,
        safe_tool_timeout + 5,
    )

    payload: Dict[str, Any] = {
        "question": question,
        "timeout": safe_tool_timeout,
        "tool_name": tool_name,
    }
    if session_id:
        payload["session_id"] = session_id
    if device_id:
        payload["device_id"] = device_id
    if photo_name:
        payload["photo_name"] = photo_name

    url = f"{base_url.rstrip('/')}/mcp/device/take_photo"
    return _request_json("POST", url, payload=payload, timeout=safe_request_timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger one camera capture via app.py")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8003",
        help="HTTP server base URL, e.g. http://127.0.0.1:8003",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Only list current online sessions",
    )
    parser.add_argument("--session-id", default="", help="Target session_id")
    parser.add_argument("--device-id", default="", help="Target device_id")
    parser.add_argument(
        "--question",
        default="Please take a photo.",
        help="Question passed to self.camera.take_photo",
    )
    parser.add_argument(
        "--photo-name",
        default="",
        help="Base photo name. Current time will be auto-appended.",
    )
    parser.add_argument(
        "--time-format",
        default="%Y%m%d_%H%M%S",
        help="strftime format for timestamp appended to --photo-name",
    )
    parser.add_argument(
        "--tool-name",
        default="self.camera.take_photo",
        help="Tool name to call (raw name, server will sanitize)",
    )
    parser.add_argument(
        "--tool-timeout",
        type=int,
        default=DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT,
        help="Tool call timeout seconds (sent to server)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT,
        help="HTTP request timeout seconds",
    )
    parser.add_argument(
        "--auto-pick-first",
        action="store_true",
        help="If no session/device provided, pick first online session",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_sessions:
        result = list_sessions(args.server, timeout=args.request_timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1

    session_id = args.session_id.strip()
    device_id = args.device_id.strip()
    photo_name = args.photo_name.strip()

    if not session_id and not device_id and args.auto_pick_first:
        sessions = list_sessions(args.server, timeout=args.request_timeout)
        if not sessions.get("success"):
            print(json.dumps(sessions, ensure_ascii=False, indent=2))
            return 1
        items = sessions.get("connections", [])
        if not items:
            print(
                json.dumps(
                    {"success": False, "message": "no online sessions"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        first = items[0]
        session_id = str(first.get("session_id", "")).strip()
        device_id = str(first.get("device_id", "")).strip()

    if photo_name:
        try:
            photo_name = build_photo_name(photo_name, args.time_format)
        except ValueError as e:
            print(json.dumps({"success": False, "message": str(e)}, ensure_ascii=False))
            return 1

    question = build_question_with_photo_name(args.question, photo_name)

    try:
        result = trigger_take_photo(
            args.server,
            session_id=session_id,
            device_id=device_id,
            question=question,
            photo_name=photo_name,
            tool_name=args.tool_name,
            tool_timeout=args.tool_timeout,
            request_timeout=args.request_timeout,
        )
    except ValueError as e:
        print(json.dumps({"success": False, "message": str(e)}, ensure_ascii=False))
        return 1

    if photo_name and isinstance(result, dict):
        result["requested_photo_name"] = photo_name

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
