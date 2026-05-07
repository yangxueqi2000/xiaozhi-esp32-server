from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_UVVIS_ROOT = Path(r"C:\Users\11979\Documents\GitHub\xiaozhi_uv_edu")
DEFAULT_SERVER = "http://127.0.0.1:8765"
DEFAULT_MCP_URL = "http://127.0.0.1:8766/mcp"
DEFAULT_MCP_PORT = 8766
DEFAULT_TOOL_PROFILE = "minimal"
DEFAULT_STARTUP_TIMEOUT = 45.0
DEFAULT_MCP_PROTOCOL_VERSION = "2025-03-26"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensure the UV-Vis streamable-http MCP server is running."
    )
    parser.add_argument("--uvvis-root", default=str(DEFAULT_UVVIS_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--default-port", default="COM7")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)
    parser.add_argument("--tool-profile", default=DEFAULT_TOOL_PROFILE)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional wrapper log file. Defaults to <uvvis-root>/data/uvvis_http_mcp.log",
    )
    return parser


def _probe_initialize_payload() -> bytes:
    payload = {
        "jsonrpc": "2.0",
        "id": "uvvis-ready-probe",
        "method": "initialize",
        "params": {
            "protocolVersion": DEFAULT_MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "ensure_uvvis_http_mcp",
                "version": "1.0",
            },
        },
    }
    return json.dumps(payload).encode("utf-8")


def _extract_probe_session_id(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    return str(headers.get("mcp-session-id", "") or "").strip()


def _close_probe_session(url: str, session_id: str, *, timeout: float) -> None:
    session_id = str(session_id or "").strip()
    if not session_id:
        return

    request = urllib.request.Request(
        url,
        method="DELETE",
        headers={"mcp-session-id": session_id},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except Exception:
        return


def _http_endpoint_ready(url: str, *, timeout: float = 1.5) -> bool:
    # Avoid probing streamable-http MCP with a bare GET /mcp. FastMCP rejects that
    # as 406 and may still allocate a transport/session before the header check.
    request = urllib.request.Request(
        url,
        data=_probe_initialize_payload(),
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            session_id = _extract_probe_session_id(response)
            try:
                response.read()
            finally:
                _close_probe_session(url, session_id, timeout=timeout)
            return True
    except urllib.error.HTTPError as exc:
        session_id = _extract_probe_session_id(exc)
        try:
            exc.read()
        except Exception:
            pass
        _close_probe_session(url, session_id, timeout=timeout)
        return False
    except Exception:
        return False


def _wait_for_http_endpoint(url: str, *, timeout: float) -> bool:
    deadline = time.monotonic() + max(1.0, float(timeout))
    while time.monotonic() < deadline:
        if _http_endpoint_ready(url):
            return True
        time.sleep(0.5)
    return _http_endpoint_ready(url)


def _load_process_rows() -> list[dict[str, Any]]:
    if os.name == "nt":
        command = (
            "$ErrorActionPreference='Stop';"
            "$rows = Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'uvvis_(http_wrapper|mcp_server)\\.py' } | "
            "Select-Object "
            "@{Name='pid';Expression={[int]$_.ProcessId}},"
            "@{Name='parent_pid';Expression={[int]$_.ParentProcessId}},"
            "@{Name='command_line';Expression={[string]$_.CommandLine}};"
            "$rows | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return []
        payload = completed.stdout.strip()
        if not payload:
            return []
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        return []

    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            parent_pid = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "parent_pid": parent_pid,
                "command_line": parts[2],
            }
        )
    return rows


def _normalized_command_line(command_line: str) -> str:
    return " ".join(str(command_line or "").lower().split())


def _is_conflicting_uvvis_process(row: dict[str, Any], *, desired_mcp_port: int) -> bool:
    command_line = _normalized_command_line(row.get("command_line", ""))
    if "uvvis_http_wrapper.py" in command_line:
        return "--transport stdio" in command_line
    if "uvvis_mcp_server.py" not in command_line:
        return False
    if "--transport stdio" in command_line:
        return True
    if "--transport streamable-http" in command_line:
        return f"--port {desired_mcp_port}" not in command_line
    return True


def _terminate_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _cleanup_conflicting_uvvis_processes(*, desired_mcp_port: int) -> None:
    for row in _load_process_rows():
        try:
            pid = int(row.get("pid", 0))
        except (TypeError, ValueError):
            continue
        if pid == os.getpid():
            continue
        if not _is_conflicting_uvvis_process(row, desired_mcp_port=desired_mcp_port):
            continue
        _terminate_process_tree(pid)


def _start_http_wrapper(
    *,
    python_executable: str,
    uvvis_root: Path,
    server: str,
    default_port: str,
    mcp_port: int,
    tool_profile: str,
    log_level: str,
    startup_timeout: float,
    log_file: Path,
) -> None:
    wrapper_path = uvvis_root / "uvvis_http_wrapper.py"
    if not wrapper_path.exists():
        raise FileNotFoundError(f"uvvis_http_wrapper.py was not found at {wrapper_path}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )

    command = [
        python_executable,
        str(wrapper_path),
        "--server",
        server,
        "--default-port",
        default_port,
        "--transport",
        "streamable-http",
        "--log-level",
        log_level,
        "--tool-profile",
        tool_profile,
        "--startup-timeout",
        str(startup_timeout),
        "--port",
        str(mcp_port),
        "--streamable-http-path",
        "/mcp",
    ]

    with log_file.open("ab") as handle:
        subprocess.Popen(
            command,
            cwd=str(uvvis_root),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            creationflags=creationflags,
            close_fds=True,
        )


def main() -> int:
    args = _build_parser().parse_args()
    uvvis_root = Path(args.uvvis_root).expanduser().resolve()
    log_file = (
        Path(args.log_file).expanduser().resolve()
        if str(args.log_file or "").strip()
        else (uvvis_root / "data" / "uvvis_http_mcp.log").resolve()
    )

    if _http_endpoint_ready(args.mcp_url):
        return 0

    _cleanup_conflicting_uvvis_processes(desired_mcp_port=args.mcp_port)

    if _http_endpoint_ready(args.mcp_url):
        return 0

    _start_http_wrapper(
        python_executable=args.python,
        uvvis_root=uvvis_root,
        server=args.server,
        default_port=args.default_port,
        mcp_port=args.mcp_port,
        tool_profile=args.tool_profile,
        log_level=args.log_level,
        startup_timeout=args.startup_timeout,
        log_file=log_file,
    )

    if _wait_for_http_endpoint(args.mcp_url, timeout=args.startup_timeout):
        return 0

    raise RuntimeError(
        f"uvvis streamable-http MCP server did not become ready at {args.mcp_url} "
        f"within {args.startup_timeout:.1f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())
