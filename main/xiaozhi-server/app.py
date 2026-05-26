import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from aioconsole import ainput

from config.logger import setup_logging
from config.settings import load_config
from core.http_server import SimpleHttpServer
from core.providers.tools.server_mcp.mcp_manager import ServerMCPManager
from core.providers.tools.server_mcp.payload_utils import finalize_server_mcp_payload
from core.utils.gc_manager import get_gc_manager
from core.utils.util import (
    check_ffmpeg_installed,
    get_local_ip,
    normalize_mcp_endpoint_for_ws,
)
from core.websocket_server import WebSocketServer

TAG = __name__
logger = setup_logging()
_voiceprint_process = None
_voiceprint_log_handle = None
_startup_server_mcp_manager = None
_startup_uvvis_session_key = ""


def _looks_like_placeholder(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True

    text_lower = text.lower()
    return (
        "你的" in text
        or "浣犵殑" in text
        or "your" in text_lower
        or "example" in text_lower
    )


async def wait_for_exit() -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
        await stop_event.wait()
        return

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _voiceprint_health_url(config: dict) -> str:
    voiceprint_config = config.get("voiceprint", {}) or {}
    return str(voiceprint_config.get("url", "") or "").strip()


def _voiceprint_health_ok(url: str, timeout: float = 3.0) -> bool:
    if not url:
        return False
    try:
        with urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return False
            body = json.loads(response.read().decode("utf-8"))
            return body.get("status") == "healthy"
    except Exception:
        return False


def _default_voiceprint_root() -> str:
    github_root = Path(__file__).resolve().parents[2].parent
    return str(github_root / "voiceprint-api")


def _resolve_experiment_yaml_path_from_config(config: dict) -> str:
    llm_map = config.get("LLM", {}) or {}
    if not isinstance(llm_map, dict):
        return ""

    selected_name = str(config.get("selected_module", {}).get("LLM", "") or "").strip()
    llm_cfg = llm_map.get(selected_name)
    if not isinstance(llm_cfg, dict):
        for candidate in llm_map.values():
            if isinstance(candidate, dict) and str(candidate.get("type", "")).strip() == "codex":
                llm_cfg = candidate
                break
    if not isinstance(llm_cfg, dict):
        return ""

    workspace = str(llm_cfg.get("workspace", "") or "").strip()
    yaml_path = str(llm_cfg.get("yaml_path", "") or "").strip()
    if not yaml_path:
        return ""
    if os.path.isabs(yaml_path):
        return str(Path(yaml_path))
    if workspace:
        return str(Path(workspace) / yaml_path)
    return str(Path(yaml_path).resolve())


def _should_startup_prewarm_uvvis(config: dict) -> bool:
    yaml_path = _resolve_experiment_yaml_path_from_config(config).replace("\\", "/").lower()
    return "exp2_uv_vis_analysis" in yaml_path


def _startup_uvvis_log_path() -> Path:
    return Path(__file__).resolve().parent / "run_logs" / "startup_uvvis_prewarm.log"


def _append_startup_uvvis_log(event: str, **fields) -> None:
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": str(event or "").strip(),
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[str(key)] = value

    log_path = _startup_uvvis_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.bind(tag=TAG).warning(f"failed to append startup uvvis log: {exc}")


def _shutdown_uvvis_http_mcp_processes(*, reason: str = "") -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "ensure_uvvis_http_mcp.py"
    if not script_path.exists():
        return
    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--shutdown-only",
            ],
            cwd=str(script_path.parent.parent),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        _append_startup_uvvis_log(
            "shutdown_uvvis_chain",
            reason=reason or None,
        )
    except Exception as exc:
        _append_startup_uvvis_log(
            "shutdown_uvvis_chain_failed",
            reason=reason or None,
            error=str(exc),
        )
        logger.bind(tag=TAG).warning(f"shutdown uvvis chain failed: {exc}")


class _StartupMCPContext:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logger
        self.device_id = ""
        self.headers = {}
        self.experiment_yaml_path = _resolve_experiment_yaml_path_from_config(config)


async def ensure_startup_uvvis_session(config: dict):
    global _startup_server_mcp_manager, _startup_uvvis_session_key

    yaml_path = _resolve_experiment_yaml_path_from_config(config)
    if not _should_startup_prewarm_uvvis(config):
        _shutdown_uvvis_http_mcp_processes(reason="non_exp2_startup")
        _append_startup_uvvis_log(
            "startup_skip_non_exp2",
            yaml_path=yaml_path,
        )
        return None
    if _startup_server_mcp_manager is not None and _startup_uvvis_session_key:
        _append_startup_uvvis_log(
            "startup_already_active",
            yaml_path=yaml_path,
            session_key=_startup_uvvis_session_key,
        )
        logger.bind(tag=TAG).info(
            f"startup uvvis prewarm already active, session_key={_startup_uvvis_session_key}"
        )
        return _startup_server_mcp_manager

    _append_startup_uvvis_log(
        "startup_begin",
        yaml_path=yaml_path,
    )
    ctx = _StartupMCPContext(config)
    manager = ServerMCPManager(ctx)
    try:
        await manager.initialize_servers()
    except Exception as exc:
        _append_startup_uvvis_log(
            "startup_initialize_servers_failed",
            yaml_path=yaml_path,
            error=str(exc),
        )
        raise

    ready = await manager.ensure_client_initialized("uvvis")
    if not ready or not manager.is_mcp_tool("uvvis_session"):
        _append_startup_uvvis_log(
            "startup_tool_not_ready",
            yaml_path=yaml_path,
            ready=ready,
            has_uvvis_session_tool=manager.is_mcp_tool("uvvis_session"),
        )
        logger.bind(tag=TAG).warning("startup uvvis prewarm skipped: uvvis_session tool not ready")
        await manager.cleanup_all()
        return None

    _append_startup_uvvis_log(
        "startup_acquire_begin",
        yaml_path=yaml_path,
    )
    raw_result = await manager.execute_tool(
        "uvvis_session",
        {"action": "acquire"},
        priority="startup",
    )
    payload = finalize_server_mcp_payload(
        raw_result,
        tool_name="uvvis_session",
        arguments={"action": "acquire"},
    )
    body = payload.get("result") if isinstance(payload, dict) and isinstance(payload.get("result"), dict) else payload
    session_key = ""
    if isinstance(body, dict):
        session_key = str(body.get("session_key", "") or payload.get("session_key", "") or "").strip()
    if not session_key:
        message = ""
        if isinstance(body, dict):
            message = str(body.get("message", "") or payload.get("message", "") or "").strip()
        _append_startup_uvvis_log(
            "startup_acquire_no_session_key",
            yaml_path=yaml_path,
            message=message or "unknown",
            payload=payload,
        )
        logger.bind(tag=TAG).warning(
            f"startup uvvis prewarm acquire returned no session_key: {message or 'unknown'}"
        )
        await manager.cleanup_all()
        return None

    _startup_server_mcp_manager = manager
    _startup_uvvis_session_key = session_key
    _append_startup_uvvis_log(
        "startup_acquire_ok",
        yaml_path=yaml_path,
        session_key=session_key,
        payload=payload,
    )
    logger.bind(tag=TAG).info(
        f"startup uvvis prewarm acquired session_key={session_key}"
    )
    return manager


async def stop_startup_uvvis_session():
    global _startup_server_mcp_manager, _startup_uvvis_session_key

    manager = _startup_server_mcp_manager
    session_key = str(_startup_uvvis_session_key or "").strip()
    _startup_server_mcp_manager = None
    _startup_uvvis_session_key = ""
    if manager is None:
        return

    try:
        if session_key and manager.is_mcp_tool("uvvis_session"):
            _append_startup_uvvis_log(
                "shutdown_release_begin",
                session_key=session_key,
            )
            await manager.execute_tool(
                "uvvis_session",
                {"action": "release", "session_key": session_key},
                priority="background_common",
            )
            _append_startup_uvvis_log(
                "shutdown_release_ok",
                session_key=session_key,
            )
    except Exception as exc:
        _append_startup_uvvis_log(
            "shutdown_release_failed",
            session_key=session_key,
            error=str(exc),
        )
        logger.bind(tag=TAG).warning(f"startup uvvis prewarm release failed: {exc}")
    finally:
        await manager.cleanup_all()


async def ensure_voiceprint_service(config: dict):
    """Start local voiceprint-api when the experiment config enables voiceprint."""
    global _voiceprint_process, _voiceprint_log_handle

    voiceprint_config = config.get("voiceprint", {}) or {}
    if not _as_bool(voiceprint_config.get("enabled"), default=False):
        logger.bind(tag=TAG).info("声纹识别总开关已关闭，不启动 voiceprint-api")
        return None

    service_config = voiceprint_config.get("service", {}) or {}
    if not _as_bool(service_config.get("auto_start"), default=True):
        logger.bind(tag=TAG).info("voiceprint-api 自动启动已关闭")
        return None

    health_url = _voiceprint_health_url(config)
    if _voiceprint_health_ok(health_url):
        logger.bind(tag=TAG).info("voiceprint-api 已在运行，健康检查通过")
        return None

    root = Path(str(service_config.get("root") or _default_voiceprint_root())).resolve()
    start_script = str(service_config.get("start_script") or "start_server.py").strip()
    script_path = root / start_script
    python_path = str(service_config.get("python") or sys.executable).strip()
    startup_timeout = float(service_config.get("startup_timeout", 120) or 120)
    log_file = Path(
        str(
            service_config.get("log_file")
            or (Path(__file__).resolve().parent / "tmp" / "voiceprint_api_stdout.log")
        )
    )

    if not script_path.exists():
        logger.bind(tag=TAG).warning(
            f"voiceprint-api 启动脚本不存在: {script_path}"
        )
        return None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    _voiceprint_log_handle = open(log_file, "a", encoding="utf-8", buffering=1)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    logger.bind(tag=TAG).info(
        "启动 voiceprint-api: "
        f"python={python_path}, script={script_path}, health={health_url}"
    )
    _voiceprint_process = await asyncio.create_subprocess_exec(
        python_path,
        str(script_path),
        cwd=str(root),
        stdout=_voiceprint_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )

    deadline = asyncio.get_running_loop().time() + max(startup_timeout, 1.0)
    while asyncio.get_running_loop().time() < deadline:
        if _voiceprint_health_ok(health_url):
            logger.bind(tag=TAG).info("voiceprint-api 启动成功，健康检查通过")
            return _voiceprint_process
        if _voiceprint_process.returncode is not None:
            logger.bind(tag=TAG).warning(
                "voiceprint-api 进程提前退出: "
                f"returncode={_voiceprint_process.returncode}, log={log_file}"
            )
            return _voiceprint_process
        await asyncio.sleep(1.0)

    logger.bind(tag=TAG).warning(
        "voiceprint-api 启动后健康检查仍未通过: "
        f"timeout={startup_timeout}s, log={log_file}"
    )
    return _voiceprint_process


async def stop_managed_voiceprint_service():
    global _voiceprint_process, _voiceprint_log_handle

    process = _voiceprint_process
    _voiceprint_process = None
    if process is not None and process.returncode is None:
        logger.bind(tag=TAG).info("停止由 xiaozhi 启动的 voiceprint-api")
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=8.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        except Exception as exc:
            logger.bind(tag=TAG).warning(f"停止 voiceprint-api 失败: {exc}")

    if _voiceprint_log_handle is not None:
        try:
            _voiceprint_log_handle.close()
        except Exception:
            pass
        _voiceprint_log_handle = None


async def monitor_stdin():
    while True:
        try:
            await ainput()
        except (EOFError, OSError):
            await asyncio.Future()


async def main():
    check_ffmpeg_installed()
    config = load_config()
    await ensure_voiceprint_service(config)
    await ensure_startup_uvvis_session(config)

    auth_key = config["server"].get("auth_key", "")
    if _looks_like_placeholder(auth_key):
        auth_key = config.get("manager-api", {}).get("secret", "")
        if _looks_like_placeholder(auth_key):
            auth_key = str(uuid.uuid4().hex)
    config["server"]["auth_key"] = auth_key

    stdin_task = asyncio.create_task(monitor_stdin())

    gc_manager = get_gc_manager(interval_seconds=300)
    await gc_manager.start()

    ws_server = WebSocketServer(config)
    ws_task = asyncio.create_task(ws_server.start())

    ota_server = SimpleHttpServer(config, ws_server)
    ota_task = asyncio.create_task(ota_server.start())

    read_config_from_api = config.get("read_config_from_api", False)
    port = int(config["server"].get("http_port", 8003))
    local_ip = get_local_ip()
    if not read_config_from_api:
        logger.bind(tag=TAG).info(
            "OTA接口是\t{}",
            f"http://{local_ip}:{port}/xiaozhi/ota/",
        )
    logger.bind(tag=TAG).info(
        "视觉分析接口是\t{}",
        f"http://{local_ip}:{port}/mcp/vision/explain",
    )
    logger.bind(tag=TAG).info(
        "Take photo trigger URL is\t{}",
        f"http://{local_ip}:{port}/mcp/device/take_photo",
    )
    logger.bind(tag=TAG).info(
        "Preview local file URL is\t{}",
        f"http://{local_ip}:{port}/mcp/device/preview_local_file",
    )
    logger.bind(tag=TAG).info(
        "Session list URL is\t{}",
        f"http://{local_ip}:{port}/mcp/device/sessions",
    )

    mcp_endpoint = str(config.get("mcp_endpoint") or "").strip()
    if mcp_endpoint:
        normalized_mcp_endpoint = normalize_mcp_endpoint_for_ws(mcp_endpoint)
        if normalized_mcp_endpoint:
            logger.bind(tag=TAG).info("MCP接入点是\t{}", mcp_endpoint)
            config["mcp_endpoint"] = normalized_mcp_endpoint
        else:
            logger.bind(tag=TAG).warning("MCP接入点不符合规范")
            config["mcp_endpoint"] = ""

    websocket_port = 8000
    server_config = config.get("server", {})
    websocket_url = ""
    if isinstance(server_config, dict):
        websocket_port = int(server_config.get("port", 8000))
        websocket_config = str(server_config.get("websocket", "") or "").strip()
        if websocket_config.startswith(("ws://", "wss://")):
            websocket_url = websocket_config
    if not websocket_url:
        websocket_url = f"ws://{local_ip}:{websocket_port}/xiaozhi/v1/"

    logger.bind(tag=TAG).info(
        "WebSocket地址是\t{}",
        websocket_url,
    )
    logger.bind(tag=TAG).info(
        "=======上面的地址是 WebSocket 协议地址，请勿用浏览器访问======"
    )
    logger.bind(tag=TAG).info(
        "如想测试 WebSocket，请用浏览器打开 test 目录下的 test_page.html"
    )
    logger.bind(tag=TAG).info(
        "=============================================================\n"
    )

    try:
        await wait_for_exit()
    except asyncio.CancelledError:
        print("任务被取消，正在清理资源...")
    finally:
        await stop_startup_uvvis_session()
        await stop_managed_voiceprint_service()
        await gc_manager.stop()

        try:
            await ota_server.stop()
        except Exception as shutdown_error:
            logger.bind(tag=TAG).error(
                f"HTTP server shutdown cleanup failed: {shutdown_error}"
            )

        try:
            await ws_server.stop()
        except Exception as shutdown_error:
            logger.bind(tag=TAG).error(
                f"WebSocket server shutdown cleanup failed: {shutdown_error}"
            )

        stdin_task.cancel()
        ws_task.cancel()
        ota_task.cancel()

        await asyncio.wait(
            [stdin_task, ws_task, ota_task],
            timeout=3.0,
            return_when=asyncio.ALL_COMPLETED,
        )
        print("服务器已关闭，程序退出。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("手动中断，程序终止。")
