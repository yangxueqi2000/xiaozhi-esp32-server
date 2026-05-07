import asyncio
import signal
import sys
import uuid

from aioconsole import ainput

from config.logger import setup_logging
from config.settings import load_config
from core.http_server import SimpleHttpServer
from core.utils.gc_manager import get_gc_manager
from core.utils.util import (
    check_ffmpeg_installed,
    get_local_ip,
    normalize_mcp_endpoint_for_ws,
)
from core.websocket_server import WebSocketServer

TAG = __name__
logger = setup_logging()


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


async def monitor_stdin():
    while True:
        await ainput()


async def main():
    check_ffmpeg_installed()
    config = load_config()

    auth_key = config["server"].get("auth_key", "")
    if not auth_key or "浣?" in auth_key:
        auth_key = config.get("manager-api", {}).get("secret", "")
        if not auth_key or "浣?" in auth_key:
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
    if not read_config_from_api:
        logger.bind(tag=TAG).info(
            "OTA鎺ュ彛鏄痋t\thttp://{}:{}/xiaozhi/ota/",
            get_local_ip(),
            port,
        )
    logger.bind(tag=TAG).info(
        "瑙嗚鍒嗘瀽鎺ュ彛鏄痋thttp://{}:{}/mcp/vision/explain",
        get_local_ip(),
        port,
    )
    logger.bind(tag=TAG).info(
        "Take photo trigger URL is\thttp://{}:{}/mcp/device/take_photo",
        get_local_ip(),
        port,
    )
    logger.bind(tag=TAG).info(
        "Preview local file URL is\thttp://{}:{}/mcp/device/preview_local_file",
        get_local_ip(),
        port,
    )
    logger.bind(tag=TAG).info(
        "Session list URL is\thttp://{}:{}/mcp/device/sessions",
        get_local_ip(),
        port,
    )

    mcp_endpoint = str(config.get("mcp_endpoint") or "").strip()
    if mcp_endpoint:
        normalized_mcp_endpoint = normalize_mcp_endpoint_for_ws(mcp_endpoint)
        if normalized_mcp_endpoint:
            logger.bind(tag=TAG).info("mcp鎺ュ叆鐐规槸\t{}", mcp_endpoint)
            config["mcp_endpoint"] = normalized_mcp_endpoint
        else:
            logger.bind(tag=TAG).warning("mcp鎺ュ叆鐐逛笉绗﹀悎瑙勮寖")
            config["mcp_endpoint"] = ""

    websocket_port = 8000
    server_config = config.get("server", {})
    if isinstance(server_config, dict):
        websocket_port = int(server_config.get("port", 8000))

    logger.bind(tag=TAG).info(
        "Websocket鍦板潃鏄痋tws://{}:{}/xiaozhi/v1/",
        get_local_ip(),
        websocket_port,
    )
    logger.bind(tag=TAG).info(
        "=======涓婇潰鐨勫湴鍧€鏄痺ebsocket鍗忚鍦板潃锛岃鍕跨敤娴忚鍣ㄨ闂?======"
    )
    logger.bind(tag=TAG).info(
        "濡傛兂娴嬭瘯websocket璇风敤璋锋瓕娴忚鍣ㄦ墦寮€test鐩綍涓嬬殑test_page.html"
    )
    logger.bind(tag=TAG).info(
        "=============================================================\n"
    )

    try:
        await wait_for_exit()
    except asyncio.CancelledError:
        print("浠诲姟琚彇娑堬紝娓呯悊璧勬簮涓?..")
    finally:
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
        print("鏈嶅姟鍣ㄥ凡鍏抽棴锛岀▼搴忛€€鍑恒€?")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("鎵嬪姩涓柇锛岀▼搴忕粓姝€?")
