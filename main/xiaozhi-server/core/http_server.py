import asyncio

from aiohttp import web

from config.logger import setup_logging
from core.api.device_mcp_handler import DeviceMCPHandler
from core.api.ota_handler import OTAHandler
from core.api.vision_handler import VisionHandler

TAG = __name__


class SimpleHttpServer:
    def __init__(self, config: dict, ws_server=None):
        self.config = config
        self.ws_server = ws_server
        self.logger = setup_logging()
        self.ota_handler = OTAHandler(config)
        self.vision_handler = VisionHandler(config)
        self.device_mcp_handler = DeviceMCPHandler(config, ws_server)
        self._app = None
        self._runner = None
        self._site = None
        self._stop_event = None

    def _get_websocket_url(self, local_ip: str, port: int) -> str:
        server_config = self.config["server"]
        websocket_config = server_config.get("websocket")

        if websocket_config:
            return websocket_config
        return f"ws://{local_ip}:{port}/xiaozhi/v1/"

    async def start(self):
        try:
            server_config = self.config["server"]
            read_config_from_api = self.config.get("read_config_from_api", False)
            host = server_config.get("ip", "0.0.0.0")
            port = int(server_config.get("http_port", 8003))

            if port:
                app = web.Application()
                self._app = app

                if not read_config_from_api:
                    app.add_routes(
                        [
                            web.get("/xiaozhi/ota/", self.ota_handler.handle_get),
                            web.post("/xiaozhi/ota/", self.ota_handler.handle_post),
                            web.options(
                                "/xiaozhi/ota/", self.ota_handler.handle_options
                            ),
                            web.get(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_download,
                            ),
                            web.options(
                                "/xiaozhi/ota/download/{filename}",
                                self.ota_handler.handle_options,
                            ),
                        ]
                    )

                app.add_routes(
                    [
                        web.get("/mcp/vision/explain", self.vision_handler.handle_get),
                        web.post(
                            "/mcp/vision/explain",
                            self.vision_handler.handle_post,
                        ),
                        web.options(
                            "/mcp/vision/explain",
                            self.vision_handler.handle_options,
                        ),
                        web.get(
                            "/mcp/device/sessions",
                            self.device_mcp_handler.handle_get,
                        ),
                        web.post(
                            "/mcp/device/take_photo",
                            self.device_mcp_handler.handle_post,
                        ),
                        web.post(
                            "/mcp/device/preview_local_file",
                            self.device_mcp_handler.handle_preview_local_file_post,
                        ),
                        web.post(
                            "/mcp/device/call_tool",
                            self.device_mcp_handler.handle_call_tool_post,
                        ),
                        web.get(
                            "/mcp/device/local_files/{file_name}",
                            self.device_mcp_handler.handle_local_file_get,
                        ),
                        web.options(
                            "/mcp/device/sessions",
                            self.device_mcp_handler.handle_options,
                        ),
                        web.options(
                            "/mcp/device/take_photo",
                            self.device_mcp_handler.handle_options,
                        ),
                        web.options(
                            "/mcp/device/preview_local_file",
                            self.device_mcp_handler.handle_options,
                        ),
                        web.options(
                            "/mcp/device/call_tool",
                            self.device_mcp_handler.handle_options,
                        ),
                        web.options(
                            "/mcp/device/local_files/{file_name}",
                            self.device_mcp_handler.handle_options,
                        ),
                    ]
                )

                self._runner = web.AppRunner(app)
                await self._runner.setup()
                self._site = web.TCPSite(self._runner, host, port)
                await self._site.start()

                self._stop_event = asyncio.Event()
                await self._stop_event.wait()
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"HTTP鏈嶅姟鍣ㄥ惎鍔ㄥけ璐? {e}")
            import traceback

            self.logger.bind(tag=TAG).error(f"閿欒鍫嗘爤: {traceback.format_exc()}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        stop_event = self._stop_event
        if stop_event is not None and not stop_event.is_set():
            stop_event.set()

        site = self._site
        runner = self._runner

        self._app = None
        self._site = None
        self._runner = None
        self._stop_event = None

        if site is not None:
            try:
                await site.stop()
            except Exception as exc:
                self.logger.bind(tag=TAG).warning(
                    f"HTTP server site stop failed: {exc}"
                )

        if runner is not None:
            try:
                await runner.cleanup()
            except Exception as exc:
                self.logger.bind(tag=TAG).warning(
                    f"HTTP server runner cleanup failed: {exc}"
                )
