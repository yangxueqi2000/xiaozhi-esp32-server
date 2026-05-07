import json
import copy
import os
import time
import uuid
import subprocess
import tempfile
from aiohttp import web
from config.logger import setup_logging
from core.api.base_handler import BaseHandler
from core.utils.util import get_vision_url, is_valid_image_file
from core.utils.vllm import create_instance
from config.config_loader import get_private_config_from_api
from core.utils.auth import AuthToken
import base64
from typing import Tuple, Optional
from plugins_func.register import Action

TAG = __name__

# 设置最大文件大小为5MB
MAX_FILE_SIZE = 5 * 1024 * 1024


class VisionHandler(BaseHandler):
    def __init__(self, config: dict):
        super().__init__(config)
        # 初始化认证工具
        self.auth = AuthToken(config["server"]["auth_key"])
        self._question_meta_prefix = "[XIAOZHI_META]"

    def _guess_image_ext(self, data: bytes) -> str:
        if data.startswith(b"\xff\xd8\xff"):
            return "jpg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "gif"
        if data.startswith(b"BM"):
            return "bmp"
        if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
            return "tiff"
        if data.startswith(b"RIFF"):
            return "webp"
        return "jpg"

    def _sanitize_filename_stem(self, stem: str) -> str:
        # Keep user-provided names safe for filesystem paths.
        val = str(stem or "").strip()
        if not val:
            return ""
        val = os.path.splitext(val)[0]
        for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            val = val.replace(ch, "_")
        val = val.strip(" .")
        if not val:
            return ""
        return val[:120]

    def _sanitize_device_for_path(self, device_id: str) -> str:
        value = str(device_id or "").strip()
        if not value:
            return "unknown"
        chars = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_", "."):
                chars.append(ch)
                continue
            if ch == ":":
                chars.append("_")
                continue
            chars.append("_")
        safe = "".join(chars).strip("._-")
        return safe or "unknown"

    def _normalize_group_number(self, value) -> Optional[int]:
        if isinstance(value, bool) or value in (None, ""):
            return None
        try:
            group_number = int(value)
        except (TypeError, ValueError):
            return None
        return group_number if group_number >= 1 else None

    def _format_group_dir_name(self, group_number: int) -> str:
        return f"group_{int(group_number):02d}"

    def _derive_experiment_data_root(self) -> str:
        cfg = self.config or {}
        llm_map = cfg.get("LLM") or {}
        selected = str((cfg.get("selected_module") or {}).get("LLM", "")).strip()
        candidates = []

        if isinstance(llm_map, dict):
            if selected:
                selected_cfg = llm_map.get(selected) or {}
                if isinstance(selected_cfg, dict):
                    candidates.append(
                        (
                            str(selected_cfg.get("workspace", "")).strip(),
                            str(selected_cfg.get("yaml_path", "")).strip(),
                        )
                    )
            for llm_cfg in llm_map.values():
                if not isinstance(llm_cfg, dict):
                    continue
                candidates.append(
                    (
                        str(llm_cfg.get("workspace", "")).strip(),
                        str(llm_cfg.get("yaml_path", "")).strip(),
                    )
                )

        prompt_template = str(cfg.get("prompt_template", "")).strip()
        if prompt_template:
            candidates.append(("", prompt_template))

        server_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )

        for workspace, path_text in candidates:
            if not path_text:
                continue
            if os.path.isabs(path_text):
                abs_path = os.path.abspath(path_text)
            else:
                base = workspace if workspace else server_root
                abs_path = os.path.abspath(os.path.join(base, path_text))
            cfg_dir = os.path.dirname(abs_path)
            if os.path.basename(cfg_dir).lower() == "configs":
                exp_root = os.path.dirname(cfg_dir)
            else:
                exp_root = cfg_dir
            if exp_root:
                return os.path.abspath(os.path.join(exp_root, "data"))
        return ""

    def _extract_question_meta(self, question: str) -> Tuple[str, str, Optional[int]]:
        src = str(question or "")
        idx = src.rfind(self._question_meta_prefix)
        if idx < 0:
            return src, "", None

        meta_raw = src[idx + len(self._question_meta_prefix) :].strip()
        clean_question = src[:idx].rstrip()
        if not meta_raw:
            return clean_question, "", None

        try:
            meta_obj = json.loads(meta_raw)
        except Exception:
            return src, "", None

        if not isinstance(meta_obj, dict):
            return clean_question, "", None

        requested = self._sanitize_filename_stem(str(meta_obj.get("photo_name", "")))
        group_number = self._normalize_group_number(meta_obj.get("group_number"))
        return clean_question, requested, group_number

    def _save_image(
        self,
        image_data: bytes,
        device_id: str,
        requested_photo_name: str = "",
        group_number: Optional[int] = None,
    ) -> str:
        data_root = self._derive_experiment_data_root()
        if not data_root:
            raise ValueError("无法从当前配置解析实验数据目录（data 根目录）")

        safe_device = self._sanitize_device_for_path(device_id)
        device_dir = os.path.join(data_root, safe_device)
        normalized_group = self._normalize_group_number(group_number)
        if normalized_group is not None:
            device_dir = os.path.join(device_dir, self._format_group_dir_name(normalized_group))
        os.makedirs(device_dir, exist_ok=True)
        ext = self._guess_image_ext(image_data)
        save_data = image_data

        if ext == "jpg":
            try:
                save_data = self._convert_image_data_to_png(image_data)
                ext = "png"
            except Exception as e:
                self.logger.bind(tag=TAG).warning(
                    f"Convert JPEG to PNG failed, fallback to jpg: {e}"
                )
                save_data = image_data
                ext = "jpg"

        filename = ""

        requested_stem = self._sanitize_filename_stem(requested_photo_name)
        if requested_stem:
            # 用户指定命名时，直接使用该名称（覆盖同名旧文件）
            filename = f"{requested_stem}.{ext}"

        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            rand = uuid.uuid4().hex[:8]
            filename = f"{timestamp}_{rand}.{ext}"

        file_path = os.path.join(device_dir, filename)

        with open(file_path, "wb") as f:
            f.write(save_data)

        return file_path

    def _convert_image_data_to_png(self, image_data: bytes) -> bytes:
        in_fd, in_path = tempfile.mkstemp(prefix="xz_vision_in_", suffix=".jpg")
        out_fd, out_path = tempfile.mkstemp(prefix="xz_vision_out_", suffix=".png")
        os.close(in_fd)
        os.close(out_fd)
        try:
            with open(in_path, "wb") as f:
                f.write(image_data)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    in_path,
                    "-frames:v",
                    "1",
                    out_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            with open(out_path, "rb") as f:
                png_data = f.read()
            if not png_data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("ffmpeg output is not PNG")
            return png_data
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            raise ValueError(f"ffmpeg convert failed: {err}") from e
        except FileNotFoundError as e:
            raise ValueError("ffmpeg is not installed or not in PATH") from e
        finally:
            for p in (in_path, out_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    def _create_error_response(self, message: str) -> dict:
        """创建统一的错误响应格式"""
        return {"success": False, "message": message}

    def _verify_auth_token(self, request) -> Tuple[bool, Optional[str]]:
        """验证认证token"""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False, None

        token = auth_header[7:]  # 移除"Bearer "前缀
        return self.auth.verify_token(token)

    async def handle_post(self, request):
        """处理 MCP Vision POST 请求"""
        response = None  # 初始化response变量
        try:
            # 验证token
            is_valid, token_device_id = self._verify_auth_token(request)
            if not is_valid:
                response = web.Response(
                    text=json.dumps(
                        self._create_error_response("无效的认证token或token已过期")
                    ),
                    content_type="application/json",
                    status=401,
                )
                return response

            # 获取请求头信息
            device_id = request.headers.get("Device-Id", "")
            client_id = request.headers.get("Client-Id", "")
            if device_id != token_device_id:
                raise ValueError("设备ID与token不匹配")
            # 解析multipart/form-data请求
            reader = await request.multipart()

            # 读取question字段
            question_field = await reader.next()
            if question_field is None:
                raise ValueError("缺少问题字段")
            question = await question_field.text()
            question, requested_photo_name, group_number = self._extract_question_meta(question)
            self.logger.bind(tag=TAG).debug(f"Question: {question}")
            if requested_photo_name:
                self.logger.bind(tag=TAG).debug(
                    f"Requested photo name: {requested_photo_name}"
                )
            if group_number is not None:
                self.logger.bind(tag=TAG).debug(
                    f"Requested photo group_number: {group_number}"
                )

            # 读取图片文件
            image_field = await reader.next()
            if image_field is None:
                raise ValueError("缺少图片文件")

            # 读取图片数据
            image_data = await image_field.read()
            if not image_data:
                raise ValueError("图片数据为空")

            # 检查文件大小
            if len(image_data) > MAX_FILE_SIZE:
                raise ValueError(
                    f"图片大小超过限制，最大允许{MAX_FILE_SIZE/1024/1024}MB"
                )

            # 检查文件格式
            if not is_valid_image_file(image_data):
                raise ValueError(
                    "不支持的文件格式，请上传有效的图片文件（支持JPEG、PNG、GIF、BMP、TIFF、WEBP格式）"
                )

            # 落盘保存图片
            try:
                saved_path = self._save_image(
                    image_data,
                    device_id,
                    requested_photo_name,
                    group_number=group_number,
                )
                self.logger.bind(tag=TAG).info(f"Saved vision image: {saved_path}")
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"Save vision image failed: {e}")

            # 将图片转换为base64编码
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 如果开启了智控台，则从智控台获取模型配置
            current_config = copy.deepcopy(self.config)
            read_config_from_api = current_config.get("read_config_from_api", False)
            if read_config_from_api:
                current_config = await get_private_config_from_api(
                    current_config,
                    device_id,
                    client_id,
                )

            select_vllm_module = current_config["selected_module"].get("VLLM")
            if not select_vllm_module:
                raise ValueError("您还未设置默认的视觉分析模块")

            vllm_type = (
                select_vllm_module
                if "type" not in current_config["VLLM"][select_vllm_module]
                else current_config["VLLM"][select_vllm_module]["type"]
            )

            if not vllm_type:
                raise ValueError(f"无法找到VLLM模块对应的供应器{vllm_type}")

            vllm = create_instance(
                vllm_type, current_config["VLLM"][select_vllm_module]
            )

            result = vllm.response(question, image_base64)

            return_json = {
                "success": True,
                "action": Action.RESPONSE.name,
                "response": result,
            }

            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        except ValueError as e:
            self.logger.bind(tag=TAG).error(f"MCP Vision POST请求异常: {e}")
            return_json = self._create_error_response(str(e))
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"MCP Vision POST请求异常: {e}")
            return_json = self._create_error_response("处理请求时发生错误")
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_get(self, request):
        """处理 MCP Vision GET 请求"""
        try:
            vision_explain = get_vision_url(self.config)
            if vision_explain and len(vision_explain) > 0 and "null" != vision_explain:
                message = (
                    f"MCP Vision 接口运行正常，视觉解释接口地址是：{vision_explain}"
                )
            else:
                message = "MCP Vision 接口运行不正常，请打开data目录下的.config.yaml文件，找到【server.vision_explain】，设置好地址"

            response = web.Response(text=message, content_type="text/plain")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"MCP Vision GET请求异常: {e}")
            return_json = self._create_error_response("服务器内部错误")
            response = web.Response(
                text=json.dumps(return_json, separators=(",", ":")),
                content_type="application/json",
            )
        finally:
            self._add_cors_headers(response)
            return response
