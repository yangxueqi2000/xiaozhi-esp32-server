import os
import re
from pathlib import Path
import yaml
from collections.abc import Mapping
from config.manage_api_client import init_service, get_server_config, get_agent_models


DEFAULT_CONFIG_CANDIDATES = ("config.yaml", "config_back.yaml")
_CONFIG_TEMPLATE_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
_CONFIG_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_CONFIG_PATH_KEY_HINTS = (
    "path",
    "dir",
    "root",
    "folder",
    "file",
    "workspace",
    "template",
    "csv",
    "json",
    "png",
    "wav",
    "yaml",
)
_CONFIG_ROUTE_KEY_HINTS = ("url", "endpoint", "http", "sse", "message", "mount", "route")
_CONFIG_PATH_VARIABLE_KEYS = {
    "workspace_root",
    "lab_runs_root",
    "experiment_root",
    "experiment_config_root",
    "experiment_data_root",
    "experiment_yaml_path",
    "prompt_template_path",
    "uvvis_scan_output_root",
}


def get_project_dir():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"


def get_default_config_path():
    project_dir = get_project_dir()
    for config_name in DEFAULT_CONFIG_CANDIDATES:
        config_path = os.path.join(project_dir, config_name)
        if os.path.exists(config_path):
            return config_path
    return os.path.join(project_dir, DEFAULT_CONFIG_CANDIDATES[0])


def read_config(config_path, required=True):
    if not os.path.exists(config_path):
        if required:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, Mapping):
        raise ValueError(
            f"Config file must contain a mapping at the top level: {config_path}"
        )
    return config


def _expand_template_string(value: str, variables: Mapping[str, str]) -> str:
    text = str(value or "")
    if not text or "${" not in text:
        return text

    def _replace(match):
        key = str(match.group(1) or "").strip()
        replacement = variables.get(key)
        if not key or replacement in (None, ""):
            return match.group(0)
        return str(replacement)

    return _CONFIG_TEMPLATE_RE.sub(_replace, text)


def _has_unexpanded_template(value: str) -> bool:
    return bool(_CONFIG_TEMPLATE_RE.search(str(value or "")))


def _join_config_path(base: str, *parts: str) -> str:
    return str(Path(base, *parts))


def _looks_like_config_path_key(key: str) -> bool:
    normalized_key = str(key or "").strip().lower()
    if any(hint in normalized_key for hint in _CONFIG_ROUTE_KEY_HINTS):
        return False
    return any(hint in normalized_key for hint in _CONFIG_PATH_KEY_HINTS)


def _normalize_config_path_string(value: str) -> str:
    text = str(value or "").strip()
    if not text or _CONFIG_URL_RE.match(text):
        return text
    return str(Path(text))


def _normalize_config_path_variable(key: str, value: str) -> str:
    if key in _CONFIG_PATH_VARIABLE_KEYS:
        return _normalize_config_path_string(value)
    return value


def _build_experiment_path_variables(config: Mapping) -> dict[str, str]:
    experiment_paths = config.get("experiment_paths", {}) or {}
    if not isinstance(experiment_paths, Mapping):
        experiment_paths = {}

    variables = {
        "workspace_root": str(experiment_paths.get("workspace_root", "") or "").strip(),
        "lab_runs_root": str(experiment_paths.get("lab_runs_root", "") or "").strip(),
        "experiment_name": str(experiment_paths.get("experiment_name", "") or "").strip(),
        "experiment_root": str(experiment_paths.get("experiment_root", "") or "").strip(),
        "experiment_config_root": str(
            experiment_paths.get("experiment_config_root", "") or ""
        ).strip(),
        "experiment_data_root": str(
            experiment_paths.get("experiment_data_root", "") or ""
        ).strip(),
        "experiment_yaml_path": str(
            experiment_paths.get("experiment_yaml_path", "") or ""
        ).strip(),
        "prompt_template_path": str(
            experiment_paths.get("prompt_template_path", "") or ""
        ).strip(),
        "uvvis_scan_output_root": str(
            experiment_paths.get("uvvis_scan_output_root", "") or ""
        ).strip(),
    }

    for _ in range(6):
        variables = {
            key: _normalize_config_path_variable(
                key,
                _expand_template_string(value, variables),
            )
            for key, value in variables.items()
        }
        if not variables.get("lab_runs_root") and variables.get("workspace_root"):
            variables["lab_runs_root"] = _join_config_path(
                variables["workspace_root"], "lab_runs"
            )
        if (
            not variables.get("experiment_root")
            and variables.get("lab_runs_root")
            and variables.get("experiment_name")
        ):
            variables["experiment_root"] = _join_config_path(
                variables["lab_runs_root"], variables["experiment_name"]
            )
        if not variables.get("experiment_config_root") and variables.get("experiment_root"):
            variables["experiment_config_root"] = _join_config_path(
                variables["experiment_root"], "configs"
            )
        if not variables.get("experiment_data_root") and variables.get("experiment_root"):
            variables["experiment_data_root"] = _join_config_path(
                variables["experiment_root"], "data"
            )
        if (
            not variables.get("experiment_yaml_path")
            and variables.get("experiment_config_root")
        ):
            variables["experiment_yaml_path"] = _join_config_path(
                variables["experiment_config_root"], "experiments.yaml"
            )
        if (
            not variables.get("prompt_template_path")
            and variables.get("experiment_config_root")
        ):
            variables["prompt_template_path"] = _join_config_path(
                variables["experiment_config_root"], "local_prompt.txt"
            )
        if (
            not variables.get("uvvis_scan_output_root")
            and variables.get("experiment_data_root")
        ):
            variables["uvvis_scan_output_root"] = _join_config_path(
                variables["experiment_data_root"], "uv_data_common"
            )

    return variables


def _expand_config_templates(value, variables: Mapping[str, str], key_name: str = ""):
    if isinstance(value, Mapping):
        return {
            key: _expand_config_templates(sub_value, variables, str(key))
            for key, sub_value in value.items()
        }
    if isinstance(value, list):
        return [_expand_config_templates(item, variables, key_name) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_config_templates(item, variables, key_name) for item in value)
    if isinstance(value, str):
        expanded = _expand_template_string(value, variables)
        if _looks_like_config_path_key(key_name):
            return _normalize_config_path_string(expanded)
        return expanded
    return value


def apply_config_path_templates(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        return config

    variables = _build_experiment_path_variables(config)
    expanded = _expand_config_templates(dict(config), variables)

    if not isinstance(expanded.get("experiment_paths"), Mapping):
        expanded["experiment_paths"] = {}
    expanded["experiment_paths"] = {
        **expanded["experiment_paths"],
        **{key: value for key, value in variables.items() if value},
    }
    return expanded


def load_config():
    """加载配置文件"""
    from core.utils.cache.manager import cache_manager, CacheType

    # 检查缓存
    cached_config = cache_manager.get(CacheType.CONFIG, "main_config")
    if cached_config is not None:
        return cached_config

    default_config_path = get_default_config_path()
    custom_config_path = get_project_dir() + "data/.config.yaml"
    default_config_exists = os.path.exists(default_config_path)
    custom_config_exists = os.path.exists(custom_config_path)

    if not custom_config_exists and not default_config_exists:
        raise FileNotFoundError(
            "No config file found. Expected data/.config.yaml, config.yaml, or config_back.yaml."
        )

    # 加载默认配置
    default_config = read_config(default_config_path, required=False)
    custom_config = read_config(custom_config_path, required=False)

    if custom_config.get("manager-api", {}).get("url"):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，使用异步版本
            config = asyncio.run_coroutine_threadsafe(
                get_config_from_api_async(custom_config), loop
            ).result()
        except RuntimeError:
            # 如果不在事件循环中（启动时），创建新的事件循环
            config = asyncio.run(get_config_from_api_async(custom_config))
    else:
        # 合并配置
        if custom_config_exists and default_config_exists:
            config = merge_configs(default_config, custom_config)
        elif custom_config_exists:
            config = custom_config
        else:
            config = default_config
    # 初始化目录
    config = apply_config_path_templates(config)
    ensure_directories(config)

    # 缓存配置
    cache_manager.set(CacheType.CONFIG, "main_config", config)
    return config


async def get_config_from_api_async(config):
    """从Java API获取配置（异步版本）"""
    # 初始化API客户端
    init_service(config)

    # 获取服务器配置
    config_data = await get_server_config()
    if config_data is None:
        raise Exception("Failed to fetch server config from API")

    config_data["read_config_from_api"] = True
    config_data["manager-api"] = {
        "url": config["manager-api"].get("url", ""),
        "secret": config["manager-api"].get("secret", ""),
    }
    if config.get("experiment_paths"):
        config_data["experiment_paths"] = config.get("experiment_paths")
    auth_enabled = config_data.get("server", {}).get("auth", {}).get("enabled", False)
    # server的配置以本地为准
    if config.get("server"):
        config_data["server"] = {
            "ip": config["server"].get("ip", ""),
            "port": config["server"].get("port", ""),
            "http_port": config["server"].get("http_port", ""),
            "vision_explain": config["server"].get("vision_explain", ""),
            "auth_key": config["server"].get("auth_key", ""),
        }
    config_data["server"]["auth"] = {"enabled": auth_enabled}
    # 如果服务器没有prompt_template，则从本地配置读取
    if not config_data.get("prompt_template"):
        config_data["prompt_template"] = config.get("prompt_template")
    return config_data


async def get_private_config_from_api(config, device_id, client_id):
    """从Java API获取私有配置"""
    return await get_agent_models(device_id, client_id, config["selected_module"])


def ensure_directories(config):
    """确保所有配置路径存在"""
    dirs_to_create = set()
    project_dir = get_project_dir()  # 获取项目根目录
    # 日志文件目录
    log_dir = config.get("log", {}).get("log_dir", "tmp")
    dirs_to_create.add(os.path.join(project_dir, log_dir))

    # ASR/TTS模块输出目录
    for module in ["ASR", "TTS"]:
        if config.get(module) is None:
            continue
        for provider in config.get(module, {}).values():
            output_dir = provider.get("output_dir", "")
            if output_dir:
                dirs_to_create.add(output_dir)

    # 根据selected_module创建模型目录
    selected_modules = config.get("selected_module", {})
    for module_type in ["ASR", "LLM", "TTS"]:
        selected_provider = selected_modules.get(module_type)
        if not selected_provider:
            continue
        module_config = config.get(module_type)
        if not isinstance(module_config, Mapping):
            continue
        provider_config = module_config.get(selected_provider, {})
        if not isinstance(provider_config, Mapping):
            continue
        output_dir = provider_config.get("output_dir")
        if output_dir:
            full_model_dir = os.path.join(project_dir, output_dir)
            dirs_to_create.add(full_model_dir)

    # 统一创建目录（保留原data目录创建）
    for dir_path in dirs_to_create:
        if _has_unexpanded_template(dir_path):
            print(f"警告：跳过未展开模板路径 {dir_path}")
            continue
        try:
            os.makedirs(dir_path, exist_ok=True)
        except PermissionError:
            print(f"警告：无法创建目录 {dir_path}，请检查写入权限")


def merge_configs(default_config, custom_config):
    """
    递归合并配置，custom_config优先级更高

    Args:
        default_config: 默认配置
        custom_config: 用户自定义配置

    Returns:
        合并后的配置
    """
    if not isinstance(default_config, Mapping) or not isinstance(
        custom_config, Mapping
    ):
        return custom_config

    merged = dict(default_config)

    for key, value in custom_config.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
