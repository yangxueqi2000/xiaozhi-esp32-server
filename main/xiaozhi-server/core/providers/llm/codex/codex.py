import json
import locale
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tomli as tomllib

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.providers.llm.system_prompt import get_system_prompt_for_function
from core.providers.tools.server_mcp.payload_utils import sync_server_mcp_payload_state
from core.utils import textUtils

TAG = __name__
logger = setup_logging()
_SERVER_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MCP_SETTINGS_PATH = _SERVER_ROOT / "data" / ".mcp_server_settings.json"
_DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
_DEFAULT_APP_SERVER_CONFIG_OVERRIDES = (
    "allow_login_shell=false",
)
_RUNTIME_CODEX_HOME_COPY_FILES = (
    "AGENTS.md",
    "auth.json",
    "cap_sid",
    "installation_id",
    "version.json",
)
_RUNTIME_CODEX_HOME_COPY_DIRS = (
    "mcp-proxies",
    "rules",
    "vendor_imports",
)
_RUNTIME_CODEX_HOME_DISABLED_DIRS = (
    "plugins",
    "skills",
)
_AGENT_INTERNAL_LEAK_MARKERS = tuple(
    marker.lower()
    for marker in (
        "We need respond",
        "We need to respond",
        "Need to respond",
        "respond as assistant",
        "We must follow instructions",
        "We need continue",
        "Need continue",
        "Need to decide",
        "Need assistant",
        "Need proceed",
        "Need ensure",
        "Need verify",
        "Let's execute",
        "Let's craft",
        "Let's produce final",
        "Let's final",
        "Use functions",
        "call mcp",
        "analysis done",
        "Wait conversation",
        "Wait now",
        "Wait given",
        "Actually conversation",
        "We'll continue",
        "Let's see upcoming",
        "Let's process next hypothetical",
        "as ChatGPT",
        "final channel",
        "No user message",
        "No more?",
        "We need final answer",
        "Already done",
        "next user maybe",
        "tool calls before final",
        "Sequence quick:",
    )
)
_AGENT_INTERNAL_LEAK_HOLD_CHARS = max(len(marker) for marker in _AGENT_INTERNAL_LEAK_MARKERS) - 1
_PHOTO_AUTHORIZATION_PHRASES = (
    "可以拍照",
    "可以拍",
    "能拍照",
    "能拍",
    "拍吧",
    "拍照吧",
    "开始拍",
    "现在拍",
    "拍一下",
    "准备好了",
    "可以了",
    "行拍",
    "行，拍",
    "好拍",
    "好，拍",
)
_EXPLICIT_PHOTO_HOT_PATH_PHRASES = (
    "可以拍照",
    "可以拍",
    "能拍照",
    "能拍",
    "拍吧",
    "拍照吧",
    "开始拍",
    "现在拍",
    "拍一个",
    "拍一下",
    "行拍",
    "行，拍",
    "好拍",
    "好，拍",
)
_PHOTO_QUESTION_MARKERS = ("?", "？", "吗", "么")
_EXP2_UVVIS_PREP_STEP_ID = "step_3_uv_vis_shared_dark_air_prep"
_EXP2_UVVIS_PREP_START_PHRASES = (
    "\u5f00\u59cb\u626b\u63cf",
    "\u53ef\u4ee5\u5f00\u59cb",
    "\u5f00\u59cb\u6d4b",
    "\u5f00\u59cb\u6821\u6b63",
    "\u626b\u63cf",
)
_EXP2_UVVIS_EMPTY_CONFIRM_PHRASES = (
    "\u90fd\u7a7a",
    "\u90fd\u662f\u7a7a",
    "\u5df2\u7a7a",
    "\u5df2\u7559\u7a7a",
    "\u7559\u7a7a\u4e86",
    "\u6ca1\u6709\u653e",
    "\u6ca1\u653e\u6db2\u4f53",
    "\u4e0d\u653e\u4efb\u4f55\u6db2\u4f53",
    "\u6837\u54c1\u4f4d\u7a7a",
    "\u53c2\u6bd4\u4f4d\u7a7a",
)
_EXP2_UVVIS_PREP_COMPLETION_CLAIM_PHRASES = (
    "\u6697\u7535\u6d41",
    "\u7a7a\u6c14\u80fd\u91cf",
    "\u7a7a\u6c14\u57fa\u7ebf",
    "\u6821\u6b63\u5df2\u7ecf\u5b8c\u6210",
    "\u6821\u6b63\u5b8c\u6210",
)
_EXP2_UVVIS_SAMPLE_LOADING_PHRASES = (
    "\u88c5\u5165\u4e94\u8054\u67b6",
    "\u653e\u5165\u4e94\u8054\u67b6",
    "\u53c2\u6bd4\u4f4d\u653e\u7eaf\u6c34",
    "\u771f\u5b9e\u6837\u54c1",
    "\u73b0\u5728\u628a",
)
_EXP2_UVVIS_PREP_NEED_EMPTY_REPLY = (
    "\u8bf7\u5148\u786e\u8ba4 1 \u5230 5 \u53f7\u6837\u54c1\u4f4d\u548c\u4eea\u5668"
    "\u539f\u751f\u53c2\u6bd4\u4f4d\u90fd\u662f\u7a7a\u7684\uff0c\u4e0d\u8981\u653e"
    "\u4efb\u4f55\u6db2\u4f53\u3002\u786e\u8ba4\u540e\u518d\u8bf4\u201c\u90fd\u7a7a\u4e86\uff0c"
    "\u5f00\u59cb\u626b\u63cf\u201d\u3002"
)
_EXP2_UVVIS_PREP_NO_TOOL_REPLY = (
    "\u6211\u8fd8\u6ca1\u6709\u771f\u6b63\u5b8c\u6210\u6697\u7535\u6d41\u548c"
    "\u7a7a\u6c14\u80fd\u91cf\u6821\u6b63\uff0c\u5148\u4e0d\u653e\u6837\u54c1\u3002"
    "\u8bf7\u786e\u8ba4\u4f4d\u7f6e\u90fd\u7a7a\u540e\u518d\u8bf4\u201c\u90fd\u7a7a\u4e86\uff0c"
    "\u5f00\u59cb\u626b\u63cf\u201d\u3002"
)

if os.name == "nt":
    try:
        import win32api
        import win32con
        import win32job
    except ImportError:
        win32api = None
        win32con = None
        win32job = None
else:
    win32api = None
    win32con = None
    win32job = None


def _ts() -> str:
    """ISO timestamp with timezone, milliseconds."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _send(proc: subprocess.Popen, obj: Dict) -> None:
    proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _ordered_encodings(*encodings: Optional[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for encoding in encodings:
        if not encoding:
            continue
        normalized = str(encoding).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _decode_pipe_line(line: Any, encodings: List[str]) -> str:
    if isinstance(line, str):
        return line
    if line is None:
        return ""

    data = bytes(line)
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _decode_stdout_line(line: Any) -> str:
    return _decode_pipe_line(line, ["utf-8"])


def _decode_stderr_line(line: Any) -> str:
    return _decode_pipe_line(
        line,
        _ordered_encodings(
            "utf-8",
            locale.getpreferredencoding(False),
            getattr(locale, "getencoding", lambda: None)(),
            "mbcs",
            "cp936",
            "gbk",
            "big5",
        ),
    )


def _create_windows_kill_job():
    if os.name != "nt" or win32job is None:
        return None

    try:
        job = win32job.CreateJobObject(None, "")
        extended_info = win32job.QueryInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
        )
        extended_info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        win32job.SetInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
            extended_info,
        )
        return job
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"failed to create windows job object for codex cleanup: {exc}"
        )
        return None


def _close_windows_job(job_handle) -> None:
    if job_handle is None or win32api is None:
        return
    try:
        win32api.CloseHandle(job_handle)
    except Exception:
        pass


def _assign_process_to_windows_job(
    proc: subprocess.Popen,
    job_handle,
):
    if (
        os.name != "nt"
        or job_handle is None
        or win32api is None
        or win32con is None
        or win32job is None
    ):
        return None

    process_handle = None
    try:
        process_handle = win32api.OpenProcess(
            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
            False,
            proc.pid,
        )
        win32job.AssignProcessToJobObject(job_handle, process_handle)
        return job_handle
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"failed to assign codex app-server process {proc.pid} to job object: {exc}"
        )
        _close_windows_job(job_handle)
        return None
    finally:
        if process_handle is not None:
            try:
                win32api.CloseHandle(process_handle)
            except Exception:
                pass


def _taskkill_process_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _terminate_process_tree(
    proc: subprocess.Popen,
    *,
    timeout_seconds: float,
    windows_job_handle=None,
) -> None:
    pid = getattr(proc, "pid", None)

    if os.name == "nt":
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if windows_job_handle is not None and win32job is not None:
                try:
                    win32job.TerminateJobObject(windows_job_handle, 1)
                except Exception:
                    pass
            elif pid:
                _taskkill_process_tree(pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            if windows_job_handle is not None and win32job is not None:
                try:
                    win32job.TerminateJobObject(windows_job_handle, 1)
                except Exception:
                    pass
            elif pid:
                _taskkill_process_tree(pid)
        finally:
            _close_windows_job(windows_job_handle)
        return

    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _resolve_optional_path(path_text: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
    raw = str(path_text or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    try:
        return path.resolve()
    except OSError:
        return path


def _snapshot_file_state(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _snapshot_watched_paths(paths: List[Path]) -> Dict[str, Optional[Tuple[int, int]]]:
    snapshot: Dict[str, Optional[Tuple[int, int]]] = {}
    for path in paths:
        key = str(path)
        if key in snapshot:
            continue
        snapshot[key] = _snapshot_file_state(path)
    return snapshot


def _toml_key(key: str) -> str:
    text = str(key or "")
    if re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise ValueError(f"unsupported float value for TOML literal: {value}")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if item is None:
                continue
            items.append(f"{_toml_key(str(key))} = {_toml_literal(item)}")
        return "{ " + ", ".join(items) + " }"
    raise TypeError(f"unsupported TOML literal type: {type(value).__name__}")


def _load_toml_document(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle) or {}
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"codex config load failed: path={path} error={exc}"
        )
        return {}

    if not isinstance(raw, dict):
        logger.bind(tag=TAG).warning(
            f"codex config ignored non-object root: path={path}"
        )
        return {}

    return raw


def _coerce_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}

    coerced: Dict[str, str] = {}
    for key, item in value.items():
        if key is None or item is None:
            continue
        coerced[str(key)] = str(item)
    return coerced


def _load_codex_mcp_server_configs(settings_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if settings_path is None or not settings_path.exists():
        return {}

    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"codex mcp settings load failed: path={settings_path} error={exc}"
        )
        return {}

    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        return {}

    merged_servers: Dict[str, Dict[str, Any]] = {}
    for name, server_cfg in servers.items():
        server_name = str(name or "").strip()
        if not server_name or not re.fullmatch(r"[A-Za-z0-9_-]+", server_name):
            logger.bind(tag=TAG).warning(
                f"codex mcp settings skipped unsupported server name: {server_name or '<empty>'}"
            )
            continue
        if not isinstance(server_cfg, dict):
            continue

        url = str(server_cfg.get("url", "") or "").strip()
        if url:
            merged_entry: Dict[str, Any] = {"url": url}
            for option_name in (
                "transport",
                "timeout",
                "tool_timeout_sec",
                "startup_timeout_sec",
                "startup_timeout_ms",
                "sse_read_timeout",
                "initialize_timeout",
                "terminate_on_close",
            ):
                if option_name in server_cfg:
                    merged_entry[option_name] = server_cfg.get(option_name)
            if "tool_timeout_sec" not in merged_entry and "timeout" in merged_entry:
                merged_entry["tool_timeout_sec"] = merged_entry["timeout"]
            if "startup_timeout_sec" not in merged_entry and "initialize_timeout" in merged_entry:
                merged_entry["startup_timeout_sec"] = merged_entry["initialize_timeout"]

            headers = _coerce_string_map(server_cfg.get("headers"))
            if headers:
                merged_entry["http_headers"] = headers

            bearer_token_env_var = str(
                server_cfg.get("bearer_token_env_var", "") or ""
            ).strip()
            if bearer_token_env_var:
                merged_entry["bearer_token_env_var"] = bearer_token_env_var
            merged_servers[server_name] = merged_entry
            continue

        command = str(server_cfg.get("command", "") or "").strip()
        if not command:
            continue

        merged_entry = {"command": command}

        args = server_cfg.get("args")
        if isinstance(args, list):
            merged_entry["args"] = [str(item) for item in args]

        env = _coerce_string_map(server_cfg.get("env"))
        if env:
            merged_entry["env"] = env

        cwd = str(server_cfg.get("cwd", "") or "").strip()
        if cwd:
            merged_entry["cwd"] = cwd

        merged_servers[server_name] = merged_entry

    return merged_servers


def _run_codex_mcp_startup_hooks(settings_path: Optional[Path]) -> None:
    if settings_path is None or not settings_path.exists():
        return

    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        logger.bind(tag=TAG).warning(
            f"codex mcp startup settings load failed: path={settings_path} error={exc}"
        )
        return

    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        return

    for name, server_cfg in servers.items():
        if not isinstance(server_cfg, dict):
            continue
        startup_cfg = server_cfg.get("startup")
        if not isinstance(startup_cfg, dict):
            continue
        command = str(startup_cfg.get("command", "") or "").strip()
        if not command:
            continue
        raw_args = startup_cfg.get("args")
        args = [str(item) for item in raw_args] if isinstance(raw_args, list) else []
        cwd = str(startup_cfg.get("cwd", "") or "").strip() or None
        timeout_value = startup_cfg.get("timeout", 120)
        try:
            timeout_seconds = max(5.0, float(timeout_value or 120))
        except (TypeError, ValueError):
            timeout_seconds = 120.0

        env = os.environ.copy()
        for source in (server_cfg.get("env"), startup_cfg.get("env")):
            env.update(_coerce_string_map(source))

        try:
            logger.bind(tag=TAG).info(
                f"running Codex MCP startup hook for {name}: {command} {' '.join(args)}"
            )
            completed = subprocess.run(
                [command, *args],
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.bind(tag=TAG).warning(
                f"Codex MCP startup hook timed out: server={name} timeout={timeout_seconds:.1f}s"
            )
            continue
        except Exception as exc:
            logger.bind(tag=TAG).warning(
                f"Codex MCP startup hook failed to start: server={name} error={exc}"
            )
            continue

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            logger.bind(tag=TAG).warning(
                f"Codex MCP startup hook exited nonzero: server={name} "
                f"returncode={completed.returncode} detail={detail[:500]}"
            )


def _merge_codex_mcp_server_configs(
    base_config: Dict[str, Any],
    server_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    merged = deepcopy(base_config) if isinstance(base_config, dict) else {}
    existing_servers = merged.get("mcp_servers")
    if not isinstance(existing_servers, dict):
        existing_servers = {}
    else:
        existing_servers = deepcopy(existing_servers)

    for server_name, server_cfg in server_configs.items():
        existing_servers[server_name] = deepcopy(server_cfg)

    merged["mcp_servers"] = existing_servers
    return merged


def _inject_experiment_yaml_env(
    server_configs: Dict[str, Dict[str, Any]],
    experiment_yaml_path: str,
) -> None:
    yaml_path = str(experiment_yaml_path or "").strip()
    if not yaml_path:
        return

    for server_cfg in server_configs.values():
        if not isinstance(server_cfg, dict) or "command" not in server_cfg:
            continue
        env = server_cfg.get("env")
        if not isinstance(env, dict):
            env = {}
            server_cfg["env"] = env
        env["EXPERIMENT_YAML_PATH"] = yaml_path


def _split_toml_table_items(
    value: Dict[str, Any],
) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Dict[str, Any]]]]:
    scalars: List[Tuple[str, Any]] = []
    child_tables: List[Tuple[str, Dict[str, Any]]] = []
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, dict):
            child_tables.append((str(key), item))
        else:
            scalars.append((str(key), item))
    return scalars, child_tables


def _toml_table_name(parts: List[str]) -> str:
    return ".".join(_toml_key(str(part)) for part in parts)


def _append_toml_table_lines(
    lines: List[str],
    table_value: Dict[str, Any],
    table_path: List[str],
) -> None:
    scalar_items, child_tables = _split_toml_table_items(table_value)
    if table_path:
        if lines:
            lines.append("")
        lines.append(f"[{_toml_table_name(table_path)}]")

    for key, value in scalar_items:
        lines.append(f"{_toml_key(key)} = {_toml_literal(value)}")

    for key, child_value in child_tables:
        _append_toml_table_lines(lines, child_value, [*table_path, key])


def _dump_toml_document(document: Dict[str, Any]) -> str:
    if not isinstance(document, dict):
        raise TypeError("toml document root must be a dict")

    lines: List[str] = []
    root_scalars, root_tables = _split_toml_table_items(document)

    for key, value in root_scalars:
        lines.append(f"{_toml_key(key)} = {_toml_literal(value)}")

    for key, child_value in root_tables:
        _append_toml_table_lines(lines, child_value, [key])

    text = "\n".join(lines).rstrip()
    return text + ("\n" if text else "")


def _override_key(override: str) -> str:
    text = str(override or "").strip()
    if not text:
        return ""
    return text.split("=", 1)[0].strip()


def _merge_app_server_config_overrides(configured_overrides: Any) -> List[str]:
    items = [
        str(item).strip()
        for item in _DEFAULT_APP_SERVER_CONFIG_OVERRIDES
        if str(item).strip()
    ]
    items.extend(
        str(item).strip()
        for item in (configured_overrides or [])
        if str(item).strip()
    )

    merged_by_key: Dict[str, str] = {}
    key_order: List[str] = []
    for item in items:
        key = _override_key(item)
        if not key:
            continue
        if key not in merged_by_key:
            key_order.append(key)
        merged_by_key[key] = item
    return [merged_by_key[key] for key in key_order]


def _is_server_request(msg: Dict) -> bool:
    return (
        "id" in msg
        and "method" in msg
        and "result" not in msg
        and "error" not in msg
    )


def _extract_elicitation_tool_name(params: Dict, meta: Dict) -> str:
    for key in ("tool", "tool_name", "name"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    message = str(params.get("message") or "")
    match = re.search(r'tool\s+"([^"]+)"', message)
    return match.group(1).strip() if match else ""


def _user_text_authorizes_photo(user_text: str) -> bool:
    text = str(user_text or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _PHOTO_QUESTION_MARKERS):
        return False
    return any(phrase in text for phrase in _PHOTO_AUTHORIZATION_PHRASES)


def _user_text_explicit_photo_hot_path(user_text: str) -> bool:
    text = str(user_text or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _PHOTO_QUESTION_MARKERS):
        return False
    return any(phrase in text for phrase in _EXPLICIT_PHOTO_HOT_PATH_PHRASES)


def _photo_authorization_hot_path_prompt_block(user_text: str) -> str:
    if not _user_text_explicit_photo_hot_path(user_text):
        return ""
    return (
        "Current-turn photo authorization hot path:\n"
        "- The latest student message is an explicit authorization to take a photo.\n"
        "- Before any student-facing reply, your next action must be the connected "
        "xiaozhi_take_photo MCP tool call, using the current step/sample photo name "
        "when available.\n"
        "- Do not explain first, do not wait, do not ask again, do not call unrelated "
        "overview/reference/schema tools first, and do not tell the student to take "
        "the photo manually.\n"
        "- After the photo tool succeeds, write the photo result to experiment_graph "
        "if the current experiment step requires a graph record; only then give a "
        "brief confirmation or the next graph-approved action."
    )


def _text_contains_any(text: str, phrases: Tuple[str, ...]) -> bool:
    normalized = str(text or "").lower()
    return any(phrase.lower() in normalized for phrase in phrases)


def _exp2_uvvis_shared_prep_exists(experiment_yaml_path: str) -> bool:
    yaml_text = _norm_str(experiment_yaml_path)
    if not yaml_text:
        return False
    try:
        yaml_path = Path(yaml_text).resolve()
        uv_common_dir = yaml_path.parent.parent / "data" / "uv_data_common"
        if not uv_common_dir.exists():
            return False
        dark_current_exists = any(uv_common_dir.glob("dark_current_*.json"))
        air_baseline_exists = (
            (uv_common_dir / "latest_air_blank_manifest.json").exists()
            or (uv_common_dir / "air_blank_latest.csv").exists()
            or any(uv_common_dir.glob("air_baseline_*_manifest.json"))
        )
        return bool(dark_current_exists and air_baseline_exists)
    except Exception:
        return False


def _is_exp2_uvvis_prep_guard_turn(
    user_text: str,
    experiment_context: Dict[str, str],
    experiment_yaml_path: str,
) -> bool:
    yaml_path = _norm_str(
        experiment_context.get("experiment_yaml_path") or experiment_yaml_path
    ).replace("\\", "/").lower()
    if "exp2_uv_vis_analysis" not in yaml_path:
        return False
    if not _text_contains_any(user_text, _EXP2_UVVIS_PREP_START_PHRASES):
        return False

    current_step_id = _norm_str(
        experiment_context.get("experiment_current_step_id", "")
    )
    if current_step_id == "" and _exp2_uvvis_shared_prep_exists(experiment_yaml_path):
        return False
    # When graph prewarm has not attached yet, this step id is empty. Treat the
    # first UV-Vis scan authorization as guarded instead of trusting memory.
    return current_step_id in {"", _EXP2_UVVIS_PREP_STEP_ID}


def _exp2_uvvis_prep_user_confirmed_empty(user_text: str) -> bool:
    return _text_contains_any(user_text, _EXP2_UVVIS_EMPTY_CONFIRM_PHRASES)


def _exp2_uvvis_prep_text_claims_success(text: str) -> bool:
    if not text:
        return False
    if _text_contains_any(text, _EXP2_UVVIS_SAMPLE_LOADING_PHRASES):
        return True
    return (
        _text_contains_any(text, _EXP2_UVVIS_PREP_COMPLETION_CLAIM_PHRASES)
        and _text_contains_any(
            text,
            ("\u5b8c\u6210", "\u5df2\u7ecf\u5b8c\u6210", "\u5df2\u5b8c\u6210"),
        )
    )


def _finalize_exp2_uvvis_prep_guard_text(
    *,
    user_text: str,
    assistant_text: str,
    called_tools: List[str],
) -> str:
    if "uvvis_prepare_dark_current" in called_tools:
        return assistant_text

    if not _exp2_uvvis_prep_user_confirmed_empty(user_text):
        return _EXP2_UVVIS_PREP_NEED_EMPTY_REPLY

    if _exp2_uvvis_prep_text_claims_success(assistant_text):
        return _EXP2_UVVIS_PREP_NO_TOOL_REPLY

    return assistant_text or _EXP2_UVVIS_PREP_NO_TOOL_REPLY


def _accept_server_request(
    proc: subprocess.Popen,
    msg: Dict,
    auto_approve: bool,
    mcp_tool_guard=None,
) -> None:
    method = str(msg.get("method") or "")
    decision = "accept" if auto_approve else "decline"

    if method == "mcpServer/elicitation/request":
        # Newer Codex app-server versions use RMCP elicitation semantics here.
        params = msg.get("params", {}) or {}
        meta = params.get("_meta", {}) or {}
        if auto_approve and meta.get("codex_approval_kind") == "mcp_tool_call":
            allow_tool_call = True
            if callable(mcp_tool_guard):
                try:
                    allow_tool_call = bool(mcp_tool_guard(params, meta))
                except Exception as exc:
                    logger.bind(tag=TAG).warning(
                        f"codex mcp tool guard failed open: {exc}"
                    )
                    allow_tool_call = True
            if allow_tool_call:
                result = {"action": "accept", "content": {}}
            else:
                result = {"action": "decline", "content": None}
        else:
            # This provider has no UI to collect structured input, so do not
            # fake accepted content for non-approval elicitation forms.
            result = {"action": "decline", "content": None}
    elif method == "item/tool/requestUserInput":
        result = {"answers": {}}
    elif method == "item/commandExecution/requestApproval":
        result = {"decision": "approved" if auto_approve else "denied"}
    elif method == "execCommandApproval":
        result = {"decision": "approved" if auto_approve else "denied"}
    elif method == "item/permissions/requestApproval":
        result = {
            "permissions": {
                "fileSystem": None,
                "network": {"enabled": bool(auto_approve)},
            },
            "scope": "turn",
        }
    else:
        result = {"decision": decision}

    _send(proc, {"id": msg["id"], "result": result})


class _StdoutReader(threading.Thread):
    def __init__(self, proc: subprocess.Popen, out_queue: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.q = out_queue

    def run(self) -> None:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.q.put({"__eof__": True})
                return
            line = _decode_stdout_line(line).strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except json.JSONDecodeError:
                self.q.put({"__non_json__": line})


def _read_one(q: queue.Queue, timeout: Optional[float] = None) -> Dict:
    while True:
        msg = q.get(timeout=timeout)
        if isinstance(msg, dict) and msg.get("__eof__"):
            raise RuntimeError("codex app-server exited (stdout closed)")
        if isinstance(msg, dict) and "__non_json__" in msg:
            logger.bind(tag=TAG).debug(f"codex non-json stdout: {msg['__non_json__']}")
            continue
        return msg


def _wait_result(
    proc: subprocess.Popen,
    q: queue.Queue,
    req_id: int,
    auto_approve: bool,
) -> Dict:
    while True:
        msg = _read_one(q, timeout=None)
        if _is_server_request(msg):
            _accept_server_request(proc, msg, auto_approve)
            continue
        if msg.get("id") == req_id and ("result" in msg or "error" in msg):
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg["result"]


def _matches_thread_turn(msg: Dict, thread_id: str, turn_id: str) -> bool:
    method = msg.get("method")
    params = msg.get("params", {}) or {}

    if method == "turn/completed":
        return params.get("threadId") == thread_id and str(
            (params.get("turn") or {}).get("id")
        ) == str(turn_id)

    if "threadId" in params and params["threadId"] != thread_id:
        return False
    if "turnId" in params and str(params["turnId"]) != str(turn_id):
        return False
    if "threadId" not in params and "turnId" not in params:
        return False
    return True


def _split_dialogue(dialogue: List[Dict]) -> Tuple[List[Dict], str, List[Dict]]:
    last_user_index = None
    for idx in range(len(dialogue) - 1, -1, -1):
        if dialogue[idx].get("role") == "user":
            last_user_index = idx
            break
    if last_user_index is None:
        return [dialogue], "", []
    history = dialogue[:last_user_index]
    last_user = dialogue[last_user_index].get("content") or ""
    tail = dialogue[last_user_index + 1 :]
    return history, last_user, tail


def _extract_system_prompt(dialogue: List[Dict]) -> str:
    for msg in dialogue:
        if msg.get("role") == "system":
            return msg.get("content") or ""
    return ""


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _system_prompt_restart_fingerprint(system_prompt: str) -> str:
    """
    Build a stable fingerprint for restart decisions.
    Ignore volatile time-like fields so we don't restart Codex session on every turn.
    """
    text = _normalize_whitespace(system_prompt)
    if not text:
        return ""

    # HH:MM or HH:MM:SS
    text = re.sub(r"\b([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b", "<TIME>", text)
    # ISO-like datetime fragments
    text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?\b",
        "<DATETIME>",
        text,
    )
    return text


def _user_already_contains_system_prompt(system_prompt: str, user_text: str) -> bool:
    system_text = str(system_prompt or "").strip()
    user = str(user_text or "").strip()
    if not system_text or not user:
        return False

    # Guard against short accidental matches.
    if len(system_text) < 120:
        return False

    if system_text in user:
        return True

    system_norm = _normalize_whitespace(system_text)
    user_norm = _normalize_whitespace(user)
    if system_norm in user_norm:
        return True

    # Also treat long shared chunks as duplicated prompt payload.
    chunk = 320
    if len(system_norm) >= chunk and system_norm[:chunk] in user_norm:
        return True
    if len(system_norm) >= chunk and system_norm[-chunk:] in user_norm:
        return True

    return False


def _build_transcript(history: List[Dict]) -> str:
    lines: List[str] = []
    for msg in history:
        role = (msg.get("role") or "").lower()
        if role == "system":
            continue
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "tool":
            prefix = "Tool"
        else:
            prefix = role or "Message"
        content = msg.get("content")
        if content is None and "tool_calls" in msg:
            content = json.dumps(msg["tool_calls"], ensure_ascii=False)
        if content:
            lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_tool_context(messages: List[Dict]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        if role == "tool":
            content = msg.get("content") or ""
            if content:
                lines.append(f"Tool result: {content}")
        elif role == "assistant" and msg.get("tool_calls"):
            payload = json.dumps(msg.get("tool_calls"), ensure_ascii=False)
            lines.append(f"Tool call: {payload}")
    return "\n".join(lines)


def _short(text: Optional[str], limit: int = 240) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    return text if len(text) <= limit else text[:limit] + " ..."


def _should_suppress_stderr_warning(text: str) -> bool:
    text = str(text or "")
    if not text:
        return False

    if (
        "failed to refresh available models" in text
        and "timeout waiting for child process to exit" in text
    ):
        return True

    # Codex app-server may emit this when its internal MCP/WHAM transport
    # hits a transient TLS/network handshake EOF. In practice the provider can
    # continue serving later turns, so keep it out of warning-level logs.
    if _is_recoverable_wham_transport_warning(text):
        return True

    # Windows may emit localized "process not found" stderr when Codex tries
    # to stop a helper that has already exited. This is noisy but harmless.
    if _is_benign_process_cleanup_warning(text):
        return True

    if _is_benign_powershell_profile_warning(text):
        return True

    if _is_benign_codex_loader_warning(text):
        return True

    return False


def _is_benign_codex_loader_warning(text: str) -> bool:
    normalized = str(text or "")
    if not normalized:
        return False

    # These are emitted by optional Codex plugin/skill metadata loaders. They
    # are not actionable in the voice assistant runtime and can flood the
    # per-turn monitor logs when Codex refreshes its app-server state.
    if "codex_core_plugins::manifest" in normalized:
        return "ignoring interface.defaultPrompt" in normalized
    if "codex_core_skills::loader" in normalized:
        return (
            "ignoring interface.icon_small" in normalized
            or "ignoring interface.icon_large" in normalized
        )
    return False


def _is_recoverable_wham_transport_warning(text: str) -> bool:
    normalized = str(text or "")
    if not normalized:
        return False

    if (
        "worker quit with fatal: Transport channel closed" not in normalized
        or "https://chatgpt.com/backend-api/wham/apps" not in normalized
    ):
        return False

    recoverable_markers = (
        "unexpected EOF during handshake",
        "http/request failed: error sending request for url",
        "error sending request for url (https://chatgpt.com/backend-api/wham/apps)",
    )
    return any(marker in normalized for marker in recoverable_markers)


def _is_benign_process_cleanup_warning(text: str) -> bool:
    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return False

    if "cannot find a process with the process identifier" in normalized:
        return True

    if "the process" in normalized and "not found" in normalized:
        return True

    chinese_markers = (
        "没有找到进程",
        "未能找到进程",
        "找不到进程",
        "找不到具有进程标识符",
        "没有此任务的实例在运行",
    )
    return any(marker in text for marker in chinese_markers)


def _is_benign_powershell_profile_warning(text: str) -> bool:
    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return False

    profile_markers = (
        "windowspowershell\\profile.ps1",
        "windowspowershell/profile.ps1",
        "microsoft.powershell_profile.ps1",
    )
    if any(marker in normalized for marker in profile_markers):
        return True

    if "about_execution_policies" in normalized:
        return True

    if "pssecurityexception" in normalized and "unauthorizedaccess" in normalized:
        return True

    return False


def _is_powershell_profile_warning_continuation(text: str) -> bool:
    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return False

    continuation_prefixes = (
        "at line:",
        "所在位置 行:",
        "+ . ",
        "+ ~",
        "+   ~",
        "categoryinfo",
        "fullyqualifiederrorid",
    )
    if any(normalized.startswith(prefix) for prefix in continuation_prefixes):
        return True

    if "pssecurityexception" in normalized or "unauthorizedaccess" in normalized:
        return True

    return False


def _recoverable_stderr_reason(text: str) -> Optional[str]:
    text = str(text or "")
    if not text:
        return None

    if (
        "failed to refresh available models" in text
        and "timeout waiting for child process to exit" in text
    ):
        return "models_refresh_timeout"

    if _is_recoverable_wham_transport_warning(text):
        return "wham_transport_eof"

    return None


def _is_recoverable_turn_failure(exc: Exception, session: Optional["_CodexSession"]) -> bool:
    if session and session._restart_required:
        return True

    text = str(exc or "")
    if not text:
        return False

    return "codex app-server exited (stdout closed)" in text


def _format_action_desc(item: Dict) -> str:
    desc = item.get("type") or "item"
    if "command" in item:
        desc += f" cmd={_short(str(item['command']))}"
    if "path" in item:
        desc += f" path={_short(str(item['path']))}"
    if "tool" in item:
        desc += f" tool={_short(str(item['tool']))}"
    if "name" in item:
        desc += f" name={_short(str(item['name']))}"
    return desc


def _sync_native_mcp_function_call_state(conn: Any, item: Dict[str, Any]) -> None:
    """Mirror app-server native MCP calls into xiaozhi's per-turn MCP state."""
    if conn is None or not isinstance(item, dict):
        return
    if item.get("type") != "function_call":
        return
    if str(item.get("status") or "").strip() != "completed":
        return

    namespace = str(item.get("namespace") or "").strip()
    tool_name = str(item.get("name") or "").strip()
    if not namespace.startswith("mcp__") or not tool_name:
        return

    arguments = {}
    raw_arguments = item.get("arguments")
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed_arguments = json.loads(raw_arguments)
            if isinstance(parsed_arguments, dict):
                arguments = parsed_arguments
        except Exception:
            arguments = {}

    # Native Codex MCP tool results are handled inside app-server, so xiaozhi
    # does not receive the result payload here. We still need to record that a
    # graph/UV/device MCP tool ran in this sentence; mutating graph tools will
    # mark the cached experiment state as refresh-required for the next turn.
    sync_server_mcp_payload_state(
        conn,
        tool_name=tool_name,
        payload=None,
        arguments=arguments,
    )


def _safe_filename(s: str) -> str:
    s = str(s or "session")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


class _PathFormatDict(dict):
    def __missing__(self, key: str) -> str:
        # Keep unknown placeholders unchanged.
        return "{" + key + "}"


def _norm_str(value: Any) -> str:
    return str(value or "").strip()


def _looks_like_experiment_record_or_flow_turn(user_text: str) -> bool:
    text = _normalize_whitespace(user_text).lower()
    if not text:
        return False

    compact = re.sub(r"\s+", "", text)
    if compact in {
        "准备好了",
        "准备好了。",
        "可以开始",
        "开始实验",
        "开始实验。",
    }:
        return False

    markers = (
        "已经",
        "已",
        "完成",
        "做完",
        "做好",
        "加完",
        "加入",
        "放入",
        "编号",
        "标记",
        "混匀",
        "搅拌",
        "观察",
        "颜色",
        "变色",
        "稳定",
        "照片",
        "拍照",
        "光谱",
        "扫描",
        "动力学",
        "下一步",
        "继续",
        "下一组",
        "导出",
        "报告",
        "结束实验",
        "done",
        "finished",
        "record",
        "next",
        "continue",
    )
    return any(marker in compact for marker in markers)


def _classify_timeout_first_turn_template(user_text: str) -> str:
    text = _normalize_whitespace(user_text)
    if not text:
        return "action"

    record_keywords = (
        "记录",
        "记一下",
        "记个",
        "填一下",
        "补记",
        "更正",
        "改成",
        "修改",
        "录入",
        "写入",
        "record",
        "log",
        "save",
    )
    theory_keywords = (
        "原理",
        "机理",
        "为什么",
        "讲解",
        "理论",
        "依据",
        "参考",
        "文献",
        "整个实验",
        "后面所有步骤",
        "全部步骤",
        "全流程",
        "theory",
        "mechanism",
        "principle",
        "reference",
        "workflow",
    )
    if any(keyword in text for keyword in record_keywords):
        return "record"
    if any(keyword in text for keyword in theory_keywords):
        return "theory"
    return "action"


def _timeout_first_turn_template_block(user_text: str) -> str:
    template_kind = _classify_timeout_first_turn_template(user_text)
    if template_kind == "record":
        return "\n".join(
            [
                "Timeout first-turn template: recording intake.",
                "- Treat the user's first goal as recording or correcting experiment data, not as a request for a full experiment recap.",
                "- First confirm only the minimum current-state fact needed to avoid writing the wrong field, step, or trial.",
                "- Once that minimum fact is clear, continue directly with the recording flow instead of detouring into theory or later steps.",
                "- Do not fetch broader reference/detail tools before hot-path record actions unless the field meaning is still too ambiguous to record safely.",
            ]
        )
    if template_kind == "theory":
        return "\n".join(
            [
                "Timeout first-turn template: theory or full-workflow request.",
                "- Acknowledge the broader explanation request, but first narrow the current state enough to avoid explaining the wrong stage of the experiment.",
                "- If the explanation depends on where the student currently is, ask one short current-state question before expanding.",
                "- Start with the most relevant current-stage explanation first; only expand to the broader workflow or references when the user explicitly still wants that broader pass.",
            ]
        )
    return "\n".join(
        [
            "Timeout first-turn template: operation or next-step guidance.",
            "- Treat the first goal as helping the student continue the experiment from the correct point, not as giving a full lecture.",
            "- Ask one narrow current-state question only if you still need it to place the student on the right step.",
            "- After that, give the immediate next action and the most relevant safety or attention point for that moment.",
        ]
    )


def _routing_context_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, str]:
    context: Dict[str, str] = {}
    for key in (
        "device_id",
        "chat_session_id",
        "model_session_key",
        "connection_session_id",
        "session_id",
        "transport_session_id",
        "user_id",
    ):
        value = _norm_str(kwargs.get(key, ""))
        if value:
            context[key] = value
    return context


def _experiment_context_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, str]:
    context: Dict[str, str] = {}
    for key in (
        "experiment_prewarm_wait_result",
        "experiment_prewarm_status",
        "experiment_prewarm_ready_level",
        "experiment_prewarm_trigger",
        "experiment_session_id",
        "experiment_current_step_id",
        "experiment_yaml_path",
        "experiment_overview_summary",
        "experiment_current_step_summary",
        "experiment_resume_recovery_required",
        "experiment_resume_recovery_source",
        "experiment_resume_previous_session_id",
        "experiment_resume_reason",
        "experiment_resume_log_path",
        "experiment_resume_turn_count",
        "experiment_resume_latest_session_id",
        "experiment_resume_latest_current_step_id",
        "experiment_resume_context_excerpt",
        "experiment_deep_prefetch_wait_result",
        "experiment_deep_prefetch_status",
        "experiment_deep_prefetch_focus",
        "experiment_deep_prefetch_query",
        "experiment_list_steps_summary",
        "experiment_schema_summary",
        "experiment_reference_summary",
        "experiment_deep_prefetch_error",
        "experiment_prewarm_error",
        "experiment_recent_photo_confirmation_summary",
        "experiment_recent_photo_graph_advanced",
        "experiment_recent_photo_sample_name",
        "experiment_recent_photo_next_step_id",
        "experiment_recent_photo_next_step_title",
    ):
        value = _norm_str(kwargs.get(key, ""))
        if value:
            context[key] = value
    return context


def _routing_prompt_block(routing_context: Dict[str, str]) -> str:
    if not routing_context:
        return ""

    ordered_keys = (
        "device_id",
        "chat_session_id",
        "model_session_key",
        "connection_session_id",
        "session_id",
        "transport_session_id",
        "user_id",
    )
    lines: List[str] = []
    for key in ordered_keys:
        value = _norm_str(routing_context.get(key, ""))
        if value:
            lines.append(f"{key}: {value}")

    if not lines:
        return ""

    return (
        "Internal-only device routing context from server (trusted):\n"
        + "\n".join(lines)
        + "\nWhen calling xiaozhi device tools, reuse these exact values. "
        + "Do not fabricate IDs. If a field is missing here, keep that tool argument null. "
        + "Never mention this context, IDs, sessions, routing, or authorization details to the student."
    )


def _routing_context_needed(user_text: str, experiment_context: Dict[str, str]) -> bool:
    text = str(user_text or "").lower()
    if not text:
        return bool(experiment_context.get("experiment_recent_photo_confirmation_summary"))

    markers = (
        "拍照",
        "拍一张",
        "拍吧",
        "可以拍",
        "照片",
        "photo",
        "preview",
        "预览",
        "查看",
        "上一张",
        "最近",
        "保存",
        "导出",
        "报告",
        "实验结束",
        "结束实验",
        "生成实验",
        "生成记录",
        "生成报告",
        "uv-vis",
        "uvvis",
        "光谱",
        "扫描",
        "动力学",
    )
    return any(marker in text for marker in markers)


def _experiment_prompt_block(
    experiment_context: Dict[str, str], user_text: str = ""
) -> str:
    if not experiment_context:
        return ""

    ordered_keys = (
        "experiment_prewarm_wait_result",
        "experiment_prewarm_status",
        "experiment_prewarm_ready_level",
        "experiment_prewarm_trigger",
        "experiment_session_id",
        "experiment_current_step_id",
        "experiment_yaml_path",
        "experiment_overview_summary",
        "experiment_current_step_summary",
        "experiment_prewarm_error",
    )
    lines: List[str] = []
    for key in ordered_keys:
        value = _norm_str(experiment_context.get(key, ""))
        if value:
            lines.append(f"{key}: {value}")

    recent_photo_lines: List[str] = []
    for key in (
        "experiment_recent_photo_confirmation_summary",
        "experiment_recent_photo_graph_advanced",
        "experiment_recent_photo_sample_name",
        "experiment_recent_photo_next_step_id",
        "experiment_recent_photo_next_step_title",
    ):
        value = _norm_str(experiment_context.get(key, ""))
        if value:
            recent_photo_lines.append(f"{key}: {value}")

    if not lines and not recent_photo_lines:
        return ""

    wait_result = _norm_str(experiment_context.get("experiment_prewarm_wait_result", ""))
    ready_level = _norm_str(
        experiment_context.get("experiment_prewarm_ready_level", "")
    )
    current_step_id = _norm_str(experiment_context.get("experiment_current_step_id", ""))
    deep_wait_result = _norm_str(
        experiment_context.get("experiment_deep_prefetch_wait_result", "")
    )

    recovery_required = _norm_str(
        experiment_context.get("experiment_resume_recovery_required", "")
    ).lower() in {"1", "true", "yes", "on"}
    recovery_block = ""
    if recovery_required:
        recovery_lines: List[str] = []
        for key in (
            "experiment_resume_recovery_required",
            "experiment_resume_recovery_source",
            "experiment_resume_previous_session_id",
            "experiment_resume_log_path",
            "experiment_resume_turn_count",
            "experiment_resume_latest_session_id",
            "experiment_resume_latest_current_step_id",
        ):
            value = _norm_str(experiment_context.get(key, ""))
            if value:
                recovery_lines.append(f"{key}: {value}")

        recovery_excerpt = experiment_context.get("experiment_resume_context_excerpt", "")
        if recovery_excerpt:
            recovery_lines.append("experiment_resume_context_excerpt:")
            recovery_lines.append(str(recovery_excerpt).strip())

        if recovery_lines:
            recovery_block = (
                "Lost-session recovery context from server (trusted):\n"
                + "\n".join(recovery_lines)
            )

    if experiment_context.get("experiment_session_id"):
        reuse_rule = (
            "Reuse this exact experiment_session_id for experiment_graph MCP calls. "
            "Do not call create_session again unless the user explicitly asks to restart or switch experiments, "
            "or the existing session proves invalid."
        )
    else:
        reuse_rule = (
            "If experiment_session_id is still missing and you need a session, create_session using "
            "experiment_yaml_path when appropriate."
        )

    deep_prefetch_lines: List[str] = []
    for key in (
        "experiment_deep_prefetch_wait_result",
        "experiment_deep_prefetch_status",
        "experiment_deep_prefetch_focus",
        "experiment_deep_prefetch_query",
        "experiment_list_steps_summary",
        "experiment_schema_summary",
        "experiment_reference_summary",
        "experiment_deep_prefetch_error",
    ):
        value = _norm_str(experiment_context.get(key, ""))
        if value:
            deep_prefetch_lines.append(f"{key}: {value}")

    parts: List[str] = []
    if lines:
        parts.append(
            "Experiment session context from server prewarm (trusted):\n"
            + "\n".join(lines)
        )
    if wait_result:
        if wait_result == "timeout":
            timeout_lines = [
                "Timeout-specific first-turn strategy for this experiment handoff:",
                "- The first-turn prewarm wait timed out before the server could confirm the full current-step context.",
                "- First stabilize the conversation with the narrowest current-state confirmation you actually need, instead of expanding into the whole experiment.",
                "- Do not invent the full experiment state, later steps, or a complete lab-handout summary on this timeout turn.",
                "- Do not fetch get_schema, get_experiment_reference, search_experiment_reference, or list_steps on this timeout turn unless the user explicitly asks for theory, references, or the broader workflow, or a safety-critical ambiguity makes that extra detail necessary.",
                "- If the user is providing data to record, asking to continue, or asking to correct an existing record, do not delay start_trial, add_field, add_fields, finish_trial, can_proceed, proceed_to_next_step, get_modifiable_records, or modify_record behind those broader reference/detail fetches.",
                "- Only pull get_schema before recording when the field meaning is still unclear enough that skipping it would risk writing the wrong value.",
            ]
            if current_step_id:
                timeout_lines.append(
                    f"- You still have current_step_id={current_step_id}; use it as the anchor, but keep the first answer concise and current-state focused."
                )
            else:
                timeout_lines.append(
                    "- If a key fact is still missing, ask one narrow current-state question first, then continue once that fact is confirmed."
                )
            parts.append("\n".join(timeout_lines))
            if not current_step_id:
                parts.append(_timeout_first_turn_template_block(user_text))
        if current_step_id:
            first_turn_lines = [
                "First real user turn strategy for this experiment handoff:",
                f"- Treat current_step_id={current_step_id} as the default center of the first reply.",
                "- Use the minimal trusted context already provided by the server before asking for more MCP detail.",
                "- On the first reply, tell the student what step they are on, what they should do now, and the immediate safety or attention points.",
                "- Do not proactively expand into the full experiment, later steps, or a full lab-handout style overview unless the user explicitly asks for that broader explanation.",
                "- Fetch deeper experiment_graph details or references only when the user explicitly asks for theory, the full workflow, later steps, schema details, or supporting references, or when extra detail is necessary to avoid a safety mistake.",
                "- Treat get_schema, get_experiment_reference, search_experiment_reference, and list_steps as detail/reference tools, not default first-turn tools.",
                "- Do not delay record/flow actions such as start_trial, add_field, add_fields, finish_trial, can_proceed, proceed_to_next_step, get_modifiable_records, or modify_record just because those detail/reference tools have not been fetched yet.",
            ]
            if ready_level == "minimal_ready":
                first_turn_lines.append(
                    "- Richer step context may still be warming in the background, so prefer a concise current-step answer over a broad lecture on this first turn."
                )
            parts.append("\n".join(first_turn_lines))
        else:
            parts.append(
                "First real user turn strategy for this experiment handoff:\n"
                "- The server has not confirmed a trusted current_step_id yet.\n"
                "- Do not invent the full experiment state or proactively narrate the whole experiment.\n"
                "- Use any trusted recovery context already provided, and if a key fact is still missing, ask only a narrow current-state question before expanding."
            )
    if current_step_id:
        if ready_level == "minimal_ready":
            parts.append(
                "Minimal-ready snapshot from server: continue from "
                f"current_step_id={current_step_id}. Richer step details may still be warming in the background."
            )
    if recovery_block:
        parts.append(recovery_block)
        latest_current_step_id = _norm_str(
            experiment_context.get("experiment_resume_latest_current_step_id", "")
        )
        if latest_current_step_id:
            parts.append(
                "Trusted recovery snapshot: continue from "
                f"current_step_id={latest_current_step_id} unless a fresh "
                "experiment_graph read immediately proves the current session is already past it."
            )
        parts.append(
            "The previous experiment_graph session is gone. The server has already created a fresh "
            "experiment_session_id for this same device. Continue the unfinished experiment in the "
            "current session instead of restarting from scratch. Recover only facts that are clearly "
            "supported by the device-log excerpt or by fixed YAML defaults; if a required field cannot "
            "be recovered confidently, ask only for that missing field."
        )
        parts.append(
            "Narration guard for recovery:\n"
            "- Do not describe this recovery as an experiment-graph outage, disconnect, or interface failure.\n"
            "- A lost previous session is not the same thing as 'the experiment graph interface did not connect'.\n"
            "- Keep any recovery explanation student-facing and minimal; focus on the current experiment action instead of backend causes."
        )
    parts.append(
        "Tool-failure narration guard:\n"
        "- Only say an interface, MCP tool, or device is disconnected, unavailable, occupied, or used by another program when a tool call on the current turn actually returned that failure.\n"
        "- Do not infer interface failure from recovery context, session recreation, old-session loss, or the absence of a fresh tool call.\n"
        "- If no current-turn tool failure exists, do not speculate about backend causes; continue with the current experiment action or ask one narrow action-level question."
    )
    if deep_prefetch_lines:
        parts.append(
            "Deep-prefetched experiment detail context from server (trusted):\n"
            + "\n".join(deep_prefetch_lines)
        )
        parts.append(
            "Use this deep-prefetched detail context first before calling list_steps, "
            "get_schema, search_experiment_reference, or get_experiment_reference again. "
            "Only fetch again if the student's question still needs missing or fresher detail."
        )
        if deep_wait_result == "timeout":
            parts.append(
                "The server only gave a micro-budget to this deep-prefetch on the current turn. "
                "If some detail is still missing, answer from the trusted current-step context first, "
                "then fetch only the narrow missing detail."
            )
    if recent_photo_lines:
        parts.append(
            "Recent trusted photo confirmation context from server:\n"
            + "\n".join(recent_photo_lines)
        )
        parts.append(
            "Treat this recent photo confirmation as trusted short-term state. "
            "Do not ask to retake the same sample photo unless the user explicitly asks for a retake "
            "or a fresh experiment_graph read clearly proves the confirmation is still missing."
        )
    if current_step_id:
        parts.append(
            "Experiment graph alignment guard:\n"
            f"- Treat trusted current_step_id={current_step_id} as the only safe step anchor until this turn's tool calls prove a change.\n"
            "- Do not verbally move the student to a later experiment step unless the current turn actually called experiment_graph state/flow tools and their results support that move.\n"
            "- When the student reports completion, observations, colors, timings, photos, or scan results for the current step, update experiment_graph records first, then narrate the next step.\n"
            "- Do not narrate backend bookkeeping such as '我先记下…', '我接着确认记录项…', or '我把这一步写回图谱…'; either give the next student-facing instruction or ask only for the still-missing field.\n"
            "- If you have not called get_step, get_state, get_progress_summary, get_current_progress, start_trial, add_field, add_fields, finish_trial, can_proceed, proceed_to_next_step, redirect_to_step, redo_trial, or modify_record on this turn, stay anchored to the trusted current step instead of improvising later steps from old dialogue, prefetched summaries, or memory."
        )
        parts.append(
            "Student sidetrack question rule:\n"
            "- If the latest user message asks a conceptual, safety, reagent, instrument, data-meaning, troubleshooting, or other explanatory question, and it does not itself report completion, observations, measurements, photos, scan results, corrections, or a request to advance, answer the question first.\n"
            "- For these sidetrack questions, do not call experiment-graph tools solely to remind an unfinished current step, do not repeat the whole unfinished step, and do not tell the student they must finish the step before you answer.\n"
            "- During and after a sidetrack question, keep experiment_graph anchored at the original current step: do not start/finish/proceed a trial, write placeholder data, or change current_step_id just because the question was answered.\n"
            "- If the student then says '可以继续', '继续做实验', or similar, treat that only as permission to resume guidance for the same current step. It is not evidence that the current step is complete and must not by itself justify proceed_to_next_step.\n"
            "- After answering, append exactly one short Chinese sentence: '我们现在能继续做实验了吗？'\n"
            "- If the experiment is already at a completed final step and the user asks a question, answer normally; do not keep urging the student to complete the final step again."
        )
        if _looks_like_experiment_record_or_flow_turn(user_text):
            parts.append(
                "Current-turn experiment_graph write barrier:\n"
                "- The latest user message looks like a completion report, observation, measurement result, correction, or flow-control request for the active experiment.\n"
                "- Do not decide, draft, or announce the next physical action from memory or YAML order alone. First make experiment-graph the source of truth for the transition.\n"
                "- Before your final student-facing answer, call the connected experiment-graph MCP tools needed to make the graph truthful: get_state/get_current_progress if the active step may be stale, start_trial if no active trial exists, add_field/add_fields for the confirmed facts, finish_trial(validate=true) when required fields are complete, and proceed_to_next_step when you are about to give the next step. You may call can_proceed first, but proceed_to_next_step itself is the required transition gate and must return ok=true before you speak the next step.\n"
                "- Hot path: when add_field/add_fields returns ok=true and missing_fields is empty, immediately call finish_trial(validate=true). When finish_trial returns ok=true and you plan to give the next physical action, immediately call proceed_to_next_step. Do not insert extra schema/reference reads or long reasoning between these calls.\n"
                "- Confirmation-step completion rule: when the active step's interaction.fast_path_mode is confirmation_step, or its capabilities/tags include step_confirmation/confirmation_step, and the student says a short completion report such as '做好了', '完成了', '做完了', '已经做好了', '已经加好了', '已经混匀了', or '已经混匀好了', treat that as confirmation of the current step's required boolean action fields unless the message contains a negative/uncertain phrase. For this case, call start_trial if needed, add_fields with all required boolean fields set to true, finish_trial(validate=true), and proceed_to_next_step before speaking. Do not ask the student to repeat the same step using more specific wording.\n"
                "- Always copy the full exact experiment_session_id into experiment-graph calls. A session_id containing '...' or '…' is invalid; replace it with the latest full session id before retrying.\n"
                "- If a graph write, finish, proceed, redirect, or export tool returns ok=false or a session_id-not-found error, retry once with the latest full exact session id. If it still fails, do not claim the record, photo, step transition, group transition, or export succeeded.\n"
                "- Only pass group_number to photo/export/UV tools when the experiment YAML explicitly has group_number fields or workflow.group_start_steps. In non-group experiments, current_group_number=1 is bookkeeping only and must not be used for storage.\n"
                "- Before calling xiaozhi_take_photo, the latest student message must explicitly authorize taking a photo. If it does not, do not call the tool on this turn; ask only '现在可以拍照吗？'. If a photo tool call is rejected with 'user rejected MCP tool call', treat that as missing authorization, not as a camera, device, network, or permission failure.\n"
                "- Photo authorization hot path: if the latest student message explicitly says '可以拍照', '拍吧', '现在拍', '能拍', '行，拍', or another clear short photo authorization, your next action must be the connected xiaozhi_take_photo MCP tool call. Do not explain first, do not wait, do not ask again, do not call unrelated overview/reference/schema tools first, and do not produce a student-facing reply before the photo tool returns.\n"
                "- Retake photo hot path: if the latest student message says '重拍/重新拍/再拍/补拍 N号样品照片', call xiaozhi_take_photo immediately with photo_name='N号样品照片' and append_timestamp=true. Treat this as an extra photo capture only: do not redirect_to_step, redo_trial, modify_record, cancel prior records, or roll back completed experiment steps.\n"
                "- For numbered sample photos, every new xiaozhi_take_photo call must preserve old files by using append_timestamp=true, producing names like 'N号样品照片_YYYYMMDD_HHMMSS'.\n"
                "- If the latest student message explicitly authorizes taking a photo and the current graph step is a required photo-confirmation step, you must call xiaozhi_take_photo on this turn. Do not ask the student to take the photo manually, do not say '拍完告诉我', and do not move on until the tool succeeds and the graph record is updated.\n"
                "- If the current step requires a photo and the student has granted permission, call the connected xiaozhi_take_photo tool before claiming the photo exists; then write the returned photo result into the graph before moving on.\n"
                "- Prefer add_fields with native JSON booleans/numbers for obvious current-step confirmations; do not write boolean facts as strings such as \"true\" unless the schema requires a string.\n"
                "- Only after proceed_to_next_step returns ok=true may you use its returned current_step_id/current step message to decide and speak the next physical action.\n"
                "- If proceed_to_next_step fails, or if no proceed_to_next_step result was obtained on this turn, do not give the next step; remain on the current graph step and ask only for the missing action-level detail.\n"
                "- A final answer that says the step is complete, gives the next physical action, enters the next group, or ends/exports the experiment without those successful MCP calls and returned graph state is invalid.\n"
                "- If a required field is missing or a graph tool rejects the write/advance, do not give the next step; ask only for the missing action-level detail."
            )
    parts.append(
        "MCP execution guard:\n"
        "- The experiment-graph, UV-Vis, and xiaozhi device capabilities for this runtime are already exposed through connected MCP tools when configured.\n"
        "- Student-facing answers must not contain English scratch text, self-corrections, or reasoning artifacts such as 'correction:', 'not right', 'Need to', or 'I should'. If you notice a draft mistake, silently replace it and output only the corrected Chinese lab instruction.\n"
        "- Local shell/commandExecution may be available for environment checks, but experiment state changes, photos, UV-Vis actions, and report export must use the connected MCP tools instead of shell substitutes.\n"
        "- Do not launch local MCP server scripts, wrapper processes, or ad-hoc Python MCP clients from shell commands just to inspect or call those capabilities.\n"
        "- When the user asks to generate/export an experiment record or report, do not use shell, commandExecution, or filesystem inspection as a substitute; call the connected experiment-graph export_records_to_yaml tool and wait for ok=true.\n"
        "- After export_records_to_yaml returns ok=true, tell the student exactly '实验报告已经生成'. Do not mention YAML, PDF, file_path, yaml_path, pdf_path, URLs, or local directories unless the user explicitly asks for the path.\n"
        "- In particular, do not run local paths such as experimental_graph_mcp.py, experiment_graph_mcp_server.py, uvvis_http_wrapper.py, or device_trigger_mcp_server.py from shell or Python probes; call the connected MCP tools directly instead.\n"
        "- If a needed MCP tool is unavailable on this turn, continue with non-tool guidance or explain what is missing, but do not bypass the runtime by spawning replacement shell-based MCP sessions."
    )
    parts.append(
        "UV-Vis execution guard:\n"
        "- The UV-Vis MCP tools are available in this runtime.\n"
        "- If the current experiment step or local prompt explicitly requires uvvis_prepare_dark_current, call only uvvis_prepare_dark_current for that preparation step and wait for its result before advancing.\n"
        "- Do not call uvvis_measure_spectra with ready_for_samples=false after uvvis_prepare_dark_current unless the active experiment step explicitly asks for an additional separate blank scan.\n"
        "- Do not start any real sample scan until the experiment graph has advanced to the sample-loading or sample-recording step and the student has confirmed the real samples are loaded.\n"
        "- Before re-measuring a shared pure-water blank, inspect the shared uv_data_common directory for reusable blank artifacts and skip the blank scan when reusable data already exists there.\n"
        "- If the shared pure-water blank is missing, keep the positions empty and call uvvis_measure_spectra with ready_for_samples=false once to prepare the shared prerequisites in the background before you ask the student to place pure water.\n"
        "- After those shared prerequisites are ready, ask for six pure-water cuvettes only when the shared pure-water blank is still missing, then use uvvis_measure_spectra with ready_for_samples=true to record the pure-water blank.\n"
        "- For the actual batch spectra measurement after the cuvettes are loaded, use uvvis_measure_spectra with ready_for_samples=true.\n"
        "- For kinetics runs, use uvvis_measure_kinetics. Use uvvis_session when you need to acquire or refresh the UV-Vis lease/session first.\n"
        "- Do not verbalize internal orchestration rules such as '先根据上一步返回结果判断是否可复用', '只有在主说话人明确回报…后才调用…', '若工具提示…则不要重复测量', or any session/tool-call wording; speak only the student's current physical action or concise readiness prompt.\n"
        "- Do not say you are starting a UV-Vis scan, baseline, or kinetics run unless one of those UV-Vis tools was actually called on the current turn."
    )
    parts.append(reuse_rule)
    return "\n\n".join(parts)


def _experiment_bootstrap_prompt_block(
    experiment_context: Dict[str, str], experiment_yaml_path: str, user_text: str = ""
) -> str:
    yaml_path = _norm_str(experiment_yaml_path)
    if not yaml_path or experiment_context.get("experiment_session_id"):
        return ""

    return (
        "No experiment_graph session has been confirmed for this Codex thread yet.\n"
        f"- The configured experiment YAML is: {yaml_path}\n"
        "- If you have already called experiment-graph create_session earlier in this same Codex thread and received a full session_id, reuse that exact session_id even if the xiaozhi server-side context below still shows experiment_session_id as missing. Do not create another graph session just because the xiaozhi-side prewarm context is empty.\n"
        "- If this Codex thread has not yet received any experiment_graph session_id and the latest user message is starting or continuing the lab, confirming readiness, reporting a completed action, authorizing a scan/photo, asking for next step/group, or giving measurement/observation data, your next backend action must be experiment-graph create_session with that exact yaml_path.\n"
        "- After create_session succeeds, immediately call get_current_progress or get_state for the returned full session_id and remember that session_id for later turns in this thread.\n"
        "- Before create_session/get_current_progress succeeds, do not claim dark current, UV-Vis scan, photo, record writing, step transition, group transition, or experiment completion has happened.\n"
        "- If a UV-Vis or photo tool is required on this same turn, create or recover the graph session first, then call the device/instrument MCP tool, then write the result back to experiment-graph before speaking the next physical step.\n"
    )


class _CodexSession:
    def __init__(self, config: Dict, session_key: str) -> None:
        self.session_key = session_key
        self.codex_bin_configured = str(config.get("codex_bin", "codex.cmd")).strip()
        self.codex_bin = self._resolve_codex_bin(self.codex_bin_configured)
        self.model = config.get("model_name") or config.get("model") or "gpt-5.2"
        self.workspace = str(Path(config.get("workspace", os.getcwd())).resolve())
        self.auto_approve = bool(config.get("auto_approve", True))
        self.approval_policy = config.get("approval_policy")
        if self.approval_policy is None and self.auto_approve:
            self.approval_policy = "never"
        self.network_access = bool(config.get("network_access", True))
        self.thread_sandbox = config.get("sandbox", "workspace-write")
        self.sandbox_policy = config.get("sandbox_policy")
        self.effort = config.get("effort")
        self.summary = config.get("summary")
        self.service_tier = config.get("service_tier") or config.get("serviceTier")
        self.system_prompt_mode = (config.get("system_prompt_mode") or "first_turn").lower()
        self.restart_on_system_prompt_change = bool(
            config.get("restart_on_system_prompt_change", False)
        )
        self.bootstrap_mode = (config.get("bootstrap_mode") or "none").lower()
        configured_yaml_path = config.get("yaml_path")
        resolved_yaml_path = _resolve_optional_path(configured_yaml_path, _SERVER_ROOT)
        self.experiment_yaml_path = str(
            resolved_yaml_path or configured_yaml_path or ""
        ).strip()
        self.api_key = config.get("api_key")
        self.export_api_key = bool(config.get("export_api_key", False))
        self.env_overrides = config.get("env", {}) or {}
        self.runtime_config = config.get("runtime_config", {}) or {}
        configured_mcp_settings_path = config.get("mcp_settings_path")
        self.mcp_settings_path = _resolve_optional_path(
            configured_mcp_settings_path,
            _SERVER_ROOT,
        ) or _DEFAULT_MCP_SETTINGS_PATH
        configured_codex_config_path = config.get("codex_config_path")
        self.codex_config_path = _resolve_optional_path(
            configured_codex_config_path,
            _SERVER_ROOT,
        ) or _DEFAULT_CODEX_CONFIG_PATH
        self.app_server_config_overrides = _merge_app_server_config_overrides(
            config.get("app_server_config_overrides")
        )
        self.disable_plugins = bool(config.get("disable_plugins", True))
        self.emit_events = bool(config.get("emit_events", False))
        self.thinking_mode = (config.get("thinking_mode") or "off").lower()
        self.show_actions = bool(config.get("show_actions", False))
        self.thinking_debug = bool(config.get("thinking_debug", False))

        # ---- Stream logging toggles ----
        # log_stream: enables writing stream output to file + end-of-turn info log
        self.log_stream = bool(config.get("log_stream", True))
        # stream_debug: enables per-token debug logs (config-only gate)
        self.stream_debug = bool(config.get("stream_debug", False))
        # log_events: also write emitted event dicts (thinking/action) into SAME log file
        self.log_events = bool(config.get("log_events", True))
        # raw_stream_log: write raw app-server messages to a separate file
        self.raw_stream_log = bool(config.get("raw_stream_log", False))
        # event_max_chars: truncate long event JSON payloads (0 = no truncation)
        self.event_max_chars = int(config.get("event_max_chars", 8000))

        default_stream_log_path = str(
            Path(self.workspace) / f"/log/codex_stream_{_safe_filename(session_key)}.log"
        )
        self.stream_log_path_template = str(
            config.get("stream_log_path", default_stream_log_path)
        )
        self.stream_log_path = self.stream_log_path_template
        self._stream_flush_bytes = int(config.get("stream_flush_bytes", 4096))
        default_raw_log_path = str(
            Path(self.workspace) / f"/log/codex_stream_raw_{_safe_filename(session_key)}.log"
        )
        self.raw_stream_log_path_template = str(
            config.get("raw_stream_log_path", default_raw_log_path)
        )
        self.raw_stream_log_path = self.raw_stream_log_path_template
        self._raw_stream_flush_bytes = int(config.get("raw_stream_flush_bytes", 8192))

        self.proc: Optional[subprocess.Popen] = None
        self._proc_windows_job = None
        self.q: Optional[queue.Queue] = None
        self.thread_id: Optional[str] = None
        self._req_id = 0
        self._last_system_prompt: Optional[str] = None
        self._system_prompt_sent = False
        self._bootstrap_history = True
        self._active_turn_id: Optional[str] = None
        self._restart_required = False
        self._restart_reason: Optional[str] = None
        self._suppress_powershell_profile_warning_lines = 0
        self._runtime_codex_home: Optional[Path] = None
        self._lock = threading.Lock()
        self._watched_config_paths = self._build_watched_config_paths()
        self._watched_config_state = _snapshot_watched_paths(self._watched_config_paths)

    @staticmethod
    def _looks_like_filesystem_path(path_text: str) -> bool:
        text = str(path_text or "").strip()
        return bool(text) and (
            os.path.isabs(text)
            or "/" in text
            or "\\" in text
            or text.startswith("~")
        )

    @classmethod
    def _resolve_codex_bin(cls, configured_bin: str) -> str:
        raw = str(configured_bin or "").strip() or "codex.cmd"
        if not cls._looks_like_filesystem_path(raw):
            return raw

        configured_path = Path(raw).expanduser()
        if configured_path.exists():
            return str(configured_path)

        matched_candidate = cls._discover_vscode_extension_codex_bin(configured_path)
        if matched_candidate:
            logger.bind(tag=TAG).warning(
                "configured codex_bin was missing; auto-discovered a newer VS Code extension binary: "
                f"configured={configured_path}, resolved={matched_candidate}"
            )
            return matched_candidate

        return str(configured_path)

    @staticmethod
    def _discover_vscode_extension_codex_bin(configured_path: Path) -> str:
        try:
            parts = configured_path.parts
        except Exception:
            return ""

        try:
            ext_idx = next(
                idx for idx, value in enumerate(parts) if str(value).lower() == "extensions"
            )
        except StopIteration:
            return ""

        if ext_idx + 2 >= len(parts):
            return ""

        extensions_dir = Path(*parts[: ext_idx + 1])
        configured_extension_dir = parts[ext_idx + 1]
        relative_suffix = Path(*parts[ext_idx + 2 :])
        if not str(configured_extension_dir).startswith("openai.chatgpt-"):
            return ""
        if not extensions_dir.exists():
            return ""

        candidate_dirs = sorted(
            (
                item
                for item in extensions_dir.iterdir()
                if item.is_dir() and item.name.startswith("openai.chatgpt-")
            ),
            key=lambda item: item.name,
            reverse=True,
        )
        for candidate_dir in candidate_dirs:
            candidate_path = candidate_dir / relative_suffix
            if candidate_path.exists():
                return str(candidate_path)
        return ""

    def _build_watched_config_paths(self) -> List[Path]:
        candidates = [
            self.mcp_settings_path,
            self.codex_config_path,
            _resolve_optional_path(Path(self.workspace) / ".mcp.json"),
            _resolve_optional_path(Path(self.workspace) / ".codex" / "config.toml"),
        ]
        paths: List[Path] = []
        seen = set()
        for candidate in candidates:
            if candidate is None:
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            paths.append(candidate)
        return paths

    def _consume_config_change_reason(self) -> Optional[str]:
        current_state = _snapshot_watched_paths(self._watched_config_paths)
        changed_paths = [
            path
            for path, state in current_state.items()
            if self._watched_config_state.get(path) != state
        ]
        if not changed_paths:
            return None

        self._watched_config_state = current_state
        labels = ", ".join(Path(path).name for path in changed_paths[:4])
        if len(changed_paths) > 4:
            labels += ", ..."
        return labels

    def _resolve_runtime_codex_home_path(self) -> Optional[Path]:
        base_home = self.codex_config_path.parent if self.codex_config_path else None
        if base_home is None:
            return None
        try:
            return (base_home / ".xiaozhi-runtime" / _safe_filename(self.session_key)).resolve()
        except OSError:
            return base_home / ".xiaozhi-runtime" / _safe_filename(self.session_key)

    def _mirror_codex_home_support_files(
        self,
        source_home: Path,
        target_home: Path,
    ) -> None:
        for file_name in _RUNTIME_CODEX_HOME_COPY_FILES:
            source_path = source_home / file_name
            if not source_path.is_file():
                continue
            shutil.copy2(source_path, target_home / file_name)

        for dir_name in _RUNTIME_CODEX_HOME_DISABLED_DIRS:
            target_path = target_home / dir_name
            if not target_path.exists():
                continue
            if target_path.is_dir():
                shutil.rmtree(target_path, ignore_errors=True)
            else:
                try:
                    target_path.unlink()
                except OSError:
                    pass

        for dir_name in _RUNTIME_CODEX_HOME_COPY_DIRS:
            source_path = source_home / dir_name
            target_path = target_home / dir_name
            if not source_path.is_dir() or target_path.exists():
                continue
            shutil.copytree(source_path, target_path)

    def _prepare_runtime_codex_home(self) -> Optional[Path]:
        _run_codex_mcp_startup_hooks(self.mcp_settings_path)
        server_configs = _load_codex_mcp_server_configs(self.mcp_settings_path)
        if not server_configs:
            self._runtime_codex_home = None
            return None
        _inject_experiment_yaml_env(server_configs, self.experiment_yaml_path)

        runtime_codex_home = self._resolve_runtime_codex_home_path()
        if runtime_codex_home is None:
            return None

        source_home = self.codex_config_path.parent
        try:
            runtime_codex_home.mkdir(parents=True, exist_ok=True)
            if source_home.exists():
                self._mirror_codex_home_support_files(source_home, runtime_codex_home)

            base_config = _load_toml_document(self.codex_config_path)
            merged_config = _merge_codex_mcp_server_configs(base_config, server_configs)
            if self.model:
                merged_config["model"] = self.model
            if self.effort:
                merged_config["model_reasoning_effort"] = self.effort
            (runtime_codex_home / "config.toml").write_text(
                _dump_toml_document(merged_config),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.bind(tag=TAG).warning(
                "failed to prepare runtime CODEX_HOME; falling back to default config: "
                f"session={self.session_key} error={exc}"
            )
            return None

        self._runtime_codex_home = runtime_codex_home
        return runtime_codex_home

    def _build_app_server_command(self) -> List[str]:
        cmd = [self.codex_bin, "app-server"]

        if self.disable_plugins:
            cmd.extend(["--disable", "plugins"])

        for override in self.app_server_config_overrides:
            cmd.extend(["-c", override])

        return cmd

    def _next_id(self) -> int:
        rid = self._req_id
        self._req_id += 1
        return rid

    def _resolve_log_path(self, template: str, context: Dict[str, Any]) -> str:
        text = str(template or "")
        if not text or "{" not in text:
            return text

        safe_context = _PathFormatDict()
        safe_context["session_key"] = _safe_filename(self.session_key)
        for key, value in (context or {}).items():
            normalized = _norm_str(value)
            if normalized:
                safe_context[key] = _safe_filename(normalized)
        if "device_id" not in safe_context:
            safe_context["device_id"] = "unknown_device"

        try:
            return text.format_map(safe_context)
        except Exception:
            return text

    def _append_stream_log(self, text: str) -> None:
        """Append to per-session stream log file (best-effort)."""
        if not self.log_stream or not self.stream_log_path or not text:
            return
        try:
            Path(self.stream_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.stream_log_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:
            logger.bind(tag=TAG).warning(f"codex stream log write failed: {exc}")

    def _append_raw_log(self, text: str) -> None:
        """Append to per-session raw log file (best-effort)."""
        if not self.raw_stream_log or not self.raw_stream_log_path or not text:
            return
        try:
            Path(self.raw_stream_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.raw_stream_log_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:
            logger.bind(tag=TAG).warning(f"codex raw log write failed: {exc}")

    def _should_suppress_stderr_line(self, text: str) -> bool:
        if self._suppress_powershell_profile_warning_lines > 0:
            if _is_powershell_profile_warning_continuation(text):
                self._suppress_powershell_profile_warning_lines -= 1
                return True
            self._suppress_powershell_profile_warning_lines = 0

        if not _should_suppress_stderr_warning(text):
            return False

        if _is_benign_powershell_profile_warning(text):
            # PowerShell execution-policy errors arrive as a short multi-line
            # block; keep the continuation lines out of warning logs too.
            self._suppress_powershell_profile_warning_lines = 6
        return True

    def _pump_stderr(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            text = _decode_stderr_line(line).rstrip()
            if not text:
                continue
            recovery_reason = _recoverable_stderr_reason(text)
            if recovery_reason:
                with self._lock:
                    self._restart_required = True
                    self._restart_reason = recovery_reason
            if self._should_suppress_stderr_line(text):
                logger.bind(tag=TAG).debug(f"codex stderr suppressed: {text}")
                continue
            logger.bind(tag=TAG).warning(f"codex stderr: {text}")

    def _write_config(self, key: str, value) -> None:
        _send(
            self.proc,
            {
                "method": "config/value/write",
                "id": self._next_id(),
                "params": {"keyPath": key, "mergeStrategy": "replace", "value": value},
            },
        )
        _wait_result(self.proc, self.q, self._req_id - 1, self.auto_approve)

    def _start_process(self) -> None:
        env = os.environ.copy()
        if self.api_key and self.export_api_key and "OPENAI_API_KEY" not in env:
            env["OPENAI_API_KEY"] = self.api_key
        if "${" in self.workspace:
            raise ValueError(
                f"Codex workspace contains an unresolved config template: {self.workspace}"
            )
        if not Path(self.workspace).exists():
            raise FileNotFoundError(
                f"Codex workspace not found: {self.workspace}"
            )
        if self._looks_like_filesystem_path(self.codex_bin) and not Path(
            self.codex_bin
        ).expanduser().exists():
            raise FileNotFoundError(
                "Codex binary not found. "
                f"configured={self.codex_bin_configured}, resolved={self.codex_bin}"
            )
        codex_bin_dir = str(Path(self.codex_bin).expanduser().resolve().parent)
        if codex_bin_dir and Path(codex_bin_dir).exists():
            env["PATH"] = codex_bin_dir + os.pathsep + env.get("PATH", "")
        env.update(self.env_overrides)
        runtime_codex_home = self._prepare_runtime_codex_home()
        if runtime_codex_home is not None:
            env["CODEX_HOME"] = str(runtime_codex_home)

        launch_cmd = self._build_app_server_command()
        self.proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workspace,
            env=env,
            start_new_session=(os.name != "nt"),
        )
        self._proc_windows_job = _assign_process_to_windows_job(
            self.proc,
            _create_windows_kill_job(),
        )
        self._watched_config_state = _snapshot_watched_paths(self._watched_config_paths)

        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self.q = queue.Queue()
        _StdoutReader(self.proc, self.q).start()

        _send(
            self.proc,
            {
                "method": "initialize",
                "id": self._next_id(),
                "params": {
                    "clientInfo": {
                        "name": "xiaozhi_codex_provider",
                        "title": "Xiaozhi Codex Provider",
                        "version": "0.0.1",
                    }
                },
            },
        )
        _wait_result(self.proc, self.q, self._req_id - 1, self.auto_approve)
        _send(self.proc, {"method": "initialized", "params": {}})

        _send(
            self.proc,
            {"method": "account/read", "id": self._next_id(), "params": {"refreshToken": False}},
        )
        auth = _wait_result(self.proc, self.q, self._req_id - 1, self.auto_approve)
        if auth.get("requiresOpenaiAuth") and auth.get("account") is None:
            if not self.api_key:
                raise RuntimeError("Codex app-server requires auth; set api_key in config.")
            _send(
                self.proc,
                {
                    "method": "account/login/start",
                    "id": self._next_id(),
                    "params": {"type": "apiKey", "apiKey": self.api_key},
                },
            )
            _wait_result(self.proc, self.q, self._req_id - 1, self.auto_approve)

        for key, value in self.runtime_config.items():
            try:
                self._write_config(key, value)
            except Exception as exc:
                logger.bind(tag=TAG).warning(f"codex config write failed: {key} ({exc})")

        if self.emit_events and self.thinking_mode != "off" and "hide_agent_reasoning" not in self.runtime_config:
            try:
                self._write_config("hide_agent_reasoning", False)
            except Exception as exc:
                logger.bind(tag=TAG).warning(
                    f"codex config write failed: hide_agent_reasoning ({exc})"
                )

        thread_params = {"model": self.model, "cwd": self.workspace, "sandbox": self.thread_sandbox}
        if self.service_tier:
            thread_params["serviceTier"] = self.service_tier
        if self.approval_policy:
            thread_params["approvalPolicy"] = self.approval_policy
        _send(self.proc, {"method": "thread/start", "id": self._next_id(), "params": thread_params})
        thread_result = _wait_result(self.proc, self.q, self._req_id - 1, self.auto_approve)
        self.thread_id = str(thread_result["thread"]["id"])

        self._system_prompt_sent = False
        self._bootstrap_history = True

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            config_change_reason = self._consume_config_change_reason()
            if config_change_reason:
                logger.bind(tag=TAG).info(
                    "restarting codex session after config change: "
                    f"session={self.session_key} files={config_change_reason}"
                )
                self.close()
            elif self._restart_required:
                logger.bind(tag=TAG).info(
                    "restarting codex session after recoverable stderr: "
                    f"session={self.session_key} reason={self._restart_reason or 'unknown'}"
                )
                self.close()
            else:
                return
        self._start_process()

    def close(self) -> None:
        if not self.proc:
            return
        proc = self.proc
        windows_job_handle = self._proc_windows_job
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            _terminate_process_tree(
                proc,
                timeout_seconds=5.0,
                windows_job_handle=windows_job_handle,
            )
        except Exception:
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
        self.proc = None
        self._proc_windows_job = None
        self.q = None
        self.thread_id = None
        self._active_turn_id = None
        self._restart_required = False
        self._restart_reason = None
        runtime_codex_home = self._runtime_codex_home
        self._runtime_codex_home = None
        if runtime_codex_home and runtime_codex_home.exists():
            try:
                shutil.rmtree(runtime_codex_home, ignore_errors=True)
            except Exception:
                pass

    def _restart(self) -> None:
        self.close()
        self.start()

    def _compose_prompt(
        self,
        dialogue: List[Dict],
        routing_context: Optional[Dict[str, str]] = None,
        experiment_context: Optional[Dict[str, str]] = None,
    ) -> str:
        history, last_user, tail = _split_dialogue(dialogue)
        strategy_user_text = last_user or ""
        tool_context = _build_tool_context(tail)
        if tool_context:
            if last_user:
                last_user = f"{last_user}\n\nTool results:\n{tool_context}"
            else:
                last_user = f"Tool results:\n{tool_context}"

        experiment_context = experiment_context or {}
        routing_block = (
            _routing_prompt_block(routing_context or {})
            if _routing_context_needed(strategy_user_text, experiment_context)
            else ""
        )
        if routing_block:
            if last_user:
                last_user = f"{last_user}\n\n{routing_block}"
            else:
                last_user = routing_block

        duration_note = textUtils.build_spoken_duration_normalization_note(
            strategy_user_text
        )
        if duration_note:
            if last_user:
                last_user = f"{last_user}\n\n{duration_note}"
            else:
                last_user = duration_note

        experiment_block = _experiment_prompt_block(
            experiment_context,
            user_text=strategy_user_text,
        )
        if experiment_block:
            if last_user:
                last_user = f"{last_user}\n\n{experiment_block}"
            else:
                last_user = experiment_block

        bootstrap_block = _experiment_bootstrap_prompt_block(
            experiment_context,
            self.experiment_yaml_path,
            user_text=strategy_user_text,
        )
        if bootstrap_block:
            if last_user:
                last_user = f"{last_user}\n\n{bootstrap_block}"
            else:
                last_user = bootstrap_block

        photo_hot_path_block = _photo_authorization_hot_path_prompt_block(
            strategy_user_text
        )
        if photo_hot_path_block:
            if last_user:
                last_user = f"{last_user}\n\n{photo_hot_path_block}"
            else:
                last_user = photo_hot_path_block

        if not last_user:
            return ""

        system_prompt = _extract_system_prompt(history)
        current_prompt_fp = _system_prompt_restart_fingerprint(system_prompt)
        if (
            self._last_system_prompt is not None
            and current_prompt_fp != self._last_system_prompt
        ):
            if self.restart_on_system_prompt_change:
                logger.bind(tag=TAG).info(
                    "restarting codex session after system prompt change: "
                    f"session={self.session_key}"
                )
                self._restart()
            else:
                logger.bind(tag=TAG).warning(
                    "codex system prompt changed; keeping existing thread to preserve "
                    f"experiment context: session={self.session_key}"
                )
        self._last_system_prompt = current_prompt_fp
        include_system = (
            system_prompt
            and self.system_prompt_mode in ("always", "first_turn")
            and (self.system_prompt_mode == "always" or not self._system_prompt_sent)
        )
        if include_system and _user_already_contains_system_prompt(system_prompt, last_user):
            include_system = False

        if self._bootstrap_history and self.bootstrap_mode != "none":
            transcript = _build_transcript(history)
            parts: List[str] = []
            if include_system:
                parts.append(system_prompt)
            if transcript:
                parts.append("Conversation so far:\n" + transcript)
            parts.append(last_user)
            prompt = "\n\n".join(parts)
            if include_system:
                self._system_prompt_sent = True
            self._bootstrap_history = False
            return prompt

        if include_system:
            self._system_prompt_sent = True

            return f"{system_prompt}\n\n{last_user}"

        return last_user

    def _stream_turn(
        self,
        prompt_text: str,
        emit_events: bool,
        user_text: Optional[str] = None,
        **kwargs,
    ):
        self.start()

        turn_params = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt_text}],
            "model": self.model,
            "cwd": self.workspace,
        }

        effort = kwargs.get("effort", self.effort)
        summary = kwargs.get("summary", self.summary)
        if effort:
            turn_params["effort"] = effort
        if summary:
            turn_params["summary"] = summary
        service_tier = kwargs.get("service_tier", kwargs.get("serviceTier", self.service_tier))
        if service_tier:
            turn_params["serviceTier"] = service_tier

        if self.sandbox_policy:
            turn_params["sandboxPolicy"] = self.sandbox_policy
        elif str(self.thread_sandbox or "").strip().lower() not in {
            "danger-full-access",
            "dangerfullaccess",
        }:
            turn_params["sandboxPolicy"] = {
                "type": "workspaceWrite",
                "writableRoots": [self.workspace],
                "networkAccess": self.network_access,
            }

        _send(self.proc, {"method": "turn/start", "id": self._next_id(), "params": turn_params})
        result = _wait_result(self.proc, self.q, self._req_id - 1, self.auto_approve)
        turn_id = str(result["turn"]["id"])
        self._active_turn_id = turn_id

        # Turn logger
        tlog = logger.bind(
            tag=TAG,
            session_key=self.session_key,
            thread_id=self.thread_id,
            turn_id=turn_id,
        )

        # ---- file buffering (assistant text + event lines) ----
        file_buf = ""
        raw_buf = ""

        def file_append(s: str) -> None:
            nonlocal file_buf
            if not self.log_stream or not s:
                return
            file_buf += s
            if len(file_buf.encode("utf-8", errors="ignore")) >= self._stream_flush_bytes:
                self._append_stream_log(file_buf)
                file_buf = ""

        def file_flush() -> None:
            nonlocal file_buf
            if file_buf:
                self._append_stream_log(file_buf)
                file_buf = ""

        def raw_append(s: str) -> None:
            nonlocal raw_buf
            if not self.raw_stream_log or not s:
                return
            raw_buf += s
            if len(raw_buf.encode("utf-8", errors="ignore")) >= self._raw_stream_flush_bytes:
                self._append_raw_log(raw_buf)
                raw_buf = ""

        def raw_flush() -> None:
            nonlocal raw_buf
            if raw_buf:
                self._append_raw_log(raw_buf)
                raw_buf = ""

        def file_event(event_obj: Dict) -> None:
            """
            Write an event line into the SAME log file, with timestamp.
            We log the *emitted* event dicts (thinking/action), not every raw app-server message.
            """
            if not (self.log_stream and self.log_events):
                return

            try:
                payload = json.dumps(event_obj, ensure_ascii=False)
            except Exception:
                payload = str(event_obj)

            if self.event_max_chars and len(payload) > self.event_max_chars:
                payload = payload[: self.event_max_chars] + " ..."

            # Ensure event starts on a new line and ends with newline.
            file_append(f"\n[{_ts()}] [EVENT] {payload}\n")

        def file_thinking_debug(method: str, delta: str) -> None:
            if not (self.log_stream and self.thinking_debug):
                return
            safe_delta = delta.replace("\r", "\\r").replace("\n", "\\n")
            file_append(f"\n[{_ts()}] [THINKING_DEBUG] {method}: {safe_delta}\n")

        def mcp_tool_guard(params: Dict, meta: Dict) -> bool:
            tool_name = _extract_elicitation_tool_name(params, meta)
            if tool_name != "xiaozhi_take_photo":
                return True
            if _user_text_authorizes_photo(user_text):
                return True

            if self.log_stream:
                file_append(
                    f"\n[{_ts()}] [BLOCKED_UNAUTHORIZED_PHOTO_TOOL_CALL]\n"
                )
            tlog.warning(
                "blocked_unauthorized_photo_tool_call",
                user_text=_short(user_text, 200),
            )
            return False

        # Log turn start
        if self.log_stream:
            file_append(f"[{_ts()}] [TURN_START] session={self.session_key} thread={self.thread_id} turn={turn_id}\n")
            if user_text:
                file_append(f"[{_ts()}] [USER] {user_text}\n")
            # Make the device log visible immediately so same-device recovery can
            # read at least the current turn header/user utterance after a rapid
            # reconnect, even if the Codex turn has not finished yet.
            file_flush()

        saw_tokens = False
        final_text = None
        thinking_buffer = ""
        out_buffer = ""
        guarded_text_buffer = ""
        mcp_tools_called: List[str] = []
        exp2_uvvis_prep_guard_active = _is_exp2_uvvis_prep_guard_turn(
            user_text,
            kwargs.get("experiment_context", {}) or {},
            self.experiment_yaml_path,
        )
        agent_pending_text = ""
        agent_internal_leak_suppressed = False

        def find_agent_internal_leak(text: str) -> int:
            lowered = text.lower()
            positions = [
                lowered.find(marker)
                for marker in _AGENT_INTERNAL_LEAK_MARKERS
                if lowered.find(marker) >= 0
            ]
            return min(positions) if positions else -1

        def emit_agent_text(text: str) -> List[str]:
            nonlocal out_buffer, guarded_text_buffer
            if not text:
                return []
            if exp2_uvvis_prep_guard_active:
                guarded_text_buffer += text
                return []
            out_buffer += text
            if self.log_stream:
                file_append(text)
            return [text]

        def emit_final_agent_text(text: str) -> List[str]:
            nonlocal out_buffer
            if not text:
                return []
            out_buffer += text
            if self.log_stream:
                file_append(text)
            return [text]

        def flush_agent_text(force: bool = False) -> List[str]:
            nonlocal agent_pending_text, agent_internal_leak_suppressed
            if agent_internal_leak_suppressed:
                agent_pending_text = ""
                return []
            if not agent_pending_text:
                return []

            marker_at = find_agent_internal_leak(agent_pending_text)
            if marker_at >= 0:
                visible_text = agent_pending_text[:marker_at].rstrip()
                agent_pending_text = ""
                agent_internal_leak_suppressed = True
                chunks = emit_agent_text(visible_text)
                if self.log_stream:
                    file_append(f"\n[{_ts()}] [FILTERED_INTERNAL_LEAK]\n")
                if self.log_stream and self.stream_debug:
                    tlog.warning("codex_agent_internal_leak_filtered")
                return chunks

            if not force and len(agent_pending_text) <= _AGENT_INTERNAL_LEAK_HOLD_CHARS:
                return []

            if force:
                visible_text = agent_pending_text
                agent_pending_text = ""
                return emit_agent_text(visible_text)

            emit_len = len(agent_pending_text) - _AGENT_INTERNAL_LEAK_HOLD_CHARS
            visible_text = agent_pending_text[:emit_len]
            agent_pending_text = agent_pending_text[emit_len:]
            return emit_agent_text(visible_text)

        def append_agent_text(delta: str) -> List[str]:
            nonlocal agent_pending_text
            if delta:
                agent_pending_text += delta
            return flush_agent_text(force=False)

        try:
            while True:
                msg = _read_one(self.q, timeout=None)
                if self.raw_stream_log:
                    try:
                        raw_payload = json.dumps(msg, ensure_ascii=False)
                    except Exception:
                        raw_payload = str(msg)
                    raw_append(f"[{_ts()}] raw_{raw_payload}\n")
                if _is_server_request(msg):
                    _accept_server_request(
                        self.proc,
                        msg,
                        self.auto_approve,
                        mcp_tool_guard=mcp_tool_guard,
                    )
                    continue

                if not _matches_thread_turn(msg, self.thread_id, turn_id):
                    continue

                method = msg.get("method")
                params = msg.get("params", {}) or {}
                
                # --- assistant text stream ---
                if method == "item/agentMessage/delta":
                    delta = params.get("delta", "")
                    if delta:
                        saw_tokens = True
                        visible_deltas = append_agent_text(delta)

                        # per-token debug (config-only gate)
                        if self.log_stream and self.stream_debug:
                            tlog.debug("codex_stream_delta", delta=_short(delta, 400))

                        for visible_delta in visible_deltas:
                            yield visible_delta

                # --- emitted events (thinking/action) ---
                if emit_events and self.thinking_mode != "off":
                    if self.thinking_mode == "summary" and method == "item/reasoning/summaryTextDelta":
                        delta = params.get("delta", "")
                        if delta:
                            thinking_buffer += delta
                            evt = {"kind": "thinking", "mode": "summary", "delta": delta, "turn_id": turn_id}
                            file_event(evt)
                            file_thinking_debug(method, delta)
                            if self.log_stream and self.stream_debug:
                                tlog.debug("codex_thinking_summary_delta", delta=_short(delta, 400))
                            yield {"kind": "thinking", "text": thinking_buffer}

                    if self.thinking_mode == "raw" and method == "item/reasoning/textDelta":
                        delta = params.get("delta", "")
                        if delta:
                            thinking_buffer += delta
                            evt = {"kind": "thinking", "mode": "raw", "delta": delta, "turn_id": turn_id}
                            file_event(evt)
                            file_thinking_debug(method, delta)
                            if self.log_stream and self.stream_debug:
                                tlog.debug("codex_thinking_raw_delta", delta=_short(delta, 400))
                            yield {"kind": "thinking", "text": thinking_buffer}
                else:
                    if method in ("item/reasoning/summaryTextDelta", "item/reasoning/textDelta"):
                        delta = params.get("delta", "")
                        if delta:
                            file_thinking_debug(method, delta)

                if emit_events and self.show_actions and method in ("item/started", "item/completed"):
                    item = params.get("item", {}) or {}
                    item_type = item.get("type")
                    if item_type not in ("userMessage", "reasoning", "agentMessage"):
                        phase = "start" if method == "item/started" else "done"
                        desc = _format_action_desc(item)
                        evt = {"kind": "action", "phase": phase, "text": desc, "turn_id": turn_id}
                        file_event(evt)

                        if self.log_stream:
                            tlog.info("codex_action", phase=phase, action=desc)

                        yield {"kind": "action", "text": desc, "phase": phase}

                # --- final text fallback ---
                if method == "item/completed":
                    item = params.get("item", {}) or {}
                    tool_name = str(item.get("name") or item.get("tool_name") or "").strip()
                    if tool_name and tool_name not in mcp_tools_called:
                        mcp_tools_called.append(tool_name)
                    _sync_native_mcp_function_call_state(
                        kwargs.get("state_conn"),
                        item,
                    )
                    if item.get("type") == "agentMessage":
                        final_text = item.get("text") or item.get("content") or ""

                if method == "turn/completed":
                    for visible_delta in flush_agent_text(force=True):
                        yield visible_delta
                    break

            # fallback if the server didn't stream deltas but gave final text
            if not saw_tokens and final_text:
                for visible_delta in append_agent_text(final_text):
                    yield visible_delta
                for visible_delta in flush_agent_text(force=True):
                    yield visible_delta

            if exp2_uvvis_prep_guard_active:
                guarded_reply = _finalize_exp2_uvvis_prep_guard_text(
                    user_text=user_text or "",
                    assistant_text=guarded_text_buffer,
                    called_tools=mcp_tools_called,
                )
                if guarded_reply != guarded_text_buffer and self.log_stream:
                    file_append(
                        f"\n[{_ts()}] [FILTERED_UNVERIFIED_UVVIS_PREP_REPLY] "
                        f"tools={','.join(mcp_tools_called) or '-'}\n"
                    )
                for visible_delta in emit_final_agent_text(guarded_reply):
                    yield visible_delta

        finally:
            self._active_turn_id = None

            # Log turn end
            if self.log_stream:
                file_append(f"\n[{_ts()}] [TURN_END] chars={len(out_buffer)}\n\n--- turn_end ---\n\n")

            file_flush()
            raw_flush()

            if self.log_stream:
                tlog.info(
                    "codex_stream_turn_completed",
                    chars=len(out_buffer),
                    preview=_short(out_buffer, 2000),
                    stream_log_path=self.stream_log_path,
                )

    def stream_response(self, dialogue: List[Dict], **kwargs):
        with self._lock:
            routing_context = _routing_context_from_kwargs(kwargs)
            experiment_context = _experiment_context_from_kwargs(kwargs)
            self.stream_log_path = self._resolve_log_path(
                self.stream_log_path_template, routing_context
            )
            self.raw_stream_log_path = self._resolve_log_path(
                self.raw_stream_log_path_template, routing_context
            )
            if routing_context:
                logger.bind(tag=TAG).info(
                    "codex_turn_routing_context "
                    f"session={self.session_key} "
                    f"context={json.dumps(routing_context, ensure_ascii=False)}"
                )
            if experiment_context:
                logger.bind(tag=TAG).info(
                    "codex_turn_experiment_context "
                    f"session={self.session_key} "
                    f"context={json.dumps(experiment_context, ensure_ascii=False)}"
                )

            # Starting the app-server/thread resets first-turn flags.
            # Do it before composing the prompt so the first real turn can
            # correctly mark system-prompt/bootstrap state.
            self.start()
            prompt_text = self._compose_prompt(
                dialogue,
                routing_context=routing_context,
                experiment_context=experiment_context,
            )
            if not prompt_text:
                return
            _, last_user, _ = _split_dialogue(dialogue)
            emit_events = kwargs.pop("emit_events", self.emit_events)
            for token in self._stream_turn(
                prompt_text,
                emit_events=emit_events,
                user_text=last_user,
                experiment_context=experiment_context,
                **kwargs,
            ):
                if isinstance(token, dict) and not emit_events:
                    continue
                yield token


class LLMProvider(LLMProviderBase):
    def __init__(self, config: Dict):
        self.config = config
        self._sessions: Dict[str, _CodexSession] = {}
        self._reuse_utility_session = bool(config.get("reuse_utility_session", True))
        self._utility_session_key = "__utility__"

    def _get_session(self, session_id: str) -> _CodexSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = _CodexSession(self.config, session_id)
        return self._sessions[session_id]

    def response(self, session_id, dialogue, **kwargs):
        if not session_id:
            if self._reuse_utility_session:
                session = self._get_session(self._utility_session_key)
                use_temp_session = False
            else:
                session = _CodexSession(self.config, "utility")
                use_temp_session = True
        else:
            session = self._get_session(session_id)
            use_temp_session = False

        try:
            for attempt in range(2):
                emitted_any = False
                try:
                    for token in session.stream_response(dialogue, **kwargs):
                        emitted_any = True
                        yield token
                    return
                except Exception as exc:
                    should_retry = (
                        attempt == 0
                        and not emitted_any
                        and _is_recoverable_turn_failure(exc, session)
                    )
                    session.close()
                    if should_retry:
                        logger.bind(tag=TAG).warning(
                            "Codex turn hit recoverable child-process failure; "
                            f"retrying once: {exc}"
                        )
                        continue
                    logger.bind(tag=TAG).error(f"Codex response error: {exc}")
                    yield "[Codex response error]"
                    return
        finally:
            if use_temp_session:
                session.close()

    def cleanup(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions = {}
        for session in sessions:
            try:
                session.close()
            except Exception as exc:
                logger.bind(tag=TAG).warning(
                    f"failed to close codex session during provider cleanup: {exc}"
                )

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        patched_dialogue = deepcopy(dialogue)
        inject_legacy_function_prompt = bool(
            self.config.get("inject_legacy_function_prompt", True)
        )

        if inject_legacy_function_prompt and len(patched_dialogue) == 2 and functions:
            last_msg = str(patched_dialogue[-1].get("content", ""))
            function_str = json.dumps(functions, ensure_ascii=False)
            patched_dialogue[-1]["content"] = (
                get_system_prompt_for_function(function_str) + last_msg
            )

        if len(patched_dialogue) > 1 and patched_dialogue[-1].get("role") == "tool":
            assistant_msg = (
                "\ntool call result: "
                + str(patched_dialogue[-1].get("content", ""))
                + "\n\n"
            )
            while len(patched_dialogue) > 1:
                if patched_dialogue[-1].get("role") == "user":
                    patched_dialogue[-1]["content"] = (
                        assistant_msg + str(patched_dialogue[-1].get("content", ""))
                    )
                    break
                patched_dialogue.pop()

        if functions:
            kwargs = dict(kwargs)
            kwargs["functions"] = functions

        for token in self.response(session_id, patched_dialogue, **kwargs):
            if isinstance(token, dict):
                yield token
            else:
                yield token, None
