from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parents[1]
EXP_GRAPH_SERVER = Path(r"C:\Users\11979\Documents\GitHub\ExperimentalAssistantServer\experiment_graph_mcp_server.py")
EXP_GRAPH_ROOT = EXP_GRAPH_SERVER.parent
LC_PYTHON = Path(r"C:\Users\11979\anaconda3\envs\lc\python.exe")


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


def _build_system_prompt(config: Dict[str, Any], device_id: str) -> str:
    _add_server_to_path()
    from core.utils.prompt_manager import PromptManager

    prompt_manager = PromptManager(config)
    return prompt_manager.build_enhanced_prompt(
        str(config.get("prompt") or ""),
        device_id=device_id,
        client_ip="127.0.0.1",
    )


def _write_fake_mcp_settings(path: Path, *, device_id: str, data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "mcpServers": {
            "experiment-graph": {
                "command": str(LC_PYTHON),
                "args": [str(EXP_GRAPH_SERVER)],
                "env": {
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            },
            "xiaozhi-device-trigger": {
                "command": str(LC_PYTHON),
                "args": [str(SERVER_ROOT / "scripts" / "fake_device_trigger_mcp_server.py")],
                "env": {
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "FAKE_XIAOZHI_DEVICE_ID": device_id,
                    "FAKE_XIAOZHI_DATA_ROOT": str(data_root),
                },
            },
        }
    }
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(obj), ensure_ascii=False) + "\n")


def _short_json(value: Any, limit: int = 1400) -> str:
    text = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _create_graph_prewarm_context(
    *,
    config: Dict[str, Any],
    model_session_key: str,
    log_dir: Path,
) -> Dict[str, Any]:
    yaml_path = Path(config["experiment_paths"]["experiment_yaml_path"]).resolve()
    experiment_session_id = "smoke_" + uuid.uuid4().hex
    code = r"""
import json
import sys

root, yaml_path, session_id = sys.argv[1:4]
sys.path.insert(0, root)
import experiment_graph_mcp_server as server

created = server.create_session(yaml_path=yaml_path, session_id=session_id, overwrite=True)
step = server.get_step(session_id) if created.get("ok") else {}
state = server.get_state(session_id) if created.get("ok") else {}
print(json.dumps({"created": created, "step": step, "state": state}, ensure_ascii=False))
"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [str(LC_PYTHON), "-c", code, str(EXP_GRAPH_ROOT), str(yaml_path), experiment_session_id],
        capture_output=True,
        env=env,
        timeout=45,
    )
    stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
    (log_dir / "graph_prewarm_stdout.log").write_text(stdout_text, encoding="utf-8")
    (log_dir / "graph_prewarm_stderr.log").write_text(stderr_text, encoding="utf-8")
    if proc.returncode != 0:
        return {
            "experiment_prewarm_wait_result": "failed",
            "experiment_prewarm_status": "failed",
            "experiment_prewarm_error": (stderr_text or stdout_text or "").strip(),
            "experiment_yaml_path": str(yaml_path),
        }

    payload: Dict[str, Any] = {}
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    created = payload.get("created", {}) if isinstance(payload, dict) else {}
    step = payload.get("step", {}) if isinstance(payload, dict) else {}
    state = payload.get("state", {}) if isinstance(payload, dict) else {}
    if not created.get("ok"):
        return {
            "experiment_prewarm_wait_result": "failed",
            "experiment_prewarm_status": "failed",
            "experiment_prewarm_error": str(created.get("message") or "graph prewarm failed"),
            "experiment_yaml_path": str(yaml_path),
        }

    current_step_id = str(created.get("current_step_id") or state.get("current_step_id") or "")
    step_body = step.get("step", {}) if isinstance(step, dict) and isinstance(step.get("step"), dict) else step
    current_step_summary = {
        "step_id": current_step_id,
        "title": step_body.get("title"),
        "description": step_body.get("description"),
        "summary": step_body.get("summary"),
        "current_progress": step_body.get("current_progress"),
    }
    return {
        "experiment_prewarm_wait_result": "ready",
        "experiment_prewarm_status": "completed",
        "experiment_prewarm_ready_level": "completed",
        "experiment_prewarm_trigger": "text-smoke-prewarm",
        "experiment_session_id": str(created.get("session_id") or experiment_session_id),
        "experiment_current_step_id": current_step_id,
        "experiment_yaml_path": str(yaml_path),
        "experiment_current_step_summary": _short_json(current_step_summary),
        "_raw_graph_prewarm": payload,
    }


def _find_recent_outputs(device_dir: Path, start_time: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "device_dir": str(device_dir),
        "exists": device_dir.exists(),
        "photos": [],
        "reports": [],
    }
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
        if suffix in {".yaml", ".yml", ".pdf"}:
            result["reports"].append(str(path))
    return result


def _find_recent_graph_sessions(start_time: float) -> List[str]:
    root = Path(r"C:\Users\11979\Documents\GitHub\ExperimentalAssistantServer\data\session_state")
    if not root.exists():
        return []
    paths = []
    for path in root.glob("*.json"):
        try:
            if path.stat().st_mtime >= start_time - 2:
                paths.append(path)
        except OSError:
            continue
    return [str(path) for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:5]]


def _run_turns(
    *,
    provider,
    system_prompt: str,
    session_id: str,
    route_kwargs: Dict[str, Any],
    experiment_kwargs: Dict[str, Any],
    turns: Iterable[str],
    transcript_path: Path,
    events_path: Path,
    stop_on_error: bool,
) -> Dict[str, Any]:
    first_turn = True
    completed_turns = 0
    error_seen = False
    last_reply = ""

    with transcript_path.open("w", encoding="utf-8") as transcript:
        transcript.write("# Exp1 Text Tutorial Smoke Transcript\n\n")
        for idx, user_text in enumerate(turns, start=1):
            print(f"\n===== TURN {idx} USER =====\n{user_text}\n", flush=True)
            transcript.write(f"## Turn {idx}\n\n**User:** {user_text}\n\n")
            messages = []
            if first_turn:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})

            chunks: List[str] = []
            started_at = time.time()
            _append_jsonl(
                events_path,
                {
                    "type": "turn_start",
                    "turn": idx,
                    "user": user_text,
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )

            for token in provider.response(
                session_id,
                messages,
                emit_events=True,
                **route_kwargs,
                **experiment_kwargs,
            ):
                if isinstance(token, dict):
                    _append_jsonl(events_path, {"type": "event", "turn": idx, "data": token})
                    continue
                text = str(token)
                chunks.append(text)
                print(text, end="", flush=True)

            print("", flush=True)
            assistant_text = "".join(chunks).strip()
            last_reply = assistant_text
            transcript.write(f"**Assistant:** {assistant_text}\n\n")
            _append_jsonl(
                events_path,
                {
                    "type": "turn_done",
                    "turn": idx,
                    "assistant": assistant_text,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                },
            )
            completed_turns = idx
            first_turn = False

            if "[Codex response error]" in assistant_text:
                error_seen = True
                if stop_on_error:
                    break

    return {
        "completed_turns": completed_turns,
        "error_seen": error_seen,
        "last_reply": last_reply,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exp1 AI-assistant text tutorial smoke test.")
    parser.add_argument("--device-id", default="codex-exp1-tutorial-smoke")
    parser.add_argument("--user-id", default="tutorial-test")
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    _add_server_to_path()
    from core.providers.llm.codex.codex import LLMProvider

    start_time = time.time()
    config = _load_exp1_config()
    device_id = str(args.device_id).strip()
    experiment_data_root = Path(config["experiment_paths"]["experiment_data_root"]).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = experiment_data_root / device_id / f"tutorial_smoke_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    mcp_settings_path = log_dir / "fake_mcp_server_settings.json"
    _write_fake_mcp_settings(
        mcp_settings_path,
        device_id=device_id,
        data_root=experiment_data_root,
    )

    llm_name = str(config.get("codex_app", {}).get("llm_name") or config["selected_module"]["LLM"])
    llm_cfg = deepcopy(config["LLM"][llm_name])
    llm_cfg["mcp_settings_path"] = str(mcp_settings_path)
    llm_cfg["stream_log_path"] = str(log_dir / "codex_stream.log")
    llm_cfg["user_utterance_log_path"] = str(log_dir / "user_utterances.jsonl")
    llm_cfg["log_stream"] = True
    llm_cfg["emit_events"] = True
    llm_cfg["show_actions"] = True
    llm_cfg["raw_stream_log"] = True
    llm_cfg["raw_stream_log_path"] = str(log_dir / "codex_stream_raw.log")
    llm_cfg.setdefault("approval_policy", "never")

    system_prompt = _build_system_prompt(config, device_id)
    (log_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    model_session_key = f"codex:{uuid.uuid4()}"
    experiment_kwargs = _create_graph_prewarm_context(
        config=config,
        model_session_key=model_session_key,
        log_dir=log_dir,
    )
    graph_prewarm_raw = experiment_kwargs.pop("_raw_graph_prewarm", {})
    route_kwargs = {
        "device_id": device_id,
        "user_id": str(args.user_id).strip(),
        "chat_session_id": model_session_key.replace("codex:", ""),
        "model_session_key": model_session_key,
        "connection_session_id": "text-smoke-connection",
        "transport_session_id": "text-smoke-transport",
    }

    turns = _turns()
    if args.max_turns and args.max_turns > 0:
        turns = turns[: args.max_turns]

    summary: Dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device_id": device_id,
        "model_session_key": model_session_key,
        "log_dir": str(log_dir),
        "mcp_settings_path": str(mcp_settings_path),
        "turn_count_requested": len(turns),
        "graph_prewarm": graph_prewarm_raw,
    }

    provider = LLMProvider(llm_cfg)
    try:
        run_result = _run_turns(
            provider=provider,
            system_prompt=system_prompt,
            session_id=model_session_key,
            route_kwargs=route_kwargs,
            experiment_kwargs=experiment_kwargs,
            turns=turns,
            transcript_path=log_dir / "transcript.md",
            events_path=log_dir / "events.jsonl",
            stop_on_error=bool(args.stop_on_error),
        )
        summary.update(run_result)
    finally:
        provider.cleanup()

    summary["outputs"] = _find_recent_outputs(experiment_data_root / device_id, start_time)
    summary["recent_graph_session_state"] = _find_recent_graph_sessions(start_time)
    summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (log_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary.get("error_seen") else 0


if __name__ == "__main__":
    raise SystemExit(main())
