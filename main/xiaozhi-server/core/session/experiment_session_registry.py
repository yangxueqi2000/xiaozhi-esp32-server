import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from config.config_loader import get_project_dir

_local_store_lock = asyncio.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _registry_config(config: Dict) -> Dict:
    return config.get("experiment_session_registry", {}) or {}


def _normalize_yaml_path(yaml_path: str) -> str:
    text = str(yaml_path or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.abspath(os.path.normpath(text)))


def _binding_key(chat_session_id: str, yaml_path: str) -> str:
    normalized_chat_session_id = str(chat_session_id or "").strip()
    normalized_yaml_path = _normalize_yaml_path(yaml_path)
    return f"{normalized_chat_session_id}::{normalized_yaml_path}"


def _resolve_local_store_path(config: Dict) -> str:
    registry_cfg = _registry_config(config)
    local_store = str(
        registry_cfg.get(
            "local_store",
            config.get("codex_app", {}).get(
                "experiment_session_store",
                "data/codex_app/experiment_session_registry.json",
            ),
        )
    ).strip()
    if not local_store:
        local_store = "data/codex_app/experiment_session_registry.json"
    if os.path.isabs(local_store):
        return local_store
    return os.path.join(get_project_dir(), local_store)


def _load_local_store(store_path: str) -> Dict:
    if not os.path.exists(store_path):
        return {"bindings": {}}

    try:
        with open(store_path, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
    except Exception:
        return {"bindings": {}}

    if not isinstance(raw, dict):
        return {"bindings": {}}

    bindings = raw.get("bindings")
    if not isinstance(bindings, dict):
        raw["bindings"] = {}
    return raw


def _save_local_store(store_path: str, store: Dict):
    parent = os.path.dirname(store_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _normalize_entry(entry: Dict) -> Dict:
    return {
        "chat_session_id": str(entry.get("chat_session_id", "")).strip(),
        "model_session_key": str(entry.get("model_session_key", "")).strip(),
        "device_id": str(entry.get("device_id", "")).strip(),
        "user_id": str(entry.get("user_id", "")).strip(),
        "yaml_path": str(entry.get("yaml_path", "")).strip(),
        "normalized_yaml_path": _normalize_yaml_path(entry.get("yaml_path", "")),
        "experiment_session_id": str(entry.get("experiment_session_id", "")).strip(),
        "status": str(entry.get("status", "")).strip(),
        "source": str(entry.get("source", "")).strip(),
        "current_step_id": str(entry.get("current_step_id", "")).strip(),
        "completed_steps_count": entry.get("completed_steps_count"),
        "total_steps": entry.get("total_steps"),
        "created_at": str(entry.get("created_at", "")).strip(),
        "updated_at": str(entry.get("updated_at", "")).strip(),
    }


async def load_experiment_session_binding(
    config: Dict,
    chat_session_id: str,
    yaml_path: str,
) -> Optional[Dict]:
    normalized_chat_session_id = str(chat_session_id or "").strip()
    normalized_yaml_path = _normalize_yaml_path(yaml_path)
    if not normalized_chat_session_id or not normalized_yaml_path:
        return None

    store_path = _resolve_local_store_path(config)
    key = _binding_key(normalized_chat_session_id, normalized_yaml_path)

    async with _local_store_lock:
        store = _load_local_store(store_path)
        bindings = store.get("bindings", {})
        entry = bindings.get(key)

    if not isinstance(entry, dict):
        return None
    return _normalize_entry(entry)


async def save_experiment_session_binding(
    config: Dict,
    *,
    chat_session_id: str,
    yaml_path: str,
    experiment_session_id: str,
    model_session_key: str = "",
    device_id: str = "",
    user_id: str = "",
    status: str = "",
    source: str = "",
    current_step_id: str = "",
    completed_steps_count=None,
    total_steps=None,
) -> Dict:
    normalized_chat_session_id = str(chat_session_id or "").strip()
    normalized_yaml_path = _normalize_yaml_path(yaml_path)
    normalized_experiment_session_id = str(experiment_session_id or "").strip()
    if (
        not normalized_chat_session_id
        or not normalized_yaml_path
        or not normalized_experiment_session_id
    ):
        raise ValueError(
            "chat_session_id, yaml_path, and experiment_session_id are required"
        )

    store_path = _resolve_local_store_path(config)
    key = _binding_key(normalized_chat_session_id, normalized_yaml_path)
    now = _utc_now()

    async with _local_store_lock:
        store = _load_local_store(store_path)
        bindings = store.setdefault("bindings", {})
        entry = bindings.get(key)
        if not isinstance(entry, dict):
            entry = {
                "chat_session_id": normalized_chat_session_id,
                "yaml_path": normalized_yaml_path,
                "created_at": now,
            }

        entry["chat_session_id"] = normalized_chat_session_id
        entry["model_session_key"] = str(model_session_key or "").strip()
        entry["device_id"] = str(device_id or "").strip()
        entry["user_id"] = str(user_id or "").strip()
        entry["yaml_path"] = normalized_yaml_path
        entry["experiment_session_id"] = normalized_experiment_session_id
        entry["status"] = str(status or "").strip()
        entry["source"] = str(source or "").strip()
        entry["current_step_id"] = str(current_step_id or "").strip()
        entry["completed_steps_count"] = completed_steps_count
        entry["total_steps"] = total_steps
        entry["updated_at"] = now
        bindings[key] = entry
        _save_local_store(store_path, store)

    return _normalize_entry(entry)


async def delete_experiment_session_binding(
    config: Dict,
    chat_session_id: str,
    yaml_path: str,
) -> bool:
    normalized_chat_session_id = str(chat_session_id or "").strip()
    normalized_yaml_path = _normalize_yaml_path(yaml_path)
    if not normalized_chat_session_id or not normalized_yaml_path:
        return False

    store_path = _resolve_local_store_path(config)
    key = _binding_key(normalized_chat_session_id, normalized_yaml_path)

    async with _local_store_lock:
        store = _load_local_store(store_path)
        bindings = store.setdefault("bindings", {})
        if key not in bindings:
            return False
        del bindings[key]
        _save_local_store(store_path, store)
    return True
