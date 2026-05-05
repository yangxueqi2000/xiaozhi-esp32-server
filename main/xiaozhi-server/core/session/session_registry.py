import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Tuple

from config.config_loader import get_project_dir

_local_store_lock = asyncio.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _registry_config(config: Dict) -> Dict:
    return config.get("session_registry", {}) or {}


def _default_user_id(config: Dict) -> str:
    user_id = str(_registry_config(config).get("default_user_id", "test")).strip()
    return user_id or "test"


def _normalize_ids(device_id: str, user_id: str, config: Dict) -> Tuple[str, str]:
    normalized_device = str(device_id or "").strip()
    normalized_user = str(user_id or "").strip() or _default_user_id(config)
    return normalized_device, normalized_user


def _binding_key(device_id: str) -> str:
    return str(device_id or "").strip()


def _entry_sort_key(entry: Dict) -> str:
    if not isinstance(entry, dict):
        return ""
    updated_at = str(entry.get("updated_at", "")).strip()
    if updated_at:
        return updated_at
    return str(entry.get("created_at", "")).strip()


def _migrate_legacy_binding(
    bindings: Dict,
    device_id: str,
    fallback_user_id: str,
):
    if not isinstance(bindings, dict):
        return None

    normalized_device_id = _binding_key(device_id)
    if not normalized_device_id:
        return None

    legacy_prefix = f"{normalized_device_id}::"
    legacy_candidates = []
    for key, entry in list(bindings.items()):
        if key.startswith(legacy_prefix) and isinstance(entry, dict):
            legacy_candidates.append((key, entry))

    if not legacy_candidates:
        return None

    selected_key, selected_entry = max(
        legacy_candidates,
        key=lambda item: _entry_sort_key(item[1]),
    )
    for key, _entry in legacy_candidates:
        bindings.pop(key, None)

    migrated_entry = dict(selected_entry)
    migrated_entry["device_id"] = normalized_device_id
    migrated_entry["user_id"] = (
        str(migrated_entry.get("user_id", "")).strip() or fallback_user_id
    )
    bindings[normalized_device_id] = migrated_entry
    return migrated_entry


def _get_binding_entry(bindings: Dict, device_id: str, fallback_user_id: str):
    binding_key = _binding_key(device_id)
    entry = bindings.get(binding_key)
    if isinstance(entry, dict):
        return binding_key, entry

    migrated_entry = _migrate_legacy_binding(bindings, binding_key, fallback_user_id)
    return binding_key, migrated_entry


def _read_response_ids(resp_json: Dict) -> Tuple[str, str, bool]:
    body = resp_json.get("data") if isinstance(resp_json.get("data"), dict) else resp_json
    chat_session_id = (
        body.get("chat_session_id")
        or body.get("chatSessionId")
        or ""
    )
    model_session_key = (
        body.get("model_session_key")
        or body.get("modelSessionKey")
        or ""
    )
    created = bool(body.get("created", False))
    return str(chat_session_id).strip(), str(model_session_key).strip(), created


async def _resolve_from_remote(config: Dict, device_id: str, user_id: str):
    registry_cfg = _registry_config(config)
    endpoint = str(registry_cfg.get("endpoint", "")).strip()
    if not endpoint:
        return None
    try:
        import aiohttp  # lazy import: remote registry may be disabled
    except ImportError as exc:
        raise RuntimeError("aiohttp is required for remote session registry") from exc

    timeout_seconds = float(registry_cfg.get("request_timeout", 3))
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    payload = {"device_id": device_id, "user_id": user_id}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, json=payload) as response:
            text = await response.text()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"session registry http {response.status}: {text[:300]}"
                )

            try:
                body = json.loads(text) if text else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError("session registry returned invalid JSON") from exc

            chat_session_id, model_session_key, created = _read_response_ids(body)
            if not chat_session_id or not model_session_key:
                raise RuntimeError(
                    "session registry response missing chat_session_id/model_session_key"
                )

            return {
                "chat_session_id": chat_session_id,
                "model_session_key": model_session_key,
                "user_id": user_id,
                "source": "remote",
                "created": created,
            }


def _resolve_local_store_path(config: Dict) -> str:
    registry_cfg = _registry_config(config)
    local_store = str(
        registry_cfg.get("local_store", "data/session_registry.json")
    ).strip()
    if not local_store:
        local_store = "data/session_registry.json"
    if os.path.isabs(local_store):
        return local_store
    return os.path.join(get_project_dir(), local_store)


def _load_local_store(store_path: str) -> Dict:
    if not os.path.exists(store_path):
        return {"bindings": {}}
    with open(store_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {"bindings": {}}
    bindings = raw.get("bindings")
    if not isinstance(bindings, dict):
        raw["bindings"] = {}
    return raw


def _save_local_store(store_path: str, store: Dict):
    dir_path = os.path.dirname(store_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


async def _resolve_from_local(config: Dict, device_id: str, user_id: str):
    store_path = _resolve_local_store_path(config)
    now = _utc_now()

    async with _local_store_lock:
        store = _load_local_store(store_path)
        bindings = store.setdefault("bindings", {})
        key, entry = _get_binding_entry(bindings, device_id, user_id)

        created = False
        if not isinstance(entry, dict):
            created = True
            chat_session_id = str(uuid.uuid4())
            model_session_key = f"codex:{chat_session_id}"
            entry = {
                "chat_session_id": chat_session_id,
                "model_session_key": model_session_key,
                "device_id": device_id,
                "user_id": user_id,
                "created_at": now,
                "updated_at": now,
            }
        else:
            entry["device_id"] = device_id
            entry["user_id"] = user_id
            entry["updated_at"] = now

        bindings[key] = entry
        _save_local_store(store_path, store)

    return {
        "chat_session_id": str(entry["chat_session_id"]),
        "model_session_key": str(entry["model_session_key"]),
        "user_id": user_id,
        "source": "local",
        "created": created,
    }


async def save_session_binding(
    config: Dict,
    device_id: str,
    user_id: str,
    chat_session_id: str,
    model_session_key: str,
):
    normalized_device_id, normalized_user_id = _normalize_ids(device_id, user_id, config)
    normalized_chat_session_id = str(chat_session_id or "").strip()
    normalized_model_session_key = str(model_session_key or "").strip()
    if not normalized_device_id:
        raise ValueError("device_id is required for session binding")
    if not normalized_chat_session_id or not normalized_model_session_key:
        raise ValueError("chat_session_id and model_session_key are required")

    store_path = _resolve_local_store_path(config)
    now = _utc_now()

    async with _local_store_lock:
        store = _load_local_store(store_path)
        bindings = store.setdefault("bindings", {})
        key, entry = _get_binding_entry(
            bindings,
            normalized_device_id,
            normalized_user_id,
        )
        created = not isinstance(entry, dict)
        created_at = now
        if isinstance(entry, dict):
            created_at = str(entry.get("created_at", "")).strip() or now
        entry = {
            "chat_session_id": normalized_chat_session_id,
            "model_session_key": normalized_model_session_key,
            "device_id": normalized_device_id,
            "user_id": normalized_user_id,
            "created_at": created_at,
            "updated_at": now,
        }
        bindings[key] = entry
        _save_local_store(store_path, store)

    return {
        "chat_session_id": normalized_chat_session_id,
        "model_session_key": normalized_model_session_key,
        "user_id": normalized_user_id,
        "source": "local",
        "created": created,
    }


async def rotate_session_binding(config: Dict, device_id: str, user_id: str):
    normalized_device_id, normalized_user_id = _normalize_ids(device_id, user_id, config)
    if not normalized_device_id:
        raise ValueError("device_id is required for session binding")

    chat_session_id = str(uuid.uuid4())
    model_session_key = f"codex:{chat_session_id}"
    return await save_session_binding(
        config,
        normalized_device_id,
        normalized_user_id,
        chat_session_id,
        model_session_key,
    )


async def resolve_or_create_session_binding(config: Dict, device_id: str, user_id: str):
    normalized_device_id, normalized_user_id = _normalize_ids(device_id, user_id, config)
    if not normalized_device_id:
        raise ValueError("device_id is required for session binding")

    try:
        remote_result = await _resolve_from_remote(
            config, normalized_device_id, normalized_user_id
        )
        if remote_result is not None:
            return remote_result
    except Exception:
        local_fallback = bool(_registry_config(config).get("enable_local_fallback", True))
        if not local_fallback:
            raise

    return await _resolve_from_local(config, normalized_device_id, normalized_user_id)
