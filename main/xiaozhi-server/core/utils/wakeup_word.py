import hashlib
import os
import re
import time
from typing import Dict

import portalocker
import yaml


class FileLock:
    def __init__(self, file, timeout=5):
        self.file = file
        self.timeout = timeout
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        while True:
            try:
                portalocker.lock(self.file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                return self.file
            except portalocker.LockException:
                if time.time() - self.start_time > self.timeout:
                    raise TimeoutError("获取文件锁超时")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        portalocker.unlock(self.file)


class WakeupWordsConfig:
    def __init__(self):
        self.config_file = "data/.wakeup_words.yaml"
        self.assets_dir = "config/assets/wakeup_words"
        self._ensure_directories()
        self._config_cache = None
        self._last_load_time = 0
        self._cache_ttl = 1
        self._lock_timeout = 5

    def _ensure_directories(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)

    @staticmethod
    def _hash_voice(voice: str) -> str:
        normalized_voice = str(voice or "default").strip() or "default"
        return hashlib.md5(normalized_voice.encode("utf-8")).hexdigest()

    def _load_config(self) -> Dict:
        current_time = time.time()
        if (
            self._config_cache is not None
            and current_time - self._last_load_time < self._cache_ttl
        ):
            return self._config_cache

        try:
            with open(self.config_file, "a+", encoding="utf-8") as file_obj:
                with FileLock(file_obj, timeout=self._lock_timeout):
                    file_obj.seek(0)
                    content = file_obj.read()
                    config = yaml.safe_load(content) if content else {}
                    if not isinstance(config, dict):
                        config = {}
                    self._config_cache = config
                    self._last_load_time = current_time
                    return config
        except (TimeoutError, IOError) as exc:
            print(f"加载唤醒词配置失败: {exc}")
            return {}
        except Exception as exc:
            print(f"加载唤醒词配置时发生未知错误: {exc}")
            return {}

    def _save_config(self, config: Dict):
        try:
            with open(self.config_file, "w", encoding="utf-8") as file_obj:
                with FileLock(file_obj, timeout=self._lock_timeout):
                    yaml.dump(config, file_obj, allow_unicode=True)
                    self._config_cache = config
                    self._last_load_time = time.time()
        except (TimeoutError, IOError) as exc:
            print(f"保存唤醒词配置失败: {exc}")
            raise
        except Exception as exc:
            print(f"保存唤醒词配置时发生未知错误: {exc}")
            raise

    def get_wakeup_response(self, voice: str) -> Dict:
        config = self._load_config()
        voice_hash = self._hash_voice(voice)
        if not config or voice_hash not in config:
            return None

        response = config[voice_hash]
        file_path = str(response.get("file_path") or "").strip()
        if not file_path or not os.path.exists(file_path):
            return None

        try:
            if os.stat(file_path).st_size < (15 * 1024):
                return None
        except OSError:
            return None

        return response

    def update_wakeup_response(self, voice: str, file_path: str, text: str):
        try:
            filtered_text = re.sub(
                r"[\U0001F600-\U0001F64F\U0001F900-\U0001F9FF]",
                "",
                str(text or ""),
            )
            config = self._load_config()
            voice_hash = self._hash_voice(voice)
            config[voice_hash] = {
                "voice": str(voice or "default").strip() or "default",
                "file_path": file_path,
                "time": time.time(),
                "text": filtered_text,
            }
            self._save_config(config)
        except Exception as exc:
            print(f"更新唤醒词配置失败: {exc}")
            raise

    def generate_file_path(self, voice: str) -> str:
        try:
            voice_hash = self._hash_voice(voice)
            return os.path.join(self.assets_dir, f"{voice_hash}.wav")
        except Exception as exc:
            print(f"生成唤醒词音频路径失败: {exc}")
            raise
