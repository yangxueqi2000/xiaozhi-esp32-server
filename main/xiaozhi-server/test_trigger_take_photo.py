import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


from trigger_take_photo import (
    DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT,
    DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT,
    trigger_take_photo,
)


class TriggerTakePhotoTest(unittest.TestCase):
    def test_trigger_take_photo_uses_shorter_defaults(self):
        captured = {}

        def fake_request_json(method, url, payload=None, timeout=0):
            captured["method"] = method
            captured["url"] = url
            captured["payload"] = dict(payload or {})
            captured["timeout"] = timeout
            return {"success": True}

        with patch("trigger_take_photo._request_json", fake_request_json):
            result = trigger_take_photo(
                "http://127.0.0.1:8003",
                session_id="session-1",
            )

        self.assertTrue(result["success"])
        self.assertEqual("POST", captured["method"])
        self.assertEqual(
            "http://127.0.0.1:8003/mcp/device/take_photo",
            captured["url"],
        )
        self.assertEqual(
            DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT,
            captured["payload"]["timeout"],
        )
        self.assertEqual(
            DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT,
            captured["timeout"],
        )

    def test_trigger_take_photo_bumps_request_timeout_above_tool_timeout(self):
        captured = {}

        def fake_request_json(method, url, payload=None, timeout=0):
            captured["payload"] = dict(payload or {})
            captured["timeout"] = timeout
            return {"success": True}

        with patch("trigger_take_photo._request_json", fake_request_json):
            trigger_take_photo(
                "http://127.0.0.1:8003",
                session_id="session-1",
                tool_timeout=42,
                request_timeout=35,
            )

        self.assertEqual(42, captured["payload"]["timeout"])
        self.assertEqual(47, captured["timeout"])


if __name__ == "__main__":
    unittest.main()
