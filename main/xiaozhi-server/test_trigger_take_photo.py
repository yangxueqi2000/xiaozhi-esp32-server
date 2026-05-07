import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


from trigger_take_photo import (
    DEFAULT_TAKE_PHOTO_REQUEST_TIMEOUT,
    DEFAULT_TAKE_PHOTO_TOOL_TIMEOUT,
    build_question_with_photo_name,
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

    def test_trigger_take_photo_carries_group_metadata(self):
        captured = {}

        def fake_request_json(method, url, payload=None, timeout=0):
            captured["payload"] = dict(payload or {})
            return {"success": True}

        with patch("trigger_take_photo._request_json", fake_request_json):
            trigger_take_photo(
                "http://127.0.0.1:8003",
                session_id="session-1",
                photo_name="sample_2",
                group_number=3,
            )

        self.assertEqual(3, captured["payload"]["group_number"])
        self.assertEqual("sample_2", captured["payload"]["photo_name"])
        self.assertEqual("Please take a photo.", captured["payload"]["question"])

    def test_build_question_without_group_preserves_existing_payload(self):
        self.assertEqual(
            'look\n[XIAOZHI_META]{"photo_name":"sample_1"}',
            build_question_with_photo_name("look", "sample_1"),
        )

    def test_build_question_with_group_metadata(self):
        question = build_question_with_photo_name("look", "sample_2", group_number=3)
        self.assertIn('"photo_name":"sample_2"', question)
        self.assertIn('"group_number":3', question)
        self.assertIn('"group_dir_name":"group_03"', question)


if __name__ == "__main__":
    unittest.main()
