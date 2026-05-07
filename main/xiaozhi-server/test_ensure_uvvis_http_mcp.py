import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

import ensure_uvvis_http_mcp as ensure_uvvis


class _FakeResponse:
    def __init__(self, headers=None, body=b""):
        self.headers = headers or {}
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class EnsureUvvisHttpMcpTest(unittest.TestCase):
    def test_http_endpoint_ready_uses_initialize_probe_and_deletes_probe_session(self):
        captured_requests = []

        def fake_urlopen(request, timeout=0):
            captured_requests.append((request, timeout))
            if request.get_method() == "POST":
                return _FakeResponse(
                    headers={"mcp-session-id": "probe-session"},
                    body=b"event: message\ndata: {}\n\n",
                )
            if request.get_method() == "DELETE":
                return _FakeResponse()
            raise AssertionError(f"unexpected method: {request.get_method()}")

        with mock.patch.object(ensure_uvvis.urllib.request, "urlopen", side_effect=fake_urlopen):
            ready = ensure_uvvis._http_endpoint_ready("http://127.0.0.1:8766/mcp", timeout=2.5)

        self.assertTrue(ready)
        self.assertEqual(2, len(captured_requests))

        post_request, post_timeout = captured_requests[0]
        self.assertEqual("POST", post_request.get_method())
        self.assertEqual(2.5, post_timeout)
        self.assertEqual(
            {
                "jsonrpc": "2.0",
                "id": "uvvis-ready-probe",
                "method": "initialize",
                "params": {
                    "protocolVersion": ensure_uvvis.DEFAULT_MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "ensure_uvvis_http_mcp",
                        "version": "1.0",
                    },
                },
            },
            json.loads(post_request.data.decode("utf-8")),
        )
        post_headers = {key.lower(): value for key, value in post_request.header_items()}
        self.assertEqual("application/json", post_headers["Content-type".lower()])
        self.assertEqual(
            "application/json, text/event-stream",
            post_headers["Accept".lower()],
        )

        delete_request, delete_timeout = captured_requests[1]
        self.assertEqual("DELETE", delete_request.get_method())
        self.assertEqual(2.5, delete_timeout)
        delete_headers = {key.lower(): value for key, value in delete_request.header_items()}
        self.assertEqual("probe-session", delete_headers["Mcp-session-id".lower()])

    def test_http_endpoint_ready_cleans_up_probe_session_after_http_error(self):
        captured_requests = []

        def fake_urlopen(request, timeout=0):
            captured_requests.append((request, timeout))
            if request.get_method() == "POST":
                raise urllib.error.HTTPError(
                    request.full_url,
                    500,
                    "boom",
                    hdrs={"mcp-session-id": "failed-probe"},
                    fp=io.BytesIO(b"error"),
                )
            if request.get_method() == "DELETE":
                return _FakeResponse()
            raise AssertionError(f"unexpected method: {request.get_method()}")

        with mock.patch.object(ensure_uvvis.urllib.request, "urlopen", side_effect=fake_urlopen):
            ready = ensure_uvvis._http_endpoint_ready("http://127.0.0.1:8766/mcp", timeout=1.0)

        self.assertFalse(ready)
        self.assertEqual(2, len(captured_requests))
        self.assertEqual("POST", captured_requests[0][0].get_method())
        self.assertEqual("DELETE", captured_requests[1][0].get_method())


if __name__ == "__main__":
    unittest.main()
