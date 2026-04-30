import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


fake_fastmcp_module = types.ModuleType("mcp.server.fastmcp")


class _DummyContext:
    pass


class _DummyFastMCP:
    pass


fake_fastmcp_module.Context = _DummyContext
fake_fastmcp_module.FastMCP = _DummyFastMCP
sys.modules.setdefault("mcp", types.ModuleType("mcp"))
sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
sys.modules.setdefault("mcp.server.fastmcp", fake_fastmcp_module)


from device_trigger_mcp_server import PhotoPathTracker


class PhotoPathTrackerTest(unittest.TestCase):
    def test_resolve_photo_after_take_photo_detects_new_shared_photo_and_mirrors_it(self):
        with TemporaryDirectory() as vision_dir, TemporaryDirectory() as by_device_dir:
            tracker = PhotoPathTracker(
                vision_dir=vision_dir,
                by_device_dir=by_device_dir,
                detect_interval_ms=1,
                enable_mirror=True,
            )
            device_id = "94:a9:90:27:3c:84"

            old_file = Path(vision_dir) / "94-a9-90-27-3c-84_20260430_100000.png"
            old_file.write_bytes(b"old")
            baseline = tracker.find_latest(device_id)

            self.assertIsNotNone(baseline)
            self.assertEqual(str(old_file.resolve()), baseline["local_path"])

            time.sleep(0.02)
            new_file = Path(vision_dir) / "94-a9-90-27-3c-84_20260430_100100.png"
            new_file.write_bytes(b"new")

            photo_meta = tracker.resolve_photo_after_take_photo(
                device_id=device_id,
                requested_photo_name="一号样品_20260430_100100",
                baseline=baseline,
                detect_timeout=0.2,
            )

            self.assertTrue(photo_meta["found"])
            self.assertTrue(photo_meta["is_new_photo"])
            self.assertEqual("new_file_after_take_photo", photo_meta["detected_by"])
            self.assertTrue(photo_meta["mirrored_path"])
            self.assertTrue(Path(photo_meta["mirrored_path"]).is_file())
            self.assertIn("一号样品_20260430_100100", Path(photo_meta["mirrored_path"]).stem)


if __name__ == "__main__":
    unittest.main()
