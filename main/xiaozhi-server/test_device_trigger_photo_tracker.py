import sys
import threading
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


from device_trigger_mcp_server import PerDevicePhotoLockManager, PhotoPathTracker


class PhotoPathTrackerTest(unittest.TestCase):
    def test_photo_lock_serializes_same_device_only(self):
        manager = PerDevicePhotoLockManager()
        first_lock, first_wait_ms = manager.acquire("94:a9:90:28:e8:d8")
        self.assertLess(first_wait_ms, 50)

        acquired = threading.Event()
        released = threading.Event()
        wait_values = []

        def acquire_same_device():
            second_lock, second_wait_ms = manager.acquire("94:a9:90:28:e8:d8")
            try:
                wait_values.append(second_wait_ms)
                acquired.set()
            finally:
                second_lock.release()
                released.set()

        worker = threading.Thread(target=acquire_same_device)
        worker.start()
        try:
            time.sleep(0.03)
            self.assertFalse(acquired.is_set())
        finally:
            first_lock.release()

        self.assertTrue(released.wait(1.0))
        worker.join(timeout=1.0)
        self.assertTrue(wait_values)
        self.assertGreaterEqual(wait_values[0], 20)

    def test_photo_lock_allows_different_devices_to_enter(self):
        manager = PerDevicePhotoLockManager()
        first_lock, _ = manager.acquire("94:a9:90:28:e8:d8")

        acquired = threading.Event()
        released = threading.Event()

        def acquire_other_device():
            other_lock, _ = manager.acquire("94:a9:90:28:ea:b4")
            try:
                acquired.set()
            finally:
                other_lock.release()
                released.set()

        worker = threading.Thread(target=acquire_other_device)
        worker.start()
        try:
            self.assertTrue(acquired.wait(0.2))
            self.assertTrue(released.wait(0.2))
        finally:
            first_lock.release()
        worker.join(timeout=1.0)

    def test_find_latest_ignores_non_image_files(self):
        with TemporaryDirectory() as vision_dir, TemporaryDirectory() as by_device_dir:
            tracker = PhotoPathTracker(
                vision_dir=vision_dir,
                by_device_dir=by_device_dir,
                detect_interval_ms=1,
                enable_mirror=True,
            )
            device_id = "94:a9:90:27:3c:84"
            safe_device = "94_a9_90_27_3c_84"
            device_dir = Path(by_device_dir) / safe_device
            device_dir.mkdir(parents=True, exist_ok=True)

            ignored_log = device_dir / "94_a9_90_27_3c_84.log"
            ignored_log.write_text("not a photo", encoding="utf-8")
            valid_photo = device_dir / "sample_20260430_100100.png"
            valid_photo.write_bytes(b"fakepng")

            latest = tracker.find_latest(device_id)

            self.assertIsNotNone(latest)
            self.assertEqual(str(valid_photo.resolve()), latest["local_path"])
            self.assertEqual("sample_20260430_100100.png", latest["file_name"])

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

    def test_grouped_photo_stays_under_group_directory(self):
        with TemporaryDirectory() as vision_dir, TemporaryDirectory() as by_device_dir:
            tracker = PhotoPathTracker(
                vision_dir=vision_dir,
                by_device_dir=by_device_dir,
                detect_interval_ms=1,
                enable_mirror=True,
            )
            device_id = "94:a9:90:27:3c:84"
            shared_photo = Path(vision_dir) / "94-a9-90-27-3c-84_20260430_100100.png"
            shared_photo.write_bytes(b"group-photo")

            photo_meta = tracker.resolve_photo_after_take_photo(
                device_id=device_id,
                requested_photo_name="sample_2",
                baseline=None,
                detect_timeout=0.01,
                group_number=2,
            )

            mirrored = Path(photo_meta["mirrored_path"])
            self.assertEqual(2, photo_meta["group_number"])
            self.assertEqual("2", photo_meta["group_dir_name"])
            self.assertEqual("2", mirrored.parent.name)
            self.assertEqual("94_a9_90_27_3c_84", mirrored.parent.parent.name)

            latest_group_2 = tracker.find_latest(device_id, group_number=2)
            latest_no_group = tracker.find_latest(device_id)
            self.assertEqual(str(mirrored.resolve()), latest_group_2["local_path"])
            self.assertEqual(str(shared_photo.resolve()), latest_no_group["local_path"])
            self.assertFalse((Path(by_device_dir) / "94_a9_90_27_3c_84" / "sample_2.png").exists())


if __name__ == "__main__":
    unittest.main()
