import socket
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


fake_logger_module = types.ModuleType("config.logger")
fake_logger_module.setup_logging = lambda: None
sys.modules.setdefault("config.logger", fake_logger_module)
sys.modules.setdefault("opuslib_next", types.ModuleType("opuslib_next"))

fake_pydub_module = types.ModuleType("pydub")
fake_pydub_module.AudioSegment = object
sys.modules.setdefault("pydub", fake_pydub_module)


from core.utils.util import get_local_ip


class _FakeSocket:
    def __init__(self, ip_address: str):
        self._ip_address = ip_address

    def connect(self, address):
        return None

    def getsockname(self):
        return (self._ip_address, 0)

    def close(self):
        return None


class GetLocalIpTest(unittest.TestCase):
    def test_prefers_real_private_lan_ip_over_benchmark_adapter(self):
        iface_addrs = {
            "Meta": [
                types.SimpleNamespace(
                    family=socket.AF_INET,
                    address="198.18.0.1",
                )
            ],
            "WLAN": [
                types.SimpleNamespace(
                    family=socket.AF_INET,
                    address="192.168.1.106",
                )
            ],
        }
        iface_stats = {
            "Meta": types.SimpleNamespace(isup=True),
            "WLAN": types.SimpleNamespace(isup=True),
        }

        with patch("core.utils.util.psutil.net_if_addrs", return_value=iface_addrs):
            with patch("core.utils.util.psutil.net_if_stats", return_value=iface_stats):
                self.assertEqual("192.168.1.106", get_local_ip())

    def test_falls_back_to_socket_probe_when_no_good_interface_candidate_exists(self):
        iface_addrs = {
            "Meta": [
                types.SimpleNamespace(
                    family=socket.AF_INET,
                    address="198.18.0.1",
                )
            ],
        }
        iface_stats = {
            "Meta": types.SimpleNamespace(isup=True),
        }

        with patch("core.utils.util.psutil.net_if_addrs", return_value=iface_addrs):
            with patch("core.utils.util.psutil.net_if_stats", return_value=iface_stats):
                with patch(
                    "core.utils.util.socket.socket",
                    return_value=_FakeSocket("10.0.0.5"),
                ):
                    self.assertEqual("10.0.0.5", get_local_ip())


if __name__ == "__main__":
    unittest.main()
