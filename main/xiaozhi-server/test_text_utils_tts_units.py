import unittest

from core.utils import textUtils


class TextUtilsTTSUnitsTest(unittest.TestCase):
    def test_normalize_tts_text_reads_standalone_nm_as_nanometer(self):
        result = textUtils.normalize_tts_text("请把单位写成nm。")

        self.assertIn("纳米", result)
        self.assertNotIn("nm", result.lower())

    def test_normalize_tts_text_reads_numeric_nm_as_nanometer(self):
        result = textUtils.normalize_tts_text("波长设到546nm。")

        self.assertIn("546纳米", result)


if __name__ == "__main__":
    unittest.main()
