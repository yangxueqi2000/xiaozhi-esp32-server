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

    def test_normalize_tts_text_reads_sample_range_dash_as_to(self):
        result = textUtils.normalize_tts_text("请把1-5号样品放好。")

        self.assertIn("到", result)
        self.assertIn("号样品", result)
        self.assertNotIn("-5", result)

    def test_normalize_tts_text_reads_aromatic_dash_without_minus(self):
        result = textUtils.normalize_tts_text("加入4-硝基苯酚。")

        self.assertIn("硝基苯酚", result)
        self.assertNotIn("-", result)


if __name__ == "__main__":
    unittest.main()
