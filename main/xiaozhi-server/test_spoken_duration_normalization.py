import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.utils import textUtils


def test_extracts_combined_chinese_minute_second_duration():
    durations = textUtils.extract_spoken_duration_expressions("深蓝色，一分二十秒。")

    assert durations
    assert durations[0]["text"] == "一分二十秒"
    assert durations[0]["seconds"] == 80
    assert durations[0]["minutes"] == 1.3333


def test_extracts_arabic_minute_second_duration():
    durations = textUtils.extract_spoken_duration_expressions("深蓝色，1分20秒。")

    assert durations
    assert durations[0]["seconds"] == 80
    assert durations[0]["minutes"] == 1.3333


def test_builds_prompt_note_for_codex():
    note = textUtils.build_spoken_duration_normalization_note("深蓝色，一分二十秒。")

    assert "一分二十秒 = 80 seconds = 1.3333 minutes" in note
    assert "do not infer or rewrite a different time" in note
