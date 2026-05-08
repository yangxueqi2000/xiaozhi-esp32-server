import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.utils import experiment_resume


def test_resume_request_is_intent_but_not_passive_llm_log_context():
    text = "继续刚才的实验步骤"

    assert experiment_resume.is_resume_experiment_request(text)
    assert not experiment_resume.should_load_device_log_context(text)


def test_export_request_can_still_load_device_log_context():
    assert experiment_resume.should_load_device_log_context("生成实验报告")
