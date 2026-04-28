from importlib import util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "preprocess.py"
_SPEC = util.spec_from_file_location("preprocess_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_compute_check_qa_probs = _MODULE._compute_check_qa_probs


def test_compute_check_qa_probs_balances_yes_no_average():
    p_yes, p_no = _compute_check_qa_probs(check_count=30, no_check_count=70, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.21428571428571427


def test_compute_check_qa_probs_handles_no_non_check_positions():
    p_yes, p_no = _compute_check_qa_probs(check_count=10, no_check_count=0, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.0
