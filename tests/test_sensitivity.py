import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config


def test_sensitivity_runs_and_reports_range():
    from scripts.sensitivity import run_sensitivity, _fixture
    cfg = config.load()
    res = run_sensitivity(cfg, _fixture(cfg))
    assert {"baseline", "scenarios", "composite_range"} <= set(res)
    assert res["scenarios"]                                    # non-empty
    lo, hi = res["composite_range"]
    assert lo <= res["baseline"]["composite"] <= hi
    assert hi > lo                                             # perturbations move the composite
    assert 1.0 <= lo and hi <= 10.0                            # stays in band
