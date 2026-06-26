import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, score
from pipeline.util import utcnow

def _ev(country, cat, w, src="Reuters"):
    return {"country": country, "category": cat, "weight": w, "credibility": 1.0,
            "source": src, "title": f"{cat} story", "link": "", "published": utcnow(), "lang": "en"}

def test_index_in_range_and_monotone():
    cfg = config.load()
    quiet = score.score([_ev("Egypt", "diplomatic_tensions", 2.5)], cfg)
    loud = score.score([_ev("Egypt", "military_conflict", 8.0)] * 3, cfg)
    assert 1.0 <= quiet["composite"] <= 10.0
    assert loud["countries"]["Egypt"]["index"] >= quiet["countries"]["Egypt"]["index"]

def test_empty_is_baseline():
    cfg = config.load()
    r = score.score([], cfg)
    assert abs(r["countries"]["Egypt"]["index"] - 1.0) < 1e-6
    assert r["status"] == "STABLE"
