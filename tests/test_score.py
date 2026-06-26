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


# ---- P2: confidence no longer punishes topical breadth (METHODOLOGY_REVIEW F7) ----

def _mixed(country="Iran", n=8):
    """A high-volume, multi-category, multi-source, corroborated country —
    exactly the shape that drove Iran to confidence 0.05 under the old model."""
    cats = [("military_conflict", 8.0), ("diplomatic_tensions", 2.5),
            ("humanitarian_crisis", 3.0), ("neutral", 0.0)]
    srcs = ["Reuters", "AP", "BBC", "Al Jazeera"]
    evs = []
    for cat, w in cats:
        for i in range(n):
            e = _ev(country, cat, w, src=srcs[i % len(srcs)])
            e["corroboration"] = 2
            evs.append(e)
    return evs


def test_confidence_not_floored_for_broad_wellsourced_country():
    cfg = config.load()
    new = score.score(_mixed(), cfg)["countries"]["Iran"]["confidence"]      # default v_d_c
    cfg.settings["score"]["confidence_model"] = "v_d_a"
    legacy = score.score(_mixed(), cfg)["countries"]["Iran"]["confidence"]   # old homogeneity term
    assert new > 0.6        # a broad, well-sourced country is no longer floored
    assert legacy < new     # the old "agreement" term dragged confidence down


# ---- P10: saturation constants are config-exposed (defaults reproduce the curve) ----

def test_transform_defaults_match_closed_form_and_are_monotone():
    import math as _m
    from pipeline.score import _transform
    assert abs(_transform(0.0, 5.0, 1.2) - 1.0) < 1e-9
    expected = 1.0 + 9.0 * (1.0 - _m.exp(-5.0 / 5.0 * 1.2))
    assert abs(_transform(5.0, 5.0, 1.2) - expected) < 1e-9
    assert _transform(3.0, 5.0, 2.0) > _transform(3.0, 5.0, 1.2)   # higher gain saturates faster
    assert 1.0 <= _transform(100.0, 5.0, 1.2) <= 10.0              # clipped to band
