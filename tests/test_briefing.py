import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, briefing


def test_llm_briefing_events_include_country(monkeypatch):
    cfg = config.load()
    seen = {}

    def fake_summarize(cfg, top_events, country_scores, log):
        seen["events"] = top_events
        return None

    monkeypatch.setattr(briefing.llm, "summarize_briefing", fake_summarize)
    result = {
        "composite": 4.2,
        "status": "ELEVATED",
        "countries": {
            "Egypt": {
                "index": 4.2,
                "components": [{"label": "Diplomatic tensions"}],
                "events": [{
                    "title": "Egypt diplomatic tension story",
                    "category": "diplomatic_tensions",
                    "source_country": "Egypt",
                    "weight": 2.5,
                    "recency_weight": 1.0,
                    "credibility": 1.0,
                }],
            }
        },
    }

    briefing.build(result, cfg)

    assert seen["events"][0]["country"] == "Egypt"


# ---- P1: region-led, severity-gated hybrid headline (METHODOLOGY_REVIEW §6) ----

def _result(composite, status, countries):
    return {"composite": composite, "status": status, "countries": countries}


def _country(index, confidence, weight=1.0, label="Military conflict"):
    return {"index": index, "confidence": confidence, "weight": weight,
            "components": [{"label": label}], "events": []}


def test_headline_is_region_led_not_argmax_country():
    cfg = config.load()
    result = _result(2.58, "STABLE", {
        "Iran": _country(4.5, 0.05, 1.5),                       # argmax index, floor confidence
        "Israel": _country(2.98, 0.60, 1.5, "Diplomatic tensions"),
    })
    h = briefing.build(result, cfg)["regional_summary_6h"]["headline"].lower()
    assert "leads regional risk" not in h
    assert "composite 2.58" in h
    assert "iran" not in h            # excluded by the confidence gate


def test_headline_names_country_that_clears_gate():
    cfg = config.load()
    result = _result(4.60, "ELEVATED", {
        "Israel": _country(5.2, 0.70, 1.5),
        "Egypt": _country(2.0, 0.50, 1.3, "Diplomatic tensions"),
    })
    h = briefing.build(result, cfg)["regional_summary_6h"]["headline"]
    assert h.lower().startswith("mena composite")
    assert "Israel" in h


def test_headline_broad_based_when_no_country_clears_gate():
    cfg = config.load()
    result = _result(4.50, "ELEVATED", {
        "Egypt": _country(3.0, 0.60, 1.3, "Diplomatic tensions"),
        "Iran": _country(4.2, 0.10, 1.5),                       # elevated index, low confidence
    })
    h = briefing.build(result, cfg)["regional_summary_6h"]["headline"].lower()
    assert "broad-based" in h


def test_headline_reports_rising_trend():
    cfg = config.load()
    result = _result(3.00, "STABLE", {"Egypt": _country(3.0, 0.6, 1.3, "Diplomatic tensions")})
    h = briefing.build(result, cfg, prev_composite=2.0)["regional_summary_6h"]["headline"].lower()
    assert "rising" in h


def test_polished_headline_rejected_without_composite_anchor():
    assert briefing._accept_polished_headline("MENA composite 2.58 holds steady", 2.58) is True
    assert briefing._accept_polished_headline("Iran leads regional risk on conflict", 2.58) is False


# ---- P9: briefing bullets are de-duplicated (METHODOLOGY_REVIEW F13) ----

def test_bullets_are_deduplicated():
    bl = [{"text": "IAEA demands verification", "cat": "diplomatic_tensions"},
          {"text": "IAEA demands verification ", "cat": "diplomatic_tensions"},   # trailing space
          {"text": "Strikes reported in the south", "cat": "military_conflict"}]
    out = briefing._dedupe_bullets(bl)
    assert len(out) == 2
    assert out[0]["text"] == "IAEA demands verification"
    assert out[1]["text"] == "Strikes reported in the south"
