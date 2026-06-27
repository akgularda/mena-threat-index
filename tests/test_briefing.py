import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, briefing, llm


def test_extract_json_object_is_not_swallowed_by_inner_array():
    # The briefing returns an OBJECT containing a bullets ARRAY. Array-first
    # parsing (classify's default) would return only the inner list.
    obj_text = '{"headline": "h", "bullets": [{"text": "t", "cat": "neutral"}]}'
    assert isinstance(llm._extract_json(obj_text), list)              # default: array-first
    got = llm._extract_json(obj_text, prefer="object")               # briefing: object-first
    assert isinstance(got, dict) and got["headline"] == "h"
    assert llm._extract_json('["a", "b"]') == ["a", "b"]             # classify path unchanged


def test_llm_briefing_events_include_country(monkeypatch):
    cfg = config.load()
    seen = {}

    def fake_summarize(cfg, top_events, country_scores, log, **kw):
        seen["events"] = top_events
        seen["kw"] = kw
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


# ---- bullets must back the headline (the Oman-in-headline / no-Oman-bullet bug) ----

def _ev(title, cat, country, w):
    return {"title": title, "category": cat, "source_country": country,
            "weight": w, "recency_weight": 1.0, "credibility": 1.0}


def test_driver_country_is_evidenced_in_bullets_and_chips(monkeypatch):
    monkeypatch.setattr(briefing.llm, "summarize_briefing",
                        lambda *a, **k: None)            # force the deterministic path
    cfg = config.load()
    # Oman is the most elevated+confident country (the headline driver) but its
    # event is less salient than Israel's two. The brief must still surface an
    # Oman bullet and list Oman among the source chips.
    result = _result(4.30, "ELEVATED", {
        "Oman": {"index": 4.35, "confidence": 0.50, "weight": 0.6, "components": [],
                 "events": [_ev("Oman moves border units after maritime alert",
                                "border_security", "Oman", 2.0)]},
        "Israel": {"index": 3.90, "confidence": 0.80, "weight": 1.5, "components": [],
                   "events": [_ev("Major strike rocks the north", "military_conflict", "Israel", 8.0),
                              _ev("Air defenses intercept a barrage", "military_conflict", "Israel", 7.0)]},
    })
    out = briefing.build(result, cfg)["regional_summary_6h"]
    assert "Oman above the elevated line" in out["headline"]
    assert any("Oman" in b["text"] for b in out["bullets"])   # bullet backs the headline
    assert "Oman" in out["source_countries"]                  # chip matches the headline


def test_brief_sees_all_countries_and_more_events(monkeypatch):
    seen = {}

    def fake(cfg, top_events, country_scores, log, **kw):
        seen["n_events"] = len(top_events)
        seen["n_countries"] = len(country_scores)
        seen["drivers"] = kw.get("drivers")
        seen["composite"] = kw.get("composite")
        return None

    monkeypatch.setattr(briefing.llm, "summarize_briefing", fake)
    cfg = config.load()
    countries = {c.name: {"index": 4.5, "confidence": 0.6, "weight": 1.0, "components": [],
                          "events": [_ev(f"{c.name} development {i}", "political_instability", c.name, 3.0 + i)
                                     for i in range(3)]}
                 for c in cfg.countries}
    briefing.build(_result(4.5, "ELEVATED", countries), cfg)
    assert seen["n_countries"] == len(cfg.countries)          # all 24 countries reach the LLM
    assert seen["n_events"] == 3 * len(cfg.countries)         # every event reaches the LLM (capped in llm.py)
    assert seen["composite"] == 4.5 and seen["drivers"]       # composite + drivers passed for the anchor


# ---- P9: briefing bullets are de-duplicated (METHODOLOGY_REVIEW F13) ----

def test_bullets_are_deduplicated():
    bl = [{"text": "IAEA demands verification", "cat": "diplomatic_tensions"},
          {"text": "IAEA demands verification ", "cat": "diplomatic_tensions"},   # trailing space
          {"text": "Strikes reported in the south", "cat": "military_conflict"}]
    out = briefing._dedupe_bullets(bl)
    assert len(out) == 2
    assert out[0]["text"] == "IAEA demands verification"
    assert out[1]["text"] == "Strikes reported in the south"
