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
