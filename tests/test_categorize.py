import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from pipeline import config, categorize


# ---- P11: optional literature-anchored weight set (default-off) ----

def test_weight_set_selection_preserves_severity_ordering():
    cats_raw = yaml.safe_load(
        open(os.path.join(config.CONFIG_DIR, "categories.yml"), encoding="utf-8"))
    default = {c.key: c.weight for c in config._build_categories(cats_raw, "default")}
    gold = {c.key: c.weight for c in config._build_categories(cats_raw, "goldstein")}
    assert default["military_conflict"] == 8.0          # default = YAML weight column
    assert gold != default                              # goldstein is a distinct set
    order = ["military_conflict", "terrorism", "border_security",
             "political_instability", "humanitarian_crisis", "diplomatic_tensions"]
    vals = [gold[k] for k in order]
    assert vals == sorted(vals, reverse=True)           # strictly descending severity
    assert gold["trade_agreement"] < 0                  # de-escalation stays negative
    assert gold["neutral"] == 0.0

def test_keyword_labels():
    cfg = config.load(); comp = categorize._compile(cfg)
    assert categorize.keyword_label("Airstrike hits border town", comp) == "military_conflict"
    assert categorize.keyword_label("Suicide bomber kills 10", comp) == "terrorism"
    assert categorize.keyword_label("Troops mass along the border", comp) == "border_security"
    assert categorize.keyword_label("Opposition leader arrested", comp) == "political_instability"
    assert categorize.keyword_label("Minister attends summit", comp) == "diplomatic_tensions"
    assert categorize.keyword_label("Two sides sign trade agreement", comp) == "trade_agreement"
    assert categorize.keyword_label("Local football match result", comp) == "neutral"


# ---- P8: categorisation robustness (METHODOLOGY_REVIEW F3/F4) ----

def test_combat_casualties_route_to_military_not_humanitarian():
    cfg = config.load(); comp = categorize._compile(cfg)
    assert categorize.keyword_label("Soldiers killed in fighting overnight", comp) == "military_conflict"


def test_ceasefire_collapse_is_military_not_border():
    cfg = config.load(); comp = categorize._compile(cfg)
    assert categorize.keyword_label("Ceasefire collapses amid renewed clashes", comp) == "military_conflict"


def test_ceasefire_holding_stays_border_security():
    cfg = config.load(); comp = categorize._compile(cfg)
    # de-escalation must NOT be escalated by the new combat patterns
    assert categorize.keyword_label("Ceasefire holds across the north", comp) == "border_security"


def test_bare_threat_no_longer_forces_diplomatic():
    cfg = config.load(); comp = categorize._compile(cfg)
    assert categorize.keyword_label("Report highlights threat to supply chains", comp) == "neutral"


def test_llm_uncertain_label_keeps_keyword(monkeypatch):
    cfg = config.load()

    class _Art:
        def __init__(self, title):
            self.title = title; self.raw_title = title; self.link = ""
            self.source = "Reuters"; self.country = "Egypt"; self.lang = "en"
            self.published = None; self.corroboration = 1

    monkeypatch.setattr(categorize.llm, "available", lambda cfg: True)
    monkeypatch.setattr(categorize.llm, "classify", lambda cfg, titles, keys, log: ["uncertain"])

    class _Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass

    events = categorize.categorize([_Art("Airstrike hits border town")], cfg, _Log())
    assert events[0]["category"] == "military_conflict"   # LLM abstention keeps the keyword label
