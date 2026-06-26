import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, categorize

def test_keyword_labels():
    cfg = config.load(); comp = categorize._compile(cfg)
    assert categorize.keyword_label("Airstrike hits border town", comp) == "military_conflict"
    assert categorize.keyword_label("Suicide bomber kills 10", comp) == "terrorism"
    assert categorize.keyword_label("Troops mass along the border", comp) == "border_security"
    assert categorize.keyword_label("Opposition leader arrested", comp) == "political_instability"
    assert categorize.keyword_label("Minister attends summit", comp) == "diplomatic_tensions"
    assert categorize.keyword_label("Two sides sign trade agreement", comp) == "trade_agreement"
    assert categorize.keyword_label("Local football match result", comp) == "neutral"
