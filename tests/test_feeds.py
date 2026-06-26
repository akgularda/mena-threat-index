import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import feeds
from pipeline.feeds import Article, _dedupe
from pipeline.util import token_set

# Pan-regional shared-feed attribution must match a country as a whole word,
# not a bare substring (METHODOLOGY_REVIEW F2 / PATCH_PLAN P4).
NAMES = ["Oman", "Sudan", "Turkey", "Saudi Arabia", "Iran", "Palestine"]
ALIASES = {"UAE": "United Arab Emirates", "Gaza": "Palestine",
           "Tehran": "Iran", "Türkiye": "Turkey"}


def test_substring_false_positive_not_attributed():
    # "woman" and "Romania" both contain "oman" as a substring but are not Oman
    assert "Oman" not in feeds._match_countries("Saudi woman wins science prize", NAMES, ALIASES)
    assert "Oman" not in feeds._match_countries("Romania and Turkey sign defence pact", NAMES, ALIASES)


def test_south_sudan_not_attributed_to_sudan():
    assert "Sudan" not in feeds._match_countries("South Sudan ceasefire holds", NAMES, ALIASES)


def test_real_mentions_still_attributed():
    assert "Oman" in feeds._match_countries("Oman mediates regional talks", NAMES, ALIASES)
    assert "Sudan" in feeds._match_countries("Sudan army advances on the capital", NAMES, ALIASES)
    assert "Turkey" in feeds._match_countries("Romania and Turkey sign defence pact", NAMES, ALIASES)


def test_aliases_resolve_to_country():
    assert "Palestine" in feeds._match_countries("Gaza strikes reported overnight", NAMES, ALIASES)
    assert "Iran" in feeds._match_countries("Tehran responds to new sanctions", NAMES, ALIASES)


# ---- P2 plumbing: dedup counts distinct corroborating sources (F7) ----

def _art(title, source, country="Iran"):
    return Article(title=title, raw_title=title, link="", published=None,
                   source=source, country=country, lang="en", _tokens=token_set(title))


def test_corroboration_counts_distinct_sources():
    arts = [_art("Iran strikes target in region", "Reuters"),
            _art("Iran strikes target in region", "AP"),          # exact-key duplicate
            _art("Iran strikes target in the region", "BBC")]     # fuzzy duplicate
    kept, dropped = _dedupe(arts, 0.5)
    assert len(kept) == 1
    assert kept[0].corroboration == 3
    assert dropped == 2


def test_corroboration_is_one_for_singleton():
    kept, dropped = _dedupe([_art("Lone Iran story", "Reuters")], 0.9)
    assert kept[0].corroboration == 1
    assert dropped == 0


# ---- P13 follow-up: verify Google-News results actually name the country ----

def test_gnews_attribution_requires_country_term():
    from pipeline.feeds import _passes_attribution
    match = ["Oman", "سلطنة عمان", "مسقط", "عُمان"]
    excl = ["Amman", "عمّان"]
    assert _passes_attribution("Oman warns ships in the Strait of Hormuz", match, excl) is True
    assert _passes_attribution("مسقط تبلغ أوروبا باحتمالية فرض رسوم", match, excl) is True   # Muscat (ar)
    assert _passes_attribution("Amman protests grow across Jordan", match, excl) is False    # excluded
    assert _passes_attribution("Oman and Amman both mentioned", match, excl) is False         # ambiguous -> precision
    assert _passes_attribution("Unrelated technology headline", match, excl) is False         # no country term
    assert _passes_attribution("anything at all", [], []) is True                             # unconfigured -> no filter


def test_config_loads_attribution_terms():
    from pipeline import config
    cfg = config.load()
    oman = next(c for c in cfg.countries if c.name == "Oman")
    assert "Amman" in oman.exclude
    assert "Oman" in oman.match
