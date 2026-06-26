"""Fetch RSS/Atom + Google News feeds, normalize, de-duplicate, window-filter."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from urllib.parse import quote

import feedparser

from .config import Config, Country
from .util import (age_hours, canon_url, http_get, jaccard, parse_dt, session,
                   strip_source_suffix, title_key, title_norm, token_set, utcnow)


@dataclass
class Article:
    title: str            # cleaned (source suffix stripped)
    raw_title: str
    link: str
    published: object     # datetime (aware UTC) or None
    source: str
    country: str
    lang: str
    _tokens: set = field(default_factory=set, repr=False)
    _sources: set = field(default_factory=set, repr=False)
    corroboration: int = 1


def _gnews_url(q: str, hl: str, gl: str, ceid: str, when_days: int) -> str:
    full_q = f"{q} when:{when_days}d"
    return ("https://news.google.com/rss/search?q=" + quote(full_q) +
            f"&hl={hl}&gl={gl}&ceid={quote(ceid)}")


def _parse_feed(sess, url: str, log) -> list:
    """Fetch bytes through our retry session (handles UA + redirects), then parse."""
    try:
        resp = http_get(sess, url)
        if resp.status_code != 200:
            log.warning("feed %s -> HTTP %s", url[:80], resp.status_code)
            return []
        parsed = feedparser.parse(resp.content)
        return parsed.entries or []
    except Exception as e:  # one bad feed must never abort the run
        log.warning("feed %s failed: %s", url[:80], e)
        return []


def _entry_to_article(e, source: str, country: str, lang: str) -> Article | None:
    raw_title = (getattr(e, "title", "") or "").strip()
    if not raw_title:
        return None
    link = (getattr(e, "link", "") or "").strip()
    published = parse_dt(getattr(e, "published_parsed", None) or
                         getattr(e, "published", None) or
                         getattr(e, "updated", None))
    src = source
    # Google News nests the real outlet under entry.source.title
    if getattr(e, "source", None) and getattr(e.source, "title", None):
        src = e.source.title
    title = strip_source_suffix(raw_title)
    return Article(title=title, raw_title=raw_title, link=link, published=published,
                   source=src, country=country, lang=lang, _tokens=token_set(title))


def _dedupe(windowed, sim):
    """Per-country de-duplication (exact title/url key, then fuzzy Jaccard).

    Counts the distinct corroborating sources of each kept article on
    `corroboration` (>=1), so the score's confidence can reward cross-source
    agreement instead of category homogeneity (METHODOLOGY_REVIEW F7 / P2).
    Returns (kept_articles, dropped_count).
    """
    by_country: dict = {}
    seen_keys: dict = {}
    kept_by_key: dict = {}
    dropped = 0
    for a in windowed:
        bucket = by_country.setdefault(a.country, [])
        keys = seen_keys.setdefault(a.country, set())
        kmap = kept_by_key.setdefault(a.country, {})
        k_title = title_key(a.title)
        k_url = canon_url(a.link)
        kept = kmap.get(k_title) or (kmap.get(k_url) if k_url else None)
        if kept is None:
            for b in bucket:
                if jaccard(a._tokens, b._tokens) >= sim:
                    kept = b
                    break
        if kept is not None:
            kept._sources.add(a.source)
            dropped += 1
            continue
        a._sources.add(a.source)
        keys.add(k_title)
        kmap[k_title] = a
        if k_url:
            keys.add(k_url)
            kmap[k_url] = a
        bucket.append(a)
    articles = [a for bucket in by_country.values() for a in bucket]
    for a in articles:
        a.corroboration = len(a._sources)
    return articles, dropped


# A few names need an explicit exclusion so a longer name doesn't trigger them.
_ATTR_EXCLUSIONS = {"Sudan": ["South Sudan"]}


def _term_in(title, term):
    """Whole-word match for Latin terms; substring for native scripts (Arabic,
    Hebrew, ... have no simple \\b)."""
    low = (title or "").lower()
    t = str(term).lower()
    if not t:
        return False
    if t.isascii():
        return re.search(r"\b" + re.escape(t) + r"\b", low) is not None
    return t in low


def _passes_attribution(title, match_terms, exclude_terms):
    """Keep a broad Google-News result for a country only if the title actually
    names it (English OR native script) and contains no excluded ambiguous term.
    Empty match_terms -> no filtering, so unconfigured countries don't regress
    (METHODOLOGY_REVIEW F2 / PATCH_PLAN P13 follow-up)."""
    if not match_terms:
        return True
    if not any(_term_in(title, m) for m in match_terms):
        return False
    if any(_term_in(title, x) for x in (exclude_terms or [])):
        return False
    return True


def _match_countries(title: str, names: list, aliases: dict) -> list:
    """Attribute a pan-regional headline to tracked countries by WHOLE-WORD match.

    A bare substring match over-attributes — "woman"/"Romania" both contain
    "oman", "South Sudan" contains "Sudan" (METHODOLOGY_REVIEW F2). Word
    boundaries plus a small exclusion map fix that without losing real mentions.
    """
    low = (title or "").lower()
    matched = []
    for name in names:
        probe = low
        for ex in _ATTR_EXCLUSIONS.get(name, []):
            probe = probe.replace(ex.lower(), " ")
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", probe):
            matched.append(name)
    for al, target in aliases.items():
        if target in matched:
            continue
        if re.search(r"\b" + re.escape(al.lower()) + r"\b", low):
            matched.append(target)
    return matched


def fetch_all(cfg: Config, log) -> tuple[list, dict]:
    """Return (articles, stats). Articles are de-duplicated globally per country."""
    s = cfg.settings.get("ingest", {})
    window_h = int(s.get("window_hours", 72))
    when_days = max(1, math.ceil(window_h / 24))
    ua = s.get("user_agent")
    timeout = int(s.get("per_feed_timeout_s", 20))
    sim = float(s.get("dedupe_similarity", 0.90))
    max_items = int(s.get("max_items_per_feed", 120))
    sess = session(ua, timeout)
    now = utcnow()

    stats = {"feeds_ok": 0, "feeds_fail": 0, "raw": 0, "kept": 0, "dropped_old": 0,
             "dropped_dup": 0}
    raw_articles: list[Article] = []

    def collect(entries, source, country, lang):
        n = 0
        for e in entries[:max_items]:
            a = _entry_to_article(e, source, country, lang)
            if a:
                raw_articles.append(a)
                n += 1
        return n

    # --- per-country curated feeds + Google News queries ---
    for c in cfg.countries:
        for f in c.feeds:
            entries = _parse_feed(sess, f["url"], log)
            stats["feeds_ok" if entries else "feeds_fail"] += 1
            collect(entries, f.get("source", "feed"), c.name, f.get("lang", c.lang))
        for g in c.gnews:
            url = _gnews_url(g["q"], g.get("hl", "en-US"), g.get("gl", "US"),
                             g.get("ceid", "US:en"), when_days)
            entries = _parse_feed(sess, url, log)
            stats["feeds_ok" if entries else "feeds_fail"] += 1
            # Broad query layer: keep only results whose title actually names the country.
            kept = [e for e in entries
                    if _passes_attribution(getattr(e, "title", "") or "", c.match, c.exclude)]
            collect(kept, "Google News", c.name, g.get("hl", "en")[:2])

    # --- shared pan-regional feeds: attribute by country mention ---
    name_tokens = {c.name: c for c in cfg.countries}
    aliases = {"UAE": "United Arab Emirates", "Emirates": "United Arab Emirates",
               "Gaza": "Palestine", "West Bank": "Palestine", "Tehran": "Iran",
               "Türkiye": "Turkey"}
    for f in cfg.shared_feeds:
        entries = _parse_feed(sess, f["url"], log)
        stats["feeds_ok" if entries else "feeds_fail"] += 1
        for e in entries[:max_items]:
            a = _entry_to_article(e, f.get("source", "feed"), "", f.get("lang", "en"))
            if not a:
                continue
            matched = _match_countries(a.raw_title, list(name_tokens), aliases)
            for name in matched:
                raw_articles.append(Article(
                    title=a.title, raw_title=a.raw_title, link=a.link,
                    published=a.published, source=a.source, country=name,
                    lang=a.lang, _tokens=a._tokens))

    stats["raw"] = len(raw_articles)

    # --- window filter ---
    windowed = []
    for a in raw_articles:
        if a.published is None:
            windowed.append(a)  # undated: keep (rare); treated as fresh-ish in scoring
            continue
        if age_hours(a.published, now) <= window_h:
            windowed.append(a)
        else:
            stats["dropped_old"] += 1

    # --- de-duplicate, per country (exact title/url key, then fuzzy) ---
    articles, dropped_dup = _dedupe(windowed, sim)
    stats["dropped_dup"] += dropped_dup
    stats["kept"] = len(articles)
    log.info("feeds: ok=%d fail=%d raw=%d kept=%d (old=%d dup=%d)",
             stats["feeds_ok"], stats["feeds_fail"], stats["raw"], stats["kept"],
             stats["dropped_old"], stats["dropped_dup"])
    return articles, stats
