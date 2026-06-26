"""Fetch RSS/Atom + Google News feeds, normalize, de-duplicate, window-filter."""
from __future__ import annotations

import math
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
            collect(entries, "Google News", c.name, g.get("hl", "en")[:2])

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
            hay = a.raw_title
            matched = [name for name in name_tokens if name in hay]
            for al, target in aliases.items():
                if al in hay and target not in matched:
                    matched.append(target)
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
    by_country: dict[str, list[Article]] = {}
    seen_keys: dict[str, set] = {}
    for a in windowed:
        bucket = by_country.setdefault(a.country, [])
        keys = seen_keys.setdefault(a.country, set())
        k_title = title_key(a.title)
        k_url = canon_url(a.link)
        if k_title in keys or (k_url and k_url in keys):
            stats["dropped_dup"] += 1
            continue
        # fuzzy: compare against already-kept titles in this country
        dup = False
        for b in bucket:
            if jaccard(a._tokens, b._tokens) >= sim:
                dup = True
                break
        if dup:
            stats["dropped_dup"] += 1
            continue
        keys.add(k_title)
        if k_url:
            keys.add(k_url)
        bucket.append(a)

    articles = [a for bucket in by_country.values() for a in bucket]
    stats["kept"] = len(articles)
    log.info("feeds: ok=%d fail=%d raw=%d kept=%d (old=%d dup=%d)",
             stats["feeds_ok"], stats["feeds_fail"], stats["raw"], stats["kept"],
             stats["dropped_old"], stats["dropped_dup"])
    return articles, stats
