"""Assign a threat category (and thus a weight) to each article.

Deterministic keyword/regex engine is the default and the floor. When an NVIDIA
API key is present, an LLM pass refines the category; any failure falls back to
the keyword label so the pipeline is never blocked on the network.
"""
from __future__ import annotations

import re

from .config import Config
from . import llm


def _compile(cfg: Config):
    compiled = []
    for cat in cfg.categories:
        if cat.keywords:
            compiled.append((cat.key, re.compile(cat.keywords, re.IGNORECASE)))
        else:
            compiled.append((cat.key, None))
    return compiled


def keyword_label(title: str, compiled) -> str:
    """First category (severity-ordered) whose pattern matches; else 'neutral'."""
    text = title or ""
    for key, pat in compiled:
        if pat is None:
            continue
        if pat.search(text):
            return key
    return "neutral"


def source_credibility(cfg: Config, source: str) -> float:
    default = float(cfg.settings.get("score", {}).get("credibility_default", 0.60))
    s = (source or "").lower()
    best = default
    matched = False
    for key, val in cfg.source_credibility.items():
        if key and key in s:
            # take the most specific (longest) matching key
            if not matched or len(key) > best_len:
                best = val
                best_len = len(key)
                matched = True
    return best


def categorize(articles, cfg: Config, log) -> list:
    compiled = _compile(cfg)
    valid_keys = set(cfg.category_keys())
    events = []
    for a in articles:
        cat = keyword_label(a.title, compiled)
        events.append({
            "title": a.title,
            "raw_title": a.raw_title,
            "link": a.link,
            "source": a.source,
            "country": a.country,
            "lang": a.lang,
            "published": a.published,
            "category": cat,
            "weight": cfg.cat_weight.get(cat, 0.0),
            "credibility": source_credibility(cfg, a.source),
        })

    # --- optional LLM refinement ---
    if llm.available(cfg):
        try:
            refined = llm.classify(cfg, [e["title"] for e in events], list(valid_keys), log)
            changed = 0
            for e, label in zip(events, refined):
                if label in valid_keys and label != e["category"]:
                    e["category"] = label
                    e["weight"] = cfg.cat_weight.get(label, 0.0)
                    changed += 1
            log.info("llm: refined %d/%d categories", changed, len(events))
        except Exception as ex:
            log.warning("llm classification failed, keeping keyword labels: %s", ex)

    # category distribution for logging
    dist = {}
    for e in events:
        dist[e["category"]] = dist.get(e["category"], 0) + 1
    log.info("categorize: %d events %s", len(events), dist)
    return events
