"""Regional briefing: a deterministic templated summary, optionally polished by
the LLM. Always produces a valid briefing even with no LLM and sparse data.
"""
from __future__ import annotations

from datetime import timedelta

from .config import Config
from . import llm
from .util import to_iso, utcnow


def _salience(e):
    return abs(float(e.get("weight", 0))) * float(e.get("recency_weight", 1)) * float(e.get("credibility", 0.6))


def build(result, cfg: Config, log=None):
    now = utcnow()
    step = int(cfg.settings.get("forecast", {}).get("step_hours", 2))
    window = int(cfg.settings.get("ingest", {}).get("score_window_hours", 48))

    # rank countries
    ranked = sorted(
        ({"name": n, **co} for n, co in result["countries"].items()),
        key=lambda c: c["index"], reverse=True)

    # gather salient threat events
    all_events = []
    for n, co in result["countries"].items():
        for e in co["events"]:
            if float(e.get("weight", 0)) > 0:
                all_events.append({**e, "country": n, "_sal": _salience(e)})
    all_events.sort(key=lambda e: e["_sal"], reverse=True)

    # bullets: top events from distinct countries where possible
    bullets = []
    used = set()
    for e in all_events:
        if e["source_country"] in used and len(used) < 3:
            continue
        bullets.append({"text": e["title"], "cat": e["category"]})
        used.add(e["source_country"])
        if len(bullets) >= 3:
            break
    if len(bullets) < 3:
        for e in all_events:
            b = {"text": e["title"], "cat": e["category"]}
            if b not in bullets:
                bullets.append(b)
            if len(bullets) >= 3:
                break

    top = ranked[0] if ranked else {"name": "the region", "index": result["composite"]}
    top_cat = (top.get("components") or [{}])[0].get("label", "regional") if top.get("components") else "regional"
    status = result["status"].lower()
    headline = (f"{top['name']} leads regional risk on {top_cat.lower()} signals; "
                f"composite holds {result['composite']:.2f} ({status}).")
    if result["composite"] < 4 and not all_events:
        headline = f"Quiet regional window; composite at {result['composite']:.2f} (stable)."

    source_countries = list(dict.fromkeys(e["source_country"] for e in all_events[:12])) or \
        [c["name"] for c in ranked[:3]]

    briefing = {
        "regional_summary_6h": {
            "window_hours": window,
            "source_countries": source_countries,
            "next_refresh_at": to_iso(now + timedelta(hours=step)),
            "headline": headline,
            "bullets": bullets if bullets else [
                {"text": "No qualifying threat signals in the current window.", "cat": "neutral"}],
        }
    }

    # optional LLM polish
    polished = llm.summarize_briefing(cfg, all_events, ranked, log)
    if polished:
        valid = set(cfg.category_keys())
        bl = [{"text": str(b.get("text", "")).strip(),
               "cat": b.get("cat") if b.get("cat") in valid else "neutral"}
              for b in polished.get("bullets", []) if b.get("text")]
        if polished.get("headline") and bl:
            briefing["regional_summary_6h"]["headline"] = str(polished["headline"]).strip()[:140]
            briefing["regional_summary_6h"]["bullets"] = bl[:4]
            if log:
                log.info("briefing: used LLM polish")
    return briefing
