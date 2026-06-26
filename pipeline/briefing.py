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


def _compute_trend(composite, prev_composite):
    if prev_composite is None:
        return "steady"
    diff = float(composite) - float(prev_composite)
    if diff > 0.05:
        return "rising"
    if diff < -0.05:
        return "easing"
    return "steady"


def _select_drivers(countries, elevated, conf_gate, limit=2):
    """Countries that are BOTH materially elevated AND adequately sourced.
    Sorted by strategic weight x (index-1) x confidence. Returns (names, rows)."""
    rows = []
    for name, co in countries.items():
        idx = float(co.get("index", 0.0))
        conf = float(co.get("confidence", 0.0))
        if idx >= elevated and conf >= conf_gate:
            rows.append((name, idx, conf, float(co.get("weight", 1.0))))
    rows.sort(key=lambda r: r[3] * (r[1] - 1.0) * r[2], reverse=True)
    rows = rows[:limit]
    return [r[0] for r in rows], rows


def _compose_headline(composite, status, trend, driver_rows, critical):
    """Region-led, severity-gated headline (METHODOLOGY_REVIEW §6)."""
    s = status.lower()
    if driver_rows:
        names = " and ".join(r[0] for r in driver_rows)
        line = "critical" if any(r[1] >= critical for r in driver_rows) else "elevated"
        return (f"MENA composite {composite:.2f} ({s}, {trend}); "
                f"{names} above the {line} line.")
    if status == "STABLE":
        return f"Quiet regional window — composite {composite:.2f} (stable, {trend})."
    return (f"MENA composite {composite:.2f} ({s}, {trend}); "
            f"risk broad-based, no single country dominant.")


def _accept_polished_headline(headline, composite):
    """Only accept an LLM-polished headline if it keeps the composite anchor, so
    the polish can't revert the region-first framing to a single-country alarm."""
    return f"{composite:.2f}" in (headline or "")


def _dedupe_bullets(bullets):
    """Drop repeated bullets (by normalised text) so the same event doesn't
    appear twice in the briefing (METHODOLOGY_REVIEW F13)."""
    seen, out = set(), []
    for b in bullets:
        key = " ".join(str(b.get("text", "")).lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(b)
    return out


def build(result, cfg: Config, log=None, prev_composite=None):
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

    thr = cfg.settings.get("thresholds", {})
    elevated = float(thr.get("elevated", 4.0))
    critical = float(thr.get("critical", 7.0))
    conf_gate = float(cfg.settings.get("briefing", {}).get("headline_conf_gate", 0.35))
    trend = _compute_trend(result["composite"], prev_composite)
    _, driver_rows = _select_drivers(result["countries"], elevated, conf_gate)
    headline = _compose_headline(result["composite"], result["status"], trend, driver_rows, critical)

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
        if bl:
            briefing["regional_summary_6h"]["bullets"] = bl[:4]
            cand = str(polished.get("headline", "")).strip()[:140]
            if cand and _accept_polished_headline(cand, result["composite"]):
                briefing["regional_summary_6h"]["headline"] = cand
            if log:
                log.info("briefing: used LLM polish (headline kept=%s)",
                         bool(cand and _accept_polished_headline(cand, result["composite"])))

    briefing["regional_summary_6h"]["bullets"] = _dedupe_bullets(
        briefing["regional_summary_6h"]["bullets"])
    return briefing
