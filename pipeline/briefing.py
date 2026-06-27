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


def _select_bullets(all_events, driver_names, limit):
    """Bullets that back the headline (METHODOLOGY_REVIEW §6 follow-up).

    Order of priority: (1) the single most-salient event of each country the
    headline names, so the brief always shows evidence for its own claim; then
    (2) the most-salient events from countries not yet represented; then (3) any
    remaining salient events. Each bullet keeps its `country` so the source-chip
    list can be aligned with what the bullets actually say.
    """
    out, used_countries, used_text = [], set(), set()

    def _add(e):
        key = " ".join(str(e.get("title", "")).lower().split())
        if not key or key in used_text:
            return False
        out.append({"text": e["title"], "cat": e["category"], "country": e["country"]})
        used_text.add(key)
        used_countries.add(e["country"])
        return True

    for name in driver_names:                         # 1. evidence each named driver
        if len(out) >= limit:
            break
        for e in all_events:
            if e["country"] == name and _add(e):
                break
    for e in all_events:                              # 2. spread across fresh countries
        if len(out) >= limit:
            break
        if e["country"] not in used_countries:
            _add(e)
    for e in all_events:                              # 3. top up with anything salient
        if len(out) >= limit:
            break
        _add(e)
    return out


def build(result, cfg: Config, log=None, prev_composite=None):
    now = utcnow()
    step = int(cfg.settings.get("forecast", {}).get("step_hours", 2))
    window = int(cfg.settings.get("ingest", {}).get("score_window_hours", 48))
    bcfg = cfg.settings.get("briefing", {})
    n_bullets = max(1, int(bcfg.get("bullets", 3)))
    src_max = max(1, int(bcfg.get("source_countries_max", 8)))

    # rank countries (all of them — the brief can reflect any tracked country)
    ranked = sorted(
        ({"name": n, **co} for n, co in result["countries"].items()),
        key=lambda c: c["index"], reverse=True)

    # gather salient threat events across every country
    all_events = []
    for n, co in result["countries"].items():
        for e in co["events"]:
            if float(e.get("weight", 0)) > 0:
                all_events.append({**e, "country": n, "_sal": _salience(e)})
    all_events.sort(key=lambda e: e["_sal"], reverse=True)

    # headline drivers FIRST, so the bullets and chips can be built to back them
    thr = cfg.settings.get("thresholds", {})
    elevated = float(thr.get("elevated", 4.0))
    critical = float(thr.get("critical", 7.0))
    conf_gate = float(bcfg.get("headline_conf_gate", 0.35))
    trend = _compute_trend(result["composite"], prev_composite)
    driver_names, driver_rows = _select_drivers(result["countries"], elevated, conf_gate)
    headline = _compose_headline(result["composite"], result["status"], trend, driver_rows, critical)

    bullets = _select_bullets(all_events, driver_names, n_bullets)

    # chips: drivers + the countries the bullets cite + the next most salient, so
    # whatever the headline names is always present in the source list.
    source_countries = list(dict.fromkeys(
        driver_names
        + [b["country"] for b in bullets]
        + [e["country"] for e in all_events]))[:src_max] or [c["name"] for c in ranked[:3]]

    briefing = {
        "regional_summary_6h": {
            "window_hours": window,
            "source_countries": source_countries,
            "next_refresh_at": to_iso(now + timedelta(hours=step)),
            "headline": headline,
            "bullets": [{"text": b["text"], "cat": b["cat"]} for b in bullets] or [
                {"text": "No qualifying threat signals in the current window.", "cat": "neutral"}],
        }
    }

    # Optional LLM polish — accepted as a *package* (headline + bullets together)
    # only when the headline keeps the composite anchor, so the polish can never
    # leave an LLM headline with deterministic bullets (or vice-versa) out of sync.
    polished = llm.summarize_briefing(
        cfg, all_events, ranked, log,
        composite=result["composite"], status=result["status"],
        trend=trend, drivers=driver_names)
    if polished:
        valid = set(cfg.category_keys())
        bl = [{"text": str(b.get("text", "")).strip(),
               "cat": b.get("cat") if b.get("cat") in valid else "neutral"}
              for b in polished.get("bullets", []) if b.get("text")]
        cand = str(polished.get("headline", "")).strip()[:140]
        if bl and cand and _accept_polished_headline(cand, result["composite"]):
            briefing["regional_summary_6h"]["headline"] = cand
            briefing["regional_summary_6h"]["bullets"] = bl[:n_bullets]
            if log:
                log.info("briefing: used LLM polish (headline + %d bullets)", len(bl[:n_bullets]))
        elif log:
            log.info("briefing: kept deterministic brief (LLM polish rejected)")

    briefing["regional_summary_6h"]["bullets"] = _dedupe_bullets(
        briefing["regional_summary_6h"]["bullets"])
    return briefing
