"""Scoring: recency decay, source credibility, volume shrinkage, confidence,
per-country index and the smoothed composite.

Per-country index preserves the BNTI transform so the in-app "why this score"
card still holds:   index = 1 + 9 * (1 - exp(-eff/5 * 1.2))
where `eff` is a recency-decayed, credibility-weighted, shrinkage-stabilized
mean event weight (clamped at >= 0 so de-escalation pulls toward baseline 1.0).
"""
from __future__ import annotations

import math

import numpy as np

from .config import Config
from .util import age_hours, clip, to_iso, utcnow


def _transform(eff, scale, gain):
    """Saturating map from mean event weight to the 1-10 index. Constants are
    config-exposed (score.saturation_scale, score.saturation_gain) so they can be
    tuned and sensitivity-tested without touching code (METHODOLOGY_REVIEW F6/P10).
    Defaults (5.0, 1.2) reproduce the inherited BNTI curve exactly."""
    return clip(1.0 + 9.0 * (1.0 - math.exp(-eff / scale * gain)), 1.0, 10.0)


def _per_country(name, events, cfg: Config, prior_raw, prev_index, now):
    sc = cfg.settings.get("score", {})
    half_life = float(sc.get("recency_half_life_h", 18))
    window_h = float(cfg.settings.get("ingest", {}).get("score_window_hours", 48))
    k_shrink = float(sc.get("shrink_k", 3))
    n0 = float(sc.get("conf_volume_scale", 4.0))
    div_target = float(sc.get("conf_diversity_target", 4))
    baseline = float(sc.get("baseline_index", 1.0))

    qualifying = []
    for e in events:
        pub = e.get("published")
        a = age_hours(pub, now) if pub else 0.0
        if a > window_h:
            continue
        lam = 0.5 ** (a / half_life)
        e2 = dict(e)
        e2["age_h"] = a
        e2["recency_weight"] = lam
        qualifying.append(e2)

    n_events = len(qualifying)
    if n_events == 0:
        return {
            "index": round(baseline, 2), "raw_score": 0.0, "confidence": 0.05,
            "n_events": 0, "components": [], "trend": 0.0, "events": [],
        }, baseline, 0.0

    lam = np.array([e["recency_weight"] for e in qualifying], dtype=float)
    cred = np.array([float(e.get("credibility", 0.6)) for e in qualifying], dtype=float)
    w = np.array([float(e.get("weight", 0.0)) for e in qualifying], dtype=float)
    aw = lam * cred                      # averaging weights
    denom = float(aw.sum())
    avg_w = float((aw * w).sum() / denom) if denom > 0 else 0.0

    # Kish effective sample size
    n_eff = (denom ** 2) / float((aw ** 2).sum()) if denom > 0 else 0.0

    # Empirical-Bayes shrinkage toward the country baseline (or 0 at cold start)
    m_prior = float(prior_raw) if prior_raw is not None else 0.0
    avg_w_shrunk = (n_eff * avg_w + k_shrink * m_prior) / (n_eff + k_shrink) if (n_eff + k_shrink) > 0 else avg_w

    eff = max(0.0, avg_w_shrunk)
    index = _transform(eff, float(sc.get("saturation_scale", 5.0)),
                       float(sc.get("saturation_gain", 1.2)))

    # ---- confidence ----
    # Volume adequacy (Kish n_eff) and source diversity (Shannon entropy) are
    # always used. The third term depends on confidence_model:
    #   v_d_c (default) -> cross-source CORROBORATION of events  [METHODOLOGY_REVIEW F7]
    #   v_d             -> drop the third term entirely
    #   v_d_a (legacy)  -> spread of event severities (category homogeneity)
    model = str(sc.get("confidence_model", "v_d_c")).lower()
    V = 1.0 - math.exp(-n_eff / n0) if n0 > 0 else 0.0
    sources = {}
    for e in qualifying:
        sources[e.get("source", "?")] = sources.get(e.get("source", "?"), 0.0) + e["recency_weight"]
    tot = sum(sources.values()) or 1.0
    shares = [v / tot for v in sources.values()]
    H = -sum(p * math.log(p) for p in shares if p > 0)
    D = clip(H / math.log(div_target), 0.0, 1.0) if div_target > 1 else 0.0
    if model == "v_d":
        confidence = clip((max(V, 1e-6) * max(D, 1e-6)) ** 0.5, 0.05, 0.99)
    elif model == "v_d_a":
        wbar = float((aw * w).sum() / denom) if denom > 0 else 0.0
        var_w = float((aw * (w - wbar) ** 2).sum() / denom) if denom > 0 else 0.0
        A = 1.0 - min(1.0, math.sqrt(max(0.0, var_w)) / 3.0)
        confidence = clip((max(V, 1e-6) * max(D, 1e-6) * max(A, 1e-6)) ** (1.0 / 3.0), 0.05, 0.99)
    else:  # v_d_c
        corr = np.array([float(e.get("corroboration", 1)) for e in qualifying], dtype=float)
        c_e = 1.0 - 0.5 * np.exp(-(corr - 1.0))   # 0.5 single-source -> ~1 multi-source
        C = float((aw * c_e).sum() / denom) if denom > 0 else 0.0
        confidence = clip((max(V, 1e-6) * max(D, 1e-6) * max(C, 1e-6)) ** (1.0 / 3.0), 0.05, 0.99)

    # ---- category components (for the "why" breakdown) ----
    comp = {}
    for e in qualifying:
        c = e["category"]
        d = comp.setdefault(c, {"count": 0, "mass": 0.0, "weight": float(e.get("weight", 0.0))})
        d["count"] += 1
        d["mass"] += e["recency_weight"] * float(e.get("credibility", 0.6)) * float(e.get("weight", 0.0))
    components = [
        {"category": k, "label": cfg.cat_label.get(k, k), "count": v["count"],
         "contribution": round(v["mass"], 3), "weight": v["weight"]}
        for k, v in comp.items()
    ]
    components.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    trend = round(index - float(prev_index), 2) if prev_index is not None else 0.0

    # ---- events payload (newest first) ----
    qualifying.sort(key=lambda e: (e.get("published") or now), reverse=True)
    ev_out = []
    for e in qualifying:
        d = e.get("published")
        ev_out.append({
            "title": e["title"],
            "translated_title": e["title"],
            "category": e["category"],
            "source_country": name,
            "source": e.get("source", "—"),
            "date": to_iso(d) if d else to_iso(now),
            "link": e.get("link", ""),
            "weight": round(float(e.get("weight", 0.0)), 2),
            "credibility": round(float(e.get("credibility", 0.6)), 2),
            "recency_weight": round(float(e["recency_weight"]), 3),
            "corroboration": int(e.get("corroboration", 1)),
        })

    country_obj = {
        "index": round(index, 2),
        "raw_score": round(avg_w_shrunk, 3),
        "confidence": round(confidence, 3),
        "n_events": n_events,
        "components": components,
        "trend": trend,
        "events": ev_out,
    }
    # min age among events => freshest signal age, for composite freshness
    fresh_age = min(e["age_h"] for e in qualifying)
    return country_obj, index, fresh_age


def score(events, cfg: Config, priors=None, prev_index=None, recent_diffs=None,
          prev_composite=None, log=None):
    priors = priors or {}
    prev_index = prev_index or {}
    recent_diffs = recent_diffs or []
    now = utcnow()
    sc = cfg.settings.get("score", {})
    thr = cfg.settings.get("thresholds", {})
    fresh_hl = float(sc.get("freshness_half_life_h", 12))

    by_country = {}
    for e in events:
        by_country.setdefault(e["country"], []).append(e)

    countries = {}
    fresh_ages = []
    n_with_events = 0
    total_events = 0
    weighted_idx_num = 0.0
    weighted_idx_den = 0.0
    weighted_conf_num = 0.0

    for c in cfg.countries:
        evs = by_country.get(c.name, [])
        obj, idx, fresh_age = _per_country(
            c.name, evs, cfg, priors.get(c.name), prev_index.get(c.name), now)
        obj["code"] = c.iso2
        obj["weight"] = c.weight
        countries[c.name] = obj
        if obj["n_events"] > 0:
            n_with_events += 1
            fresh_ages.append(fresh_age)
        total_events += obj["n_events"]
        weighted_idx_num += c.weight * idx
        weighted_idx_den += c.weight
        weighted_conf_num += c.weight * obj["confidence"]

    composite_raw = weighted_idx_num / weighted_idx_den if weighted_idx_den else 1.0

    # ---- EWMA smoothing with asymmetric spike guard ----
    alpha = float(sc.get("ewma_alpha", 0.20))
    alpha_spike = float(sc.get("ewma_alpha_spike", 0.50))
    spike_min = float(sc.get("spike_min_jump", 1.0))
    if prev_composite is None:
        composite = composite_raw
    else:
        diff = composite_raw - float(prev_composite)
        sigma = float(np.std(recent_diffs)) if len(recent_diffs) >= 3 else 0.0
        a = alpha_spike if abs(diff) > max(spike_min, 2.0 * sigma) else alpha
        composite = a * composite_raw + (1.0 - a) * float(prev_composite)
    composite = clip(composite, 1.0, 10.0)

    coverage = n_with_events / max(1, len(cfg.countries))
    staleness = min(fresh_ages) if fresh_ages else 6.0
    freshness = 0.5 ** (staleness / fresh_hl)
    conf_comp = clip(coverage * freshness *
                     (weighted_idx_den and weighted_conf_num / weighted_idx_den or 0.0),
                     0.02, 0.99)

    elevated = float(thr.get("elevated", 4.0))
    critical = float(thr.get("critical", 7.0))
    status = "CRITICAL" if composite >= critical else "ELEVATED" if composite >= elevated else "STABLE"

    if log:
        log.info("score: composite_raw=%.2f smoothed=%.2f status=%s coverage=%.0f%% conf=%.2f events=%d",
                 composite_raw, composite, status, coverage * 100, conf_comp, total_events)

    return {
        "composite_raw": round(composite_raw, 3),
        "composite": round(composite, 2),
        "status": status,
        "confidence": round(conf_comp, 3),
        "coverage": round(coverage, 3),
        "n_events": total_events,
        "countries": countries,
    }
