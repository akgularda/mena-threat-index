"""Index forecasting: a data-driven selector over simple models, defaulting to
persistence, with conformal (empirical) widening predictive bands.

Why this shape (validated by `scripts/forecast_backtest.py` over BNTI + MENA score
series): for these short, noisy, random-walk-dominated, regime-shifting 1-10 indices
**persistence (last value) is the most reliable point forecast**, and global
mean-reversion (the former AR(1)+drift) is the *worst* — it over-predicts because it
reverts to a stale global average. So we anchor at the last observed value, only adopt
a trend model when it demonstrably beats persistence on the recent record (hysteresis),
and put the intelligence into honest, empirically-calibrated, widening bands.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from .util import clip, to_iso, utcnow


# ---- candidate point models: f(values, H, phi=...) -> [f_1..f_H], anchored, numpy ----
def _persistence(y, H, **_):
    return [y[-1]] * H


def _ses(y, H, alpha=0.5, **_):
    L = y[0]
    for v in y[1:]:
        L = alpha * v + (1 - alpha) * L
    return [L] * H


def _holt_damped(y, H, alpha=0.5, beta=0.1, phi=0.85, **_):
    if len(y) < 2:
        return [y[-1]] * H
    L, T = y[0], y[1] - y[0]
    for v in y[1:]:
        Lp = L
        L = alpha * v + (1 - alpha) * (L + phi * T)
        T = beta * (L - Lp) + (1 - beta) * phi * T
    return [L + sum(phi ** i for i in range(1, h + 1)) * T for h in range(1, H + 1)]


def _theta(y, H, alpha=0.5, **_):
    n = len(y)
    if n < 3:
        return _ses(y, H, alpha)
    b = float(np.polyfit(np.arange(n), y, 1)[0])          # long-term (OLS) slope
    L = y[0]
    for v in y[1:]:
        L = alpha * v + (1 - alpha) * L
    return [L + 0.5 * b * h for h in range(1, H + 1)]      # SES level + half-trend


def _damped_drift(y, H, phi=0.85, k_recent=4, **_):
    if len(y) < 2:
        return [y[-1]] * H
    d = float(np.mean(np.diff(y[-(k_recent + 1):])))
    return [y[-1] + sum(phi ** i for i in range(1, h + 1)) * d for h in range(1, H + 1)]


MODELS = {
    "persistence": _persistence,
    "ses": _ses,
    "holt_damped": _holt_damped,
    "theta": _theta,
    "damped_drift": _damped_drift,
}


# ---- rolling-origin evaluation + data-driven model selection ----
def _rolling_origin(values, phi=0.85, min_train=5):
    """One-step out-of-sample abs errors per model on ONE series.
    Returns {model: {"mae": float, "n": int}}."""
    err = {m: [] for m in MODELS}
    n = len(values)
    for t in range(min_train, n):
        train = values[:t]
        for m, fn in MODELS.items():
            err[m].append(abs(fn(train, 1, phi=phi)[0] - values[t]))
    return {m: {"mae": (sum(e) / len(e) if e else math.inf), "n": len(e)}
            for m, e in err.items()}


def select_model(values, cfg):
    """Pick the model with the best recent 1-step skill, but only leave persistence
    when a challenger beats it by >= select_margin over >= select_min_folds folds and
    we have >= select_min_points history. Returns (model_name, skill_vs_persistence)."""
    fc = cfg.settings.get("forecast", {})
    min_pts = int(fc.get("select_min_points", 16))
    min_folds = int(fc.get("select_min_folds", 8))
    margin = float(fc.get("select_margin", 0.05))
    phi = float(fc.get("drift_phi", 0.85))
    if len(values) < min_pts:
        return "persistence", 0.0
    res = _rolling_origin(values, phi=phi)
    base = res["persistence"]["mae"]
    if not math.isfinite(base) or base <= 0 or res["persistence"]["n"] < min_folds:
        return "persistence", 0.0
    best, best_skill = "persistence", 0.0
    for m, r in res.items():
        if m == "persistence" or not math.isfinite(r["mae"]):
            continue
        skill = 1.0 - r["mae"] / base
        if skill > best_skill:
            best, best_skill = m, skill
    return (best, best_skill) if best != "persistence" and best_skill >= margin else ("persistence", 0.0)


def _band_halfwidths(values, model_name, n_steps, cfg, sigma_prior, phi):
    """Conformal/empirical half-width per horizon: the band_quantile of recent
    |h-step residuals| of the selected model on this series; fall back to
    z*sigma*sqrt(h) when too few residuals. Floored at band_floor."""
    fc = cfg.settings.get("forecast", {})
    q = float(fc.get("band_quantile", 0.80))
    floor = float(fc.get("band_floor", 0.15))
    z = float(fc.get("band_z", 1.2815))
    fn = MODELS[model_name]
    resid = {h: [] for h in range(1, n_steps + 1)}
    n = len(values)
    for t in range(2, n):                                  # train = values[:t]
        f = fn(values[:t], n_steps, phi=phi)
        for h in range(1, n_steps + 1):
            idx = t - 1 + h
            if idx >= n:
                break
            resid[h].append(abs(f[h - 1] - values[idx]))
    widths = []
    for h in range(1, n_steps + 1):
        rs = resid[h]
        hw = float(np.quantile(rs, q)) if len(rs) >= 4 else z * sigma_prior * math.sqrt(h)
        widths.append(max(hw, floor))
    return widths


def forecast(hist, cfg, conf_comp: float, seed_sigma: float | None, log=None):
    fc = cfg.settings.get("forecast", {})
    horizon = int(fc.get("horizon_hours", 24))
    step = int(fc.get("step_hours", 2))
    R = float(fc.get("precision_range", 3.0))
    phi = float(fc.get("drift_phi", 0.85))
    cal_window = int(fc.get("calibration_window", 60))
    q = float(fc.get("band_quantile", 0.80))
    n_steps = max(1, horizon // step)

    # Use only REAL (non-seed) points — the BNTI seed is a different regime and
    # biases any fit (METHODOLOGY_REVIEW: forecast was seed-contaminated).
    rows = [r for r in hist if r.get("index") is not None and not r.get("seed")
            and r.get("_dt") is not None]
    if not rows:                                           # cold start: nothing real yet
        rows = [r for r in hist if r.get("index") is not None and r.get("_dt") is not None]
    rows.sort(key=lambda r: r["_dt"])
    rows = rows[-cal_window:]
    values = [float(r["index"]) for r in rows]
    n = len(values)
    last_dt = rows[-1]["_dt"] if rows else utcnow()
    S_t = values[-1] if values else 1.0

    # fallback sigma for sparse horizons (seed diff std, else recent diff std, else 0.6)
    if seed_sigma and seed_sigma > 0:
        sigma_prior = float(seed_sigma)
    elif n >= 3:
        sigma_prior = float(np.std(np.diff(values[-12:]))) or 0.5
    else:
        sigma_prior = 0.6

    model_name, skill = select_model(values, cfg)
    means = MODELS[model_name](values, n_steps, phi=phi) if values else [S_t] * n_steps
    widths = _band_halfwidths(values, model_name, n_steps, cfg, sigma_prior, phi)

    points = []
    for k in range(1, n_steps + 1):
        ts = last_dt + timedelta(hours=step * k)
        mean = clip(means[k - 1], 1.0, 10.0)
        hw = widths[k - 1]
        precision = clip(1.0 - hw / R, 0.0, 1.0)
        conf = clip(math.sqrt(max(precision, 1e-4) * max(conf_comp, 1e-4)), 0.0, 0.99)
        points.append({"timestamp": to_iso(ts), "main_index": round(mean, 2),
                       "confidence": round(conf, 3),
                       "low": round(clip(mean - hw, 1.0, 10.0), 2),
                       "high": round(clip(mean + hw, 1.0, 10.0), 2)})

    meta = {"method": f"auto:{model_name}", "selected_model": model_name,
            "recent_skill": round(float(skill), 3), "horizon_hours": horizon,
            "band": f"{int(q * 100)}% predictive interval", "n_points": n}
    if log:
        log.info("forecast: model=%s skill=%.2f n=%d next=%.2f..%.2f",
                 model_name, skill, n, points[0]["low"], points[0]["high"])
    return points, meta
