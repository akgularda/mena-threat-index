"""Index forecasting: AR(1)-with-drift, with a cold-start ladder for thin series.

Returns forward points with widening predictive bands and a per-horizon
confidence in [0,1]. Honest by construction: mean-reversion + damping flatten the
long horizon rather than extrapolating a trend to absurdity.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from .util import clip, to_iso, utcnow


def _ar1(values):
    """Fit S_t = c + phi*S_{t-1}; return (phi_hat, c, sigma2, n)."""
    x = np.array(values[:-1], dtype=float)
    y = np.array(values[1:], dtype=float)
    n = len(x)
    if n < 2 or np.std(x) < 1e-9:
        return 0.0, float(np.mean(values)), float(np.var(values)) or 0.25, n
    phi, c = np.polyfit(x, y, 1)
    resid = y - (c + phi * x)
    dof = max(1, n - 2)
    sigma2 = float((resid ** 2).sum() / dof)
    return float(phi), float(c), max(sigma2, 1e-4), n


def forecast(hist, cfg, conf_comp: float, seed_sigma: float | None, log=None):
    fc = cfg.settings.get("forecast", {})
    horizon = int(fc.get("horizon_hours", 24))
    step = int(fc.get("step_hours", 2))
    z = float(fc.get("band_z", 1.2815))
    R = float(fc.get("precision_range", 3.0))
    phi_prior = float(fc.get("phi_prior", 0.90))
    phi_k = float(fc.get("phi_shrink_k", 10))
    min_ar1 = int(fc.get("min_points_ar1", 12))
    min_holt = int(fc.get("min_points_holt", 5))
    holt_phi = float(fc.get("holt_phi", 0.80))

    rows = [r for r in hist if r.get("index") is not None]
    values = [float(r["index"]) for r in rows]
    n = len(values)
    last_dt = rows[-1]["_dt"] if rows and rows[-1].get("_dt") else utcnow()
    S_t = values[-1] if values else 1.0
    n_steps = max(1, horizon // step)

    # fallback sigma for cold start: BNTI seed diff std, else recent diff std, else 0.6
    if seed_sigma and seed_sigma > 0:
        sigma_prior = float(seed_sigma)
    elif n >= 3:
        diffs = np.diff(values[-12:])
        sigma_prior = float(np.std(diffs)) or 0.5
    else:
        sigma_prior = 0.6

    points = []

    def emit(k, mean, halfwidth):
        ts = last_dt + timedelta(hours=step * k)
        low = clip(mean - halfwidth, 1.0, 10.0)
        high = clip(mean + halfwidth, 1.0, 10.0)
        precision = clip(1.0 - halfwidth / R, 0.0, 1.0)
        conf = clip(math.sqrt(max(precision, 1e-4) * max(conf_comp, 1e-4)), 0.0, 0.99)
        points.append({"timestamp": to_iso(ts), "main_index": round(clip(mean, 1.0, 10.0), 2),
                       "confidence": round(conf, 3), "low": round(low, 2), "high": round(high, 2)})

    if n >= min_ar1:
        method = "AR(1)+drift"
        phi_hat, c, sigma2, nfit = _ar1(values)
        phi = clip((nfit * phi_hat + phi_k * phi_prior) / (nfit + phi_k), 0.0, 0.995)
        mu = c / (1.0 - phi) if abs(1.0 - phi) > 1e-6 else S_t
        sigma2_ = sigma2
        for k in range(1, n_steps + 1):
            mean = mu + (phi ** k) * (S_t - mu)
            if phi >= 0.999:
                var = sigma2_ * k
            else:
                var = sigma2_ * (1.0 - phi ** (2 * k)) / (1.0 - phi ** 2)
            emit(k, mean, z * math.sqrt(max(var, 1e-6)))
    elif n >= min_holt:
        method = "damped Holt"
        # simple Holt init
        level = values[-1]
        trend = float(np.mean(np.diff(values[-min_holt:]))) if n >= 2 else 0.0
        a, b = 0.4, 0.2
        L, T = values[0], (values[1] - values[0]) if n >= 2 else 0.0
        resid = []
        for v in values[1:]:
            f = L + holt_phi * T
            resid.append(v - f)
            Ln = a * v + (1 - a) * (L + holt_phi * T)
            T = b * (Ln - L) + (1 - b) * holt_phi * T
            L = Ln
        sigma = float(np.std(resid)) if resid else sigma_prior
        for k in range(1, n_steps + 1):
            damp = sum(holt_phi ** i for i in range(1, k + 1))
            mean = L + T * damp
            emit(k, mean, z * sigma * math.sqrt(k))
    else:
        method = "persistence" + (f" (cold start, n={n})" if n else " (no history)")
        for k in range(1, n_steps + 1):
            emit(k, S_t, z * sigma_prior * math.sqrt(k / 3.0))

    meta = {"method": method, "horizon_hours": horizon, "band": "80% predictive interval",
            "n_points": n}
    if log:
        log.info("forecast: method=%s n=%d next=%.2f..%.2f",
                 method, n, points[0]["low"], points[0]["high"])
    return points, meta
