"""Market reaction + forecasting.

For each instrument: fetch a daily price series (Yahoo v8 chart, FRED fallback —
both keyless), align the (intraday) threat index to the instrument's own trading
days via an as-of join, estimate lagged correlation and an OLS "threat beta"
(return per +1 index point), then project expected moves from the forecasted
index path. Everything is gated by minimum-sample / significance rules and is
clearly caveated. Launch-day numbers lean on the BNTI-seeded history.

This module never raises into the pipeline: on any failure it returns an empty
instrument set with a note, and the rest of the run still publishes.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone

from .config import Config
from .util import clip, http_get, safe_float, session, to_iso, utcnow


# ---------- small stats helpers (numpy-free, robust on tiny N) ----------

def _phi(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return 0.0, 1.0
    r = sxy / math.sqrt(sxx * syy)
    r = clip(r, -0.999999, 0.999999)
    # Fisher-z two-sided p
    if n > 3:
        z = math.atanh(r) * math.sqrt(n - 3)
        p = 2.0 * (1.0 - _phi(abs(z)))
    else:
        p = 1.0
    return r, p


def _ols(xs, ys):
    """Return (beta, alpha, r2, se_beta)."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0, float("inf")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0, float("inf")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    beta = sxy / sxx
    alpha = my - beta * mx
    resid = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    sse = sum(e * e for e in resid)
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sse / syy if syy > 0 else 0.0
    dof = max(1, n - 2)
    se_beta = math.sqrt((sse / dof) / sxx) if sxx > 0 else float("inf")
    return beta, alpha, r2, se_beta


def _bh_significant(pvals, q):
    """Benjamini-Hochberg: return a set of indices that pass at FDR q."""
    m = len(pvals)
    if m == 0:
        return set()
    order = sorted(range(m), key=lambda i: pvals[i])
    passed = set()
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            passed = set(order[:rank])
    return passed


# ---------- price fetching ----------

def _fetch_yahoo(sess, symbol, log):
    for host in ("query1", "query2"):
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?range=1y&interval=1d")
        try:
            r = http_get(sess, url)
            if r.status_code != 200:
                continue
            j = r.json()
            res = (j.get("chart", {}).get("result") or [None])[0]
            if not res:
                continue
            ts = res.get("timestamp") or []
            closes = ((res.get("indicators", {}).get("quote") or [{}])[0]).get("close") or []
            out = {}
            for t, c in zip(ts, closes):
                if c is None:
                    continue
                d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
                out[d] = float(c)
            if out:
                return out
        except Exception as e:
            log.debug("yahoo %s via %s failed: %s", symbol, host, e)
    return {}


def _fetch_fred(sess, symbol, log):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}"
    try:
        r = http_get(sess, url)
        if r.status_code != 200:
            return {}
        rows = list(csv.reader(io.StringIO(r.text)))
        if not rows or len(rows) < 2:
            return {}
        out = {}
        for row in rows[1:]:
            if len(row) < 2:
                continue
            d, v = row[0].strip(), row[1].strip()
            if v in (".", "", "NaN"):
                continue
            try:
                out[d] = float(v)
            except Exception:
                continue
        return out
    except Exception as e:
        log.debug("fred %s failed: %s", symbol, e)
        return {}


def _fetch_prices(sess, inst, log):
    fn = _fetch_yahoo if inst.source == "yahoo" else _fetch_fred
    prices = fn(sess, inst.symbol, log)
    if not prices and inst.fallback_symbol:
        fb = _fetch_yahoo if inst.fallback_source == "yahoo" else _fetch_fred
        prices = fb(sess, inst.fallback_symbol, log)
    return prices


# ---------- index daily series ----------

def _index_daily(index_hist):
    """Last index reading per UTC calendar day -> {date: (value, is_seed)}."""
    daily = {}
    for r in index_hist:
        dt = r.get("_dt")
        if dt is None or r.get("index") is None:
            continue
        d = dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        daily[d] = (float(r["index"]), bool(r.get("seed", False)))
    return daily


# ---------- per-instrument analysis ----------

def _analyse(inst, prices, idx_daily, cfg, index_forecast, current_index, log):
    mk = cfg.settings.get("markets", {})
    lag_set = mk.get("lag_set", [-3, -2, -1, 0, 1, 2, 3])
    fdays = mk.get("forecast_days", [1, 3, 5])
    min_obs = int(mk.get("min_paired_obs", 20))
    min_r = float(mk.get("min_abs_corr", 0.30))
    beta_t_min = float(mk.get("beta_t_min", 1.5))
    inflate = float(mk.get("model_risk_inflation", 1.5))
    z = 1.2815

    pdates = sorted(prices.keys())
    if len(pdates) < 5:
        return None
    last = prices[pdates[-1]]
    prev = prices[pdates[-2]]
    change_pct = (last / prev - 1.0) * 100.0 if prev else 0.0

    # instrument log returns on its own trading days
    rets = {}
    for i in range(1, len(pdates)):
        p0, p1 = prices[pdates[i - 1]], prices[pdates[i]]
        if p0 > 0 and p1 > 0:
            rets[pdates[i]] = math.log(p1 / p0)

    # daily index deltas across available index days
    idates = sorted(idx_daily.keys())
    didx = {}
    native = {}
    for i in range(1, len(idates)):
        d = idates[i]
        didx[d] = idx_daily[d][0] - idx_daily[idates[i - 1]][0]
        native[d] = not idx_daily[d][1]

    def aligned(lag):
        xs, ys, nnat = [], [], 0
        for d in rets:
            # market day d reacts to index change `lag` days earlier
            di = idates_index_for(d, lag)
            if di is None:
                continue
            xs.append(didx[di]); ys.append(rets[d])
            if native.get(di):
                nnat += 1
        return xs, ys, nnat

    # precompute date->position for index days
    ipos = {d: i for i, d in enumerate(idates)}

    def idates_index_for(market_date, lag):
        # find the index day == market_date shifted back by `lag` days (by position
        # in the index-day sequence, approximating trading-day lag)
        if market_date in ipos:
            j = ipos[market_date] - lag
        else:
            # nearest prior index day
            j = None
            for k in range(len(idates) - 1, -1, -1):
                if idates[k] <= market_date:
                    j = k - lag
                    break
            if j is None:
                return None
        if j is None or j < 1 or j >= len(idates):
            return None
        return idates[j]

    best = None
    for lag in lag_set:
        xs, ys, nnat = aligned(lag)
        if len(xs) < 5:
            continue
        r, p = _pearson(xs, ys)
        if best is None or abs(r) > abs(best["r"]):
            best = {"lag": lag, "r": r, "p": p, "xs": xs, "ys": ys, "n": len(xs), "nnat": nnat}
    if best is None:
        return None

    beta, alpha, r2, se_beta = _ols(best["xs"], best["ys"])
    n_obs = best["n"]
    n_native = best["nnat"]
    source = "mena" if n_native >= min_obs else ("blended" if n_native >= 5 else "bnti_seeded")

    # rolling correlation sign stability (window over the aligned series)
    win = int(mk.get("rolling_window", 25))
    xs, ys = best["xs"], best["ys"]
    signs = []
    for i in range(len(xs)):
        if i + 1 >= win:
            rr, _ = _pearson(xs[i + 1 - win:i + 1], ys[i + 1 - win:i + 1])
            signs.append(1 if rr >= 0 else -1)
    sign_stability = (max(signs.count(1), signs.count(-1)) / len(signs)) if signs else 0.0

    # gates
    passes = (n_native >= min_obs and abs(best["r"]) >= min_r and
              (se_beta > 0 and abs(beta) / se_beta >= beta_t_min) and
              sign_stability >= float(mk.get("sign_stability_min", 0.60)))

    # ---- market forecast from index forecast path ----
    # daily expected index change over next ~24h:
    fc_vals = [p["main_index"] for p in index_forecast] if index_forecast else []
    daily_dindex = (fc_vals[-1] - current_index) if fc_vals else 0.0
    forecasts = {}
    resid_sigma = math.sqrt(max(0.0, (1 - r2))) * (abs(beta) * abs(daily_dindex) + 0.001)
    for h in fdays:
        damp = sum(0.8 ** (i - 1) for i in range(1, h + 1))  # damped accumulation
        cum = daily_dindex * damp
        e_ret = beta * cum
        move = (math.exp(e_ret) - 1.0) * 100.0
        var = (cum ** 2) * (se_beta ** 2) + h * (resid_sigma ** 2)
        hw = z * math.sqrt(max(var, 1e-8)) * inflate * 100.0
        forecasts[f"{h}d"] = {"expected_move_pct": round(move, 2),
                              "low_pct": round(move - hw, 2), "high_pct": round(move + hw, 2)}

    direction = "positive" if beta > 0 else "negative" if beta < 0 else "flat"

    # scenarios
    scen = []
    for delta, lbl in ((1.0, "Index +1.0"), (-1.0, "Index −1.0")):
        scen.append({"label": lbl, "index_delta": delta,
                     "expected_move_pct": round((math.exp(beta * delta) - 1.0) * 100.0, 2)})

    # confidence
    conf = clip((max(r2, 1e-4) * abs(best["r"]) * (1.0 if passes else 0.4)) ** (1 / 3), 0.0, 0.95)

    # sparkline history (last 60 closes)
    hist = [{"timestamp": d, "value": round(prices[d], 4)} for d in pdates[-60:]]

    fc_primary = forecasts.get("1d", forecasts[list(forecasts)[0]])
    return {
        "id": inst.id, "name": inst.name, "symbol": inst.symbol,
        "asset_class": inst.asset_class, "currency": inst.currency,
        "last": round(last, 4), "change_pct": round(change_pct, 2),
        "correlation": round(best["r"], 3), "p_value": round(best["p"], 4),
        "beta": round(beta, 4), "beta_se": round(se_beta, 4) if se_beta != float("inf") else None,
        "r2": round(r2, 3), "best_lag_days": best["lag"],
        "n_obs": n_obs, "n_native": n_native,
        "rolling_sign_stability": round(sign_stability, 2),
        "direction": direction, "significant": bool(passes), "source": source,
        "forecast": {"horizon_days": 1, "expected_move_pct": fc_primary["expected_move_pct"],
                     "low_pct": fc_primary["low_pct"], "high_pct": fc_primary["high_pct"],
                     "confidence": round(conf, 3), "by_horizon": forecasts},
        "scenarios": scen, "history": hist,
        "caveat_flags": ([] if passes else ["insufficient_native_data"]) +
                        (["pegged_fx"] if inst.pegged else []),
    }


def build(cfg: Config, index_hist, index_forecast, current_index, log):
    mk = cfg.settings.get("markets", {})
    try:
        sess = session(cfg.settings.get("ingest", {}).get("user_agent"), 20)
        idx_daily = _index_daily(index_hist)
        instruments = []
        for inst in cfg.instruments:
            prices = _fetch_prices(sess, inst, log)
            if not prices:
                log.warning("markets: no prices for %s (%s)", inst.id, inst.symbol)
                continue
            row = _analyse(inst, prices, idx_daily, cfg, index_forecast, current_index, log)
            if row:
                instruments.append(row)

        # multiple-comparison control across instruments' best-lag correlations
        if instruments:
            pvals = [r["p_value"] for r in instruments]
            passed = _bh_significant(pvals, float(mk.get("fdr_q", 0.10)))
            for i, r in enumerate(instruments):
                fdr_ok = i in passed
                r["significant"] = bool(r["significant"] and fdr_ok)
                if not fdr_ok and "fdr_reject" not in r["caveat_flags"]:
                    r["caveat_flags"].append("fdr_reject")

        regime, headline = _summary(instruments, current_index, index_forecast)
        log.info("markets: %d instruments (%d significant)",
                 len(instruments), sum(1 for r in instruments if r["significant"]))
        return {
            "generated_at": to_iso(utcnow()),
            "correlation_window_days": int(mk.get("correlation_window_days", 120)),
            "regime": regime, "headline": headline,
            "method_note": ("Threat beta = OLS of daily log-returns on the index change at "
                            "the lead lag; launch-day estimates lean on BNTI-seeded history. "
                            "Association, not causation."),
            "instruments": instruments,
        }
    except Exception as e:  # never break the pipeline on markets
        log.warning("markets module failed (non-fatal): %s", e)
        return {"generated_at": to_iso(utcnow()), "instruments": [],
                "regime": "unknown", "headline": "Market data unavailable this run.",
                "method_note": "Market module error; index still published."}


def _summary(instruments, current_index, index_forecast):
    if not instruments:
        return "unknown", "Market data unavailable this run."
    by = {r["id"]: r for r in instruments}
    fc_vals = [p["main_index"] for p in index_forecast] if index_forecast else []
    rising = bool(fc_vals and fc_vals[-1] > current_index)
    brent = by.get("brent")
    gold = by.get("gold")
    vix = by.get("vix")
    parts = []
    if rising:
        regime = "risk-off"
        parts.append("Index forecast rising")
    elif fc_vals and fc_vals[-1] < current_index:
        regime = "risk-on"
        parts.append("Index forecast easing")
    else:
        regime = "neutral"
        parts.append("Index forecast flat")
    if brent and brent.get("forecast"):
        parts.append(f"Brent {brent['forecast']['expected_move_pct']:+.1f}%")
    if gold and gold.get("forecast"):
        parts.append(f"gold {gold['forecast']['expected_move_pct']:+.1f}%")
    if vix and vix.get("forecast"):
        parts.append(f"VIX {vix['forecast']['expected_move_pct']:+.1f}%")
    return regime, "; ".join(parts) + " (model estimate)."
