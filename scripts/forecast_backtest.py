"""Forecast model bake-off on the project's own recorded score series.

Standalone — NOT part of the pipeline hot path. Runs a rolling-origin (time-series
cross-validation) backtest of the candidate point models in `pipeline.forecast.MODELS`,
plus the legacy AR(1)-global model as a reference, pooled across the composite and
every per-country series in `data/`. Reports 1-step MAE, skill vs persistence, and
MASE (Hyndman & Koehler 2006) — the evidence behind the default model.

Run:  python -m scripts.forecast_backtest
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from pipeline import config as cfgmod
from pipeline.forecast import MODELS
from pipeline.util import ROOT

DATA = os.path.join(ROOT, "data")


def _series_from_jsonl(path, seed_ok=True):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows = [r for r in rows if r.get("index") is not None and (seed_ok or not r.get("seed"))]
    rows.sort(key=lambda r: r.get("ts", ""))
    return [float(r["index"]) for r in rows]


def load_series(min_len=6):
    """Composite (non-seed) + every per-country series with enough points."""
    series = []
    comp = _series_from_jsonl(os.path.join(DATA, "history.jsonl"), seed_ok=False)
    if len(comp) >= min_len:
        series.append(("composite", comp))
    for fp in sorted(glob.glob(os.path.join(DATA, "countries", "*.jsonl"))):
        vals = _series_from_jsonl(fp)
        if len(vals) >= min_len:
            series.append((os.path.basename(fp)[:-6].replace("_", " "), vals))
    return series


def _ar1_global(y, H, **_):
    """Legacy model (removed from MODELS): mean-revert to the global AR(1) mean."""
    if len(y) < 3 or np.std(y[:-1]) < 1e-9:
        return [y[-1]] * H
    x, z = np.array(y[:-1]), np.array(y[1:])
    phi, c = np.polyfit(x, z, 1)
    phi = min(max(phi, 0.0), 0.995)
    mu = c / (1 - phi) if abs(1 - phi) > 1e-6 else y[-1]
    return [mu + (phi ** h) * (y[-1] - mu) for h in range(1, H + 1)]


def backtest(series, phi=0.85, min_train=5):
    models = dict(MODELS)
    models["ar1_global(legacy)"] = _ar1_global
    err = {m: [] for m in models}
    mase = {m: [] for m in models}
    for _, y in series:
        n = len(y)
        scale = float(np.mean(np.abs(np.diff(y)))) or None       # in-sample naive MAE
        for t in range(min_train, n):
            train = y[:t]
            for m, fn in models.items():
                e = abs(fn(train, 1, phi=phi)[0] - y[t])
                err[m].append(e)
                if scale:
                    mase[m].append(e / scale)
    base = sum(err["persistence"]) or 1e-9
    rows = []
    for m in models:
        es = err[m]
        mae = sum(es) / max(1, len(es))
        skill = 1.0 - sum(es) / base
        ms = (sum(mase[m]) / len(mase[m])) if mase[m] else float("nan")
        rows.append((m, mae, skill, ms))
    return sorted(rows, key=lambda r: r[1]), len(err["persistence"])


def main():
    cfgmod.load()                                            # sanity: config loads
    series = load_series()
    if not series:
        print("no recorded series yet (need data/history.jsonl or data/countries/*.jsonl)")
        return
    rows, n1 = backtest(series)
    print(f"forecast bake-off — {len(series)} series, {n1} one-step evaluations\n")
    print(f"{'model':22} {'MAE':>7} {'skill_vs_naive':>15} {'MASE':>7}")
    for m, mae, sk, ms in rows:
        print(f"{m:22} {mae:7.3f} {sk * 100:+14.1f}% {ms:7.3f}")
    print(f"\nbest (lowest MAE): {rows[0][0]}")


if __name__ == "__main__":
    main()
