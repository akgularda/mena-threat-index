"""Append-only JSONL persistence for the index, per-country, and market series.

`data/history.jsonl` is the source of truth and grows over time. Each run appends
one composite line (idempotent on run_id), per-country lines, and a markets line.
The published mena_data.json carries only a trimmed slice.
"""
from __future__ import annotations

import json
import os

from .util import ROOT, parse_dt, to_iso, utcnow

DATA = os.path.join(ROOT, "data")
HISTORY = os.path.join(DATA, "history.jsonl")
COUNTRIES_DIR = os.path.join(DATA, "countries")
MARKETS = os.path.join(DATA, "markets", "instruments.jsonl")
FORECAST_EVAL = os.path.join(DATA, "forecast_eval.jsonl")
SEED = os.path.join(DATA, "seed", "bnti_history_seed.json")


def _country_path(name: str) -> str:
    safe = name.replace(" ", "_").replace("/", "_")
    return os.path.join(COUNTRIES_DIR, f"{safe}.jsonl")


def _read_jsonl(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _append_idempotent(path: str, row: dict, key: str = "run_id") -> None:
    rows = _read_jsonl(path)
    rid = row.get(key)
    if rid is not None:
        rows = [r for r in rows if r.get(key) != rid]
    rows.append(row)
    rows.sort(key=lambda r: r.get("ts", ""))
    _write_jsonl(path, rows)


def ensure_seeded(log=None) -> int:
    """If the composite history is empty and a BNTI seed exists, load it."""
    if _read_jsonl(HISTORY):
        return 0
    if not os.path.exists(SEED):
        return 0
    with open(SEED, "r", encoding="utf-8") as f:
        seed = json.load(f)
    rows = []
    for p in seed.get("history", []):
        rows.append({"run_id": "seed-" + p["timestamp"], "ts": p["timestamp"],
                     "index": round(float(p["index"]), 3), "raw": None,
                     "status": None, "confidence": None, "n_events": None,
                     "forecast_next": None, "seed": True})
    rows.sort(key=lambda r: r["ts"])
    _write_jsonl(HISTORY, rows)
    if log:
        log.info("history: seeded %d points from BNTI", len(rows))
    return len(rows)


def read_composite() -> list:
    rows = _read_jsonl(HISTORY)
    for r in rows:
        r["_dt"] = parse_dt(r.get("ts"))
    rows = [r for r in rows if r["_dt"] is not None]
    rows.sort(key=lambda r: r["_dt"])
    return rows


def read_country(name: str) -> list:
    rows = _read_jsonl(_country_path(name))
    for r in rows:
        r["_dt"] = parse_dt(r.get("ts"))
    rows = [r for r in rows if r["_dt"] is not None]
    rows.sort(key=lambda r: r["_dt"])
    return rows


def prev_composite():
    rows = [r for r in read_composite() if not r.get("seed")]
    if not rows:
        # fall back to seed value if that's all we have
        allrows = read_composite()
        return float(allrows[-1]["index"]) if allrows else None
    return float(rows[-1]["index"])


def recent_diffs(n: int = 12) -> list:
    rows = read_composite()
    vals = [float(r["index"]) for r in rows[-(n + 1):]]
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def prev_forecast_next():
    """Most recent non-seed run's stored 1-step forecast (the prediction whose
    target is the current run), for realized-accuracy logging."""
    for r in reversed([r for r in read_composite() if not r.get("seed")]):
        fn = r.get("forecast_next")
        if fn is not None:
            return float(fn)
    return None


def prev_index_map() -> dict:
    out = {}
    if not os.path.isdir(COUNTRIES_DIR):
        return out
    for fn in os.listdir(COUNTRIES_DIR):
        if not fn.endswith(".jsonl"):
            continue
        rows = _read_jsonl(os.path.join(COUNTRIES_DIR, fn))
        rows = [r for r in rows if r.get("index") is not None]
        if rows:
            name = rows[-1].get("name") or fn[:-6].replace("_", " ")
            out[name] = float(rows[-1]["index"])
    return out


def priors(cfg, log=None) -> dict:
    """Per-country trailing-mean raw score for empirical-Bayes shrinkage."""
    days = float(cfg.settings.get("score", {}).get("baseline_window_days", 14))
    now = utcnow()
    out = {}
    for c in cfg.countries:
        rows = read_country(c.name)
        vals = [float(r["raw"]) for r in rows
                if r.get("raw") is not None and (now - r["_dt"]).total_seconds() / 86400.0 <= days]
        if vals:
            out[c.name] = sum(vals) / len(vals)
    return out


def append_run(run_id, ts_iso, result, forecast_next, market_rows, log=None) -> None:
    """Append the composite reading, per-country lines, and market snapshot."""
    _append_idempotent(HISTORY, {
        "run_id": run_id, "ts": ts_iso,
        "index": result["composite"], "raw": result["composite_raw"],
        "status": result["status"], "confidence": result["confidence"],
        "n_events": result["n_events"],
        "forecast_next": forecast_next, "seed": False,
    })
    for name, co in result["countries"].items():
        _append_idempotent(_country_path(name), {
            "run_id": run_id, "ts": ts_iso, "name": name,
            "index": co["index"], "raw": co["raw_score"],
            "confidence": co["confidence"], "n_events": co["n_events"],
        })
    for m in (market_rows or []):
        row = {"run_id": run_id, "ts": ts_iso, "id": m.get("id"),
               "last": m.get("last"), "change_pct": m.get("change_pct"),
               "correlation": m.get("correlation"), "beta": m.get("beta"),
               "r2": m.get("r2")}
        _append_idempotent(MARKETS, row)
    if log:
        log.info("history: appended run %s", run_id)


def append_forecast_eval(run_id, ts_iso, actual, model_pred, naive_pred, log=None) -> None:
    """Record the realized 1-step error of the previous run's forecast vs the new
    actual, alongside the naive (persistence) error, so forecast skill is monitored
    over time. Idempotent on run_id. Read with recent_forecast_skill()."""
    em = abs(float(model_pred) - float(actual))
    en = abs(float(naive_pred) - float(actual))
    _append_idempotent(FORECAST_EVAL, {
        "run_id": run_id, "ts": ts_iso, "actual": round(float(actual), 3),
        "model_pred": round(float(model_pred), 3), "naive_pred": round(float(naive_pred), 3),
        "err_model": round(em, 3), "err_naive": round(en, 3),
    })
    if log:
        log.info("forecast eval: err_model=%.3f err_naive=%.3f (skill %+.0f%%)",
                 em, en, (1 - em / en) * 100 if en > 0 else 0.0)


def recent_forecast_skill(n: int = 20) -> dict | None:
    """Pooled realized skill of the last n logged forecasts vs naive persistence."""
    rows = _read_jsonl(FORECAST_EVAL)
    if not rows:
        return None
    rows = rows[-n:]
    sm = sum(float(r.get("err_model", 0.0)) for r in rows)
    sn = sum(float(r.get("err_naive", 0.0)) for r in rows)
    return {"n": len(rows), "mae_model": round(sm / len(rows), 3),
            "mae_naive": round(sn / len(rows), 3),
            "skill": round(1.0 - sm / sn, 3) if sn > 0 else 0.0}


def published_composite(days: int) -> list:
    now = utcnow()
    rows = read_composite()
    cut = [r for r in rows if (now - r["_dt"]).total_seconds() / 86400.0 <= days]
    if not cut:
        cut = rows
    return [{"timestamp": to_iso(r["_dt"]), "main_index": round(float(r["index"]), 2),
             "seed": bool(r.get("seed", False))} for r in cut]


def published_country(name: str, days: int) -> list:
    now = utcnow()
    rows = read_country(name)
    cut = [r for r in rows if (now - r["_dt"]).total_seconds() / 86400.0 <= days]
    return [{"timestamp": to_iso(r["_dt"]), "index": round(float(r["index"]), 2)}
            for r in cut]
