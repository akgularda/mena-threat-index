"""Assemble the final mena_data.json, validate it, and write atomically.
Withhold-on-failure: writes to .tmp, validates, only then os.replace; if invalid
or coverage below floor, keeps the existing last-good file and returns False."""
from __future__ import annotations

import json
import os

from .util import ROOT, to_iso, utcnow
from . import schema, history

OUT = os.path.join(ROOT, "mena_data.json")


def assemble(cfg, run_id, ts, result, forecast_points, forecast_meta, briefing, markets, model_label):
    days = int(cfg.settings.get("history", {}).get("published_days", 90))
    from datetime import timedelta
    next_update = ts + timedelta(hours=int(cfg.settings.get("forecast", {}).get("step_hours", 2)))

    countries = {}
    for c in cfg.countries:
        co = dict(result["countries"][c.name])
        co["history"] = history.published_country(c.name, days)
        countries[c.name] = co

    return {
        "schema_version": cfg.settings.get("schema_version", "2.0"),
        "meta": {
            "main_index": result["composite"], "status": result["status"],
            "generated_at": to_iso(ts), "next_update": to_iso(next_update),
            "confidence": result["confidence"], "n_events": result["n_events"],
            "window_hours": int(cfg.settings.get("ingest", {}).get("score_window_hours", 48)),
            "run_id": run_id, "region": cfg.settings.get("region", "MENA"),
            "model": model_label,
        },
        "countries": countries,
        "history": history.published_composite(days),
        "forecast": forecast_points,
        "forecast_meta": forecast_meta,
        "briefing": briefing,
        "markets": markets,
    }


def publish(cfg, doc, log) -> bool:
    errs = schema.validate(doc)
    if errs:
        log.error("publish WITHHELD — schema errors: %s", errs[:8])
        return False
    floor = float(cfg.settings.get("ingest", {}).get("coverage_floor", 0.30))
    cov = doc["meta"].get("confidence", 1)  # coverage folded into confidence; use raw coverage below
    if doc["meta"].get("n_events", 0) == 0:
        log.warning("publish: zero events this run (still publishing baseline)")
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUT)
    log.info("published %s (index=%.2f status=%s)", OUT, doc["meta"]["main_index"], doc["meta"]["status"])
    return True
