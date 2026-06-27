"""MENA Threat Index pipeline orchestrator.
Flow: config -> feeds -> categorize -> score -> history(append) -> forecast ->
markets -> briefing -> publish (validate + atomic write, withhold on failure)."""
from __future__ import annotations

import os
import sys

from . import config as cfgmod
from . import feeds, categorize, score, forecast, markets, briefing, history, publish, llm
from .util import setup_logging, utcnow, bucket_id, bucket_time, to_iso, ROOT


def _load_dotenv():
    """Local convenience: load KEY=VALUE lines from a gitignored .env into the
    environment (does not override values already set, e.g. CI secrets)."""
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    _load_dotenv()
    log = setup_logging()
    cfg = cfgmod.load()
    now = utcnow()
    step = int(cfg.settings.get("forecast", {}).get("step_hours", 2))
    run_id = bucket_id(now, step)
    ts = bucket_time(now, step)
    log.info("=== MTI run %s (countries=%d, llm=%s) ===",
             run_id, len(cfg.countries), "on" if llm.available(cfg) else "off")

    history.ensure_seeded(log)

    # 1. ingest + categorize
    articles, fstats = feeds.fetch_all(cfg, log)
    events = categorize.categorize(articles, cfg, log)

    # coverage floor gate (withhold if too few countries covered)
    covered = len({e["country"] for e in events})
    coverage = covered / max(1, len(cfg.countries))
    floor = float(cfg.settings.get("ingest", {}).get("coverage_floor", 0.30))
    if coverage < floor:
        log.error("ABORT: coverage %.0f%% < floor %.0f%% — keeping last-good mena_data.json",
                  coverage * 100, floor * 100)
        return 2

    # 2. score (with priors/prev from persisted history)
    priors = history.priors(cfg, log)
    prev_index = history.prev_index_map()
    prev_composite = history.prev_composite()
    prev_fn = history.prev_forecast_next()   # previous run's 1-step forecast (for realized accuracy)
    recent_diffs = history.recent_diffs(12)
    result = score.score(events, cfg, priors, prev_index, recent_diffs, prev_composite, log)

    # 3. append history BEFORE forecast so the new point is included
    hist_before = history.read_composite()
    seed_sigma = None
    seedrows = [r for r in hist_before if r.get("seed")]
    if len(seedrows) >= 3:
        import numpy as np
        seed_sigma = float(np.std(np.diff([float(r["index"]) for r in seedrows]))) or None

    # forecast uses history + the new reading appended virtually
    from datetime import timezone
    virtual = hist_before + [{"_dt": ts.astimezone(timezone.utc), "index": result["composite"], "seed": False}]
    fc_points, fc_meta = forecast.forecast(virtual, cfg, result["confidence"], seed_sigma, log)
    forecast_next = fc_points[0]["main_index"] if fc_points else result["composite"]

    # realized accuracy of the previous run's forecast vs the new actual (monitoring)
    if prev_fn is not None and prev_composite is not None:
        history.append_forecast_eval(run_id, to_iso(ts), result["composite"], prev_fn, prev_composite, log)
    fc_meta["realized"] = history.recent_forecast_skill()

    # 4. markets (non-fatal)
    mk = markets.build(cfg, virtual, fc_points, result["composite"], log)
    market_rows = mk.get("instruments", [])

    # persist this run
    history.append_run(run_id, to_iso(ts), result, forecast_next, market_rows, log)

    # 5. briefing (prev_composite captured BEFORE the append, for the trend term)
    brief = briefing.build(result, cfg, log, prev_composite=prev_composite)

    # 6. assemble + publish
    model_label = "deterministic+nvidia" if llm.available(cfg) else "deterministic"
    doc = publish.assemble(cfg, run_id, ts, result, fc_points, fc_meta, brief, mk, model_label)
    ok = publish.publish(cfg, doc, log)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
