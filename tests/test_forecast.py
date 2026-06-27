import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import timezone
from pipeline import config, forecast
from pipeline.util import utcnow

def _hist(vals):
    now = utcnow(); out = []
    for i, v in enumerate(vals):
        from datetime import timedelta
        out.append({"_dt": now - timedelta(hours=2 * (len(vals) - i)), "index": v, "seed": False})
    return out

def test_coldstart_and_ar1():
    cfg = config.load()
    p1, m1 = forecast.forecast(_hist([4.0]), cfg, 0.5, 0.6)
    assert p1 and 1 <= p1[0]["main_index"] <= 10 and "persistence" in m1["method"]
    p2, m2 = forecast.forecast(_hist([3,3.2,3.1,3.5,3.8,4.0,4.1,4.0,4.2,4.3,4.1,4.4,4.5]), cfg, 0.7, None)
    assert p2 and all(1 <= x["main_index"] <= 10 for x in p2)


def test_persistence_is_default_on_short_history():
    cfg = config.load()
    pts, meta = forecast.forecast(_hist([3.0, 3.1, 2.9, 3.0, 3.2, 3.0, 2.8, 3.1]), cfg, 0.6, None)
    assert meta["selected_model"] == "persistence"
    assert abs(pts[0]["main_index"] - 3.1) < 1e-6              # anchored at the last value


def test_seed_rows_are_excluded_from_the_fit():
    cfg = config.load()
    seed = _hist([5.5, 5.4, 5.6])                              # stale high seed regime
    for r in seed:
        r["seed"] = True
    real = _hist([2.0, 2.1, 1.9, 2.0])                        # current low regime
    pts, _ = forecast.forecast(seed + real, cfg, 0.6, None)
    assert pts[0]["main_index"] < 3.0                          # anchored to real ~2.0, not seed ~5.5


def test_no_global_mean_reversion_on_regime_shift():
    cfg = config.load()
    pts, _ = forecast.forecast(_hist([5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2]), cfg, 0.6, None)
    assert pts[0]["main_index"] <= 2.5                         # stays in the current regime...
    assert all(p["main_index"] <= 3.0 for p in pts)            # ...unlike the old AR(1)


def test_bands_are_bounded_and_widen():
    cfg = config.load()
    pts, _ = forecast.forecast(_hist([3.0, 3.2, 2.8, 3.1, 2.9, 3.3, 2.7, 3.0, 3.1, 2.9]), cfg, 0.6, None)
    assert all(1.0 <= p["low"] <= p["main_index"] <= p["high"] <= 10.0 for p in pts)
    assert (pts[-1]["high"] - pts[-1]["low"]) >= (pts[0]["high"] - pts[0]["low"]) - 1e-9


def test_selector_upgrades_off_persistence_on_a_clear_trend():
    cfg = config.load()
    vals = [round(2.0 + 0.2 * i, 3) for i in range(20)]        # clean trend, enough folds
    name, skill = forecast.select_model(vals, cfg)
    assert name != "persistence" and skill > 0


def test_forecast_eval_logging_and_skill(tmp_path, monkeypatch):
    from pipeline import history
    monkeypatch.setattr(history, "FORECAST_EVAL", str(tmp_path / "fe.jsonl"))
    history.append_forecast_eval("r1", "2026-01-01T00:00:00Z", actual=2.0, model_pred=2.4, naive_pred=2.1)
    history.append_forecast_eval("r2", "2026-01-01T02:00:00Z", actual=2.0, model_pred=2.0, naive_pred=2.3)
    sk = history.recent_forecast_skill()
    assert sk["n"] == 2
    assert sk["mae_model"] == 0.2 and sk["mae_naive"] == 0.2   # (0.4+0.0)/2, (0.1+0.3)/2
