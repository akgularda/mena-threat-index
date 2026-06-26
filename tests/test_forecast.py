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
