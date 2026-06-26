import json, urllib.request, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

def test_map_names_present():
    cfg = config.load()
    url = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json"
    try:
        topo = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception:
        import pytest; pytest.skip("network unavailable")
    names = {g["properties"]["name"] for g in topo["objects"]["countries"]["geometries"]}
    missing = [c.map_name for c in cfg.countries if c.map_name not in names]
    assert not missing, f"map_name not in TopoJSON: {missing}"
