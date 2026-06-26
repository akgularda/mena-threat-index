"""Export the former BNTI composite history into data/seed/bnti_history_seed.json
and seed data/history.jsonl (rows flagged seed:true). Idempotent."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.util import ROOT
from pipeline import history as H

# BNTI history (from BNTI.dc.html DATA.history) as [timestamp, index] pairs.
BNTI = [
    ["2026-06-24T07:51",3.34],["2026-06-24T10:59",3.4],["2026-06-24T14:35",3.5],["2026-06-24T18:06",3.7],
    ["2026-06-24T19:48",3.26],["2026-06-24T21:35",3.32],["2026-06-24T23:22",3.56],["2026-06-25T14:30",5.26],
    ["2026-06-25T18:26",5.25],["2026-06-25T21:48",4.21],["2026-06-25T23:36",4.7],["2026-06-26T02:52",3.84],
]

def main():
    seed = {"source": "BNTI", "history": [{"timestamp": t + ":00Z" if len(t) == 16 else t, "index": v}
                                          for t, v in BNTI]}
    os.makedirs(os.path.join(ROOT, "data", "seed"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "seed", "bnti_history_seed.json"), "w", encoding="utf-8") as f:
        json.dump(seed, f, indent=2)
    n = H.ensure_seeded()
    print(f"seed file written; ensure_seeded added {n} rows to data/history.jsonl")

if __name__ == "__main__":
    main()
