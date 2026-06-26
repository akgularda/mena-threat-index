"""Lightweight validation of the assembled mena_data.json.
validate(doc) -> list of error strings (empty == valid). publish.py refuses to
overwrite the live file when this returns errors, so a bad run keeps last-good."""
from __future__ import annotations


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _in_range(x, lo, hi):
    return _num(x) and lo - 1e-9 <= x <= hi + 1e-9


def validate(doc) -> list:
    errs = []
    if not isinstance(doc, dict):
        return ["root: not an object"]
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        errs.append("meta: missing")
    else:
        if not _in_range(meta.get("main_index"), 1, 10):
            errs.append(f"meta.main_index out of range: {meta.get('main_index')}")
        if meta.get("status") not in ("STABLE", "ELEVATED", "CRITICAL"):
            errs.append(f"meta.status invalid: {meta.get('status')}")
        for k in ("generated_at", "next_update"):
            if not isinstance(meta.get(k), str) or not meta.get(k):
                errs.append(f"meta.{k} missing")
        if not _in_range(meta.get("confidence", 0), 0, 1):
            errs.append("meta.confidence out of [0,1]")
    countries = doc.get("countries")
    if not isinstance(countries, dict) or not countries:
        errs.append("countries: missing or empty")
    else:
        for name, co in countries.items():
            if not isinstance(co, dict):
                errs.append(f"countries.{name}: not an object"); continue
            if not _in_range(co.get("index"), 1, 10):
                errs.append(f"countries.{name}.index out of range: {co.get('index')}")
            if not isinstance(co.get("events"), list):
                errs.append(f"countries.{name}.events not a list")
            if not _in_range(co.get("confidence", 0), 0, 1):
                errs.append(f"countries.{name}.confidence out of [0,1]")
    hist = doc.get("history")
    if not isinstance(hist, list) or not hist:
        errs.append("history: missing or empty")
    else:
        for i, p in enumerate(hist[:5]):
            if not isinstance(p, dict) or "timestamp" not in p or not _in_range(p.get("main_index"), 1, 10):
                errs.append(f"history[{i}] malformed")
    fc = doc.get("forecast")
    if not isinstance(fc, list):
        errs.append("forecast: not a list")
    else:
        for i, p in enumerate(fc[:3]):
            if not isinstance(p, dict) or "timestamp" not in p or not _in_range(p.get("main_index"), 1, 10):
                errs.append(f"forecast[{i}] malformed")
    br = doc.get("briefing", {}).get("regional_summary_6h") if isinstance(doc.get("briefing"), dict) else None
    if not isinstance(br, dict) or not br.get("headline"):
        errs.append("briefing.regional_summary_6h.headline missing")
    elif not isinstance(br.get("bullets"), list):
        errs.append("briefing.bullets not a list")
    mk = doc.get("markets")
    if mk is not None:
        if not isinstance(mk, dict):
            errs.append("markets: not an object")
        elif not isinstance(mk.get("instruments", []), list):
            errs.append("markets.instruments not a list")
    return errs
