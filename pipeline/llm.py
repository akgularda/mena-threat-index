"""NVIDIA OpenAI-compatible LLM client (optional).

The API key is read ONLY from the environment variable NVIDIA_API_KEY and is
never persisted. If the key is absent or llm.enabled is false, available()
returns False and the pipeline uses the deterministic keyword engine.
"""
from __future__ import annotations

import json
import os
import re

import requests

ENV_KEY = "NVIDIA_API_KEY"


def _key() -> str | None:
    k = os.environ.get(ENV_KEY)
    return k.strip() if k else None


def available(cfg) -> bool:
    llm = cfg.settings.get("llm", {})
    return bool(llm.get("enabled", True)) and _key() is not None


def _chat(cfg, messages, log, max_tokens=1500):
    llm = cfg.settings.get("llm", {})
    url = llm.get("base_url", "https://integrate.api.nvidia.com/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": llm.get("model", "meta/llama-3.3-70b-instruct"),
        "messages": messages,
        "temperature": float(llm.get("temperature", 0.0)),
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
               "Accept": "application/json"}
    timeout = int(llm.get("timeout_s", 40))
    retries = int(llm.get("max_retries", 2))
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:
            last = str(e)
    raise RuntimeError(f"LLM call failed after {retries + 1} tries: {last}")


def _extract_json(text: str):
    """Pull the first JSON array/object out of a possibly markdown-wrapped reply."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        j = text.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return None


def classify(cfg, titles, valid_keys, log) -> list:
    """Return a category key per title (aligned, same length)."""
    if not titles:
        return []
    llm = cfg.settings.get("llm", {})
    batch = int(llm.get("batch_size", 25))
    keys = sorted(valid_keys)
    out: list[str] = []
    cat_list = ", ".join(keys)
    for start in range(0, len(titles), batch):
        chunk = titles[start:start + batch]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(chunk))
        sys = ("You are a geopolitical news classifier for a Middle East & North "
               "Africa threat index. Assign each headline to exactly ONE category.")
        usr = (f"Categories: {cat_list}\n\n"
               f"Headlines:\n{numbered}\n\n"
               f"Return ONLY a JSON array of {len(chunk)} strings (one category key per "
               f"headline, in order). Use 'neutral' only for genuine non-threat news; use "
               f"'uncertain' if you cannot confidently classify a headline — do not guess. "
               f"An 'uncertain' label keeps the deterministic keyword category.")
        content = _chat(cfg, [{"role": "system", "content": sys},
                              {"role": "user", "content": usr}], log,
                        max_tokens=min(2000, 40 + 12 * len(chunk)))
        arr = _extract_json(content)
        if not isinstance(arr, list) or len(arr) != len(chunk):
            # alignment failed for this batch -> signal caller to keep keyword labels
            raise ValueError(f"LLM returned {type(arr).__name__} len "
                             f"{len(arr) if isinstance(arr, list) else '?'} != {len(chunk)}")
        out.extend(str(x).strip().lower().replace(" ", "_") for x in arr)
    return out


def summarize_briefing(cfg, top_events, country_scores, log):
    """Optional: return {headline, bullets:[{text,cat}]} or None on failure."""
    if not available(cfg):
        return None
    llm = cfg.settings.get("llm", {})
    if not llm.get("use_for_briefing", True):
        return None
    lines = []
    for e in top_events[:14]:
        lines.append(f"- [{e['country']} | {e['category']}] {e['title']}")
    ranked = ", ".join(f"{c['name']} {c['index']:.1f}" for c in country_scores[:6])
    sys = ("You write a terse, factual regional security briefing for a MENA threat "
           "index. No speculation, no fluff, present tense.")
    usr = (f"Top countries by index: {ranked}.\n\nKey headlines (last 48h):\n"
           + "\n".join(lines) +
           "\n\nReturn ONLY JSON: {\"headline\": \"<=120 chars\", \"bullets\": "
           "[{\"text\": \"one sentence\", \"cat\": \"category_key\"}, ...]} with 3 bullets.")
    try:
        content = _chat(cfg, [{"role": "system", "content": sys},
                              {"role": "user", "content": usr}], log, max_tokens=600)
        obj = _extract_json(content)
        if isinstance(obj, dict) and obj.get("headline") and isinstance(obj.get("bullets"), list):
            return obj
    except Exception as e:
        log.warning("llm briefing failed: %s", e)
    return None
