# MENA Threat Index — Patch Plan (proposal)

**Status:** Each item is scoped so it can be implemented, reviewed, and reverted independently. Findings (F#) and recommendations (R#) reference [`METHODOLOGY_REVIEW.md`](./METHODOLOGY_REVIEW.md).

**Implementation status (branch `methodology-review`):** ✅ **P1–P5 (Tier 1)** and ✅ **P6** implemented, each test-first with an atomic commit; full `pytest` suite green (26 tests). ⬜ **P7–P9 (Tier 2)** and ⬜ **P10–P13 (Tier 3)** not yet implemented. All shipped changes default to the new, corrected behaviour where it is a defect fix (P1, P2, P4, P5) and are config-selectable back to the legacy behaviour where the change is model-altering (confidence_model, lag_selection).

**Ground rules carried through every item:**
- The system stays **deterministic and explainable**; the only ML is the *existing optional* LLM refinement.
- **Model-changing items default to current behaviour behind a config flag**, so the existing `pytest` suite stays green until each change is explicitly turned on and re-pinned.
- `NVIDIA_API_KEY` remains **environment-only** — never written to any file.
- The **withhold-on-failure** publish discipline (`publish.py` + `schema.py`) is preserved.

**Invariants that must remain true after every patch** (existing pins in `tests/`):
country `index` ∈ [1,10] and monotone in severity (`test_score`); empty events → index 1.0 / STABLE (`test_score`); schema ranges & status enum (`test_schema`); forecast points ∈ [1,10] with a `method` label (`test_forecast`); every `map_name` in the TopoJSON (`test_map_names`); briefing events carry `country` (`test_briefing`); keyword mappings (`test_categorize`).

---

## Tier 1 — Must-fix

### P1 · Region-led, severity-gated headline (R1 / F1)
- **Files:** `pipeline/briefing.py`; `config/settings.yml`; `pipeline/llm.py` (polish guard).
- **Change:** replace the argmax-country headline (`briefing.py:53-57`) with the deterministic algorithm in METHODOLOGY_REVIEW §6: lead with composite + status + trend; name ≤ 2 countries only when `index ≥ thresholds.elevated` **and** `confidence ≥ briefing.headline_conf_gate`, ranked by `country_weight·(index−1)·confidence`; otherwise emit the "quiet window" or "broad-based" forms. Constrain the optional LLM polish (`llm.summarize_briefing`) so a polished headline is **rejected** (keep deterministic) unless it preserves the composite value and region-first framing.
- **Config (new):** `briefing.headline_conf_gate: 0.35` (tunable). Reuses `thresholds.elevated`.
- **Why:** the maximum of 24 noisy series is biased high and reverts (Galton 1886; Forde et al. 2023); a STABLE region should not read as a single-country alarm (OECD 2008 — communicate the headline with its drivers).
- **Tests:** existing `test_briefing` (events carry `country`) unaffected. **Add** `tests/test_briefing.py` cases: (a) STABLE composite + only a low-confidence argmax → headline is region-led / "quiet window", **not** "<country> leads"; (b) a country with `index≥4` and `confidence≥gate` → it is named; (c) elevated composite, no country clears the gate → "broad-based" wording; (d) LLM polish that drops the composite is rejected.
- **Manual check:** on the current `mena_data.json` (composite 2.58, no country ≥4 with conf ≥0.35) the headline becomes *"Quiet regional window — composite 2.58 (stable, steady)."*

### P2 · Fix the confidence "agreement" term (R2 / F7)
- **Files:** `pipeline/score.py` (`_per_country` confidence block, lines 64-78); `config/settings.yml`; (target) `pipeline/feeds.py` to retain a corroboration count.
- **Change:** introduce `score.confidence_model` with three options and **default to current** so nothing breaks:
  - `"v_d_a"` *(current default)* — unchanged.
  - `"v_d"` *(interim fix)* — `confidence = (V·D)^(1/2)`; drops the harmful homogeneity term.
  - `"v_d_c"` *(target)* — replace `A` with a **corroboration** term `C`: during per-country de-dup (`feeds.py:138-162`) store, on each kept article, the count of distinct sources whose titles fuzzy-matched it; then `C = Σ(aw·1[corroboration≥2]) / Σaw`, and `confidence = (V·D·C)^(1/3)`.
- **Why:** `A = 1 − σ_w/3` measures category homogeneity, not corroboration — it floors large, well-sourced, topically-broad countries (Iran: 221 events → conf 0.05). Corroboration-with-uncertainty is the established pattern (Kaufmann et al. 2010 WGI margins of error; Pemstein et al. V-Dem rater model).
- **Tests:** `test_score` pins **index** bounds/monotonicity and empty→baseline only — no confidence values — so it stays green; `test_schema` still satisfied (`confidence ∈ [0,1]`). **Add** a test: a synthetic high-volume, multi-category, multi-source country yields confidence well above the 0.05 floor under `"v_d_c"`.

### P3 · Correct the website "Translation"/"multilingual" claims (R3 / F12, F15)
- **Files:** `index.html` (pipeline `pipeline` array ~line 1012; new Confidence + Methodology-basis blocks; Limitations sentence). Copy is given verbatim in METHODOLOGY_REVIEW §8.
- **Change:** delete the false **Translation** step; relabel steps 02–04; add the Confidence subsection (currently the page never explains the confidence number it ships); add the short "Methodology basis" reference block; add the non-English-coverage caveat to Limitations.
- **Why:** no translation module exists (`score.py:104` copies `translated_title = title`); the deterministic lexicon is English-only, so "multilingual" only holds with the LLM on. Accurate self-description is a credibility requirement (and matches the honesty already shown in Limitations).
- **Tests:** none automated (static frontend). **Manual:** `python -m http.server 8000` → open Methodology screen, confirm steps, Confidence block, and references render.

### P4 · Disambiguate country attribution (R4 / F2)
- **Files:** `pipeline/feeds.py` (shared-feed attribution, lines 103-123); `config/countries.yml` (Oman query).
- **Change:** (a) replace bare substring `name in hay` with **word-boundary** matching (regex `\b<name>\b`, case-insensitive, plus the existing alias map) so "Amman" no longer matches "Oman" and "Sudan" no longer matches "South Sudan" by accident; (b) fix the Oman Google-News query from `q: 'عمان'` to a disambiguated form, e.g. `q: 'سلطنة عمان OR "عُمان" -عمّان -الأردن'` (Sultanate of Oman, excluding Amman/Jordan); (c) optionally require a second cue before a shared-feed article is attributed.
- **Why:** `عمان` is ambiguous between Oman and Amman; substring matching over-attributes. Geolocation accuracy is the core quality axis for media event data (Weidmann 2015). Live Oman (#2, 4.05, 209 events) is the visible symptom.
- **Tests:** `test_map_names` unaffected (`map_name` values unchanged). **Add** `tests/test_feeds.py`: a headline containing "Amman" is **not** attributed to Oman; "South Sudan" is **not** attributed to Sudan; a clean "Oman signs deal" **is** attributed to Oman.

### P5 · Control the market lag selection (R5 / F10)
- **Files:** `pipeline/markets.py` (`_analyse` lag loop, lines 234-262); `config/settings.yml`.
- **Change:** add `markets.lag_selection: "sidak"` (default) | `"best"` (current) | `"preregistered"`. For `"sidak"`, after picking the max-|r| lag, inflate its p-value for the search of `n = len(lag_set)` lags: `p_adj = 1 − (1 − p_best)^n` (or Bonferroni `min(1, p_best·n)`), and feed `p_adj` (not `p_best`) into both the publication gate and the across-instrument Benjamini–Hochberg step. For `"preregistered"`, restrict `lag_set` to `[0, 1]` and skip the search.
- **Why:** choosing the luckiest of 7 lags inflates significance; multiple-testing control must cover the lag search, not just the instrument set (Benjamini & Hochberg 1995).
- **Tests:** no existing market tests. **Add** `tests/test_markets.py`: `_pearson`/`_ols` numeric sanity; `_bh_significant` known-input check; and that `p_adj ≥ p_best` and a borderline correlation flips to non-significant under `"sidak"`.

---

## Tier 2 — Should-fix

### P6 · README shrinkage wording (R6 / F8)
- **Files:** `README.md` (Methodology summary) — replace with METHODOLOGY_REVIEW §7 text, which states shrinkage correctly (cold-start pulls toward calm; the real anti-"false-calm" protection is the confidence value).
- **Tests:** none.

### P7 · Wire up or remove the dead robustness knobs (R7 / F11)
- **Files:** `pipeline/markets.py`; `config/settings.yml`.
- **Change (recommended):** **implement** a stationary/block bootstrap (Politis & Romano 1994; Künsch 1989) over the aligned `(Δindex, return)` series using `bootstrap_resamples` and `bootstrap_block`, and publish `correlation_ci` / `beta_ci` (low, high) on each instrument; **remove** `bnti_prior_k` (there are no per-instrument BNTI betas to shrink toward, so the knob is misleading) — or, if a defensible per-instrument prior is supplied later, implement `beta_shrunk = (n·beta + k·beta_prior)/(n+k)` explicitly. Gate the bootstrap behind `markets.bootstrap_enabled: true` (adds fields only).
- **Why:** config currently advertises robustness the code never applies. Either deliver it or stop implying it.
- **Tests:** `test_schema` only checks required keys, so new optional fields are safe. **Add** a bootstrap-CI test in `tests/test_markets.py` (CI brackets the point estimate; widens on smaller `bootstrap_block`).

### P8 · Categorisation robustness (R8 / F3, F4)
- **Files:** `pipeline/categorize.py`; `config/categories.yml`; `pipeline/llm.py`.
- **Change:** (a) add a lightweight **negation guard** — if a matched keyword span is immediately preceded by a negator (`no|denies|dismisses|averts|prevents|ends|de-escalat`), demote to the next non-matching category; (b) **narrow over-broad tokens** — drop bare `threat`/`warn`/`accuse` from `diplomatic_tensions` or require co-occurrence; (c) **casualty routing** — stop sending bare `killed|wounded|death toll` to `humanitarian_crisis`; keep humanitarian gated on humanitarian context (`refugee|famine|flood|epidemic|aid`), letting combat phrasing carry military weight; (d) **LLM abstention** — instruct the model to return `uncertain` when unsure and **do not override** the keyword label on `uncertain`.
- **Why:** title-only first-match regex is brittle; coded event schemes handle negation/severity explicitly (Goldstein 1992; CAMEO).
- **Tests:** must keep `test_categorize`'s seven pinned mappings green (airstrike→military, suicide bomber→terrorism, troops→border, opposition arrested→political, summit→diplomatic, trade agreement→trade, football→neutral). **Add** cases: "ceasefire collapses" ≠ de-escalation; "no airstrike reported" ≠ military_conflict; a combat-casualty headline → military_conflict.

### P9 · Surface confidence in the UI + de-duplicate briefing bullets (R9, R10 / F13)
- **Files:** `index.html` (overview + country detail); `pipeline/briefing.py`.
- **Change:** (a) display the existing `confidence` on the composite and country-detail screens (data already present; no contract change); (b) de-dup briefing bullets by `title_key`/Jaccard before finalising and constrain LLM bullets to unique events (live data currently shows the same IAEA headline twice).
- **Tests:** `test_briefing` unaffected. **Add** a bullet-dedup test (duplicate input events → distinct bullets).

---

## Tier 3 — Optional re-grounding (clearly separated; default off)

### P10 · Expose the saturation & country weights as config (R13 / F6, F9)
- **Files:** `pipeline/score.py`; `config/settings.yml` (`score.saturation_scale: 5`, `score.saturation_gain: 1.2`); country weights already in `countries.yml`.
- **Change:** read the transform constants from settings instead of the hard-coded `−eff/5·1.2`. **Defaults reproduce the current formula exactly**, so `test_score` is unchanged. Prereq for P11/P12.

### P11 · Literature-anchored alternative category weights (R11 / F5)
- **Files:** `config/categories.yml` (add a second `weights_goldstein` set); `pipeline/config.py`; `config/settings.yml` (`score.weight_set: "default"|"goldstein"`).
- **Change:** ship an alternative weight vector whose **ordering and spacing are derived from the Goldstein/CAMEO conflict-intensity scale** (documented derivation, not invented), selectable via config. Default `"default"` preserves current scores/tests.
- **Tests:** `test_categorize` (keyword mapping) unaffected. **Add** a test that `"goldstein"` loads and preserves the severity ordering.

### P12 · Sensitivity-analysis appendix (R12)
- **Files (new):** `scripts/sensitivity.py`; output `docs/methodology_sensitivity.md`.
- **Change:** on a fixed event fixture, perturb category weights, country weights, half-life, and saturation constants (±, OAT and a small Monte-Carlo) and report how the **composite value and country ranking** move (Saisana, Saltelli & Tarantola 2005; OECD 2008). Standalone — **not** in the pipeline hot path.
- **Tests:** **Add** a smoke test that the script runs on the fixture and emits a report; no pipeline behaviour changes.

### P13 · (Future, optional) local-language lexicons (R14 / F15)
- **Files:** `config/categories.yml` (per-language keyword sets); `pipeline/categorize.py` (compile per `lang`).
- **Change:** add Arabic/Persian/Hebrew/Turkish keyword sets so the **deterministic** baseline categorises local-language headlines without the LLM. Larger effort; flagged as a roadmap item, not Tier 1.

---

## Suggested order & verification

1. **Tier 1** (P1–P5) — each independent; land P1 and P2 first (they fix the most visible, evidence-backed defects), then P3 (copy), P4, P5.
2. **Tier 2** (P6–P9).
3. **Tier 3** (P10 → P11/P12 → P13), P10 first as plumbing.

**After each patch:**
- `pytest -q` stays green (defaults preserve behaviour); new tests listed per item are added in the same change.
- For frontend items: `python -m pipeline.run` (or use the committed `mena_data.json`) then `python -m http.server 8000` and click through **Overview → Country → Methodology**.
- For markets/sensitivity: run `tests/test_markets.py` and `scripts/sensitivity.py` against the fixture.
- Re-run a full pipeline (`python -m pipeline.run`) and confirm the run still **publishes** (or correctly withholds) and that `schema.validate` returns `[]`.

**Definition of done for the whole plan:** the methodology page describes only what the pipeline actually does; the headline represents the region; confidence is meaningful and visible; market betas are not lag-cherry-picked; and every model-changing constant is either justified in `METHODOLOGY_REVIEW.md`, exposed in config, or covered by the sensitivity appendix — with the deterministic baseline and `NVIDIA_API_KEY`-env-only guarantee intact.
