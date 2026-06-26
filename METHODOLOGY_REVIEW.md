# MENA Threat Index — Methodology Review

**Status:** Review only — no pipeline code is changed by this document.
**Date:** 2026-06-26 · **Scope:** the full MTI pipeline (`pipeline/*.py`, `config/*.yml`), the published contract (`mena_data.json`), and the user-facing methodology (`index.html`).
**Companion:** proposed changes are itemised separately in [`PATCH_PLAN.md`](./PATCH_PLAN.md). Nothing here should be implemented until that plan is reviewed and approved.

### Provenance of citations
Every one of the 31 references in §9 was **independently fetched from its publisher / DOI page and then adversarially re-checked** by a second pass instructed to *refute* it; all 31 survived (verdict *confirmed*, working DOI/URL, metadata corrected where needed). No citation in this document is from memory alone. This was done specifically to honour the "do not fabricate citations" requirement.

---

## 1. Executive summary

MTI is, in its bones, a **news-frequency risk index** in the same family as the academic Geopolitical Risk index (Caldara & Iacoviello 2022) and the Economic Policy Uncertainty index (Baker, Bloom & Davis 2016). That lineage is a genuine strength: the design is deterministic, auditable, withholds bad runs, and already uses several techniques with strong scholarly backing (empirical-Bayes shrinkage, Kish effective sample size, Shannon-entropy diversity, exponential smoothing, Benjamini–Hochberg FDR). The problem is not the skeleton — it is that **almost none of the constants, weights, and ad-hoc terms are justified**, a few claims are **factually wrong**, and the **headline misrepresents the region**.

The five things most worth fixing:

1. **The headline represents one country, not the region.** The live headline is *"Iran leads regional risk on military conflict signals; composite holds 2.58 (stable)"* — the composite is **STABLE (2.58)**, yet the single argmax country is foregrounded. Selecting the maximum of 24 noisy series is biased high and unstable (Galton 1886 regression to the mean; Forde, Hemani & Ferguson 2023 "Winner's Curse"). **Decision taken: a region-led, severity-gated hybrid headline.** (§6)
2. **The confidence model punishes normal countries.** The third confidence term `A = 1 − σ_w/3` measures *category homogeneity*, not corroboration. Iran has 221 events but **confidence 0.05** purely because its events span many categories. The argmax headline therefore elevates the *least* reliable signal. (§2.7, §4-F7)
3. **The website claims a translation step that does not exist.** Methodology step 02 says "Non-English items are machine-translated"; there is **no translator anywhere** in `pipeline/` (`score.py` sets `translated_title = title`). Worse, the deterministic keyword lexicon is **English-only**, so without the optional LLM, Arabic/Persian/Hebrew/Turkish headlines mostly score as `neutral`. (§4-F12, §4-F15)
4. **The market module's lag is cherry-picked.** The "threat beta" uses the *best of 7 lags by |r|* (`markets.py:234-240`) — a garden-of-forking-paths that inflates correlations — while the Benjamini–Hochberg gate is applied only *across instruments*, not *across lags*. (§4-F10)
5. **Hand-set numbers with no basis.** Category weights (8/7/5/4/3/2.5/−2), the saturation constants (`/5 · 1.2`), and country strategic weights are inherited or invented. The literature gives defensible anchors (Goldstein 1992 / CAMEO for event severity; the OECD composite-indicator handbook + Saisana et al. 2005 for weighting and sensitivity analysis). **Decision taken: re-ground these, behind config flags, with a sensitivity analysis — keeping the deterministic baseline intact.** (§5)

Everything recommended keeps the system deterministic and explainable; any ML stays optional and clearly separated; `NVIDIA_API_KEY` remains environment-only.

---

## 2. Current methodology in plain language

The orchestration (`pipeline/run.py`) is: **config → feeds → categorize → score → history(append) → forecast → markets → briefing → publish**, every 2 hours, withholding the last-good `mena_data.json` if a run fails validation or coverage is too thin.

### 2.1 Event ingestion (`feeds.py`, `config/countries.yml`, `config/settings.yml`)
For each of 24 countries the pipeline fetches curated outlet RSS plus Google-News RSS queries (one English, one local-language, with Google's `when:Nd` recency operator). Four pan-regional "shared feeds" are fetched once and **attributed to a country when the country's name appears as a substring of the headline** (`feeds.py:103-123`), with a few aliases (UAE, Gaza→Palestine, Tehran→Iran, Türkiye→Turkey). Articles are kept if published within `window_hours = 72`, then de-duplicated **per country** by exact title/URL key and fuzzy Jaccard ≥ `0.90`. A **coverage floor** (`coverage_floor = 0.30`) withholds the whole run if fewer than 30 % of countries have any event.

### 2.2 Categorization — keyword engine + optional LLM (`categorize.py`, `config/categories.yml`, `llm.py`)
The deterministic floor is a **severity-ordered regex over the headline title only**; the *first* category whose pattern matches wins (`categorize.py:25-33`). If `NVIDIA_API_KEY` is present and `llm.enabled`, an LLM (Llama-3.3-70B, temperature 0) re-labels each headline in batches and **overrides** the keyword label when it returns a valid category (`categorize.py:71-82`). On any LLM failure the keyword label stands. There is no confidence/abstention on either path.

### 2.3 Category weights (`config/categories.yml`)
Fixed weights, never changed between runs (for auditability): military conflict **8**, terrorism **7**, border security **5**, political instability **4**, humanitarian crisis **3**, diplomatic tensions **2.5**, trade/de-escalation **−2**, neutral **0**. A **source-credibility** multiplier (0.5–1.0; default 0.6) is applied to the *averaging* weight, not to severity (`categorize.py:36-48`).

### 2.4 Recency weighting (`score.py:34`, `settings.yml`)
Each event gets `λ = 0.5 ** (age_h / 18)` (an 18-hour half-life), and a hard cutoff drops events older than `score_window_hours = 48`.

### 2.5 Country scoring (`score.py:_per_country`)
Let `w` = event severity weight, `cred` = source credibility, `λ` = recency. Averaging weights `aw = λ·cred`. The mean event weight is `avg_w = Σ(aw·w)/Σaw`. The **Kish effective sample size** is `n_eff = (Σaw)² / Σ(aw²)` (`score.py:54-55`). An **empirical-Bayes shrinkage** pulls this toward a per-country prior `m_prior` (the trailing 14-day mean raw score, or 0 at cold start) with pseudo-count `k = 3`:

```
avg_w_shrunk = (n_eff · avg_w + k · m_prior) / (n_eff + k)
eff          = max(0, avg_w_shrunk)
index        = clip( 1 + 9 · (1 − e^(−eff/5 · 1.2)), 1, 10 )
```

Note: `index` is driven by the **mean** event weight — *volume* enters only through `n_eff` (shrinkage) and confidence, **not** the level. The transform constants `/5` and `·1.2` are inherited verbatim from BNTI.

### 2.6 Regional composite (`score.py:150-178`)
A strategic-weighted mean of country indices (`composite_raw = Σ wᶜ·indexᶜ / Σ wᶜ`, country weights from `countries.yml`), then an **asymmetric EWMA**: smoothing `α = 0.20` normally, `α = 0.50` when a spike is detected (`|Δ| > max(1.0, 2σ)` over recent diffs). Result clipped to [1, 10]. Status bands: STABLE < 4.0 ≤ ELEVATED < 7.0 ≤ CRITICAL.

### 2.7 Confidence (`score.py:64-78`, `180-185`)
Per country, the geometric mean of three terms:
- **Volume adequacy** `V = 1 − e^(−n_eff/4)`,
- **Source diversity** `D = H/ln(4)`, where `H = −Σ p log p` is Shannon entropy of recency-weighted source shares,
- **"Agreement"** `A = 1 − min(1, σ_w/3)`, where `σ_w` is the weighted std-dev of event *severity weights*.

`confidence = (V·D·A)^(1/3)`, clipped to [0.05, 0.99]. **This is the source of the Iran 0.05 anomaly** (§4-F7): `A` collapses when a country's events span many severity categories, which is normal for a large, heavily-covered country. Composite confidence = `coverage · freshness · (weighted-mean country confidence)`, where `freshness = 0.5 ** (staleness_h / 12)`.

### 2.8 Forecasting (`forecast.py`)
A graceful ladder on the smoothed composite series:
- **≥ 12 points → AR(1)+drift:** fit `S_t = c + φ·S_{t−1}`, shrink `φ` toward a 0.90 prior with strength `k=10`, forecast `mean = μ + φ^k·(S_t − μ)` with variance growing toward the stationary `σ²/(1−φ²)`.
- **5–11 points → damped Holt** (`holt_phi = 0.80`).
- **< 5 points → persistence** (flat last value, widening bands).

Bands are an 80 % predictive interval (`band_z = 1.2815`); each horizon also reports a confidence from band width × composite confidence.

### 2.9 Market correlation / threat beta (`markets.py`)
For each instrument (oil, gold, VIX, S&P, defense ETF, EM, regional FX/equities): fetch a daily price series (keyless Yahoo v8, FRED fallback), take log-returns, difference the index to a daily series, and over lags −3…+3 **pick the lag with the largest |r|** (`markets.py:234-240`). At that lag, OLS gives a **"threat beta"** (return per +1 index point). Publication gates: ≥ 20 native paired obs, |r| ≥ 0.30, |β|/SE ≥ 1.5, rolling sign stability ≥ 0.60, and a **Benjamini–Hochberg FDR** at `q = 0.10` *across instruments* (`markets.py:330-337`). Market moves are projected from the forecast path with damped accumulation and inflated bands. Launch-day estimates are BNTI-seeded and badged.

### 2.10 Briefing & headline (`briefing.py`)
Countries are ranked by index; salient threat events (weight > 0) are ranked by `|weight|·recency·credibility`; up to three bullets are taken from distinct countries. The **headline is templated from the top-ranked country**: `"{top.name} leads regional risk on {top_category} signals; composite holds {composite} ({status})."` (`briefing.py:53-57`). An optional LLM polish can replace headline + bullets.

### 2.11 Publishing & integrity (`publish.py`, `schema.py`)
`schema.validate` enforces required keys and ranges (`main_index` ∈ [1,10], `status` ∈ {STABLE,ELEVATED,CRITICAL}, per-country `index` ∈ [1,10], `confidence` ∈ [0,1], briefing headline non-empty). On any error the run is **withheld** and the last-good file kept; otherwise an atomic temp-then-replace write. This withhold-on-failure discipline is a real strength and should be preserved.

---

## 3. Literature-backed justification (component → source)

| MTI component | What it does | Authoritative basis (verified) |
|---|---|---|
| Whole-index concept | News-frequency → single risk number, validated against markets | **Caldara & Iacoviello (2022)** GPR index; **Baker, Bloom & Davis (2016)** EPU |
| Counting + normalising news terms; auditing the classifier | Frequency counts, scaling before averaging, accuracy audit vs. hand reads | **Baker, Bloom & Davis (2016)** |
| Category → fixed numeric weight | Assigning intensity scores to event types | **Goldstein (1992)** WEIS scale; **Schrodt (2012)** / **Gerner et al. (2002, 2008)** CAMEO |
| Event taxonomy & sourcing/bias context | Disaggregated event categories from media reports | **Raleigh et al. (2010)** ACLED; **Leetaru & Schrodt (2013)** GDELT; **Boschee et al. (2015)** ICEWS |
| "Reporting reflects coverage, not ground truth" limitation | Selection & description bias in media event data | **Weidmann (2015)**; **Earl et al. (2004)** |
| Source-credibility weighting + reporting uncertainty | Combine heterogeneous sources with source-specific weights and margins of error | **Kaufmann, Kraay & Mastruzzi (2010)** WGI |
| Confidence from noisy, disagreeing signals | Aggregate noisy ratings into estimates *with* explicit uncertainty | **Pemstein et al.** V-Dem measurement model |
| Shrinkage of sparse country scores toward a prior | James–Stein / empirical Bayes lowers MSE | **Efron & Morris (1975)**; **Morris (1983)** |
| Effective sample size of weighted events | `n_eff = (Σw)²/Σw²` | **Kish (1965)** |
| Source-diversity term | Entropy `H = −Σ p log p` | **Shannon (1948)** |
| Recency half-life, composite EWMA, damped Holt | Exponential / damped smoothing | **Hyndman & Athanasopoulos (2021)**; **Gardner (2006)**; **Gardner & McKenzie (1985)** |
| AR / persistence baseline, predictive intervals, accuracy | Naïve/random-walk benchmark, scaled error | **Hyndman & Koehler (2006)** |
| Market reaction to events (threat beta) | Market-model abnormal returns around events | **MacKinlay (1997)** |
| Instrument choice & expected signs | Gold safe-haven; oil supply/geopolitical shocks | **Baur & McDermott (2010)**; **Kilian (2009)** |
| FDR across many correlation tests | Benjamini–Hochberg procedure | **Benjamini & Hochberg (1995)** |
| Honest CIs for dependent (time-series) data | Block / stationary bootstrap | **Künsch (1989)**; **Politis & Romano (1994)** |
| **Not** headlining the argmax country | Extreme-of-many is biased high and reverts | **Galton (1886)**; **Forde, Hemani & Ferguson (2023)** |
| Weighting, aggregation, normalisation, robustness of any composite | Construction pipeline + sensitivity analysis | **Nardo et al. / OECD (2008)**; **Saisana, Saltelli & Tarantola (2005)** |

Full metadata, DOIs, and one-line relevance for each appear in §9.

---

## 4. Weak spots & limitations

Each finding: **what · evidence · why it matters · literature context.**

**F1 — Headline = single argmax country (high priority).** `briefing.py:53-57` builds the headline from `ranked[0]`. Live: composite 2.58 STABLE but headline foregrounds Iran. The maximum of many noisy series is upward-biased and unstable (Galton 1886; Forde et al. 2023). → §6.

**F2 — Ambiguous / over-broad attribution.** `feeds.py:115` attributes a shared-feed article to a country if the country name is a **substring** of the headline, and `countries.yml` queries Oman with `q: 'عمان'` — which is *also* Amman (Jordan's capital). Live Oman ranks #2 (4.05, 209 events), almost certainly an artifact. Media event-data accuracy depends on correct geolocation (Weidmann 2015).

**F3 — Title-only, English-only, first-match categorisation.** Regex runs on the headline only, with no body and no negation ("ceasefire collapses" and "ceasefire holds" both → border_security). "first match wins" over a hand-ordered list is crude vs. a coded taxonomy (CAMEO; Goldstein 1992).

**F4 — Over-broad keywords.** `threat` (an extremely common word) sits in `diplomatic_tensions`; casualties (`killed`, `death toll`, `wounded`) route to `humanitarian_crisis` (weight 3) even when they describe combat — arguably mis-severitised relative to a coded scheme (Goldstein 1992; CAMEO).

**F5 — Ungrounded category weights.** 8/7/5/4/3/2.5/−2 are asserted, never derived or sensitivity-tested. Composite-indicator practice requires documenting the weighting scheme and testing ranking robustness to it (OECD 2008; Saisana et al. 2005). Goldstein/CAMEO give a defensible anchor.

**F6 — Inherited saturation constants.** `index = 1 + 9·(1 − e^(−eff/5·1.2))`: the `/5` and `·1.2` come from BNTI with no rationale, and the **mean**-based `eff` makes the level volume-insensitive (one airstrike headline ≈ ten). This may be a *defensible* "severity not volume" choice — but it is undocumented and untested.

**F7 — Confidence "agreement" term is mis-specified (high priority).** `A = 1 − σ_w/3` (`score.py:73-78`) measures spread of event *severity weights*, i.e. **category homogeneity, not corroboration**. A large, well-covered country with a normal mix of categories (war + diplomacy + humanitarian + neutral) gets `A ≈ 0` and is floored to confidence 0.05 — exactly what happens to **Iran (221 events, conf 0.05)**. Corroboration should be "do independent sources report the *same event*", in the spirit of WGI margins-of-error (Kaufmann et al. 2010) and V-Dem rater modelling (Pemstein et al.).

**F8 — README overstates shrinkage behaviour.** The README says shrinkage means "sparse coverage doesn't read as calm." True for an *established* country (prior = its hot 14-day mean), but **false at cold start**, where `m_prior = 0` pulls sparse new coverage *toward* calm (`score.py:58-59`). The real protection against over-reading sparse data is the (separate) confidence value. Wording should be corrected.

**F9 — Ungrounded country strategic weights.** Iran/Israel 1.5, Syria 1.4, … (`countries.yml`) are exposure judgements with no stated basis. Composite weights should be documented and sensitivity-tested (OECD 2008; Saisana et al. 2005), and could be anchored to an observable (GDP, population, military spend, oil throughput).

**F10 — Market lag-selection bias (high priority).** Choosing the lag with the largest |r| over 7 candidates (`markets.py:234-240`) inflates the reported correlation; the BH-FDR gate corrects *across instruments* but not *across the 7 lags*. Multiple-testing control must cover the lag search too (Benjamini & Hochberg 1995); better, pre-register lag 0/+1 or penalise the search.

**F11 — Dead robustness knobs.** `settings.yml` exposes `bootstrap_resamples`, `bootstrap_block`, and `bnti_prior_k`, but `markets.py` uses **none** of them — no block-bootstrap CIs and no actual shrinkage of the MENA beta toward a BNTI beta. The code implies more robustness than it delivers. Either wire them up (Künsch 1989; Politis & Romano 1994) or remove them.

**F12 — Website claims a non-existent "Translation" step (integrity).** `index.html` pipeline step 02 ("Non-English items are machine-translated") describes a translator that **does not exist** (`score.py:104` copies `translated_title = title`; no translation module). Must be corrected.

**F13 — Duplicate briefing bullets.** Live data shows the same IAEA headline twice in the briefing bullets — the de-dupe in `briefing.py` (bullet selection / LLM-polish merge) does not catch near/exact duplicates of the *same* event across the bullet list.

**F14 — Markets are frequently empty.** The live run returned `instruments: 0` / "Market data unavailable" (keyless Yahoo/FRED blocked). This is handled gracefully, but the UI should make the "no data this run" state explicit and the BNTI-seeded provenance honest (the module is association, not causation, per its own note and MacKinlay 1997).

**F15 — "Multilingual" is only true with the LLM on.** Because the keyword lexicon is English-only (F3), the **deterministic baseline effectively ignores severity in Arabic/Persian/Hebrew/Turkish headlines** (they fall to `neutral`). The whole "multilingual regional reporting" pitch holds only when `NVIDIA_API_KEY` is set. This is a coverage/selection bias (Earl et al. 2004; Weidmann 2015) and should be stated plainly — or addressed with local-language lexicons.

**Standing limitations (correct and worth keeping front-of-house):** the index measures the *volume and severity of reporting*, not ground truth; coverage is uneven across languages; machine categorisation is noisy; short-sample correlations are unstable and are association, not causation.

---

## 5. Recommended changes (tiered; deterministic; ML optional)

Decision on scope: **re-ground the model.** Model-changing items below are proposed **behind config flags with the current behaviour as default**, so the deterministic baseline and existing tests are preserved until each change is reviewed. Full mechanics are in `PATCH_PLAN.md`.

### Tier 1 — Must-fix (integrity / clear defects)
- **R1 Headline → region-led, severity-gated hybrid** (F1). §6.
- **R2 Fix the confidence "agreement" term** (F7): replace category-homogeneity with an event-corroboration measure (multi-source coverage of the same event cluster), or drop `A` and report `V·D` with a separate corroboration flag. Anchor: Kaufmann et al. (2010); Pemstein et al.
- **R3 Correct the website "Translation" claim** (F12) and the "multilingual" framing (F15).
- **R4 Disambiguate attribution** (F2): replace bare-substring country matching with word-boundary + alias disambiguation; split the `عمان` query (e.g. `"عُمان" -عمّان` or require sultanate context), and treat shared-feed matches as candidates needing a second cue.
- **R5 Market lag-selection control** (F10): apply multiple-testing control across the 7 lags (or pre-register lag 0/+1), so a "significant" beta is not just the luckiest lag.

### Tier 2 — Should-fix (soundness)
- **R6 README wording on shrinkage** (F8) and **R7 dead config knobs** (F11): either implement block-bootstrap CIs / BNTI-beta shrinkage, or delete the knobs so config matches behaviour.
- **R8 Categorisation robustness** (F3, F4): add simple negation handling, move combat-casualty phrasing toward conflict, and narrow `threat`/over-broad keywords; add an LLM abstention threshold so low-confidence LLM labels don't override the deterministic floor.
- **R9 Surface confidence in the UI** (currently computed but absent from the methodology page) and **R10 de-duplicate briefing bullets** (F13).

### Tier 3 — Optional re-grounding (clearly separated; default off)
- **R11 Literature-anchored category weights** (F5): provide an alternative weight set derived from the Goldstein/CAMEO intensity ordering, selectable via config, with both sets shipped.
- **R12 Sensitivity analysis** (OECD 2008; Saisana et al. 2005): a `scripts/sensitivity.py` that perturbs category weights, country weights, half-life, and the saturation constants and reports how composite value and country ranking move — published as a methodology appendix, not in the hot path.
- **R13 Revisit the saturation constants** (F6) and **country weights** (F9): document the current choice, expose `/5·1.2` and country weights as config, and report the sensitivity result.
- **R14 (optional, separate) local-language lexicons** to make the deterministic baseline genuinely multilingual (F15).

No recommendation requires an opaque model; the only ML is the *existing optional* LLM, which stays a refinement on top of the deterministic floor.

---

## 6. Headline & regional briefing (task #9)

**Decision: severity-gated hybrid.** Lead with the region; name a country only when it crosses a status threshold *and* clears a confidence gate; otherwise a pure regional summary.

**Why not the argmax country.** Picking the single highest of 24 noisy country scores is a textbook selection problem: the maximum order statistic is biased upward and tends to revert next period (Galton 1886, regression to the mean; Forde, Hemani & Ferguson 2023, the "Winner's Curse" — estimates chosen *because* they are the most extreme are systematically inflated). It is also confidence-blind: today it elevates Iran, whose confidence is 0.05. And it is dissonant with the composite — "leads regional risk … composite holds 2.58 (stable)" reads as alarming about a STABLE region. Composite-indicator practice is to communicate the headline number with its **drivers**, not to replace it with one component (OECD 2008).

**Proposed algorithm (deterministic):**
```
status     = band(composite)                       # STABLE / ELEVATED / CRITICAL
trend      = sign(composite − prev_composite)      # rising / easing / steady
# a country is "nameable" only if it is materially elevated AND adequately sourced
elevated_threshold = thresholds.elevated (4.0)
conf_gate          = 0.35   # tunable; excludes argmax artifacts like Iran@0.05
drivers = [c for c in countries
           if c.index >= elevated_threshold and c.confidence >= conf_gate]
drivers = sort(drivers, by = country_weight · (index − 1) · confidence)[:2]

if status == STABLE and not drivers:
    headline = f"Quiet regional window — composite {composite:.2f} (stable, {trend})."
elif drivers:
    names = join(drivers.name)
    headline = f"MENA composite {composite:.2f} ({status}, {trend}); {names} above the {status_of(drivers)} line."
else:   # elevated/critical composite but no single country clears the gate
    headline = f"MENA composite {composite:.2f} ({status}, {trend}); risk broad-based, no single country dominant."
```
The country mention is **gated by confidence**, so the Iran-style artifact is excluded; the region always leads; and a genuinely dangerous single-country situation (high index *and* high confidence) is still surfaced. The optional LLM polish must be constrained to preserve the region-first framing and the composite value (it currently may override freely).

**On today's data** (composite 2.58 STABLE; no country both ≥ 4.0 and confidence ≥ 0.35): the headline becomes *"Quiet regional window — composite 2.58 (stable, steady)."* — which is what a reader should take away.

---

## 7. Exact wording — README "Methodology (summary)" section

Drop-in replacement for the current bullet block in `README.md` (§ "Methodology (summary)"):

> ## Methodology (summary)
>
> MTI is a **news-frequency risk index**: it converts the volume and severity of regional reporting into a 1–10 reading per country and a regional composite. The approach follows the published geopolitical-risk and uncertainty-index literature (Caldara & Iacoviello 2022; Baker, Bloom & Davis 2016) and standard composite-indicator practice (OECD 2008).
>
> - **Categories → weights:** each headline is assigned one category, which maps to a fixed severity weight — military conflict 8, terrorism 7, border security 5, political instability 4, humanitarian crisis 3, diplomatic tensions 2.5, trade/de-escalation −2, neutral 0. The category ordering and fixed-weight design follow event-data severity scales (Goldstein 1992; CAMEO). Weights never change between runs (auditability).
> - **Per-country score:** `1 + 9·(1 − e^(−eff/5·1.2))`, where `eff` is a **recency-decayed** (18 h half-life), **source-credibility-weighted mean** event weight, **empirical-Bayes shrunk** toward the country's trailing baseline (Efron & Morris 1975; Kish 1965 for the effective sample size). The level reflects **severity**, not raw volume.
> - **Composite:** strategic-weighted mean of country scores, smoothed with an **asymmetric EWMA** (exponential smoothing; Hyndman & Athanasopoulos 2021) — responsive to genuine spikes, resistant to two-hour whipsaw.
> - **Confidence (0–1):** per country and composite, combining event volume (Kish effective sample size), **source diversity** (Shannon entropy), and **cross-source corroboration**, reported with each reading (cf. WGI margins of error, Kaufmann et al. 2010).
> - **Forecast:** an AR(1)/damped-Holt/persistence ladder on the smoothed series (Gardner & McKenzie 1985; Hyndman & Koehler 2006), 24 h ahead with widening 80 % predictive bands.
> - **Markets:** for each instrument, a lagged "threat beta" via the market-model event-study approach (MacKinlay 1997), gated by minimum sample, sign stability, and **Benjamini–Hochberg false-discovery control** (1995). Gold and oil signs follow the safe-haven / supply-shock literature (Baur & McDermott 2010; Kilian 2009). Association, **not** causation; launch-day stats are BNTI-seeded and badged.
> - **Headline:** the regional briefing leads with the **composite and its status**; an individual country is named only when it is materially elevated *and* adequately sourced — the index never headlines the single highest country on its own, because the maximum of many noisy series is biased high and unstable (Galton 1886; "Winner's Curse", Forde et al. 2023).
>
> Full formulas and constants live in `config/settings.yml`, the in-app **Methodology** page, and `METHODOLOGY_REVIEW.md`.

---

## 8. Exact wording — website Methodology page (`index.html`)

These map to the existing structures in `index.html`. (Implementation is in `PATCH_PLAN.md`; this is the copy.)

**8.1 Pipeline steps** — replace the `pipeline` array bodies (`index.html` ~line 1012). The key change is **deleting the false Translation step** and stating the multilingual caveat honestly:

> - **01 · Regional ingestion** — "Headlines are pulled on a two-hour cycle from RSS and Google News feeds across 24 MENA countries, in their original languages when available."
> - **02 · Attribution & categorisation** — "A deterministic keyword engine assigns each headline to one country and one threat category. When an NVIDIA API key is configured, an LLM reviews and may refine the category; English-language signals are categorised directly, and local-language coverage relies on the LLM pass."
> - **03 · Deterministic scoring** — "Each category maps to a fixed weight. Per-country and composite scores follow a closed formula, so the same inputs always produce the same index. Recency, source credibility and coverage adequacy are built in."
> - **04 · Confidence & validation** — "Every reading carries a 0–1 confidence from event volume, source diversity and corroboration. Runs that fail validation, or whose coverage is too thin, are withheld rather than published."

**8.2 Category weights note** (replace the line under "Category weights"):
> "Each headline is assigned one canonical category; each category maps to a fixed severity weight, ordered from open conflict down to de-escalation. The ordering follows event-data severity scales used in conflict research (Goldstein 1992; CAMEO). Weights never change between runs."

**8.3 Scoring formula block** — keep the formula, add one explanatory line:
> "`eff` is the recency-weighted, credibility-weighted **mean** event weight, stabilised toward the country's recent baseline. The score reflects **severity**, not how many times a story is filed."

**8.4 New "Confidence" subsection** (currently missing from the page):
> ## Confidence
> "Each country and the regional composite carry a confidence between 0 and 1. It rises with the **effective number of events**, the **diversity of sources** reporting them, and **corroboration** across independent outlets; it falls when coverage is thin, stale, or single-sourced. Confidence is shown alongside every reading so a well-evidenced score is not read the same as a tentative one."

**8.5 New "Methodology basis" block** (short, linking the design to the literature — add near Sources/Limitations):
> ## Methodology basis
> "MTI follows established practice for news-based risk indices (Caldara & Iacoviello, *Measuring Geopolitical Risk*, AER 2022; Baker, Bloom & Davis, *Measuring Economic Policy Uncertainty*, QJE 2016) and for composite indicators (OECD/EC-JRC *Handbook on Constructing Composite Indicators*, 2008). Event-severity weighting draws on conflict event-data scales (Goldstein 1992; CAMEO); the forecast uses exponential-smoothing methods (Gardner & McKenzie 1985); market reactions use the event-study market model (MacKinlay 1997) with false-discovery control (Benjamini & Hochberg 1995). Full references are in the project's `METHODOLOGY_REVIEW.md`."

**8.6 Limitations** — keep the current text and add one sentence:
> "Categorisation of non-English headlines depends on the optional language model; without it, local-language coverage contributes little to scores. Market correlations are short-sample, unstable, and indicate association, not causation."

---

## 9. References (all verified 2026-06-26)

1. **Caldara, D. & Iacoviello, M. (2022).** "Measuring Geopolitical Risk." *American Economic Review* 112(4): 1194–1225. https://doi.org/10.1257/aer.20191823 — *Canonical news-text risk index validated against macro/market outcomes; the direct MTI template.*
2. **Baker, S. R., Bloom, N. & Davis, S. J. (2016).** "Measuring Economic Policy Uncertainty." *The Quarterly Journal of Economics* 131(4): 1593–1636. https://doi.org/10.1093/qje/qjw024 — *Newspaper-frequency index with normalisation and a ~12,000-article classifier audit.*
3. **Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffmann, A. & Giovannini, E. (2008).** *Handbook on Constructing Composite Indicators: Methodology and User Guide.* OECD Publishing / EC-JRC. https://doi.org/10.1787/9789264043466-en — *Canonical normalisation, weighting, aggregation, and robustness pipeline for composite indices.*
4. **Saisana, M., Saltelli, A. & Tarantola, S. (2005).** "Uncertainty and sensitivity analysis techniques as tools for the quality assessment of composite indicators." *JRSS Series A* 168(2): 307–323. https://doi.org/10.1111/j.1467-985X.2005.00350.x — *Framework for testing how rankings respond to weighting/modelling choices.*
5. **Goldstein, J. S. (1992).** "A Conflict-Cooperation Scale for WEIS Events Data." *Journal of Conflict Resolution* 36(2): 369–385. https://doi.org/10.1177/0022002792036002007 — *Precedent for fixed numeric intensity weights on discrete event categories.*
6. **Schrodt, P. A. (2012).** *CAMEO: Conflict and Mediation Event Observations — Event and Actor Codebook (v1.1b3).* Penn State / Parus Analytics. https://eventdata.parusanalytics.com/data.dir/cameo.html — *Standardised event taxonomy and intensity coding; alternative weight anchor.*
7. **Gerner, D. J., Schrodt, P. A., Yilmaz, Ö. & Abu-Jabr, R. (2002).** "Conflict and Mediation Event Observations (CAMEO): A New Event Data Framework…" ISA/APSA working paper, Univ. of Kansas. https://www.semanticscholar.org/paper/775d7f7262ffb42972e5b87a245bc4b63c20396d — *Foundational CAMEO ontology and intensity rationale.*
8. **Gerner, D. J., Schrodt, P. A. & Yilmaz, Ö. (2008).** "CAMEO: An Event Data Framework for a Post-Cold War World." In *International Conflict Mediation*, Routledge. https://doi.org/10.4324/9780203885130 — *Peer-reviewed published record of the CAMEO framework.*
9. **Raleigh, C., Linke, A., Hegre, H. & Karlsen, J. (2010).** "Introducing ACLED: An Armed Conflict Location and Event Dataset." *Journal of Peace Research* 47(5): 651–660. https://doi.org/10.1177/0022343310378914 — *Disaggregated event coding, media sourcing, and known reporting biases.*
10. **Leetaru, K. & Schrodt, P. A. (2013).** "GDELT: Global Data on Events, Location and Tone, 1979–2012." ISA Annual Convention. http://data.gdeltproject.org/documentation/ISA.2013.GDELT.pdf — *Automated machine-coded global event extraction; sourcing-bias context.*
11. **Boschee, E., Lautenschlager, J., O'Brien, S., Shellman, S., Starz, J. & Ward, M. (2015).** "ICEWS Coded Event Data." *Harvard Dataverse.* https://doi.org/10.7910/DVN/28075 — *Canonical machine-coded political event data from ~30M multilingual reports.*
12. **Weidmann, N. B. (2015).** "On the Accuracy of Media-based Conflict Event Data." *Journal of Conflict Resolution* 59(6): 1129–1149. https://doi.org/10.1177/0022002714530431 — *Systematic location/severity inaccuracy driven by reporting density.*
13. **Earl, J., Martin, A., McCarthy, J. D. & Soule, S. A. (2004).** "The Use of Newspaper Data in the Study of Collective Action." *Annual Review of Sociology* 30: 65–80. https://doi.org/10.1146/annurev.soc.30.012703.110603 — *Defines selection and description bias in newspaper-derived event data.*
14. **Kaufmann, D., Kraay, A. & Mastruzzi, M. (2010).** "The Worldwide Governance Indicators: Methodology and Analytical Issues." *World Bank Policy Research WP 5430.* https://documents1.worldbank.org/curated/en/630421468336563314/pdf/WPS5430.pdf — *Combines heterogeneous sources with source-specific weights and reported margins of error.*
15. **Pemstein, D., Marquardt, K. L., Tzelgov, E., Wang, Y., Medzihorsky, J., Krusell, J., Miri, F. & von Römer, J.** "The V-Dem Measurement Model: Latent Variable Analysis for Cross-National and Cross-Temporal Expert-Coded Data." *V-Dem Working Paper 21*, Univ. of Gothenburg. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3595962 — *Bayesian IRT that aggregates noisy, disagreeing ratings into estimates with explicit uncertainty.*
16. **Efron, B. & Morris, C. (1975).** "Data Analysis Using Stein's Estimator and Its Generalizations." *JASA* 70(350): 311–319. https://doi.org/10.1080/01621459.1975.10479864 — *James–Stein shrinkage toward a common mean lowers MSE.*
17. **Morris, C. N. (1983).** "Parametric Empirical Bayes Inference: Theory and Applications." *JASA* 78(381): 47–55. https://doi.org/10.1080/01621459.1983.10477920 — *Formal parametric empirical-Bayes shrinkage framework.*
18. **Kish, L. (1965).** *Survey Sampling.* Wiley. (Design effect / effective sample size `n_eff = (Σw)²/Σw²`.) https://www.scirp.org/reference/referencespapers?referenceid=1259743 — *Source of the Kish effective sample size used in `score.py`.*
19. **Shannon, C. E. (1948).** "A Mathematical Theory of Communication." *Bell System Technical Journal* 27: 379–423 & 623–656. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x — *Origin of entropy `H = −Σ p log p`, the basis for the source-diversity term.*
20. **Hyndman, R. J. & Athanasopoulos, G. (2021).** *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/ — *Simple/Holt/damped exponential smoothing and ETS; basis for EWMA and damped-trend forecasts.*
21. **Gardner, E. S. Jr. (2006).** "Exponential smoothing: The state of the art — Part II." *International Journal of Forecasting* 22(4): 637–666. https://doi.org/10.1016/j.ijforecast.2006.03.005 — *Authoritative survey underpinning EWMA and damped-trend choices.*
22. **Gardner, E. S. Jr. & McKenzie, E. (1985).** "Forecasting Trends in Time Series." *Management Science* 31(10): 1237–1246. https://doi.org/10.1287/mnsc.31.10.1237 — *Foundational damped-trend exponential smoothing model.*
23. **Hyndman, R. J. & Koehler, A. B. (2006).** "Another look at measures of forecast accuracy." *International Journal of Forecasting* 22(4): 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001 — *MASE scales errors against a naïve/persistence benchmark.*
24. **MacKinlay, A. C. (1997).** "Event Studies in Economics and Finance." *Journal of Economic Literature* 35(1): 13–39. https://www.jstor.org/stable/2729691 — *Foundational market-model event-study methodology (the threat-beta approach).*
25. **Baur, D. G. & McDermott, T. K. (2010).** "Is gold a safe haven? International evidence." *Journal of Banking & Finance* 34(8): 1886–1898. https://doi.org/10.1016/j.jbankfin.2009.12.008 — *Gold as safe haven during extreme shocks; expected positive sign under threat.*
26. **Kilian, L. (2009).** "Not All Oil Price Shocks Are Alike…" *American Economic Review* 99(3): 1053–1069. https://doi.org/10.1257/aer.99.3.1053 — *Distinguishes supply/geopolitical vs. demand oil shocks; shapes oil sign expectations.*
27. **Benjamini, Y. & Hochberg, Y. (1995).** "Controlling the False Discovery Rate…" *JRSS Series B* 57(1): 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x — *The BH procedure used to control FDR across correlation tests.*
28. **Künsch, H. R. (1989).** "The Jackknife and the Bootstrap for General Stationary Observations." *The Annals of Statistics* 17(3): 1217–1241. https://doi.org/10.1214/aos/1176347265 — *Moving-block bootstrap for honest CIs on dependent data.*
29. **Politis, D. N. & Romano, J. P. (1994).** "The Stationary Bootstrap." *JASA* 89(428): 1303–1313. https://doi.org/10.1080/01621459.1994.10476870 — *Random-length block resampling for valid SEs on weakly dependent series.*
30. **Galton, F. (1886).** "Regression towards Mediocrity in Hereditary Stature." *Journal of the Anthropological Institute* 15: 246–263. https://doi.org/10.2307/2841583 — *Extreme observations regress to the mean: an argmax country tends to fall back.*
31. **Forde, A., Hemani, G. & Ferguson, J. P. (2023).** "Review and further developments in statistical corrections for Winner's Curse…" *PLOS Genetics* 19(9): e1010546. https://doi.org/10.1371/journal.pgen.1010546 — *Formalises upward inflation of estimates selected as most extreme among many — the case against an argmax headline.*

---

*Prepared as a methodology audit for the MENA Threat Index. No pipeline code was modified in producing this review; proposed code changes are in `PATCH_PLAN.md` and await approval.*
