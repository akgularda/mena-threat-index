# MENA Threat Index (MTI)

**An open, repeatable geopolitical-risk index for the Middle East & North Africa.**

MTI turns the region's news flow into a single 1–10 threat reading per country and a regional composite, refreshed every two hours. It forecasts the index forward, and correlates it with global markets (oil, gold, FX, defense equities) to estimate **how markets react to regional risk** — and to project market moves from the forecasted index path.

It is the successor to the **Border Neighbor Threat Index (BNTI)**, expanded from Türkiye's seven neighbours to all of MENA, with a substantially improved methodology. The former BNTI history is used to seed market correlations at launch.

> Situational-awareness aid — **not** an intelligence assessment. See *Limitations* below.

---

## How it works

```
RSS / Google News (24 countries, multilingual)
        │  feeds.py        fetch · dedupe · 72h window
        ▼
   categorize.py           keyword engine (+ optional NVIDIA LLM)
        ▼
     score.py              recency-decay · source credibility · volume shrinkage
        │                  per-country index  = 1 + 9·(1 − e^(−raw/5·1.2))
        │                  composite          = strategic-weighted mean, EWMA spike-guard
        ▼
   history.py              append reading to data/history.jsonl  (persistent, growing)
        ▼
   forecast.py             AR(1)-with-drift (cold-start ladder) → 24h path + bands
        ▼
   markets.py              Yahoo + FRED prices · lagged correlation · OLS "threat beta"
        │                  · BNTI-seeded · project market moves from forecast path
        ▼
   briefing.py             templated regional summary (+ optional LLM polish)
        ▼
   publish.py              assemble + validate → mena_data.json  (atomic, withhold-on-failure)
```

The frontend (`index.html` + `support.js`) is a static single-page app that fetches `mena_data.json` and renders the map, country detail, trend + forecast, the **Markets** screen, and the methodology. `support.js` is a self-contained React-based template runtime; no build step.

## Repository layout

| Path | What |
|---|---|
| `index.html` | The web app (served by GitHub Pages). |
| `support.js` | Template runtime (unchanged from BNTI). |
| `mena_data.json` | Latest published snapshot — **committed by the pipeline**. |
| `config/*.yml` | Countries + feeds, market instruments, category lexicon, settings. |
| `pipeline/*.py` | The data pipeline (run with `python -m pipeline.run`). |
| `data/history.jsonl` | Append-only composite history (source of truth). |
| `data/countries/*.jsonl` | Per-country history. |
| `data/markets/instruments.jsonl` | Per-run market snapshots. |
| `data/seed/` | Former BNTI history, used to seed correlations. |
| `scripts/seed_from_bnti.py` | One-time history seeding. |
| `.github/workflows/` | Scheduled pipeline + Pages deploy. |
| `tests/` | `pytest` suite. |

## Running locally

```bash
pip install -r requirements.txt

# Deterministic run (keyword categorizer, no LLM):
python -m pipeline.run

# With LLM categorization + briefing polish (NVIDIA, OpenAI-compatible API):
export NVIDIA_API_KEY=nvapi-xxxxxxxx        # Windows PowerShell: $env:NVIDIA_API_KEY="nvapi-..."
python -m pipeline.run

# Serve the site:
python -m http.server 8000      # then open http://localhost:8000/
```

`python -m pipeline.run` writes/refreshes `mena_data.json`, appends to `data/`, and **withholds** the update (keeps the last-good file) if the run fails validation or feed coverage is too low.

Run the tests with `pytest`.

## Deploying (GitHub Pages + Actions)

1. Create a **public** repo `akgularda/mena-threat-index` and push this folder.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. **Settings → Secrets and variables → Actions → New repository secret**: add `NVIDIA_API_KEY` (optional — the pipeline runs fully without it, using the keyword categorizer).
4. The `pipeline` workflow runs every 2 hours (cron), recomputes the index, commits `mena_data.json` + `data/`, and deploys Pages. You can also trigger it manually via **Actions → MTI Pipeline → Run workflow**.

> 🔐 **Security:** the NVIDIA key lives **only** as a GitHub secret and is read from `NVIDIA_API_KEY` at runtime. It is never written to any file in this repo. If a key was ever shared in plaintext, rotate it at <https://build.nvidia.com>.

## Methodology (summary)

MTI is a **news-frequency risk index**: it converts the volume and severity of regional reporting into a 1–10 reading per country and a regional composite. The approach follows the published geopolitical-risk and uncertainty-index literature (Caldara & Iacoviello 2022; Baker, Bloom & Davis 2016) and standard composite-indicator practice (OECD 2008).

- **Categories → weights:** each headline is assigned one category, which maps to a fixed severity weight — military conflict 8, terrorism 7, border security 5, political instability 4, humanitarian crisis 3, diplomatic tensions 2.5, trade/de-escalation −2, neutral 0. The ordering and fixed-weight design follow event-data severity scales (Goldstein 1992; CAMEO). Weights never change between runs (auditability).
- **Per-country score:** `1 + 9·(1 − e^(−eff/5·1.2))`, where `eff` is a **recency-decayed** (18 h half-life), **source-credibility-weighted mean** event weight, **empirical-Bayes shrunk** toward the country's trailing baseline (Efron & Morris 1975; Kish 1965 for the effective sample size). The level reflects **severity**, not raw volume.
- **Composite:** strategic-weighted mean of country scores, smoothed with an **asymmetric EWMA** (exponential smoothing; Hyndman & Athanasopoulos 2021) — responsive to genuine spikes, resistant to two-hour whipsaw.
- **Confidence (0–1):** per country and composite, combining event volume (Kish effective sample size), **source diversity** (Shannon entropy), and **cross-source corroboration**, reported with each reading (cf. WGI margins of error, Kaufmann et al. 2010).
- **Forecast:** an AR(1)/damped-Holt/persistence ladder on the smoothed series (Gardner & McKenzie 1985; Hyndman & Koehler 2006), 24 h ahead with widening 80 % predictive bands.
- **Markets:** for each instrument, a lagged "threat beta" via the market-model event-study approach (MacKinlay 1997), gated by minimum sample, sign stability, a lag-search correction, and **Benjamini–Hochberg false-discovery control** (1995). Gold and oil signs follow the safe-haven / supply-shock literature (Baur & McDermott 2010; Kilian 2009). Association, **not** causation; launch-day stats are **seeded from BNTI** and badged as such.
- **Headline:** the regional briefing leads with the **composite and its status**; an individual country is named only when it is materially elevated *and* adequately sourced — the index never headlines the single highest country on its own, because the maximum of many noisy series is biased high and unstable (Galton 1886; "Winner's Curse", Forde et al. 2023).

Full formulas, constants, and references live in `config/settings.yml`, the in-app **Methodology** page, and [`METHODOLOGY_REVIEW.md`](METHODOLOGY_REVIEW.md).

## Limitations

The index reflects the **volume and severity of published reporting**, not ground truth. Coverage is uneven across languages; machine categorization introduces noise; correlations on short samples are unstable and are association, **not** causation. Runs that fail validation are withheld rather than published. Treat MTI as a monitoring aid.

---

Built and maintained by **Monarch Castle Technologies** · Apache-2.0 · 2026
