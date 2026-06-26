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

- **Categories → weights:** military conflict 8, terrorism 7, border security 5, political instability 4, humanitarian crisis 3, diplomatic tensions 2.5, trade/de-escalation −2, neutral 0.
- **Per-country score:** `1 + 9·(1 − e^(−raw/5·1.2))`, where `raw` is a **recency-decayed, source-credibility-weighted** mean event weight, **shrunk toward a country baseline** so sparse coverage doesn't read as calm.
- **Composite:** strategic-weighted mean of country scores, smoothed with an **asymmetric EWMA** (responsive to genuine spikes, resistant to two-hour whipsaw).
- **Confidence (0–1):** per country and composite, from event volume, source diversity, and corroboration.
- **Forecast:** AR(1)-with-drift on the smoothed series (cold-start ladder when data is thin), 24h ahead with widening predictive bands.
- **Markets:** for each instrument, lagged cross-correlation and an OLS "threat beta" (return per +1 index point), with multiple-comparison control and minimum-sample gates; market moves are projected from the forecasted index path. Launch-day stats are **seeded from BNTI** and badged as such.

Full formulas and constants live in `config/settings.yml` and the in-app **Methodology** page.

## Limitations

The index reflects the **volume and severity of published reporting**, not ground truth. Coverage is uneven across languages; machine categorization introduces noise; correlations on short samples are unstable and are association, **not** causation. Runs that fail validation are withheld rather than published. Treat MTI as a monitoring aid.

---

Built and maintained by **Monarch Castle Technologies** · Apache-2.0 · 2026
