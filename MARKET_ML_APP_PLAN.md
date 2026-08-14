# Market ML Entry — Application Plan

A NextJS/React tab in the `agent` repo that drives the `ml_analysis` ML pipeline: pull a
ticker, build features, check multicollinearity, fit XGBoost, review charts (00–03),
calibrate threshold, predict next-day entry — and save/reload fitted models with their
featuresets.

## Architecture (two repos, one ML engine)

`ml_analysis` stays the single source of truth for features/model (AGENTS.md: no second
feature stack; charts only via `plot_*`). The `agent` NextJS app gets a new top-level
**MARKET** tab that talks to a **Python sidecar** wrapping `market_ml.pipeline`.

```
agent (NextJS :3000) ──HTTP──▶ ml_analysis sidecar (FastAPI :5050) ──▶ market_ml.pipeline.run_ticker
   │                              │                                          (reuses plot_00/01/02/03)
   │                              ├─ improves_{T}_{DATE}/  (00-03 PNG + JSON/CSV)
   │                              └─ models/{T}/{id}/      (saved model + featureset)
   └─ data/briefings/*.html      (printable, surfaces in existing BRIEFINGS tab)
```

A sidecar (not shell-out) because fit + feature-health ablation is long-running (expanding
WF × per-feature retrain over ~170 OOS steps) and needs job tracking + SSE. It mirrors the
existing `agent/daemon/` (Express :4000) pattern. Runs in `ml_analysis/.venv`
(xgboost/sklearn/yfinance/matplotlib already installed).

## What gets materialized locally

| Artifact | Path | Producer |
|---|---|---|
| OHLCV caches | `ml_analysis/data/cache_ohlcv_{T}.parquet` | pipeline `load_ohlcv` |
| Run folder 00–03 + JSON/CSV | `ml_analysis/improves_{T}_{DATE}/` | `run_ticker` (`plot_*`) |
| Saved model + featureset | `ml_analysis/models/{T}/{id}/` | `market_ml/models.py` |
| Printable briefing | `agent/data/briefings/{id}.html` (+`.md`) | `agent/scripts/export-marketml-briefing-html.ts` |

`models/{T}/{id}/`:
- `xgb_clf.joblib`, `lasso_pipe.joblib`
- `featureset.json` — FEATURE_COLS, FEATURE_SPECS, BB/feature constants, `auto_invert`,
  `entry_score_min`, `chosen_alpha`, train span, source run id
- `model_card.json` — AUC raw+oriented, IC, base UP, `is_discriminating`, ablation table
- `last_pred.json` — most recent prediction through this model

`.gitignore` already covers `*.joblib`, `improves_*/`, `data/*`.

## Python sidecar (`market_ml/server.py`, FastAPI :5050)

| Endpoint | Menu item |
|---|---|
| `POST /runs {ticker, predict_target_date?, force_recalibrate?}` → `run_id` | FIT MODEL |
| `GET /runs/{id}` · `GET /runs/{id}/logs` (SSE) · `GET /runs/{id}/artifacts` | all — status/progress/artifacts |
| `GET /static/{path}` | display PNG/CSV/parquet |
| `POST /features/test {ticker, candidate}` → ΔAUC | TEST A FEATURE |
| `POST /models/save {ticker, run_id, name}` | SAVE MODEL |
| `GET /models?ticker=` · `POST /models/{id}/predict` | PREDICT (SAVED) |
| `POST /briefings/export {run_id}` | printable HTML |

- **Test feature:** reuses notebook `feature_health` ablation (full vs without-candidate
  AUC). Candidates in `market_ml/candidates.py`; gate = Δ oriented AUC ≥ +1pp. No free-form
  code exec.
- **Predict-with-saved-model (no refit):** executes canonical stages
  `setup→load_ohlcv→feature_registry→feat_*→assemble` from `_STAGE_SOURCES` (same builders
  — no second feature stack), loads saved XGB/Lasso, scores latest feature-complete session.
  `auto_invert`+`entry_score_min` from `featureset.json`. Output JSON matches a fresh run.

## Frontend (`agent`, new top-level MARKET route)

Extract a shared `<SiteNav/>` component (current links appear duplicated per-page in
`app/page.tsx`, `app/briefings/page.tsx`, `app/cor/page.tsx`). Add MARKET link; reuse on
all pages to avoid drift.

- `app/market-ml/page.tsx` (ticker picker + recent runs) · `app/market-ml/[ticker]/page.tsx` (workspace).
- Left vertical sidebar (font-mono, matches existing) — 7 items:
  **TICKER · RUNS/BRIEFINGS · FIT MODEL · PREDICT (SAVED) · TEST A FEATURE · SAVE MODEL · MODEL REGISTRY**
- `components/market-ml/{TickerPicker,RunsBriefings,FitModelForm,ChartBoard,DecisionCard,TestFeature,SaveModel,ModelRegistry,PredictSaved}.tsx`
- 00–03 PNGs served by sidecar → `<img>` (never re-plotted in JS).
  - AUC with/without each feature = `feature_ablation_oos_*.csv` / `feature_health_*.json` (ablation ΔAUC) beside 00 chart
  - thr from `02` + `entry_rule_80_*.json`
  - dir-hits from `03` + `buy_decision_*.json`

## HTML briefing export (briefings pattern)

`scripts/export-marketml-briefing-html.ts` mirrors `export-briefing-html.ts`: embed 00–03
(base64) + decision card + entry_rule + feature_health/ablation + top thr rows →
self-contained `agent/data/briefings/{ticker}-{date}-marketml.html` (+`.md` stub so it
appears in the BRIEFINGS tab). npm script `export:marketml:html`; UI button →
`POST /briefings/export`.

## Files

**`ml_analysis/`** (no edits to `pipeline.py`/`extract.py`/notebook — extract contract intact):
- `market_ml/server.py` — FastAPI app
- `market_ml/models.py` — save/load/predict with featureset
- `market_ml/candidates.py` — curated candidate feature library
- `market_ml/jobs.py` — job tracker + stage runner (returns exec globals)
- `requirements-server.txt` — fastapi + uvicorn

**`agent/`:**
- `app/market-ml/page.tsx`, `app/market-ml/[ticker]/page.tsx`
- `components/market-ml/*.tsx`
- `components/SiteNav.tsx` (shared)
- `lib/market-ml/{client,types,briefing-html}.ts`
- `scripts/export-marketml-briefing-html.ts`
- `package.json` (`ml` in `concurrently` dev; `export:marketml:html`)
- `.env.local` (`MARKET_ML_API=http://localhost:5050`)

## Constraints honored

Single feature stack · charts only via `plot_*` · walk-forward OOS only · no imputation ·
discrimination gate (precision > max(50%, base UP), n≥10) drives BUY/SKIP · force-recalibrate
exposed · saved model stores `auto_invert` + `entry_score_min`.

## Build order

1. Sidecar: `server.py`+`models.py`+`candidates.py`+`jobs.py`; smoke `POST /runs` CRM →
   verify 00–03 regenerate.
2. `models/save`+`predict`+`features/test`; confirm saved-model pred == fresh-run pred on
   same as-of.
3. `agent` shared `SiteNav` + MARKET route + FIT + RUNS/BRIEFINGS screens.
4. TEST FEATURE, SAVE, PREDICT, REGISTRY.
5. HTML briefing export; confirm BRIEFINGS tab pickup.
6. Wire `concurrently` to start web+daemon+ml sidecar on `pnpm dev`.
