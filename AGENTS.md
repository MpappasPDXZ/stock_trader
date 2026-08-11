# AGENTS.md — Market ML Analysis

Guidance for AI agents (and humans) working in this repo. **CRM is the prototype ticker**; the same notebook pipeline is reused for other symbols.

## Project in one paragraph

**One script:** `market_ml_analysis.ipynb`. It builds a **session-close feature frame**, trains **Lasso** (next log return) + **XGBoost** (next close up/down) with **walk-forward** discipline, then:

1. **Calibrates** a long-entry threshold so buy-in accuracy beats coin-flip **and** the population base UP rate.
2. **Scores a target session** (usually tomorrow / clock-aware horizon).
3. Writes charts + JSON under that run’s folder.

No second analysis script. Feature ideation and thr picking for a new stock all happen in this notebook by changing config and re-running stages.

### Charts — only via notebook functions

**Every PNG under `improves_*` must be produced by calling a `plot_*` function defined in `market_ml_analysis.ipynb`.**  
Do **not** regenerate charts from one-off agent scripts, external `.py` plotters, or ad-hoc `matplotlib` outside those functions.

| Function | Artifact |
|----------|----------|
| `plot_00_candles_feature_health()` | `00_candles_feature_health_{TICKER}.png` (only 00 board; no aliases) |
| `plot_01_features_outcome_bb()` | `01_features_outcome_bb_{TICKER}.png` |
| `plot_02_entry_calibration()` | `02_entry_calibration_{TICKER}.png` |
| `plot_03_dir_hits_lastn()` | `03_dir_hits_last{N}_{TICKER}.png` (buy-rule thr; preferred) |
| `plot_03_dir_hits_lastn_val()` | same `03_…` path from last-N WF (pre-cal thr) |

To refresh a board: re-run the notebook stage that **calls** the function (or call it in a notebook cell after its deps exist).

---

## Repo layout

```
market_ml_analysis.ipynb     # only pipeline / ideation surface
data/                        # OHLCV downloads & parquet caches (any ticker)
  cache_ohlcv_{TICKER}.parquet
  cache_ohlcv_hist_{TICKER}.parquet
improves_{TICKER}_{DATE}/    # one run’s outputs (DATE = PREDICT_TARGET_DATE)
  00_candles_feature_health_….png
  01_features_outcome_bb_….png
  02_entry_calibration_….png
  03_dir_hits_last*_….png
  entry_rule_80_….json
  buy_decision_{DATE}_….json
  pred_{DATE}_….json
  …
FEATURE_MATRIX.md            # paste-ready feature research brief
AGENTS.md                    # this file
```

| Path | Role |
|------|------|
| `data/` | **Only** place for market data downloads / refreshable caches |
| `improves_{TICKER}_{DATE}/` | Charts, thr scan, WF panels, buy/pred JSON for that stock + target session |
| `market_ml_analysis.ipynb` | Setup → features → assemble → WF → **calibrate thr** → buy rule |

Setup creates:

- `DATA_DIR = data/`
- `OUTPUT_DIR = improves_{TICKER}_{RUN_DATE}/` with `RUN_DATE = PREDICT_TARGET_DATE`

Stale WF panels **lie** after feature/ticker changes — delete `improves_…/wf_dir_panel_{TICKER}.parquet` (or the whole run folder) and regenerate.

---

## Canonical pipeline order

Run top-to-bottom when features or ticker change:

| Stage | What it does |
|-------|----------------|
| Config + imports | `TICKER`, dates, buy thr, feature windows; sets `DATA_DIR` + `OUTPUT_DIR` |
| Load OHLCV | Fetch/cache under **`data/cache_ohlcv_{TICKER}.parquet`** (+ hist warm-up) |
| Features | Registry → builders → `bars[col]` → hand-check on validate date |
| Assemble | `FEATURE_COLS` + `y_reg` / `y_cls` → `model_df` |
| Feature board | `OUTPUT_DIR/01_features_outcome_bb_{TICKER}.png` |
| Last-N WF + pending pred | `val_panel`, `pred_target`, `pred_{DATE}_{TICKER}.json` |
| **Entry calibration** | Orient score, thr scan, discrimination gate → thr / invert |
| **Buy rule + dir hits** | Apply thr; `03_dir_hits_…png`, `buy_decision_…json` |
| Optional full WF / live / persist | Secondary; day-to-day decisions use calibration + last-N |

Read/write under the run folder:

- `entry_rule_80_{TICKER}.json` — thr, `auto_invert`, `is_discriminating`, pending  
- `buy_decision_{PREDICT_TARGET_DATE}_{TICKER}.json` — BUY/SKIP + rationale  
- `pred_{PREDICT_TARGET_DATE}_{TICKER}.json` — raw model outputs  

---

## Common ask 1 — “Try this feature”

### Goal
Add a candidate column to **X**, keep labels fixed, re-evaluate discrimination + thr.

### Contract
- **No look-ahead:** only data known at session close of feature as-of day.
- **Name match:** `bars["name"]` == `FEATURE_SPECS["name"]` == entry in `FEATURE_COLS`.
- Labels (do not change unless explicitly asked):
  - `y_cls` — 1 if next session close > this session close  
  - `y_reg` — log(next close / this close)

### Checklist
1. **Config (setup cell):** lookback constant if needed; fold into `FEATURE_MAX_SESSIONS` so warm-up (`PRE_START_DAYS`) still covers it.
2. **Registry cell:** append to `FEATURE_SPECS`:
   - `name`, `sessions_before_start` (honest trading-session buffer), `title`
3. **Builder cell:** pure function → assign `bars["name"]` → hand-recompute on `FEATURE_VALIDATE_TS` → assert ≈ match → print `PASS name`.
4. **Assemble:** re-run so `model_df[FEATURE_COLS]` includes the column (NaNs at validate date → increase buffer).
5. **Optional board:** `FEATURE_ZONES` if oversold/overbought bands help the multi-panel chart.
6. **Re-run from assemble down:** last-N WF → **delete or ignore stale** `wf_dir_panel_*.parquet` if calibration would short-circuit → entry calibration → buy rule / dir-hits. Artifacts land in `improves_{TICKER}_{DATE}/`.
7. **Judge success** on `02_entry_calibration_{TICKER}.png` + `entry_rule_80_{TICKER}.json`:
   - Does some thr with `n_buys ≥ min_trades` get buy-in precision **> max(50%, base UP rate)**?
   - Net **% gain captured** improved?
   - Did **auto-invert** flip?

### Do not
- Hard-code feature names into train loops (use `FEATURE_COLS`).
- Leave stub “feature dropped” cells as the pattern for new work — add real builders only.
- Ship a thr that never clears discrimination floors.
- Write new parallel `.py` / second notebook for experiments — use this notebook.

---

## Common ask 2 — “Run this ticker & calibrate threshold”

### Goal
Swap prototype CRM for e.g. `AAPL`, full refresh, pick operating long thr.

### Checklist
1. Set **`TICKER = "…"`** in the setup cell.
2. Set **`FEATURE_VALIDATE_DATE`** and **`PREDICT_TARGET_DATE`** to sensible sessions on the new symbol’s calendar (creates `improves_{TICKER}_{PREDICT_TARGET_DATE}/`).
3. Full forward run: load → features → assemble → board → last-N WF → **entry calibration** → buy rule.
   - Fresh download caches go to **`data/`** (shared across runs for that ticker).
4. Force fresh calibration when needed:
   - Remove `improves_…/wf_dir_panel_{TICKER}.parquet` if wrong features; rebuild from current `model_df` + `FEATURE_COLS`.
5. Read results in that run folder:
   - **`02_entry_calibration_{TICKER}.png`**
   - **`entry_rule_80_{TICKER}.json`**
     - `is_discriminating`
     - `auto_invert` / `entry_score_definition`
     - `best_rule.entry_score_min`
     - `pending.enter_under_best_rule`
6. Sync config thr: set `BUY_ENTRY_SCORE_MIN` from calibrated `entry_score_min` when discrimination **passes**; if it **fails**, stay flat / SKIP.

### Discrimination gate (hard)
- Buy-in precision = TP / (TP+FP) among days the rule **buys**.
- Eligible only if precision **> 50%**, **> base UP rate**, and `n_buys ≥ MIN_TRADES` (typically 10).
- Prefer eligible thr with highest precision (then n_wins, then n_buys).
- If none eligible: **`is_discriminating: false`** → no tradeable thr.

### Entry score orientation
- If OOS Spearman(IC) of raw `xgb_prob_up` vs `y_up` is **negative**, set **`auto_invert=true`** and  
  `entry_score = 1 - xgb_prob_up`  
  so **higher score still means stronger long signal**.
- Always report both **raw P(up)** and **entry_score** in decisions.

---

## Common ask 3 — “Display prediction for tomorrow”

### Goal
Show clear BUY/SKIP for the next target session with numbers a human can trust.

### Always pair decision with `03_dir_hits`
Every prediction response **must** include both:

1. **Text decision card** (JSON-backed numbers), and  
2. **`improves_{TICKER}_{DATE}/03_dir_hits_last*_{TICKER}.png`**

Do **not** answer with JSON-only or chart-only. If `03_…` is missing or stale vs `buy_decision` / `PREDICT_TARGET_DATE`, **regenerate** before displaying.

| Piece | Source |
|-------|--------|
| Pending bar + candles | `improves_…/03_dir_hits_last*_{TICKER}.png` |
| Decision, thr, invert | `improves_…/buy_decision_{DATE}_{TICKER}.json` |
| Raw model fields | `improves_…/pred_{DATE}_{TICKER}.json` |
| Thr / discrimination | `improves_…/entry_rule_80_{TICKER}.json`, `02_entry_calibration_….png` |

### Checklist
1. Horizon (America/Chicago): before 15:00 → current-day vs prior; at/after 15:00 → next session vs latest close.
2. Pipeline run for **`TICKER`** with fresh pred, calibration, and dir-hits for that as-of.
3. Load artifacts from **`OUTPUT_DIR = improves_{TICKER}_{PREDICT_TARGET_DATE}/`**.
4. **Display order:** (1) `03_dir_hits…` image (2) decision card with raw P, entry_score, thr, BUY/SKIP.
5. If `is_discriminating` is false → **SKIP**.

### Example response shape
```
[display improves_CRM_2026-08-11/03_dir_hits_last30_CRM.png]

CRM | target 2026-08-11 | features as-of 2026-08-10 close=197.51
Raw XGB P(up)=64.5% → entry_score=35.5% (auto-invert)
Thr entry_score ≥ 84% | OOS buy-in 80% | discriminating=yes
Decision: SKIP (score 35.5% < 84%)
```

---

## Key outputs (by ticker + date folder)

| File (under `improves_{TICKER}_{DATE}/`) | Role |
|------|------|
| `00_candles_feature_health_{TICKER}.png` | Candles/BB + Spearman + OOS lift (`plot_00_…`) |
| `01_features_outcome_bb_{TICKER}.png` | Candles/BB + feature strip + outcomes |
| `02_entry_calibration_{TICKER}.png` | Thr scan, floors, net gain capture |
| `03_dir_hits_last{N}_{TICKER}.png` | Candles + entry-score bars + actuals / pending |
| `entry_rule_80_{TICKER}.json` | Calibration truth for thr / invert / pending |
| `entry_threshold_scan_{TICKER}.csv` | Full thr grid |
| `wf_dir_panel_{TICKER}.parquet` | Expanding OOS panel used in calibration |
| `pred_{DATE}_{TICKER}.json` | Model outputs for target date |
| `buy_decision_{DATE}_{TICKER}.json` | BUY/SKIP decision object |
| `buy_panel_last{N}_{TICKER}.parquet` | Last-N buy outcomes |

Caches (under **`data/`** only):

| File | Role |
|------|------|
| `cache_ohlcv_{TICKER}.parquet` | Analysis-window OHLCV |
| `cache_ohlcv_hist_{TICKER}.parquet` | Warm-up + analysis history |

---

## Modeling rules of the road

- **Walk-forward only** for OOS claims.
- **Score only BUY days** for win rate / buy-in accuracy; SKIP never enters the denominator.
- **Gain captured:** net = Σ_TP r + Σ_FP r over total UP-day gains (`r = close_next/close_feature − 1`).
- Prefer editing the **notebook cells that own a stage** — do not invent parallel scripts.
- Charts: call the owning **`plot_*` function** in-notebook (or re-run that stage). No external plotters.

### Feature lookback (no imputation)

- Each `FEATURE_SPECS` entry has `sessions_before_start` (trading sessions needed at as-of day \(t\)).
- `FEATURE_MAX_SESSIONS = max(sessions_before_start)` = largest honest buffer for a **feature-complete** row.
- A session is kept only if **every** `FEATURE_COLS` value is finite (known at that session’s close).
- **Never impute / ffill / zero-fill** features. Incomplete rows are **dropped**.
- Short history (IPO / thin Yahoo span): effective analysis start slides to the first complete session after config `START`; pipeline prints `EFFECTIVE_FEATURE_START` and counts dropped.

---

## Feature system (quick)

```
FEATURE_SPECS → FEATURE_COLS
bars[name] → feature builders + PASS assert on FEATURE_VALIDATE_DATE
assemble → model_df[FEATURE_COLS], X, y_cls, y_reg
all later train steps use FEATURE_COLS only
```

---

## Config knobs (setup cell)

| Name | Typical use |
|------|-------------|
| `TICKER` | Symbol (CRM prototype) |
| `START` / `PRE_START_DAYS` | Analysis window + warm-up fetch; `PRE_START_DAYS = ceil((FEATURE_MAX_SESSIONS+5)×365.25/252)` |
| `FEATURE_VALIDATE_DATE` | Hand-check board date |
| `VAL_WINDOW_SESSIONS` | Last-N WF for dir-hits (often 30) |
| `PREDICT_TARGET_DATE` | Session to score (**also** sets `improves_*` folder date) |
| `BUY_ENTRY_SCORE_MIN` | Operating thr on **entry score** (prefer calibrated) |
| `BUY_REQUIRE_LASSO_UP` | Optional Lasso up filter |
| `FEAT_*` | Feature window lengths |
| `DATA_DIR` / `OUTPUT_DIR` | Set automatically from ticker + date |

---

## Anti-patterns

- Dumping caches into `improves_*` or chart outputs into `data/`.
- **Writing/overwriting run-folder PNGs outside a notebook `plot_*` function** (no external plot scripts, no agent-only savefig).
- Running buy decision from an **old** `entry_rule_80` after changing ticker/features without re-calibrating.
- Treating raw `P(up) ≥ 0.5` as a trade when calibration uses **oriented** score + high thr.
- Celebrating high thr precision on **n &lt; min_trades**.
- Adding a second notebook/script for “experiments” instead of iterating in `market_ml_analysis.ipynb`.

---

## When the user says…

| Ask | Agent action |
|-----|----------------|
| Try feature X | Registry + builder in the one notebook; re-run assemble→calibrate→decide into `improves_{TICKER}_{DATE}/` |
| Run ticker Y | Set `TICKER` + dates; full pipeline; caches in `data/`; deliver `02` + thr + pending |
| Tomorrow’s prediction | Refresh if needed; **always show `03_dir_hits` + decision card** from that run folder |
| Tidy notebook | Keep registry contract and calibration gates; keep single-notebook contract |

When in doubt: **regenerate calibration + `03_dir_hits`**, then state BUY/SKIP from entry score vs thr (with raw P), **always with the `03_…` chart**.
