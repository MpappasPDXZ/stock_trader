"""FastAPI sidecar for the market-ML pipeline.

Runs the canonical notebook-extracted stages (via ``market_ml.jobs.run_stages``)
for any ticker, serves run artifacts (charts, JSON, CSV), manages saved models,
and tests candidate features — all from the same feature stack as the notebook.

Run:
    .venv/bin/python -m market_ml.server
    # or
    .venv/bin/uvicorn market_ml.server:app --host 127.0.0.1 --port 5050
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from market_ml.pipeline import RunConfig, _STAGE_ORDER
from market_ml.jobs import JOBS, run_stages
from market_ml import models as M
from market_ml import candidates as C
from market_ml import calibration as CAL

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Market ML Sidecar", version="0.1.0")


# --------------------------------------------------------------------------- #
# Preflight — check if a ticker has enough history for the pipeline
# --------------------------------------------------------------------------- #
# near_high uses min_periods=1 (adaptive), so the 200-session buffer is no
# longer a hard gate: the pipeline runs with whatever history exists. The
# binding floor is now MIN_TRAIN_ROWS (walk-forward needs >65 labeled rows).
# FEATURE_MAX_SESSIONS (200) + PAD is still reported as `buffer_needed` for
# the fully-featured quality flag, but `executable` reflects the real floor.

NYSE_SESSIONS_PER_YEAR = 252.0
CAL_DAYS_PER_YEAR = 365.25
FEATURE_MAX_SESSIONS = 200
PRE_START_SESSION_PAD = 5
PRE_START_DAYS = int(
    (FEATURE_MAX_SESSIONS + PRE_START_SESSION_PAD) * CAL_DAYS_PER_YEAR / NYSE_SESSIONS_PER_YEAR
)  # ≈ 298

# Pipeline floor: walk-forward raises if labeled rows <= MIN_TRAIN_ROWS + 5.
# With adaptive near_high, the binding feature lookback is ~21 (vol_ratio),
# so ~22 sessions cover feature completeness; >65 labeled rows is the real
# gate. We require a small cushion above it.
MIN_EXEC_SESSIONS = 66


def _preflight(ticker: str) -> dict[str, Any]:
    """Check if a ticker has enough history to run the pipeline.

    Returns a dict with:
      - executable: bool
      - sessions_available: total trading sessions from yfinance
      - first_session: date of first available bar
      - last_session: date of last available bar
      - sessions_needed: FEATURE_MAX_SESSIONS + PAD
      - start_date: the pipeline's START date (1 year ago)
      - buffer_sessions: sessions available before START
      - buffer_needed: sessions needed before START
      - days_until_executable: calendar days from today until enough buffer exists
      - estimated_executable_date: the date when the ticker will become runnable
      - days_since_first_session: calendar days since IPO/first bar
    """
    import pandas as pd
    import numpy as np
    import yfinance as yf

    t = ticker.upper().strip()
    START = (pd.Timestamp.today().normalize() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    START_TS = pd.Timestamp(START)
    FETCH_START = (START_TS - pd.Timedelta(days=PRE_START_DAYS)).strftime("%Y-%m-%d")

    sessions_needed = FEATURE_MAX_SESSIONS + PRE_START_SESSION_PAD

    try:
        raw = yf.download(t, start=FETCH_START, auto_adjust=True, progress=False)
    except Exception as e:
        return {
            "ticker": t,
            "executable": False,
            "error": f"yfinance fetch failed: {e}",
            "sessions_available": 0,
            "sessions_needed": sessions_needed,
            "buffer_sessions": 0,
            "buffer_needed": sessions_needed,
            "days_until_executable": None,
        }

    if raw is None or raw.empty:
        # Ticker might not exist yet, or might be too new for even FETCH_START.
        # Try a max-history fetch to see if it exists at all.
        try:
            raw_max = yf.download(t, period="max", auto_adjust=True, progress=False)
        except Exception:
            raw_max = None
        if raw_max is None or raw_max.empty:
            return {
                "ticker": t,
                "executable": False,
                "error": "No data from yfinance — ticker may not exist",
                "sessions_available": 0,
                "sessions_needed": sessions_needed,
                "buffer_sessions": 0,
                "buffer_needed": sessions_needed,
                "days_until_executable": None,
            }
        raw = raw_max

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=["Close"]).sort_index()

    sessions_available = len(df)
    first_session = df.index.min()
    last_session = df.index.max()
    buffer_sessions = int((df.index < START_TS).sum())

    # Executable threshold: the pipeline runs once it has enough labeled,
    # feature-complete rows to clear MIN_TRAIN_ROWS (>65). near_high is
    # adaptive (min_periods=1), so the 200-session buffer is no longer a
    # hard gate — the binding floor is MIN_EXEC_SESSIONS total sessions.
    #
    # We report two thresholds:
    #   - executable: sessions_available >= MIN_EXEC_SESSIONS (pipeline runs)
    #   - fully_featured: buffer_sessions >= sessions_needed (full 200-session
    #     near_high window + all other features at their canonical length)
    if sessions_available >= MIN_EXEC_SESSIONS:
        return {
            "ticker": t,
            "executable": True,
            "fully_featured": buffer_sessions >= sessions_needed,
            "sessions_available": sessions_available,
            "first_session": first_session.strftime("%Y-%m-%d"),
            "last_session": last_session.strftime("%Y-%m-%d"),
            "sessions_needed": sessions_needed,
            "start_date": START,
            "buffer_sessions": buffer_sessions,
            "buffer_needed": sessions_needed,
            "days_until_executable": 0,
            "estimated_executable_date": None,
            "days_since_first_session": (pd.Timestamp.today().normalize() - first_session).days,
            "pre_start_days": PRE_START_DAYS,
            "feature_max_sessions": FEATURE_MAX_SESSIONS,
        }

    # Not enough history to run yet. The deficit is against MIN_EXEC_SESSIONS
    # (the real floor), not the 205 fully-featured target. Convert the session
    # deficit to calendar days (252 sessions ≈ 365 calendar days).
    sessions_deficit = MIN_EXEC_SESSIONS - sessions_available
    days_until = int(np.ceil(sessions_deficit * CAL_DAYS_PER_YEAR / NYSE_SESSIONS_PER_YEAR))
    days_until = max(0, days_until)

    # Estimate the calendar date when it becomes executable
    est_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=days_until)

    return {
        "ticker": t,
        "executable": False,
        "sessions_available": sessions_available,
        "first_session": first_session.strftime("%Y-%m-%d"),
        "last_session": last_session.strftime("%Y-%m-%d"),
        "sessions_needed": sessions_needed,
        "start_date": START,
        "buffer_sessions": buffer_sessions,
        "buffer_needed": sessions_needed,
        "days_until_executable": days_until,
        "estimated_executable_date": est_date.strftime("%Y-%m-%d") if days_until > 0 else None,
        "days_since_first_session": (pd.Timestamp.today().normalize() - first_session).days,
        "pre_start_days": PRE_START_DAYS,
        "feature_max_sessions": FEATURE_MAX_SESSIONS,
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run_dirs_for_ticker(ticker: str) -> list[dict[str, Any]]:
    """List ``runs/improves_{TICKER}_*`` folders sorted by date desc.

    Each entry includes ``decision`` ("BUY"/"SKIP"/None) read from the
    ``buy_decision_{date}_{TICKER}.json`` artifact when present.
    """
    import glob
    t = ticker.upper()
    pattern = str(REPO_ROOT / "runs" / f"improves_{t}_*")
    dirs = sorted(
        (p for p in glob.glob(pattern) if Path(p).is_dir()),
        key=lambda p: Path(p).name,
        reverse=True,
    )
    out = []
    for d in dirs:
        p = Path(d)
        name = p.name
        date = name.rsplit("_", 1)[-1] if "_" in name else name
        entry: dict[str, Any] = {"folder": name, "date": date, "path": str(p)}
        bd_path = p / f"buy_decision_{date}_{t}.json"
        if bd_path.exists():
            try:
                bd = json.loads(bd_path.read_text())
                entry["decision"] = bd.get("decision")
            except Exception:
                entry["decision"] = None
        else:
            entry["decision"] = None
        out.append(entry)
    return out


def _run_dir(ticker: str, date: str) -> Path:
    d = REPO_ROOT / "runs" / f"improves_{ticker.upper()}_{date}"
    if not d.is_dir():
        raise HTTPException(404, f"run folder not found: runs/improves_{ticker.upper()}_{date}")
    return d


def _list_artifacts(ticker: str, date: str) -> dict[str, Any]:
    d = _run_dir(ticker, date)
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    charts = [f for f in files if f.endswith(".png")]
    jsons = [f for f in files if f.endswith(".json")]
    csvs = [f for f in files if f.endswith(".csv")]
    parquets = [f for f in files if f.endswith(".parquet")]

    # Load key JSONs for convenience
    loaded: dict[str, Any] = {}
    for fn in jsons:
        try:
            loaded[fn] = json.loads((d / fn).read_text())
        except Exception:
            pass

    return {
        "ticker": ticker.upper(),
        "date": date,
        "folder": d.name,
        "files": files,
        "charts": charts,
        "jsons": jsons,
        "csvs": csvs,
        "parquets": parquets,
        "loaded": loaded,
    }


def _summary_from_globals(g: dict[str, Any]) -> dict[str, Any]:
    """Extract a run summary from exec globals (mirrors run_ticker's summary)."""
    t = g.get("TICKER", "?")
    out = g.get("OUTPUT_DIR")
    ptd = g.get("PREDICT_TARGET_DATE", "?")
    pred_path = out / f"pred_{ptd}_{t}.json" if out else None
    buy_path = out / f"buy_decision_{ptd}_{t}.json" if out else None
    rule_path = out / f"entry_rule_80_{t}.json" if out else None

    summary: dict[str, Any] = {
        "ok": True,
        "ticker": t,
        "predict_target_date": ptd,
        "output_dir": str(out) if out else None,
        "feature_cols": list(g.get("FEATURE_COLS", [])),
        "stages_done": [s for s in _STAGE_ORDER],
        "pred_path": str(pred_path) if pred_path and pred_path.exists() else None,
        "buy_path": str(buy_path) if buy_path and buy_path.exists() else None,
        "rule_path": str(rule_path) if rule_path and rule_path.exists() else None,
    }
    if rule_path and rule_path.exists():
        rule = json.loads(rule_path.read_text())
        summary["is_discriminating"] = rule.get("is_discriminating")
        summary["auto_invert"] = rule.get("auto_invert")
        summary["operating_entry_score_min"] = rule.get("operating_entry_score_min")
        summary["oriented_auc"] = rule.get("oriented_auc")
        summary["raw_xgb_auc"] = rule.get("raw_xgb_auc")
        pend = rule.get("pending") or {}
        summary["enter_under_best_rule"] = pend.get("enter_under_best_rule")
        summary["entry_score"] = pend.get("entry_score")
        summary["entry_rule"] = rule
    if buy_path and buy_path.exists():
        buy = json.loads(buy_path.read_text())
        summary["decision"] = buy.get("decision")
        summary["buy_decision"] = buy
    if pred_path and pred_path.exists():
        summary["pred"] = json.loads(pred_path.read_text())
    return summary


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    ticker: str
    predict_target_date: str | None = None
    feature_validate_date: str | None = None
    val_window_sessions: int = 30
    buy_entry_thr_mode: str = "composite"
    buy_entry_score_min: float = 0.72
    buy_require_lasso_up: bool = False
    force_recalibrate: bool = False
    skip_feature_health: bool = False
    skip_plot_01: bool = False
    n_jobs_model: int = 2


class SaveModelRequest(BaseModel):
    ticker: str
    run_id: str
    name: str


class CandidateTestRequest(BaseModel):
    ticker: str
    candidate_name: str
    predict_target_date: str | None = None


class PredictRequest(BaseModel):
    predict_target_date: str | None = None


class LivePredictRequest(BaseModel):
    ticker: str
    target_date: str | None = None
    override_close: float | None = None


class CalibrateRequest(BaseModel):
    predict_target_date: str | None = None
    val_window_sessions: int = 60
    name: str | None = None
    n_jobs_model: int = 2
    features: list[str] | None = None


class UpdateFeaturesRequest(BaseModel):
    features: list[str]


class PredictRequest(BaseModel):
    predict_target_date: str | None = None
    override_close: float | None = None
    model_id: str | None = None


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "market-ml-sidecar",
        "repo_root": str(REPO_ROOT),
        "models_dir": str(M.MODELS_DIR),
        "models_dir_exists": M.MODELS_DIR.exists(),
        "n_candidates": len(C.CANDIDATES),
        "queue_len": len([j for j in JOBS.all() if j.status in ("queued", "running")]),
    }


# --------------------------------------------------------------------------- #
# Preflight — check if a ticker has enough history
# --------------------------------------------------------------------------- #
@app.get("/preflight/{ticker}")
async def preflight(ticker: str) -> dict[str, Any]:
    """Check if a ticker has enough history for the pipeline."""
    return _preflight(ticker)


# --------------------------------------------------------------------------- #
# Bars — OHLCV history for candlestick charts
# --------------------------------------------------------------------------- #
@app.get("/bars/{ticker}")
async def get_bars(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Serve OHLCV bars for a ticker as JSON.

    Tries the cached parquet first, falls back to yfinance.
    period: 1y (default), 6m, 3m, 1m, max, or ytd
    """
    import pandas as pd

    t = ticker.upper().strip()
    cache_path = REPO_ROOT / "data" / f"cache_ohlcv_{t}.parquet"
    hist_cache = REPO_ROOT / "data" / f"cache_ohlcv_hist_{t}.parquet"

    df = None
    # Prefer hist cache (has full history including pre-START buffer)
    if hist_cache.exists():
        try:
            df = pd.read_parquet(hist_cache)
        except Exception:
            df = None
    if df is None and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            df = None

    if df is None:
        import yfinance as yf
        yf_period = period if period != "1y" else "2y"
        raw = yf.download(t, period=yf_period, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise HTTPException(404, f"No data for ticker {t}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna(subset=["Close"]).sort_index()

    # Filter by period
    if period != "max":
        if period == "ytd":
            cutoff = pd.Timestamp.now().normalize().replace(month=1, day=1)
        else:
            period_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730}
            days = period_map.get(period, 365)
            cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        df = df.loc[df.index >= cutoff]

    bars = []
    for ts, row in df.iterrows():
        bars.append({
            "date": ts.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    return {
        "ticker": t,
        "period": period,
        "bars": bars,
        "first_date": bars[0]["date"] if bars else None,
        "last_date": bars[-1]["date"] if bars else None,
        "n_bars": len(bars),
    }


# --------------------------------------------------------------------------- #
# Runs — submit / poll
# --------------------------------------------------------------------------- #
@app.post("/runs")
async def submit_run(req: RunRequest) -> dict[str, Any]:
    """Submit a full pipeline run for a ticker. Returns job_id for polling."""
    job_id = uuid.uuid4().hex[:12]
    ticker = req.ticker.upper().strip()
    cfg = RunConfig(
        ticker=ticker,
        predict_target_date=req.predict_target_date,
        feature_validate_date=req.feature_validate_date,
        val_window_sessions=req.val_window_sessions,
        buy_entry_thr_mode=req.buy_entry_thr_mode,
        buy_entry_score_min=req.buy_entry_score_min,
        buy_require_lasso_up=req.buy_require_lasso_up,
        force_recalibrate=req.force_recalibrate,
        skip_feature_health=req.skip_feature_health,
        skip_plot_01=req.skip_plot_01,
        n_jobs_model=req.n_jobs_model,
    )

    def _run_job(job, args):
        # Preflight: check history before creating the output folder
        pf = _preflight(ticker)
        if not pf["executable"]:
            return {
                "ok": False,
                "error": "insufficient_history",
                "preflight": pf,
                "ticker": ticker,
            }
        g, s = run_stages(cfg, quiet=True)
        if s["ok"]:
            M.cache_fitted(job.id, g)
            result = _summary_from_globals(g)
            result["job_id"] = job.id
            return result
        else:
            return {"ok": False, "error": s["error"], "traceback": s.get("traceback")}

    JOBS.submit(job_id, "run", _run_job, {"ticker": ticker})
    return {"job_id": job_id, "ticker": ticker, "status": "queued"}


@app.post("/predict/live")
async def submit_live_predict(req: LivePredictRequest) -> dict[str, Any]:
    """Submit a live intraday prediction. Pulls today's session-so-far from
    yfinance, appends as a completed bar, and runs the full pipeline to predict
    the next session.  Returns job_id for polling."""
    from market_ml.live import live_predict

    job_id = uuid.uuid4().hex[:12]
    ticker = req.ticker.upper().strip()

    def _run_job(job, args):
        # Preflight: check history before running
        pf = _preflight(ticker)
        if not pf["executable"]:
            return {
                "ok": False,
                "error": "insufficient_history",
                "preflight": pf,
                "ticker": ticker,
                "live": True,
            }
        result = live_predict(
            ticker=ticker,
            target_date=req.target_date,
            override_close=req.override_close,
            quiet=True,
        )
        result["job_id"] = job.id
        return result

    JOBS.submit(job_id, "live_predict", _run_job, {"ticker": ticker})
    return {"job_id": job_id, "ticker": ticker, "status": "queued"}


@app.get("/jobs")
async def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_public() for j in reversed(JOBS.all())]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job.to_public()


@app.get("/jobs/{job_id}/logs")
async def job_logs_sse(job_id: str):
    """SSE stream of log lines for a job. Closes when job finishes."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")

    async def event_gen():
        idx = 0
        while True:
            while idx < len(job.logs):
                yield f"data: {json.dumps(job.logs[idx])}\n\n"
                idx += 1
            if job.status in ("done", "error") and idx >= len(job.logs):
                yield f"data: {json.dumps({'__done__': True, 'status': job.status})}\n\n"
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# Run folders — list / inspect / serve artifacts
# --------------------------------------------------------------------------- #
@app.get("/runs/{ticker}")
async def list_runs(ticker: str) -> dict[str, Any]:
    return {"ticker": ticker.upper(), "runs": _run_dirs_for_ticker(ticker)}


@app.get("/runs/{ticker}/{date}")
async def get_run(ticker: str, date: str) -> dict[str, Any]:
    return _list_artifacts(ticker, date)


@app.get("/artifacts/{ticker}/{date}/{filename:path}")
async def serve_artifact(ticker: str, date: str, filename: str):
    d = _run_dir(ticker, date)
    f = d / filename
    if not f.is_file():
        raise HTTPException(404, f"artifact not found: {filename}")
    # Prevent path traversal
    if not f.resolve().parent == d.resolve():
        raise HTTPException(403, "path traversal denied")
    return FileResponse(f)


# --------------------------------------------------------------------------- #
# Saved models
# --------------------------------------------------------------------------- #
@app.get("/models")
async def list_models(ticker: str | None = None) -> dict[str, Any]:
    return {"models": M.list_models(ticker)}


@app.get("/models/{model_id}")
async def get_model(model_id: str) -> dict[str, Any]:
    try:
        bundle = M.load_model(model_id)
        return {
            "model_id": model_id,
            "featureset": bundle["featureset"],
            "model_card": bundle["model_card"],
            "dir": bundle["dir"],
        }
    except FileNotFoundError:
        raise HTTPException(404, f"model {model_id} not found")


@app.post("/models")
async def save_model(req: SaveModelRequest) -> dict[str, Any]:
    try:
        result = M.save_model(req.ticker, req.run_id, req.name)
        return result
    except Exception as e:
        raise HTTPException(500, f"save failed: {e}")


@app.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict[str, Any]:
    ok = M.delete_model(model_id)
    if not ok:
        raise HTTPException(404, f"model {model_id} not found")
    return {"ok": True, "deleted": model_id}


@app.post("/models/{model_id}/predict")
async def predict_with_model(model_id: str, req: PredictRequest) -> dict[str, Any]:
    try:
        result = M.predict_with_saved_model(model_id, req.predict_target_date)
        return result
    except FileNotFoundError:
        raise HTTPException(404, f"model {model_id} not found")
    except Exception as e:
        raise HTTPException(500, f"predict failed: {e}")


# --------------------------------------------------------------------------- #
# Candidate features
# --------------------------------------------------------------------------- #
@app.get("/candidates")
async def list_candidates() -> dict[str, Any]:
    return {"candidates": C.list_candidates()}


@app.post("/candidates/test")
async def test_candidate(req: CandidateTestRequest) -> dict[str, Any]:
    """Test a candidate feature vs baseline (ΔAUC). Runs synchronously (~30-60s)."""
    job_id = uuid.uuid4().hex[:12]
    ticker = req.ticker.upper().strip()

    def _test_job(job, args):
        try:
            return C.test_candidate(
                ticker=ticker,
                candidate_name=req.candidate_name,
                predict_target_date=req.predict_target_date,
            )
        except Exception as e:
            import traceback
            return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

    JOBS.submit(job_id, "candidate_test", _test_job, {"ticker": ticker, "candidate": req.candidate_name})
    return {"job_id": job_id, "ticker": ticker, "candidate": req.candidate_name, "status": "queued"}


# --------------------------------------------------------------------------- #
# Calibration — per-ticker feature config + calibrate / predict lifecycle
# --------------------------------------------------------------------------- #
@app.get("/calibration")
async def calibration_overview() -> dict[str, Any]:
    """Calibration status for all configured tickers + the feature pool + pilot set."""
    return {
        "pool": CAL.get_feature_pool(),
        "baseline_6f": CAL.load_config().get("baseline_6f", []),
        "pilot": CAL.get_pilot_tickers(),
        "tickers": CAL.calibration_status_all(),
    }


@app.get("/calibration/{ticker}")
async def get_calibration(ticker: str, model_id: str | None = None) -> dict[str, Any]:
    """Show calibration for this ticker: features, metrics, model_id.

    If ``model_id`` query param is provided, returns that specific model's details.
    Otherwise returns the most recent model.
    If not yet calibrated, returns the configured feature set and ``calibrated: false``.
    """
    return CAL.calibration_status(ticker, model_id=model_id)


@app.put("/calibration/{ticker}/features")
async def update_ticker_features(ticker: str, req: UpdateFeaturesRequest) -> dict[str, Any]:
    """Update the feature set for a ticker in the config."""
    try:
        CAL.set_ticker_features(ticker, req.features)
        return {"ok": True, "ticker": ticker.upper(), "features": req.features}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/calibration/{ticker}")
async def calibrate_ticker(ticker: str, req: CalibrateRequest) -> dict[str, Any]:
    """Run a full pipeline calibration on the ticker's configured features and save the model.

    Returns job_id for polling. When done, the saved model is the new calibration.
    """
    job_id = uuid.uuid4().hex[:12]
    t = ticker.upper().strip()

    def _run_job(job, args):
        result = CAL.calibrate(
            t,
            predict_target_date=req.predict_target_date,
            val_window_sessions=req.val_window_sessions,
            name=req.name,
            n_jobs_model=req.n_jobs_model,
            features=req.features,
        )
        return result

    JOBS.submit(job_id, "calibrate", _run_job, {"ticker": t})
    return {"job_id": job_id, "ticker": t, "status": "queued"}


@app.post("/calibration/{ticker}/predict")
async def predict_with_calibration(ticker: str, req: PredictRequest) -> dict[str, Any]:
    """Score the latest session through the ticker's saved calibration (no refit).

    If ``override_close`` is provided, fetches today's intraday bar, overrides
    the close, and predicts the next session through the saved model.
    """
    t = ticker.upper().strip()
    job_id = uuid.uuid4().hex[:12]

    def _run_job(job, args):
        result = CAL.predict(
            t,
            predict_target_date=req.predict_target_date,
            override_close=req.override_close,
            model_id=req.model_id,
        )
        return result

    JOBS.submit(job_id, "predict", _run_job, {"ticker": t})
    return {"job_id": job_id, "ticker": t, "status": "queued"}
@app.get("/tickers")
async def list_tickers() -> dict[str, Any]:
    """List all tickers that have at least one run folder or cached data."""
    import glob
    run_folders = glob.glob(str(REPO_ROOT / "runs" / "improves_*"))
    tickers: set[str] = set()
    for p in run_folders:
        name = Path(p).name
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[0].startswith("improves_"):
            tickers.add(parts[0].replace("improves_", ""))
    # Also include tickers with cached data (exclude _hist_ warm-up caches)
    for p in glob.glob(str(REPO_ROOT / "data" / "cache_ohlcv_*.parquet")):
        name = Path(p).stem
        if name.startswith("cache_ohlcv_") and "_hist_" not in name:
            tickers.add(name.replace("cache_ohlcv_", ""))
    # Include saved models
    if M.MODELS_DIR.exists():
        for d in M.MODELS_DIR.iterdir():
            if d.is_dir():
                tickers.add(d.name)
    return {"tickers": sorted(tickers)}


@app.get("/research/{ticker}")
async def get_research(ticker: str) -> dict[str, Any]:
    """Serve research markdown files for a ticker from the research/ directory."""
    import glob
    pattern = str(REPO_ROOT / "research" / f"{ticker.upper()}_*.md")
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    notes = []
    for fp in files:
        p = Path(fp)
        notes.append({
            "filename": p.name,
            "content": p.read_text(),
            "modified": p.stat().st_mtime,
        })
    return {"ticker": ticker.upper(), "notes": notes}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050, log_level="info")
