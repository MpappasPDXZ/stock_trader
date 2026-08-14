"""Saved models + featuresets for market-ml.

A saved model is a directory under ``models/{TICKER}/{model_id}/`` holding the fitted
XGB + Lasso (joblib) plus a ``featureset.json`` (FEATURE_COLS, FEATURE_SPECS, feature
constants, ``auto_invert``, ``entry_score_min``, ``chosen_alpha``, train span) and a
``model_card.json`` (AUC raw+oriented, IC, base UP, ``is_discriminating``, ablation).

Predict-with-saved-model runs the canonical stages ``setup→…→assemble`` from
``market_ml.pipeline._STAGE_SOURCES`` (the *same* builders as the notebook — no second
feature stack), then loads the saved XGB/Lasso and scores the latest feature-complete
session. ``auto_invert`` + ``entry_score_min`` come from ``featureset.json``.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from market_ml.pipeline import RunConfig
from market_ml.jobs import run_stages

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# In-memory cache of fitted objects from a completed run, keyed by run_id, so SAVE
# doesn't need to refit. If absent (server restart), SAVE falls back to re-running
# stages up through walkforward_pred.
_FIT_CACHE: dict[str, dict[str, Any]] = {}


def cache_fitted(run_id: str, g: dict[str, Any]) -> None:
    """Stash the non-serialisable fitted objects + featureset from a run's exec globals."""
    _FIT_CACHE[run_id] = {
        "live_xgb": g.get("live_xgb"),
        "live_lasso": g.get("live_lasso"),
        "chosen_alpha": g.get("chosen_alpha"),
        "auto_invert": g.get("auto_invert", g.get("XGB_AUTO_INVERT", False)),
        "entry_score_min": g.get("ENTRY_OPERATING") or g.get("BUY_ENTRY_SCORE_MIN"),
        "FEATURE_COLS": list(g.get("FEATURE_COLS", [])),
        "FEATURE_SPECS": g.get("FEATURE_SPECS"),
        "feature_constants": _feature_constants(g),
        "pred_target": _safe_series_to_dict(g.get("pred_target")),
        "ticker": g.get("TICKER"),
        "predict_target_date": g.get("PREDICT_TARGET_DATE"),
        "train_rows": int(len(g["train_df"])) if "train_df" in g else None,
    }


def _feature_constants(g: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "BB_WINDOW", "BB_K", "FEAT_RSI_SESSIONS", "FEAT_OBV_Z_SESSIONS",
        "FEAT_BB_SESSIONS", "FEAT_VOL_RATIO_SHORT", "FEAT_VOL_RATIO_LONG",
        "FEAT_NEAR_HIGH_SESSIONS", "FEAT_TREND_SESSIONS",
        "FEAT_ADX_SESSIONS", "FEAT_BODY_EMA", "FEAT_STRESS_Z_WEEKS",
        "FRED_FETCH_YEARS_BACK",
        "FEATURE_MAX_SESSIONS", "PRE_START_DAYS",
        "RANDOM_STATE", "MIN_TRAIN_MONTHS", "VAL_WINDOW_SESSIONS",
    ]
    return {k: _jsonable(g.get(k)) for k in keys}


def _jsonable(v: Any) -> Any:
    import numpy as np
    import pandas as pd
    if isinstance(v, (bool,)):
        return v
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (float,)):
        return None if pd.isna(v) else float(v)
    if v is None:
        return None
    try:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if pd.isna(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
    except Exception:
        pass
    return v


def _safe_series_to_dict(s: Any) -> dict[str, Any] | None:
    if s is None:
        return None
    try:
        return {k: _jsonable(v) for k, v in dict(s).items()}
    except Exception:
        return None


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower() or "model"


def _model_dir(ticker: str, model_id: str) -> Path:
    return MODELS_DIR / ticker.upper() / model_id


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_model(ticker: str, run_id: str, name: str) -> dict[str, Any]:
    """Persist the fitted model + featureset from a completed run.

    Pulls fitted objects from ``_FIT_CACHE[run_id]`` when available; otherwise re-runs
    canonical stages up through ``walkforward_pred`` to regenerate them.
    """
    from joblib import dump

    ticker = ticker.upper().strip()
    cached = _FIT_CACHE.get(run_id)
    if cached is None or cached.get("live_xgb") is None:
        # Fallback: refit by re-running stages. We need a RunConfig matching the run.
        # Read it from the run's entry_rule / pred if available; otherwise default cfg.
        cfg = RunConfig(ticker=ticker)
        g, summary = run_stages(cfg, up_to="walkforward_pred")
        if not summary["ok"]:
            raise RuntimeError(f"refit for save failed: {summary['error']}")
        cache_fitted(run_id, g)
        cached = _FIT_CACHE[run_id]

    model_id = f"{ticker}__{_slug(name)}__{time.strftime('%Y%m%d%H%M%S')}"
    out = _model_dir(ticker, model_id)
    out.mkdir(parents=True, exist_ok=True)

    dump(cached["live_xgb"], out / "xgb_clf.joblib")
    dump(cached["live_lasso"], out / "lasso_pipe.joblib")

    featureset = {
        "model_id": model_id,
        "ticker": ticker,
        "name": name,
        "run_id": run_id,
        "saved_at": time.time(),
        "FEATURE_COLS": cached["FEATURE_COLS"],
        "FEATURE_SPECS": cached["FEATURE_SPECS"],
        "feature_constants": cached["feature_constants"],
        "auto_invert": bool(cached["auto_invert"]),
        "entry_score_min": _jsonable(cached["entry_score_min"]),
        "chosen_alpha": _jsonable(cached["chosen_alpha"]),
        "train_rows": cached["train_rows"],
        "predict_target_date": cached["predict_target_date"],
    }
    (out / "featureset.json").write_text(json.dumps(featureset, indent=2, default=str))

    model_card = _build_model_card(run_id, ticker)
    (out / "model_card.json").write_text(json.dumps(model_card, indent=2, default=str))

    return {"model_id": model_id, "path": str(out), "featureset": featureset, "model_card": model_card}


def _build_model_card(run_id: str, ticker: str) -> dict[str, Any]:
    """Pull discrimination / AUC / ablation from the run folder artifacts."""
    import glob
    # Find the run folder for this ticker (most recent matching improves_{T}_*)
    candidates = sorted(
        (p for p in glob.glob(str(REPO_ROOT / f"improves_{ticker}_*")) if Path(p).is_dir()),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    card: dict[str, Any] = {"ticker": ticker, "run_id": run_id}
    if not candidates:
        card["note"] = "no run folder found"
        return card
    d = Path(candidates[0])
    card["run_folder"] = str(d)
    for fn, key in [
        (f"entry_rule_80_{ticker}.json", "entry_rule"),
        (f"feature_health_{ticker}.json", "feature_health"),
        (f"pred_{Path(d).name.split('_')[-1]}_{ticker}.json", "pred"),
    ]:
        p = d / fn
        if p.exists():
            try:
                card[key] = json.loads(p.read_text())
            except Exception:
                pass
    return card


# --------------------------------------------------------------------------- #
# List / load
# --------------------------------------------------------------------------- #
def list_models(ticker: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return out
    pattern = "*" if ticker is None else ticker.upper()
    for tdir in sorted(MODELS_DIR.glob(pattern)):
        if not tdir.is_dir():
            continue
        for mdir in sorted(tdir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            fs = mdir / "featureset.json"
            if not fs.exists():
                continue
            try:
                meta = json.loads(fs.read_text())
            except Exception:
                continue
            card_p = mdir / "model_card.json"
            card = json.loads(card_p.read_text()) if card_p.exists() else {}
            out.append({
                "model_id": meta.get("model_id", mdir.name),
                "ticker": tdir.name,
                "name": meta.get("name", ""),
                "saved_at": meta.get("saved_at"),
                "FEATURE_COLS": meta.get("FEATURE_COLS", []),
                "auto_invert": meta.get("auto_invert"),
                "entry_score_min": meta.get("entry_score_min"),
                "train_rows": meta.get("train_rows"),
                "is_discriminating": (card.get("entry_rule") or {}).get("is_discriminating"),
                "oriented_auc": (card.get("entry_rule") or {}).get("oriented_auc"),
            })
    return out


def load_model(model_id: str) -> dict[str, Any]:
    from joblib import load
    # model_id = TICKER__name__ts ; find its dir
    if not MODELS_DIR.exists():
        raise FileNotFoundError(model_id)
    for tdir in MODELS_DIR.iterdir():
        mdir = tdir / model_id
        if mdir.is_dir():
            fs = json.loads((mdir / "featureset.json").read_text())
            xgb = load(mdir / "xgb_clf.joblib")
            lasso = load(mdir / "lasso_pipe.joblib")
            card = {}
            cp = mdir / "model_card.json"
            if cp.exists():
                card = json.loads(cp.read_text())
            return {"xgb": xgb, "lasso": lasso, "featureset": fs, "model_card": card, "dir": str(mdir)}
    raise FileNotFoundError(f"model not found: {model_id}")


def delete_model(model_id: str) -> bool:
    if not MODELS_DIR.exists():
        return False
    for tdir in MODELS_DIR.iterdir():
        mdir = tdir / model_id
        if mdir.is_dir():
            shutil.rmtree(mdir)
            return True
    return False


# --------------------------------------------------------------------------- #
# Predict with saved model (no refit)
# --------------------------------------------------------------------------- #
def predict_with_saved_model(
    model_id: str,
    predict_target_date: str | None = None,
) -> dict[str, Any]:
    """Score the latest feature-complete session through a saved model.

    Runs canonical stages setup→assemble (same builders as notebook), verifies the
    assembled FEATURE_COLS matches the saved featureset, then loads the saved XGB/Lasso
    and scores. ``auto_invert`` + ``entry_score_min`` from ``featureset.json``.
    """
    import numpy as np
    import pandas as pd

    bundle = load_model(model_id)
    fs = bundle["featureset"]
    ticker = fs["ticker"]

    cfg = RunConfig(
        ticker=ticker,
        predict_target_date=predict_target_date,
        n_jobs_model=2,
    )
    g, summary = run_stages(cfg, up_to="assemble")
    if not summary["ok"]:
        raise RuntimeError(f"feature rebuild failed: {summary['error']}")

    live_cols = list(g.get("FEATURE_COLS", []))
    if live_cols != list(fs["FEATURE_COLS"]):
        raise RuntimeError(
            f"featureset mismatch: live={live_cols} vs saved={list(fs['FEATURE_COLS'])}. "
            "Refit the model with the current notebook feature stack."
        )

    bars: pd.DataFrame = g["bars"]
    PRED_TARGET_TS = pd.Timestamp(predict_target_date or g["PREDICT_TARGET_DATE"])
    eligible = bars.index[bars.index < PRED_TARGET_TS]
    if len(eligible) == 0:
        raise RuntimeError(f"no feature session before {PRED_TARGET_TS.date()}")
    asof_ts = eligible[-1]
    if bars.loc[asof_ts, live_cols].isna().any():
        raise RuntimeError(f"NaN features on asof {asof_ts.date()}")

    X_live = bars.loc[[asof_ts], live_cols]
    close_asof = float(bars.loc[asof_ts, "Close"])

    xgb_p = float(bundle["xgb"].predict_proba(X_live)[0, 1])
    lasso_lr = float(bundle["lasso"].predict(X_live)[0])
    pred_close = close_asof * float(np.exp(lasso_lr))

    auto_invert = bool(fs.get("auto_invert", False))
    entry_score = (1.0 - xgb_p) if auto_invert else xgb_p
    thr = float(fs.get("entry_score_min") or 0.72)
    enter = bool(np.isfinite(entry_score) and entry_score >= thr)

    result = {
        "ticker": ticker,
        "model_id": model_id,
        "target_session": str(PRED_TARGET_TS.date()),
        "feature_asof": str(asof_ts.date()),
        "close_asof": close_asof,
        "xgb_prob_up": xgb_p,
        "xgb_pred_up": int(xgb_p >= 0.5),
        "lasso_pred_logret": lasso_lr,
        "lasso_implied_close": pred_close,
        "lasso_pred_up": int(lasso_lr > 0),
        "auto_invert": auto_invert,
        "entry_score": entry_score,
        "entry_score_min": thr,
        "decision": "BUY" if enter else "SKIP",
        "rationale": (
            f"entry_score={entry_score:.1%} {'≥' if enter else '<'} thr {thr:.1%}"
            f"  (raw P(up)={xgb_p:.1%}{', inverted' if auto_invert else ''})"
        ),
        "features": {c: _jsonable(bars.loc[asof_ts, c]) for c in live_cols},
    }

    # persist last_pred.json next to the model
    mdir = Path(bundle["dir"])
    (mdir / "last_pred.json").write_text(json.dumps(result, indent=2, default=str))
    return result
