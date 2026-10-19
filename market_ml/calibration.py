"""Per-ticker feature config + calibration lifecycle.

Calibration = fit model on a ticker's configured feature set and save it.
Prediction = score the latest session through the saved model (no refit).

Config file: ``data/ticker_features.json``
  - pool: all 18 available features
  - baseline_6f: the original 6
  - pilot: 10 tickers for testing new features
  - tickers: { TICKER: [feature names] }

Calibration artifacts are saved models under ``models/{TICKER}/{model_id}/``
(via ``market_ml.models``).  The calibration status (last calibration date,
features used, discrimination metrics) is read from the saved model's
``featureset.json`` + ``model_card.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "data" / "ticker_features.json"


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"ticker feature config not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, default=str))


def get_ticker_features(ticker: str) -> list[str]:
    cfg = load_config()
    t = ticker.upper().strip()
    features = cfg.get("tickers", {}).get(t)
    if features is None:
        features = cfg.get("baseline_6f", [])
    return list(features)


def get_pilot_tickers() -> list[str]:
    return list(load_config().get("pilot", []))


def get_feature_pool() -> list[str]:
    return list(load_config().get("pool", []))


def set_ticker_features(ticker: str, features: list[str]) -> None:
    cfg = load_config()
    t = ticker.upper().strip()
    pool = set(cfg.get("pool", []))
    invalid = [f for f in features if f not in pool]
    if invalid:
        raise ValueError(f"features not in pool: {invalid}")
    cfg.setdefault("tickers", {})[t] = list(features)
    save_config(cfg)


def is_pilot(ticker: str) -> bool:
    return ticker.upper().strip() in get_pilot_tickers()


def get_inactive_tickers() -> list[str]:
    """Tickers the user has hidden from the market landing page."""
    return list(load_config().get("inactive", []))


def is_inactive(ticker: str) -> bool:
    return ticker.upper().strip() in get_inactive_tickers()


def set_inactive(ticker: str, inactive: bool) -> list[str]:
    """Add or remove a ticker from the inactive set. Returns the new inactive list."""
    cfg = load_config()
    t = ticker.upper().strip()
    inactive_set: list[str] = list(cfg.get("inactive", []))
    if inactive and t not in inactive_set:
        inactive_set.append(t)
    elif not inactive and t in inactive_set:
        inactive_set = [x for x in inactive_set if x != t]
    cfg["inactive"] = inactive_set
    save_config(cfg)
    return inactive_set


# --------------------------------------------------------------------------- #
# Calibration status — what was last calibrated for this ticker
# --------------------------------------------------------------------------- #
def calibration_status(ticker: str, model_id: str | None = None) -> dict[str, Any]:
    """Return calibration status for a ticker.

    If ``model_id`` is provided, returns details for that specific model.
    Otherwise picks the most recent saved model.
    If no saved model exists, returns ``calibrated: false`` with the configured
    feature set so the caller knows what *would* be calibrated.
    """
    from market_ml import models as M

    t = ticker.upper().strip()
    configured_features = get_ticker_features(t)
    pilot = is_pilot(t)

    status: dict[str, Any] = {
        "ticker": t,
        "pilot": pilot,
        "configured_features": configured_features,
        "calibrated": False,
        "model_id": None,
    }

    saved = M.list_models(t)
    if not saved:
        return status

    if model_id is not None:
        latest = next((m for m in saved if m["model_id"] == model_id), None)
        if latest is None:
            return status
    else:
        latest = saved[0]
    model_id = latest["model_id"]

    # Load full model card for feature health + run folder
    card_path = M.MODELS_DIR / t / model_id / "model_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    fh = card.get("feature_health", {})
    run_folder = card.get("run_folder", "")

    # Derive run date from run_folder path (e.g. improves_GEV_2026-08-17 → 2026-08-17)
    run_date = ""
    if run_folder:
        from pathlib import Path as _P
        run_date = _P(run_folder).name.split("_")[-1]

    status.update({
        "calibrated": True,
        "model_id": model_id,
        "calibrated_at": latest.get("saved_at"),
        "calibrated_features": latest.get("FEATURE_COLS", []),
        "auto_invert": latest.get("auto_invert"),
        "entry_score_min": latest.get("entry_score_min"),
        "is_discriminating": latest.get("is_discriminating"),
        "oriented_auc": latest.get("oriented_auc"),
        "train_rows": latest.get("train_rows"),
        "features_match_config": latest.get("FEATURE_COLS", []) == configured_features,
        "run_date": run_date,
        "feature_health": {
            "auc_full_raw": fh.get("auc_full_raw"),
            "auc_full_oriented": fh.get("auc_full_oriented"),
            "ic_full": fh.get("ic_full"),
            "base_up": fh.get("base_up"),
            "oos_steps": fh.get("oos_steps"),
            "model_beats_null": fh.get("model_beats_null"),
            "high_vif_features": fh.get("high_vif_features", []),
            "high_corr_pairs": [
                pair for pair in fh.get("high_corr_pairs", [])
                if isinstance(pair.get("r"), (int, float))
            ],
            "uni": [
                u for u in fh.get("uni", [])
                if isinstance(u.get("oos_auc_oriented"), (int, float))
            ],
            "ablation": [
                a for a in fh.get("ablation", [])
                if isinstance(a.get("delta_vs_full"), (int, float))
            ],
        },
    })
    return status


def calibration_status_all() -> list[dict[str, Any]]:
    """Calibration status for every ticker in the config."""
    cfg = load_config()
    tickers = sorted(cfg.get("tickers", {}).keys())
    return [calibration_status(t) for t in tickers]


# --------------------------------------------------------------------------- #
# Calibrate — fit model on configured features and save it
# --------------------------------------------------------------------------- #
def calibrate(
    ticker: str,
    *,
    predict_target_date: str | None = None,
    val_window_sessions: int = 60,
    name: str | None = None,
    n_jobs_model: int = 2,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Run the pipeline with the ticker's configured features and save the model.

    This is a full pipeline run (setup → buy_rule) using the per-ticker feature
    subset, then save the fitted model via ``market_ml.models.save_model``.

    If ``features`` is provided, those features are used instead of the config
    file's ``active_features``.  The config file is NOT modified.

    Returns the calibration result: model_id, features, discrimination metrics.
    """
    from market_ml.pipeline import RunConfig
    from market_ml.jobs import run_stages
    from market_ml import models as M

    t = ticker.upper().strip()
    if features is not None:
        active = list(features)
    else:
        active = get_ticker_features(t)

    cfg = RunConfig(
        ticker=t,
        active_features=active,
        predict_target_date=predict_target_date,
        val_window_sessions=val_window_sessions,
        skip_feature_health=False,
        skip_plot_01=False,
        n_jobs_model=n_jobs_model,
        quiet=True,
    )

    g, summary = run_stages(cfg, quiet=True)
    if not summary["ok"]:
        # Young-ticker fallback: if walkforward failed because there aren't
        # enough labeled rows for the walk-forward window, skip walkforward
        # / calibrate / buy_rule and fit a simple model on all labeled rows.
        err = str(summary.get("error") or "")
        n_labeled = 0
        try:
            n_labeled = len(g.get("model_df", []))
        except Exception:
            pass
        if n_labeled and n_labeled < 65:
            young = _simple_fit_young_ticker(cfg, g, n_labeled)
            if young is not None:
                run_id = f"calibrate_{t}_{int(time.time())}"
                M.cache_fitted(run_id, young)
                model_name = name or f"young_ticker_{time.strftime('%Y%m%d')}"
                save_result = M.save_model(t, run_id, model_name)
                young_summary = _young_summary_from_globals(young, save_result)
                return young_summary
        return {"ok": False, "error": summary["error"], "traceback": summary.get("traceback")}

    run_id = f"calibrate_{t}_{int(time.time())}"
    M.cache_fitted(run_id, g)

    model_name = name or f"calibrated_{time.strftime('%Y%m%d')}"
    save_result = M.save_model(t, run_id, model_name)

    from market_ml.server import _summary_from_globals
    run_summary = _summary_from_globals(g)

    return {
        "ok": True,
        "ticker": t,
        "model_id": save_result["model_id"],
        "features": active,
        "calibrated_at": time.time(),
        "is_discriminating": run_summary.get("is_discriminating"),
        "oriented_auc": run_summary.get("oriented_auc"),
        "raw_xgb_auc": run_summary.get("raw_xgb_auc"),
        "entry_score_min": run_summary.get("operating_entry_score_min"),
        "auto_invert": run_summary.get("auto_invert"),
        "decision": run_summary.get("decision"),
        "predict_target_date": run_summary.get("predict_target_date"),
        "output_dir": run_summary.get("output_dir"),
    }


# --------------------------------------------------------------------------- #
# Young-ticker fallback — simple fit on all labeled rows (no walkforward)
# --------------------------------------------------------------------------- #
def _simple_fit_young_ticker(cfg: "Any", g: dict, n_labeled: int) -> dict | None:
    """Fit Lasso + XGB on all available labeled rows; predict the live session.

    Used when ``n_labeled < 65`` (the walkforward floor). Produces a fitted
    model with no out-of-sample AUC-PR / discrimination (too few rows). Sets
    ``XGB_AUTO_INVERT=False``, ``XGB_IS_DISCRIMINATING=False``, and a
    conservative default ``ENTRY_OPERATING=0.50`` (no calibrated threshold).

    Mutates ``g`` in place with: live_xgb, live_lasso, chosen_alpha, train_df,
    pred_target, ENTRY_OPERATING, BUY_ENTRY_SCORE_MIN, XGB_AUTO_INVERT,
    XGB_IS_DISCRIMINATING, summary_80 (a minimal entry_rule stub), and
    RUN_DATE for ``_build_model_card``.
    """
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier

    bars = g.get("bars")
    model_df = g.get("model_df")
    FEATURE_COLS = g.get("FEATURE_COLS", [])
    TARGET_REG = g.get("TARGET_REG", "y_reg")
    TARGET_CLS = g.get("TARGET_CLS", "y_cls")
    PREDICT_TARGET_DATE = g.get("PREDICT_TARGET_DATE")
    PRED_TARGET_TS = pd.Timestamp(PREDICT_TARGET_DATE)
    RANDOM_STATE = g.get("RANDOM_STATE", 42)
    n_jobs = int(getattr(cfg, "n_jobs_model", 2) or 2)
    OUTPUT_DIR = g.get("OUTPUT_DIR")

    if bars is None or model_df is None or not FEATURE_COLS:
        return None

    # Feature session = latest bar strictly before target date
    eligible = bars.index[bars.index < PRED_TARGET_TS]
    if len(eligible) == 0:
        return None
    asof_ts = eligible[-1]
    if bars.loc[asof_ts, FEATURE_COLS].isna().any():
        return None

    # Train on all labeled rows with feature date < asof (no leakage)
    train_df = model_df.loc[model_df.index < asof_ts]
    if len(train_df) < 8:
        return None

    X_tr = train_df[FEATURE_COLS]
    y_reg_tr = train_df[TARGET_REG]
    y_cls_tr = train_df[TARGET_CLS].astype(int)

    # Lasso alpha via CV (few rows → 2 splits, min 2)
    n_splits = max(2, min(3, len(train_df) // 4))
    try:
        alpha_search = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(
                alphas=np.logspace(-5, -1, 40),
                cv=TimeSeriesSplit(n_splits=n_splits),
                max_iter=20_000,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            )),
        ])
        alpha_search.fit(X_tr, y_reg_tr)
        chosen_alpha = float(alpha_search.named_steps["model"].alpha_)
    except Exception:
        chosen_alpha = 0.001

    live_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=chosen_alpha, max_iter=20_000, random_state=RANDOM_STATE)),
    ])
    live_lasso.fit(X_tr, y_reg_tr)
    X_live = bars.loc[[asof_ts], FEATURE_COLS]
    live_logret = float(live_lasso.predict(X_live)[0])
    close_asof = float(bars.loc[asof_ts, "Close"])
    pred_close = close_asof * float(np.exp(live_logret))

    live_xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, objective="binary:logistic",
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=n_jobs,
    )
    live_xgb.fit(X_tr, y_cls_tr, verbose=False)
    live_prob = float(live_xgb.predict_proba(X_live)[0, 1])
    live_up = int(live_prob >= 0.5)

    pred_target = pd.Series({
        "ticker": cfg.ticker.upper().strip(),
        "target_session": PREDICT_TARGET_DATE,
        "feature_asof": str(asof_ts.date()),
        "close_asof": close_asof,
        "lasso_pred_logret": live_logret,
        "lasso_implied_close": pred_close,
        "lasso_pred_up": int(live_logret > 0),
        "xgb_prob_up": live_prob,
        "xgb_pred_up": live_up,
        "train_rows": int(len(train_df)),
        "lasso_alpha": chosen_alpha,
    })

    # Young-ticker defaults: no calibration, conservative 0.50 threshold
    g["live_xgb"] = live_xgb
    g["live_lasso"] = live_lasso
    g["chosen_alpha"] = chosen_alpha
    g["train_df"] = train_df
    g["pred_target"] = pred_target
    g["XGB_AUTO_INVERT"] = False
    g["XGB_IS_DISCRIMINATING"] = False
    g["ENTRY_OPERATING"] = 0.50
    g["BUY_ENTRY_SCORE_MIN"] = 0.50
    g["BUY_XGB_PROB_MIN"] = 0.50
    g["auto_invert"] = False
    # Minimal entry_rule stub for the model card / summary extraction
    auto_inv = False
    entry_score = live_prob  # not inverted
    enter = bool(np.isfinite(entry_score) and entry_score >= 0.50)
    g["summary_80"] = {
        "is_discriminating": False,
        "auto_invert": False,
        "operating_entry_score_min": 0.50,
        "oriented_auc": None,
        "raw_xgb_auc": None,
        "best_rule": {
            "entry_score_min": 0.50,
            "xgb_p_min": 0.50,
            "max_precision_eligible_thr": None,
            "operator_entry_score_min": 0.50,
            "n_buys": None,
            "n_wins": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "beats_coin_50": False,
            "beats_base_up": False,
            "meets_target": False,
            "er_per_buy": None,
            "gain_captured": None,
        },
        "pending": {
            "date": PREDICT_TARGET_DATE,
            "xgb_prob_up_raw": live_prob,
            "entry_score": entry_score,
            "enter_under_best_rule": enter,
        },
        "young_ticker": True,
        "n_labeled_rows": int(n_labeled),
        "note": (
            f"Young ticker — only {n_labeled} labeled rows (< 65 walkforward floor). "
            "Fit on all available rows; no out-of-sample AUC-PR / discrimination. "
            "Threshold defaults to 0.50 (no calibrated entry rule)."
        ),
    }
    # Persist artifacts so _summary_from_globals can read them
    if OUTPUT_DIR is not None:
        import json as _json
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"entry_rule_80_{cfg.ticker.upper().strip()}.json").write_text(
            _json.dumps(g["summary_80"], indent=2, default=str)
        )
        (OUTPUT_DIR / f"pred_{PREDICT_TARGET_DATE}_{cfg.ticker.upper().strip()}.json").write_text(
            _json.dumps(
                {k: (float(v) if hasattr(v, "item") else v) for k, v in pred_target.items()},
                indent=2, default=str,
            )
        )
    g["RUN_DATE"] = g.get("RUN_DATE") or PREDICT_TARGET_DATE
    return g


def _young_summary_from_globals(g: dict, save_result: dict) -> dict[str, Any]:
    """Build a calibration result summary for the young-ticker simple-fit path."""
    from market_ml.server import _summary_from_globals
    s = _summary_from_globals(g)
    s["ok"] = True
    s["ticker"] = g.get("TICKER")
    s["model_id"] = save_result["model_id"]
    s["features"] = list(g.get("FEATURE_COLS", []))
    s["calibrated_at"] = time.time()
    s["is_discriminating"] = False
    s["entry_score_min"] = 0.50
    s["auto_invert"] = False
    s["young_ticker"] = True
    s["n_labeled_rows"] = int(len(g.get("train_df", [])))
    s["predict_target_date"] = g.get("PREDICT_TARGET_DATE")
    return s


# --------------------------------------------------------------------------- #
# Predict — score latest session through saved calibration (no refit)
# --------------------------------------------------------------------------- #
def predict(
    ticker: str,
    predict_target_date: str | None = None,
    override_close: float | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Score the latest session through a saved model.

    If ``model_id`` is provided, uses that specific model. Otherwise uses the
    ticker's most recent saved model.

    If ``override_close`` is provided, fetches today's intraday bar from
    yfinance, overrides the close, injects it as the feature-asof session,
    and predicts the next session through the saved model.
    """
    from market_ml import models as M

    t = ticker.upper().strip()

    if model_id is None:
        saved = M.list_models(t)
        if not saved:
            return {
                "ok": False,
                "error": "no calibrated model — run calibrate first",
                "ticker": t,
            }
        model_id = saved[0]["model_id"]

    if override_close is not None:
        result = _predict_live(t, model_id, override_close, predict_target_date)
    else:
        result = M.predict_with_saved_model(model_id, predict_target_date)

    result["ok"] = True
    result["model_id"] = model_id

    # Attach model precision / trade counts
    models = M.list_models(t)
    m = next((x for x in models if x["model_id"] == model_id), None)
    if m:
        result["n_wins"] = m.get("n_wins")
        result["n_buys"] = m.get("n_buys")
        result["precision"] = m.get("precision")

    return result


def _predict_live(
    ticker: str,
    model_id: str,
    override_close: float,
    predict_target_date: str | None,
) -> dict[str, Any]:
    """Inject today's live bar with an overridden close, then score through saved model."""
    import numpy as np
    import pandas as pd
    from market_ml.pipeline import RunConfig
    from market_ml.jobs import run_stages
    from market_ml.live import fetch_live_bar, _inject_live_bar
    from market_ml import models as M

    bundle = M.load_model(model_id)
    fs = bundle["featureset"]

    if predict_target_date is None:
        predict_target_date = (
            pd.Timestamp.now().normalize() + pd.offsets.BDay(1)
        ).strftime("%Y-%m-%d")

    cfg = RunConfig(
        ticker=ticker,
        predict_target_date=predict_target_date,
        n_jobs_model=2,
        active_features=list(fs["FEATURE_COLS"]),
    )

    g, summary = run_stages(cfg, up_to="load_ohlcv")
    if not summary["ok"]:
        raise RuntimeError(f"setup/load failed: {summary['error']}")

    live_bar = fetch_live_bar(ticker, override_close=override_close)
    if live_bar is None:
        raise RuntimeError(
            "No intraday bar available — market may be closed. "
            "Use PREDICT without override close, or try during market hours."
        )
    _inject_live_bar(g, ticker, live_bar, quiet=True)

    g, summary = run_stages(cfg, up_to="assemble", _globals=g)
    if not summary["ok"]:
        raise RuntimeError(f"feature rebuild failed: {summary['error']}")

    live_cols = list(g.get("FEATURE_COLS", []))
    if live_cols != list(fs["FEATURE_COLS"]):
        raise RuntimeError(
            f"featureset mismatch: live={live_cols} vs saved={list(fs['FEATURE_COLS'])}. "
            "Recalibrate with the current feature stack."
        )

    bars: pd.DataFrame = g["bars"]
    PRED_TARGET_TS = pd.Timestamp(predict_target_date)
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
        "override_close": override_close,
        "live_bar": {
            "date": str(pd.Timestamp(live_bar.name).date()),
            "open": float(live_bar["Open"]),
            "high": float(live_bar["High"]),
            "low": float(live_bar["Low"]),
            "close": float(live_bar["Close"]),
            "volume": float(live_bar["Volume"]),
            "override_close": True,
        },
    }

    # ------------------------------------------------------------------ #
    # Run remaining stages (walkforward_pred → calibrate → buy_rule) and
    # persist last_pred.json  — previously the live path computed the
    # decision in memory only, leaving the run folder empty: run history
    # showed decision None and the date page loaded no artifacts.
    # ------------------------------------------------------------------ #
    from pathlib import Path as _Path  # noqa: PLC0415

    try:
        _done = (
            "setup", "load_ohlcv", "feature_registry",
            "feat_rsi", "feat_vol_ratio", "feat_near_high", "feat_obv",
            "feat_bb", "feat_trend", "feat_adx", "feat_candle", "feat_fred",
            "assemble", "feature_health", "plot_01",
        )
        g, _chart_summary = run_stages(
            cfg, up_to="buy_rule", skip=_done, _globals=g, quiet=True,
        )
        _out = g.get("OUTPUT_DIR")
        if _out is not None:
            result["output_dir"] = str(_out)
            for _p in _Path(_out).glob(f"03_dir_hits_last*_{ticker}.png"):
                result["chart_03"] = _p.name
                break
        if not _chart_summary.get("ok"):
            result["chart_error"] = _chart_summary.get("error", "unknown")
    except Exception as _ce:
        result["chart_error"] = f"{type(_ce).__name__}: {_ce}"

    # persist last_pred.json next to the model
    mdir = _Path(bundle["dir"])
    (mdir / "live_last_pred.json").write_text(json.dumps(result, indent=2, default=str))
    return result
