#!/usr/bin/env python3
"""
6f baseline with custom XGBoost:
  - eval_metric='aucpr' (PR-AUC)
  - Custom asymmetric objective: penalty_weight=1.25 (FP penalized 1.25x)
  - Native API xgb.train with early_stopping_rounds=10
  - 3 tickers in parallel

22 valid tickers (RACK and SKHY excluded — no data)

Walk-forward returns raw OOS arrays (probs, actuals, returns).
Threshold scan runs post-hoc with early-stop on precision decrease.
Discrimination gate: precision > pass_floor (max(50%, base_up)) AND n_buys >= MIN_TRADES.
"""
import sys, os, json, time, traceback, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path("/Users/matthewpappas/code/ml_analysis")
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

# ─── Config ───
TICKERS = ["AMZN", "CAT", "CRM", "CSCO", "FLR", "GEV", "GOOG", "MCD", "META", "MHVIY",
           "MSFT", "MU", "NFLX", "NOK", "OUST", "PLTR", "PWR", "SMERY", "SNDK", "VRT", "WMT", "XLI"]
BASE_6 = "rsi_14,obv_chg_z_20,bb_dist_lower,vol_ratio_7_21,near_high_200,med_term_21wk"
VARIANTS = {
    "6f": BASE_6,
}
PARALLEL = 3
PENALTY_WEIGHT = 1.25
NUM_BOOST_ROUND = 100
EARLY_STOPPING = 10
MIN_TRAIN = 80
VAL_WINDOW = 60  # last 60 sessions OOS

# Discrimination gate constants (match pipeline)
COIN_FLOOR = 0.50
MIN_TRADES = 10


# ─── Custom asymmetric logistic loss ───
def asymmetric_logistic_loss(y_true, y_pred_prob, penalty_weight=5.0):
    p = 1.0 / (1.0 + np.exp(-y_pred_prob))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    y_true = np.asarray(y_true, dtype=np.float64)
    grad = p * (1.0 + (penalty_weight - 1.0) * (1.0 - y_true)) - y_true
    hess = p * (1.0 - p) * (1.0 + (penalty_weight - 1.0) * (1.0 - y_true))
    return grad, hess


def custom_aucpr_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    return "aucpr", average_precision_score(y_true, p)


# ─── Get model_df from pipeline ───
def get_model_df(ticker: str, active_features: str) -> pd.DataFrame | None:
    from market_ml.pipeline import _STAGE_SOURCES, RunConfig

    cfg = RunConfig(
        ticker=ticker,
        active_features=active_features.split(","),
        quiet=True,
        skip_feature_health=True,
        skip_plot_01=True,
    )

    stages_through_assemble = [
        "setup", "load_ohlcv", "feature_registry",
        "feat_rsi", "feat_vol_ratio", "feat_near_high",
        "feat_obv", "feat_bb", "feat_trend", "feat_adx", "feat_candle",
        "feat_fred", "assemble",
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score,
        roc_auc_score, roc_curve,
    )
    from scipy import stats
    from zoneinfo import ZoneInfo

    g: dict = {
        "__name__": f"market_ml.pipeline.{ticker}",
        "__builtins__": __builtins__,
        "display": lambda *a, **k: None,
        "np": np, "pd": pd, "plt": plt, "Path": Path,
        "warnings": __import__("warnings"),
        "json": json, "datetime": __import__("datetime"),
        "time": __import__("time"),
        "ZoneInfo": ZoneInfo,
        "Pipeline": Pipeline, "StandardScaler": StandardScaler,
        "Lasso": Lasso, "LassoCV": LassoCV,
        "TimeSeriesSplit": TimeSeriesSplit,
        "XGBClassifier": XGBClassifier,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
        "mdates": __import__("matplotlib.dates", fromlist=["dates"]),
        "stats": stats, "spearmanr": spearmanr,
        "Rectangle": __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle,
        "stats": stats,
    }

    def _patch_model_n_jobs(src, n_jobs):
        src = re.sub(r"n_jobs\s*=\s*-1", f"n_jobs={n_jobs}", src)
        src = re.sub(r"n_jobs\s*=\s*2\b", f"n_jobs={n_jobs}", src)
        return src

    try:
        for stage in stages_through_assemble:
            src = _patch_model_n_jobs(_STAGE_SOURCES[stage], cfg.n_jobs_model)

            if stage == "setup":
                src = re.sub(
                    r'^TICKER\s*=\s*["\'][^"\']*["\']',
                    f'TICKER = {ticker.upper().strip()!r}',
                    src, count=1, flags=re.M,
                )
                exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
                from market_ml.pipeline import _apply_overrides
                _apply_overrides(cfg, g)
            else:
                exec(compile(src, f"<stage:{stage}>", "exec"), g, g)

            if stage == "feature_registry" and cfg.active_features is not None:
                af = set(cfg.active_features)
                g["FEATURE_SPECS"] = [s for s in g["FEATURE_SPECS"] if s["name"] in af]
                g["FEATURE_COLS"] = [s["name"] for s in g["FEATURE_SPECS"]]
                g["FEATURE_MAX_SESSIONS"] = max(s["sessions_before_start"] for s in g["FEATURE_SPECS"])

        model_df = g["model_df"]
        feature_cols = g["FEATURE_COLS"]
        return model_df, feature_cols

    except Exception as e:
        print(f"  [ERROR {ticker}] {e}")
        traceback.print_exc()
        return None, None


# ─── Walk-forward with custom XGBoost (returns raw OOS arrays) ───
def walk_forward(model_df: pd.DataFrame, feature_cols: list[str], ticker: str = "") -> dict:
    """
    Expanding walk-forward CV using custom XGBoost.
    Returns raw OOS arrays for post-hoc threshold scan.
    """
    X = model_df[feature_cols]
    y = model_df["y_cls"].astype(int).to_numpy()
    n = len(model_df)
    dates = model_df.index

    val_n = min(VAL_WINDOW, n - MIN_TRAIN)
    if val_n < 5:
        return {"error": f"not enough data: n={n}, need {MIN_TRAIN + 5}"}

    wf_start = n - val_n

    oos_probs = []
    oos_actual = []
    oos_dates = []
    best_iterations = []

    for i in range(wf_start, n):
        X_train = X.iloc[:i].to_numpy()
        y_train = y[:i]
        X_test = X.iloc[i : i + 1].to_numpy()
        y_test = y[i]

        split_idx = max(int(len(X_train) * 0.8), MIN_TRAIN - 10)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        if len(set(y_val)) < 2:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)
            params = {"disable_default_eval_metric": 1}
            bst = xgb.train(
                params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                obj=lambda y_pred, dtrain: asymmetric_logistic_loss(
                    dtrain.get_label(), y_pred, penalty_weight=PENALTY_WEIGHT),
                custom_metric=custom_aucpr_eval,
                evals=[(dtrain, "train")],
                verbose_eval=False,
            )
        else:
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)
            params = {"disable_default_eval_metric": 1}
            bst = xgb.train(
                params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                obj=lambda y_pred, dtrain: asymmetric_logistic_loss(
                    dtrain.get_label(), y_pred, penalty_weight=PENALTY_WEIGHT),
                custom_metric=custom_aucpr_eval,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=EARLY_STOPPING,
                verbose_eval=False,
            )

        prob = float(bst.predict(dtest)[0])
        oos_probs.append(prob)
        oos_actual.append(y_test)
        oos_dates.append(dates[i])
        best_iterations.append(bst.best_iteration if hasattr(bst, "best_iteration") and bst.best_iteration is not None else NUM_BOOST_ROUND)

    oos_probs = np.array(oos_probs)
    oos_actual = np.array(oos_actual)

    # Simple returns for gain capture: Close_next / Close_feat - 1
    # model_df index = feature_asof date; y_cls = (Close_next > Close_feat)
    # We need the actual close-to-close return aligned to OOS sessions
    try:
        if "bars" in dir() and "Close" in bars.columns:
            closes = bars["Close"]
        else:
            data_dir = REPO / "data"
            cp = data_dir / f"cache_ohlcv_hist_{ticker}.parquet"
            if not cp.exists():
                cp = data_dir / f"cache_ohlcv_{ticker}.parquet"
            closes = pd.read_parquet(cp)["Close"]
    except Exception:
        closes = None

    # Fallback: derive returns from model_df itself if possible
    if closes is not None:
        simple_ret = (closes.shift(-1) / closes - 1.0)
        oos_returns = simple_ret.reindex(oos_dates).to_numpy(dtype=float)
        oos_returns = np.where(np.isfinite(oos_returns), oos_returns, 0.0)
    else:
        oos_returns = np.zeros(len(oos_actual))

    # Overall AUC metrics
    if len(set(oos_actual)) < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(oos_actual, oos_probs))
        roc_auc = float(roc_auc_score(oos_actual, oos_probs))

    rho = float(spearmanr(oos_probs, oos_actual).correlation) if len(set(oos_actual)) > 1 else float("nan")

    return {
        "oos_probs": oos_probs.tolist(),
        "oos_actual": oos_actual.tolist(),
        "oos_returns": oos_returns.tolist(),
        "n_oos": len(oos_actual),
        "n_pos": int(oos_actual.sum()),
        "base_up": float(oos_actual.mean()),
        "oos_pr_auc": pr_auc,
        "oos_roc_auc": roc_auc,
        "spearman_ic": rho,
        "avg_best_iter": float(np.mean(best_iterations)) if best_iterations else 0,
    }


# ─── Threshold scan with early-stop on precision decrease ───
def threshold_scan(oos_probs, oos_actual, oos_returns):
    """
    Replicate pipeline threshold scan with early-stop when precision decreases.

    Steps through thresholds 0.50 → 0.95 (0.01 steps) on oriented entry_score.
    Stops scanning as soon as precision starts decreasing (monotonic drop),
    since higher thresholds only reduce n_buys further.

    Returns dict with scan rows, best eligible threshold, and discrimination verdict.
    """
    oos_probs = np.asarray(oos_probs, dtype=float)
    oos_actual = np.asarray(oos_actual, dtype=int)
    oos_returns = np.asarray(oos_returns, dtype=float)

    base_up = float(oos_actual.mean())
    pass_floor = max(COIN_FLOOR, base_up)
    n_pos = int(oos_actual.sum())

    # Orient: if Spearman IC < 0, invert
    rho = float(spearmanr(oos_probs, oos_actual).correlation) if len(set(oos_actual)) > 1 else float("nan")
    auto_invert = bool(np.isfinite(rho) and rho < 0)
    entry_score = (1.0 - oos_probs) if auto_invert else oos_probs.copy()

    total_up_gain = float(oos_returns[oos_actual == 1].sum())

    thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)
    scan_rows = []
    prev_precision = -1.0
    stopped_early = False
    stop_reason = ""

    for thr in thresholds:
        m = entry_score >= thr
        n_buys = int(m.sum())
        n_wins = int((m & (oos_actual == 1)).sum())
        n_fp = int((m & (oos_actual == 0)).sum())

        precision = (n_wins / n_buys) if n_buys else float("nan")
        recall = (n_wins / n_pos) if n_pos else float("nan")

        tp_gain = float(oos_returns[m & (oos_actual == 1)].sum()) if n_wins else 0.0
        fp_loss = float(oos_returns[m & (oos_actual == 0)].sum()) if n_fp else 0.0
        net_captured = tp_gain + fp_loss
        gain_captured = (net_captured / total_up_gain) if total_up_gain > 0 else float("nan")
        er_per_buy = (net_captured / n_buys) if n_buys else float("nan")

        beats_both = bool(n_buys and np.isfinite(precision) and precision > pass_floor)
        eligible = bool(beats_both and n_buys >= MIN_TRADES)

        scan_rows.append({
            "thr": float(thr),
            "n_buys": n_buys,
            "n_wins": n_wins,
            "n_fp": n_fp,
            "precision": precision,
            "recall": recall,
            "er_per_buy": er_per_buy,
            "gain_captured": gain_captured,
            "eligible": eligible,
        })

        # Early-stop: if n_buys > 0 and precision decreased from previous threshold
        if n_buys > 0 and np.isfinite(precision) and np.isfinite(prev_precision) and precision < prev_precision:
            stopped_early = True
            stop_reason = f"precision dropped at thr={thr:.2f} ({precision:.1%} < {prev_precision:.1%})"
            break

        # Also stop if no buys at all
        if n_buys == 0 and len(scan_rows) > 1:
            stopped_early = True
            stop_reason = f"no buys at thr={thr:.2f}"
            break

        if n_buys > 0 and np.isfinite(precision):
            prev_precision = precision

    # Find best eligible threshold (max precision among eligible)
    eligible_rows = [r for r in scan_rows if r["eligible"]]
    is_discriminating = len(eligible_rows) > 0

    if is_discriminating:
        best = max(eligible_rows, key=lambda r: (r["precision"], r["n_wins"], r["n_buys"]))
        best_thr = best["thr"]
        best_prec = best["precision"]
        best_n_buys = best["n_buys"]
        best_n_wins = best["n_wins"]
        best_er = best["er_per_buy"]
        margin = best_prec - pass_floor
    else:
        best_thr = float("nan")
        best_prec = float("nan")
        best_n_buys = 0
        best_n_wins = 0
        best_er = float("nan")
        margin = float("nan")

    # Also find peak precision anywhere (even if not eligible)
    with_buys = [r for r in scan_rows if r["n_buys"] >= 1 and np.isfinite(r["precision"])]
    peak_prec = max((r["precision"] for r in with_buys), default=float("nan"))

    return {
        "scan_rows": scan_rows,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "n_thresholds_scanned": len(scan_rows),
        "auto_invert": auto_invert,
        "pass_floor": pass_floor,
        "base_up": base_up,
        "is_discriminating": is_discriminating,
        "best_thr": best_thr,
        "best_precision": best_prec,
        "best_n_buys": best_n_buys,
        "best_n_wins": best_n_wins,
        "best_er_per_buy": best_er,
        "margin_vs_floor": margin,
        "peak_precision": peak_prec,
    }


# ─── Worker for parallel execution ───
g_ticker = ""

def run_one_ticker_variant(args):
    """Run one ticker + one variant. Returns result dict with OOS arrays."""
    global g_ticker
    ticker, label, features = args
    g_ticker = ticker

    t0 = time.time()
    model_df, feature_cols = get_model_df(ticker, features)
    if model_df is None:
        return {"ticker": ticker, "label": label, "error": "failed to get model_df"}

    wf = walk_forward(model_df, feature_cols, ticker)
    elapsed = time.time() - t0

    if "error" in wf:
        return {"ticker": ticker, "label": label, "features": features,
                "elapsed": elapsed, "error": wf["error"]}

    result = {
        "ticker": ticker,
        "label": label,
        "features": features,
        "elapsed": elapsed,
        **wf,
    }
    print(f"  [{ticker}] {label}: PR-AUC={wf.get('oos_pr_auc', '?'):.4f}  "
          f"ROC-AUC={wf.get('oos_roc_auc', '?'):.4f}  "
          f"n_oos={wf['n_oos']}  "
          f"({elapsed:.1f}s)", flush=True)
    return result


# ─── Main ───
def main():
    jobs = []
    for label, features in VARIANTS.items():
        for ticker in TICKERS:
            jobs.append((ticker, label, features))

    print(f"Running {len(jobs)} jobs: {len(TICKERS)} tickers x {len(VARIANTS)} variants, {PARALLEL} parallel")
    print(f"XGBoost: custom asymmetric loss (penalty_weight={PENALTY_WEIGHT}), eval=aucpr, early_stop={EARLY_STOPPING}")
    print(f"OOS window: {VAL_WINDOW} sessions  |  Threshold scan: 0.50→0.95 with early-stop on precision decrease")
    print(f"Discrimination gate: precision > max({COIN_FLOOR:.0%}, base_up) AND n_buys >= {MIN_TRADES}")
    print()

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(run_one_ticker_variant, job): job for job in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  [{job[0]}] {job[1]}: FAILED — {e}")
                traceback.print_exc()
                results.append({"ticker": job[0], "label": job[1], "error": str(e)})

    total_elapsed = time.time() - t0
    print(f"\nWalk-forward total elapsed: {total_elapsed:.1f}s")

    # ─── Post-hoc threshold scan (instant) ───
    print("\nRunning threshold scan with early-stop...")
    scan_results = {}
    for r in results:
        if "error" in r or "oos_probs" not in r:
            continue
        key = (r["ticker"], r["label"])
        scan = threshold_scan(r["oos_probs"], r["oos_actual"], r["oos_returns"])
        scan_results[key] = scan

    # Save raw results (without OOS arrays to keep file small)
    slim_results = []
    for r in results:
        if "error" in r:
            slim_results.append(r)
            continue
        slim = {k: v for k, v in r.items() if k not in ("oos_probs", "oos_actual", "oos_returns")}
        key = (r["ticker"], r["label"])
        if key in scan_results:
            s = scan_results[key]
            slim["scan"] = {
                "stopped_early": s["stopped_early"],
                "stop_reason": s["stop_reason"],
                "n_thresholds_scanned": s["n_thresholds_scanned"],
                "auto_invert": s["auto_invert"],
                "pass_floor": s["pass_floor"],
                "is_discriminating": s["is_discriminating"],
                "best_thr": s["best_thr"],
                "best_precision": s["best_precision"],
                "best_n_buys": s["best_n_buys"],
                "best_n_wins": s["best_n_wins"],
                "best_er_per_buy": s["best_er_per_buy"],
                "margin_vs_floor": s["margin_vs_floor"],
                "peak_precision": s["peak_precision"],
                "scan_rows": s["scan_rows"],
            }
        slim_results.append(slim)

    out_path = REPO / "custom_xgb_6f_22tickers.json"
    out_path.write_text(json.dumps(slim_results, indent=2, default=str))
    print(f"Saved to {out_path}")

    # ─── Discrimination table ───
    print(f"\n{'='*140}")
    print(f"DISCRIMINATION TABLE: Custom XGBoost (asymmetric loss, aucpr eval) — {VAL_WINDOW} OOS sessions, threshold scan with early-stop")
    print(f"{'='*140}")
    print(f"\n{'Ticker':<7} {'Var':<9} {'base_up':>8} {'floor':>7} {'best_thr':>9} {'prec':>7} {'n_buys':>7} {'n_wins':>7} {'E[r|buy]':>9} {'margin':>8} {'PR-AUC':>7} {'ROC-AUC':>8} {'IC':>7} {'invert':>7} {'scanned':>8} {'Verdict':<10} Justification")
    print("-" * 140)

    for ticker in TICKERS:
        for label in VARIANTS:
            key = (ticker, label)
            r = next((x for x in results if x.get("ticker") == ticker and x.get("label") == label), None)
            if r is None or "error" in r:
                print(f"{ticker:<7} {label:<9}  ERROR: {r.get('error', '?') if r else 'missing'}")
                continue

            s = scan_results.get(key)
            if s is None:
                print(f"{ticker:<7} {label:<9}  SCAN MISSING")
                continue

            base_up = s["base_up"]
            pass_floor = s["pass_floor"]
            best_thr = s["best_thr"]
            best_prec = s["best_precision"]
            best_n_buys = s["best_n_buys"]
            best_n_wins = s["best_n_wins"]
            best_er = s["best_er_per_buy"]
            margin = s["margin_vs_floor"]
            pr_auc = r.get("oos_pr_auc", float("nan"))
            roc_auc = r.get("oos_roc_auc", float("nan"))
            ic = r.get("spearman_ic", float("nan"))
            invert = "YES" if s["auto_invert"] else "no"
            scanned = s["n_thresholds_scanned"]

            if s["is_discriminating"]:
                verdict = "PASS"
                justify = f"prec {best_prec:.1%} > floor {pass_floor:.1%} (+{margin:.1%}), n_buys={best_n_buys}≥{MIN_TRADES}"
            elif np.isfinite(best_prec) and best_prec > pass_floor and best_n_buys < MIN_TRADES:
                verdict = "MARGINAL"
                justify = f"prec {best_prec:.1%} > floor {pass_floor:.1%} but n_buys={best_n_buys}<{MIN_TRADES}"
            elif np.isfinite(s["peak_precision"]) and s["peak_precision"] > pass_floor:
                # Peak precision cleared floor but not enough trades
                peak = s["peak_precision"]
                justify = f"peak prec {peak:.1%} > floor {pass_floor:.1%} but n_buys<{MIN_TRADES} at all eligible thr"
                verdict = "MARGINAL"
            else:
                peak = s["peak_precision"]
                verdict = "FAIL"
                if np.isfinite(peak):
                    justify = f"peak prec {peak:.1%} ≤ floor {pass_floor:.1%} (gap {pass_floor - peak:+.1%})"
                else:
                    justify = "no buys at any threshold"

            er_str = f"{best_er:+.2%}" if np.isfinite(best_er) else "N/A"
            margin_str = f"+{margin:.1%}" if np.isfinite(margin) else "N/A"

            print(f"{ticker:<7} {label:<9} {base_up:>8.1%} {pass_floor:>7.1%} {best_thr:>9.0%} {best_prec:>7.1%} {best_n_buys:>7} {best_n_wins:>7} {er_str:>9} {margin_str:>8} {pr_auc:>7.3f} {roc_auc:>8.3f} {ic:>7.3f} {invert:>7} {scanned:>8} {verdict:<10} {justify}")

    # ─── Summary counts ───
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for label in VARIANTS:
        passes = sum(1 for t in TICKERS if scan_results.get((t, label), {}).get("is_discriminating", False))
        marg = sum(1 for t in TICKERS if not scan_results.get((t, label), {}).get("is_discriminating", False)
                   and np.isfinite(scan_results.get((t, label), {}).get("peak_precision", float("nan")))
                   and scan_results.get((t, label), {}).get("peak_precision", 0) > scan_results.get((t, label), {}).get("pass_floor", 1))
        fails = len(TICKERS) - passes - marg
        avg_pr = np.nanmean([r.get("oos_pr_auc", float("nan")) for r in results if r.get("label") == label and "oos_pr_auc" in r])
        avg_roc = np.nanmean([r.get("oos_roc_auc", float("nan")) for r in results if r.get("label") == label and "oos_roc_auc" in r])
        print(f"  {label:<10}: PASS={passes}  MARGINAL={marg}  FAIL={fails}  (of {len(TICKERS)} tickers)  avg PR-AUC={avg_pr:.4f}  avg ROC-AUC={avg_roc:.4f}")

    # List tickers by verdict
    print(f"\n  PASS tickers:")
    for t in TICKERS:
        s = scan_results.get((t, "6f"))
        if s and s["is_discriminating"]:
            print(f"    {t:<7} prec={s['best_precision']:.1%} thr={s['best_thr']:.0%} n_buys={s['best_n_buys']} margin=+{s['margin_vs_floor']:.1%}")
    print(f"\n  MARGINAL tickers:")
    for t in TICKERS:
        s = scan_results.get((t, "6f"))
        if s and not s["is_discriminating"] and np.isfinite(s["peak_precision"]) and s["peak_precision"] > s["pass_floor"]:
            print(f"    {t:<7} peak_prec={s['peak_precision']:.1%} floor={s['pass_floor']:.1%} (n_buys < {MIN_TRADES})")
    print(f"\n  FAIL tickers:")
    for t in TICKERS:
        s = scan_results.get((t, "6f"))
        if s and not s["is_discriminating"]:
            if not (np.isfinite(s["peak_precision"]) and s["peak_precision"] > s["pass_floor"]):
                peak = s["peak_precision"]
                if np.isfinite(peak):
                    print(f"    {t:<7} peak_prec={peak:.1%} ≤ floor={s['pass_floor']:.1%}")
                else:
                    print(f"    {t:<7} no buys at any threshold")


if __name__ == "__main__":
    main()
