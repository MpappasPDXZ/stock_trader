#!/usr/bin/env python3
"""
6f baseline penalty sweep: 1.05, 1.10, 1.15, 1.20, 1.25
22 tickers, custom XGBoost (aucpr eval, asymmetric loss).
Walk-forward runs all 5 penalties per ticker in one worker (model_df loaded once).
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
PENALTIES = [1.05, 1.10, 1.15, 1.20, 1.25]
PARALLEL = 3
NUM_BOOST_ROUND = 100
EARLY_STOPPING = 10
MIN_TRAIN = 80
VAL_WINDOW = 60

COIN_FLOOR = 0.50
MIN_TRADES = 10


def asymmetric_logistic_loss(y_true, y_pred, penalty_weight=1.25):
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    y_true = np.asarray(y_true, dtype=np.float64)
    grad = p * (1.0 + (penalty_weight - 1.0) * (1.0 - y_true)) - y_true
    hess = p * (1.0 - p) * (1.0 + (penalty_weight - 1.0) * (1.0 - y_true))
    return grad, hess


def custom_aucpr_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    return "aucpr", average_precision_score(y_true, p)


def get_model_df(ticker, active_features):
    from market_ml.pipeline import _STAGE_SOURCES, RunConfig

    cfg = RunConfig(
        ticker=ticker,
        active_features=active_features.split(","),
        quiet=True,
        skip_feature_health=True,
        skip_plot_01=True,
    )

    stages = [
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
    }

    def _patch_n_jobs(src, n_jobs):
        src = re.sub(r"n_jobs\s*=\s*-1", f"n_jobs={n_jobs}", src)
        src = re.sub(r"n_jobs\s*=\s*2\b", f"n_jobs={n_jobs}", src)
        return src

    try:
        for stage in stages:
            src = _patch_n_jobs(_STAGE_SOURCES[stage], cfg.n_jobs_model)
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

        return g["model_df"], g["FEATURE_COLS"]
    except Exception as e:
        print(f"  [ERROR {ticker}] {e}")
        traceback.print_exc()
        return None, None


def walk_forward(model_df, feature_cols, ticker, penalty):
    X = model_df[feature_cols]
    y = model_df["y_cls"].astype(int).to_numpy()
    n = len(model_df)
    dates = model_df.index

    val_n = min(VAL_WINDOW, n - MIN_TRAIN)
    if val_n < 5:
        return {"error": f"not enough data: n={n}"}

    wf_start = n - val_n
    oos_probs, oos_actual, oos_dates = [], [], []

    for i in range(wf_start, n):
        X_train = X.iloc[:i].to_numpy()
        y_train = y[:i]
        X_test = X.iloc[i:i+1].to_numpy()

        split_idx = max(int(len(X_train) * 0.8), MIN_TRAIN - 10)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        if len(set(y_val)) < 2:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)
            bst = xgb.train(
                {"disable_default_eval_metric": 1},
                dtrain, num_boost_round=NUM_BOOST_ROUND,
                obj=lambda yp, dt: asymmetric_logistic_loss(dt.get_label(), yp, penalty_weight=penalty),
                custom_metric=custom_aucpr_eval,
                evals=[(dtrain, "train")],
                verbose_eval=False,
            )
        else:
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)
            bst = xgb.train(
                {"disable_default_eval_metric": 1},
                dtrain, num_boost_round=NUM_BOOST_ROUND,
                obj=lambda yp, dt: asymmetric_logistic_loss(dt.get_label(), yp, penalty_weight=penalty),
                custom_metric=custom_aucpr_eval,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=EARLY_STOPPING,
                verbose_eval=False,
            )

        oos_probs.append(float(bst.predict(dtest)[0]))
        oos_actual.append(y[i])
        oos_dates.append(dates[i])

    oos_probs = np.array(oos_probs)
    oos_actual = np.array(oos_actual)

    # returns
    try:
        data_dir = REPO / "data"
        cp = data_dir / f"cache_ohlcv_hist_{ticker}.parquet"
        if not cp.exists():
            cp = data_dir / f"cache_ohlcv_{ticker}.parquet"
        closes = pd.read_parquet(cp)["Close"]
        simple_ret = closes.shift(-1) / closes - 1.0
        oos_returns = simple_ret.reindex(oos_dates).to_numpy(dtype=float)
        oos_returns = np.where(np.isfinite(oos_returns), oos_returns, 0.0)
    except Exception:
        oos_returns = np.zeros(len(oos_actual))

    pr_auc = float(average_precision_score(oos_actual, oos_probs)) if len(set(oos_actual)) > 1 else float("nan")
    roc_auc = float(roc_auc_score(oos_actual, oos_probs)) if len(set(oos_actual)) > 1 else float("nan")
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
    }


def threshold_scan(oos_probs, oos_actual, oos_returns):
    oos_probs = np.asarray(oos_probs, dtype=float)
    oos_actual = np.asarray(oos_actual, dtype=int)
    oos_returns = np.asarray(oos_returns, dtype=float)

    base_up = float(oos_actual.mean())
    pass_floor = max(COIN_FLOOR, base_up)
    n_pos = int(oos_actual.sum())

    rho = float(spearmanr(oos_probs, oos_actual).correlation) if len(set(oos_actual)) > 1 else float("nan")
    auto_invert = bool(np.isfinite(rho) and rho < 0)
    entry_score = (1.0 - oos_probs) if auto_invert else oos_probs.copy()

    thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)
    scan_rows = []
    prev_precision = -1.0
    stopped_early = False

    for thr in thresholds:
        m = entry_score >= thr
        n_buys = int(m.sum())
        n_wins = int((m & (oos_actual == 1)).sum())
        precision = (n_wins / n_buys) if n_buys else float("nan")
        eligible = bool(n_buys and np.isfinite(precision) and precision > pass_floor and n_buys >= MIN_TRADES)
        scan_rows.append({"thr": float(thr), "n_buys": n_buys, "n_wins": n_wins,
                          "precision": precision, "eligible": eligible})
        if n_buys > 0 and np.isfinite(precision) and np.isfinite(prev_precision) and precision < prev_precision:
            stopped_early = True
            break
        if n_buys == 0 and len(scan_rows) > 1:
            stopped_early = True
            break
        if n_buys > 0 and np.isfinite(precision):
            prev_precision = precision

    eligible_rows = [r for r in scan_rows if r["eligible"]]
    is_disc = len(eligible_rows) > 0
    if is_disc:
        best = max(eligible_rows, key=lambda r: (r["precision"], r["n_wins"], r["n_buys"]))
        best_thr = best["thr"]
        best_prec = best["precision"]
        best_n_buys = best["n_buys"]
        best_n_wins = best["n_wins"]
        margin = best_prec - pass_floor
    else:
        best_thr = float("nan")
        best_prec = float("nan")
        best_n_buys = 0
        best_n_wins = 0
        margin = float("nan")

    with_buys = [r for r in scan_rows if r["n_buys"] >= 1 and np.isfinite(r["precision"])]
    peak_prec = max((r["precision"] for r in with_buys), default=float("nan"))

    return {
        "is_discriminating": is_disc,
        "best_thr": best_thr,
        "best_precision": best_prec,
        "best_n_buys": best_n_buys,
        "best_n_wins": best_n_wins,
        "margin_vs_floor": margin,
        "peak_precision": peak_prec,
        "pass_floor": pass_floor,
        "base_up": base_up,
        "auto_invert": auto_invert,
        "n_thresholds_scanned": len(scan_rows),
    }


def run_one_ticker(ticker):
    t0 = time.time()
    model_df, feature_cols = get_model_df(ticker, BASE_6)
    if model_df is None:
        return {"ticker": ticker, "error": "failed to get model_df"}

    results = {}
    for penalty in PENALTIES:
        wf = walk_forward(model_df, feature_cols, ticker, penalty)
        if "error" in wf:
            results[penalty] = {"error": wf["error"]}
            continue
        scan = threshold_scan(wf["oos_probs"], wf["oos_actual"], wf["oos_returns"])
        results[penalty] = {
            "oos_pr_auc": wf["oos_pr_auc"],
            "oos_roc_auc": wf["oos_roc_auc"],
            "spearman_ic": wf["spearman_ic"],
            "base_up": wf["base_up"],
            "scan": scan,
        }
        print(f"  [{ticker}] penalty={penalty:.2f}: PR-AUC={wf['oos_pr_auc']:.4f}  "
              f"disc={'PASS' if scan['is_discriminating'] else 'FAIL'}  "
              f"prec={scan['best_precision']:.1%}  thr={scan['best_thr']:.0%}  "
              f"n_buys={scan['best_n_buys']}", flush=True)

    elapsed = time.time() - t0
    return {"ticker": ticker, "elapsed": elapsed, **results}


def main():
    print(f"Penalty sweep: {PENALTIES}")
    print(f"22 tickers, 6f, custom XGB (aucpr), {PARALLEL} parallel, {VAL_WINDOW} OOS sessions")
    print()

    all_results = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(run_one_ticker, t): t for t in TICKERS}
        for fut in as_completed(futs):
            ticker = futs[fut]
            try:
                r = fut.result()
                all_results[r["ticker"]] = r
            except Exception as e:
                print(f"  [{ticker}] FAILED: {e}")
                traceback.print_exc()
                all_results[ticker] = {"ticker": ticker, "error": str(e)}

    total = time.time() - t0
    print(f"\nTotal elapsed: {total:.1f}s")

    # Save
    out = REPO / "penalty_sweep_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Saved to {out}")

    # ─── Comparison table ───
    print(f"\n{'='*120}")
    print("PENALTY SWEEP: Discrimination comparison (22 tickers × 5 penalties)")
    print(f"{'='*120}")
    header = f"{'Ticker':<7}"
    for p in PENALTIES:
        header += f" | {p:.2f} prec  n_buys  verdict"
    print(header)
    print("-" * 120)

    pass_counts = {p: 0 for p in PENALTIES}
    avg_prauc = {p: [] for p in PENALTIES}

    for ticker in TICKERS:
        row = f"{ticker:<7}"
        for p in PENALTIES:
            r = all_results.get(ticker, {})
            if "error" in r:
                row += f" |    ERROR              "
                continue
            pr = r.get(str(p) if str(p) in r else p, r.get(p, {}))
            if "error" in pr:
                row += f" |    ERROR              "
                continue
            scan = pr.get("scan", {})
            is_disc = scan.get("is_discriminating", False)
            prec = scan.get("best_precision", float("nan"))
            n_buys = scan.get("best_n_buys", 0)
            verdict = "PASS" if is_disc else "FAIL"
            if is_disc:
                pass_counts[p] += 1
            if "oos_pr_auc" in pr and np.isfinite(pr["oos_pr_auc"]):
                avg_prauc[p].append(pr["oos_pr_auc"])

            prec_str = f"{prec:.1%}" if np.isfinite(prec) else "  —  "
            row += f" | {prec_str:>7}  {n_buys:>4}  {verdict:<6}"
        print(row)

    # ─── Summary ───
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"{'Penalty':<10} {'PASS':>6} {'FAIL':>6} {'avg PR-AUC':>12}")
    print("-" * 40)
    for p in PENALTIES:
        n_pass = pass_counts[p]
        n_fail = 22 - n_pass
        avg_p = np.mean(avg_prauc[p]) if avg_prauc[p] else float("nan")
        print(f"{p:<10.2f} {n_pass:>6} {n_fail:>6} {avg_p:>12.4f}")

    # Per-ticker best penalty
    print(f"\nBest penalty per ticker (by precision, PASS only):")
    print(f"{'Ticker':<7} {'Best penalty':>13} {'prec':>8} {'n_buys':>8}")
    print("-" * 40)
    for ticker in TICKERS:
        r = all_results.get(ticker, {})
        if "error" in r:
            continue
        best_p = None
        best_prec = -1
        best_n = 0
        for p in PENALTIES:
            pr = r.get(str(p) if str(p) in r else p, r.get(p, {}))
            if "error" in pr:
                continue
            scan = pr.get("scan", {})
            if scan.get("is_discriminating") and scan.get("best_precision", 0) > best_prec:
                best_p = p
                best_prec = scan["best_precision"]
                best_n = scan["best_n_buys"]
        if best_p:
            print(f"{ticker:<7} {best_p:>13.2f} {best_prec:>8.1%} {best_n:>8}")
        else:
            print(f"{ticker:<7} {'none (FAIL)':>13}")


if __name__ == "__main__":
    main()
