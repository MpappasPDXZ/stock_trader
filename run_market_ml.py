#!/usr/bin/env python3
"""CLI: run market ML pipeline for one or many tickers (optionally in parallel).

Examples
--------
  # Single ticker (full stages)
  python run_market_ml.py CRM

  # Many tickers, 4 worker processes (inner model threads pinned to 1)
  python run_market_ml.py CRM CAT OUST --jobs 4

  # Fixed target date + force fresh WF panel for calibration
  python run_market_ml.py VRT CAT --target 2026-08-12 --force-recalibrate

  # Faster smoke path (skip slow feature-health + chart 01)
  python run_market_ml.py CRM --skip-feature-health --skip-plot-01

Notebook ``market_ml_analysis.ipynb`` remains the place for feature ideation.
Re-sync production stages after notebook edits::

  python -m market_ml.extract
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run market ML analysis pipeline for ticker(s) in parallel.",
    )
    p.add_argument(
        "tickers",
        nargs="+",
        help="One or more stock tickers (e.g. CRM CAT OUST)",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Process pool size (default 1). Use N>1 for parallel tickers.",
    )
    p.add_argument(
        "--target",
        dest="predict_target_date",
        default=None,
        help="PREDICT_TARGET_DATE YYYY-MM-DD (default: clock-aware like notebook)",
    )
    p.add_argument(
        "--validate-date",
        dest="feature_validate_date",
        default=None,
        help="FEATURE_VALIDATE_DATE YYYY-MM-DD (default: BDay before target)",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Analysis START date YYYY-MM-DD (default: ~1y lookback)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="yfinance end date (default: latest)",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (default: directory containing this script)",
    )
    p.add_argument(
        "--val-window",
        type=int,
        default=30,
        help="VAL_WINDOW_SESSIONS (default 30)",
    )
    p.add_argument(
        "--thr-mode",
        choices=("composite", "manual"),
        default="composite",
        help="BUY_ENTRY_THR_MODE",
    )
    p.add_argument(
        "--entry-score-min",
        type=float,
        default=0.72,
        help="BUY_ENTRY_SCORE_MIN when thr-mode=manual (or composite fallback)",
    )
    p.add_argument(
        "--require-lasso-up",
        action="store_true",
        help="Require Lasso pred_logret > 0 in addition to entry score thr",
    )
    p.add_argument(
        "--force-recalibrate",
        action="store_true",
        help="Delete wf_dir_panel_*.parquet before calibration (fresh OOS panel)",
    )
    p.add_argument(
        "--skip-feature-health",
        action="store_true",
        help="Skip Step 2b diagnostic WF + 00 chart (much faster)",
    )
    p.add_argument(
        "--skip-plot-01",
        action="store_true",
        help="Skip multi-panel 01_features_outcome_bb chart",
    )
    p.add_argument(
        "--model-jobs",
        type=int,
        default=None,
        help="Threads per model fit (default: 1 when --jobs>1, else 2)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less per-stage logging",
    )
    p.add_argument(
        "--active-features",
        default=None,
        help="Comma-separated feature names to use (default: all in FEATURE_SPECS). "
             "Example: --active-features rsi_14,obv_chg_z_20,bb_dist_lower,vol_ratio_7_21,near_high_200,med_term_21wk,wick_asymmetry",
    )
    p.add_argument(
        "--summary-json",
        default=None,
        help="Write run summaries JSON to this path",
    )
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent
    # Ensure local package import when run as script
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from market_ml.pipeline import run_tickers

    jobs = max(1, int(args.jobs))
    model_jobs = args.model_jobs
    if model_jobs is None:
        model_jobs = 1 if jobs > 1 else 2

    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if not tickers:
        p.error("No tickers provided")

    print(
        f"Running {len(tickers)} ticker(s) with jobs={jobs} model_jobs={model_jobs}: "
        f"{', '.join(tickers)}"
    )
    active_features = None
    if args.active_features:
        active_features = [f.strip() for f in args.active_features.split(",") if f.strip()]

    t0 = time.perf_counter()
    results = run_tickers(
        tickers,
        jobs=jobs,
        repo_root=str(repo),
        start=args.start,
        end=args.end,
        predict_target_date=args.predict_target_date,
        feature_validate_date=args.feature_validate_date,
        val_window_sessions=args.val_window,
        buy_entry_thr_mode=args.thr_mode,
        buy_entry_score_min=args.entry_score_min,
        buy_require_lasso_up=args.require_lasso_up,
        force_recalibrate=args.force_recalibrate,
        skip_feature_health=args.skip_feature_health,
        skip_plot_01=args.skip_plot_01,
        n_jobs_model=model_jobs,
        quiet=args.quiet,
        active_features=active_features,
    )
    elapsed = time.perf_counter() - t0

    # Compact card
    print("\n" + "=" * 72)
    print(f"DONE in {elapsed:.1f}s")
    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"ok={ok_n}/{len(results)}")
    for r in results:
        t = r.get("ticker", "?")
        if not r.get("ok"):
            print(f"  {t}: FAIL — {r.get('error')}")
            continue
        dec = r.get("decision")
        disc = r.get("is_discriminating")
        thr = r.get("operating_entry_score_min")
        es = r.get("entry_score")
        thr_s = f"{thr:.0%}" if isinstance(thr, (int, float)) and thr == thr else "?"
        es_s = f"{es:.1%}" if isinstance(es, (int, float)) and es == es else "?"
        print(
            f"  {t}: decision={dec}  disc={disc}  thr≥{thr_s}  "
            f"score={es_s}  out={r.get('output_dir')}"
        )
    print("=" * 72)

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        # drop large nested buy payload for summary file if present
        slim = []
        for r in results:
            d = {k: v for k, v in r.items() if k not in ("buy_decision", "traceback")}
            if not d.get("ok") and r.get("traceback"):
                d["traceback"] = r["traceback"]
            slim.append(d)
        path.write_text(json.dumps(slim, indent=2, default=str))
        print(f"Wrote {path}")

    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
