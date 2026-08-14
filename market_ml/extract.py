"""Re-extract production stages from market_ml_analysis.ipynb → pipeline.py."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "market_ml_analysis.ipynb"
OUT = Path(__file__).resolve().parent / "pipeline.py"

# Canonical production path (setup → buy rule / dir-hits). Optional full-WF cells excluded.
# NOTE: feat_adx, feat_candle, feat_fred stages are hand-added to pipeline.py
# (not yet in the notebook). If re-extracting from notebook, these stages
# will be missing — re-add them manually or add cells to the notebook first.
PROD_CELLS = [2, 4, 6, 9, 10, 11, 12, 13, 15, 14, 16, 18, 20, 30, 22]

STAGE_NAMES = {
    2: "setup",
    4: "load_ohlcv",
    6: "feature_registry",
    9: "feat_rsi",
    10: "feat_vol_ratio",
    11: "feat_near_high",
    12: "feat_obv",
    13: "feat_bb",
    15: "feat_trend",
    14: "assemble",
    16: "feature_health",
    18: "plot_01",
    20: "walkforward_pred",
    30: "calibrate",
    22: "buy_rule",
}


def transform(src: str) -> str:
    src = src.replace("from IPython.display import display\n", "")
    src = src.replace("from IPython.display import display", "")
    src = re.sub(r"^\s*plt\.show\(\)\s*$", "    plt.close('all')", src, flags=re.M)
    src = src.replace("plt.show()", "plt.close('all')")
    src = re.sub(
        r"display\(([^)]+)\.style\.format\([^)]*\)\)",
        r"display(\1)",
        src,
    )
    src = src.replace(
        "This notebook is the only script: single stock, feature ideation, thr pick, predict.",
        "Production runner reuses notebook pipeline stages; ideation stays in the notebook.",
    )
    return src


def build() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    cells = nb["cells"]

    def cell_src(i: int) -> str:
        return "".join(cells[i].get("source", []))

    header = dedent(
        '''\
        """Market ML production pipeline (extracted from market_ml_analysis.ipynb).

        Stages: config → OHLCV → features → assemble → health/boards →
        last-N WF + pred → entry calibration → buy rule + dir hits.

        Call ``run_ticker(ticker, ...)`` from CLI or import. Parallelize across
        processes via ``run_tickers(...)`` / ``run_market_ml.py``.

        Charts are produced only via the same ``plot_*`` functions as the notebook.
        Do not edit stages by hand for experiments — iterate in the notebook, then
        re-run: ``python -m market_ml.extract``.
        """
        from __future__ import annotations

        # Headless / multi-process safe
        import matplotlib
        matplotlib.use("Agg")

        import json
        import os
        import re
        import sys
        import traceback
        import warnings
        from dataclasses import dataclass
        from datetime import datetime, time
        from pathlib import Path
        from typing import Any
        from zoneinfo import ZoneInfo

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Lasso, LassoCV
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        warnings.filterwarnings("ignore")


        def display(obj: Any = None, *args, **kwargs) -> None:
            """Notebook compatibility shim (print frames / series)."""
            if obj is None:
                return
            try:
                if hasattr(obj, "to_string"):
                    print(obj.to_string())
                else:
                    print(obj)
            except Exception:
                print(repr(obj))


        @dataclass
        class RunConfig:
            ticker: str
            repo_root: Path | str | None = None
            start: str | None = None
            end: str | None = None
            predict_target_date: str | None = None
            feature_validate_date: str | None = None
            val_window_sessions: int = 30
            buy_entry_thr_mode: str = "composite"
            buy_entry_score_min: float = 0.72
            buy_require_lasso_up: bool = False
            force_recalibrate: bool = False
            skip_feature_health: bool = False
            skip_plot_01: bool = False
            n_jobs_model: int = 1
            quiet: bool = False


        def _apply_overrides(cfg: RunConfig, g: dict[str, Any]) -> None:
            """Mutate stage globals after setup block with CLI overrides."""
            g["TICKER"] = cfg.ticker.upper().strip()
            if cfg.repo_root is not None:
                g["REPO_ROOT"] = Path(cfg.repo_root).resolve()
            else:
                g["REPO_ROOT"] = Path(__file__).resolve().parent.parent
            g["DATA_DIR"] = g["REPO_ROOT"] / "data"
            g["DATA_DIR"].mkdir(parents=True, exist_ok=True)
            if cfg.start:
                g["START"] = cfg.start
                g["START_TS"] = pd.Timestamp(cfg.start)
                g["FETCH_START"] = (
                    g["START_TS"] - pd.Timedelta(days=g["PRE_START_DAYS"])
                ).strftime("%Y-%m-%d")
            if cfg.end is not None:
                g["END"] = cfg.end
            if cfg.predict_target_date:
                g["PREDICT_TARGET_DATE"] = cfg.predict_target_date
                g["RUN_DATE"] = str(cfg.predict_target_date)
                if not cfg.feature_validate_date:
                    g["FEATURE_VALIDATE_DATE"] = (
                        pd.Timestamp(cfg.predict_target_date) - pd.offsets.BDay(1)
                    ).strftime("%Y-%m-%d")
            if cfg.feature_validate_date:
                g["FEATURE_VALIDATE_DATE"] = cfg.feature_validate_date
            g["VAL_WINDOW_SESSIONS"] = int(cfg.val_window_sessions)
            g["BUY_ENTRY_THR_MODE"] = cfg.buy_entry_thr_mode
            g["BUY_ENTRY_SCORE_MIN"] = float(cfg.buy_entry_score_min)
            g["BUY_XGB_PROB_MIN"] = float(cfg.buy_entry_score_min)
            g["BUY_REQUIRE_LASSO_UP"] = bool(cfg.buy_require_lasso_up)
            g["OUTPUT_DIR"] = g["REPO_ROOT"] / f"improves_{g['TICKER']}_{g['RUN_DATE']}"
            g["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
            g["_N_JOBS_MODEL"] = int(cfg.n_jobs_model)
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cfg.n_jobs_model)))
            os.environ.setdefault("MKL_NUM_THREADS", str(max(1, cfg.n_jobs_model)))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, cfg.n_jobs_model)))


        '''
    )

    parts = [header]
    parts.append("\n_STAGE_SOURCES: dict[str, str] = {}\n\n")
    for idx in PROD_CELLS:
        name = STAGE_NAMES[idx]
        safe = transform(cell_src(idx)).replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
        parts.append(f"_STAGE_SOURCES[{name!r}] = '''\\\n{safe}\n'''\n\n")

    parts.append(
        dedent(
            '''\
            _STAGE_ORDER = [
                "setup",
                "load_ohlcv",
                "feature_registry",
                "feat_rsi",
                "feat_vol_ratio",
                "feat_near_high",
                "feat_obv",
                "feat_bb",
                "feat_trend",
                "feat_adx",
                "feat_candle",
                "feat_fred",
                "assemble",
                "feature_health",
                "plot_01",
                "walkforward_pred",
                "calibrate",
                "buy_rule",
            ]


            def _patch_model_n_jobs(src: str, n_jobs: int) -> str:
                """Pin inner XGB / LassoCV jobs for process-level parallelism."""
                src = re.sub(r"n_jobs\\s*=\\s*-1", f"n_jobs={n_jobs}", src)
                src = re.sub(r"n_jobs\\s*=\\s*2\\b", f"n_jobs={n_jobs}", src)
                return src


            def run_ticker(cfg: RunConfig | str, **kwargs) -> dict[str, Any]:
                """Run full production pipeline for one ticker. Returns summary dict."""
                if isinstance(cfg, str):
                    cfg = RunConfig(ticker=cfg, **kwargs)
                elif kwargs:
                    for k, v in kwargs.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)

                g: dict[str, Any] = {
                    "__name__": f"market_ml.pipeline.{cfg.ticker}",
                    "__builtins__": __builtins__,
                    "display": display,
                    "np": np,
                    "pd": pd,
                    "plt": plt,
                    "Path": Path,
                    "warnings": warnings,
                    "json": json,
                    "datetime": datetime,
                    "time": time,
                    "ZoneInfo": ZoneInfo,
                    "Pipeline": Pipeline,
                    "StandardScaler": StandardScaler,
                    "Lasso": Lasso,
                    "LassoCV": LassoCV,
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
                }

                import matplotlib.dates as mdates
                from matplotlib.patches import Rectangle, Patch
                from matplotlib.lines import Line2D
                from scipy import stats
                from scipy.stats import spearmanr

                g["mdates"] = mdates
                g["Rectangle"] = Rectangle
                g["Patch"] = Patch
                g["Line2D"] = Line2D
                g["stats"] = stats
                g["spearmanr"] = spearmanr

                # Defined in notebook feature-health cell; needed by plot_01 / plot_03
                # even when --skip-feature-health is set.
                def annotate_prev_close_right(ax, close: float, *, linewidth: float = 1.15) -> None:
                    """Outboard right label only (no hline): ← $XX.XX at last/prev close level."""
                    try:
                        c = float(close)
                    except (TypeError, ValueError):
                        return
                    if not np.isfinite(c):
                        return
                    _ = linewidth
                    ax.annotate(
                        f"← ${c:.2f}",
                        xy=(1.0, c),
                        xycoords=("axes fraction", "data"),
                        xytext=(6, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                        clip_on=False,
                        annotation_clip=False,
                        zorder=8,
                    )

                g["annotate_prev_close_right"] = annotate_prev_close_right

                stages = list(_STAGE_ORDER)
                if cfg.skip_feature_health:
                    stages = [s for s in stages if s != "feature_health"]
                if cfg.skip_plot_01:
                    stages = [s for s in stages if s != "plot_01"]

                summary: dict[str, Any] = {
                    "ticker": cfg.ticker.upper().strip(),
                    "ok": False,
                    "error": None,
                    "stages_done": [],
                }

                try:
                    for stage in stages:
                        src = _patch_model_n_jobs(_STAGE_SOURCES[stage], cfg.n_jobs_model)

                        if stage == "setup":
                            src = re.sub(
                                r'^TICKER\\s*=\\s*["\\'][^"\\']*["\\']',
                                f'TICKER = {cfg.ticker.upper().strip()!r}',
                                src,
                                count=1,
                                flags=re.M,
                            )
                            exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
                            _apply_overrides(cfg, g)
                            if not cfg.quiet:
                                print(
                                    f"[{g['TICKER']}] OUTPUT_DIR={g['OUTPUT_DIR']}  "
                                    f"target={g['PREDICT_TARGET_DATE']}  "
                                    f"validate={g['FEATURE_VALIDATE_DATE']}"
                                )
                        elif stage == "calibrate":
                            if cfg.force_recalibrate:
                                panel = g["OUTPUT_DIR"] / f"wf_dir_panel_{g['TICKER']}.parquet"
                                if panel.exists():
                                    panel.unlink()
                                    if not cfg.quiet:
                                        print(f"[{g['TICKER']}] removed stale {panel.name}")
                            exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
                        else:
                            exec(compile(src, f"<stage:{stage}>", "exec"), g, g)

                        summary["stages_done"].append(stage)
                        if not cfg.quiet:
                            print(f"[{g.get('TICKER', cfg.ticker)}] stage ok: {stage}")

                    t = g["TICKER"]
                    out = g["OUTPUT_DIR"]
                    pred_path = out / f"pred_{g['PREDICT_TARGET_DATE']}_{t}.json"
                    buy_path = out / f"buy_decision_{g['PREDICT_TARGET_DATE']}_{t}.json"
                    rule_path = out / f"entry_rule_80_{t}.json"
                    summary.update({
                        "ok": True,
                        "ticker": t,
                        "predict_target_date": g["PREDICT_TARGET_DATE"],
                        "output_dir": str(out),
                        "pred_path": str(pred_path) if pred_path.exists() else None,
                        "buy_path": str(buy_path) if buy_path.exists() else None,
                        "rule_path": str(rule_path) if rule_path.exists() else None,
                    })
                    if rule_path.exists():
                        rule = json.loads(rule_path.read_text())
                        summary["is_discriminating"] = rule.get("is_discriminating")
                        summary["auto_invert"] = rule.get("auto_invert")
                        summary["operating_entry_score_min"] = rule.get(
                            "operating_entry_score_min"
                        )
                        pend = rule.get("pending") or {}
                        summary["enter_under_best_rule"] = pend.get("enter_under_best_rule")
                        summary["entry_score"] = pend.get("entry_score")
                    if buy_path.exists():
                        buy = json.loads(buy_path.read_text())
                        for k in ("decision", "action", "buy", "enter", "live_buy"):
                            if k in buy:
                                summary["decision"] = buy[k]
                                break
                        summary["buy_decision"] = buy
                    plt.close("all")
                except Exception as e:
                    summary["ok"] = False
                    summary["error"] = f"{type(e).__name__}: {e}"
                    summary["traceback"] = traceback.format_exc()
                    plt.close("all")
                    if not cfg.quiet:
                        print(f"[{cfg.ticker}] FAILED: {summary['error']}", file=sys.stderr)
                        print(summary["traceback"], file=sys.stderr)
                return summary


            def _worker(payload: dict[str, Any]) -> dict[str, Any]:
                return run_ticker(RunConfig(**payload))


            def run_tickers(
                tickers: list[str],
                *,
                jobs: int = 1,
                **cfg_kwargs,
            ) -> list[dict[str, Any]]:
                """Run many tickers; process pool when jobs > 1."""
                from concurrent.futures import ProcessPoolExecutor, as_completed

                payloads: list[dict[str, Any]] = []
                for t in tickers:
                    p = {"ticker": t, **cfg_kwargs}
                    if p.get("repo_root") is not None:
                        p["repo_root"] = str(p["repo_root"])
                    payloads.append(p)

                if jobs <= 1 or len(payloads) == 1:
                    return [run_ticker(RunConfig(**p)) for p in payloads]

                results: list[dict[str, Any]] = []
                with ProcessPoolExecutor(max_workers=jobs) as ex:
                    futs = {ex.submit(_worker, p): p["ticker"] for p in payloads}
                    for fut in as_completed(futs):
                        t = futs[fut]
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            results.append({
                                "ticker": t,
                                "ok": False,
                                "error": f"{type(e).__name__}: {e}",
                                "traceback": traceback.format_exc(),
                            })
                by_t = {str(r.get("ticker", "")).upper(): r for r in results}
                ordered = []
                for t in tickers:
                    ordered.append(
                        by_t.get(
                            t.upper(),
                            {"ticker": t, "ok": False, "error": "missing result"},
                        )
                    )
                return ordered
            '''
        )
    )
    return "".join(parts)


def main() -> int:
    text = build()
    compile(text, str(OUT), "exec")
    OUT.write_text(text)
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"Stages: {list(STAGE_NAMES.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
