"""Live intraday inference.

Pull today's session-so-far OHLCV from yfinance, append as a completed bar,
and run the same canonical pipeline stages so features + prediction are
as-of *today's* session (not yesterday's). Reuses ``_STAGE_SOURCES`` — no
second feature stack.

Public API:
    fetch_live_bar(ticker, override_close=None) -> pd.Series | None
    live_predict(ticker, target_date=None, override_close=None, ...) -> dict
"""
from __future__ import annotations

import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_ml.pipeline import (
    RunConfig,
    _STAGE_ORDER,
    _STAGE_SOURCES,
    _apply_overrides,
    _patch_model_n_jobs,
)
from market_ml.jobs import _seed_globals

_TZ = ZoneInfo("America/Chicago")
_NY_TZ = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# 1. Fetch today's intraday bar from yfinance
# --------------------------------------------------------------------------- #
def fetch_live_bar(
    ticker: str,
    *,
    override_close: float | None = None,
    interval: str = "5m",
) -> pd.Series | None:
    """Today's session-so-far OHLCV from yfinance intraday.

    Returns a Series (name = today's date as tz-naive Timestamp) with
    Open / High / Low / Close / Volume.  Close = last traded price unless
    ``override_close`` is given (hypothetical).  Returns ``None`` when no
    intraday data is available (pre-market, holiday, bad ticker).
    """
    import yfinance as yf

    raw = yf.download(ticker, period="1d", interval=interval, progress=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in raw.columns:
            return None

    bar = pd.Series({
        "Open": float(raw["Open"].iloc[0]),
        "High": float(raw["High"].max()),
        "Low": float(raw["Low"].min()),
        "Close": float(raw["Close"].iloc[-1]),
        "Volume": float(raw["Volume"].sum()),
    })

    if override_close is not None:
        oc = float(override_close)
        bar["Close"] = oc
        bar["High"] = max(bar["High"], oc)
        bar["Low"] = min(bar["Low"], oc)

    session_date = pd.Timestamp(raw.index[-1]).tz_localize(None).normalize()
    bar.name = session_date

    # Guard: reject stale bars when the market isn't open today.
    # yfinance period="1d" returns the *last* trading day's bars on weekends,
    # holidays, and pre-market — injecting those would duplicate an existing
    # session.  Only accept bars whose date == today (America/New_York).
    today_ny = datetime.now(_NY_TZ).date()
    if session_date.date() != today_ny:
        return None
    return bar


# --------------------------------------------------------------------------- #
# 2. Helpers — calibration artifact reuse
# --------------------------------------------------------------------------- #
def _find_recent_artifact(ticker: str, repo_root: Path, pattern: str) -> Path | None:
    """Most recent matching artifact across all runs/improves_{TICKER}_* folders."""
    matches = sorted(repo_root.glob(f"runs/improves_{ticker}_*/{pattern}"))
    return matches[-1] if matches else None


def _copy_calibration_artifacts(g: dict, ticker: str) -> dict[str, Any]:
    """Copy wf_dir_panel + entry_rule_80 from a prior run into the live OUTPUT_DIR.

    The calibration panel is built from historical labeled data and does not
    depend on the live bar, so reusing a prior panel avoids a slow expanding-
    window rebuild (minutes).  Returns a dict describing what was reused.
    """
    out_dir: Path = g["OUTPUT_DIR"]
    repo_root: Path = g["REPO_ROOT"]
    info: dict[str, Any] = {"copied_panel": None, "copied_rule": None}

    panel = _find_recent_artifact(ticker, repo_root, f"wf_dir_panel_{ticker}.parquet")
    if panel is not None:
        dest = out_dir / panel.name
        if not dest.exists():
            dest.write_bytes(panel.read_bytes())
            info["copied_panel"] = str(panel)
            print(f"[live] reused panel {panel.parent.name}/{panel.name}")

    rule = _find_recent_artifact(ticker, repo_root, f"entry_rule_80_{ticker}.json")
    if rule is not None:
        dest = out_dir / rule.name
        if not dest.exists():
            dest.write_bytes(rule.read_bytes())
            info["copied_rule"] = str(rule)
            print(f"[live] reused entry_rule {rule.parent.name}/{rule.name}")

    return info


# --------------------------------------------------------------------------- #
# 3. Stage execution (mirrors run_stages / run_ticker — same exec loop)
# --------------------------------------------------------------------------- #
def _exec_stage(stage: str, cfg: RunConfig, g: dict[str, Any], quiet: bool = False) -> None:
    src = _patch_model_n_jobs(_STAGE_SOURCES[stage], cfg.n_jobs_model)
    if stage == "setup":
        src = re.sub(
            r'^TICKER\s*=\s*["\'][^"\']*["\']',
            f'TICKER = {cfg.ticker.upper().strip()!r}',
            src, count=1, flags=re.M,
        )
        exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
        _apply_overrides(cfg, g)
    else:
        exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
    if not quiet:
        print(f"[{g.get('TICKER', cfg.ticker)}] stage ok: {stage}")


def _inject_live_bar(g: dict, ticker: str, live_bar: pd.Series, quiet: bool = False) -> None:
    """Append the intraday bar to bars/ohlcv_hist and recompute move features."""
    today_ts: pd.Timestamp = pd.Timestamp(live_bar.name)

    bars: pd.DataFrame = g["bars"]
    if today_ts in bars.index:
        bars = bars.drop(index=today_ts)
    live_row = pd.DataFrame(
        [{c: live_bar[c] for c in ("Open", "High", "Low", "Close", "Volume")}],
        index=pd.DatetimeIndex([today_ts], name=bars.index.name),
    )
    # Align columns: only OHLCV from live_row, then concat (move cols become NaN, recomputed below)
    common = [c for c in bars.columns if c in live_row.columns]
    missing = [c for c in bars.columns if c not in live_row.columns]
    live_aligned = live_row[common].copy()
    for c in missing:
        live_aligned[c] = np.nan
    live_aligned = live_aligned[bars.columns]
    bars = pd.concat([bars, live_aligned]).sort_index()

    # Recompute prior-close move features on the augmented frame
    bars = g["add_session_move_features"](bars)
    g["bars"] = bars
    g["ohlcv_hist"] = bars.copy()

    if not quiet:
        print(
            f"[{ticker}] injected live bar {today_ts.date()}  "
            f"O={live_bar['Open']:.2f} H={live_bar['High']:.2f} "
            f"L={live_bar['Low']:.2f} C={live_bar['Close']:.2f}  "
            f"bars now {len(bars)} rows → {bars.index.max().date()}"
        )


# --------------------------------------------------------------------------- #
# 4. Result extraction
# --------------------------------------------------------------------------- #
def _extract_summary(g: dict, cfg: RunConfig, summary: dict) -> dict:
    t = g["TICKER"]
    out: Path = g["OUTPUT_DIR"]
    target = g["PREDICT_TARGET_DATE"]

    summary.update({
        "ok": True,
        "ticker": t,
        "predict_target_date": target,
        "output_dir": str(out),
    })

    pred_path = out / f"pred_{target}_{t}.json"
    buy_path = out / f"buy_decision_{target}_{t}.json"
    rule_path = out / f"entry_rule_80_{t}.json"
    panel03 = None
    for p in out.glob(f"03_dir_hits_last*_{t}.png"):
        panel03 = str(p)
        break

    summary["pred_path"] = str(pred_path) if pred_path.exists() else None
    summary["buy_path"] = str(buy_path) if buy_path.exists() else None
    summary["rule_path"] = str(rule_path) if rule_path.exists() else None
    summary["chart_03"] = panel03

    if pred_path.exists():
        summary["pred"] = json.loads(pred_path.read_text())
    if buy_path.exists():
        bd = json.loads(buy_path.read_text())
        summary["buy_decision"] = bd
        summary["decision"] = bd.get("decision")
    if rule_path.exists():
        rule = json.loads(rule_path.read_text())
        summary["is_discriminating"] = rule.get("is_discriminating")
        summary["auto_invert"] = rule.get("auto_invert")
        summary["operating_entry_score_min"] = rule.get("operating_entry_score_min")
        summary["entry_score_definition"] = rule.get("entry_score_definition")
        summary["oriented_auc"] = rule.get("oriented_auc")
        summary["raw_xgb_auc"] = rule.get("raw_xgb_auc")
        pend = rule.get("pending") or {}
        summary["entry_score"] = pend.get("entry_score")
        summary["enter_under_best_rule"] = pend.get("enter_under_best_rule")

    return summary


# --------------------------------------------------------------------------- #
# 5. Public: live_predict
# --------------------------------------------------------------------------- #
def live_predict(
    ticker: str,
    *,
    target_date: str | None = None,
    override_close: float | None = None,
    repo_root: Path | str | None = None,
    skip_feature_health: bool = True,
    skip_plot_01: bool = True,
    quiet: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run inference using today's live (intraday) bar as the feature-asof session.

    Pulls today's session-so-far OHLCV from yfinance (or uses ``override_close``
    for a hypothetical), appends to cached history, rebuilds features, trains on
    all labeled data through yesterday, and predicts the next session.

    Reuses the exact notebook-extracted stages (``_STAGE_SOURCES``) — no second
    feature stack.  Calibration panel is reused from prior runs when available
    so the slow expanding-window rebuild is avoided.
    """
    import matplotlib.pyplot as plt
    from market_ml.pipeline import display as _display

    ticker = ticker.upper().strip()

    # Target = next business day after today (live mode always predicts tomorrow)
    if target_date is None:
        today = datetime.now(_TZ).date()
        target_date = (pd.Timestamp(today) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    # Fail fast, BEFORE creating the run folder — a live predict with no live
    # bar (pre-market/holiday/weekend) used to mkdir runs/improves_{T}_YYYY-MM-DD
    # and then error out, littering empty future-dated folders.
    if override_close is None:
        live_bar_probe = fetch_live_bar(ticker)
        if live_bar_probe is None:
            today_ny = datetime.now(_NY_TZ).date()
            if today_ny.weekday() >= 5:
                raise RuntimeError(
                    "Markets are closed (weekend) — live predict requires "
                    "an open or recently-closed session."
                )
            raise RuntimeError(
                f"No intraday data for {ticker} with today's date ({today_ny}). "
                f"Market may be closed (holiday/pre-market) or ticker is invalid."
            )

    cfg = RunConfig(
        ticker=ticker,
        predict_target_date=target_date,
        repo_root=repo_root,
        skip_feature_health=skip_feature_health,
        skip_plot_01=skip_plot_01,
        quiet=quiet,
        **kwargs,
    )

    # Exec globals — same seed as run_stages / run_ticker
    g: dict[str, Any] = {"__builtins__": __builtins__}
    _seed_globals(g)

    summary: dict[str, Any] = {
        "ok": False,
        "error": None,
        "stages_done": [],
        "ticker": ticker,
        "live": True,
        "target_date": target_date,
        "override_close": override_close,
    }

    try:
        # Determine stage list
        stages = list(_STAGE_ORDER)
        if skip_feature_health:
            stages = [s for s in stages if s != "feature_health"]
        if skip_plot_01:
            stages = [s for s in stages if s != "plot_01"]

        load_idx = stages.index("load_ohlcv")
        pre = stages[: load_idx + 1]       # setup, load_ohlcv
        post = stages[load_idx + 1:]        # feature_registry → buy_rule

        # Run pre-stages (setup + load_ohlcv — drops today's incomplete bar)
        for stage in pre:
            _exec_stage(stage, cfg, g, quiet=quiet)
            summary["stages_done"].append(stage)

        # Inject live bar
        live_bar = fetch_live_bar(ticker, override_close=override_close)
        if live_bar is None:
            today_ny = datetime.now(_NY_TZ).date()
            is_weekend = today_ny.weekday() >= 5
            if is_weekend:
                raise RuntimeError(
                    f"Markets are closed (weekend) — live predict requires "
                    f"an open or recently-closed session."
                )
            raise RuntimeError(
                f"No intraday data for {ticker} with today's date ({today_ny}). "
                f"Market may be closed (holiday/pre-market) or ticker is invalid."
            )
        _inject_live_bar(g, ticker, live_bar, quiet=quiet)
        summary["live_bar"] = {
            "date": str(pd.Timestamp(live_bar.name).date()),
            "open": float(live_bar["Open"]),
            "high": float(live_bar["High"]),
            "low": float(live_bar["Low"]),
            "close": float(live_bar["Close"]),
            "volume": float(live_bar["Volume"]),
            "override_close": override_close is not None,
        }

        # Reuse calibration panel + entry rule from prior runs
        reuse_info = _copy_calibration_artifacts(g, ticker)
        summary["reused_artifacts"] = reuse_info

        # Run post-stages (features → assemble → walkforward_pred → calibrate → buy_rule)
        for stage in post:
            _exec_stage(stage, cfg, g, quiet=quiet)
            summary["stages_done"].append(stage)

        summary = _extract_summary(g, cfg, summary)
        plt.close("all")
    except Exception as e:
        summary["ok"] = False
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()
        plt.close("all")
        if not quiet:
            print(f"[{ticker}] FAILED: {summary['error']}")
            print(summary["traceback"])
    return summary
