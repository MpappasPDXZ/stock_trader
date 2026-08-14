"""Curated candidate feature library for market-ML.

Each candidate is a pre-built, look-ahead-free feature builder that can be injected
into the canonical pipeline (``run_stages``) and re-evaluated via ΔAUC.  No free-form
code execution — only builders registered here are testable.

A candidate is a dict with:
  - ``name``           — column name (must match ``bars["name"]`` == ``FEATURE_COLS`` entry)
  - ``title``          — human-readable title for charts
  - ``sessions_before_start`` — trading sessions needed at as-of day t
  - ``builder``        — pure function ``(bars: pd.DataFrame) -> None`` that adds the column
  - ``category``       — "momentum" | "volatility" | "volume" | "trend" | "structure"
  - ``description``    — one-liner

Builders receive the OHLCV ``bars`` DataFrame (which already has Open/High/Low/Close/Volume
and any existing feature columns).  They must add exactly one column named ``name``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# Candidate builders — each adds a single column to bars
# --------------------------------------------------------------------------- #
def _roc_10(bars: pd.DataFrame) -> None:
    bars["roc_10"] = bars["Close"].pct_change(10)


def _willr_14(bars: pd.DataFrame) -> None:
    hh = bars["High"].rolling(14).max()
    ll = bars["Low"].rolling(14).min()
    bars["willr_14"] = -100 * (hh - bars["Close"]) / (hh - ll).replace(0, np.nan)


def _atr_pct_14(bars: pd.DataFrame) -> None:
    bars["atr_pct_14"] = _wilder_atr(bars, 14) / bars["Close"]


def _cci_20(bars: pd.DataFrame) -> None:
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    bars["cci_20"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _mfi_14(bars: pd.DataFrame) -> None:
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    mf = tp * bars["Volume"]
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(14).sum()
    neg_mf = mf.where(delta < 0, 0).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    bars["mfi_14"] = (100 - 100 / (1 + mfr)).fillna(50)


def _ema_cross_12_26(bars: pd.DataFrame) -> None:
    ema12 = bars["Close"].ewm(span=12, adjust=False).mean()
    ema26 = bars["Close"].ewm(span=26, adjust=False).mean()
    bars["ema_cross_12_26"] = (ema12 - ema26) / bars["Close"]


def _stoch_14_3(bars: pd.DataFrame) -> None:
    ll = bars["Low"].rolling(14).min()
    hh = bars["High"].rolling(14).max()
    pct_k = 100 * (bars["Close"] - ll) / (hh - ll).replace(0, np.nan)
    bars["stoch_14_3"] = pct_k.rolling(3).mean()


def _adx_14(bars: pd.DataFrame) -> None:
    period = 14
    up = bars["High"].diff()
    down = -bars["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    atr = _wilder_atr(bars, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    bars["adx_14"] = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _close_z_20(bars: pd.DataFrame) -> None:
    bars["close_z_20"] = (bars["Close"] - bars["Close"].rolling(20).mean()) / bars["Close"].rolling(20).std().replace(0, np.nan)


def _gap_1(bars: pd.DataFrame) -> None:
    bars["gap_1"] = (bars["Open"] / bars["Close"].shift(1) - 1)


def _obv_slope_10(bars: pd.DataFrame) -> None:
    obv = (np.sign(bars["Close"].diff()) * bars["Volume"]).cumsum()
    bars["obv_slope_10"] = obv.diff(10) / (bars["Volume"].rolling(10).sum().replace(0, np.nan))


def _macd_hist_12_26_9(bars: pd.DataFrame) -> None:
    ema12 = bars["Close"].ewm(span=12, adjust=False).mean()
    ema26 = bars["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    bars["macd_hist_12_26_9"] = (macd - signal) / bars["Close"]


def _return_skew_20(bars: pd.DataFrame) -> None:
    rets = bars["Close"].pct_change()
    bars["return_skew_20"] = rets.rolling(20).skew()


def _dollar_vol_z_20(bars: pd.DataFrame) -> None:
    dvol = bars["Close"] * bars["Volume"]
    bars["dollar_vol_z_20"] = (dvol - dvol.rolling(20).mean()) / dvol.rolling(20).std().replace(0, np.nan)


def _near_low_50(bars: pd.DataFrame) -> None:
    ll = bars["Low"].rolling(50).min()
    bars["near_low_50"] = (bars["Close"] - ll) / (bars["High"].rolling(50).max() - ll).replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
CANDIDATES: list[dict] = [
    {
        "name": "roc_10",
        "title": "10-session Rate of Change",
        "sessions_before_start": 10,
        "builder": _roc_10,
        "category": "momentum",
        "description": "10-session price rate of change (pct return over 10 sessions)",
    },
    {
        "name": "willr_14",
        "title": "Williams %R (14)",
        "sessions_before_start": 14,
        "builder": _willr_14,
        "category": "momentum",
        "description": "Williams %R — momentum oscillator measuring close relative to 14-session high-low range",
    },
    {
        "name": "atr_pct_14",
        "title": "ATR % (14)",
        "sessions_before_start": 14,
        "builder": _atr_pct_14,
        "category": "volatility",
        "description": "Wilder ATR(14) normalized by close price",
    },
    {
        "name": "cci_20",
        "title": "Commodity Channel Index (20)",
        "sessions_before_start": 20,
        "builder": _cci_20,
        "category": "momentum",
        "description": "CCI(20) — deviation of typical price from its moving average",
    },
    {
        "name": "mfi_14",
        "title": "Money Flow Index (14)",
        "sessions_before_start": 14,
        "builder": _mfi_14,
        "category": "volume",
        "description": "MFI(14) — volume-weighted RSI analog",
    },
    {
        "name": "ema_cross_12_26",
        "title": "EMA 12/26 Cross",
        "sessions_before_start": 26,
        "builder": _ema_cross_12_26,
        "category": "trend",
        "description": "Normalized EMA(12) − EMA(26) spread",
    },
    {
        "name": "stoch_14_3",
        "title": "Stochastic %K (14,3)",
        "sessions_before_start": 16,
        "builder": _stoch_14_3,
        "category": "momentum",
        "description": "Smoothed stochastic oscillator (14,3)",
    },
    {
        "name": "adx_14",
        "title": "ADX (14)",
        "sessions_before_start": 28,
        "builder": _adx_14,
        "category": "trend",
        "description": "Wilder ADX(14) — trend strength regardless of direction",
    },
    {
        "name": "close_z_20",
        "title": "Close Z-Score (20)",
        "sessions_before_start": 20,
        "builder": _close_z_20,
        "category": "structure",
        "description": "20-session z-score of close price",
    },
    {
        "name": "gap_1",
        "title": "Overnight Gap",
        "sessions_before_start": 1,
        "builder": _gap_1,
        "category": "structure",
        "description": "Open vs prior close gap",
    },
    {
        "name": "obv_slope_10",
        "title": "OBV Slope (10)",
        "sessions_before_start": 10,
        "builder": _obv_slope_10,
        "category": "volume",
        "description": "10-session OBV change normalized by total volume",
    },
    {
        "name": "macd_hist_12_26_9",
        "title": "MACD Histogram (12,26,9)",
        "sessions_before_start": 35,
        "builder": _macd_hist_12_26_9,
        "category": "momentum",
        "description": "Normalized MACD histogram (EMA12−EMA26 − signal)",
    },
    {
        "name": "return_skew_20",
        "title": "Return Skewness (20)",
        "sessions_before_start": 20,
        "builder": _return_skew_20,
        "category": "structure",
        "description": "20-session rolling skewness of daily returns",
    },
    {
        "name": "dollar_vol_z_20",
        "title": "Dollar Volume Z (20)",
        "sessions_before_start": 20,
        "builder": _dollar_vol_z_20,
        "category": "volume",
        "description": "20-session z-score of dollar volume (Close × Volume)",
    },
    {
        "name": "near_low_50",
        "title": "Near Low (50)",
        "sessions_before_start": 50,
        "builder": _near_low_50,
        "category": "structure",
        "description": "Position within 50-session high-low range (0 = at low, 1 = at high)",
    },
]

CANDIDATES_BY_NAME = {c["name"]: c for c in CANDIDATES}


def list_candidates() -> list[dict]:
    return [
        {
            "name": c["name"],
            "title": c["title"],
            "sessions_before_start": c["sessions_before_start"],
            "category": c["category"],
            "description": c["description"],
        }
        for c in CANDIDATES
    ]


# --------------------------------------------------------------------------- #
# Candidate test — inject feature, re-run assemble→calibrate, report ΔAUC
# --------------------------------------------------------------------------- #
def test_candidate(
    ticker: str,
    candidate_name: str,
    *,
    predict_target_date: str | None = None,
) -> dict:
    """Test a candidate feature vs baseline, returning ΔAUC and discrimination.

    Runs the canonical pipeline (same stages as notebook) for baseline, then
    injects the candidate feature and re-runs assemble→calibrate. No second
    feature stack — candidate builder operates on the same ``bars`` DataFrame.
    """
    from market_ml.pipeline import RunConfig, _STAGE_SOURCES, _STAGE_ORDER
    from market_ml.jobs import run_stages

    if candidate_name not in CANDIDATES_BY_NAME:
        raise ValueError(
            f"unknown candidate {candidate_name!r}; have {list(CANDIDATES_BY_NAME)}"
        )
    cand = CANDIDATES_BY_NAME[candidate_name]

    cfg = RunConfig(
        ticker=ticker,
        predict_target_date=predict_target_date,
        n_jobs_model=2,
    )

    # 1) Baseline run through calibrate
    g_base, s_base = run_stages(cfg, up_to="calibrate", quiet=True)
    if not s_base["ok"]:
        raise RuntimeError(f"baseline failed: {s_base['error']}")
    base_auc = float(g_base.get("auc_oriented", float("nan")))
    base_raw_auc = float(g_base.get("auc_raw", float("nan")))
    base_disc = bool(g_base.get("XGB_IS_DISCRIMINATING", False))

    # 2) Candidate run: stages up to feat_bb, inject candidate, then assemble→calibrate
    g_cand, s_cand = run_stages(cfg, up_to="feat_bb", quiet=True)
    if not s_cand["ok"]:
        raise RuntimeError(f"candidate feat stage failed: {s_cand['error']}")

    bars: pd.DataFrame = g_cand["bars"]
    cand["builder"](bars)
    feat_name = cand["name"]
    if feat_name not in bars.columns:
        raise RuntimeError(f"builder did not add column {feat_name!r}")

    # Register candidate in FEATURE_SPECS + FEATURE_COLS
    g_cand["FEATURE_SPECS"] = list(g_cand["FEATURE_SPECS"]) + [{
        "name": feat_name,
        "sessions_before_start": cand["sessions_before_start"],
        "title": cand["title"],
    }]
    g_cand["FEATURE_COLS"] = list(g_cand["FEATURE_COLS"]) + [feat_name]

    # Force-rebuild the WF panel (remove cached parquet)
    import os
    panel_path = g_cand["OUTPUT_DIR"] / f"wf_dir_panel_{g_cand['TICKER']}.parquet"
    if panel_path.exists():
        panel_path.unlink()
    # Also clear any stale panel_80 in globals
    g_cand.pop("panel_80", None)

    # Exec remaining stages in the modified globals
    import traceback
    remaining = [s for s in _STAGE_ORDER if s not in s_cand["stages_done"]]
    # We need assemble through calibrate (skip plot_01 which needs feature_health)
    remaining = [s for s in remaining if s in (
        "assemble", "feature_health", "plot_01", "walkforward_pred", "calibrate",
    )]
    try:
        for stage in remaining:
            src = _STAGE_SOURCES[stage]
            exec(compile(src, f"<candidate:{stage}>", "exec"), g_cand, g_cand)
    except Exception as e:
        raise RuntimeError(
            f"candidate stage failed at {remaining}: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )

    cand_auc = float(g_cand.get("auc_oriented", float("nan")))
    cand_raw_auc = float(g_cand.get("auc_raw", float("nan")))
    cand_disc = bool(g_cand.get("XGB_IS_DISCRIMINATING", False))

    delta_auc = cand_auc - base_auc if np.isfinite(cand_auc) and np.isfinite(base_auc) else None

    return {
        "ticker": ticker.upper(),
        "candidate": candidate_name,
        "candidate_title": cand["title"],
        "category": cand["category"],
        "baseline_oriented_auc": base_auc,
        "baseline_raw_auc": base_raw_auc,
        "baseline_is_discriminating": base_disc,
        "candidate_oriented_auc": cand_auc,
        "candidate_raw_auc": cand_raw_auc,
        "candidate_is_discriminating": cand_disc,
        "delta_auc": delta_auc,
        "delta_auc_pct": (delta_auc / base_auc) if (delta_auc is not None and base_auc != 0) else None,
        "passes_1pct_threshold": (
            delta_auc is not None and delta_auc >= 0.01
        ),
        "feature_cols_baseline": list(g_base.get("FEATURE_COLS", [])),
        "feature_cols_candidate": list(g_cand.get("FEATURE_COLS", [])),
    }
