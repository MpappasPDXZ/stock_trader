"""Alpha Vantage historical put-call ratio backfiller — free-tier aware.

Fetches daily per-symbol historical option put-call ratios (full chain +
per-expiration term structure) from the FREE-tier ALPHA Vantage endpoints and
appends them into long-format parquet caches per ticker:

    data/cache_pcr_<TICKER>.parquet   (columns: date, expiry, pcr)

with ``expiry == "full"`` marking the full-chain aggregate. Complete history
from 2008-01-01 forward.

Free-tier realities (as of 2026-10-12):
  * HISTORICAL_PUT_CALL_RATIO: free, 25 requests/day per key, 1 req/sec
  * Pace: with --budget 22, one ticker advances ~22 sessions/run → ~6
    trading-months of backlog cleared over 30 runs (~40 calendar days).
  * HISTORICAL_VOLUME_OPEN_INTEREST_RATIO: believed free (rate-limit message,
    not a premium message) but unverified — enable via --endpoint volume_oi
    once verified.

State is tracked in ``data/av_backfill_state.json`` so consecutive runs
resume exactly where the last one left off and never exceed the daily
budget. API key is read from ``$AV_API_KEY`` or ``data/av_api_key.txt``
(both gitignored — never put the key in source).

Usage (from ml_analysis root):
    .venv/bin/python -m scripts.av_backfill_pcr                 # CAT, default budget
    .venv/bin/python -m scripts.av_backfill_pcr --budget 20     # explicit budget
    .venv/bin/python -m scripts.av_backfill_pcr CAT TSLA        # several tickers (budget is shared!)
    .venv/bin/python -m scripts.av_backfill_pcr --stats         # just show progress
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
STATE_PATH = DATA_DIR / "av_backfill_state.json"
KEY_PATH = DATA_DIR / "av_api_key.txt"
BASE_URL = "https://www.alphavantage.co/query"

ENDPOINTS = {
    "put_call": "HISTORICAL_PUT_CALL_RATIO",
    "volume_oi": "HISTORICAL_VOLUME_OPEN_INTEREST_RATIO",  # unverified free-tier
}
DEFAULT_START = date(2022, 1, 3)
FULL_CHAIN = "full"


def load_key() -> str:
    key = (KEY_PATH.read_text().strip() if KEY_PATH.exists() else "")
    if not key:
        raise SystemExit("No API key: set $AV_API_KEY or create data/av_api_key.txt")
    return key


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def trading_days(start: date, end: date) -> list[date]:
    """ISO weekdays in range (holidays are skipped at fetch time via nulls)."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def cache_path(ticker: str) -> Path:
    return DATA_DIR / f"cache_pcr_{ticker}.parquet"


def backfill_progress(ticker: str, start: date) -> dict:
    cp = cache_path(ticker)
    if cp.exists():
        have = pd.read_parquet(cp)
        have_days = pd.to_datetime(have["date"]).dt.date
        got = set(have_days[have_days >= start])
    else:
        got = set()
    want = set(trading_days(start, date.today()))
    missing = sorted(want - got)
    return {
        "have": len(got),
        "want": len(want),
        "missing": len(missing),
        "next_missing": max(missing).isoformat() if missing else None,
        "missing_list": list(reversed(missing)),  # newest → oldest
    }


def fetch_one(ticker: str, day: date, endpoint: str, key: str) -> pd.DataFrame:
    """One API call → rows (date, expiry, pcr). Raises RuntimeError on gates."""
    resp = requests.get(
        BASE_URL,
        params={
            "function": ENDPOINTS[endpoint],
            "symbol": ticker,
            "date": day.isoformat(),
            "apikey": key,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    if "Information" in data:
        info = data["Information"]
        if "premium" in info.lower():
            raise RuntimeError(f"premium gate: {info[:160]}")
        raise RuntimeError(f"rate limit/note: {info[:160]}")

    if endpoint == "put_call":
        full = data.get("put_call_ratio_full_chain")
        exps = data.get("put_call_ratio_by_expiration") or []
        rows = [{"date": day.isoformat(), "expiry": FULL_CHAIN, "pcr": full}]
        rows += [
            {"date": day.isoformat(), "expiry": e.get("date"), "pcr": e.get("value")}
            for e in exps
        ]
    else:
        raise RuntimeError(f"parsing for endpoint {endpoint!r} not wired yet")

    # Holiday sessions return a null/full-chain-empty payload — keep the row
    # with an empty pcr so the cache stays dense over weekdays.
    out = pd.DataFrame(rows)
    out["pcr"] = pd.to_numeric(out["pcr"], errors="coerce")
    return out


def backfill(tickers: list[str], budget: int, endpoint: str, start: date,
             dry_run: bool = False) -> int:
    key = load_key()
    state = load_state()

    today = date.today().isoformat()
    used_today = int(state.get("budget", {}).get(today, 0))
    refreshed = {k: v for k, v in state.get("budget", {}).items() if k == today}
    state["budget"] = refreshed

    if endpoint not in ENDPOINTS:
        raise SystemExit(f"unknown endpoint {endpoint!r}")

    stats_lines = []
    for ticker in tickers:
        prog = backfill_progress(ticker, start)
        stats_lines.append(f"  {ticker}: have={prog['have']} missing={prog['missing']}"
                           f" next_missing={prog['next_missing']}")
    print("Progress (newest→oldest):\n" + "\n".join(stats_lines), flush=True)
    print(f"Budget today: {today} used={used_today} limit={budget}", flush=True)
    total_used = used_today
    for ticker in tickers:
        prog = backfill_progress(ticker, start)
        # Walk newest → oldest: the model's analysis window is the trailing
        # year, so filling the recent end first makes candidates testable ASAP.
        if not prog["next_missing"]:
            print(f"  {ticker}: nothing missing — cache current.", flush=True)
            continue
        # full deficit, newest → oldest — previously this only spanned
        # (next_missing, today], so after the daily-led row landed the run
        # never walked further back and the 291-day deficit never shrank
        missing = prog["missing_list"]

        new_rows: list[pd.DataFrame] = []
        any_gates = 0  # consecutive API notes for THIS ticker only — previously
        # a global counter, so the first ticker's 3 notes foreclosed every
        # later ticker's depth and each run only ever landed 1 session
        for day in missing:
            if total_used >= used_today + budget and budget > 0:
                print(f"  {ticker}: daily budget exhausted (used {total_used} today)", flush=True)
                break
            if dry_run:
                total_used += 1
                continue
            time.sleep(1.1)  # free tier: 1 request/sec
            try:
                df = fetch_one(ticker, day, endpoint, key)
            except RuntimeError as e:
                print(f"  {ticker} {day}: {e}", flush=True)
                if "premium gate" in str(e):
                    print("Stopping — endpoint is premium-locked.", flush=True)
                    save_state(state)
                    return 2
                any_gates += 1
                if any_gates >= 3:
                    print("Stopping after 3 consecutive API notes (likely daily cap hit).", flush=True)
                    break
                continue
            total_used += 1
            # recent-session nulls = AV publish lag (premium feed trails by a
            # few days), NOT holiday emptiness — drop them so the deficit
            # stays and a later run retries the session
            fc = df[df["expiry"] == FULL_CHAIN]
            if fc["pcr"].isna().all() and (date.today() - day).days < 7:
                print(f"  {ticker} {day}: null pcr on a recent session "
                      f"(AV publish lag) — dropped, will retry", flush=True)
                continue
            df["ticker"] = ticker
            new_rows.append(df)

        if new_rows and not dry_run:
            cp = cache_path(ticker)
            fresh = pd.concat(new_rows, ignore_index=True)
            if cp.exists():
                old = pd.read_parquet(cp)
                fresh = pd.concat([old, fresh], ignore_index=True).drop_duplicates(
                    subset=["date", "expiry"], keep="last")
            fresh.to_parquet(cp, index=False)
            state[f"last_fetch_{ticker}"] = date.today().isoformat()
            print(f"  {ticker}: +{len(new_rows)} day(s) → {cp.name} "
                  f"(total rows {len(fresh)})", flush=True)

    state["budget"][today] = total_used
    save_state(state)
    print(f"API calls today: {total_used} (budget {budget}/day)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpha Vantage PCR backfiller (free-tier aware)")
    parser.add_argument("tickers", nargs="*", default=["CAT"])
    parser.add_argument("--budget", type=int, default=22,
                        help="Max NEW API calls per calendar day (default 22; free tier allows 25)")
    parser.add_argument("--endpoint", choices=list(ENDPOINTS), default="put_call")
    parser.add_argument("--start", type=str, default=DEFAULT_START.isoformat())
    parser.add_argument("--stats-only", action="store_true", help="Print progress, fetch nothing")
    args = parser.parse_args()

    tickers = [t.upper() for t in (args.tickers or ["CAT"])]
    start = date.fromisoformat(args.start)

    if args.stats_only:
        for t in tickers:
            prog = backfill_progress(t, start)
            print(f"{t}: have={prog['have']} want={prog['want']} missing={prog['missing']}"
                  f" next_missing={prog['next_missing']} (newest→oldest)")
        return 0

    return backfill(tickers, args.budget, args.endpoint, start)


if __name__ == "__main__":
    sys.exit(main())
