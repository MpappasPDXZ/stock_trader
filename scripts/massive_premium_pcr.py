"""Build historical daily put/call volume ratios (PCR) from Massive (Polygon)
free-tier per-contract daily aggregate bars.

Free (Options Basic) entitlements used:
  * /v3/reference/options/contracts — strike/expiry universe incl. expired, paginated
  * /v2/aggs/ticker/{contract}/range/1/day/{from}/{to} — one call returns the
    contract's whole daily-bar history within the free ~2-year window

Per-contract daily volumes aggregate to chain-wide and per-expiry volume PCR
into ``data/cache_pcr_massive_<TICKER>.parquet``
(date, expiry, pcr, put_vol, call_vol, n_contracts, source, ticker).
The candidate builders merge this with the Alpha Vantage cache.

Pacing: 13s between calls (4.6 req/min vs the free 5/min cap). Resume state in
``data/massive_pcr_state_<TICKER>.json``; --stats-only prints progress.

Usage:
    .venv/bin/python -m scripts.massive_premium_pcr CAT
    .venv/bin/python -m scripts.massive_premium_pcr CAT --stats-only
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BASE = "https://api.massive.com"
PACING_SECONDS = 13
KEY_PATH = DATA_DIR / "massive_api_key.txt"
HISTORY_FROM = "2023-10-01"  # free window ~2 years; older per-contract from-dates are rejected

CONTRACT_RE = re.compile(r"O:([A-Z0-9\.]+?)(\d{6})([CP])(\d{8})$")


def load_key() -> str:
    import os
    key = os.environ.get("MASSIVE_API_KEY") or (KEY_PATH.read_text().strip() if KEY_PATH.exists() else "")
    if not key:
        raise SystemExit("No key: set $MASSIVE_API_KEY or data/massive_api_key.txt")
    return key


def state_path(ticker: str) -> Path:
    return DATA_DIR / f"massive_pcr_state_{ticker}.json"


def cache_path(ticker: str) -> Path:
    return DATA_DIR / f"cache_pcr_massive_{ticker}.parquet"


def raw_cache_path(ticker: str) -> Path:
    return DATA_DIR / f"cache_pcrmassive_raw_{ticker}.parquet"


def fetch_json(url: str, key: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code == 429 and attempt < retries - 1:
                wait = 65 * (attempt + 1)
                print(f"    429 rate limit; sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            if e.code == 404:
                return {}
            raise RuntimeError(f"HTTP {e.code}: {body}") from e
    raise RuntimeError("retries exhausted")


def list_contracts(ticker: str, strike_min: float, strike_max: float,
                   expiry_min: str, expiry_max: str, key: str,
                   expired: str | None = None) -> set[str]:
    """Active AND expired contracts (the pull-date-active list alone misses
    every front-weekly that has already expired — the bulk of a 2-year PCR
    history). Two listing passes, union. Pass ``expired`` ("true"/"false")
    to restrict to one pass."""
    tickers: set[str] = set()
    passes = (expired,) if expired else ("false", "true")
    for expired_flag in passes:
        band = (f"&expiration_date.gte={expiry_min}" if expiry_min else "")
        url = (f"{BASE}/v3/reference/options/contracts?underlying_ticker={ticker}"
               f"&strike_price.gte={strike_min}&strike_price.lte={strike_max}"
               f"{band}&expiration_date.lt={expiry_max}&limit=1000"
               f"&expired={expired_flag}")
        while url:
            d = fetch_json(url, key)
            for row in d.get("results") or []:
                tickers.add(row["ticker"])
            url = d.get("next_url")
            time.sleep(PACING_SECONDS)
    return tickers


def load_state(ticker: str) -> dict:
    p = state_path(ticker)
    return json.loads(p.read_text()) if p.exists() else {"done_contracts": []}


def save_state(ticker: str, st: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    state_path(ticker).write_text(json.dumps(st, indent=1))


def pull_contract(contract: str, key: str, ticker: str) -> int:
    """Pull one contract's full daily-bar history; append to the raw cache.
    Returns bar count."""
    m = CONTRACT_RE.match(contract)
    if not m:
        print(f"    skip unrecognized {contract}", flush=True)
        return 0
    underlying, _rest, side, strike_digits = m.group(1), m.group(1), m.group(3), m.group(4)
    expiry_digits = CONTRACT_RE.match(contract).group(2)
    expiry = f"20{expiry_digits[:2]}-{expiry_digits[2:4]}-{expiry_digits[4:]}"
    strike = int(strike_digits) / 1000.0

    url = f"{BASE}/v2/aggs/ticker/{contract}/range/1/day/{HISTORY_FROM}/{date.today().isoformat()}?limit=50000"
    try:
        d = fetch_json(url, key)
    except RuntimeError as e:
        if "403" in str(e):
            print(f"    {contract}: not authorized (out of window) — skipping", flush=True)
            return 0
        raise
    bars = []
    for bar in d.get("results") or []:
        ts = (pd.Timestamp(bar["t"], unit="ms", tz="UTC")
              .tz_convert("America/Chicago").normalize().tz_localize(None))
        bars.append({"date": ts.date().isoformat(), "close": bar["c"],
                     "volume": bar["v"], "transactions": bar.get("n", 0)})

    if bars:
        raw_path = raw_cache_path(ticker)
        old = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame(
            columns=["date", "expiry", "side", "strike", "close", "volume", "transactions"])
        df = pd.DataFrame(bars).assign(expiry=expiry, side=side, strike=strike)
        # dedupe (refresh pulls re-send the contract's full history; fresh wins)
        combined = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["date", "expiry", "side", "strike"], keep="last")
        combined.to_parquet(raw_path, index=False)
    return len(bars)


def rebuild_daily_cache(ticker: str) -> pd.DataFrame:
    raw_path = raw_cache_path(ticker)
    if not raw_path.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(raw_path)
    pivot = raw.groupby(["date", "expiry", "side"], as_index=False)["volume"].sum()
    puts = (pivot[pivot["side"] == "P"][["date", "expiry", "volume"]]
            .rename(columns={"volume": "put_vol"}))
    calls = (pivot[pivot["side"] == "C"][["date", "expiry", "volume"]]
             .rename(columns={"volume": "call_vol"}))
    m = puts.merge(calls, on=["date", "expiry"], how="outer").fillna(0)
    m["pcr"] = m["put_vol"] / m["call_vol"].replace(0, pd.NaT)
    counts = raw.groupby(["date", "expiry"]).size().reset_index(name="n_contracts")
    m = m.merge(counts, on=["date", "expiry"], how="left")

    daily = (m.groupby("date", as_index=False)[["put_vol", "call_vol", "n_contracts"]].sum())
    daily["pcr"] = daily["put_vol"] / daily["call_vol"].replace(0, pd.NaT)
    daily["expiry"] = "full"
    out = pd.concat([m, daily], ignore_index=True)
    out["source"] = "massive_strikes"
    out["ticker"] = ticker
    return out[["date", "expiry", "pcr", "put_vol", "call_vol", "n_contracts", "source", "ticker"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--strike-min", type=float, default=500.0)
    parser.add_argument("--strike-max", type=float, default=950.0)
    parser.add_argument("--expiry-min", type=str, default="")
    parser.add_argument("--expiry-max", type=str, default="2027-01-01")
    parser.add_argument("--pace", type=float, default=13.0)
    parser.add_argument("--refresh-active", action="store_true", default=True,
                        help="re-pull ACTIVE contracts every night so their "
                             "daily tail grows (raw cache dedupes; fresh wins)")
    parser.add_argument("--no-refresh-active", dest="refresh_active",
                        action="store_false")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    key = load_key()
    st = load_state(ticker)
    done = set(st["done_contracts"])

    if args.stats_only:
        print(f"{ticker}: contracts done={len(done)}")
        return 0

    cp = cache_path(ticker)
    print(f"[{ticker}] listing contracts {args.strike_min}-{args.strike_max} "
          f"exp {args.expiry_min or '*'}..{args.expiry_max} ...", flush=True)
    contracts = sorted(list_contracts(ticker, args.strike_min, args.strike_max,
                                      args.expiry_min or "2000-01-01",
                                      args.expiry_max, key))
    todo = [c for c in contracts if c not in done]
    if args.refresh_active:
        # active contracts re-pull nightly — the pull contract's daily-bar
        # history now runs to today, so this is what publishes the modulus
        # most recent sessions the AV feed lags on
        actives = sorted(list_contracts(ticker, args.strike_min, args.strike_max,
                                        args.expiry_min or "2000-01-01",
                                        args.expiry_max, key, expired="false"))
        refresh = [c for c in actives if c in done]
        todo = sorted(set(todo + refresh))
    print(f"[{ticker}] {len(contracts)} contracts, {len(todo)} to pull "
          f"(~{len(todo) * (args.pace + 2) / 60:.0f} min)", flush=True)

    import time as _time
    for i, contract in enumerate(todo):
        pull_contract(contract, key, ticker)
        st["done_contracts"].append(contract)
        _time.sleep(args.pace)
        if (i + 1) % 25 == 0:
            save_state(ticker, st)
            print(f"[{ticker}] {i + 1}/{len(todo)} pulled", flush=True)
    save_state(ticker, st)

    df = rebuild_daily_cache(ticker)
    if len(df):
        df.to_parquet(cp, index=False)
        print(f"[{ticker}] daily cache → {cp.name}: {df['date'].nunique()} days, {len(df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
