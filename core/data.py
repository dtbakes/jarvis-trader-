"""
Data.py — historical candle fetcher + local cache for JARVIS backtesting.

OANDA caps candle requests at 5000 bars, but STRATEGY.md §10.1 requires 4+
years of H1 (~25,000 bars/pair). This module walks forward through time in
5000-bar pages, stitches the result into one DataFrame, and caches it to disk
so a 10-pair backtest hits the API once, not on every run.

Output contract (matches regime._validate):
    * columns: open, high, low, close, volume  (lowercase)
    * index:   tz-aware UTC DatetimeIndex of bar OPEN times, sorted ascending
    * only COMPLETE candles are kept

Usage:
    from data import get_history
    h1 = get_history("EUR_USD", "H1", years=4)
    h4 = get_history("EUR_USD", "H4", years=4)

    # force a re-download (e.g. to extend the cache to today):
    h1 = get_history("EUR_USD", "H1", years=4, refresh=True)

Cache lives in data/candles/{instrument}_{granularity}_{years}y.pkl
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "candles"
)

_GRANULARITY_SECONDS = {
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 3600,
    "H4": 4 * 3600,
    "D": 24 * 3600,
}

_MAX_COUNT = 5000          # OANDA per-request cap
_THROTTLE_SECONDS = 0.25   # be polite to the practice API


def _client() -> oandapyV20.API:
    return oandapyV20.API(
        access_token=config.OANDA_API_KEY,
        environment=config.OANDA_ENVIRONMENT,
    )


def _cache_path(instrument: str, granularity: str, years: float) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{instrument}_{granularity}_{years}y.pkl")


def _fetch_page(
    api: oandapyV20.API,
    instrument: str,
    granularity: str,
    start: datetime,
) -> list[dict]:
    """One paginated request: up to 5000 candles from `start` forward."""
    params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "granularity": granularity,
        "count": _MAX_COUNT,
        "price": "M",
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    api.request(r)
    return r.response.get("candles", [])


def fetch_history(
    instrument: str,
    granularity: str = "H1",
    years: float = 4.0,
    end: datetime | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download `years` of candles ending at `end` (default: now), paginated."""
    if granularity not in _GRANULARITY_SECONDS:
        raise ValueError(
            f"granularity {granularity!r} not supported; "
            f"choose from {sorted(_GRANULARITY_SECONDS)}"
        )

    end = end or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    cursor = end - timedelta(days=365.25 * years)

    api = _client()
    bar = timedelta(seconds=_GRANULARITY_SECONDS[granularity])
    rows: list[dict] = []
    page = 0

    while cursor < end:
        page += 1
        candles = _fetch_page(api, instrument, granularity, cursor)
        if not candles:
            break

        for c in candles:
            if not c.get("complete", False):
                continue
            ts = pd.Timestamp(c["time"]).tz_convert("UTC")
            rows.append(
                {
                    "time": ts,
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c.get("volume", 0)),
                }
            )

        last_ts = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC")
        new_cursor = last_ts.to_pydatetime() + bar
        if new_cursor <= cursor:      # safety: no forward progress -> stop
            break
        cursor = new_cursor

        if verbose:
            print(
                f"  [{instrument} {granularity}] page {page}: "
                f"{len(rows)} candles, up to {last_ts:%Y-%m-%d %H:%M}"
            )
        time.sleep(_THROTTLE_SECONDS)

    if not rows:
        raise RuntimeError(
            f"No candles returned for {instrument} {granularity}. "
            "Check the instrument name and your practice-account API key."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset="time").set_index("time")
    df = df.sort_index()
    df.index.name = None
    return df


def get_history(
    instrument: str,
    granularity: str = "H1",
    years: float = 4.0,
    refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cached wrapper around fetch_history. Set refresh=True to re-download."""
    path = _cache_path(instrument, granularity, years)
    if not refresh and os.path.exists(path):
        df = pd.read_pickle(path)
        if verbose:
            print(
                f"[cache] {instrument} {granularity}: {len(df)} candles "
                f"({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})"
            )
        return df

    if verbose:
        print(f"[fetch] {instrument} {granularity}: downloading {years}y ...")
    df = fetch_history(instrument, granularity, years, verbose=verbose)
    df.to_pickle(path)
    if verbose:
        print(
            f"[saved] {path} — {len(df)} candles "
            f"({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})"
        )
    return df


if __name__ == "__main__":
    # Smoke test: pull a small window first so failures are fast, then 4y H4+H1.
    print("=== data.py smoke test ===")
    small = fetch_history("EUR_USD", "H1", years=0.05)  # ~2.5 weeks
    print(f"small pull OK: {len(small)} candles\n{small.tail(3)}\n")

    h4 = get_history("EUR_USD", "H4", years=4.0)
    h1 = get_history("EUR_USD", "H1", years=4.0)
    print(f"\nEUR_USD H4: {len(h4)} candles | H1: {len(h1)} candles")
    print("data.py ready for M1.")