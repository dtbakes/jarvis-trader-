"""Synthetic market data helpers shared by the test suite."""

import numpy as np
import pandas as pd


def make_ohlc(closes, start="2024-01-02 00:00", freq="4h", wick=0.0002, seed=0):
    """Build a tz-aware OHLC frame from a close path (bar OPEN timestamps)."""
    rng = np.random.default_rng(seed)
    close = pd.Series(np.asarray(closes, dtype=float))
    idx = pd.date_range(start, periods=len(close), freq=freq, tz="UTC")
    close.index = idx
    open_ = close.shift(1).fillna(close.iloc[0])
    w = np.abs(rng.normal(0, wick, len(close))) if wick else np.zeros(len(close))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + w,
            "low": np.minimum(open_, close) - w,
            "close": close,
        },
        index=idx,
    )


def random_walk(n, start=1.10, drift=0.0, noise=0.0006, seed=0):
    rng = np.random.default_rng(seed)
    return start + np.cumsum(drift + noise * rng.standard_normal(n))
