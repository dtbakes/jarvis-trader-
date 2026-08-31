"""
indicators.py — pure indicator math for JARVIS.

Design contract
---------------
* All functions are vectorized and **lookahead-free**: the value at row i is
  fully determined by rows 0..i. This gives the "prefix property" — computing
  on a truncated DataFrame yields identical values for the overlapping rows —
  which regime.py exploits as an automatic lookahead self-test.
* Head values are NaN until enough history exists. Consumers must handle NaN.
* Input DataFrames need lowercase columns: open, high, low, close.
* Wilder-family indicators (ATR, ADX, RSI) use Wilder's smoothing, i.e. an
  EMA with alpha = 1/period, matching the classic definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["true_range", "atr", "adx", "rsi", "bollinger", "donchian", "ema"]


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing = EMA with alpha = 1/period, seeded recursively."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Classic true range: max of (H-L, |H-prevC|, |L-prevC|)."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). Used for stops, sizing, and the vol gate."""
    return _wilder(true_range(df), period).rename(f"atr_{period}")


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index.

    Returns a DataFrame with columns: adx, plus_di, minus_di.
    adx measures trend *strength* (direction-agnostic); the DI spread gives
    direction: plus_di > minus_di implies an up-trend and vice versa.
    """
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr_ = _wilder(true_range(df), period)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ = _wilder(dx, period)

    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Relative Strength Index (Wilder). Engine B uses period=2 (Connors)."""
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), period)
    loss = _wilder(-delta.clip(upper=0.0), period)

    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)

    # Degenerate cases: pure gains -> 100, dead-flat market -> 50.
    zero_loss = loss.eq(0.0)  # False for NaN head, so warmup NaNs survive
    out = out.mask(zero_loss & gain.gt(0.0), 100.0)
    out = out.mask(zero_loss & gain.eq(0.0), 50.0)
    return out.rename(f"rsi_{period}")


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands. Returns bb_mid, bb_upper, bb_lower."""
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + num_std * sd, "bb_lower": mid - num_std * sd}
    )


def donchian(df: pd.DataFrame, period: int = 55) -> pd.DataFrame:
    """Donchian channel, INCLUSIVE of the current bar.

    For breakout tests against the *prior* N-bar extreme (Engine A), shift the
    result by 1 in the engine:  close > dc_upper.shift(1).
    """
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_width": upper - lower})


def ema(close: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average (span parameterization)."""
    return close.ewm(span=period, adjust=False, min_periods=period).mean().rename(
        f"ema_{period}"
    )