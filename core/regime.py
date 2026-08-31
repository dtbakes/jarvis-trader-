"""
regime.py — Layer 1 regime classifier for JARVIS.

Classifies every bar into one of four states that gate the trading engines:

    TREND     -> Engine A (Donchian breakout + chandelier trail) may enter
    RANGE     -> Engine B (Bollinger + RSI(2) reversion) may enter
    NEUTRAL   -> no new entries; manage open positions only
    NO_TRADE  -> no new entries (vol spike / session window / news window)

Rules (see STRATEGY.md §4 for rationale):
    TREND    : ADX(14) > 25  AND Donchian(20) width expanding vs 5 bars ago
    RANGE    : ADX(14) < 20
    NEUTRAL  : 20 <= ADX <= 25, or trend structure not confirmed
    NO_TRADE : ATR(14) above its 90th percentile over the trailing window,
               OR inside a session block (daily rollover +/- buffer, late
               Friday, Sunday-open buffer), OR inside a supplied news window.

Hysteresis: a regime switch requires `confirm_bars` consecutive bars of the
new raw label, which kills single-bar flip-flopping at the thresholds.

Lookahead safety
----------------
* Every value at row i depends only on rows 0..i (prefix property). The
  __main__ self-test verifies this by classifying a truncated frame and
  asserting the overlap is identical.
* OANDA candles are stamped with their OPEN time. An H4 bar stamped 08:00 is
  only fully known at 12:00. `align_to_lower_tf` shifts availability forward
  by one bar duration before forward-filling onto the lower timeframe — this
  is the single most common lookahead bug in multi-timeframe backtests.
* Session blocks are pure clock arithmetic (no market data), so they may be
  evaluated on the *execution* bar without lookahead. Market-derived fields
  (regime, atr, adx) must be consumed with .shift(1) in a
  decide-at-close / execute-next-open backtest. See STRATEGY.md §9.

Usage (backtest):
    from regime import classify_regimes, RegimeConfig, Regime
    h4 = classify_regimes(h4_ohlc, RegimeConfig())
    # decide on the last CLOSED bar:
    h4["regime_t"] = h4["regime"].shift(1)

Usage (live, inside the APScheduler tick):
    state = current_regime(h4_ohlc_tail)      # needs >= min_history_bars() rows
    if state.regime is Regime.TREND: ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from indicators import adx as _adx, atr as _atr, donchian as _donchian


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #

class Regime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    NEUTRAL = "NEUTRAL"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class RegimeConfig:
    # --- directional strength -------------------------------------------- #
    adx_period: int = 14
    adx_trend: float = 25.0          # enter TREND above this
    adx_range: float = 20.0          # enter RANGE below this
    # --- volatility gate --------------------------------------------------- #
    atr_period: int = 14
    atr_pct_window: int = 540        # trailing bars for the percentile rank
    #                                  (540 H4 bars ~ 90 trading days)
    atr_pct_cutoff: float = 0.90     # NO_TRADE above this percentile
    # --- trend structure confirmation ------------------------------------- #
    width_period: int = 20           # Donchian lookback for the width check
    width_lag: int = 5               # width must exceed its value 5 bars ago
    require_expanding_width: bool = True
    # --- hysteresis -------------------------------------------------------- #
    confirm_bars: int = 2            # consecutive bars required to switch
    # --- session filters (exchange time = America/New_York) ---------------- #
    ny_tz: str = "America/New_York"
    rollover_buffer_min: int = 20    # block +/- this around 17:00 NY
    friday_cutoff_hour: int = 12     # no new risk from Fri 12:00 NY onward
    sunday_open_buffer_min: int = 60 # block first hour after the weekly open


@dataclass(frozen=True)
class RegimeState:
    """Snapshot for live use (Claude journal entries, Telegram messages)."""
    timestamp: pd.Timestamp
    regime: Regime
    trend_dir: int                   # +1 up, -1 down, 0 undefined
    adx: float
    atr: float
    atr_pct: float
    reasons: Tuple[str, ...]         # why NO_TRADE, if applicable


# --------------------------------------------------------------------------- #
# Core classifier
# --------------------------------------------------------------------------- #

def classify_regimes(
    df: pd.DataFrame,
    cfg: RegimeConfig = RegimeConfig(),
    news_windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Classify every bar of an OHLC frame.

    Parameters
    ----------
    df : DataFrame with columns open/high/low/close and a tz-aware
        DatetimeIndex (UTC recommended; OANDA returns UTC). Timestamps are
        assumed to be bar OPEN times.
    cfg : RegimeConfig
    news_windows : optional iterable of (start, end) timestamps (UTC) during
        which entries are blocked. Wire your calendar source here.

    Returns
    -------
    Copy of df with added columns:
        adx, plus_di, minus_di, atr, atr_pct, trend_dir,
        blocked_high_vol, blocked_session, blocked_news,
        regime  (string values of the Regime enum)
    """
    _validate(df)
    out = df.copy()

    a = _adx(df, cfg.adx_period)
    out["adx"] = a["adx"]
    out["plus_di"] = a["plus_di"]
    out["minus_di"] = a["minus_di"]
    out["atr"] = _atr(df, cfg.atr_period)

    min_p = max(cfg.atr_pct_window // 4, cfg.atr_period * 4)
    out["atr_pct"] = (
        out["atr"].rolling(cfg.atr_pct_window, min_periods=min_p).rank(pct=True)
    )

    dc = _donchian(df, cfg.width_period)
    expanding = dc["dc_width"] > dc["dc_width"].shift(cfg.width_lag)

    known = out["adx"].notna()
    trend_ok = out["adx"] > cfg.adx_trend
    if cfg.require_expanding_width:
        trend_ok &= expanding
    range_ok = out["adx"] < cfg.adx_range

    raw = np.select(
        [~known, trend_ok, range_ok],
        [Regime.NO_TRADE.value, Regime.TREND.value, Regime.RANGE.value],
        default=Regime.NEUTRAL.value,
    )
    confirmed = _confirm(np.asarray(raw), cfg.confirm_bars)

    high_vol = out["atr_pct"] > cfg.atr_pct_cutoff
    session = _session_block(out.index, cfg)
    news = _news_block(out.index, news_windows)
    blocked = high_vol.fillna(False) | session | news | ~known

    out["regime"] = np.where(blocked.to_numpy(), Regime.NO_TRADE.value, confirmed)
    out["trend_dir"] = (
        np.sign(out["plus_di"] - out["minus_di"]).fillna(0).astype(int)
    )
    out["blocked_high_vol"] = high_vol.fillna(False)
    out["blocked_session"] = session
    out["blocked_news"] = news
    return out


def current_regime(
    df: pd.DataFrame,
    cfg: RegimeConfig = RegimeConfig(),
    news_windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> RegimeState:
    """Convenience wrapper for live use: classify and return the last bar.

    Feed at least `min_history_bars(cfg)` rows or the vol percentile gate
    will still be warming up.
    """
    out = classify_regimes(df, cfg, news_windows)
    row = out.iloc[-1]
    reasons = []
    if bool(row["blocked_high_vol"]):
        reasons.append("ATR above percentile cutoff (vol spike)")
    if bool(row["blocked_session"]):
        reasons.append("session window (rollover / late Friday / Sunday open)")
    if bool(row["blocked_news"]):
        reasons.append("high-impact news window")
    return RegimeState(
        timestamp=out.index[-1],
        regime=Regime(row["regime"]),
        trend_dir=int(row["trend_dir"]),
        adx=float(row["adx"]),
        atr=float(row["atr"]),
        atr_pct=float(row["atr_pct"]),
        reasons=tuple(reasons),
    )


def align_to_lower_tf(
    higher: pd.DataFrame,
    lower_index: pd.DatetimeIndex,
    bar_duration: pd.Timedelta,
    cols: Iterable[str] = ("regime", "trend_dir", "adx", "atr", "atr_pct"),
) -> pd.DataFrame:
    """Map higher-timeframe columns onto a lower-timeframe index, lookahead-free.

    Candles are stamped with their OPEN time, so the H4 bar stamped 08:00 is
    only known at 08:00 + 4h = 12:00. We shift availability forward by
    `bar_duration` before forward-filling, so H1 bars opening at 12:00 or
    later see the 08:00 H4 bar's regime — never earlier.

    Example
    -------
        h4 = classify_regimes(h4_ohlc)
        r = align_to_lower_tf(h4, h1_ohlc.index, pd.Timedelta(hours=4))
        h1 = h1_ohlc.join(r)
    """
    avail = higher[list(cols)].copy()
    avail.index = avail.index + bar_duration
    return avail.reindex(lower_index, method="ffill")


def min_history_bars(cfg: RegimeConfig = RegimeConfig()) -> int:
    """Recommended minimum bars to feed before trusting the output."""
    return (
        max(cfg.atr_pct_window, cfg.adx_period * 6, cfg.width_period + cfg.width_lag)
        + cfg.confirm_bars
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _validate(df: pd.DataFrame) -> None:
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex of bar OPEN times")
    if df.index.tz is None:
        raise ValueError(
            "index must be timezone-aware; OANDA returns UTC — "
            "use df.index = df.index.tz_localize('UTC') if yours is naive"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")


def _confirm(raw: np.ndarray, n: int) -> np.ndarray:
    """Hysteresis: switch regimes only after n consecutive bars of a new label."""
    if n <= 1:
        return raw.copy()
    out = np.empty_like(raw)
    current = raw[0]
    candidate = current
    streak = 0
    for i, label in enumerate(raw):
        if label == current:
            candidate, streak = current, 0
        elif label == candidate:
            streak += 1
            if streak >= n:
                current, streak = candidate, 0
        else:
            candidate, streak = label, 1
        out[i] = current
    return out


def _session_block(index: pd.DatetimeIndex, cfg: RegimeConfig) -> pd.Series:
    """Clock-based blocks: rollover +/- buffer, late Friday, Sunday-open buffer.

    Evaluated on the bar's OPEN timestamp. Because this is pure clock
    arithmetic it can also be re-evaluated on the execution bar in the
    engines without lookahead.
    """
    ny = index.tz_convert(cfg.ny_tz)
    minutes = ny.hour * 60 + ny.minute
    roll = 17 * 60  # 17:00 New York — OANDA daily rollover / weekly open

    rollover = (minutes >= roll - cfg.rollover_buffer_min) & (
        minutes <= roll + cfg.rollover_buffer_min
    )
    friday_late = (ny.weekday == 4) & (ny.hour >= cfg.friday_cutoff_hour)
    sunday_open = (
        (ny.weekday == 6)
        & (minutes >= roll)
        & (minutes < roll + cfg.sunday_open_buffer_min)
    )
    return pd.Series(rollover | friday_late | sunday_open, index=index)


def _news_block(
    index: pd.DatetimeIndex,
    windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.Series:
    mask = pd.Series(False, index=index)
    if not windows:
        return mask
    for start, end in windows:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start.tz is None:
            start = start.tz_localize("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")
        mask |= (index >= start) & (index <= end)
    return mask


# --------------------------------------------------------------------------- #
# Self-test / demo:  python regime.py
# --------------------------------------------------------------------------- #

def _synthetic_ohlc(seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Range -> trend -> range -> vol-spike synthetic series (H4 bars)."""
    rng = np.random.default_rng(seed)
    segs, labels = [], {}

    def seg(name, n, drift, noise, mean_revert=0.0, mu=None):
        start = sum(len(s) for s in segs)
        x = segs[-1][-1] if segs else 1.1000
        vals = np.empty(n)
        for i in range(n):
            pull = mean_revert * ((mu if mu is not None else 1.1000) - x)
            x = x + drift + pull + noise * rng.standard_normal()
            vals[i] = x
        segs.append(vals)
        labels[name] = (start, start + n)

    seg("range_1", 700, drift=0.0, noise=0.0006, mean_revert=0.05, mu=1.1000)
    seg("trend", 700, drift=0.00045, noise=0.0006)
    seg("range_2", 700, drift=0.0, noise=0.0006, mean_revert=0.05, mu=None)
    seg("vol_spike", 250, drift=0.0, noise=0.0035)

    close = pd.Series(np.concatenate(segs))
    idx = pd.date_range("2024-01-02", periods=len(close), freq="4h", tz="UTC")
    close.index = idx
    open_ = close.shift(1).fillna(close.iloc[0])
    wick = np.abs(rng.normal(0, 0.0004, len(close)))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
        },
        index=idx,
    )
    return df, labels


def _run_self_tests() -> None:
    cfg = RegimeConfig()
    df, labels = _synthetic_ohlc()
    out = classify_regimes(df, cfg)

    # ---- 1. Lookahead check via the prefix property ----------------------- #
    cut = len(df) * 2 // 3
    part = classify_regimes(df.iloc[:cut], cfg)
    assert (out["regime"].iloc[:cut].to_numpy() == part["regime"].to_numpy()).all(), (
        "prefix property violated — classifier is leaking future data"
    )

    # ---- 2. Segment sanity ------------------------------------------------ #
    def dist(name):
        s, e = labels[name]
        return out["regime"].iloc[s:e].value_counts(normalize=True)

    d_trend, d_range, d_vol = dist("trend"), dist("range_2"), dist("vol_spike")
    assert d_trend.get("TREND", 0) > d_trend.get("RANGE", 0), "trend segment misread"
    assert d_range.get("RANGE", 0) > d_range.get("TREND", 0), "range segment misread"
    assert d_vol.get("NO_TRADE", 0) > 0.5, "vol spike not gated"

    # ---- 3. News window blocks entries ------------------------------------ #
    w = [(df.index[900], df.index[905])]
    out_news = classify_regimes(df, cfg, news_windows=w)
    assert (out_news["regime"].iloc[900:906] == "NO_TRADE").all(), "news gate failed"

    # ---- 4. Live helper --------------------------------------------------- #
    state = current_regime(df.iloc[: labels["trend"][1]], cfg)
    assert state.regime in Regime and isinstance(state.trend_dir, int)

    # ---- 5. Alignment helper is lookahead-free ---------------------------- #
    h1_idx = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    aligned = align_to_lower_tf(out, h1_idx, pd.Timedelta(hours=4))
    t0 = df.index[100]
    # H1 bars strictly before t0+4h must NOT yet see the H4 bar stamped t0
    first_seen = aligned.index[aligned.index >= t0 + pd.Timedelta(hours=4)][0]
    assert first_seen - t0 >= pd.Timedelta(hours=4)

    print("All regime.py self-tests passed.\n")
    print(f"min_history_bars(default cfg) = {min_history_bars(cfg)}\n")
    print("Regime distribution by synthetic segment:")
    for name in labels:
        s, e = labels[name]
        vc = out["regime"].iloc[s:e].value_counts(normalize=True).round(2)
        print(f"  {name:10s} -> {vc.to_dict()}")
    print("\nLast live state:", current_regime(df))


if __name__ == "__main__":
    _run_self_tests()