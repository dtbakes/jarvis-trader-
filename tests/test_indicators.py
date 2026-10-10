"""Unit tests for core/indicators.py — hand-checked values plus the
lookahead-free 'prefix property' every indicator promises."""

import numpy as np
import pandas as pd
import pytest

import indicators as ind
from synth import make_ohlc, random_walk


@pytest.fixture
def df():
    return make_ohlc(random_walk(600, seed=3), seed=3)


# ---- hand-checked values -------------------------------------------------- #

def test_true_range_uses_prior_close_gap():
    df = pd.DataFrame(
        {"open": [10, 12], "high": [11, 13], "low": [9, 12.5], "close": [10, 12.8]},
        index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
    )
    tr = ind.true_range(df)
    assert tr.iloc[0] == 2.0           # H-L on the first bar
    assert tr.iloc[1] == 3.0           # gap: |13 - prior close 10| beats H-L of 0.5


def test_donchian_includes_current_bar():
    df = make_ohlc([1, 2, 3, 2, 1], freq="h", wick=0.0)
    dc = ind.donchian(df, period=3)
    assert dc["dc_upper"].iloc[:2].isna().all()
    assert dc["dc_upper"].iloc[2] == 3.0
    assert dc["dc_lower"].iloc[4] == 1.0
    pd.testing.assert_series_equal(
        dc["dc_width"], dc["dc_upper"] - dc["dc_lower"], check_names=False
    )


def test_bollinger_flat_series_collapses_bands():
    close = pd.Series([1.1] * 30)
    bb = ind.bollinger(close, period=20)
    last = bb.iloc[-1]
    assert last["bb_upper"] == pytest.approx(1.1)
    assert last["bb_lower"] == pytest.approx(1.1)


def test_rsi_degenerate_cases():
    rising = pd.Series(np.arange(1, 20, dtype=float))
    flat = pd.Series([5.0] * 20)
    assert ind.rsi(rising, period=2).iloc[-1] == 100.0
    assert ind.rsi(flat, period=2).iloc[-1] == 50.0


def test_rsi_stays_in_bounds(df):
    r = ind.rsi(df["close"], period=14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_adx_components_are_bounded(df):
    a = ind.adx(df).dropna()
    for col in ("adx", "plus_di", "minus_di"):
        assert ((a[col] >= 0) & (a[col] <= 100)).all(), col


def test_atr_is_positive_after_warmup(df):
    a = ind.atr(df, 14)
    assert a.iloc[:13].isna().all()
    assert (a.dropna() > 0).all()


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        ind.atr(pd.DataFrame({"close": [1.0, 2.0]}))


# ---- no lookahead --------------------------------------------------------- #

PREFIX_CASES = [
    ("atr", lambda d: ind.atr(d, 14)),
    ("adx", lambda d: ind.adx(d, 14)["adx"]),
    ("rsi2", lambda d: ind.rsi(d["close"], 2)),
    ("rsi14", lambda d: ind.rsi(d["close"], 14)),
    ("bb_mid", lambda d: ind.bollinger(d["close"])["bb_mid"]),
    ("donchian", lambda d: ind.donchian(d, 55)["dc_upper"]),
    ("ema", lambda d: ind.ema(d["close"], 50)),
]


@pytest.mark.parametrize("name,fn", PREFIX_CASES, ids=[c[0] for c in PREFIX_CASES])
@pytest.mark.parametrize("cut", [100, 333, 599])
def test_prefix_property(df, name, fn, cut):
    """Value at bar i must depend only on bars 0..i: computing on a truncated
    frame gives identical values for the overlap. A failure here means the
    indicator is peeking at the future and every backtest is invalid."""
    full = fn(df).iloc[:cut]
    part = fn(df.iloc[:cut])
    pd.testing.assert_series_equal(full, part, check_names=False)
