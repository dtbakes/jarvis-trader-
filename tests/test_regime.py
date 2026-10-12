"""Tests for core/regime.py — the Layer 1 classifier that gates both engines.

Ports the __main__ self-test into pytest and adds edge cases for input
validation, hysteresis, session/news gates and multi-timeframe alignment."""

import numpy as np
import pandas as pd
import pytest

import regime as rg
from regime import Regime, RegimeConfig

CFG = RegimeConfig()


@pytest.fixture(scope="module")
def synthetic():
    """Range -> trend -> range -> vol-spike series with known segment labels."""
    return rg._synthetic_ohlc(seed=7)


@pytest.fixture(scope="module")
def classified(synthetic):
    df, _ = synthetic
    return rg.classify_regimes(df, CFG)


# ---- input validation ----------------------------------------------------- #

def test_rejects_missing_columns(synthetic):
    df, _ = synthetic
    with pytest.raises(ValueError, match="missing OHLC"):
        rg.classify_regimes(df.drop(columns=["low"]))


def test_rejects_naive_index(synthetic):
    df, _ = synthetic
    with pytest.raises(ValueError, match="timezone-aware"):
        rg.classify_regimes(df.tz_localize(None))


def test_rejects_unsorted_index(synthetic):
    df, _ = synthetic
    with pytest.raises(ValueError, match="sorted"):
        rg.classify_regimes(df.iloc[::-1])


# ---- correctness on known segments ---------------------------------------- #

def _dist(out, labels, name):
    s, e = labels[name]
    return out["regime"].iloc[s:e].value_counts(normalize=True)


def test_trend_segment_reads_as_trend(classified, synthetic):
    d = _dist(classified, synthetic[1], "trend")
    assert d.get("TREND", 0) > d.get("RANGE", 0)


def test_range_segment_reads_as_range(classified, synthetic):
    d = _dist(classified, synthetic[1], "range_2")
    assert d.get("RANGE", 0) > d.get("TREND", 0)


def test_vol_spike_is_gated(classified, synthetic):
    assert _dist(classified, synthetic[1], "vol_spike").get("NO_TRADE", 0) > 0.5


def test_warmup_bars_are_no_trade(classified):
    assert (classified["regime"].iloc[: CFG.adx_period] == "NO_TRADE").all()


def test_output_values_are_valid_enum_members(classified):
    assert set(classified["regime"].unique()) <= {r.value for r in Regime}
    assert set(classified["trend_dir"].unique()) <= {-1, 0, 1}


# ---- lookahead -------------------------------------------------------------- #

@pytest.mark.parametrize("frac", [0.4, 0.66, 0.9])
def test_prefix_property(synthetic, classified, frac):
    df, _ = synthetic
    cut = int(len(df) * frac)
    part = rg.classify_regimes(df.iloc[:cut], CFG)
    np.testing.assert_array_equal(
        classified["regime"].iloc[:cut].to_numpy(), part["regime"].to_numpy()
    )


def test_align_to_lower_tf_never_sees_unfinished_bar(classified, synthetic):
    df, _ = synthetic
    h1 = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    aligned = rg.align_to_lower_tf(classified, h1, pd.Timedelta(hours=4))
    # Mark one H4 bar uniquely and confirm H1 only sees it once it has closed.
    marked = classified.copy()
    t0 = df.index[200]
    marked.loc[t0, "adx"] = -999.0
    aligned = rg.align_to_lower_tf(marked, h1, pd.Timedelta(hours=4))
    seen = aligned.index[aligned["adx"] == -999.0]
    assert len(seen) > 0
    assert seen.min() == t0 + pd.Timedelta(hours=4)


# ---- gates -------------------------------------------------------------------- #

def test_news_window_blocks_entries(synthetic):
    df, _ = synthetic
    w = [(df.index[900], df.index[905])]
    out = rg.classify_regimes(df, CFG, news_windows=w)
    assert (out["regime"].iloc[900:906] == "NO_TRADE").all()
    assert out["blocked_news"].iloc[900:906].all()
    assert not out["blocked_news"].iloc[910]


def test_news_window_accepts_naive_timestamps(synthetic):
    df, _ = synthetic
    w = [(df.index[900].tz_localize(None), df.index[905].tz_localize(None))]
    out = rg.classify_regimes(df, CFG, news_windows=w)
    assert out["blocked_news"].iloc[900:906].all()


@pytest.mark.parametrize(
    "ny_time,blocked",
    [
        ("2026-10-07 16:30", False),  # Wed, before rollover buffer
        ("2026-10-07 16:45", True),   # Wed, inside 17:00 +/- 20 min
        ("2026-10-07 17:15", True),
        ("2026-10-07 17:30", False),
        ("2026-10-09 11:59", False),  # Fri morning still open
        ("2026-10-09 12:00", True),   # Fri cutoff
        ("2026-10-11 17:30", True),   # Sun, first hour after weekly open
        ("2026-10-11 18:30", False),
    ],
)
def test_session_block(ny_time, blocked):
    ts = pd.DatetimeIndex([pd.Timestamp(ny_time, tz="America/New_York")]).tz_convert("UTC")
    assert bool(rg._session_block(ts, CFG).iloc[0]) is blocked


# ---- hysteresis ------------------------------------------------------------- #

def test_confirm_ignores_single_bar_flicker():
    raw = np.array(["RANGE", "RANGE", "TREND", "RANGE", "RANGE"], dtype=object)
    assert list(rg._confirm(raw, 2)) == ["RANGE"] * 5


def test_confirm_switches_after_n_bars():
    raw = np.array(["RANGE", "TREND", "TREND", "TREND"], dtype=object)
    assert list(rg._confirm(raw, 2)) == ["RANGE", "RANGE", "TREND", "TREND"]


def test_confirm_n1_is_passthrough():
    raw = np.array(["A", "B", "A"], dtype=object)
    assert list(rg._confirm(raw, 1)) == ["A", "B", "A"]


# ---- live helper ---------------------------------------------------------- #

def test_current_regime_returns_last_bar_state(synthetic, classified):
    df, _ = synthetic
    state = rg.current_regime(df)
    assert state.timestamp == df.index[-1]
    assert state.regime.value == classified["regime"].iloc[-1]
    assert isinstance(state.trend_dir, int)
    if state.regime is Regime.NO_TRADE and classified["blocked_high_vol"].iloc[-1]:
        assert any("vol" in r for r in state.reasons)


def test_min_history_covers_percentile_window():
    assert rg.min_history_bars(CFG) >= CFG.atr_pct_window
