"""Tests for Engine C (daily zone rejection, core/engine_c.py)."""

import numpy as np
import pandas as pd
import pytest

import engine_c as C
from synth import make_ohlc


def _daily(closes, seed=3):
    return make_ohlc(closes, start="2016-01-04 21:00", freq="1D", wick=0.004, seed=seed)


@pytest.fixture(scope="module")
def closes():
    # A ranging market with repeated tests of the same highs and lows,
    # plus noise, so zones form and get retested.
    n = 1500
    rng = np.random.default_rng(42)
    t = np.arange(n)
    return 1.30 + 0.04 * np.sin(t / 25) + 0.02 * np.sin(t / 7) + np.cumsum(0.002 * rng.standard_normal(n)) * 0.3


@pytest.fixture(scope="module")
def market(closes):
    return _daily(closes)


def run(df, **kw):
    return C.run_engine_c(df, "GBP_NZD", C.EngineCConfig(**kw))


# ---- zone building ------------------------------------------------------- #

def test_build_zones_needs_enough_touches():
    zones = C.build_zones([1.10, 1.1003, 1.20], tol=0.001, min_touches=2, min_width=0.0)
    assert zones == [(1.10, 1.1003, 2)]


def test_build_zones_widens_thin_zones():
    (bottom, top, n), = C.build_zones([1.2, 1.2], tol=0.001, min_touches=2, min_width=0.01)
    assert top - bottom == pytest.approx(0.01) and n == 2


def test_swings_are_known_only_after_lookback(market):
    highs, lows = C.confirmed_swings(market, 5)
    assert highs and lows
    assert all(c == i + 5 for c, i, _ in highs + lows)


# ---- engine behaviour -------------------------------------------------- #

def test_produces_trades(market):
    trades, stats = run(market)
    assert len(trades) >= 10
    assert stats["n_trades"] == len(trades)
    assert set(stats["exit_mix"]) == {"target", "stop", "early"}


def test_sells_only_takes_no_buys(market):
    trades, _ = run(market, sells_only=True)
    assert trades and all(t.direction == -1 for t in trades)


def test_both_directions_takes_buys(market):
    trades, _ = run(market, sells_only=False)
    assert any(t.direction == 1 for t in trades)


def test_stop_is_minus_one_r_and_target_is_rr_without_costs(market):
    for rr in (1.0, 2.0):
        trades, _ = run(market, costs_on=False, sells_only=False, rr=rr)
        stops = [t for t in trades if t.exit_reason == "stop"]
        targets = [t for t in trades if t.exit_reason == "target"]
        assert stops and targets
        for t in stops:
            assert t.r_multiple == pytest.approx(-1.0)
        for t in targets:
            assert t.r_multiple == pytest.approx(rr)


def test_costs_reduce_returns_without_changing_signals(market):
    gross, _ = run(market, costs_on=False)
    net, _ = run(market, costs_on=True)
    assert [t.entry_ts for t in gross] == [t.entry_ts for t in net]
    assert sum(t.pnl_pct for t in net) < sum(t.pnl_pct for t in gross)


def test_stop_sits_above_entry_for_sells(market):
    trades, _ = run(market, sells_only=False)
    for t in trades:
        assert (t.initial_stop > t.entry_px) if t.direction == -1 else (t.initial_stop < t.entry_px)


def test_early_exit_toggle(market):
    on, _ = run(market, early_exit=True)
    off, _ = run(market, early_exit=False)
    assert any(t.exit_reason == "early" for t in on)
    assert not any(t.exit_reason == "early" for t in off)


def test_no_friday_session_entries(market):
    trades, _ = run(market, sells_only=False)
    entries = pd.DatetimeIndex([t.entry_ts for t in trades])
    assert (C._session_weekday(entries) != 4).all()
    with_fri, _ = run(market, sells_only=False, skip_friday=False)
    assert (C._session_weekday(pd.DatetimeIndex([t.entry_ts for t in with_fri])) == 4).any()


def test_trades_never_overlap(market):
    trades, _ = run(market, sells_only=False)
    for a, b in zip(trades, trades[1:]):
        assert b.entry_ts > a.exit_ts


# ---- no lookahead ---------------------------------------------------------- #

@pytest.mark.parametrize("frac", [0.5, 0.8])
def test_no_lookahead_truncation(market, frac):
    cut = int(len(market) * frac)
    full, _ = run(market, sells_only=False)
    part, _ = run(market.iloc[:cut], sells_only=False)
    cutoff = market.index[cut - 1]
    key = lambda ts: [(t.entry_ts, t.exit_ts, round(t.r_multiple, 10)) for t in ts if t.exit_ts < cutoff]
    assert key(full) == key(part)


def test_no_lookahead_future_shock(closes):
    """Shock every bar from an entry bar onward; entries up to it must not change."""
    base = _daily(closes)
    trades, _ = run(base, sells_only=False)
    pos = {ts: i for i, ts in enumerate(base.index)}
    for t in trades[:8]:
        k = pos[t.entry_ts]
        for factor in (0.97, 1.03):
            shocked = closes.copy()
            shocked[k:] = shocked[k:] * factor
            got, _ = run(_daily(shocked), sells_only=False)
            before = lambda ts: [(x.entry_ts, x.direction) for x in ts if x.entry_ts <= t.entry_ts]
            assert before(got) == before(trades)
