"""Integration tests for the backtest engines (core/engine_a.py, engine_b.py).

These run the full pipeline (regime -> signals -> fills -> stats) on synthetic
data and check the properties a backtest must have to be trusted: no
lookahead, costs always hurt, stops honoured, config switches respected."""

import numpy as np
import pandas as pd
import pytest

import engine_a as A
import engine_b as B
import regime as rg

ENGINES = {
    "A": (A.run_engine_a, A.EngineAConfig),
    "B": (B.run_engine_b, B.EngineBConfig),
}


@pytest.fixture(scope="module")
def market():
    df, _ = rg._synthetic_ohlc(seed=7)
    return df


def run(engine, df, **cfg_kwargs):
    fn, Cfg = ENGINES[engine]
    return fn(df, df, "EUR_USD", Cfg(**cfg_kwargs))


@pytest.mark.parametrize("engine", ["A", "B"])
def test_engine_produces_trades_and_stats(market, engine):
    trades, stats = run(engine, market)
    assert len(trades) > 0
    assert stats["n_trades"] == len(trades)
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert 0.0 <= stats["max_dd"] <= 1.0


@pytest.mark.parametrize("engine", ["A", "B"])
def test_trades_are_chronological_and_non_overlapping(market, engine):
    trades, _ = run(engine, market)
    for t in trades:
        assert t.exit_ts > t.entry_ts
    for prev, nxt in zip(trades, trades[1:]):
        assert nxt.entry_ts >= prev.exit_ts, "one position per pair, no pyramiding"


@pytest.mark.parametrize("engine", ["A", "B"])
def test_initial_stop_on_correct_side(market, engine):
    trades, _ = run(engine, market)
    for t in trades:
        if t.direction == 1:
            assert t.initial_stop < t.entry_px
        else:
            assert t.initial_stop > t.entry_px


@pytest.mark.parametrize("engine", ["A", "B"])
def test_costs_strictly_reduce_returns(market, engine):
    gross, _ = run(engine, market, costs_on=False)
    net, _ = run(engine, market, costs_on=True)
    # Same signals either way (costs never change entries), worse results.
    assert [t.entry_ts for t in gross] == [t.entry_ts for t in net]
    assert sum(t.pnl_pct for t in net) < sum(t.pnl_pct for t in gross)


@pytest.mark.parametrize("engine", ["A", "B"])
def test_hard_stop_without_costs_is_exactly_minus_one_r(market, engine):
    trades, _ = run(engine, market, costs_on=False)
    stops = [t for t in trades if t.exit_reason == "stop"]
    assert stops, "fixture should exercise the stop path"
    for t in stops:
        assert t.r_multiple == pytest.approx(-1.0)


@pytest.mark.parametrize("engine", ["A", "B"])
def test_hard_stop_with_costs_loses_more_than_one_r(market, engine):
    trades, _ = run(engine, market, costs_on=True)
    for t in (t for t in trades if t.exit_reason == "stop"):
        assert t.r_multiple < -1.0


@pytest.mark.parametrize("engine", ["A", "B"])
def test_long_only_config_takes_no_shorts(market, engine):
    trades, _ = run(engine, market, allow_shorts=False)
    assert all(t.direction == 1 for t in trades)


@pytest.mark.parametrize("engine", ["A", "B"])
@pytest.mark.parametrize("frac", [0.5, 0.8])
def test_no_lookahead_truncation(market, engine, frac):
    """Trades that closed before the cut must be identical whether or not the
    engine could see the bars after it. Any difference = future leakage."""
    cut = int(len(market) * frac)
    full, _ = run(engine, market)
    part, _ = run(engine, market.iloc[:cut])
    cutoff = market.index[cut - 1]
    done_full = [(t.entry_ts, t.exit_ts, round(t.r_multiple, 10)) for t in full if t.exit_ts < cutoff]
    done_part = [(t.entry_ts, t.exit_ts, round(t.r_multiple, 10)) for t in part if t.exit_ts < cutoff]
    assert done_full == done_part


@pytest.fixture(scope="module")
def rebuilt_closes(market):
    return market["close"].to_numpy()


def _with_future_shock(closes, k, factor):
    """Same path up to bar k-1, then every close from bar k onward jumps by
    `factor`. open[k] == close[k-1] is unchanged, so the only thing that
    differs is information that did not exist yet when bar k's entry was
    decided."""
    from synth import make_ohlc
    shocked = closes.copy()
    shocked[k:] = shocked[k:] * factor
    return make_ohlc(shocked, seed=11)


@pytest.mark.parametrize("engine", ["A", "B"])
def test_no_lookahead_future_shock(rebuilt_closes, engine):
    """Decide-at-close / execute-at-next-open means an entry filled at bar k
    may only use bars 0..k-1 plus open[k]. Shock every bar from k onward (up
    and down) and every entry at or before bar k must be unchanged. This
    catches even a one-bar `.shift(-1)` leak that truncation tests miss."""
    from synth import make_ohlc
    base = make_ohlc(rebuilt_closes, seed=11)
    trades, _ = run(engine, base)
    assert trades
    pos = {ts: i for i, ts in enumerate(base.index)}
    for t in trades[:8]:
        k = pos[t.entry_ts]
        for factor in (0.97, 1.03):
            shocked = _with_future_shock(rebuilt_closes, k, factor)
            got, _ = run(engine, shocked)
            before = lambda ts: [(x.entry_ts, x.direction) for x in ts if x.entry_ts <= t.entry_ts]
            assert before(got) == before(trades), (
                f"entry at {t.entry_ts} changed when only bars >= it were shocked "
                f"(factor {factor}) -> engine is reading the future"
            )


@pytest.mark.parametrize("engine", ["A", "B"])
def test_entries_respect_session_blocks(market, engine):
    trades, _ = run(engine, market)
    entries = pd.DatetimeIndex([t.entry_ts for t in trades])
    assert not rg._session_block(entries, rg.RegimeConfig()).any()


def test_engine_a_handles_no_signal_market():
    """A dead-flat market should produce zero trades and a clean summary."""
    from synth import make_ohlc
    flat = make_ohlc(np.full(900, 1.1), wick=0.0)
    trades, stats = A.run_engine_a(flat, flat, "EUR_USD")
    assert trades == []
    assert stats["n_trades"] == 0


@pytest.mark.parametrize("pair,pip", [("EUR_USD", 0.0001), ("USD_JPY", 0.01), ("GBP_JPY", 0.01)])
def test_pip_size(pair, pip):
    assert A.pip_size(pair) == pip
