"""Tests for Engine D (momentum + carry portfolio, core/engine_d.py).
Synthetic prices and rates only; nothing touches OANDA or FRED."""

import numpy as np
import pandas as pd
import pytest

import engine_d as D

PAIRS = [pair for pair, _ in D.LEGS.values()]


def make_pairs(n=1600, seed=0, drift=None, shock_from=None, factor=1.0):
    idx = pd.bdate_range("2017-01-02", periods=n, tz="UTC")
    rng = np.random.default_rng(seed)
    out = {}
    for j, pair in enumerate(PAIRS):
        start = 110.0 if pair.endswith("JPY") else 1.2
        mu = (drift or {}).get(pair, 0.0)
        r = mu + 0.006 * rng.standard_normal(n)
        px = start * np.exp(np.cumsum(r))
        if shock_from is not None:
            px[shock_from:] *= factor
        out[pair] = pd.Series(px, index=idx)
    return out


def make_rates(start="2016-01-01", months=96, overrides=None):
    idx = pd.date_range(start, periods=months, freq="MS")
    df = pd.DataFrame({c: 1.0 for c in D.FRED_SERIES}, index=idx)
    for (ccy, from_date), val in (overrides or {}).items():
        df.loc[from_date:, ccy] = val
    return df


@pytest.fixture(scope="module")
def pairs():
    return make_pairs()


def test_runs_and_reports(pairs):
    daily, w, st = D.run_engine_d(pairs, make_rates())
    assert len(w) > 30 and st["n_trades"] > 30
    assert {"profit_factor", "sharpe", "max_dd", "cagr"} <= st.keys()
    assert daily.index[0] > w.index[0]


def test_inverted_pairs_flip_direction():
    pairs = make_pairs(drift={"USD_JPY": 0.002})
    px = D.currency_prices(pairs)
    assert px["JPY"].iloc[-1] < px["JPY"].iloc[0]


def test_leg_weights_are_capped(pairs):
    cfg = D.EngineDConfig(target_vol=50.0, max_leg_weight=1.0)
    _, w, _ = D.run_engine_d(pairs, make_rates(), cfg)
    assert w.abs().max().max() <= 1.0 + 1e-12


def test_carry_only_goes_long_the_high_yielder(pairs):
    rates = make_rates(overrides={("AUD", "2016-01-01"): 5.0, ("JPY", "2016-01-01"): -0.5})
    _, w, _ = D.run_engine_d(pairs, rates, use_mom=False, use_carry=True)
    assert (w["AUD"] > 0).all() and (w["JPY"] < 0).all()
    assert (w["EUR"] == 0).all()        # same rate as USD -> no position


def test_rate_change_is_used_only_after_the_lag(pairs):
    base = make_rates()
    changed = make_rates(overrides={("NZD", "2020-03-01"): 9.0})
    _, w0, _ = D.run_engine_d(pairs, base, use_mom=False)
    _, w1, _ = D.run_engine_d(pairs, changed, use_mom=False)
    march_end = w0.index[(w0.index.year == 2020) & (w0.index.month == 3)][0]
    april_end = w0.index[(w0.index.year == 2020) & (w0.index.month == 4)][0]
    assert w1.loc[march_end, "NZD"] == w0.loc[march_end, "NZD"]   # March rate not known yet
    assert w1.loc[april_end, "NZD"] > 0                            # known in April


def test_positive_carry_is_earned(pairs):
    flat = make_rates()
    rich = make_rates(overrides={("AUD", "2016-01-01"): 6.0})
    cfg = D.EngineDConfig(costs_on=False)
    d0, _, _ = D.run_engine_d(pairs, flat, cfg, use_mom=False)
    d1, _, _ = D.run_engine_d(pairs, rich, cfg, use_mom=False)
    # With AUD long and AUD yielding more, accrual adds to the return.
    assert d1.sum() > d0.sum()


def test_costs_reduce_returns(pairs):
    rates = make_rates()
    gross, w0, _ = D.run_engine_d(pairs, rates, D.EngineDConfig(costs_on=False))
    net, w1, _ = D.run_engine_d(pairs, rates, D.EngineDConfig(costs_on=True))
    pd.testing.assert_frame_equal(w0, w1)
    assert net.sum() < gross.sum()


def test_no_lookahead_future_shock():
    base = make_pairs()
    _, w, _ = D.run_engine_d(base, make_rates())
    pos = {ts: i for i, ts in enumerate(base[PAIRS[0]].index)}
    for day in w.index[::7][:6]:
        k = pos[day] + 1                  # everything after the decision close
        for factor in (0.95, 1.05):
            _, ws, _ = D.run_engine_d(make_pairs(shock_from=k, factor=factor), make_rates())
            pd.testing.assert_frame_equal(ws.loc[:day], w.loc[:day])


def test_weights_start_after_the_rebalance_day(pairs):
    daily, w, _ = D.run_engine_d(pairs, make_rates(), D.EngineDConfig(costs_on=False))
    # The rebalance day itself earns nothing new; weights go live the next day.
    assert w.index[0] not in daily.index


def test_stale_rates_stop_the_run():
    rates = make_rates(months=24)        # ends early 2017
    with pytest.raises(ValueError, match="stale"):
        D.check_rates_fresh(rates, pd.Timestamp("2026-10-01", tz="UTC"))
    D.check_rates_fresh(make_rates(months=130), pd.Timestamp("2026-10-01", tz="UTC"))


def test_month_end_days():
    idx = pd.bdate_range("2024-01-01", "2024-03-31", tz="UTC")
    got = D.month_end_days(idx)
    assert [d.day for d in got] == [31, 29, 29]


# ---- rate data splicing ------------------------------------------------------- #

def _monthly(start, months, value):
    return pd.Series(float(value), index=pd.date_range(start, periods=months, freq="MS"))


def test_splice_prefers_primary_and_fills_later_months():
    primary = _monthly("2016-01-01", 24, 1.0)             # stops end of 2017
    backup = _monthly("2016-06-01", 60, 2.0)              # runs to 2021
    out = D.splice([primary, None, backup])
    assert out.loc["2016-06-01"] == 1.0                   # primary wins on overlap
    assert out.loc["2020-01-01"] == 2.0                   # backup fills the gap
    assert out.index[-1] == backup.index[-1]


def test_monthly_mean_from_daily():
    idx = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-15"])
    out = D.monthly_mean(pd.Series([1.0, 3.0, 5.0], index=idx))
    assert list(out.index) == list(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    assert list(out) == [2.0, 5.0]


def test_fetch_rates_splices_backups_and_manual(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(D, "FRED_SERIES", {"USD": ["U1"], "CHF": ["C1", "C2"]})
    fake = {"U1": _monthly("2016-01-01", 130, 2.0),
            "C1": _monthly("2016-01-01", 98, -0.75),       # stops early 2024
            "C2": None}                                   # series not found
    monkeypatch.setattr(D, "_read_fred", lambda sid, cache, refresh: fake[sid])
    (tmp_path / "rates").mkdir()
    (tmp_path / "rates" / "manual.csv").write_text(
        "month,currency,rate\n2024-03,CHF,1.5\n2026-09,CHF,0.0\n")
    rates = D.fetch_rates(verbose=False)
    assert rates.loc["2020-01-01", "CHF"] == -0.75
    assert rates.loc["2024-03-01", "CHF"] == 1.5
    assert rates["CHF"].last_valid_index() == pd.Timestamp("2026-09-01")
