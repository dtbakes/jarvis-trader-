"""Tests for the Engine D monthly signal report (core/signal_report.py).
Synthetic data only; nothing touches OANDA, FRED or Telegram."""

import pandas as pd
import pytest

import core.signal_report as SR
from test_engine_d import make_pairs, make_rates


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize("ts,last", [
    ("2026-10-29 21:00", True),    # Thu open (Fri 30 Oct session); next opens Sun 1 Nov
    ("2026-10-28 21:00", False),   # Wed open; next opens Thu 29 Oct
    ("2026-09-29 21:00", False),   # Tue open; next opens Wed 30 Sep
    ("2026-09-30 21:00", True),    # Wed open; next opens Thu 1 Oct
])
def test_is_last_bar_of_month(ts, last):
    assert SR.is_last_bar_of_month(_ts(ts)) is last


def test_due_month_waits_for_the_month_to_finish():
    idx = pd.DatetimeIndex([_ts("2026-09-29 21:00"), _ts("2026-09-30 21:00"),
                            _ts("2026-10-01 21:00"), _ts("2026-10-27 21:00")])
    assert SR.due_month(idx) == _ts("2026-09-30 21:00")      # September is done
    idx2 = idx.append(pd.DatetimeIndex([_ts("2026-10-29 21:00")]))
    assert SR.due_month(idx2) == _ts("2026-10-29 21:00")     # October just closed


@pytest.fixture
def inputs():
    pairs = make_pairs(n=1600)
    rates = make_rates(overrides={("AUD", "2016-01-01"): 5.0})
    return pairs, rates


def test_report_explains_every_currency_and_places_no_orders(inputs):
    pairs, rates = inputs
    px = SR.D.currency_prices(pairs)
    day = SR.D.month_end_days(px.index)[-2]
    text, details = SR.build_report(pairs, rates, day)
    assert "no orders placed" in text
    for ccy in SR.D.LEGS:
        assert f"• {ccy}:" in text
    # The report's weights are exactly the backtest's weights for that day.
    expected = SR.D.target_weights(px, rates, SR.D.EngineDConfig()).loc[day]
    pd.testing.assert_series_equal(details["weight"], expected, check_names=False)


def test_explain_wording():
    row = pd.Series({"weight": 0.3, "mom": 1.0, "trend_12m": 0.05, "rate_vs_usd": 2.0, "vol": 0.1})
    assert "LONG 0.30x (full signal)" in SR.explain("AUD", row)
    row = pd.Series({"weight": 0.0, "mom": 1.0, "trend_12m": 0.05, "rate_vs_usd": -1.0, "vol": 0.1})
    assert "flat" in SR.explain("EUR", row) and "cancel" in SR.explain("EUR", row)
    row = pd.Series({"weight": -0.1, "mom": -1.0, "trend_12m": -0.03, "rate_vs_usd": 0.0, "vol": 0.1})
    assert "SHORT 0.10x (half signal)" in SR.explain("CHF", row)


def test_run_monthly_sends_once_per_month(inputs, tmp_path):
    pairs, rates = inputs
    sent = []
    db = str(tmp_path / "memory.db")
    load = lambda: (pairs, rates)
    first = SR.run_monthly(sent.append, load=load, path=db)
    assert first is not None and len(sent) == 1
    assert SR.run_monthly(sent.append, load=load, path=db) is None
    assert len(sent) == 1
    assert SR.already_reported(SR._month(SR.due_month(SR.D.currency_prices(pairs).index)), db)


def test_change_vs_last_month_is_shown(inputs, tmp_path):
    pairs, rates = inputs
    db = str(tmp_path / "memory.db")
    px = SR.D.currency_prices(pairs)
    ends = SR.D.month_end_days(px.index)
    _, d1 = SR.build_report(pairs, rates, ends[-3])
    SR.log_signals(SR._month(ends[-3]), ends[-3], d1, db)
    prev = SR.previous_weights(SR._month(ends[-2]), db)
    pd.testing.assert_series_equal(prev.sort_index(), d1["weight"].sort_index(), check_names=False)
    text, d2 = SR.build_report(pairs, rates, ends[-2], prev)
    if ((d2["weight"] - d1["weight"]).abs() > 0.05).any():
        assert "(was " in text


def test_tracking_counts_only_returns_after_the_start(inputs, monkeypatch):
    pairs, rates = inputs
    assert "starts with" in SR.tracking_line(pairs, rates)     # synthetic data ends 2023
    monkeypatch.setattr(SR, "TRACKING_START", pd.Timestamp("2022-06-30", tz="UTC"))
    line = SR.tracking_line(pairs, rates)
    assert "Out-of-sample since" in line and "month(s)" in line
