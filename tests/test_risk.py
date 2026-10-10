"""Tests for the hard risk gate in core/risk.py. These limits are the last line
of defence between the LLM and the broker, so every rule gets a test."""

from datetime import datetime, timezone

import pytest

import core.risk as risk

OK = dict(instrument="EUR_USD", units=100, stop_loss_price=1.05)


def test_valid_order_passes():
    assert risk.check_order(**OK) is True


@pytest.mark.parametrize("pair", ["BTC_USD", "XAU_USD", "eur_usd", ""])
def test_rejects_non_whitelisted_instrument(pair):
    with pytest.raises(risk.RiskViolation, match="not in the allowed list"):
        risk.check_order(pair, 100, stop_loss_price=1.05)


@pytest.mark.parametrize("units", [risk.MAX_POSITION_UNITS, -risk.MAX_POSITION_UNITS])
def test_max_size_is_allowed_both_directions(units):
    assert risk.check_order("EUR_USD", units, stop_loss_price=1.05)


@pytest.mark.parametrize("units", [risk.MAX_POSITION_UNITS + 1, -(risk.MAX_POSITION_UNITS + 1)])
def test_oversize_rejected_both_directions(units):
    with pytest.raises(risk.RiskViolation, match="exceeds max"):
        risk.check_order("EUR_USD", units, stop_loss_price=1.05)


def test_zero_units_rejected():
    with pytest.raises(risk.RiskViolation, match="cannot be zero"):
        risk.check_order("EUR_USD", 0, stop_loss_price=1.05)


def test_daily_loss_limit_is_inclusive():
    with pytest.raises(risk.RiskViolation, match="Daily loss"):
        risk.check_order(**OK, daily_pnl=-risk.MAX_DAILY_LOSS)
    assert risk.check_order(**OK, daily_pnl=-risk.MAX_DAILY_LOSS + 0.01)


def test_master_switch_off_blocks_everything(monkeypatch):
    monkeypatch.setattr(risk, "TRADING_ENABLED", False)
    with pytest.raises(risk.RiskViolation, match="disabled"):
        risk.check_order(**OK)


class _FrozenClock(datetime):
    frozen = None

    @classmethod
    def now(cls, tz=None):
        return cls.frozen


@pytest.mark.parametrize(
    "utc,open_",
    [
        ("2026-10-07 12:00", True),   # Wednesday
        ("2026-10-09 20:59", True),   # Friday before close
        ("2026-10-09 21:00", False),  # Friday after close
        ("2026-10-10 12:00", False),  # Saturday
        ("2026-10-11 21:59", False),  # Sunday before open
        ("2026-10-11 22:00", True),   # Sunday open
    ],
)
def test_is_market_hours(monkeypatch, utc, open_):
    _FrozenClock.frozen = datetime.fromisoformat(utc).replace(tzinfo=timezone.utc)
    monkeypatch.setattr(risk, "datetime", _FrozenClock)
    assert risk.is_market_hours() is open_
