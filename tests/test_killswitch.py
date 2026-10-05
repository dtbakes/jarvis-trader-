from datetime import datetime, timezone

import pytest

import core.risk as risk
from core import killswitch

# 14:00 UTC = 10:00 New York (EDT), well inside one trading day
T0 = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)


def test_fresh_state_is_not_engaged():
    assert killswitch.is_engaged() == (False, None)


def test_daily_loss_trips_and_latches():
    killswitch.evaluate(1000.0, now=T0)
    ks = killswitch.evaluate(1000.0 - risk.MAX_DAILY_LOSS, now=T0)
    assert ks["engaged"] and ks["tripped_now"]
    assert "Daily loss" in ks["reason"]

    # Recovery does not un-trip it; only a human release does.
    ks = killswitch.evaluate(2000.0, now=T0)
    assert ks["engaged"] and not ks["tripped_now"]


def test_small_loss_does_not_trip():
    killswitch.evaluate(1000.0, now=T0)
    assert not killswitch.evaluate(1000.0 - risk.MAX_DAILY_LOSS + 1, now=T0)["engaged"]


def test_daily_baseline_resets_at_ny_rollover():
    killswitch.evaluate(1000.0, now=T0)
    killswitch.evaluate(970.0, now=T0)                       # -30 today
    next_day = datetime(2026, 10, 6, 21, 30, tzinfo=timezone.utc)  # 17:30 NY
    ks = killswitch.evaluate(940.0, now=next_day)            # new day starts at 940
    assert not ks["engaged"]
    assert ks["day_start_nav"] == 940.0


def test_drawdown_from_peak_trips_across_days(monkeypatch):
    monkeypatch.setattr(risk, "MAX_DAILY_LOSS", 10_000.0)
    killswitch.evaluate(1000.0, now=T0)
    later = datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)
    ks = killswitch.evaluate(1000.0 * (1 - risk.MAX_DRAWDOWN_PCT), now=later)
    assert ks["engaged"] and "Drawdown" in ks["reason"]


def test_missing_nav_fails_closed():
    ks = killswitch.evaluate(None, now=T0)
    assert ks["engaged"] and ks["tripped_now"]


def test_corrupt_state_fails_closed(isolated_state):
    with open(isolated_state, "w") as f:
        f.write("{not json")
    engaged, reason = killswitch.is_engaged()
    assert engaged and "unreadable" in reason


def test_consecutive_errors_trip_and_success_resets():
    for _ in range(risk.MAX_CONSECUTIVE_ERRORS - 1):
        assert not killswitch.record_error("timeout")["engaged"]
    killswitch.record_success()
    for _ in range(risk.MAX_CONSECUTIVE_ERRORS - 1):
        killswitch.record_error("timeout")
    ks = killswitch.record_error("timeout")
    assert ks["engaged"] and ks["tripped_now"]


def test_release_clears_baselines_so_it_does_not_retrip():
    killswitch.evaluate(1000.0, now=T0)
    killswitch.evaluate(900.0, now=T0)
    killswitch.release()
    assert killswitch.is_engaged() == (False, None)
    assert not killswitch.evaluate(900.0, now=T0)["engaged"]


def test_engage_keeps_first_reason():
    killswitch.engage("first")
    killswitch.engage("second")
    assert killswitch.is_engaged() == (True, "first")


def test_env_override(monkeypatch):
    monkeypatch.setenv("JARVIS_KILL", "1")
    assert killswitch.is_engaged()[0]


def test_flatten_all_continues_past_failures():
    closed = []

    def close(instrument):
        if instrument == "GBP_USD":
            raise RuntimeError("rejected")
        closed.append(instrument)
        return {}

    positions = [{"instrument": "EUR_USD"}, {"instrument": "GBP_USD"}, {"instrument": "USD_JPY"}]
    results = killswitch.flatten_all(lambda: positions, close)
    assert closed == ["EUR_USD", "USD_JPY"]
    assert [r["ok"] for r in results] == [True, False, True]


def test_trading_day_label_rolls_at_5pm_new_york():
    before = datetime(2026, 10, 6, 20, 59, tzinfo=timezone.utc)   # 16:59 EDT
    after = datetime(2026, 10, 6, 21, 0, tzinfo=timezone.utc)     # 17:00 EDT
    assert killswitch.trading_day(before) != killswitch.trading_day(after)


# --- risk.check_order integration ---

def test_check_order_blocks_when_engaged():
    risk.check_order("EUR_USD", 100, stop_loss_price=1.05)
    killswitch.engage("test")
    with pytest.raises(risk.RiskViolation, match="Kill switch"):
        risk.check_order("EUR_USD", 100, stop_loss_price=1.05)


def test_check_order_requires_stop():
    with pytest.raises(risk.RiskViolation, match="stop_loss_price"):
        risk.check_order("EUR_USD", 100)
