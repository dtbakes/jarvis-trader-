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


# --- losing streak ---

def _trade(day, pl):
    # OANDA-style timestamp with nanoseconds
    return {"close_time": f"2026-10-{day:02d}T14:00:00.123456789Z", "pl": pl}


def test_losing_streak_counts_newest_first():
    trades = [_trade(1, -5), _trade(2, 10), _trade(3, -5), _trade(4, -5)]
    assert killswitch.losing_streak(trades) == 2
    assert killswitch.losing_streak(list(reversed(trades))) == 2


def test_losing_streak_trips_at_limit():
    losers = [_trade(d, -5) for d in range(1, risk.MAX_CONSECUTIVE_LOSSES)]
    assert not killswitch.evaluate_trades(losers, now=T0)["engaged"]
    losers.append(_trade(20, -1))
    ks = killswitch.evaluate_trades(losers, now=T0)
    assert ks["engaged"] and ks["tripped_now"] and "losing trades" in ks["reason"]


def test_release_starts_a_fresh_streak(monkeypatch):
    losers = [_trade(d, -5) for d in range(1, risk.MAX_CONSECUTIVE_LOSSES + 1)]
    assert killswitch.evaluate_trades(losers, now=T0)["engaged"]
    monkeypatch.setattr(killswitch, "_now", lambda: datetime(2026, 10, 25, tzinfo=timezone.utc))
    killswitch.release()
    assert not killswitch.evaluate_trades(losers, now=T0)["engaged"]
    assert killswitch.evaluate_trades(losers + [_trade(26, -5)], now=T0)["losing_streak"] == 1


def test_unreadable_trade_history_fails_closed():
    ks = killswitch.evaluate_trades(None, now=T0)
    assert ks["engaged"] and ks["tripped_now"]


def test_refresh_runs_all_checks_and_reports_read_errors():
    def no_trades():
        raise ConnectionError("down")
    ks = killswitch.refresh(lambda: {"NAV": "1000"}, no_trades, now=T0)
    assert ks["engaged"] and ks["tripped_now"]
    assert ks["errors"] and "closed trades" in ks["errors"][0]


def test_refresh_ok_when_everything_is_fine():
    ks = killswitch.refresh(lambda: {"NAV": "1000"}, lambda: [_trade(1, 5)], now=T0)
    assert not ks["engaged"] and ks["errors"] == []
