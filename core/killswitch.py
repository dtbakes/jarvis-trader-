"""
killswitch.py — latched circuit breaker that protects the account.

Once tripped, NOTHING can open a new position until a human releases it.
The AI has no tool that can release it.

Trips automatically when:
    * today's loss (NAV vs NAV at the 17:00 New York rollover) >= MAX_DAILY_LOSS
    * drawdown from peak NAV >= MAX_DRAWDOWN_PCT
    * MAX_CONSECUTIVE_ERRORS broker errors in a row
    * the account can't be read (fail closed: no NAV, no trading)
    * the state file exists but can't be read (fail closed)

Trips manually when:
    * python -m core.killswitch engage "reason" [--flatten]
    * env var JARVIS_KILL=1 is set (no file write needed)

On an automatic trip, open positions are flattened if FLATTEN_ON_TRIP.

State lives in data/killswitch.json so it survives restarts.

Usage:
    python -m core.killswitch status
    python -m core.killswitch engage "going on holiday" --flatten
    python -m core.killswitch release
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.risk as risk

STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "killswitch.json"
)
NEW_YORK = ZoneInfo("America/New_York")
ROLLOVER_HOUR_NY = 17          # forex trading day starts 17:00 New York


def _now() -> datetime:
    return datetime.now(timezone.utc)


def trading_day(now: datetime) -> str:
    """Forex trading day label: rolls over at 17:00 New York, DST-aware."""
    ny = now.astimezone(NEW_YORK)
    return (ny - timedelta(hours=ROLLOVER_HOUR_NY) + timedelta(days=1)).date().isoformat()


def _default_state() -> dict:
    return {
        "engaged": False,
        "reason": None,
        "source": None,
        "engaged_at": None,
        "day": None,
        "day_start_nav": None,
        "peak_nav": None,
        "consecutive_errors": 0,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def load_state(path: str | None = None) -> dict:
    """Missing file = fresh state. Unreadable file = engaged (fail closed)."""
    path = path or STATE_PATH
    if not os.path.exists(path):
        return _default_state()
    try:
        with open(path) as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state is not an object")
        return {**_default_state(), **state}
    except Exception as e:
        state = _default_state()
        state.update(engaged=True, source="auto",
                     reason=f"Kill switch state unreadable ({e}); failing closed.")
        return state


def save_state(state: dict, path: str | None = None) -> None:
    """Atomic write so a crash mid-write can't leave a half-file."""
    path = path or STATE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Core API
# --------------------------------------------------------------------------- #

def is_engaged(path: str | None = None) -> tuple[bool, str | None]:
    """(engaged, reason). The env override wins over the file."""
    if os.getenv("JARVIS_KILL", "").strip().lower() in ("1", "true", "yes", "on"):
        return True, "JARVIS_KILL environment variable is set."
    state = load_state(path)
    return bool(state["engaged"]), state["reason"]


def engage(reason: str, source: str = "manual", path: str | None = None,
           now: datetime | None = None) -> dict:
    """Trip the switch. Idempotent: an already-engaged switch keeps its first reason."""
    state = load_state(path)
    if not state["engaged"]:
        state.update(engaged=True, reason=reason, source=source,
                     engaged_at=(now or _now()).isoformat())
        save_state(state, path)
    return state


def release(path: str | None = None) -> dict:
    """
    Human-only reset. Also clears the NAV baselines, otherwise the same
    drawdown would re-trip on the very next check.
    """
    state = load_state(path)
    state.update(engaged=False, reason=None, source=None, engaged_at=None,
                 day=None, day_start_nav=None, peak_nav=None, consecutive_errors=0)
    save_state(state, path)
    return state


def evaluate(nav: float | None, now: datetime | None = None,
             path: str | None = None) -> dict:
    """
    Feed the latest account NAV. Updates baselines and trips the switch if a
    limit is breached. Returns {"engaged", "reason", "tripped_now", ...}.
    """
    now = now or _now()
    state = load_state(path)
    was_engaged = state["engaged"]

    if nav is None:
        if not was_engaged:
            state.update(engaged=True, source="auto", engaged_at=now.isoformat(),
                         reason="Account NAV unavailable; failing closed.")
            save_state(state, path)
        return {**state, "tripped_now": not was_engaged}

    nav = float(nav)
    day = trading_day(now)
    if state["day"] != day or state["day_start_nav"] is None:
        state["day"] = day
        state["day_start_nav"] = nav
    if state["peak_nav"] is None or nav > state["peak_nav"]:
        state["peak_nav"] = nav

    daily_pnl = nav - state["day_start_nav"]
    drawdown = (state["peak_nav"] - nav) / state["peak_nav"] if state["peak_nav"] > 0 else 0.0

    reason = None
    if daily_pnl <= -risk.MAX_DAILY_LOSS:
        reason = (f"Daily loss limit hit: {daily_pnl:.2f} "
                  f"(limit -{risk.MAX_DAILY_LOSS:.2f}).")
    elif drawdown >= risk.MAX_DRAWDOWN_PCT:
        reason = (f"Drawdown limit hit: {drawdown:.1%} from peak "
                  f"{state['peak_nav']:.2f} (limit {risk.MAX_DRAWDOWN_PCT:.0%}).")

    if reason and not was_engaged:
        state.update(engaged=True, reason=reason, source="auto",
                     engaged_at=now.isoformat())

    save_state(state, path)
    return {**state, "tripped_now": state["engaged"] and not was_engaged,
            "daily_pnl": daily_pnl, "drawdown": drawdown}


def record_error(error: str, path: str | None = None, now: datetime | None = None) -> dict:
    """Count a broker/execution error. Trips after MAX_CONSECUTIVE_ERRORS in a row."""
    state = load_state(path)
    was_engaged = state["engaged"]
    state["consecutive_errors"] = int(state["consecutive_errors"]) + 1
    if state["consecutive_errors"] >= risk.MAX_CONSECUTIVE_ERRORS and not was_engaged:
        state.update(engaged=True, source="auto", engaged_at=(now or _now()).isoformat(),
                     reason=f"{state['consecutive_errors']} consecutive broker errors; "
                            f"last: {error}")
    save_state(state, path)
    return {**state, "tripped_now": state["engaged"] and not was_engaged}


def record_success(path: str | None = None) -> None:
    state = load_state(path)
    if state["consecutive_errors"]:
        state["consecutive_errors"] = 0
        save_state(state, path)


def flatten_all(get_positions, close_position) -> list[dict]:
    """Close every open position. Keeps going if one close fails."""
    results = []
    for p in get_positions():
        instrument = p["instrument"]
        try:
            results.append({"instrument": instrument, "ok": True,
                            "result": close_position(instrument)})
        except Exception as e:
            results.append({"instrument": instrument, "ok": False, "error": str(e)})
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _flatten_live() -> list[dict]:
    from brokers.oanda import close_position, get_open_positions
    return flatten_all(get_open_positions, close_position)


def main() -> None:
    ap = argparse.ArgumentParser(description="JARVIS kill switch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    e = sub.add_parser("engage")
    e.add_argument("reason", nargs="?", default="Manually engaged.")
    e.add_argument("--flatten", action="store_true", help="also close all open positions")
    sub.add_parser("release")
    args = ap.parse_args()

    if args.cmd == "status":
        engaged, reason = is_engaged()
        print(json.dumps({**load_state(), "effective_engaged": engaged,
                          "effective_reason": reason}, indent=2))
    elif args.cmd == "engage":
        engage(args.reason, source="manual")
        print(f"Kill switch ENGAGED: {args.reason}")
        if args.flatten:
            for r in _flatten_live():
                print(f"  close {r['instrument']}: {'ok' if r['ok'] else r['error']}")
    elif args.cmd == "release":
        release()
        print("Kill switch released. NAV baselines reset on next check.")


if __name__ == "__main__":
    main()
