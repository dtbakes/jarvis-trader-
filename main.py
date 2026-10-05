"""
JARVIS — autonomous agent entry point.
Runs the agent on a schedule during forex market hours.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from core.agent import run_agent
from core.memory import log_event
from notifications.telegram import send_message
from brokers.oanda import get_account_summary, get_open_positions, close_position
from core import killswitch
import core.risk as risk

CHECK_INTERVAL_MINUTES = 30
FLATTEN_ON_TRIP = True   # close all open positions when the kill switch trips automatically

def check_killswitch():
    """Feed the latest NAV to the kill switch; flatten and alert if it just tripped."""
    try:
        nav = float(get_account_summary()["NAV"])
    except Exception as e:
        log_event("killswitch_nav_error", f"Could not read NAV: {e}")
        nav = None

    ks = killswitch.evaluate(nav)
    if not ks["tripped_now"]:
        return ks

    log_event("killswitch_tripped", ks["reason"])
    msg = f"🛑 KILL SWITCH TRIPPED: {ks['reason']}\nNo new trades until released."
    if FLATTEN_ON_TRIP:
        results = killswitch.flatten_all(get_open_positions, close_position)
        log_event("killswitch_flatten", f"Flattened {len(results)} positions", results)
        failed = [r["instrument"] for r in results if not r["ok"]]
        msg += f"\nClosed {len(results) - len(failed)} position(s)."
        if failed:
            msg += f" FAILED to close: {', '.join(failed)}. Check OANDA now."
    msg += "\nRelease with: python -m core.killswitch release"
    send_message(msg, markdown=False)
    return ks

def market_check():
    """One scheduled cycle: observe market, log, notify if notable."""
    if not risk.is_market_hours():
        print("[scheduler] Market closed. Skipping cycle.")
        return

    ks = check_killswitch()

    print("[scheduler] Running market check...")
    try:
        run_agent(
            "Do a routine market check. Look at EUR_USD and GBP_USD prices, "
            "compare to recent history in your memory, and note anything interesting. "
            "Do NOT place any trades — you are in observe-only mode. "
            "End with a one-line summary."
            + (f" Note: the kill switch is ENGAGED ({ks['reason']})." if ks["engaged"] else ""),
            allow_trading=False,
        )
        log_event("scheduled_run", "Completed scheduled market check")
    except Exception as e:
        log_event("scheduler_error", f"Cycle failed: {e}")
        send_message(f"⚠️ *JARVIS error during scheduled run:*\n`{e}`")

if __name__ == "__main__":
    print(f"JARVIS starting. Checking markets every {CHECK_INTERVAL_MINUTES} min during market hours.")
    send_message(f"🤖 *JARVIS autopilot started.* Observing markets every {CHECK_INTERVAL_MINUTES} min. Observe-only mode — no trades.")
    engaged, reason = killswitch.is_engaged()
    if engaged:
        send_message(f"🛑 Kill switch is ENGAGED: {reason}", markdown=False)

    scheduler = BlockingScheduler()
    scheduler.add_job(market_check, "interval", minutes=CHECK_INTERVAL_MINUTES)

    # Run one check immediately on startup
    market_check()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        send_message("🛑 *JARVIS autopilot stopped.*")
        print("\nJARVIS stopped.")