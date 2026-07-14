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
import core.risk as risk

CHECK_INTERVAL_MINUTES = 30

def market_check():
    """One scheduled cycle: observe market, log, notify if notable."""
    if not risk.is_market_hours():
        print("[scheduler] Market closed. Skipping cycle.")
        return

    print("[scheduler] Running market check...")
    try:
        run_agent(
            "Do a routine market check. Look at EUR_USD and GBP_USD prices, "
            "compare to recent history in your memory, and note anything interesting. "
            "Do NOT place any trades — you are in observe-only mode. "
            "End with a one-line summary."
        )
        log_event("scheduled_run", "Completed scheduled market check")
    except Exception as e:
        log_event("scheduler_error", f"Cycle failed: {e}")
        send_message(f"⚠️ *JARVIS error during scheduled run:*\n`{e}`")

if __name__ == "__main__":
    print(f"JARVIS starting. Checking markets every {CHECK_INTERVAL_MINUTES} min during market hours.")
    send_message(f"🤖 *JARVIS autopilot started.* Observing markets every {CHECK_INTERVAL_MINUTES} min. Observe-only mode — no trades.")

    scheduler = BlockingScheduler()
    scheduler.add_job(market_check, "interval", minutes=CHECK_INTERVAL_MINUTES)

    # Run one check immediately on startup
    market_check()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        send_message("🛑 *JARVIS autopilot stopped.*")
        print("\nJARVIS stopped.")