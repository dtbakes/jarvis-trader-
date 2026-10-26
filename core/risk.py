from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# --- Hard limits. The AI CANNOT override these. ---
MAX_POSITION_UNITS = 1000          # max size per trade (1k units = micro lot)
MAX_INSTRUMENT_UNITS = 2000        # max net position per instrument, so stacked orders can't add up past it
MAX_TOTAL_UNITS = 3000             # max sum of |net position| across all instruments
MAX_DAILY_LOSS = 50.0              # kill switch trips if down this much today (account currency)
MAX_DRAWDOWN_PCT = 0.10            # kill switch trips at this drawdown from peak NAV
MAX_CONSECUTIVE_ERRORS = 3         # kill switch trips after this many broker errors in a row
MAX_CONSECUTIVE_LOSSES = 4         # kill switch trips after this many losing trades in a row
FRIDAY_ENTRY_CUTOFF_HOUR_NY = 12   # no new entries after Friday noon New York (no weekend gap risk)
ALLOWED_INSTRUMENTS = {            # whitelist — nothing else can be traded
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"
}
TRADING_ENABLED = True             # hard-coded master switch (the runtime one is core/killswitch.py)

class RiskViolation(Exception):
    """Raised when a proposed trade breaks a rule."""
    pass

def check_order(instrument, units, stop_loss_price=None, daily_pnl=0.0, open_units=None):
    """
    Validates a proposed order against all risk rules.
    Returns True if safe. Raises RiskViolation if not.

    open_units: {instrument: net units currently open}. Used for the
    per-instrument and total exposure caps.
    """
    # Master kill switch
    if not TRADING_ENABLED:
        raise RiskViolation("Trading is disabled (kill switch is OFF).")

    # Runtime kill switch (latched; only a human can release it)
    from core import killswitch
    engaged, reason = killswitch.is_engaged()
    if engaged:
        raise RiskViolation(f"Kill switch engaged: {reason}")

    # Instrument whitelist
    if instrument not in ALLOWED_INSTRUMENTS:
        raise RiskViolation(
            f"{instrument} is not in the allowed list: {ALLOWED_INSTRUMENTS}"
        )

    # Position size limit
    if abs(units) > MAX_POSITION_UNITS:
        raise RiskViolation(
            f"Order size {abs(units)} exceeds max {MAX_POSITION_UNITS} units."
        )

    # Units can't be zero
    if units == 0:
        raise RiskViolation("Order units cannot be zero.")

    # Every order carries a stop
    if stop_loss_price is None:
        raise RiskViolation("Every order needs a stop_loss_price.")

    # Exposure caps: one order can be small while five stacked ones are not
    open_units = open_units or {}
    current = open_units.get(instrument, 0)
    after = current + units
    if abs(after) > MAX_INSTRUMENT_UNITS:
        raise RiskViolation(
            f"{instrument} position would be {after} units; "
            f"max {MAX_INSTRUMENT_UNITS} per instrument."
        )
    total_after = sum(abs(u) for i, u in open_units.items() if i != instrument) + abs(after)
    if total_after > MAX_TOTAL_UNITS:
        raise RiskViolation(
            f"Total exposure would be {total_after} units; max {MAX_TOTAL_UNITS}."
        )

    # Daily loss limit
    if daily_pnl <= -MAX_DAILY_LOSS:
        raise RiskViolation(
            f"Daily loss limit hit ({daily_pnl}). Trading halted for safety."
        )

    return True

def is_market_hours(now=None):
    """Forex runs Sun 5pm ET to Fri 5pm ET. Rough UTC check."""
    now = now or datetime.now(timezone.utc)
    weekday = now.weekday()  # Mon=0 ... Sun=6
    # Closed: Saturday, and Friday after ~21:00 UTC, and before Sunday ~22:00 UTC
    if weekday == 5:  # Saturday
        return False
    if weekday == 4 and now.hour >= 21:  # Friday evening
        return False
    if weekday == 6 and now.hour < 22:  # Sunday before open
        return False
    return True


def is_entry_window(now=None):
    """
    New positions are allowed only during market hours and before the
    Friday cutoff, so nothing new is left exposed to the weekend gap.
    Closing positions is never restricted by this.
    """
    now = now or datetime.now(timezone.utc)
    ny = now.astimezone(ZoneInfo("America/New_York"))
    if ny.weekday() == 4 and ny.hour >= FRIDAY_ENTRY_CUTOFF_HOUR_NY:
        return False
    return is_market_hours(now)
