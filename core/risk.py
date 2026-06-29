from datetime import datetime, timezone

# --- Hard limits. The AI CANNOT override these. ---
MAX_POSITION_UNITS = 1000          # max size per trade (1k units = micro lot)
MAX_DAILY_LOSS = 50.0              # stop trading if down this much (account currency)
ALLOWED_INSTRUMENTS = {            # whitelist — nothing else can be traded
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"
}
TRADING_ENABLED = True             # master kill switch

class RiskViolation(Exception):
    """Raised when a proposed trade breaks a rule."""
    pass

def check_order(instrument, units, daily_pnl=0.0):
    """
    Validates a proposed order against all risk rules.
    Returns True if safe. Raises RiskViolation if not.
    """
    # Master kill switch
    if not TRADING_ENABLED:
        raise RiskViolation("Trading is disabled (kill switch is OFF).")

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

    # Daily loss limit
    if daily_pnl <= -MAX_DAILY_LOSS:
        raise RiskViolation(
            f"Daily loss limit hit ({daily_pnl}). Trading halted for safety."
        )

    return True

def is_market_hours():
    """Forex runs Sun 5pm ET to Fri 5pm ET. Rough UTC check."""
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # Mon=0 ... Sun=6
    # Closed: Saturday, and Friday after ~21:00 UTC, and before Sunday ~22:00 UTC
    if weekday == 5:  # Saturday
        return False
    if weekday == 4 and now.hour >= 21:  # Friday evening
        return False
    if weekday == 6 and now.hour < 22:  # Sunday before open
        return False
    return True