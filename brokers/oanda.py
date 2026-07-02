import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
# practice or live, based on config
client = oandapyV20.API(
    access_token=config.OANDA_API_KEY,
    environment=config.OANDA_ENVIRONMENT
)

def get_price(instrument):
    """Get current bid/ask for an instrument like EUR_USD."""
    params = {"instruments": instrument}
    r = pricing.PricingInfo(accountID=config.OANDA_ACCOUNT_ID, params=params)
    client.request(r)
    price_data = r.response["prices"][0]

    bid = float(price_data["bids"][0]["price"])
    ask = float(price_data["asks"][0]["price"])
    mid = round((bid + ask) / 2, 5)

    return {
        "instrument": instrument,
        "bid": bid,
        "ask": ask,
        "price": mid
    }

def get_candles(instrument, count=50, granularity="M15"):
    """Get recent candles. granularity: M1, M5, M15, H1, H4, D, etc."""
    params = {"count": count, "granularity": granularity, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)

    candles = []
    for c in r.response["candles"]:
        if c["complete"]:
            candles.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c["volume"]
            })
    return candles

def place_market_order(instrument, units, stop_loss_price=None):
    """Place a market order. units: +buy / -sell. Optionally attach a stop loss."""
    order = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT"
        }
    }
    if stop_loss_price is not None:
        order["order"]["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}

    r = orders.OrderCreate(accountID=config.OANDA_ACCOUNT_ID, data=order)
    client.request(r)
    return r.response

def get_open_positions():
    """Return all currently open positions."""
    r = positions.OpenPositions(accountID=config.OANDA_ACCOUNT_ID)
    client.request(r)
    return r.response.get("positions", [])

def close_position(instrument):
    """Close the position for an instrument — only the side that's actually open."""
    # First, find out what's actually open
    r = positions.OpenPositions(accountID=config.OANDA_ACCOUNT_ID)
    client.request(r)
    open_pos = r.response.get("positions", [])

    target = None
    for p in open_pos:
        if p["instrument"] == instrument:
            target = p
            break

    if target is None:
        return {"closed": False, "reason": f"No open position for {instrument}"}

    # Build close request for only the side(s) with units
    data = {}
    if float(target["long"]["units"]) > 0:
        data["longUnits"] = "ALL"
    if float(target["short"]["units"]) < 0:
        data["shortUnits"] = "ALL"

    r = positions.PositionClose(
        accountID=config.OANDA_ACCOUNT_ID,
        instrument=instrument,
        data=data
    )
    client.request(r)
    return r.response

def get_account_summary():
    """Return balance, unrealized P&L, margin, etc."""
    r = accounts.AccountSummary(accountID=config.OANDA_ACCOUNT_ID)
    client.request(r)
    return r.response.get("account", {})




# Quick test when run directly
if __name__ == "__main__":
    print("Testing OANDA connection...")
    price = get_price("EUR_USD")
    print(f"EUR_USD live price: {price}")

    print("\nFetching last 3 candles (M15)...")
    candles = get_candles("EUR_USD", count=3)
    for c in candles:
        print(f"  {c['time'][:16]} | close: {c['close']}")