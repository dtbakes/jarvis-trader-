import anthropic
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.memory import log_event, get_recent_events, log_trade, get_recent_trades, get_all_facts, set_fact
from brokers.oanda import get_price, get_candles, place_market_order, close_position, get_open_positions, get_account_summary
import core.risk as risk
from core import killswitch

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
MODEL = "claude-haiku-4-5-20251001"

# --- Tools ---
tools = [
    {
        "name": "get_market_price",
        "description": "Get the current price for a currency pair like EUR_USD",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "description": "Currency pair, e.g. EUR_USD"}
            },
            "required": ["instrument"]
        }
    },
    {
        "name": "remember_fact",
        "description": "Store a persistent fact in memory, like a preference or observation",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short label, e.g. 'risk_tolerance'"},
                "value": {"type": "string", "description": "The fact to remember"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "place_trade",
        "description": "Place a market order. units positive=buy, negative=sell. Always set a stop_loss_price to limit downside.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "description": "e.g. EUR_USD"},
                "units": {"type": "integer", "description": "positive=buy, negative=sell. Max 1000."},
                "stop_loss_price": {"type": "number", "description": "Price at which to auto-close to limit loss"},
                "reasoning": {"type": "string", "description": "Why you're making this trade"}
            },
            "required": ["instrument", "units", "stop_loss_price", "reasoning"]
        }
    },
    {
        "name": "close_trade",
        "description": "Close an open position for an instrument entirely",
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "description": "e.g. EUR_USD"},
                "reasoning": {"type": "string", "description": "Why you're closing"}
            },
            "required": ["instrument", "reasoning"]
        }
    },
    {
        "name": "check_account",
        "description": "Get account balance, open positions, and P&L",
        "input_schema": {"type": "object", "properties": {}}
    }
]

TRADING_TOOLS = {"place_trade", "close_trade"}

# --- Tool implementations ---
def get_market_price(instrument):
    data = get_price(instrument)
    log_event("price_check", f"Checked {instrument}: {data['price']}", data)
    return data

def remember_fact(key, value):
    set_fact(key, value)
    log_event("fact_stored", f"Stored fact: {key} = {value}")
    return {"stored": True, "key": key, "value": value}

def place_trade(instrument, units, reasoning, stop_loss_price=None):
    # Fresh NAV check right before the order, so the kill switch sees the
    # latest P&L. If the account can't be read, evaluate() fails closed.
    try:
        nav = float(get_account_summary()["NAV"])
    except Exception as e:
        log_event("killswitch_nav_error", f"Could not read NAV: {e}")
        nav = None
    ks = killswitch.evaluate(nav)
    if ks["tripped_now"]:
        log_event("killswitch_tripped", ks["reason"])

    # RISK CHECK FIRST -- this is the gate. Nothing bypasses it.
    try:
        risk.check_order(instrument, units, stop_loss_price)
    except risk.RiskViolation as e:
        log_event("trade_blocked", f"BLOCKED {instrument} {units}: {e}")
        return {"executed": False, "blocked": True, "reason": str(e)}

    if not risk.is_market_hours():
        log_event("trade_blocked", f"BLOCKED {instrument}: market closed")
        return {"executed": False, "blocked": True, "reason": "Market is closed."}

    # Passed all checks -- execute
    try:
        result = place_market_order(instrument, units, stop_loss_price)
        killswitch.record_success()
        log_trade(f"{'BUY' if units > 0 else 'SELL'} {abs(units)}", instrument, reasoning, "executed")
        log_event("trade_executed", f"{'BUY' if units > 0 else 'SELL'} {abs(units)} {instrument}")
        return {"executed": True, "instrument": instrument, "units": units, "result": result}
    except Exception as e:
        log_event("trade_error", f"ERROR {instrument}: {e}")
        if killswitch.record_error(str(e))["tripped_now"]:
            log_event("killswitch_tripped", "Too many consecutive broker errors")
        return {"executed": False, "error": str(e)}

def close_trade(instrument, reasoning):
    try:
        result = close_position(instrument)
        log_trade("CLOSE", instrument, reasoning, "closed")
        log_event("trade_closed", f"Closed {instrument}")
        return {"closed": True, "instrument": instrument, "result": result}
    except Exception as e:
        return {"closed": False, "error": str(e)}

def check_account():
    summary = get_account_summary()
    pos = get_open_positions()
    return {
        "balance": summary.get("balance"),
        "unrealized_pl": summary.get("unrealizedPL"),
        "open_positions": len(pos),
        "positions": [{"instrument": p["instrument"]} for p in pos]
    }

def run_tool(name, tool_input):
    if name == "get_market_price":
        return get_market_price(**tool_input)
    elif name == "remember_fact":
        return remember_fact(**tool_input)
    elif name == "place_trade":
        return place_trade(**tool_input)
    elif name == "close_trade":
        return close_trade(**tool_input)
    elif name == "check_account":
        return check_account()
    return {"error": f"unknown tool {name}"}

# --- Build context from memory ---
def build_memory_context():
    parts = []

    facts = get_all_facts()
    if facts:
        parts.append("KNOWN FACTS:")
        for k, v in facts.items():
            parts.append(f"  {k}: {v}")

    events = get_recent_events(10)
    if events:
        parts.append("\nRECENT EVENTS:")
        for e in events:
            parts.append(f"  [{e['timestamp']}] {e['event_type']}: {e['summary']}")

    trades = get_recent_trades(5)
    if trades:
        parts.append("\nRECENT TRADES:")
        for t in trades:
            parts.append(f"  [{t['timestamp']}] {t['action']} {t['instrument']} -- {t['reasoning']}")

    return "\n".join(parts) if parts else "No memory yet -- this is the first run."

# --- The agent loop ---
def run_agent(user_message="What's the current price of EUR_USD and GBP_USD?", allow_trading=False):
    """
    allow_trading=False hides place_trade/close_trade from the model entirely,
    so observe-only is enforced in code, not just asked for in the prompt.
    """
    memory_context = build_memory_context()
    run_tools = tools if allow_trading else [t for t in tools if t["name"] not in TRADING_TOOLS]

    system_prompt = (
        "You are JARVIS, an autonomous trading agent. "
        "You have access to tools for checking prices, storing facts, placing and closing trades, and checking the account. "
        "Always use a stop loss when placing a trade. Use your tools when needed. Be concise.\n\n"
        f"YOUR MEMORY:\n{memory_context}"
    )

    print(f"--- JARVIS Agent Run ---")
    print(f"[memory] {memory_context}\n")

    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model=MODEL, max_tokens=1024,
        system=system_prompt, tools=run_tools, messages=messages,
    )

    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool call]   {block.name}({block.input})")
                if block.name in TRADING_TOOLS and not allow_trading:
                    result = {"error": "Trading tools are disabled in this run."}
                else:
                    result = run_tool(block.name, block.input)
                print(f"[tool result] {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model=MODEL, max_tokens=1024,
            system=system_prompt, tools=run_tools, messages=messages,
        )

    for block in response.content:
        if block.type == "text":
            print(f"\n[agent] {block.text}")
            from notifications.telegram import send_message
            send_message(f"🤖 *JARVIS:*\n{block.text}")

    log_event("agent_run", f"Completed run. Query: {user_message}")


if __name__ == "__main__":
    run_agent("Check my account balance and tell me my open positions.")