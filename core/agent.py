import anthropic
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.memory import log_event, get_recent_events, log_trade, get_recent_trades, get_all_facts, set_fact
from brokers.oanda import get_price, get_candles, place_market_order, close_position, get_open_positions, get_account_summary, get_closed_trades
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
    # Fresh kill switch checks right before the order (NAV, drawdown, losing
    # streak). Any broker read failure fails closed.
    ks = killswitch.refresh(get_account_summary, get_closed_trades)
    for err in ks["errors"]:
        log_event("killswitch_read_error", err)
    if ks["tripped_now"]:
        log_event("killswitch_tripped", ks["reason"])

    # Current net exposure per instrument, for the stacking caps.
    try:
        open_units = {
            p["instrument"]: int(float(p["long"]["units"]) + float(p["short"]["units"]))
            for p in get_open_positions()
        }
    except Exception as e:
        log_event("trade_blocked", f"BLOCKED {instrument}: can't read open positions: {e}")
        return {"executed": False, "blocked": True,
                "reason": f"Could not read open positions: {e}"}

    # RISK CHECK FIRST -- this is the gate. Nothing bypasses it.
    try:
        risk.check_order(instrument, units, stop_loss_price, open_units=open_units)
    except risk.RiskViolation as e:
        log_event("trade_blocked", f"BLOCKED {instrument} {units}: {e}")
        return {"executed": False, "blocked": True, "reason": str(e)}

    if not risk.is_entry_window():
        log_event("trade_blocked", f"BLOCKED {instrument}: outside entry window")
        return {"executed": False, "blocked": True,
                "reason": "Outside the entry window (market closed or past the Friday cutoff)."}

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
MAX_TURNS = 10  # hard cap on model round-trips per run, so a tool loop can't spin forever


def run_agent(user_message="What's the current price of EUR_USD and GBP_USD?", allow_trading=False,
              model=None, notify=True, max_turns=MAX_TURNS):
    """
    allow_trading=False hides place_trade/close_trade from the model entirely,
    so observe-only is enforced in code, not just asked for in the prompt.

    model / notify / max_turns exist so tests and evals can run the same loop
    against a different model without sending Telegram alerts.

    Returns a trace dict: final text, every tool call (name, input, result),
    token usage and turn count. main.py ignores it; the eval harness scores it.
    """
    model = model or MODEL
    trace = {"model": model, "text": "", "tool_calls": [], "turns": 0,
             "input_tokens": 0, "output_tokens": 0, "hit_turn_limit": False}
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

    def call_model():
        r = client.messages.create(
            model=model, max_tokens=1024,
            system=system_prompt, tools=run_tools, messages=messages,
        )
        trace["turns"] += 1
        usage = getattr(r, "usage", None)
        if usage is not None:
            trace["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            trace["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        return r

    response = call_model()

    while response.stop_reason == "tool_use":
        if trace["turns"] >= max_turns:
            trace["hit_turn_limit"] = True
            log_event("agent_turn_limit", f"Stopped after {max_turns} turns. Query: {user_message}")
            break
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool call]   {block.name}({block.input})")
                is_error = False
                if block.name in TRADING_TOOLS and not allow_trading:
                    result = {"error": "Trading tools are disabled in this run."}
                    is_error = True
                else:
                    # A failing tool (network blip, bad args from the model)
                    # becomes an error the model can see, not a crashed run.
                    try:
                        result = run_tool(block.name, block.input)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                        is_error = True
                        log_event("tool_error", f"{block.name} failed: {e}")
                print(f"[tool result] {result}")
                trace["tool_calls"].append({"name": block.name, "input": dict(block.input), "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                    **({"is_error": True} if is_error else {}),
                })

        messages.append({"role": "user", "content": tool_results})
        response = call_model()

    texts = [block.text for block in response.content if block.type == "text"]
    trace["text"] = "\n".join(texts)
    for text in texts:
        print(f"\n[agent] {text}")
        if notify:
            from notifications.telegram import send_message
            send_message(f"🤖 *JARVIS:*\n{text}")

    log_event("agent_run", f"Completed run. Query: {user_message}")
    return trace


if __name__ == "__main__":
    run_agent("Check my account balance and tell me my open positions.")