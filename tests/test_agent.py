"""Integration tests for the agent loop in core/agent.py.

Claude, OANDA and Telegram are all replaced with fakes, so these run offline
in CI. They check the safety properties that matter most for an LLM that can
touch a brokerage account: observe-only is enforced in code, the risk gate
cannot be bypassed, and tool failures don't crash a scheduled run."""

import pytest

import core.agent as agent
import core.memory as memory
from fakes import ScriptedClient, response, text, tool_use


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """Isolated memory DB, fake broker, no Telegram, entry window open."""
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "memory.db"))
    memory.init_db()

    calls = {"orders": [], "closes": [], "telegram": []}
    monkeypatch.setattr(agent, "get_price", lambda i: {"instrument": i, "price": 1.0850, "bid": 1.0849, "ask": 1.0851})
    monkeypatch.setattr(agent, "get_account_summary", lambda: {"NAV": "1000.0", "balance": "1000.0", "unrealizedPL": "0.0"})
    monkeypatch.setattr(agent, "get_open_positions", lambda: [])
    monkeypatch.setattr(agent, "place_market_order", lambda *a, **k: calls["orders"].append((a, k)) or {"id": "1"})
    monkeypatch.setattr(agent, "close_position", lambda i: calls["closes"].append(i) or {})
    import notifications.telegram as tg
    monkeypatch.setattr(tg, "send_message", lambda msg, **k: calls["telegram"].append(msg) or True)
    monkeypatch.setattr(agent, "get_closed_trades", lambda: [])
    monkeypatch.setattr(agent.risk, "is_entry_window", lambda now=None: True)
    return calls


def use(monkeypatch, *responses):
    client = ScriptedClient(*responses)
    monkeypatch.setattr(agent, "client", client)
    return client


# ---- observe-only ------------------------------------------------------------ #

def test_observe_only_hides_trading_tools(monkeypatch):
    client = use(monkeypatch, response(text("EUR_USD is quiet.")))
    agent.run_agent("check", allow_trading=False, notify=False)
    assert client.offered_tools() == {"get_market_price", "remember_fact", "check_account"}


def test_trading_mode_offers_trading_tools(monkeypatch):
    client = use(monkeypatch, response(text("ok")))
    agent.run_agent("check", allow_trading=True, notify=False)
    assert {"place_trade", "close_trade"} <= client.offered_tools()


def test_observe_only_refuses_hallucinated_trade_call(monkeypatch, offline):
    """Even if the model calls a tool it was never offered, nothing executes."""
    use(
        monkeypatch,
        response(tool_use("place_trade", {"instrument": "EUR_USD", "units": 100,
                                          "stop_loss_price": 1.08, "reasoning": "yolo"})),
        response(text("Could not trade.")),
    )
    trace = agent.run_agent("check", allow_trading=False, notify=False)
    assert offline["orders"] == []
    assert trace["tool_calls"][0]["result"] == {"error": "Trading tools are disabled in this run."}


# ---- risk gate ------------------------------------------------------------- #

@pytest.mark.parametrize(
    "order,why",
    [
        ({"instrument": "BTC_USD", "units": 100, "stop_loss_price": 1.0}, "allowed list"),
        ({"instrument": "EUR_USD", "units": 50_000, "stop_loss_price": 1.08}, "exceeds max"),
    ],
)
def test_risk_gate_blocks_bad_orders_from_the_model(monkeypatch, offline, order, why):
    use(
        monkeypatch,
        response(tool_use("place_trade", {**order, "reasoning": "test"})),
        response(text("blocked")),
    )
    trace = agent.run_agent("trade", allow_trading=True, notify=False)
    assert offline["orders"] == []
    result = trace["tool_calls"][0]["result"]
    assert result["blocked"] and why in result["reason"]


def test_valid_order_reaches_broker(monkeypatch, offline):
    use(
        monkeypatch,
        response(tool_use("place_trade", {"instrument": "EUR_USD", "units": 100,
                                          "stop_loss_price": 1.08, "reasoning": "test"})),
        response(text("done")),
    )
    trace = agent.run_agent("trade", allow_trading=True, notify=False)
    assert len(offline["orders"]) == 1
    assert trace["tool_calls"][0]["result"]["executed"] is True


def test_kill_switch_blocks_orders(monkeypatch, offline):
    agent.killswitch.engage("test")
    use(
        monkeypatch,
        response(tool_use("place_trade", {"instrument": "EUR_USD", "units": 100,
                                          "stop_loss_price": 1.08, "reasoning": "test"})),
        response(text("blocked")),
    )
    trace = agent.run_agent("trade", allow_trading=True, notify=False)
    assert offline["orders"] == []
    assert "Kill switch" in trace["tool_calls"][0]["result"]["reason"]


def test_unreadable_nav_fails_closed(monkeypatch, offline):
    def boom():
        raise ConnectionError("OANDA down")
    monkeypatch.setattr(agent, "get_account_summary", boom)
    use(
        monkeypatch,
        response(tool_use("place_trade", {"instrument": "EUR_USD", "units": 100,
                                          "stop_loss_price": 1.08, "reasoning": "test"})),
        response(text("blocked")),
    )
    agent.run_agent("trade", allow_trading=True, notify=False)
    assert offline["orders"] == []
    assert agent.killswitch.is_engaged()[0]


# ---- robustness ------------------------------------------------------------ #

def test_tool_exception_is_returned_to_model_not_raised(monkeypatch):
    def boom(_):
        raise TimeoutError("pricing timed out")
    monkeypatch.setattr(agent, "get_price", boom)
    client = use(
        monkeypatch,
        response(tool_use("get_market_price", {"instrument": "EUR_USD"})),
        response(text("Price feed is down.")),
    )
    trace = agent.run_agent("check", notify=False)
    assert trace["text"] == "Price feed is down."
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "TimeoutError" in sent["content"]


def test_bad_tool_arguments_do_not_crash(monkeypatch):
    use(
        monkeypatch,
        response(tool_use("get_market_price", {"pair": "EUR_USD"})),  # wrong arg name
        response(text("retrying later")),
    )
    trace = agent.run_agent("check", notify=False)
    assert "TypeError" in trace["tool_calls"][0]["result"]["error"]


def test_turn_limit_stops_runaway_loop(monkeypatch):
    looping = [response(tool_use("get_market_price", {"instrument": "EUR_USD"})) for _ in range(5)]
    use(monkeypatch, *looping)
    trace = agent.run_agent("check", notify=False, max_turns=3)
    assert trace["hit_turn_limit"] and trace["turns"] == 3


# ---- plumbing -------------------------------------------------------------- #

def test_trace_records_usage_and_calls(monkeypatch):
    use(
        monkeypatch,
        response(tool_use("get_market_price", {"instrument": "EUR_USD"}), in_tok=200, out_tok=30),
        response(text("EUR_USD at 1.0850"), in_tok=300, out_tok=40),
    )
    trace = agent.run_agent("check", notify=False, model="claude-test")
    assert trace["model"] == "claude-test"
    assert (trace["input_tokens"], trace["output_tokens"], trace["turns"]) == (500, 70, 2)
    assert trace["tool_calls"][0]["name"] == "get_market_price"
    assert trace["text"] == "EUR_USD at 1.0850"


def test_notify_sends_final_text_to_telegram(monkeypatch, offline):
    use(monkeypatch, response(text("summary line")))
    agent.run_agent("check", notify=True)
    assert offline["telegram"] and "summary line" in offline["telegram"][0]


def test_notify_false_sends_nothing(monkeypatch, offline):
    use(monkeypatch, response(text("summary line")))
    agent.run_agent("check", notify=False)
    assert offline["telegram"] == []


def test_memory_context_reaches_system_prompt(monkeypatch):
    memory.set_fact("risk_tolerance", "low")
    client = use(monkeypatch, response(text("ok")))
    agent.run_agent("check", notify=False)
    assert "risk_tolerance: low" in client.requests[0]["system"]


# ---- stacking, losing streak, Friday cutoff --------------------------------- #

def _order(units=100):
    return response(tool_use("place_trade", {"instrument": "EUR_USD", "units": units,
                                             "stop_loss_price": 1.08, "reasoning": "test"}))


def test_stacked_orders_hit_the_instrument_cap(monkeypatch, offline):
    held = agent.risk.MAX_INSTRUMENT_UNITS - 500
    monkeypatch.setattr(agent, "get_open_positions", lambda: [
        {"instrument": "EUR_USD", "long": {"units": str(held)}, "short": {"units": "0"}}])
    use(monkeypatch, _order(agent.risk.MAX_POSITION_UNITS), response(text("blocked")))
    trace = agent.run_agent("trade", allow_trading=True, notify=False)
    assert offline["orders"] == []
    assert "per instrument" in trace["tool_calls"][0]["result"]["reason"]


def test_losing_streak_blocks_orders(monkeypatch, offline):
    losers = [{"close_time": f"2026-10-{d:02d}T10:00:00Z", "pl": -3.0}
              for d in range(1, agent.risk.MAX_CONSECUTIVE_LOSSES + 1)]
    monkeypatch.setattr(agent, "get_closed_trades", lambda: losers)
    use(monkeypatch, _order(), response(text("blocked")))
    trace = agent.run_agent("trade", allow_trading=True, notify=False)
    assert offline["orders"] == []
    assert "losing trades" in trace["tool_calls"][0]["result"]["reason"]


def test_entry_window_closed_blocks_orders(monkeypatch, offline):
    monkeypatch.setattr(agent.risk, "is_entry_window", lambda now=None: False)
    use(monkeypatch, _order(), response(text("blocked")))
    trace = agent.run_agent("trade", allow_trading=True, notify=False)
    assert offline["orders"] == []
    assert "entry window" in trace["tool_calls"][0]["result"]["reason"]
