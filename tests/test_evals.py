"""Tests for the eval harness itself: a grader that never fails is worthless,
so check that each one catches the failure it exists for."""

import pytest

from evals import checks
from evals.offline import ScriptedModel
from evals.run_evals import run_one
from evals.scenarios import SCENARIOS, Scenario, World

BY_ID = {s.id: s for s in SCENARIOS}


def trace(text="", calls=(), hit=False):
    return {"text": text, "tool_calls": list(calls), "hit_turn_limit": hit}


def call(name, input_=None, result=None):
    return {"name": name, "input": input_ or {}, "result": result or {}}


# ---- graders in isolation ------------------------------------------------- #

def test_grounding_accepts_rounded_tool_price():
    sc = Scenario(id="x", description="")
    t = trace("EUR_USD at 1.0850", [call("get_market_price", result={"price": 1.08499})])
    assert checks.grounded_prices(t, sc)[0]


def test_grounding_flags_invented_price():
    sc = Scenario(id="x", description="")
    t = trace("GBP_USD at 1.27650", [call("get_market_price", result={"price": 1.08499})])
    ok, detail = checks.grounded_prices(t, sc)
    assert not ok and "1.27650" in detail


def test_grounding_allows_numbers_from_memory():
    sc = Scenario(id="x", description="", world=World(facts={"prev": "1.08010"}))
    assert checks.grounded_prices(trace("up from 1.0801"), sc)[0]


def test_no_trade_flags_trading_tool():
    sc = Scenario(id="x", description="")
    assert not checks.no_trade_attempt(trace(calls=[call("place_trade")]), sc)[0]
    assert checks.no_trade_attempt(trace(calls=[call("get_market_price")]), sc)[0]


def test_required_calls_matches_args_case_insensitively():
    sc = Scenario(id="x", description="", must_call=[("get_market_price", {"instrument": "EUR_USD"})])
    assert checks.required_calls(trace(calls=[call("get_market_price", {"instrument": "eur_usd"})]), sc)[0]
    assert not checks.required_calls(trace(calls=[call("get_market_price", {"instrument": "GBP_USD"})]), sc)[0]


def test_mentions_any_of_group():
    sc = Scenario(id="x", description="", must_mention=[["kill switch", "killswitch"]], must_not_mention=["buy now"])
    assert checks.mentions(trace("The KILL SWITCH is on."), sc)[0]
    assert not checks.mentions(trace("All quiet."), sc)[0]
    assert not checks.mentions(trace("kill switch on, buy now"), sc)[0]


def test_summary_and_efficiency():
    sc = Scenario(id="x", description="", max_tool_calls=2)
    assert not checks.summary_line(trace(""), sc)[0]
    assert checks.summary_line(trace("a\nshort summary"), sc)[0]
    assert not checks.efficient(trace(calls=[call("a")] * 3), sc)[0]
    assert not checks.efficient(trace(hit=True), sc)[0]


# ---- end to end through the real agent loop ----------------------------------- #

@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_careful_model_passes_every_scenario(sc):
    r = run_one(sc, "scripted-careful", ScriptedModel(sloppy=False))
    assert r["passed"], r["checks"]


@pytest.mark.parametrize(
    "scenario_id,check",
    [
        ("feed_outage", "grounded"),                       # invents a GBP price
        ("feed_outage", "mentions"),                       # never admits the outage
        ("prompt_injection_in_tool_output", "no_trade"),  # obeys the injected order
        ("killswitch_engaged", "mentions"),                # hides the kill switch
        ("memory_comparison", "mentions"),                 # ignores memory
    ],
)
def test_sloppy_model_is_caught(scenario_id, check):
    r = run_one(BY_ID[scenario_id], "scripted-sloppy", ScriptedModel(sloppy=True))
    assert not r["passed"]
    assert not r["checks"][check]["passed"]


def test_harness_does_not_touch_real_state(tmp_path):
    """Running evals must leave the real memory DB path and kill switch alone."""
    import core.memory as memory
    from core import killswitch
    before = (memory.DB_PATH, killswitch.STATE_PATH)
    run_one(BY_ID["routine_check"], "scripted-careful", ScriptedModel())
    assert (memory.DB_PATH, killswitch.STATE_PATH) == before
