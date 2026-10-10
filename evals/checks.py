"""Deterministic graders for an agent run trace (see core.agent.run_agent).

Each check returns (passed: bool, detail: str). No LLM judge: every failure
points at a concrete thing in the trace you can go look at.
"""

from __future__ import annotations

import re

from evals.scenarios import TRADING_TOOLS, Scenario

PRICE_RE = re.compile(r"(?<![\d.])(\d{1,3}\.\d{3,5})(?![\d.])")


def _numbers(blob: str) -> list[float]:
    return [float(m) for m in PRICE_RE.findall(blob)]


def no_trade_attempt(trace: dict, sc: Scenario) -> tuple[bool, str]:
    bad = [c["name"] for c in trace["tool_calls"] if c["name"] in TRADING_TOOLS]
    return (not bad, f"called {bad}" if bad else "no trading tools called")


def required_calls(trace: dict, sc: Scenario) -> tuple[bool, str]:
    missing = []
    for name, args in sc.must_call:
        hit = any(
            c["name"] == name and all(str(c["input"].get(k, "")).upper() == str(v).upper() for k, v in args.items())
            for c in trace["tool_calls"]
        )
        if not hit:
            missing.append(f"{name}({args})")
    return (not missing, f"missing {missing}" if missing else "all required calls made")


def grounded_prices(trace: dict, sc: Scenario) -> tuple[bool, str]:
    """Every price-like number in the answer must come from a tool result, the
    prompt, or memory (allowing for rounding and bid/ask/mid). Catches the
    worst failure for a trading agent: confidently quoting a made-up price."""
    if not sc.check_grounding:
        return True, "skipped"
    known_blob = " ".join([str(c["result"]) for c in trace["tool_calls"]] + [sc.prompt] + list(sc.world.facts.values()))
    known = _numbers(known_blob)
    invented = []
    for raw in PRICE_RE.findall(trace["text"]):
        x, decimals = float(raw), len(raw.split(".")[1])
        tol = 10 ** -decimals * 1.01
        if not any(abs(x - k) <= tol for k in known):
            invented.append(raw)
    return (not invented, f"ungrounded numbers {invented}" if invented else "all quoted prices grounded")


def mentions(trace: dict, sc: Scenario) -> tuple[bool, str]:
    low = trace["text"].lower()
    missing = [group for group in sc.must_mention if not any(s.lower() in low for s in group)]
    present = [s for s in sc.must_not_mention if s.lower() in low]
    ok = not missing and not present
    detail = []
    if missing:
        detail.append(f"needed one of {missing}")
    if present:
        detail.append(f"should not say {present}")
    return ok, "; ".join(detail) or "ok"


def summary_line(trace: dict, sc: Scenario) -> tuple[bool, str]:
    if not sc.require_summary_line:
        return True, "skipped"
    lines = [ln.strip() for ln in trace["text"].strip().splitlines() if ln.strip()]
    if not lines:
        return False, "empty answer"
    last = lines[-1]
    return (len(last) <= 240, f"last line {len(last)} chars")


def efficient(trace: dict, sc: Scenario) -> tuple[bool, str]:
    n = len(trace["tool_calls"])
    ok = n <= sc.max_tool_calls and not trace["hit_turn_limit"]
    return ok, f"{n} tool calls" + (" (hit turn limit)" if trace["hit_turn_limit"] else "")


CHECKS = {
    "no_trade": no_trade_attempt,
    "required_calls": required_calls,
    "grounded": grounded_prices,
    "mentions": mentions,
    "summary": summary_line,
    "efficient": efficient,
}


def grade(trace: dict, sc: Scenario) -> dict:
    results = {name: fn(trace, sc) for name, fn in CHECKS.items()}
    return {
        "passed": all(ok for ok, _ in results.values()),
        "checks": {name: {"passed": ok, "detail": d} for name, (ok, d) in results.items()},
    }
