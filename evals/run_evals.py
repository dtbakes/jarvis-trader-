"""Run the JARVIS agent eval suite against one or more Claude models.

The real agent loop (core.agent.run_agent) runs unchanged; only the outside
world is swapped out: OANDA returns scenario prices, Telegram is muted, and
memory / kill-switch state live in a temp dir. Your real account and DB are
never touched, and place_market_order is replaced with a recorder.

Usage
-----
    # compare the current production model against newer releases
    python -m evals.run_evals --models claude-haiku-4-5-20251001 claude-haiku-5-5 --trials 3

    # offline self-check of the harness (no API key; what CI runs)
    python -m evals.run_evals --offline

Writes evals/results/<timestamp>.json and .md and prints the markdown table.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evals.checks import CHECKS, grade  # noqa: E402
from evals.scenarios import SCENARIOS, Scenario  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evals", "results")


@contextlib.contextmanager
def scenario_world(sc: Scenario, client):
    """Patch the agent's dependencies to the scenario's world, then restore."""
    import core.agent as agent
    import core.memory as memory
    import notifications.telegram as tg
    from core import killswitch

    w = sc.world
    orders = []

    def get_price(instrument):
        if instrument in w.price_errors:
            raise TimeoutError(f"pricing request for {instrument} timed out")
        if instrument not in w.prices:
            raise ValueError(f"unknown instrument {instrument}")
        p = w.prices[instrument]
        out = {"instrument": instrument, "bid": p["bid"], "ask": p["ask"],
               "price": round((p["bid"] + p["ask"]) / 2, 5)}
        if w.price_note:
            out["note"] = w.price_note
        return out

    patches = {
        (agent, "client"): client,
        (agent, "get_price"): get_price,
        (agent, "get_account_summary"): lambda: {"NAV": str(w.nav), "balance": f"{w.nav:.2f}",
                                                 "unrealizedPL": str(w.unrealized_pl)},
        (agent, "get_open_positions"): lambda: list(w.positions),
        (agent, "place_market_order"): lambda *a, **k: orders.append((a, k)) or {"id": "eval"},
        (agent, "close_position"): lambda i: {},
        (agent, "get_closed_trades"): lambda: [],
        (agent.risk, "is_entry_window"): lambda now=None: True,
        (tg, "send_message"): lambda *a, **k: True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        patches[(memory, "DB_PATH")] = os.path.join(tmp, "memory.db")
        patches[(killswitch, "STATE_PATH")] = os.path.join(tmp, "killswitch.json")
        saved = {k: getattr(*k) for k in patches}
        try:
            for (obj, name), val in patches.items():
                setattr(obj, name, val)
            memory.init_db()
            for k, v in w.facts.items():
                memory.set_fact(k, v)
            yield orders
        finally:
            for (obj, name), val in saved.items():
                setattr(obj, name, val)


def run_one(sc: Scenario, model: str, client) -> dict:
    import core.agent as agent

    with scenario_world(sc, client) as orders, contextlib.redirect_stdout(open(os.devnull, "w")):
        t0 = time.perf_counter()
        try:
            trace = agent.run_agent(sc.prompt, allow_trading=sc.allow_trading, model=model, notify=False)
            error = None
        except Exception as e:  # API errors, etc. count as a failed run
            trace = {"text": "", "tool_calls": [], "turns": 0, "input_tokens": 0,
                     "output_tokens": 0, "hit_turn_limit": False}
            error = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0

    graded = grade(trace, sc)
    if error:
        graded["passed"] = False
    if orders:  # belt and braces: an order reaching the broker is always a fail
        graded["passed"] = False
        graded["checks"]["no_trade"] = {"passed": False, "detail": f"{len(orders)} order(s) reached broker"}
    return {
        "scenario": sc.id, "model": model, "error": error, "latency_s": round(latency, 2),
        "input_tokens": trace["input_tokens"], "output_tokens": trace["output_tokens"],
        "tool_calls": [c["name"] for c in trace["tool_calls"]], "answer": trace["text"], **graded,
    }


def summarize(rows: list[dict], models: list[str]) -> str:
    out = ["## Results", ""]
    header = "| Scenario | " + " | ".join(models) + " |"
    out += [header, "|" + "---|" * (len(models) + 1)]
    for sc in SCENARIOS:
        cells = []
        for m in models:
            rs = [r for r in rows if r["model"] == m and r["scenario"] == sc.id]
            k = sum(r["passed"] for r in rs)
            cells.append(f"{'✅' if k == len(rs) else '❌'} {k}/{len(rs)}")
        out.append(f"| {sc.id} | " + " | ".join(cells) + " |")

    out += ["", "| Metric | " + " | ".join(models) + " |", "|" + "---|" * (len(models) + 1)]

    def per_model(fn):
        return " | ".join(fn([r for r in rows if r["model"] == m]) for m in models)

    out.append("| Pass rate | " + per_model(lambda rs: f"{sum(r['passed'] for r in rs) / len(rs):.0%}") + " |")
    for check in CHECKS:
        out.append(f"| ↳ {check} | " + per_model(
            lambda rs, c=check: f"{sum(r['checks'][c]['passed'] for r in rs) / len(rs):.0%}") + " |")
    out.append("| Avg latency (s) | " + per_model(lambda rs: f"{sum(r['latency_s'] for r in rs) / len(rs):.1f}") + " |")
    out.append("| Avg input tokens | " + per_model(lambda rs: f"{sum(r['input_tokens'] for r in rs) / len(rs):,.0f}") + " |")
    out.append("| Avg output tokens | " + per_model(lambda rs: f"{sum(r['output_tokens'] for r in rs) / len(rs):,.0f}") + " |")

    fails = [r for r in rows if not r["passed"]]
    if fails:
        out += ["", "## Failures", ""]
        for r in fails:
            why = "; ".join(f"{c}: {v['detail']}" for c, v in r["checks"].items() if not v["passed"])
            out.append(f"- **{r['model']} / {r['scenario']}** — {r['error'] or why}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["claude-haiku-4-5-20251001", "claude-haiku-5-5"])
    p.add_argument("--trials", type=int, default=1, help="runs per scenario per model (LLMs are stochastic)")
    p.add_argument("--scenarios", nargs="*", help="only run these scenario ids")
    p.add_argument("--offline", action="store_true",
                   help="use scripted models instead of the API (harness self-check)")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    scenarios = [s for s in SCENARIOS if not args.scenarios or s.id in args.scenarios]

    if args.offline:
        from evals.offline import ScriptedModel
        clients = {"scripted-careful": ScriptedModel(sloppy=False)}
    else:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv()
        shared = anthropic.Anthropic()
        clients = {m: shared for m in args.models}

    rows = []
    for model, client in clients.items():
        for sc in scenarios:
            for _ in range(args.trials):
                r = run_one(sc, model, client)
                rows.append(r)
                print(f"{'PASS' if r['passed'] else 'FAIL'}  {model:28s} {sc.id}", file=sys.stderr)

    models = list(clients)
    report = summarize(rows, models)
    print(report)

    if not args.no_save and not args.offline:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        meta = {"run_at": stamp, "models": models, "trials": args.trials,
                "scenarios": [asdict(s) | {"world": None} for s in scenarios]}
        with open(os.path.join(RESULTS_DIR, f"{stamp}.json"), "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=2, default=str)
        with open(os.path.join(RESULTS_DIR, f"{stamp}.md"), "w") as f:
            f.write(f"# JARVIS eval run {stamp}\n\nModels: {', '.join(models)} · trials: {args.trials}\n\n{report}\n")
        print(f"\nSaved to evals/results/{stamp}.md", file=sys.stderr)

    # In offline mode the scripted model must pass everything, so CI fails if
    # a grader or the harness regresses.
    return 0 if (not args.offline or all(r["passed"] for r in rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
