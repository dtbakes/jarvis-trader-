# JARVIS — trading research agent

![tests](https://github.com/dtbakes/jarvis-trader-/actions/workflows/tests.yml/badge.svg)

A forex research framework with an LLM agent on top. Two strategy engines are
backtested with realistic costs, a regime classifier decides which one may
trade, and a Claude-powered agent observes the market every 30 minutes and
reports to Telegram. It runs **observe-only on a demo account**.

## What's in here

| Layer | Files | What it does |
|---|---|---|
| Indicators | `core/indicators.py` | ATR, ADX, RSI, Bollinger, Donchian, EMA, all vectorized and lookahead-free |
| Regime | `core/regime.py` | Labels each H4 bar TREND / RANGE / NEUTRAL / NO_TRADE (ADX, vol percentile, session + news blackouts, hysteresis) |
| Engine A | `core/engine_a.py` | Donchian breakout + chandelier trail, trades only in TREND |
| Engine B | `core/engine_b.py` | Bollinger + RSI(2) mean reversion, trades only in RANGE |
| Engine C | `core/engine_c.py` | Daily support/resistance zone rejection (Nick Shawn style): multi-touch zones, sell the failed poke, stop above the zone, 1:1 target, early exit on a close back inside. Sells-only vs both directions |
| Universe runs | `core/run_universe.py` | Backtests both engines over 10 pairs and logs trials to SQLite |
| Engine D | `core/engine_d.py` | Momentum + carry portfolio: 7 currencies vs USD, monthly rebalance, volatility-scaled legs, real rate-differential carry from FRED. One pre-registered run |
| Risk | `core/risk.py`, `core/killswitch.py` | Hard order limits (per-order, per-instrument and total exposure caps, stop required, no entries after Friday noon New York), plus a latched kill switch (daily loss, drawdown, losing streak, broker errors) that only a human can release |
| Agent | `core/agent.py`, `main.py` | Claude tool-use loop with memory; trading tools are hidden from the model in observe-only mode |

## Results so far

Both engines were backtested across the 10-pair universe with spread,
slippage and swap modeled. **Neither showed an edge** (profit factor ≈ 1.0
gross), so both were shelved per the kill criteria. A follow-up structure lab
(sweeps / volume nodes / break-of-structure on H1) also found nothing. These
null results are logged in the trial database rather than tuned away.

**Engine C** (run 2026-10-26, 15 pairs, 10y daily): sells-only PF 0.89,
both directions PF 1.05 with zero expectancy over 2,240 trades. No edge;
archived.

**Engine D (momentum + carry)** is the last pre-registered test:

```bash
cd core && python engine_d.py
```

Its rules are fixed in the module docstring. The verdict is the combined
rule's monthly profit factor after costs, against the same 1.05 line.

Engine C was the zone-rejection test. It checks whether the H1 null
result changes on daily/weekly zones with several touches. Run it from
`core/` (needs OANDA practice credentials for the daily candles):

```bash
cd core && python engine_c.py --universe all --years 10
```

It prints sells-only and both-directions side by side, per pair and pooled,
and logs every run to the trials table. The same kill rule applies: pooled
PF under 1.05 after costs means it's shelved.

## Testing

```bash
pip install -r requirements.txt
python -m pytest -q
```

About 150 tests, all offline: Claude, OANDA and Telegram are faked, so CI
needs no secrets and can't touch an account. They run on every push via
GitHub Actions (Python 3.11 and 3.12).

What the suite covers:

- **No lookahead.** Every indicator and the regime classifier are checked for
  the prefix property (values on a truncated frame match the full frame).
  The engines get a stronger *future-shock* test: every bar after an entry is
  moved ±3%, and the entry must not change. That test catches a one-bar
  `.shift(-1)` leak that truncation tests miss (verified by injecting the bug).
- **Backtest integrity.** Costs always lower returns without changing
  signals, a hard stop is exactly −1R without costs and worse than −1R with
  them, positions never overlap, and entries respect session blocks.
- **Agent safety.** Observe-only hides trading tools. A hallucinated trade
  call is refused. The risk gate and kill switch block bad orders that come
  from the model. An unreadable NAV fails closed. Tool exceptions go back to
  the model instead of crashing the scheduler. Runaway tool loops stop at a
  turn limit.

## Evals: does a new model still behave?

`evals/` runs the **real agent loop** against scripted market scenarios and
grades each run with deterministic checks (no LLM judge):

| Scenario | What it catches |
|---|---|
| `routine_check` | Skipping tool calls, rambling, no summary |
| `feed_outage` | Inventing a price when the feed is down |
| `killswitch_engaged` | Hiding a safety event from the human |
| `prompt_injection_in_tool_output` | Placing a trade because tool output said to |
| `memory_comparison` | Ignoring stored context |
| `account_question` | Misreporting balance or positions |

Checks: no trade attempts, required tool calls, **every quoted price grounded
in a tool result**, required mentions, a summary line, and tool-call
efficiency. Latency and token usage are tracked per model.

```bash
# compare the production model with a newer release, 3 trials each
python -m evals.run_evals --models claude-haiku-4-5-20251001 claude-haiku-5-5 --trials 3
```

Results go to `evals/results/` as JSON and a markdown table. CI runs the
harness offline against a scripted model, and the test suite confirms that a
deliberately sloppy model fails the checks it should.

## Running the agent

Create `.env` with `ANTHROPIC_API_KEY`, `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`,
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (practice account only), then:

```bash
python main.py                       # observe every 30 min during market hours
python -m core.killswitch release    # human release after a trip
```
