"""
run_daily.py — ONE-SHOT test: does Engine A work at the horizon the
trend-following literature actually documents?

Rationale
---------
STRATEGY.md §1 cites Hurst/Ooi/Pedersen (2017), whose century of evidence is
measured on DAILY-and-slower bars with multi-week holds. The H1/H4 universe
run produced 0.6-0.9 day holds and a pooled PF of 0.59 across 751 trades.
This script tests the premise at its native horizon before Engine A is
declared dead per §11.

Deliberate deviations from §4, all logged to trials as entry_tf='D':
  1. Regime AND execution both on daily bars (no cross-timeframe mapping).
  2. Session blocks NEUTRALIZED. OANDA daily candles are stamped 17:00 NY —
     precisely the rollover window regime.py blocks — so the default config
     discards ~49% of daily bars for a filter that is meaningless at this
     granularity. Verified empirically before writing this.
  3. atr_pct_window 540 -> 90, preserving the ORIGINAL INTENT ("~90 trading
     days") rather than the H4 bar count.
  4. 10 years instead of 4: daily data is ~2,600 bars/pair (one API page),
     which finally clears the §10.6 200-trade bar and spans more macro
     regimes per §10.1.

Engine parameters are left at spec defaults (Donchian 55, stop 2.0 ATR,
trail 3.0 ATR). This is a premise test, not a parameter hunt.

Usage:
    python run_daily.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from data import get_history
from engine_a import EngineAConfig, log_trial, run_engine_a
from regime import RegimeConfig
from run_universe import PAIRS_DEFAULT, pooled_stats

# Session filtering is meaningless on daily bars (see module docstring).
# These values make every clock-based block condition unsatisfiable.
DAILY_REGIME_CFG = RegimeConfig(
    atr_pct_window=90,          # ~90 trading days, matching original intent
    rollover_buffer_min=-1,     # window becomes empty
    friday_cutoff_hour=25,      # hour >= 25 never true
    sunday_open_buffer_min=0,   # window becomes empty
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS_DEFAULT))
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--donchian", type=int, default=55)
    ap.add_argument("--trail", type=float, default=3.0)
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--no-costs", action="store_true")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    cfg = EngineAConfig(
        donchian_period=args.donchian,
        trail_atr=args.trail,
        initial_stop_atr=args.stop,
        costs_on=not args.no_costs,
    )

    db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "memory.db",
    )

    all_trades, rows = [], []

    for pair in pairs:
        try:
            d = get_history(pair, "D", years=args.years, verbose=False)
        except Exception as e:
            print(f"  !! {pair}: daily data unavailable ({e})")
            continue

        # regime and execution share the daily frame
        trades, stats = run_engine_a(
            d, d, pair=pair, cfg=cfg, regime_cfg=DAILY_REGIME_CFG
        )
        all_trades.extend(trades)

        if stats.get("n_trades", 0) > 0:
            rows.append({
                "pair": pair,
                "bars": len(d),
                "trades": stats["n_trades"],
                "win%": f"{stats['win_rate']:.0%}",
                "payoff": f"{stats['payoff_ratio']:.2f}",
                "exp(R)": f"{stats['expectancy_r']:+.3f}",
                "PF": f"{stats['profit_factor']:.2f}",
                "maxDD": f"{stats['max_dd']:.1%}",
                "hold(d)": f"{stats['avg_bars_held']:.0f}",
            })
            log_trial(
                db, stats,
                {**asdict(cfg), "pair": pair, "years": args.years,
                 "entry_tf": "D", "run": "daily_premise_test"},
                period=f"{d.index[0]:%Y-%m-%d}..{d.index[-1]:%Y-%m-%d}", oos=0,
            )
        else:
            rows.append({"pair": pair, "bars": len(d), "trades": 0, "win%": "-",
                         "payoff": "-", "exp(R)": "-", "PF": "-",
                         "maxDD": "-", "hold(d)": "-"})

    print(f"\n=== ENGINE A | DAILY BARS | donchian {args.donchian} | "
          f"trail {args.trail} | stop {args.stop} | "
          f"costs {'ON' if cfg.costs_on else 'OFF'} | {args.years:.0f}y ===\n")
    print(pd.DataFrame(rows).to_string(index=False))

    if not all_trades:
        print("\nNo trades. Check the daily data pull.")
        return

    agg = pooled_stats(all_trades)
    print(f"\n--- POOLED ({agg['n']} trades across {len(pairs)} pairs) ---")
    print(f"Win rate:     {agg['win_rate']:.1%}   (spec expects 30-45%)")
    print(f"Avg win:      {agg['avg_win_r']:+.2f}R   Avg loss: {agg['avg_loss_r']:+.2f}R")
    print(f"Payoff:       {agg['payoff']:.2f}   (spec expects > 2)")
    print(f"Expectancy:   {agg['expectancy_r']:+.3f}R per trade")
    print(f"PF:           {agg['pf']:.2f}   (kill threshold 1.05)")
    print(f"Max DD:       {agg['max_dd']:.1%}   (pooled, unsized)")
    print(f"Sample:       {agg['n']} trades"
          + ("" if agg["n"] >= 200 else f"   !! still under 200"))

    hold = np.mean([t.bars_held for t in all_trades])
    print(f"Avg hold:     {hold:.0f} days   "
          f"(H1 universe run was 0.7 days; literature horizon is weeks)")

    # breakeven win rate implied by the realized payoff
    if agg["payoff"] > 0:
        be = 1.0 / (1.0 + agg["payoff"])
        print(f"\nBreakeven win rate at this payoff: {be:.1%}  "
              f"| actual: {agg['win_rate']:.1%}  "
              f"| gap: {agg['win_rate'] - be:+.1%}")

    print("\nDecision rule agreed in advance: this is ONE run. If the premise "
          "shows no life here, Engine A v1.0 is archived per §11 and we move "
          "to Engine B. No parameter cascade.")


if __name__ == "__main__":
    main()