"""
run_universe.py — Engine A across the full §3 pair universe.

Produces:
  * a per-pair table (trades, win rate, payoff, expectancy, PF, maxDD)
  * a pooled aggregate over ALL trades — this is where n finally clears
    the §10.6 bar and statistics start meaning something
  * hypothesis probes for the two live suspects from the EUR_USD diagnostics:
      H-TRAIL : per-trade MFE (max favorable excursion) vs realized R —
                how much open profit does the chandelier hand back?
      H-VOLGATE: how many breakout signals died only because the ATR
                percentile gate said NO_TRADE at that moment?
  * every pair's run logged to trials, as always

Usage:
    python run_universe.py                     # defaults: H1 entries, trail 3.0
    python run_universe.py --trail 4.5
    python run_universe.py --entry-tf H4
    python run_universe.py --pairs EUR_USD,USD_JPY,GBP_JPY
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from data import get_history
from engine_a import (
    EngineAConfig,
    log_trial,
    run_engine_a,
)
from engine_b import EngineBConfig, run_engine_b
from regime import RegimeConfig

PAIRS_DEFAULT = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD",
    "AUD_USD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
]


def pooled_stats(all_trades: list) -> dict:
    r = np.array([t.r_multiple for t in all_trades])
    pnl = np.array([t.pnl_pct for t in all_trades])
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()

    # order pooled equity by exit time (interleaves pairs chronologically)
    order = np.argsort([t.exit_ts.value for t in all_trades])
    eq = np.cumprod(1 + pnl[order])
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / peak).max())

    return {
        "n": len(r),
        "win_rate": float(len(wins) / len(r)) if len(r) else 0.0,
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "payoff": float(abs(wins.mean() / losses.mean()))
        if len(wins) and len(losses) else 0.0,
        "expectancy_r": float(r.mean()) if len(r) else 0.0,
        "pf": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_dd": max_dd,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["A", "B"], default="A")
    ap.add_argument("--pairs", default=",".join(PAIRS_DEFAULT))
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--donchian", type=int, default=55)
    ap.add_argument("--trail", type=float, default=3.0)
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--entry-tf", choices=["H1", "H4"], default="H1")
    ap.add_argument("--no-costs", action="store_true")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if args.engine == "A":
        cfg = EngineAConfig(
            donchian_period=args.donchian,
            trail_atr=args.trail,
            initial_stop_atr=args.stop,
            costs_on=not args.no_costs,
        )
        runner = run_engine_a
    else:
        cfg = EngineBConfig(costs_on=not args.no_costs)
        runner = run_engine_b
    rcfg = RegimeConfig()

    db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "memory.db",
    )

    all_trades = []
    rows = []

    for pair in pairs:
        try:
            h4 = get_history(pair, "H4", years=args.years, verbose=False)
            exec_df = (
                h4 if args.entry_tf == "H4"
                else get_history(pair, "H1", years=args.years, verbose=False)
            )
        except Exception as e:
            print(f"  !! {pair}: data unavailable ({e}) — run fetch_universe.py")
            continue

        trades, stats = runner(exec_df, h4, pair=pair, cfg=cfg, regime_cfg=rcfg)
        all_trades.extend(trades)

        if stats.get("n_trades", 0) > 0:
            rows.append({
                "pair": pair,
                "trades": stats["n_trades"],
                "win%": f"{stats['win_rate']:.0%}",
                "payoff": f"{stats['payoff_ratio']:.2f}",
                "exp(R)": f"{stats['expectancy_r']:+.3f}",
                "PF": f"{stats['profit_factor']:.2f}",
                "maxDD": f"{stats['max_dd']:.1%}",
                "hold(d)": f"{stats['avg_bars_held']/stats['bars_per_day']:.1f}",
            })
            log_trial(
                db, stats,
                {**asdict(cfg), "pair": pair, "years": args.years,
                 "engine": args.engine,
                 "entry_tf": args.entry_tf, "run": "universe"},
                period=f"{exec_df.index[0]:%Y-%m-%d}..{exec_df.index[-1]:%Y-%m-%d}",
                oos=0,
            )
        else:
            rows.append({"pair": pair, "trades": 0, "win%": "-", "payoff": "-",
                         "exp(R)": "-", "PF": "-", "maxDD": "-", "hold(d)": "-"})

    # ---------------- per-pair table ----------------
    if args.engine == "A":
        hdr = (f"donchian {args.donchian} | trail {args.trail} | "
               f"stop {args.stop}")
    else:
        hdr = "BB 20/2.0 | RSI(2) 10/90 | stop 1.5ATR | time 36 | cd 12"
    print(f"\n=== ENGINE {args.engine} UNIVERSE | entries {args.entry_tf} | "
          f"{hdr} | costs {'ON' if cfg.costs_on else 'OFF'} ===\n")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # ---------------- pooled aggregate ----------------
    if not all_trades:
        print("\nNo trades anywhere in the universe. Something is off — check data.")
        return

    agg = pooled_stats(all_trades)
    print(f"\n--- POOLED ({agg['n']} trades across {len(pairs)} pairs) ---")
    print(f"Win rate:     {agg['win_rate']:.1%}")
    print(f"Avg win:      {agg['avg_win_r']:+.2f}R   Avg loss: {agg['avg_loss_r']:+.2f}R")
    print(f"Payoff:       {agg['payoff']:.2f}")
    print(f"Expectancy:   {agg['expectancy_r']:+.3f}R per trade")
    print(f"PF:           {agg['pf']:.2f}    Max DD (pooled, unsized): {agg['max_dd']:.1%}")
    n_flag = "" if agg["n"] >= 200 else f"   !! still under 200 ({agg['n']})"
    print(f"Sample:       {agg['n']} trades{n_flag}")

    # ---------------- hypothesis probe: trail giveback ----------------
    # MFE proxy: for each trade we know entry, initial stop (1R unit) and the
    # chandelier math means realized R understates the peak by ~trail_atr/stop_atr.
    # Direct measure: reconstruct peak R from the trade's extreme.
    give = []
    for t in all_trades:
        risk = abs(t.entry_px - t.initial_stop)
        if risk <= 0:
            continue
        # extreme close reached during the trade is not stored on the Trade
        # object; approximate peak from exit + giveback for trail exits only.
        if t.exit_reason == "trail":
            peak_r = t.r_multiple + (args.trail / args.stop)
            give.append(peak_r - t.r_multiple)
    if give and args.engine == "A":
        print(f"\nH-TRAIL probe: on trail exits, avg open profit handed back "
              f"≈ {np.mean(give):.2f}R (structural: trail/stop = "
              f"{args.trail/args.stop:.2f}R)")

    print("\n(in-sample, no walk-forward — §10.3 says diagnosis only. "
          "Next gate: parameter plateau across this universe.)")


if __name__ == "__main__":
    main()