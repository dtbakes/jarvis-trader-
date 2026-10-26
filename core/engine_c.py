"""
engine_c.py — Engine C (higher-timeframe zone rejection) backtester.

Hypothesis
----------
Nick Shawn (MissionFX) trades obvious daily/weekly support and resistance
zones: price pokes into a zone it has turned from before, fails, and he
sells with the stop above the zone and a 1:1 target, exiting early if
price moves back against him. He says he takes sells only.

structure_lab.py found no edge in H1 "sweeps" with 5-bar swings. This is
the higher-timeframe version that was never tested: daily bars, zones built
from multi-touch swing clusters over the past year.

Rules (daily bars)
------------------
    Swings     : bar i is a swing high if its high is the max of
                 [i-L, i+L]; it only becomes KNOWN at bar i+L.
    Zones      : known swing highs from the last `zone_lookback` bars are
                 clustered when within `cluster_atr` x ATR of each other.
                 A cluster with >= `min_touches` swings is a resistance zone
                 [min high, max high] (widened to at least `min_width_atr`).
                 Support zones mirror this on swing lows.
    Sell signal: bar high reaches the zone bottom AND the bar closes back
                 below the zone bottom (poked in, rejected).
    Buy signal : mirror at support (disabled when sells_only=True).
    Entry      : next bar's open (decide at close, execute at next open).
    Stop       : above max(zone top, signal bar high) + stop_buffer_atr x ATR.
    Target     : rr x initial risk (1:1 by default).
    Early exit : a daily close back inside the zone (price moved against
                 the idea) exits at that close. Toggle with early_exit.
    Friday     : no entries on the Friday session (weekend gap).
    Position   : 1 per pair; `cooldown_bars` after any exit.

Costs and conventions match engine_a: half-spread per fill, slippage on
stop fills, a flat swap drag per night, results in R.

Usage (from core/):
    python engine_c.py --universe all                 # the real test: all pairs,
                                                      # sells-only vs both, pooled
    python engine_c.py --pair GBP_NZD                # sells only, 10y daily
    python engine_c.py --pair GBP_NZD --both          # sells and buys
    python engine_c.py --pair EUR_JPY --no-early-exit
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from indicators import atr as _atr
from engine_a import (
    DEFAULT_SPREAD_PIPS,
    SPREAD_PIPS,
    STOP_SLIPPAGE_PIPS,
    SWAP_PIPS_PER_NIGHT,
    Trade,
    log_trial,
    pip_size,
    summarize,
)

# Crosses Nick trades that aren't in the A/B universe. Rough retail spreads.
CROSS_SPREAD_PIPS = {
    "GBP_NZD": 4.0, "GBP_CAD": 3.0, "GBP_AUD": 3.0, "NZD_JPY": 2.5,
    "AUD_JPY": 2.0, "EUR_AUD": 2.5, "EUR_NZD": 3.5, "EUR_CAD": 2.5,
    "CAD_JPY": 2.5, "AUD_NZD": 3.0,
}

NICK_PAIRS = ["GBP_NZD", "GBP_CAD", "GBP_AUD", "NZD_JPY", "EUR_JPY"]


@dataclass(frozen=True)
class EngineCConfig:
    swing_lookback: int = 5           # L: bars each side to confirm a swing
    zone_lookback: int = 250          # ~1 trading year of swings feed zones
    cluster_atr: float = 0.5          # swings within this x ATR join one zone
    min_touches: int = 2              # "clear and obvious": tested before
    min_width_atr: float = 0.25       # thin zones widened to this x ATR
    stop_buffer_atr: float = 0.25     # stop sits this far above the zone
    rr: float = 1.0                   # target = rr x risk (Nick: 1:1)
    early_exit: bool = True           # exit on a close back inside the zone
    sells_only: bool = True           # Nick: sells only
    skip_friday: bool = True          # no Friday-session entries
    cooldown_bars: int = 3
    atr_period: int = 14
    costs_on: bool = True


# --------------------------------------------------------------------------- #
# Zones (lookahead-free)
# --------------------------------------------------------------------------- #

def confirmed_swings(df: pd.DataFrame, L: int) -> tuple[list, list]:
    """
    Returns (highs, lows) as lists of (confirm_idx, swing_idx, price).
    A swing at i is only knowable at i + L, so confirm_idx = i + L.
    """
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(df)
    highs, lows = [], []
    for i in range(L, n - L):
        win_h = h[i - L:i + L + 1]
        win_l = lo[i - L:i + L + 1]
        if h[i] == win_h.max() and np.argmax(win_h) == L:
            highs.append((i + L, i, float(h[i])))
        if lo[i] == win_l.min() and np.argmin(win_l) == L:
            lows.append((i + L, i, float(lo[i])))
    return highs, lows


def build_zones(prices: list[float], tol: float, min_touches: int,
                min_width: float) -> list[tuple[float, float, int]]:
    """Cluster sorted prices within `tol`; keep clusters with enough touches.
    Returns [(bottom, top, touches)]."""
    if not prices:
        return []
    ps = sorted(prices)
    clusters, cur = [], [ps[0]]
    for p in ps[1:]:
        if p - cur[-1] <= tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)

    zones = []
    for c in clusters:
        if len(c) < min_touches:
            continue
        bottom, top = min(c), max(c)
        if top - bottom < min_width:
            mid = (top + bottom) / 2
            bottom, top = mid - min_width / 2, mid + min_width / 2
        zones.append((bottom, top, len(c)))
    return zones


def _active(swings: list, t: int, lookback: int) -> list[float]:
    """Prices of swings known at bar t (confirm_idx <= t) within the window."""
    return [p for c, i, p in swings if c <= t and i >= t - lookback]


def _session_weekday(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Trading-session weekday of each daily bar. FX days roll at 17:00 New
    York, so a bar opening Thursday 17:00 NY is Friday's session. Adding 7h
    to the NY open time lands inside the session whatever the alignment.
    """
    ny = index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return np.asarray(ny.dayofweek)


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #

def run_engine_c(
    df: pd.DataFrame,
    pair: str = "GBP_NZD",
    cfg: EngineCConfig = EngineCConfig(),
) -> tuple[list[Trade], dict]:
    """Run Engine C on daily bars. Returns (trades, stats)."""
    df = df.copy()
    df["atr"] = _atr(df, cfg.atr_period)
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = df["atr"].to_numpy()
    idx = df.index
    weekday = _session_weekday(idx)
    n = len(df)

    swing_highs, swing_lows = confirmed_swings(df, cfg.swing_lookback)

    pip = pip_size(pair)
    spread = CROSS_SPREAD_PIPS.get(pair, SPREAD_PIPS.get(pair, DEFAULT_SPREAD_PIPS))
    half_spread = spread * pip / 2.0
    slip = STOP_SLIPPAGE_PIPS * pip
    swap = SWAP_PIPS_PER_NIGHT * pip
    if not cfg.costs_on:
        half_spread = slip = swap = 0.0

    trades: list[Trade] = []
    pos = None
    pending = None          # signal decided at close of t, filled at open of t+1
    cooldown = 0

    for t in range(n):
        # ---- 1. fill a pending entry at this bar's open ------------------- #
        if pending is not None and pos is None:
            d = pending["dir"]
            entry = o[t] + d * half_spread
            stop = pending["stop"]
            risk = (stop - entry) * -d
            if risk > 0:
                pos = {
                    "dir": d, "entry": entry, "entry_t": t, "stop": stop,
                    "target": entry + d * cfg.rr * risk, "risk": risk,
                    "zone": pending["zone"], "atr": pending["atr"],
                }
            pending = None

        # ---- 2. manage the open position --------------------------------- #
        exited = False
        if pos is not None:
            d = pos["dir"]
            exit_px = reason = None
            hit_stop = (h[t] >= pos["stop"]) if d == -1 else (lo[t] <= pos["stop"])
            hit_target = (lo[t] <= pos["target"]) if d == -1 else (h[t] >= pos["target"])
            if hit_stop:
                # if both are inside one bar, assume the worst: the stop.
                # A gap through the stop fills at the open, not the stop.
                fill = pos["stop"]
                if t > pos["entry_t"]:
                    fill = max(fill, o[t]) if d == -1 else min(fill, o[t])
                exit_px = fill - d * (slip + half_spread)
                reason = "stop"
            elif hit_target:
                exit_px = pos["target"] - d * half_spread
                reason = "target"
            elif cfg.early_exit:
                bottom, top = pos["zone"]
                back_inside = c[t] >= bottom if d == -1 else c[t] <= top
                if back_inside:
                    exit_px = c[t] - d * half_spread
                    reason = "early"

            if exit_px is not None:
                bars_held = t - pos["entry_t"] + 1
                net = (exit_px - pos["entry"]) * d - bars_held * swap
                trades.append(Trade(
                    pair=pair, direction=d,
                    entry_ts=idx[pos["entry_t"]], exit_ts=idx[t],
                    entry_px=pos["entry"], exit_px=exit_px,
                    initial_stop=pos["stop"], exit_reason=reason,
                    r_multiple=net / pos["risk"], pnl_pct=net / pos["entry"],
                    bars_held=bars_held, adx=0.0, atr=pos["atr"],
                ))
                pos = None
                exited = True
                cooldown = cfg.cooldown_bars

        if pos is not None or exited:
            continue  # in a trade, or no new decision on the exit bar
        if cooldown > 0:
            cooldown -= 1
            continue

        # ---- 3. decide at this bar's close (entry at t+1 open) ------------ #
        if t + 1 >= n or np.isnan(atr[t]) or atr[t] <= 0:
            continue
        if cfg.skip_friday and weekday[t + 1] == 4:
            continue

        tol = cfg.cluster_atr * atr[t]
        width = cfg.min_width_atr * atr[t]
        buf = cfg.stop_buffer_atr * atr[t]

        res = build_zones(_active(swing_highs, t, cfg.zone_lookback),
                          tol, cfg.min_touches, width)
        # nearest resistance whose bottom was reached and rejected
        sells = [z for z in res if h[t] >= z[0] and c[t] < z[0]]
        if sells:
            bottom, top, _ = min(sells, key=lambda z: z[0])
            pending = {"dir": -1, "stop": max(top, h[t]) + buf,
                       "zone": (bottom, top), "atr": float(atr[t])}
            continue

        if not cfg.sells_only:
            sup = build_zones(_active(swing_lows, t, cfg.zone_lookback),
                              tol, cfg.min_touches, width)
            buys = [z for z in sup if lo[t] <= z[1] and c[t] > z[1]]
            if buys:
                bottom, top, _ = max(buys, key=lambda z: z[1])
                pending = {"dir": 1, "stop": min(bottom, lo[t]) - buf,
                           "zone": (bottom, top), "atr": float(atr[t])}

    stats = summarize(trades, df, pair, cfg)
    stats["engine"] = "C"
    stats["exec_tf"] = "D"
    stats["bars_per_day"] = 1
    if trades:
        stats["exit_mix"] = {r: sum(1 for x in trades if x.exit_reason == r) / len(trades)
                             for r in ("target", "stop", "early")}
    return trades, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def run_universe(pairs: list[str], years: float, cfg_kwargs: dict, log: bool = True) -> None:
    """Sells-only vs both-directions, side by side, per pair and pooled."""
    from data import get_history
    from run_universe import pooled_stats

    db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "memory.db")
    variants = {"sells_only": True, "both": False}
    pooled = {v: [] for v in variants}
    rows = []
    for pair in pairs:
        try:
            d = get_history(pair, "D", years=years, verbose=False)
        except Exception as e:
            print(f"  !! {pair}: daily data unavailable ({e})")
            continue
        for name, so in variants.items():
            cfg = EngineCConfig(**{**cfg_kwargs, "sells_only": so})
            trades, st = run_engine_c(d, pair=pair, cfg=cfg)
            pooled[name].extend(trades)
            rows.append({
                "pair": pair, "variant": name, "trades": st.get("n_trades", 0),
                "win%": f"{st['win_rate']:.0%}" if trades else "-",
                "exp(R)": f"{st['expectancy_r']:+.3f}" if trades else "-",
                "PF": f"{st['profit_factor']:.2f}" if trades else "-",
                "maxDD": f"{st['max_dd']:.1%}" if trades else "-",
            })
            if log and trades:
                log_trial(db, st, {**asdict(cfg), "engine": "C", "pair": pair,
                                   "years": years, "entry_tf": "D",
                                   "run": "engine_c_universe"},
                          period=f"{d.index[0]:%Y-%m-%d}..{d.index[-1]:%Y-%m-%d}", oos=0)

    print(f"\n=== ENGINE C | daily zone rejection | {years:.0f}y | "
          f"rr {cfg_kwargs.get('rr', 1.0)} | early exit {cfg_kwargs.get('early_exit', True)} | "
          f"costs {'ON' if cfg_kwargs.get('costs_on', True) else 'OFF'} ===\n")
    print(pd.DataFrame(rows).to_string(index=False))
    for name, trades in pooled.items():
        if not trades:
            continue
        agg = pooled_stats(trades)
        be = 1.0 / (1.0 + agg["payoff"]) if agg["payoff"] > 0 else float("nan")
        print(f"\n--- POOLED {name} ({agg['n']} trades) ---")
        print(f"Win rate {agg['win_rate']:.1%} (breakeven at this payoff {be:.1%}) | "
              f"payoff {agg['payoff']:.2f} | expectancy {agg['expectancy_r']:+.3f}R | "
              f"PF {agg['pf']:.2f} (kill < 1.05) | maxDD {agg['max_dd']:.1%}"
              + ("" if agg["n"] >= 200 else "  !! under 200 trades"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Engine C (zone rejection) backtest")
    ap.add_argument("--pair", default="GBP_NZD")
    ap.add_argument("--universe", choices=["nick", "default", "all"],
                    help="run many pairs, sells-only vs both side by side")
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--both", action="store_true", help="take buys at support too")
    ap.add_argument("--rr", type=float, default=1.0)
    ap.add_argument("--no-early-exit", action="store_true")
    ap.add_argument("--no-costs", action="store_true")
    args = ap.parse_args()

    cfg_kwargs = {"rr": args.rr, "early_exit": not args.no_early_exit,
                  "costs_on": not args.no_costs}

    if args.universe:
        from run_universe import PAIRS_DEFAULT
        pairs = {"nick": NICK_PAIRS, "default": PAIRS_DEFAULT,
                 "all": list(dict.fromkeys(NICK_PAIRS + PAIRS_DEFAULT))}[args.universe]
        run_universe(pairs, args.years, cfg_kwargs)
        return

    from data import get_history

    cfg = EngineCConfig(sells_only=not args.both, **cfg_kwargs)
    d = get_history(args.pair, "D", years=args.years)
    trades, stats = run_engine_c(d, pair=args.pair, cfg=cfg)
    if not trades:
        print(f"\n=== ENGINE C | {args.pair} === no trades triggered.")
        return
    s = stats
    print(f"\n=== ENGINE C | {args.pair} | {'sells only' if cfg.sells_only else 'both'} | "
          f"{s['span_years']:.1f}y daily | costs {'ON' if cfg.costs_on else 'OFF'} ===")
    print(f"Trades {s['n_trades']} | win {s['win_rate']:.1%} | payoff {s['payoff_ratio']:.2f} | "
          f"expectancy {s['expectancy_r']:+.3f}R | PF {s['profit_factor']:.2f} | "
          f"maxDD {s['max_dd']:.1%} | avg hold {s['avg_bars_held']:.1f}d")
    print("Exits: " + ", ".join(f"{k} {v:.0%}" for k, v in s["exit_mix"].items()))
    if s["n_trades"] < 200:
        print("  !! single pair, under 200 trades: diagnosis only. Use --universe.")

    db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "memory.db")
    log_trial(db, stats, {**asdict(cfg), "engine": "C", "pair": args.pair,
                          "years": args.years, "entry_tf": "D"},
              period=f"{d.index[0]:%Y-%m-%d}..{d.index[-1]:%Y-%m-%d}", oos=0)


if __name__ == "__main__":
    main()
