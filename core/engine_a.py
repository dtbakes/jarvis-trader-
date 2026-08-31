"""
engine_a.py — Engine A (trend breakout) backtester. STRATEGY.md §5, M1.

Rules
-----
    Long entry  : H1 close > prior 55-bar Donchian high  AND regime==TREND
                  AND trend_dir == +1
    Short entry : mirror on the 55-bar low with trend_dir == -1
    Initial stop: 2.0 x ATR(14, H1) from entry
    Trail       : chandelier — extreme close since entry -/+ 3.0 x ATR,
                  ratchets one way only
    Target      : NONE. Exit only via trail or initial stop.
    Position    : 1 per pair, no pyramiding.

Execution conventions (§9)
--------------------------
* Decide at bar close, execute at next bar open. All market-derived columns
  are consumed with .shift(1); fills happen at the execution bar's OPEN.
* Session blocks are clock-based, so they're checked on the execution bar
  directly (no shift) — see regime.py's note.
* Exits evaluated on H1 close, EXCEPT the hard stop: if the bar's range
  crosses it, we assume a fill at stop price + slippage against us.
* Costs on every fill: half-spread each side + slippage on stop-market exits.
  Swap is modeled per night held.

Everything is reported in R-multiples as well as %, because Engine A's whole
profile (§5: win rate 30-45%, payoff > 2) only makes sense in R.

Usage:
    python engine_a.py                       # EUR_USD, 4y, costs on
    python engine_a.py --pair GBP_USD
    python engine_a.py --no-costs            # debug only, never a headline
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from indicators import atr as _atr, donchian as _donchian
from regime import (
    Regime,
    RegimeConfig,
    align_to_lower_tf,
    classify_regimes,
    _news_block,
    _session_block,
)

# --------------------------------------------------------------------------- #
# Cost model (§9) — spreads in pips, per pair
# --------------------------------------------------------------------------- #

SPREAD_PIPS = {
    "EUR_USD": 1.3, "GBP_USD": 1.8, "USD_JPY": 1.4, "USD_CHF": 1.9,
    "USD_CAD": 1.9, "AUD_USD": 1.5, "NZD_USD": 2.2, "EUR_GBP": 2.0,
    "EUR_JPY": 2.2, "GBP_JPY": 3.0,
}
DEFAULT_SPREAD_PIPS = 2.5
STOP_SLIPPAGE_PIPS = 0.3          # §9: stop-market fills slip against us
SWAP_PIPS_PER_NIGHT = 0.15        # rough financing drag; refine from OANDA later


def pip_size(pair: str) -> float:
    """JPY pairs quote to 2dp, everything else to 4dp."""
    return 0.01 if pair.endswith("_JPY") else 0.0001


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngineAConfig:
    donchian_period: int = 55         # §12: sane range 40-80
    atr_period: int = 14
    initial_stop_atr: float = 2.0     # §12: 1.5-2.5
    trail_atr: float = 3.0            # §12: 2.5-3.5
    allow_shorts: bool = True
    costs_on: bool = True


@dataclass
class Trade:
    pair: str
    direction: int                    # +1 long, -1 short
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_px: float
    exit_px: float
    initial_stop: float
    exit_reason: str                  # 'trail' | 'stop'
    r_multiple: float
    pnl_pct: float
    bars_held: int
    adx: float
    atr: float


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #

def run_engine_a(
    exec_df: pd.DataFrame,
    h4: pd.DataFrame,
    pair: str = "EUR_USD",
    cfg: EngineAConfig = EngineAConfig(),
    regime_cfg: RegimeConfig = RegimeConfig(),
    news_windows=None,
) -> tuple[list[Trade], dict]:
    """Run Engine A. Regime always comes from H4; entries/exits/ATR run on
    `exec_df`, which may be H1 (§4 default) or H4 itself (single-timeframe
    variant). Returns (trades, stats)."""

    # Bar duration drives the swap accrual and the alignment path.
    bar_delta = pd.Timedelta(exec_df.index[1] - exec_df.index[0])
    bars_per_day = max(int(round(pd.Timedelta(days=1) / bar_delta)), 1)
    same_tf = exec_df.index.equals(h4.index)

    # ---- Layer 1: regime on H4, aligned onto H1 without lookahead --------- #
    h4c = classify_regimes(h4, regime_cfg, news_windows)
    _cols = ("regime", "trend_dir", "adx", "atr", "atr_pct")
    if same_tf:
        # Regime and execution share the frame: no cross-timeframe mapping is
        # needed (and none must be applied). The .shift(1) below still enforces
        # decide-at-close / execute-at-next-open.
        aligned = h4c[list(_cols)].copy()
    else:
        aligned = align_to_lower_tf(
            h4c, exec_df.index, pd.Timedelta(hours=4), cols=_cols,
        )
    # H4's atr/adx must not collide with the H1 ATR used for stops and sizing.
    aligned = aligned.rename(columns={c: f"h4_{c}" for c in ("adx", "atr", "atr_pct")})
    df = exec_df.join(aligned)

    # Session blocks are pure clock arithmetic, so per §9 / regime.py they are
    # evaluated on the H1 EXECUTION bar — not inherited from the coarse H4 bar
    # (an H4 bar opening 16:00 NY straddles the 17:00 rollover).
    df["blocked_session"] = _session_block(df.index, regime_cfg)
    if news_windows:
        df["blocked_news_h1"] = _news_block(df.index, news_windows)
    else:
        df["blocked_news_h1"] = False

    # ---- H1 indicators ---------------------------------------------------- #
    dc = _donchian(df, cfg.donchian_period)
    df["dc_upper"] = dc["dc_upper"]
    df["dc_lower"] = dc["dc_lower"]
    df["atr"] = _atr(df, cfg.atr_period)

    # ---- Decide-at-close / execute-at-next-open (§9) ---------------------- #
    # Breakout tests the PRIOR N-bar extreme, then the whole signal is shifted
    # once more so the decision only uses the last CLOSED bar.
    sig_long = df["close"] > df["dc_upper"].shift(1)
    sig_short = df["close"] < df["dc_lower"].shift(1)

    df["sig_long"] = sig_long.shift(1).fillna(False)
    df["sig_short"] = sig_short.shift(1).fillna(False)
    df["regime_t"] = df["regime"].shift(1)
    df["dir_t"] = df["trend_dir"].shift(1)
    df["atr_t"] = df["atr"].shift(1)
    df["adx_t"] = df["h4_adx"].shift(1)
    # session block is clock-based -> evaluated on the execution bar itself
    df["blocked_now"] = df["blocked_session"] | df["blocked_news_h1"]

    pip = pip_size(pair)
    half_spread = (SPREAD_PIPS.get(pair, DEFAULT_SPREAD_PIPS) * pip) / 2.0
    slip = STOP_SLIPPAGE_PIPS * pip
    swap = SWAP_PIPS_PER_NIGHT * pip
    if not cfg.costs_on:
        half_spread = slip = swap = 0.0

    trades: list[Trade] = []
    pos = None

    rows = df.itertuples()
    for bar in rows:
        # ------------------------------------------------------------------ #
        # 1. Manage an open position first (always runs, even when blocked)
        # ------------------------------------------------------------------ #
        if pos is not None:
            exit_px = None
            reason = None

            if pos["dir"] == 1:
                # hard stop: did the bar's range cross it?
                if bar.low <= pos["stop"]:
                    exit_px = pos["stop"] - slip - half_spread
                    # 'stop' = original risk point; 'trail' = ratcheted chandelier
                    reason = "stop" if pos["stop"] <= pos["init_stop"] else "trail"
                else:
                    pos["extreme"] = max(pos["extreme"], bar.close)
                    if not np.isnan(bar.atr):
                        new_trail = pos["extreme"] - cfg.trail_atr * bar.atr
                        pos["stop"] = max(pos["stop"], new_trail)  # ratchet only
                    if bar.close <= pos["stop"]:
                        exit_px = bar.close - half_spread
                        reason = "trail"
            else:
                if bar.high >= pos["stop"]:
                    exit_px = pos["stop"] + slip + half_spread
                    reason = "stop" if pos["stop"] >= pos["init_stop"] else "trail"
                else:
                    pos["extreme"] = min(pos["extreme"], bar.close)
                    if not np.isnan(bar.atr):
                        new_trail = pos["extreme"] + cfg.trail_atr * bar.atr
                        pos["stop"] = min(pos["stop"], new_trail)
                    if bar.close >= pos["stop"]:
                        exit_px = bar.close + half_spread
                        reason = "trail"

            if exit_px is not None:
                bars_held = pos["bars"]
                nights = bars_held // bars_per_day
                gross = (exit_px - pos["entry"]) * pos["dir"]
                net = gross - nights * swap
                risk = abs(pos["entry"] - pos["init_stop"])
                trades.append(
                    Trade(
                        pair=pair,
                        direction=pos["dir"],
                        entry_ts=pos["entry_ts"],
                        exit_ts=bar.Index,
                        entry_px=pos["entry"],
                        exit_px=exit_px,
                        initial_stop=pos["init_stop"],
                        exit_reason=reason,
                        r_multiple=net / risk if risk > 0 else 0.0,
                        pnl_pct=net / pos["entry"],
                        bars_held=bars_held,
                        adx=pos["adx"],
                        atr=pos["atr"],
                    )
                )
                pos = None
            else:
                pos["bars"] += 1
                continue  # still in the trade; no new entry this bar

        # ------------------------------------------------------------------ #
        # 2. Entry checks (only when flat)
        # ------------------------------------------------------------------ #
        if pos is not None or bar.blocked_now:
            continue
        if bar.regime_t != Regime.TREND.value:
            continue
        if np.isnan(bar.atr_t) or bar.atr_t <= 0:
            continue

        direction = 0
        if bar.sig_long and bar.dir_t == 1:
            direction = 1
        elif cfg.allow_shorts and bar.sig_short and bar.dir_t == -1:
            direction = -1
        if direction == 0:
            continue

        # fill at this bar's OPEN, paying half the spread
        entry = bar.open + direction * half_spread
        init_stop = entry - direction * cfg.initial_stop_atr * bar.atr_t
        pos = {
            "dir": direction,
            "entry": entry,
            "entry_ts": bar.Index,
            "stop": init_stop,
            "init_stop": init_stop,
            "extreme": bar.close,
            "bars": 1,
            "adx": float(bar.adx_t) if not np.isnan(bar.adx_t) else 0.0,
            "atr": float(bar.atr_t),
        }

    stats = summarize(trades, df, pair, cfg)
    stats["exec_tf"] = "H4" if same_tf else "H1"
    stats["bars_per_day"] = bars_per_day
    return trades, stats


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def summarize(trades: list[Trade], df: pd.DataFrame, pair: str, cfg: EngineAConfig) -> dict:
    if not trades:
        return {"pair": pair, "n_trades": 0, "note": "no trades triggered"}

    r = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.pnl_pct for t in trades])
    wins, losses = r[r > 0], r[r <= 0]

    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    equity = np.cumprod(1 + pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((peak - equity) / peak).max())

    # Sharpe on the per-trade R series, annualized by realized trade frequency
    span_years = (df.index[-1] - df.index[0]).days / 365.25
    trades_per_year = len(trades) / span_years if span_years > 0 else 0.0
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(trades_per_year))
        if len(r) > 1 and r.std(ddof=1) > 0
        else 0.0
    )

    return {
        "pair": pair,
        "engine": "A",
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(r)),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float(abs(wins.mean() / losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0
        else float("inf"),
        "expectancy_r": float(r.mean()),
        "total_return_pct": float(equity[-1] - 1),
        "profit_factor": float(pf),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trades_per_year": float(trades_per_year),
        "span_years": float(span_years),
        "pct_stopped": float(sum(1 for t in trades if t.exit_reason == "stop") / len(trades)),
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
        "costs_on": cfg.costs_on,
    }


def print_report(stats: dict) -> None:
    if stats.get("n_trades", 0) == 0:
        print(f"\n=== ENGINE A | {stats['pair']} === no trades triggered.\n")
        return

    s = stats
    print(f"\n=== ENGINE A (trend breakout) | {s['pair']} | "
          f"entries {s.get('exec_tf','H1')} | {s['span_years']:.1f}y | "
          f"costs {'ON' if s['costs_on'] else 'OFF'} ===")
    print(f"Trades:            {s['n_trades']}  ({s['trades_per_year']:.0f}/yr)")
    print(f"Win rate:          {s['win_rate']:.1%}   (spec expects 30-45%)")
    print(f"Avg win:           {s['avg_win_r']:+.2f}R")
    print(f"Avg loss:          {s['avg_loss_r']:+.2f}R")
    print(f"Payoff ratio:      {s['payoff_ratio']:.2f}   (spec expects > 2)")
    print(f"Expectancy:        {s['expectancy_r']:+.3f}R per trade")
    print(f"Profit factor:     {s['profit_factor']:.2f}   (kill threshold 1.05)")
    print(f"Total return:      {s['total_return_pct']:+.1%}  (unsized, 1R-per-trade basis)")
    print(f"Max drawdown:      {s['max_dd']:.1%}")
    print(f"Sharpe (approx):   {s['sharpe']:.2f}")
    print(f"Stopped out:       {s['pct_stopped']:.0%} of exits")
    print(f"Avg hold:          {s['avg_bars_held']:.0f} {s.get('exec_tf','')} bars "
          f"({s['avg_bars_held']/s.get('bars_per_day',24):.1f} days)")

    flags = []
    if s["n_trades"] < 200:
        flags.append(f"SAMPLE TOO SMALL ({s['n_trades']} < 200 per §10.6)")
    if s["profit_factor"] < 1.05:
        flags.append("PF below §11 kill threshold")
    if s["max_dd"] > 0.20:
        flags.append("max DD above the §11 20% line")
    print("\n" + ("\n".join("  !! " + f for f in flags) if flags
                  else "  no §10/§11 flags on this run"))
    print("  (in-sample, single pair — diagnosis only, NOT a headline metric)\n")


# --------------------------------------------------------------------------- #
# trials logging (§8 / §10.5) — every run gets recorded, no exceptions
# --------------------------------------------------------------------------- #

def log_trial(db_path: str, stats: dict, params: dict, period: str, oos: int = 0) -> None:
    if stats.get("n_trades", 0) == 0:
        return
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trials (
          id INTEGER PRIMARY KEY,
          ts TEXT, params_json TEXT, period TEXT,
          n_trades INTEGER, sharpe REAL, pf REAL, maxdd REAL,
          oos INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO trials (ts, params_json, period, n_trades, sharpe, pf, maxdd, oos) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(params, sort_keys=True),
            period,
            stats["n_trades"],
            stats["sharpe"],
            stats["profit_factor"],
            stats["max_dd"],
            oos,
        ),
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Engine A backtest")
    ap.add_argument("--pair", default="EUR_USD")
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--donchian", type=int, default=55)
    ap.add_argument("--trail", type=float, default=3.0)
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--no-costs", action="store_true")
    ap.add_argument("--no-shorts", action="store_true")
    ap.add_argument("--entry-tf", choices=["H1", "H4"], default="H1",
                    help="timeframe for entries/exits/ATR; regime is always H4")
    args = ap.parse_args()

    from data import get_history

    cfg = EngineAConfig(
        donchian_period=args.donchian,
        trail_atr=args.trail,
        initial_stop_atr=args.stop,
        allow_shorts=not args.no_shorts,
        costs_on=not args.no_costs,
    )

    h4 = get_history(args.pair, "H4", years=args.years)
    exec_df = h4 if args.entry_tf == "H4" else get_history(args.pair, "H1", years=args.years)

    trades, stats = run_engine_a(exec_df, h4, pair=args.pair, cfg=cfg)
    print_report(stats)

    db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memory.db"
    )
    log_trial(
        db, stats,
        {**asdict(cfg), "pair": args.pair, "years": args.years,
         "entry_tf": args.entry_tf},
        period=f"{exec_df.index[0]:%Y-%m-%d}..{exec_df.index[-1]:%Y-%m-%d}", oos=0,
    )
    print(f"  logged to trials table in {db}")


if __name__ == "__main__":
    main()