"""
engine_b.py — Engine B (range mean-reversion) backtester. STRATEGY.md §6, M1.

Rules
-----
    Long entry  : H1 close < Bollinger(20, 2.0) lower band AND RSI(2) < 10
                  AND regime == RANGE
    Short entry : H1 close > upper band AND RSI(2) > 90 AND regime == RANGE
    Initial stop: 1.5 x ATR(14, H1) — HARD, non-negotiable
    Exit        : first of:
                    * close crosses the BB mid-band (profit take)
                    * 36 H1 bars elapsed (time stop)
                    * hard stop hit
    Cooldown    : after a stop-out, no same-direction re-entry on that pair
                  for 12 H1 bars
    Position    : 1 per pair

The four guards against the documented RSI(2)-in-FX failure mode (naive
transplants that cut winners with an RSI exit while carrying NO stop):
    1. hard ATR stop        -> loss per trade is capped at ~1R
    2. time stop            -> no zombie positions bleeding swap
    3. cooldown             -> no re-entering a falling knife every bar
    4. RANGE-only gating    -> the trade only exists in its native regime

Execution conventions identical to engine_a (§9): decide at bar close,
execute next bar open, market-derived columns consumed with .shift(1),
session blocks clock-checked on the execution bar, hard stop fills at
stop + slippage when the bar's range crosses it, spread half-paid per side,
swap per night held.

Expected profile (§6): win rate 55-70%, payoff < 1. The mirror image of
Engine A. Judge it by expectancy, not by either number alone.

Usage:
    python engine_b.py                        # EUR_USD, 4y, costs on
    python engine_b.py --pair USD_JPY
    python engine_b.py --rsi-band 5           # entry bands 5/95
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from indicators import atr as _atr, bollinger as _bollinger, rsi as _rsi
from regime import (
    Regime,
    RegimeConfig,
    align_to_lower_tf,
    classify_regimes,
    _news_block,
    _session_block,
)
from engine_a import (
    SPREAD_PIPS,
    DEFAULT_SPREAD_PIPS,
    STOP_SLIPPAGE_PIPS,
    SWAP_PIPS_PER_NIGHT,
    Trade,
    log_trial,
    pip_size,
)


@dataclass(frozen=True)
class EngineBConfig:
    bb_period: int = 20               # §12: 18-25
    bb_std: float = 2.0               # §12: 1.8-2.5
    rsi_period: int = 2               # §12: 2-4
    rsi_low: float = 10.0             # §12: 5-15
    rsi_high: float = 90.0            # §12: 85-95
    initial_stop_atr: float = 1.5     # §6: hard, non-negotiable
    atr_period: int = 14
    time_stop_bars: int = 36          # §12: 24-48
    cooldown_bars: int = 12           # §6
    allow_shorts: bool = True
    costs_on: bool = True


def run_engine_b(
    exec_df: pd.DataFrame,
    h4: pd.DataFrame,
    pair: str = "EUR_USD",
    cfg: EngineBConfig = EngineBConfig(),
    regime_cfg: RegimeConfig = RegimeConfig(),
    news_windows=None,
) -> tuple[list[Trade], dict]:
    """Run Engine B. Regime from H4, execution on exec_df (H1 per §4)."""

    bar_delta = pd.Timedelta(exec_df.index[1] - exec_df.index[0])
    bars_per_day = max(int(round(pd.Timedelta(days=1) / bar_delta)), 1)
    same_tf = exec_df.index.equals(h4.index)

    # ---- Layer 1: regime ---------------------------------------------------
    h4c = classify_regimes(h4, regime_cfg, news_windows)
    _cols = ("regime", "trend_dir", "adx", "atr", "atr_pct")
    if same_tf:
        aligned = h4c[list(_cols)].copy()
    else:
        aligned = align_to_lower_tf(
            h4c, exec_df.index, pd.Timedelta(hours=4), cols=_cols
        )
    aligned = aligned.rename(columns={c: f"h4_{c}" for c in ("adx", "atr", "atr_pct")})
    df = exec_df.join(aligned)

    # session/news on the execution bar (clock arithmetic, no lookahead)
    df["blocked_session"] = _session_block(df.index, regime_cfg)
    df["blocked_news_h1"] = (
        _news_block(df.index, news_windows) if news_windows else False
    )
    df["blocked_now"] = df["blocked_session"] | df["blocked_news_h1"]

    # ---- indicators on the execution timeframe -----------------------------
    bb = _bollinger(df["close"], cfg.bb_period, cfg.bb_std)
    df["bb_lower"] = bb["bb_lower"]
    df["bb_mid"] = bb["bb_mid"]
    df["bb_upper"] = bb["bb_upper"]
    df["rsi"] = _rsi(df["close"], cfg.rsi_period)
    df["atr"] = _atr(df, cfg.atr_period)

    # ---- decide-at-close -> execute-next-open (§9) --------------------------
    sig_long_raw = (df["close"] < df["bb_lower"]) & (df["rsi"] < cfg.rsi_low)
    sig_short_raw = (df["close"] > df["bb_upper"]) & (df["rsi"] > cfg.rsi_high)

    df["sig_long"] = sig_long_raw.shift(1).fillna(False)
    df["sig_short"] = sig_short_raw.shift(1).fillna(False)
    df["regime_t"] = df["regime"].shift(1)
    df["atr_t"] = df["atr"].shift(1)
    df["adx_t"] = df["h4_adx"].shift(1)
    # mid-band exit is decided on the PRIOR bar's close vs prior mid too:
    df["mid_t"] = df["bb_mid"].shift(1)

    pip = pip_size(pair)
    half_spread = (SPREAD_PIPS.get(pair, DEFAULT_SPREAD_PIPS) * pip) / 2.0
    slip = STOP_SLIPPAGE_PIPS * pip
    swap = SWAP_PIPS_PER_NIGHT * pip
    if not cfg.costs_on:
        half_spread = slip = swap = 0.0

    trades: list[Trade] = []
    pos = None
    cooldown_until = -1          # bar counter gate
    cooldown_dir = 0
    bar_i = -1

    for bar in df.itertuples():
        bar_i += 1

        # ---------------- manage open position ------------------------------
        if pos is not None:
            exit_px = None
            reason = None

            if pos["dir"] == 1:
                if bar.low <= pos["stop"]:
                    exit_px = pos["stop"] - slip - half_spread
                    reason = "stop"
                elif not np.isnan(bar.mid_t) and bar.close >= bar.mid_t:
                    exit_px = bar.close - half_spread
                    reason = "midband"
            else:
                if bar.high >= pos["stop"]:
                    exit_px = pos["stop"] + slip + half_spread
                    reason = "stop"
                elif not np.isnan(bar.mid_t) and bar.close <= bar.mid_t:
                    exit_px = bar.close + half_spread
                    reason = "midband"

            if exit_px is None and pos["bars"] >= cfg.time_stop_bars:
                exit_px = bar.close - pos["dir"] * half_spread
                reason = "time"

            if exit_px is not None:
                nights = pos["bars"] // bars_per_day
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
                        bars_held=pos["bars"],
                        adx=pos["adx"],
                        atr=pos["atr"],
                    )
                )
                if reason == "stop":
                    cooldown_until = bar_i + cfg.cooldown_bars
                    cooldown_dir = pos["dir"]
                pos = None
            else:
                pos["bars"] += 1
                continue

        # ---------------- entries (flat only) --------------------------------
        if pos is not None or bar.blocked_now:
            continue
        if bar.regime_t != Regime.RANGE.value:
            continue
        if np.isnan(bar.atr_t) or bar.atr_t <= 0:
            continue

        direction = 0
        if bar.sig_long:
            direction = 1
        elif cfg.allow_shorts and bar.sig_short:
            direction = -1
        if direction == 0:
            continue

        # cooldown: block same-direction re-entry after a stop-out.
        # "for 12 bars" = the next 12 bars are ALL blocked; entry OK on the 13th.
        if bar_i <= cooldown_until and direction == cooldown_dir:
            continue

        entry = bar.open + direction * half_spread
        init_stop = entry - direction * cfg.initial_stop_atr * bar.atr_t
        pos = {
            "dir": direction,
            "entry": entry,
            "entry_ts": bar.Index,
            "stop": init_stop,
            "init_stop": init_stop,
            "bars": 1,
            "adx": float(bar.adx_t) if not np.isnan(bar.adx_t) else 0.0,
            "atr": float(bar.atr_t),
        }

    stats = summarize_b(trades, df, pair, cfg)
    stats["exec_tf"] = "H4" if same_tf else "H1"
    stats["bars_per_day"] = bars_per_day
    return trades, stats


def summarize_b(trades: list[Trade], df: pd.DataFrame, pair: str, cfg: EngineBConfig) -> dict:
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

    span_years = (df.index[-1] - df.index[0]).days / 365.25
    tpy = len(trades) / span_years if span_years > 0 else 0.0
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(tpy))
        if len(r) > 1 and r.std(ddof=1) > 0
        else 0.0
    )

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        "pair": pair,
        "engine": "B",
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(r)),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float(abs(wins.mean() / losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0 else float("inf"),
        "expectancy_r": float(r.mean()),
        "total_return_pct": float(equity[-1] - 1),
        "profit_factor": float(pf),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trades_per_year": float(tpy),
        "span_years": float(span_years),
        "exit_mix": reasons,
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
        "costs_on": cfg.costs_on,
    }


def print_report_b(stats: dict) -> None:
    if stats.get("n_trades", 0) == 0:
        print(f"\n=== ENGINE B | {stats['pair']} === no trades triggered.\n")
        return
    s = stats
    mix = s["exit_mix"]
    n = s["n_trades"]
    mix_str = ", ".join(f"{k} {v/n:.0%}" for k, v in sorted(mix.items()))
    print(f"\n=== ENGINE B (range reversion) | {s['pair']} | "
          f"entries {s.get('exec_tf','H1')} | {s['span_years']:.1f}y | "
          f"costs {'ON' if s['costs_on'] else 'OFF'} ===")
    print(f"Trades:            {n}  ({s['trades_per_year']:.0f}/yr)")
    print(f"Win rate:          {s['win_rate']:.1%}   (spec expects 55-70%)")
    print(f"Avg win:           {s['avg_win_r']:+.2f}R")
    print(f"Avg loss:          {s['avg_loss_r']:+.2f}R")
    print(f"Payoff ratio:      {s['payoff_ratio']:.2f}   (spec expects < 1)")
    print(f"Expectancy:        {s['expectancy_r']:+.3f}R per trade")
    print(f"Profit factor:     {s['profit_factor']:.2f}   (kill threshold 1.05)")
    print(f"Max drawdown:      {s['max_dd']:.1%}")
    print(f"Sharpe (approx):   {s['sharpe']:.2f}")
    print(f"Exit mix:          {mix_str}")
    print(f"Avg hold:          {s['avg_bars_held']:.0f} bars "
          f"({s['avg_bars_held']/s.get('bars_per_day',24):.1f} days)")
    flags = []
    if n < 200:
        flags.append(f"SAMPLE TOO SMALL ({n} < 200 per §10.6)")
    if s["profit_factor"] < 1.05:
        flags.append("PF below §11 kill threshold")
    if s["max_dd"] > 0.20:
        flags.append("max DD above the §11 20% line")
    print("\n" + ("\n".join("  !! " + f for f in flags) if flags
                  else "  no §10/§11 flags on this run"))
    print("  (in-sample, single pair — diagnosis only, NOT a headline metric)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Engine B backtest")
    ap.add_argument("--pair", default="EUR_USD")
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--bb", type=int, default=20)
    ap.add_argument("--bb-std", type=float, default=2.0)
    ap.add_argument("--rsi-band", type=float, default=10.0,
                    help="entry bands: long < band, short > 100-band")
    ap.add_argument("--stop", type=float, default=1.5)
    ap.add_argument("--time-stop", type=int, default=36)
    ap.add_argument("--no-costs", action="store_true")
    ap.add_argument("--no-shorts", action="store_true")
    args = ap.parse_args()

    from data import get_history

    cfg = EngineBConfig(
        bb_period=args.bb,
        bb_std=args.bb_std,
        rsi_low=args.rsi_band,
        rsi_high=100.0 - args.rsi_band,
        initial_stop_atr=args.stop,
        time_stop_bars=args.time_stop,
        allow_shorts=not args.no_shorts,
        costs_on=not args.no_costs,
    )

    h4 = get_history(args.pair, "H4", years=args.years)
    h1 = get_history(args.pair, "H1", years=args.years)

    trades, stats = run_engine_b(h1, h4, pair=args.pair, cfg=cfg)
    print_report_b(stats)

    db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "memory.db",
    )
    log_trial(
        db, stats,
        {**asdict(cfg), "pair": args.pair, "years": args.years, "engine": "B"},
        period=f"{h1.index[0]:%Y-%m-%d}..{h1.index[-1]:%Y-%m-%d}", oos=0,
    )
    print(f"  logged to trials table in {db}")


if __name__ == "__main__":
    main()