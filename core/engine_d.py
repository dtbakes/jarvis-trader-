"""
engine_d.py — Engine D: FX momentum + carry portfolio. ONE pre-registered run.

Why this one
------------
Engines A, B, C and the structure lab all tried to time individual entries
and found no edge after costs. Time-series momentum and carry are different:
they hold positions for weeks to months and rank whole currencies. They are
the two best-documented return sources in FX (Moskowitz/Ooi/Pedersen 2012
on time-series momentum; Lustig/Roussanov/Verdelhan 2011 and Koijen et al.
2018 on carry). Expect modest results, not a money printer.

Rules (fixed BEFORE the first run — no tuning afterwards)
--------------------------------------------------------
Universe : EUR, GBP, JPY, CHF, CAD, AUD, NZD, each traded against USD
           through the 7 USD majors.
Rebalance: last trading day of each month, at that day's close.
Momentum : sign of the currency's trailing 252-day return vs USD.
Carry    : sign of (currency short rate - USD short rate), using the rate
           for the month BEFORE the rebalance month (publication lag).
Signal   : 0.5 x momentum sign + 0.5 x carry sign  ->  -1, -0.5, 0, +0.5, +1.
Sizing   : each leg targets 10% / sqrt(7) annualized vol, using 63-day
           realized vol, capped at 1.0x notional per leg.
Carry P&L: the rate differential accrues daily on what is held.
Costs    : half the spread on every change in position, plus a 0.5% a year
           broker financing markup on gross exposure.
Verdict  : monthly profit factor (sum of up months / sum of down months)
           of the COMBINED rule. Under 1.05 -> archived, like A, B and C.
           Momentum-only and carry-only are printed as diagnostics only.

Rates come from FRED (OECD "immediate rates", monthly, percent), cached in
data/rates/. A series that stopped updating more than 6 months before the
end of the price data stops the run, so stale rates can't fake a result.

Usage (from core/):
    python engine_d.py                # 10y daily, prints the verdict
    python engine_d.py --refresh      # re-download rates and candles
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from engine_a import DEFAULT_SPREAD_PIPS, SPREAD_PIPS, log_trial, pip_size

# currency -> (OANDA pair, +1 if the currency is the base, -1 if it's the quote)
LEGS = {
    "EUR": ("EUR_USD", 1), "GBP": ("GBP_USD", 1), "AUD": ("AUD_USD", 1),
    "NZD": ("NZD_USD", 1), "JPY": ("USD_JPY", -1), "CHF": ("USD_CHF", -1),
    "CAD": ("USD_CAD", -1),
}

FRED_SERIES = {
    "USD": "IRSTCI01USM156N", "EUR": "IRSTCI01EZM156N", "GBP": "IRSTCI01GBM156N",
    "JPY": "IRSTCI01JPM156N", "CHF": "IRSTCI01CHM156N", "CAD": "IRSTCI01CAM156N",
    "AUD": "IRSTCI01AUM156N", "NZD": "IRSTCI01NZM156N",
}
MAX_RATE_STALENESS_DAYS = 183

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass(frozen=True)
class EngineDConfig:
    mom_lookback: int = 252
    vol_lookback: int = 63
    target_vol: float = 0.10          # whole portfolio, annualized
    max_leg_weight: float = 1.0
    w_momentum: float = 0.5
    w_carry: float = 0.5
    rate_lag_months: int = 1
    financing_markup: float = 0.005   # per year, on gross exposure
    costs_on: bool = True


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def fetch_rates(refresh: bool = False) -> pd.DataFrame:
    """Monthly short rates in percent, columns = currencies, index = month start."""
    cache = os.path.join(_DATA_DIR, "rates")
    os.makedirs(cache, exist_ok=True)
    cols = {}
    for ccy, series in FRED_SERIES.items():
        path = os.path.join(cache, f"{series}.csv")
        if refresh or not os.path.exists(path):
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
            pd.read_csv(url).to_csv(path, index=False)
        raw = pd.read_csv(path)
        # FRED has used both "DATE" and "observation_date" as the first column.
        s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(),
                      index=pd.to_datetime(raw.iloc[:, 0]))
        cols[ccy] = s.dropna()
    return pd.DataFrame(cols).sort_index()


def check_rates_fresh(rates: pd.DataFrame, price_end: pd.Timestamp) -> None:
    """Refuse to run if any series stopped well before the price data ends."""
    end = price_end.tz_localize(None) if price_end.tzinfo else price_end
    stale = {c: rates[c].last_valid_index() for c in rates.columns
             if (end - rates[c].last_valid_index()).days > MAX_RATE_STALENESS_DAYS}
    if stale:
        detail = ", ".join(f"{c} ends {d:%Y-%m}" for c, d in stale.items())
        raise ValueError(
            f"Rate data is stale ({detail}). Swap in a current FRED series for "
            f"those currencies in FRED_SERIES before trusting a result."
        )


def currency_prices(pair_closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily price of each currency in USD (inverting USD_xxx pairs)."""
    out = {}
    for ccy, (pair, side) in LEGS.items():
        s = pair_closes[pair].astype(float)
        out[ccy] = s if side == 1 else 1.0 / s
    return pd.DataFrame(out).dropna()


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #

def month_end_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month in the index."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def target_weights(px: pd.DataFrame, rates: pd.DataFrame, cfg: EngineDConfig,
                   use_mom: bool = True, use_carry: bool = True) -> pd.DataFrame:
    """
    Weights decided at each month-end close, from data up to that close only.
    Rows = rebalance days, columns = currencies (+ = long the currency vs USD).
    """
    logret = np.log(px).diff()
    mom = np.sign(np.log(px) - np.log(px.shift(cfg.mom_lookback)))
    vol = logret.rolling(cfg.vol_lookback).std() * np.sqrt(252)

    # Rate known at a rebalance in month M = the value for month M - lag.
    rates_m = rates.copy()
    rates_m.index = rates_m.index.to_period("M")
    ccys = list(px.columns)

    rows = {}
    for day in month_end_days(px.index):
        period = pd.Timestamp(day).tz_localize(None).to_period("M") - cfg.rate_lag_months
        known = rates_m.loc[:period]
        if known.empty or np.isnan(mom.loc[day]).any() or np.isnan(vol.loc[day]).any():
            continue
        r = known.ffill().iloc[-1]
        carry = np.sign(r[ccys] - r["USD"])
        sig = (cfg.w_momentum * mom.loc[day] if use_mom else 0.0) \
            + (cfg.w_carry * carry if use_carry else 0.0)
        if use_mom != use_carry:          # a single component gets the full weight
            sig = np.sign(sig)
        leg_target = cfg.target_vol / np.sqrt(len(ccys))
        w = (sig * leg_target / vol.loc[day]).clip(-cfg.max_leg_weight, cfg.max_leg_weight)
        rows[day] = w.astype(float)
    return pd.DataFrame(rows).T.reindex(columns=ccys)


def run_engine_d(pair_closes: dict[str, pd.Series], rates: pd.DataFrame,
                 cfg: EngineDConfig = EngineDConfig(),
                 use_mom: bool = True, use_carry: bool = True):
    """
    Returns (daily_returns, weights_at_rebalance, stats).
    Weights set at a month-end close earn from the next day's return on.
    """
    px = currency_prices(pair_closes)
    ccys = list(px.columns)
    simple = px.pct_change()
    w_reb = target_weights(px, rates, cfg, use_mom, use_carry)
    if w_reb.empty:
        raise ValueError("No rebalance had enough history; need > mom_lookback bars.")

    # Hold each month's weights from the day after the rebalance.
    held = w_reb.reindex(px.index).ffill().shift(1).fillna(0.0)

    port = (held * simple).sum(axis=1)

    # Real carry, always on: the rate differential accrues on calendar days
    # held (it is part of the return of holding a currency, not a cost).
    days = px.index.to_series().diff().dt.days.fillna(0).to_numpy()
    rates_d = rates.copy()
    rates_d.index = rates_d.index.to_period("M")
    month = pd.DatetimeIndex(px.index).tz_localize(None).to_period("M")
    rd = rates_d.ffill().reindex(month).ffill()
    diff = (rd[ccys].to_numpy() - rd[["USD"]].to_numpy()) / 100.0
    port = port + (held.to_numpy() * np.nan_to_num(diff)).sum(axis=1) * days / 365.0

    if cfg.costs_on:
        markup = held.abs().sum(axis=1).to_numpy() * cfg.financing_markup * days / 365.0
        # Half the spread on every change in position, as a fraction of price.
        turnover = held.diff().abs().fillna(held.abs())
        half_spread = pd.DataFrame({
            c: (SPREAD_PIPS.get(LEGS[c][0], DEFAULT_SPREAD_PIPS) * pip_size(LEGS[c][0]) / 2.0)
            / pair_closes[LEGS[c][0]].reindex(px.index).astype(float)
            for c in ccys
        })
        spread_cost = (turnover * half_spread).sum(axis=1)
        port = port - markup - spread_cost

    port = port.loc[w_reb.index[0]:].iloc[1:]   # start once the first weights are live
    return port, w_reb, summarize_d(port, len(w_reb))


def summarize_d(daily: pd.Series, n_rebalances: int) -> dict:
    monthly = (1 + daily).groupby([daily.index.year, daily.index.month]).prod() - 1
    up, down = monthly[monthly > 0].sum(), -monthly[monthly < 0].sum()
    equity = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    vol = daily.std() * np.sqrt(252)
    return {
        "n_trades": int(len(monthly)),          # months; named for log_trial
        "n_rebalances": n_rebalances,
        "profit_factor": float(up / down) if down > 0 else float("inf"),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        "ann_vol": float(vol),
        "sharpe": float(daily.mean() * 252 / vol) if vol > 0 else 0.0,
        "max_dd": float((1 - equity / equity.cummax()).max()),
        "pct_up_months": float((monthly > 0).mean()),
        "span_years": float(years),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Engine D: momentum + carry (one-shot)")
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    from data import get_history

    pairs = {pair: get_history(pair, "D", years=args.years, refresh=args.refresh,
                               verbose=False)["close"]
             for pair, _ in LEGS.values()}
    rates = fetch_rates(refresh=args.refresh)
    end = min(s.index[-1] for s in pairs.values())
    check_rates_fresh(rates, end)
    print("Rates through: " + ", ".join(
        f"{c} {rates[c].last_valid_index():%Y-%m}" for c in rates.columns))

    cfg = EngineDConfig()
    runs = {
        "COMBINED (verdict)": (True, True),
        "momentum only (diagnostic)": (True, False),
        "carry only (diagnostic)": (False, True),
    }
    results = {}
    for name, (m, c) in runs.items():
        daily, w, st = run_engine_d(pairs, rates, cfg, use_mom=m, use_carry=c)
        results[name] = st

    print(f"\n=== ENGINE D | momentum + carry | 7 currencies vs USD | "
          f"{results['COMBINED (verdict)']['span_years']:.1f}y | monthly rebalance | costs ON ===\n")
    print(pd.DataFrame({
        k: {"months": v["n_trades"], "PF (monthly)": f"{v['profit_factor']:.2f}",
            "CAGR": f"{v['cagr']:+.1%}", "vol": f"{v['ann_vol']:.1%}",
            "Sharpe": f"{v['sharpe']:.2f}", "maxDD": f"{v['max_dd']:.1%}",
            "up months": f"{v['pct_up_months']:.0%}"}
        for k, v in results.items()
    }).T.to_string())

    st = results["COMBINED (verdict)"]
    verdict = "PASSES" if st["profit_factor"] >= 1.05 else "FAILS"
    print(f"\nVerdict: combined PF {st['profit_factor']:.2f} {verdict} the 1.05 kill line.")
    print("The diagnostics don't count: picking the better half after the fact "
          "would be curve-fitting.")

    db = os.path.join(_DATA_DIR, "memory.db")
    log_trial(db, st, {**asdict(cfg), "engine": "D", "years": args.years,
                       "run": "momentum_carry_one_shot"},
              period=f"{end - pd.Timedelta(days=int(args.years * 365)):%Y-%m-%d}..{end:%Y-%m-%d}",
              oos=0)


if __name__ == "__main__":
    main()
