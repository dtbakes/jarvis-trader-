"""
signal_report.py — Engine D paper tracking. Reports only; it never places orders.

At each month-end, Engine D decides new positions. This module recomputes
them from fresh daily candles and FRED rates with exactly the backtest's
rules (engine_d.target_weights), then:
    * logs them to the engine_d_signals table in data/memory.db
    * sends a plain-English summary to Telegram
    * reports out-of-sample performance since TRACKING_START, i.e. the
      backtest rules applied to data that didn't exist when they were fixed

Each month is reported once. main.py checks every weekday evening; a month
is reported as soon as its last daily bar has closed.

Usage:
    python -m core.signal_report            # preview the latest positions, no log/send
    python -m core.signal_report --send     # log + send if a month-end is due
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
# engine_d and its siblings import each other as top-level modules.
sys.path.append(os.path.join(_ROOT, "core"))

import numpy as np
import pandas as pd

import engine_d as D

# Engine D's rules were fixed and backtested on 2026-10-26. Positions decided
# at the October 2026 month-end are the first the backtest never saw, so
# returns after this date are out-of-sample.
TRACKING_START = pd.Timestamp("2026-10-31", tz="UTC")
HISTORY_YEARS = 3          # enough for the 252-day momentum + 63-day vol windows
DB_PATH = os.path.join(_ROOT, "data", "memory.db")


# --------------------------------------------------------------------------- #
# Month-end detection
# --------------------------------------------------------------------------- #

def _month(ts: pd.Timestamp) -> str:
    return f"{ts:%Y-%m}"


def is_last_bar_of_month(ts: pd.Timestamp) -> bool:
    """
    True if no later daily bar can fall in the same calendar month, using the
    backtest's convention (bars grouped by their index timestamp). OANDA
    daily bars open Sun-Thu evenings UTC, so a Thursday bar is followed by
    Sunday's.
    """
    nxt = ts + pd.Timedelta(days=3 if ts.weekday() == 3 else 1)
    return nxt.month != ts.month


def due_month(index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """
    The latest month-end bar whose month is complete: either a bar from a
    later month exists, or the last bar is the month's final one.
    """
    ends = D.month_end_days(index)
    last = index[-1]
    if is_last_bar_of_month(last):
        return ends[-1]
    return ends[-2] if len(ends) >= 2 else None


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def position_details(px: pd.DataFrame, rates: pd.DataFrame, day: pd.Timestamp,
                     cfg: D.EngineDConfig = D.EngineDConfig()) -> pd.DataFrame:
    """Weight and the inputs behind it, per currency, at one rebalance day."""
    weights = D.target_weights(px, rates, cfg)
    if day not in weights.index:
        raise ValueError(f"No Engine D decision on {day:%Y-%m-%d} (not enough history?)")
    hist = px.loc[:day]
    mom = np.sign(np.log(hist.iloc[-1]) - np.log(hist.iloc[-1 - cfg.mom_lookback]))
    ret_12m = hist.iloc[-1] / hist.iloc[-1 - cfg.mom_lookback] - 1
    vol = np.log(hist).diff().iloc[-cfg.vol_lookback:].std() * np.sqrt(252)
    rates_m = rates.copy()
    rates_m.index = rates_m.index.to_period("M")
    period = pd.Timestamp(day).tz_localize(None).to_period("M") - cfg.rate_lag_months
    r = rates_m.loc[:period].ffill().iloc[-1]
    return pd.DataFrame({
        "weight": weights.loc[day],
        "trend_12m": ret_12m,
        "mom": mom,
        "rate": r[px.columns],
        "rate_vs_usd": r[px.columns] - r["USD"],
        "vol": vol,
    })


def explain(ccy: str, row: pd.Series) -> str:
    """One plain-English line per currency."""
    w = row["weight"]
    trend = "up" if row["mom"] > 0 else "down" if row["mom"] < 0 else "flat"
    gap = row["rate_vs_usd"]
    rate = (f"pays {gap:.2f} pts more than USD" if gap > 0
            else f"pays {abs(gap):.2f} pts less than USD" if gap < 0
            else "pays the same as USD")
    why = f"12-month trend {trend} ({row['trend_12m']:+.1%}), {rate}"
    if abs(w) < 1e-12:
        return f"• {ccy}: flat. {why}, so the two signals cancel."
    side = "LONG" if w > 0 else "SHORT"
    size = "full" if (row["mom"] != 0 and np.sign(row["mom"]) == np.sign(gap)) else "half"
    return f"• {ccy}: {side} {abs(w):.2f}x ({size} signal). {why}."


def tracking_line(pairs: dict, rates: pd.DataFrame) -> str:
    daily, _, _ = D.run_engine_d(pairs, rates)
    live = daily[daily.index > TRACKING_START]
    if live.empty:
        return "Out-of-sample tracking starts with the October 2026 positions."
    ret = (1 + live).prod() - 1
    months = len(live.groupby([live.index.year, live.index.month]))
    return (f"Out-of-sample since the October 2026 positions: {ret:+.2%} over "
            f"{months} month(s). The backtest expects about +1.3% a year with "
            f"25% drawdowns, so read nothing into a few months.")


def build_report(pairs: dict, rates: pd.DataFrame, day: pd.Timestamp,
                 prev: pd.Series | None = None) -> tuple[str, pd.DataFrame]:
    px = D.currency_prices(pairs)
    details = position_details(px, rates, day)
    lines = [f"📊 Engine D positions for {day:%B %Y} (paper only, no orders placed)", ""]
    for ccy, row in details.iterrows():
        line = explain(ccy, row)
        if prev is not None and ccy in prev.index and abs(prev[ccy] - row["weight"]) > 0.05:
            line += f" (was {prev[ccy]:+.2f}x)"
        lines.append(line)
    gross = details["weight"].abs().sum()
    lines += ["", f"Gross exposure {gross:.2f}x of the account, all against USD.",
              tracking_line(pairs, rates)]
    return "\n".join(lines), details


# --------------------------------------------------------------------------- #
# Log + send
# --------------------------------------------------------------------------- #

def _db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS engine_d_signals (
            month TEXT, as_of TEXT, currency TEXT, weight REAL, trend_12m REAL,
            rate_vs_usd REAL, vol REAL, created_at TEXT,
            PRIMARY KEY (month, currency)
        )
    """)
    return conn


def already_reported(month: str, path: str = DB_PATH) -> bool:
    with _db(path) as conn:
        return conn.execute("SELECT 1 FROM engine_d_signals WHERE month = ? LIMIT 1",
                            (month,)).fetchone() is not None


def previous_weights(month: str, path: str = DB_PATH) -> pd.Series | None:
    with _db(path) as conn:
        rows = conn.execute(
            "SELECT currency, weight FROM engine_d_signals WHERE month = "
            "(SELECT MAX(month) FROM engine_d_signals WHERE month < ?)", (month,)).fetchall()
    return pd.Series(dict(rows)) if rows else None


def log_signals(month: str, day: pd.Timestamp, details: pd.DataFrame,
                path: str = DB_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db(path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO engine_d_signals VALUES (?,?,?,?,?,?,?,?)",
            [(month, day.isoformat(), ccy, float(r["weight"]), float(r["trend_12m"]),
              float(r["rate_vs_usd"]), float(r["vol"]), now)
             for ccy, r in details.iterrows()],
        )


def load_inputs(refresh: bool = True):
    from data import get_history
    pairs = {pair: get_history(pair, "D", years=HISTORY_YEARS, refresh=refresh,
                               verbose=False)["close"]
             for pair, _ in D.LEGS.values()}
    rates = D.fetch_rates(refresh=refresh, verbose=False)
    D.check_rates_fresh(rates, min(s.index[-1] for s in pairs.values()))
    return pairs, rates


def run_monthly(send, load=load_inputs, path: str = DB_PATH) -> str | None:
    """
    Report the latest completed month once. Returns the text it sent, or
    None if nothing was due. Never places orders.
    """
    pairs, rates = load()
    px = D.currency_prices(pairs)
    day = due_month(px.index)
    if day is None:
        return None
    month = _month(day)
    if already_reported(month, path):
        return None
    text, details = build_report(pairs, rates, day, previous_weights(month, path))
    log_signals(month, day, details, path)
    send(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Engine D monthly signal report (no orders)")
    ap.add_argument("--send", action="store_true",
                    help="log and send to Telegram if a month-end is due")
    args = ap.parse_args()

    if args.send:
        from notifications.telegram import send_message
        text = run_monthly(lambda t: send_message(t, markdown=False))
        print(text or "Nothing due: this month's report was already sent, or the month isn't over.")
        return

    pairs, rates = load_inputs()
    px = D.currency_prices(pairs)
    text, _ = build_report(pairs, rates, D.month_end_days(px.index)[-1])
    print("PREVIEW (positions if the month ended at the latest close; nothing logged or sent)\n")
    print(text)


if __name__ == "__main__":
    main()
