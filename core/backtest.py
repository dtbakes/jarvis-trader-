"""
Backtester — runs a rule set against historical candles and reports the stats that matter.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.oanda import get_candles

# ---------- Indicators ----------

def compute_rsi(closes, period=14):
    """Standard RSI. Returns a list aligned with closes (None until enough data)."""
    rsis = [None] * len(closes)
    gains, losses = 0.0, 0.0

    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period

    def rsi_from(avg_g, avg_l):
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))

    rsis[period] = rsi_from(avg_gain, avg_loss)

    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsis[i] = rsi_from(avg_gain, avg_loss)

    return rsis

# ---------- The backtest engine ----------

def backtest_rsi_reversion(
    instrument="EUR_USD",
    granularity="H1",
    candle_count=5000,
    rsi_buy_below=30,
    rsi_exit_above=50,
    stop_loss_pct=0.005,
    spread_cost_pct=0.00013,
):
    """
    Rule set:
      ENTER long when RSI crosses below rsi_buy_below.
      EXIT when RSI rises above rsi_exit_above, or stop loss (-0.5%) hits.
      One position at a time. Long only (keep v1 simple).
    """
    candles = get_candles(instrument, count=candle_count, granularity=granularity)
    closes = [c["close"] for c in candles]
    lows = [c["low"] for c in candles]
    rsis = compute_rsi(closes)

    trades = []
    in_position = False
    entry_price = 0.0
    stop_price = 0.0

    for i in range(len(candles)):
        if rsis[i] is None:
            continue

        if not in_position:
            if rsis[i] < rsi_buy_below:
                in_position = True
                entry_price = closes[i]
                stop_price = entry_price * (1 - stop_loss_pct)
        else:
            # Stop loss check first (uses candle low — did price touch the stop?)
            if lows[i] <= stop_price:
                pnl_pct = (stop_price - entry_price) / entry_price
                trades.append(pnl_pct - spread_cost_pct)
                in_position = False
            elif rsis[i] > rsi_exit_above:
                pnl_pct = (closes[i] - entry_price) / entry_price
                trades.append(pnl_pct - spread_cost_pct)
                in_position = False

    return report(instrument, granularity, len(candles), trades)

# ---------- Stats ----------

def report(instrument, granularity, n_candles, trades):
    if not trades:
        print("No trades triggered. Rules never fired.")
        return None

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Equity curve + max drawdown
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= (1 + t)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    print(f"=== BACKTEST: RSI mean reversion | {instrument} {granularity} | {n_candles} candles ===")
    print(f"Total trades:    {len(trades)}")
    print(f"Win rate:        {win_rate:.1%}")
    print(f"Avg win:         {avg_win:+.3%}")
    print(f"Avg loss:        {avg_loss:+.3%}")
    print(f"Expectancy:      {expectancy:+.4%} per trade")
    print(f"Total return:    {(equity - 1):+.2%} (compounded, full period)")
    print(f"Max drawdown:    {max_dd:.2%}")
    verdict = "POSITIVE EDGE (on this data)" if expectancy > 0 else "NO EDGE — loses over time"
    print(f"Verdict:         {verdict}")
    return {"trades": len(trades), "win_rate": win_rate, "expectancy": expectancy, "max_dd": max_dd}


if __name__ == "__main__":
    backtest_rsi_reversion()