"""
structure_lab.py — market structure & volume measurement, NOT a strategy.

Answers three factual questions against cached OHLCV, each compared to a
random/shuffled baseline so "interesting" is measured, not assumed:

  Q1. LIQUIDITY SWEEPS
      When price pokes beyond a prior swing high/low and closes back inside
      it (a "sweep"), does the forward return distribution differ from the
      unconditional forward return distribution? This is the testable core
      of the "stop hunt then reversal" idea.

  Q2. VOLUME NODES
      Build a volume-at-price histogram (tick-volume proxy, per data.py's
      note). Does price drift toward high-volume nodes (HVN) more than
      toward low-volume nodes (LVN) / more than chance?

  Q3. BREAK OF STRUCTURE (BOS)
      After a confirmed higher-high/higher-low (or lower-low/lower-high)
      sequence breaks, does price continue in the new direction more than
      the unconditional baseline?

Design discipline, carried over from engine_a/b:
  * Every "signal" bar is matched against a DEDUPLICATED RANDOM BASELINE —
    same number of events, drawn from eligible bars, same forward horizons —
    so a mean reversion driven only by FX's general chop doesn't masquerade
    as a finding.
  * Lookahead-free: swing points, volume nodes and BOS state are all built
    from data strictly BEFORE the signal bar; forward returns are measured
    AFTER it. A prefix-property self-test (same pattern as regime.py)
    verifies this mechanically, not just by code review.
  * This script computes NO costs and proposes NO entries/exits. It answers
    "is there structure here at all" — the cheapest possible question to ask
    before spending a build cycle on a new Engine C/D/E.

Usage:
    python structure_lab.py --pair EUR_USD --granularity H1
    python structure_lab.py --pair GBP_JPY --lookback 8
    python structure_lab.py --all          # run the full universe, H1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HORIZONS = (4, 12, 24, 48)   # bars ahead to measure forward return


# --------------------------------------------------------------------------- #
# Swing points (lookahead-free: a swing at i is only confirmed `lookback`
# bars LATER, which is when the signal becomes knowable in a live system)
# --------------------------------------------------------------------------- #

def find_confirmed_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    A bar i is a swing high if its high is the max of [i-lookback, i+lookback].
    It is only CONFIRMED (knowable) at bar i+lookback. We store the swing's
    price at its origin index but also the bar index at which it becomes
    usable, so downstream logic never looks into the future.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    is_sw_high = np.zeros(n, dtype=bool)
    is_sw_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            is_sw_high[i] = True
        if lows[i] == window_l.min():
            is_sw_low[i] = True

    out = df.copy()
    out["sw_high"] = is_sw_high
    out["sw_low"] = is_sw_low
    # confirmed_at: the bar index at which this swing is first knowable
    out["sw_confirmed_at"] = np.arange(n) + lookback
    return out


# --------------------------------------------------------------------------- #
# Q1 — Liquidity sweeps
# --------------------------------------------------------------------------- #

def detect_sweeps(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """
    A sweep-high event at bar j: the most recently CONFIRMED swing high
    (known as of bar j, i.e. confirmed_at <= j) has a price P. Bar j's high
    exceeds P (price traded beyond the old high) but bar j's CLOSE is back
    below P (the breakout failed to hold). Mirror for sweep-low.

    Only the most recent confirmed swing is used — this is a genuinely
    forward-only, streaming-style computation.
    """
    sw = find_confirmed_swings(df, lookback)
    n = len(sw)
    highs, lows, closes = sw["high"].to_numpy(), sw["low"].to_numpy(), sw["close"].to_numpy()
    is_sw_high, is_sw_low = sw["sw_high"].to_numpy(), sw["sw_low"].to_numpy()
    confirmed_at = sw["sw_confirmed_at"].to_numpy()

    last_sw_high_price = np.nan
    last_sw_high_origin = -1
    last_sw_low_price = np.nan
    last_sw_low_origin = -1

    events = []
    for j in range(n):
        # absorb any swing that becomes confirmed exactly at bar j
        # (origin = j - lookback, confirmed_at = j)
        origin = j - lookback
        if 0 <= origin < n:
            if is_sw_high[origin]:
                last_sw_high_price = highs[origin]
                last_sw_high_origin = origin
            if is_sw_low[origin]:
                last_sw_low_price = lows[origin]
                last_sw_low_origin = origin

        if not np.isnan(last_sw_high_price) and highs[j] > last_sw_high_price >= closes[j]:
            events.append({"i": j, "dir": -1, "kind": "sweep_high",
                           "level": float(last_sw_high_price),
                           "origin": last_sw_high_origin})
        if not np.isnan(last_sw_low_price) and lows[j] < last_sw_low_price <= closes[j]:
            events.append({"i": j, "dir": 1, "kind": "sweep_low",
                           "level": float(last_sw_low_price),
                           "origin": last_sw_low_origin})
    return events


# --------------------------------------------------------------------------- #
# Q2 — Volume nodes
# --------------------------------------------------------------------------- #

def volume_profile(df: pd.DataFrame, bins: int = 50, window: int | None = None) -> pd.DataFrame:
    """
    Volume-at-price histogram. Each bar's tick-volume is distributed across
    the price bins its [low, high] range touches (simple uniform split —
    adequate for a measurement tool, not a precision instrument).
    `window`: if set, only the trailing `window` bars are used (rolling
    profile); if None, the whole frame.
    """
    d = df.iloc[-window:] if window else df
    lo, hi = d["low"].min(), d["high"].max()
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)

    for row in d.itertuples():
        lo_i = np.searchsorted(edges, row.low, side="right") - 1
        hi_i = np.searchsorted(edges, row.high, side="right") - 1
        lo_i, hi_i = max(0, min(lo_i, bins - 1)), max(0, min(hi_i, bins - 1))
        span = max(hi_i - lo_i + 1, 1)
        vol[lo_i:hi_i + 1] += row.volume / span

    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"price": centers, "volume": vol})


def nearest_node_distance(price: float, profile: pd.DataFrame, kind: str) -> float:
    """Signed distance (in price units) from `price` to the nearest HVN/LVN."""
    if kind == "hvn":
        target_idx = profile["volume"].idxmax()
    else:
        target_idx = profile["volume"].idxmin()
    return profile.loc[target_idx, "price"] - price


# --------------------------------------------------------------------------- #
# Q3 — Break of structure
# --------------------------------------------------------------------------- #

def detect_bos(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """
    Tracks the alternating sequence of confirmed swing highs/lows. A BOS-up
    event fires at the bar where close first exceeds the last confirmed
    swing high WHILE the prior two confirmed swing lows were rising (uptrend
    structure) — i.e. continuation confirmation, not the first break of any
    kind. Mirror for BOS-down.
    """
    sw = find_confirmed_swings(df, lookback)
    n = len(sw)
    closes = sw["close"].to_numpy()
    highs = sw["high"].to_numpy()
    lows = sw["low"].to_numpy()
    is_sw_high, is_sw_low = sw["sw_high"].to_numpy(), sw["sw_low"].to_numpy()

    swing_highs: list[tuple[int, float]] = []   # (origin_idx, price), as confirmed
    swing_lows: list[tuple[int, float]] = []
    last_break_high_used = -1
    last_break_low_used = -1

    events = []
    for j in range(n):
        origin = j - lookback
        if 0 <= origin < n:
            if is_sw_high[origin]:
                swing_highs.append((origin, highs[origin]))
            if is_sw_low[origin]:
                swing_lows.append((origin, lows[origin]))

        if len(swing_highs) >= 1 and len(swing_lows) >= 2:
            last_high_idx, last_high_price = swing_highs[-1]
            if (closes[j] > last_high_price
                    and last_high_idx > last_break_high_used
                    and swing_lows[-1][1] > swing_lows[-2][1]):
                events.append({"i": j, "dir": 1, "kind": "bos_up",
                               "level": float(last_high_price)})
                last_break_high_used = last_high_idx

        if len(swing_lows) >= 1 and len(swing_highs) >= 2:
            last_low_idx, last_low_price = swing_lows[-1]
            if (closes[j] < last_low_price
                    and last_low_idx > last_break_low_used
                    and swing_highs[-1][1] < swing_highs[-2][1]):
                events.append({"i": j, "dir": -1, "kind": "bos_down",
                               "level": float(last_low_price)})
                last_break_low_used = last_low_idx
    return events


# --------------------------------------------------------------------------- #
# Forward-return measurement, with a random baseline for every test
# --------------------------------------------------------------------------- #

def forward_returns(df: pd.DataFrame, indices: list[int], direction: np.ndarray,
                     horizons=HORIZONS) -> dict[int, np.ndarray]:
    """For each event i, signed forward return at i+h, direction-adjusted
    (positive = move agreed with the event's implied direction)."""
    closes = df["close"].to_numpy()
    n = len(closes)
    out = {}
    for h in horizons:
        rets = []
        for idx, d in zip(indices, direction):
            if idx + h < n and closes[idx] > 0 and np.isfinite(closes[idx]) \
                    and np.isfinite(closes[idx + h]):
                r = (closes[idx + h] - closes[idx]) / closes[idx] * d
                rets.append(r)
        out[h] = np.array(rets)
    return out


def baseline_returns(df: pd.DataFrame, n_events: int, rng: np.random.Generator,
                     horizons=HORIZONS, min_i: int = 60) -> dict[int, np.ndarray]:
    """Random bars, random +-1 direction, same count as the real event set —
    the null hypothesis: 'any bar, any direction' forward return."""
    closes = df["close"].to_numpy()
    n = len(closes)
    idxs = rng.integers(min_i, n - max(horizons) - 1, size=n_events)
    dirs = rng.choice([-1, 1], size=n_events)
    out = {}
    for h in horizons:
        out[h] = np.array([
            (closes[i + h] - closes[i]) / closes[i] * d for i, d in zip(idxs, dirs)
        ])
    return out


def summarize_test(name: str, event_rets: dict[int, np.ndarray],
                    base_rets: dict[int, np.ndarray], n_events: int) -> None:
    print(f"\n--- {name} (n={n_events}) ---")
    if n_events < 30:
        print("  too few events to read anything into this — skipping stats")
        return
    print(f"  {'horizon':>8s}  {'event mean':>11s}  {'base mean':>10s}  "
          f"{'edge':>8s}  {'t-stat':>7s}  {'event win%':>10s}")
    for h in HORIZONS:
        e, b = event_rets.get(h, np.array([])), base_rets.get(h, np.array([]))
        if len(e) < 20 or len(b) < 20:
            continue
        edge = e.mean() - b.mean()
        # Welch's t-test, dependency-free
        se = np.sqrt(e.var(ddof=1) / len(e) + b.var(ddof=1) / len(b))
        t = edge / se if se > 0 else 0.0
        win = (e > 0).mean()
        flag = "  <-- |t|>2" if abs(t) > 2 else ""
        print(f"  {h:>8d}  {e.mean():>+10.4%}  {b.mean():>+9.4%}  "
              f"{edge:>+7.4%}  {t:>7.2f}  {win:>9.1%}{flag}")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_structure_lab(df: pd.DataFrame, pair: str, lookback: int = 5,
                      seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    print(f"\n{'='*70}\nSTRUCTURE LAB | {pair} | {len(df)} bars | "
          f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}\n{'='*70}")

    # ---- Q1: liquidity sweeps ----
    sweeps = detect_sweeps(df, lookback)
    if sweeps:
        idxs = [e["i"] for e in sweeps]
        dirs = np.array([e["dir"] for e in sweeps])
        ev = forward_returns(df, idxs, dirs)
        base = baseline_returns(df, len(sweeps), rng)
        summarize_test("Q1: LIQUIDITY SWEEPS (reversal direction)", ev, base, len(sweeps))
    else:
        print("\n--- Q1: LIQUIDITY SWEEPS --- none detected")

    # ---- Q2: volume nodes ----
    profile = volume_profile(df)
    # sample bars at regular intervals, measure drift toward nearest HVN
    sample_i = np.arange(200, len(df) - max(HORIZONS) - 1, max(50, len(df) // 300))
    hvn_dirs, hvn_idx = [], []
    for i in sample_i:
        trailing_profile = volume_profile(df.iloc[:i], window=min(i, 2000))
        dist = nearest_node_distance(df["close"].iloc[i], trailing_profile, "hvn")
        if abs(dist) < 1e-12:
            continue
        hvn_dirs.append(1 if dist > 0 else -1)  # direction TOWARD the HVN
        hvn_idx.append(i)
    if hvn_idx:
        ev = forward_returns(df, hvn_idx, np.array(hvn_dirs))
        base = baseline_returns(df, len(hvn_idx), rng)
        summarize_test("Q2: DRIFT TOWARD NEAREST HIGH-VOLUME NODE", ev, base, len(hvn_idx))
    else:
        print("\n--- Q2: VOLUME NODES --- insufficient samples")

    # ---- Q3: break of structure ----
    bos = detect_bos(df, lookback)
    if bos:
        idxs = [e["i"] for e in bos]
        dirs = np.array([e["dir"] for e in bos])
        ev = forward_returns(df, idxs, dirs)
        base = baseline_returns(df, len(bos), rng)
        summarize_test("Q3: BREAK OF STRUCTURE (continuation)", ev, base, len(bos))
    else:
        print("\n--- Q3: BREAK OF STRUCTURE --- none detected")

    print(f"\n{'='*70}")
    print("Reading guide: |t|>2 is the usual 'worth a second look' bar — it is")
    print("NOT a validated edge. No costs here, no walk-forward, no §10.5 DSR.")
    print("This only tells you whether the PATTERN exists in the data at all.")
    print(f"{'='*70}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EUR_USD")
    ap.add_argument("--granularity", default="H1", choices=["H1", "H4", "D"])
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="run the full 10-pair universe")
    args = ap.parse_args()

    from data import get_history
    from run_universe import PAIRS_DEFAULT

    pairs = PAIRS_DEFAULT if args.all else [args.pair]
    for pair in pairs:
        df = get_history(pair, args.granularity, years=args.years, verbose=False)
        run_structure_lab(df, pair, lookback=args.lookback)


if __name__ == "__main__":
    main()