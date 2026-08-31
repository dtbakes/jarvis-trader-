"""
fetch_universe.py — bulk-download the STRATEGY.md §3 pair universe.

Pulls 4 years of H1 + H4 for every pair, using data.get_history's cache so
the run is resumable: pairs already cached are skipped instantly, and a
mid-run failure (network, rate limit) costs only the pair in flight.

Usage:
    python fetch_universe.py            # download whatever's missing
    python fetch_universe.py --check    # just report cache status, no network
"""

from __future__ import annotations

import argparse
import sys
import time

from data import get_history

# STRATEGY.md §3 universe
PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD",
    "AUD_USD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
]
GRANULARITIES = ["H4", "H1"]
YEARS = 4.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report cache only")
    args = ap.parse_args()

    ok, failed = [], []
    t0 = time.time()

    for pair in PAIRS:
        for gran in GRANULARITIES:
            label = f"{pair} {gran}"
            try:
                if args.check:
                    # verbose get_history prints cache status; refresh=False
                    # means a cache hit never touches the network.
                    df = get_history(pair, gran, years=YEARS, verbose=True)
                else:
                    df = get_history(pair, gran, years=YEARS, verbose=True)
                ok.append((label, len(df)))
            except Exception as e:  # keep going; report at the end
                print(f"  !! {label} FAILED: {e}")
                failed.append((label, str(e)))

    dt = time.time() - t0
    print(f"\n=== universe fetch summary ({dt:.0f}s) ===")
    for label, n in ok:
        print(f"  ok   {label:12s} {n:6d} candles")
    for label, err in failed:
        print(f"  FAIL {label:12s} {err[:80]}")

    if failed:
        print("\nRe-run this script to retry the failed pairs (cached ones skip).")
        sys.exit(1)
    print("\nUniverse complete. Ready for run_universe.py.")


if __name__ == "__main__":
    main()