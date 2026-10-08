"""Does the unstable sort actually change any fill, or only reorder rows?

The defect is established: IBKR truncates tick times to whole seconds but preserves
sequence, and pandas' default quicksort is unstable, so `sort_values("Date")` permuted
1,343,654 of 1,360,157 rows. That is a demonstration of REORDERING. It is not a
demonstration that anything downstream changes, which was inferred from the mechanism:
`passive_fill` consumes size resting at a price oldest-first and fills on a print
through the level, so whether a trade-through lands before or after the queue is
consumed is decided by order within the second.

This runs the identical study twice on one session, same seed, same file, differing only
in `kind=`, and compares the fill sets row by row. If they match, the fix bought
certainty rather than a correction, and that is worth knowing plainly.

usage: sort_stability_impact.py [DECISIONS_PER_SIDE]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from markout import BUY, MARKOUT_HORIZONS, SELL, attempt, fresh_seconds, markout_curve
from passive_fill import Book
from tick_probe import TICKS_DIR

SESSION = "2026-09-29"
DECISIONS_PER_SIDE = 3000
ENTRY_LIMIT_SECONDS = 30.0
SEED = 11


def load(kind: str) -> tuple[Book, set[int]]:
    trades = pd.read_csv(f"{TICKS_DIR}/{SESSION}_ES_TRADES.txt").sort_values("Date", kind=kind)
    quotes = pd.read_csv(f"{TICKS_DIR}/{SESSION}_ES_BIDASK.txt")
    quotes = quotes[(quotes["Bid"] > 0) & (quotes["Ask"] >= quotes["Bid"])].sort_values("Date", kind=kind)
    marked = quotes.assign(second=quotes["Date"].astype(int),
                           mid=(quotes["Bid"] + quotes["Ask"]) / 2)
    grid = marked.groupby("second").last()
    grid = grid.reindex(range(grid.index.min(), grid.index.max() + 1))
    grid["quote_is_fresh"] = grid["mid"].notna()
    return Book(trades, quotes), fresh_seconds(grid.ffill())


def arm(kind: str, decisions_per_side: int) -> pd.DataFrame:
    book, fresh = load(kind)
    candidates = np.array(sorted(fresh))
    generator = np.random.default_rng(SEED)
    rows = []
    for side in (BUY, SELL):
        chosen = generator.choice(candidates, size=min(decisions_per_side, len(candidates)),
                                  replace=False)
        for moment in chosen:
            outcome = attempt(book, float(moment), side, ENTRY_LIMIT_SECONDS)
            if outcome is None:
                rows.append({"moment": int(moment), "side": side, "filled": False,
                             "fill_time": np.nan, "fill_price": np.nan})
                continue
            fill_time, fill_price = outcome
            curve = markout_curve(book, fresh, fill_time, fill_price, side) or {}
            rows.append({"moment": int(moment), "side": side, "filled": True,
                         "fill_time": fill_time, "fill_price": fill_price, **curve})
    return pd.DataFrame(rows).set_index(["moment", "side"]).sort_index()


def main(decisions_per_side: int = DECISIONS_PER_SIDE) -> None:
    stable, quick = arm("stable", decisions_per_side), arm("quicksort", decisions_per_side)
    print(f"{len(stable):,} decisions per arm on {SESSION}\n")

    both_filled = stable["filled"] & quick["filled"]
    print(f"fill decision differs: {(stable['filled'] != quick['filled']).sum():,} of {len(stable):,}")
    print(f"  stable fills {stable['filled'].sum():,}, quicksort fills {quick['filled'].sum():,}")
    for column in ("fill_time", "fill_price"):
        differ = (stable.loc[both_filled, column] != quick.loc[both_filled, column]).sum()
        print(f"{column} differs: {differ:,} of {both_filled.sum():,} commonly filled")

    print(f"\n  {'horizon':<9}{'stable':>10}{'quicksort':>11}{'difference':>12}")
    for horizon in MARKOUT_HORIZONS:
        column = f"markout_{horizon}"
        if column in stable and column in quick:
            left, right = stable[column].mean(), quick[column].mean()
            print(f"  {f'{horizon}s':<9}{left:>10.3f}{right:>11.3f}{right - left:>12.3f}")

    identical = stable.equals(quick)
    print(f"\nthe two arms are {'IDENTICAL' if identical else 'DIFFERENT'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DECISIONS_PER_SIDE)
