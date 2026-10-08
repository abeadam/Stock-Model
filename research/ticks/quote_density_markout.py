"""Does the fuller quote feed change the adverse-selection verdict, on one session?

The markout study ran on quotes fetched with ignoreSize=True, which returned only ticks
where a PRICE changed. On 2026-09-29 that is 196,338 ticks over 79% of the day against
2,350,320 over 100% of it. Two things in that study depend directly on quote density:

  * `fresh_seconds` drops a markout endpoint whose quote is stale, because a carried-
    forward mid understates the move. With a fifth of the ticks, far more endpoints were
    dropped, and they were dropped non-randomly -- a second with no price change is a
    quiet second, so the surviving sample was tilted toward moves.
  * the fill simulation consumes size resting at the limit, and size-only updates were
    exactly what was missing. Depletion was invisible.

Both arms run the same code on the same session with the same seed. The only difference
is which quote file is read, so any change in the decomposition is the data.

usage: quote_density_markout.py [DECISIONS_PER_SIDE]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from markout import (BUY, MARKOUT_HORIZONS, SELL, attempt, drift_dollars, fresh_seconds,
                     markout_curve)
from passive_fill import Book
from tick_probe import TICKS_DIR

SESSION = "2026-09-29"
THIN_DIR = f"{TICKS_DIR}/thinned_bidask"
DECISIONS_PER_SIDE = 3000
ENTRY_LIMIT_SECONDS = 30.0
SEED = 11


def load_quotes(path: str) -> pd.DataFrame:
    quotes = pd.read_csv(path)
    quotes = quotes[(quotes["Bid"] > 0) & (quotes["Ask"] >= quotes["Bid"])]
    return quotes.sort_values("Date", kind="stable")


def second_grid(quotes: pd.DataFrame) -> pd.DataFrame:
    """A second-indexed grid carrying quote_is_fresh, built as tick_probe.load_day builds it.

    A second is fresh when it carried a quote of its own rather than a carried-forward
    one. That is the whole point of the comparison: with size-only updates dropped, a
    second in which only size moved looked stale.
    """
    quotes = quotes.assign(second=quotes["Date"].astype(int),
                           mid=(quotes["Bid"] + quotes["Ask"]) / 2)
    grid = quotes.groupby("second").last()
    grid = grid.reindex(range(grid.index.min(), grid.index.max() + 1))
    grid["quote_is_fresh"] = grid["mid"].notna()
    return grid.ffill()


def arm(label: str, quote_path: str, decisions_per_side: int) -> dict:
    trades = pd.read_csv(f"{TICKS_DIR}/{SESSION}_ES_TRADES.txt").sort_values("Date", kind="stable")
    quotes = load_quotes(quote_path)
    book = Book(trades, quotes)
    grid = second_grid(quotes)
    fresh = fresh_seconds(grid)
    candidates = np.array(sorted(fresh))
    generator = np.random.default_rng(SEED)

    rows, attempts = [], 0
    for side in (BUY, SELL):
        chosen = generator.choice(candidates, size=min(decisions_per_side, len(candidates)),
                                  replace=False)
        for moment in chosen:
            attempts += 1
            outcome = attempt(book, float(moment), side, ENTRY_LIMIT_SECONDS)
            if outcome is None:
                continue
            fill_time, fill_price = outcome
            curve = markout_curve(book, fresh, fill_time, fill_price, side)
            if curve is not None:
                rows.append(curve)
    fills = pd.DataFrame(rows)
    return {"label": label, "ticks": len(quotes), "fresh": len(fresh),
            "attempts": attempts, "fills": fills}


def report(result: dict) -> None:
    fills = result["fills"]
    print(f"\n{result['label']}: {result['ticks']:,} quote ticks, {result['fresh']:,} fresh "
          f"seconds, {len(fills):,} fills of {result['attempts']:,} attempts "
          f"({len(fills) / result['attempts']:.1%})")
    if fills.empty:
        return
    print(f"  {'horizon':<9}{'n':>8}{'capture':>10}{'markout':>10}{'adverse':>10}")
    for horizon in MARKOUT_HORIZONS:
        column = f"markout_{horizon}"
        if column not in fills:
            continue
        rows = fills[fills[column].notna()]
        print(f"  {f'{horizon}s':<9}{len(rows):>8,}{rows['capture'].mean():>10.2f}"
              f"{rows[column].mean():>10.2f}{rows[f'adverse_{horizon}'].mean():>10.2f}")


def main(decisions_per_side: int = DECISIONS_PER_SIDE) -> None:
    arms = [arm("thin  (ignoreSize=True)", f"{THIN_DIR}/{SESSION}_ES_BIDASK.txt", decisions_per_side),
            arm("full  (ignoreSize=False)", f"{TICKS_DIR}/{SESSION}_ES_BIDASK.txt", decisions_per_side)]
    for result in arms:
        report(result)

    thin, full = (result["fills"] for result in arms)
    if thin.empty or full.empty:
        return
    print(f"\ndifference, full minus thin, $ per fill")
    print(f"  {'horizon':<9}{'markout':>10}{'adverse':>10}")
    for horizon in MARKOUT_HORIZONS:
        column = f"markout_{horizon}"
        if column in thin and column in full:
            print(f"  {f'{horizon}s':<9}{full[column].mean() - thin[column].mean():>10.2f}"
                  f"{full[f'adverse_{horizon}'].mean() - thin[f'adverse_{horizon}'].mean():>10.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DECISIONS_PER_SIDE)
