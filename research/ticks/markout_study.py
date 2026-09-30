"""How much of the captured half spread survives, and how much of that is being filled.

Step 1 is the markout curve: what a passive fill is worth at 1, 5, 10, 30 and 60 seconds.
Step 2 is the counterfactual it needs -- the same mid drift measured from random seconds
matched on direction and hour of day. Without it the curve cannot distinguish "the flow
that filled me knew something" from "the market happened to be drifting", and only the
first is adverse selection.

Fills come from `passive_fill.simulate_limit_order`, whose rule is deliberately
pessimistic: the size resting at your price is ahead of you, prints at your price consume
it oldest first, a print through your level fills you, and cancellations ahead are
invisible. Fill rates are therefore a floor.

Both ends of every markout must sit on a **fresh** quote. IBKR's historical BID_ASK ticks
are thinned, so a carried-forward mid understates the move.

usage: markout_study.py [SESSIONS_PER_SIDE]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from markout import (BUY, MARKOUT_HORIZONS, POINT_VALUE, SELL, attempt, drift_dollars,
                     fresh_seconds, markout_curve)
from passive_fill import Book
from tick_probe import TICKS_DIR, load_day

CLEAN_SESSIONS = ("2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23",
                  "2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29")
DECISIONS_PER_SIDE = 3000
ENTRY_LIMIT_SECONDS = 30.0
SEED = 11
RESULTS_PATH = "/Users/abeadam/dev/model/Stock-Model/scratchpad_backup/markout_study.json"


def load_book(day: str) -> Book:
    """kind="stable": the file is already chronological and IBKR strips sub-second times,
    so the ORDER OF TIES is the only record of what happened first inside a second.
    The default quicksort reorders 98.8% of rows, which randomises whether a
    trade-through print lands before or after the queue is consumed."""
    trades = pd.read_csv(f"{TICKS_DIR}/{day}_ES_TRADES.txt").sort_values("Date", kind="stable")
    quotes = pd.read_csv(f"{TICKS_DIR}/{day}_ES_BIDASK.txt")
    quotes = quotes[(quotes["Bid"] > 0) & (quotes["Ask"] >= quotes["Bid"])].sort_values("Date", kind="stable")
    return Book(trades, quotes)


def hour_of(second: int) -> int:
    return int(pd.Timestamp(second, unit="s", tz="UTC").tz_convert("America/New_York").hour)


def session_rows(day: str, decisions_per_side: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Fills with their markout curves, and matched random seconds with their drift."""
    book, grid = load_book(day), load_day(day)
    fresh = fresh_seconds(grid)
    candidates = np.array(sorted(fresh))
    generator = np.random.default_rng(seed)
    by_hour: dict[int, list[int]] = {}
    for second in candidates:
        by_hour.setdefault(hour_of(second), []).append(int(second))

    fills, matched = [], []
    for side in (BUY, SELL):
        chosen = generator.choice(candidates, size=min(decisions_per_side, len(candidates)),
                                  replace=False)
        for moment in chosen:
            outcome = attempt(book, float(moment), side, ENTRY_LIMIT_SECONDS)
            if outcome is None:
                continue
            fill_time, fill_price = outcome
            curve = markout_curve(book, fresh, fill_time, fill_price, side)
            if curve is None:
                continue
            fills.append({"day": day, "side": side, "hour": hour_of(int(fill_time)),
                          "attempted": 1, **curve})
            pool = by_hour.get(hour_of(int(fill_time)), [])
            if pool:
                drift = drift_dollars(book, fresh, int(generator.choice(pool)), side)
                if drift:
                    matched.append({"day": day, "side": side, **drift})
    return fills, matched


def summarise(frame: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(f"  {'horizon':<10}{'fills':>9}{'capture':>11}{'markout':>11}{'adverse':>11}")
    for horizon in MARKOUT_HORIZONS:
        column = f"markout_{horizon}"
        if column not in frame:
            continue
        rows = frame[frame[column].notna()]
        capture = rows["capture"].mean() if "capture" in rows else float("nan")
        print(f"  {f'{horizon}s':<10}{len(rows):>9,}{capture:>11.2f}"
              f"{rows[column].mean():>11.2f}{rows[f'adverse_{horizon}'].mean():>11.2f}")


def main(decisions_per_side: int = DECISIONS_PER_SIDE) -> None:
    started = time.time()
    all_fills, all_matched = [], []
    for index, day in enumerate(CLEAN_SESSIONS):
        fills, matched = session_rows(day, decisions_per_side, SEED + index)
        all_fills.extend(fills)
        all_matched.extend(matched)
        print(f"{day}: {len(fills):,} fills, {len(matched):,} matched seconds "
              f"({time.time() - started:.0f}s)", flush=True)
    fills = pd.DataFrame(all_fills)
    matched = pd.DataFrame(all_matched)
    attempts = len(CLEAN_SESSIONS) * 2 * decisions_per_side
    print(f"\n{len(fills):,} fills from {attempts:,} attempts "
          f"({len(fills) / attempts:.1%} fill rate, a floor)")

    summarise(fills, "STEP 1 - markout of an actual fill, $ per fill")
    print(f"\nSTEP 2 - matched random seconds, same side and hour, $ per second")
    print(f"  {'horizon':<10}{'rows':>9}{'drift':>11}")
    for horizon in MARKOUT_HORIZONS:
        column = f"adverse_{horizon}"
        if column in matched:
            rows = matched[matched[column].notna()]
            print(f"  {f'{horizon}s':<10}{len(rows):>9,}{rows[column].mean():>11.2f}")
    print(f"\n  adverse selection attributable to being filled")
    print(f"  {'horizon':<10}{'filled':>11}{'random':>11}{'difference':>13}")
    for horizon in MARKOUT_HORIZONS:
        column = f"adverse_{horizon}"
        if column in fills and column in matched:
            filled = fills[column].mean()
            random_drift = matched[column].mean()
            print(f"  {f'{horizon}s':<10}{filled:>11.2f}{random_drift:>11.2f}"
                  f"{filled - random_drift:>13.2f}")
    with open(RESULTS_PATH, "w") as handle:
        json.dump({"fills": len(fills), "attempts": attempts,
                   "markout": {f"{h}": float(fills[f'markout_{h}'].mean())
                               for h in MARKOUT_HORIZONS if f"markout_{h}" in fills},
                   "adverse_filled": {f"{h}": float(fills[f'adverse_{h}'].mean())
                                      for h in MARKOUT_HORIZONS if f"adverse_{h}" in fills},
                   "adverse_random": {f"{h}": float(matched[f'adverse_{h}'].mean())
                                      for h in MARKOUT_HORIZONS if f"adverse_{h}" in matched}},
                  handle, indent=1)
    fills.to_pickle("/Users/abeadam/dev/model/Stock-Model/scratchpad_backup/markout_fills.pkl")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DECISIONS_PER_SIDE)
