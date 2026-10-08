"""Trading a decision rather than a score.

The usual harness takes a score, picks a threshold on validation, and trades what passes.
A cost-aware model does not need that step: it says long, short or nothing for each bar,
and this turns that sequence into money.

One position at a time, held a fixed number of bars, never across a session gap. Fills at
the mid or by crossing the spread, with the round trip charged once per trade.
"""
from __future__ import annotations

import numpy as np

from quote_passive_entry import LONG, Quotes, session_end_index

MID_FILL, CROSS_FILL = "mid", "cross"


def simulate_decisions(direction: np.ndarray, quotes: Quotes, mid: np.ndarray, hold_bars: int,
                       point_value: float, cost: float, mode: str = MID_FILL) -> dict:
    """Total the result of acting on `direction`, one position at a time."""
    session_end = session_end_index(quotes.is_last)
    buy = mid if mode == MID_FILL else quotes.ask
    sell = mid if mode == MID_FILL else quotes.bid
    profits, next_free = [], 0
    for row in np.flatnonzero(direction != 0):
        if row < next_free or quotes.is_last[row]:
            continue
        exit_row = min(row + hold_bars, session_end[row])
        if exit_row <= row:
            continue
        gain = (sell[exit_row] - buy[row]) if direction[row] == LONG else (sell[row] - buy[exit_row])
        profits.append(gain * point_value - cost)
        next_free = exit_row + 1
    if not profits:
        return {"trades": 0, "net": 0.0, "per_trade": float("nan"), "win_rate": float("nan")}
    profits = np.array(profits)
    return {"trades": len(profits), "net": float(profits.sum()),
            "per_trade": float(profits.mean()), "win_rate": float((profits > 0).mean())}
