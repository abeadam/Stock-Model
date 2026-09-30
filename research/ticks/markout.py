"""What a passive fill is worth once the market has had time to react to it.

A resting order captures half the spread the instant it fills -- about $6.25 on ES -- and
then loses some of it back, because the flow that filled you was going somewhere. The
markout curve is how much survives at each horizon:

    half spread captured  d * (mid(t)     - P)        what the fill is worth immediately
    markout(tau)          d * (mid(t+tau) - P)        what it is worth later
    adverse selection     -d * (mid(t+tau) - mid(t))  the difference, i.e. what the
                                                      market took back

with d = +1 for a buy resting on the bid. By construction markout = capture - adverse
selection, which the tests check rather than assume.

**Both ends must sit on a fresh quote.** IBKR's historical BID_ASK ticks are thinned --
roughly 1,988 of 3,600 seconds carry an update -- so a carried-forward mid at t+tau is
really the mid from some earlier second, which understates how far the market moved. That
single detail decides whether the curve means anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from passive_fill import Book, simulate_limit_order

TICK = 0.25
POINT_VALUE = 50.0
MARKOUT_HORIZONS = (1, 5, 10, 30, 60)
BUY, SELL = 1, -1


def mid_at(book: Book, moment: float) -> float:
    """Mid of the last quote at or before `moment`; NaN before the first quote."""
    row = book.quote_row(moment)
    if row < 0:
        return float("nan")
    return float((book.bid[row] + book.ask[row]) / 2.0)


def touch_price(book: Book, moment: float, side: int) -> float:
    """Where a resting order would sit: the bid when buying, the ask when selling."""
    row = book.quote_row(moment)
    if row < 0:
        return float("nan")
    return float(book.bid[row] if side == BUY else book.ask[row])


def queue_ahead(book: Book, moment: float, side: int) -> float:
    row = book.quote_row(moment)
    if row < 0:
        return float("nan")
    return float(book.bid_size[row] if side == BUY else book.ask_size[row])


def capture_dollars(side: int, fill_price: float, mid_then: float) -> float:
    """Half the spread, banked the moment the order fills."""
    return side * (mid_then - fill_price) * POINT_VALUE


def markout_dollars(side: int, fill_price: float, mid_later: float) -> float:
    """What the fill is worth once the market has moved on."""
    return side * (mid_later - fill_price) * POINT_VALUE


def adverse_dollars(side: int, mid_then: float, mid_later: float) -> float:
    """What the market took back after the fill."""
    return -side * (mid_later - mid_then) * POINT_VALUE


def fresh_seconds(grid: pd.DataFrame) -> set[int]:
    """Seconds whose quote actually updated, as a set for fast membership tests."""
    return set(grid.index[grid["quote_is_fresh"].to_numpy()].astype(int))


def attempt(book: Book, moment: float, side: int, entry_limit: float) -> tuple[float, float] | None:
    """(fill time, fill price) for an order resting at the touch, or None if it never fills."""
    price = touch_price(book, moment, side)
    ahead = queue_ahead(book, moment, side)
    if not np.isfinite(price) or not np.isfinite(ahead):
        return None
    filled_at = simulate_limit_order(book, moment, price, side, ahead, entry_limit)
    return None if filled_at is None else (filled_at, price)


def markout_curve(book: Book, fresh: set[int], fill_time: float, fill_price: float, side: int,
                  horizons=MARKOUT_HORIZONS) -> dict[int, float] | None:
    """Markout in dollars at each horizon, or None if the fill second is not fresh.

    A horizon whose far end lands on a stale quote is left out rather than filled in with
    a carried-forward mid, which would flatter the curve."""
    if int(fill_time) not in fresh:
        return None
    mid_then = mid_at(book, fill_time)
    if not np.isfinite(mid_then):
        return None
    curve = {"capture": capture_dollars(side, fill_price, mid_then)}
    for horizon in horizons:
        later = int(fill_time) + horizon
        if later not in fresh:
            continue
        mid_later = mid_at(book, float(later))
        if not np.isfinite(mid_later):
            continue
        curve[f"markout_{horizon}"] = markout_dollars(side, fill_price, mid_later)
        curve[f"adverse_{horizon}"] = adverse_dollars(side, mid_then, mid_later)
    return curve


def drift_dollars(book: Book, fresh: set[int], moment: int, side: int,
                  horizons=MARKOUT_HORIZONS) -> dict[int, float] | None:
    """Mid drift from an arbitrary second, signed as if a fill had happened there.

    This is the counterfactual for the markout curve: without it, adverse selection cannot
    be told apart from the market simply drifting."""
    if moment not in fresh:
        return None
    mid_then = mid_at(book, float(moment))
    if not np.isfinite(mid_then):
        return None
    drift = {}
    for horizon in horizons:
        later = moment + horizon
        if later not in fresh:
            continue
        mid_later = mid_at(book, float(later))
        if np.isfinite(mid_later):
            drift[f"adverse_{horizon}"] = adverse_dollars(side, mid_then, mid_later)
    return drift
