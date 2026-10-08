"""Passive entries priced against the real bid and ask.

`passive_entry.py` was written the day before the ES quote data arrived, so it rests its
limit at `close - offset_ticks` and treats that as a proxy for the spread. The quotes exist
now, so an order can rest where an order actually rests: on the bid when buying, on the ask
when selling.

Fills use the same conservative rule as the original, and for the same reason: an order
counts as filled only when a later bar trades **strictly through** its price. The market has
to go past you, not merely touch you, so queue position never has to be modelled and every
fill counted is one that carried its adverse selection. Signals whose price never comes back
are simply lost, which is the honest cost of resting.

Exits come in two kinds, because they are economically opposite:

  cross     leave by crossing the spread -- sell the bid, buy the ask. Guaranteed, and it
            gives back on the way out what resting earned on the way in. Long at the bid
            and out at the bid nets to the mid-to-mid assumption every P&L number in this
            project already uses, which is what makes this the honest baseline rather than
            an improvement on it.
  passive   rest on the far side for up to `wait_bars` and cross only if it never fills.
            Earns the spread twice when it works, at the cost of holding longer than
            intended when it does not.

One position or resting order at a time. Nothing reaches across a session gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TICK = 0.25
CROSS_EXIT, PASSIVE_EXIT = "cross", "passive"
EXIT_MODES = (CROSS_EXIT, PASSIVE_EXIT)
LONG, SHORT = 1, -1


@dataclass(frozen=True)
class QuotePassiveRule:
    threshold: float          # on |prob - 0.5|
    entry_offset_ticks: int   # 0 rests on the touch, 1 rests a tick behind it
    wait_bars: int            # how long the entry order rests before being cancelled
    hold_bars: int            # bars held after the fill
    exit_mode: str


@dataclass(frozen=True)
class Quotes:
    """Per-bar prices: the last quote in each bar, and the bar's traded extremes."""
    bid: np.ndarray
    ask: np.ndarray
    high: np.ndarray
    low: np.ndarray
    is_last: np.ndarray       # True where the next bar is not the next 5-second bar


def session_end_index(is_last: np.ndarray) -> np.ndarray:
    """For each bar, the index of the last bar before the next gap."""
    last_indices = np.flatnonzero(is_last)
    return last_indices[np.searchsorted(last_indices, np.arange(len(is_last)))]


def entry_limit(quotes: Quotes, bar: int, direction: int, offset_ticks: int) -> float:
    """Where a resting entry order sits: on the near touch, or `offset_ticks` behind it."""
    touch = quotes.bid[bar] if direction == LONG else quotes.ask[bar]
    return touch - direction * offset_ticks * TICK


def first_trade_through(quotes: Quotes, window: slice, limit: float, direction: int) -> int | None:
    """Offset within `window` of the first bar trading strictly through `limit`, if any."""
    through = quotes.low[window] < limit if direction == LONG else quotes.high[window] > limit
    return int(np.argmax(through)) if through.any() else None


def exit_fill(quotes: Quotes, exit_bar: int, direction: int, rule: QuotePassiveRule,
              session_end: np.ndarray) -> tuple[float, int]:
    """Price and bar the position leaves at. Crossing is immediate; resting may time out."""
    if rule.exit_mode == CROSS_EXIT:
        return (quotes.bid[exit_bar] if direction == LONG else quotes.ask[exit_bar]), exit_bar
    limit = quotes.ask[exit_bar] if direction == LONG else quotes.bid[exit_bar]
    last_wait_bar = min(exit_bar + rule.wait_bars, session_end[exit_bar])
    filled = first_trade_through(quotes, slice(exit_bar + 1, last_wait_bar + 1), limit, -direction)
    if filled is None:
        return (quotes.bid[last_wait_bar] if direction == LONG else quotes.ask[last_wait_bar]), last_wait_bar
    return limit, exit_bar + 1 + filled


def totals(profits: list[float], orders: int, point_value: float, round_trip_cost: float) -> dict:
    if not profits:
        return {"orders": orders, "fills": 0, "fill_rate": 0.0, "net": 0.0, "gross": 0.0,
                "win_rate": float("nan")}
    gross = np.array(profits) * point_value
    return {"orders": orders, "fills": len(gross), "fill_rate": len(gross) / max(orders, 1),
            "net": float(gross.sum() - round_trip_cost * len(gross)), "gross": float(gross.sum()),
            "win_rate": float(np.mean(np.array(profits) > 0))}


def simulate_crossing(score: np.ndarray, quotes: Quotes, threshold: float, hold_bars: int,
                      point_value: float, round_trip_cost: float) -> dict:
    """Every signal taken at once by crossing: buy the ask, sell the bid, and back again.

    The aggressive baseline. It never misses a signal, and pays the spread twice for it."""
    session_end = session_end_index(quotes.is_last)
    signals = np.flatnonzero((np.abs(score) >= threshold) & ~quotes.is_last)
    profits, next_free = [], 0
    for decision in signals:
        if decision < next_free:
            continue
        direction = LONG if score[decision] > 0 else SHORT
        entry = quotes.ask[decision] if direction == LONG else quotes.bid[decision]
        exit_bar = min(decision + hold_bars, session_end[decision])
        price = quotes.bid[exit_bar] if direction == LONG else quotes.ask[exit_bar]
        profits.append(direction * (price - entry))
        next_free = exit_bar + 1
    return totals(profits, len(profits), point_value, round_trip_cost)


def simulate(score: np.ndarray, quotes: Quotes, rule: QuotePassiveRule, point_value: float,
             round_trip_cost: float) -> dict:
    """Trade every signal that fills, one at a time, and total the result."""
    session_end = session_end_index(quotes.is_last)
    signals = np.flatnonzero((np.abs(score) >= rule.threshold) & ~quotes.is_last)
    profits, orders, next_free = [], 0, 0
    for decision in signals:
        if decision < next_free:
            continue
        orders += 1
        direction = LONG if score[decision] > 0 else SHORT
        limit = entry_limit(quotes, decision, direction, rule.entry_offset_ticks)
        last_wait_bar = min(decision + rule.wait_bars, session_end[decision])
        filled = first_trade_through(quotes, slice(decision + 1, last_wait_bar + 1), limit, direction)
        if filled is None:
            next_free = last_wait_bar + 1
            continue
        fill_bar = decision + 1 + filled
        exit_bar = min(fill_bar + rule.hold_bars, session_end[fill_bar])
        price, left_at = exit_fill(quotes, exit_bar, direction, rule, session_end)
        profits.append(direction * (price - limit))
        next_free = left_at + 1
    return totals(profits, orders, point_value, round_trip_cost)
