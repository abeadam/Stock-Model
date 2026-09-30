"""Signal-hold trading policy on 5-second bars: enter on a confident signal, stay in
while the model still leans the same way, exit when it stops agreeing.

The signal is a score centred on zero (a probability minus 0.5, or an expected move).
Long entry when score >= entry_threshold; the long exits at the first bar (after a
minimum hold) where score < exit_threshold. Shorts mirror that around zero. Every
position is closed at the last bar before a gap in the 5-second series (the daily
break, a halt, missing data), so nothing is held across one. One position at a time.
Fills are fill_delay_bars after the decision bar (never past a gap): a long buys
at buy_price and sells at sell_price, a short the reverse. Pass the ask and bid to
cross the spread, the mid twice for mid fills, or the close twice for last-trade
fills (then a delay of 1 keeps the decision bar's own bid-ask bounce out).

Jumps between candidate bars with searchsorted, so cost is per trade, not per bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BAR_SECONDS = 5


@dataclass(frozen=True)
class HoldRule:
    entry_threshold: float
    exit_threshold: float
    min_hold_bars: int
    fill_delay_bars: int = 0


def last_bar_before_gap(seconds: np.ndarray) -> np.ndarray:
    """True where the next bar is not the next 5-second bar (and on the final bar)."""
    is_last = np.ones(len(seconds), bool)
    is_last[:-1] = np.diff(seconds) != BAR_SECONDS
    return is_last


def first_at_or_after(sorted_indices: np.ndarray, position: int) -> int:
    """The first index in sorted_indices >= position, or -1 if none."""
    slot = np.searchsorted(sorted_indices, position)
    return int(sorted_indices[slot]) if slot < len(sorted_indices) else -1


def _fill_bar(decision_bar: int, forced_exits: np.ndarray, delay: int) -> int:
    """The bar whose close fills a decision: `delay` bars later, but not past a gap."""
    return min(decision_bar + delay, first_at_or_after(forced_exits, decision_bar))


def simulate_hold(score: np.ndarray, buy_price: np.ndarray, sell_price: np.ndarray, is_last: np.ndarray,
                  rule: HoldRule, point_value: float, round_trip_cost: float) -> dict:
    tradable = ~is_last
    long_entries = np.flatnonzero(tradable & (score >= rule.entry_threshold))
    short_entries = np.flatnonzero(tradable & (score <= -rule.entry_threshold))
    long_exits = np.flatnonzero((score < rule.exit_threshold) | is_last)
    short_exits = np.flatnonzero((score > -rule.exit_threshold) | is_last)
    forced_exits = np.flatnonzero(is_last)

    pnl_points, bars_held = [], []
    position = 0
    while True:
        next_long = first_at_or_after(long_entries, position)
        next_short = first_at_or_after(short_entries, position)
        if next_long < 0 and next_short < 0:
            break
        is_long = next_short < 0 or (0 <= next_long < next_short)
        entry = next_long if is_long else next_short
        signal_exit = first_at_or_after(long_exits if is_long else short_exits, entry + rule.min_hold_bars)
        forced_exit = first_at_or_after(forced_exits, entry)
        exit_bar = forced_exit if signal_exit < 0 else min(signal_exit, forced_exit)
        entry_fill = _fill_bar(entry, forced_exits, rule.fill_delay_bars)
        exit_fill = _fill_bar(exit_bar, forced_exits, rule.fill_delay_bars)
        if is_long:
            pnl_points.append(sell_price[exit_fill] - buy_price[entry_fill])
        else:
            pnl_points.append(sell_price[entry_fill] - buy_price[exit_fill])
        bars_held.append(exit_fill - entry_fill)
        position = exit_fill + 1

    pnl_points = np.array(pnl_points)
    if len(pnl_points) == 0:
        return {"trades": 0, "net": 0.0, "gross": 0.0, "hit_rate": np.nan, "mean_bars_held": np.nan}
    gross = pnl_points * point_value
    return {"trades": len(gross), "net": float(gross.sum() - round_trip_cost * len(gross)),
            "gross": float(gross.sum()), "hit_rate": float(np.mean(pnl_points > 0)),
            "mean_bars_held": float(np.mean(bars_held))}
