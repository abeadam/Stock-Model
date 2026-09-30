"""Take-profit exits: enter on a confident signal and exit at the first of
  - a gain of max(2 standard deviations of the last 5 minutes' price movement, 0.25 points),
  - a confident signal the other way (the next bar can then open that side if the
    signal is still there),
  - the end of the trading period (the last bar before a gap in the rows given:
    a segment's end, the 17:00 ET halt),
  - optionally, a loss of stop_points (the user's rule has no stop; this is a variant).
One position at a time.

The gain is checked at each 5-second bar close. A long buys at buy_price and exits
when sell_price has risen by the target; a short mirrors that. Pass the ask and bid
to cross the spread, or the mid twice for mid fills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hold_policy import first_at_or_after

BAR_SECONDS = 5
VOLATILITY_WINDOW_BARS = 60          # 5 minutes of 5-second bars
TARGET_STANDARD_DEVIATIONS = 2.0
MIN_TARGET_POINTS = 0.25             # one ES tick
VOLATILITY_MEASURES = ("changes", "levels")


def _window_is_contiguous(seconds: np.ndarray, bars: int) -> np.ndarray:
    span = np.full(len(seconds), -1, np.int64)
    if bars <= len(seconds):
        span[bars - 1:] = seconds[bars - 1:] - seconds[:len(seconds) - bars + 1]
    return span == (bars - 1) * BAR_SECONDS


def recent_volatility(mid: np.ndarray, seconds: np.ndarray, measure: str) -> np.ndarray:
    """Standard deviation over the last 5 minutes, ending at each bar, NaN across a gap.
    "changes": of the 5-second mid changes. "levels": of the mid price itself."""
    prices = pd.Series(mid)
    if measure == "changes":
        volatility = prices.diff().rolling(VOLATILITY_WINDOW_BARS).std()
        contiguous = _window_is_contiguous(seconds, VOLATILITY_WINDOW_BARS + 1)
    elif measure == "levels":
        volatility = prices.rolling(VOLATILITY_WINDOW_BARS).std()
        contiguous = _window_is_contiguous(seconds, VOLATILITY_WINDOW_BARS)
    else:
        raise ValueError(f"unknown volatility measure {measure!r}; expected one of {VOLATILITY_MEASURES}")
    return volatility.where(contiguous).to_numpy()


def take_profit_points(volatility: np.ndarray, multiple: float = TARGET_STANDARD_DEVIATIONS) -> np.ndarray:
    """max(multiple standard deviations, 0.25 points); the floor alone where volatility is unknown."""
    target = multiple * np.nan_to_num(volatility, nan=0.0)
    return np.maximum(target, MIN_TARGET_POINTS)


def simulate_take_profit(score: np.ndarray, buy_price: np.ndarray, sell_price: np.ndarray,
                         target_points: np.ndarray, is_last: np.ndarray, threshold: float,
                         point_value: float, round_trip_cost: float,
                         stop_points: np.ndarray | None = None) -> dict:
    long_signals = np.flatnonzero(score >= threshold)
    short_signals = np.flatnonzero(score <= -threshold)
    long_entries = long_signals[~is_last[long_signals]]
    short_entries = short_signals[~is_last[short_signals]]
    period_ends = np.flatnonzero(is_last)
    pnl_points, bars_held, exit_reasons, directions = [], [], [], []
    position = 0
    while True:
        next_long = first_at_or_after(long_entries, position)
        next_short = first_at_or_after(short_entries, position)
        if next_long < 0 and next_short < 0:
            break
        is_long = next_short < 0 or (0 <= next_long < next_short)
        entry = next_long if is_long else next_short
        period_end = first_at_or_after(period_ends, entry)
        reversal = first_at_or_after(short_signals if is_long else long_signals, entry + 1)
        last_bar = period_end if reversal < 0 else min(reversal, period_end)
        later = slice(entry + 1, last_bar + 1)
        entry_price = buy_price[entry] if is_long else sell_price[entry]
        # Gain the position would book by exiting at each later bar.
        gain = sell_price[later] - entry_price if is_long else entry_price - buy_price[later]
        exit_bar, reason = last_bar, "reversal" if last_bar == reversal else "period_end"
        hits = np.flatnonzero(gain >= target_points[entry])
        stops = np.flatnonzero(gain <= -stop_points[entry]) if stop_points is not None else np.array([], int)
        first_hit = hits[0] if len(hits) else np.inf
        first_stop = stops[0] if len(stops) else np.inf
        if min(first_hit, first_stop) < np.inf:
            offset = int(min(first_hit, first_stop))
            exit_bar, reason = entry + 1 + offset, "target" if first_hit <= first_stop else "stop"
        pnl_points.append(sell_price[exit_bar] - entry_price if is_long else entry_price - buy_price[exit_bar])
        bars_held.append(exit_bar - entry)
        exit_reasons.append(reason)
        directions.append(1 if is_long else -1)
        position = exit_bar + 1

    if not pnl_points:
        return {"trades": 0, "net": 0.0, "gross": 0.0, "hit_rate": np.nan, "target_rate": np.nan,
                "reversal_rate": np.nan, "stop_rate": np.nan, "mean_bars_held": np.nan}
    pnl_points = np.array(pnl_points)
    gross = pnl_points * point_value
    return {"trades": len(gross), "net": float(gross.sum() - round_trip_cost * len(gross)),
            "gross": float(gross.sum()), "hit_rate": float(np.mean(pnl_points > 0)),
            "target_rate": float(np.mean(np.array(exit_reasons) == "target")),
            "reversal_rate": float(np.mean(np.array(exit_reasons) == "reversal")),
            "stop_rate": float(np.mean(np.array(exit_reasons) == "stop")),
            "mean_bars_held": float(np.mean(bars_held)), "pnl_points": pnl_points,
            "directions": np.array(directions)}
