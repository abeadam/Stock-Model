"""Contract rolls in the stitched ES price series.

The daily 5-second files follow the front ES contract and switch to the next
quarterly contract across the 17:00-18:00 ET halt shortly before expiry. The
next contract trades above the expiring one by the cost of carry (~50-70
points), so the switch shows up as a jump that nobody holding a position earns.
Any P&L measured across that halt has to take the jump out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from session_prices import first_price_after, prices_at_clock

ROLL_MONTHS = (3, 6, 9, 12)
# The data switches contracts within this many days before the third Friday
# (the day before expiry, earlier when a holiday moves expiry up).
ROLL_WINDOW_DAYS = 7
# Normal moves across the halt are a few points (median ~2); carry makes a roll
# jump 50+ points, so 40 separates them without catching ordinary news gaps
# outside roll weeks, which the calendar check excludes anyway.
MIN_ROLL_GAP_POINTS = 40.0
HALT_START_CLOCK = "17:00"
FRIDAY = 4
WEEKS_TO_THIRD_FRIDAY = 2


def third_friday(day: pd.Timestamp) -> pd.Timestamp:
    """The third Friday of `day`'s month."""
    first_of_month = day.replace(day=1)
    days_to_first_friday = (FRIDAY - first_of_month.weekday()) % 7
    return first_of_month + pd.Timedelta(days=days_to_first_friday + 7 * WEEKS_TO_THIRD_FRIDAY)


def is_roll_week(days: pd.DatetimeIndex) -> np.ndarray:
    """True for days in a quarterly expiry month within ROLL_WINDOW_DAYS before its third Friday."""
    days_before_expiry = np.array([(third_friday(day) - day).days for day in days])
    return (days.month.isin(ROLL_MONTHS) & (days_before_expiry >= 0)
            & (days_before_expiry <= ROLL_WINDOW_DAYS))


def roll_gaps(closes: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Points added by a contract switch across the halt that starts at 17:00 on each day
    (Friday's lasts the weekend); zero on other days."""
    halt_start = days + pd.Timedelta(f"{HALT_START_CLOCK}:00")
    halt_gap = first_price_after(closes, halt_start) - prices_at_clock(closes, days, HALT_START_CLOCK)
    gap = pd.Series(halt_gap, index=days).fillna(0.0)
    candidates = gap[is_roll_week(days) & (gap.to_numpy() > MIN_ROLL_GAP_POINTS)]
    # One switch per expiry: a big weekend news gap in the same week is a real move.
    switch_days = candidates.groupby(candidates.index.to_period("M")).idxmax()
    return gap.where(gap.index.isin(switch_days), 0.0)
