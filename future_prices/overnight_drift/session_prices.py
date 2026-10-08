"""ES prices at fixed Eastern wall-clock moments, read from 5-second bars.

Bars are keyed by their start (Unix seconds); a bar starting at t closes at
t + 5 s, so the price "at 16:00" is the close of the bar that ends at or just
before 16:00. Times are naive US/Eastern, as everywhere else in this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "basic_model"))
from market_clock import epoch_to_eastern  # noqa: E402

BAR_SECONDS = 5
# A price older than this at the requested moment means the market was not
# trading then (a holiday or early close), so the moment has no price.
MAX_STALENESS = pd.Timedelta(minutes=1)


def closes_by_bar_end(es_bars: pd.DataFrame) -> pd.Series:
    """Close prices indexed by the naive Eastern time each bar ends."""
    bar_end = epoch_to_eastern(es_bars["Date"] + BAR_SECONDS)
    closes = pd.Series(es_bars["Close"].to_numpy(float), index=pd.DatetimeIndex(bar_end))
    return closes[~closes.index.duplicated(keep="last")].sort_index()


def trading_weekdays(closes: pd.Series) -> pd.DatetimeIndex:
    """Midnight of every Monday-Friday date that has at least one bar."""
    days = closes.index.normalize().unique()
    return days[days.weekday < 5]


def price_at(closes: pd.Series, moments: pd.DatetimeIndex) -> np.ndarray:
    """The last close at or before each moment; NaN if none within MAX_STALENESS."""
    positions = closes.index.searchsorted(moments, side="right") - 1
    prices = np.full(len(moments), np.nan)
    has_bar = positions >= 0
    bar_times = closes.index[positions[has_bar]]
    fresh = (moments[has_bar] - bar_times) <= MAX_STALENESS
    rows = np.flatnonzero(has_bar)[fresh]
    prices[rows] = closes.to_numpy()[positions[has_bar][fresh]]
    return prices


def first_price_after(closes: pd.Series, moments: pd.DatetimeIndex) -> np.ndarray:
    """The close of the first bar ending strictly after each moment; NaN if there is none."""
    positions = closes.index.searchsorted(moments, side="right")
    prices = np.full(len(moments), np.nan)
    has_bar = positions < len(closes)
    prices[has_bar] = closes.to_numpy()[positions[has_bar]]
    return prices


def prices_at_clock(closes: pd.Series, days: pd.DatetimeIndex, clock: str) -> np.ndarray:
    """The price at wall-clock time `clock` ("HH:MM") on each day."""
    return price_at(closes, days + pd.Timedelta(f"{clock}:00"))
