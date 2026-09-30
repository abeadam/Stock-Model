"""Technical indicators over a price series, as plain functions.

The cross-asset and macro features so far are pure returns: how far a symbol moved
over the last 1, 3, 12 or 60 bars. A return says nothing about where the symbol
stands -- stretched or compressed, trending or mean-reverting, quiet or agitated --
which is what these add.

Every function takes a close series and returns one array of the same length, and
none of them looks forward. Windows are counted in bars, not minutes, so the caller
decides what a bar is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERCENT = 100.0
NEUTRAL_RSI = 50.0
BOLLINGER_BAND_DEVIATIONS = 2.0
MACD_FAST_BARS, MACD_SLOW_BARS, MACD_SIGNAL_BARS = 12, 26, 9


def relative_strength_index(closes: pd.Series, window: int) -> np.ndarray:
    """Wilder's RSI: 100 means every bar in the window rose, 0 that every bar fell."""
    change = closes.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    index = PERCENT - PERCENT / (1 + gain / loss.replace(0, np.nan))
    # Nothing has fallen: the index is 100 if the symbol rose, neutral if it never moved.
    index = index.mask((loss == 0) & (gain > 0), PERCENT)
    return index.fillna(NEUTRAL_RSI).to_numpy()


def bollinger_position(closes: pd.Series, window: int) -> np.ndarray:
    """Where the close sits in its band: 0 at the lower edge, 1 at the upper, 0.5 at the mean."""
    mean = closes.rolling(window).mean()
    deviation = closes.rolling(window).std()
    width = 2 * BOLLINGER_BAND_DEVIATIONS * deviation
    lower = mean - BOLLINGER_BAND_DEVIATIONS * deviation
    return ((closes - lower) / width.replace(0, np.nan)).to_numpy()


def distance_from_average(closes: pd.Series, window: int) -> np.ndarray:
    """Percent the close stands above its exponential moving average."""
    average = closes.ewm(span=window, adjust=False).mean()
    return ((closes / average - 1) * PERCENT).to_numpy()


def realized_volatility(closes: pd.Series, window: int) -> np.ndarray:
    """Standard deviation of bar returns over the window, in percent."""
    return (closes.pct_change().rolling(window).std() * PERCENT).to_numpy()


def volatility_ratio(closes: pd.Series, fast_window: int, slow_window: int) -> np.ndarray:
    """Recent volatility against slower volatility: above 1 means the symbol is waking up."""
    fast = realized_volatility(closes, fast_window)
    slow = realized_volatility(closes, slow_window)
    return np.divide(fast, slow, out=np.full_like(fast, np.nan), where=slow > 0)


def money_flow_index(highs: pd.Series, lows: pd.Series, closes: pd.Series, volumes: pd.Series,
                     window: int) -> np.ndarray:
    """MFI: RSI weighted by traded value, so a move on volume counts for more.
    Matches futures_price.calculate_mfi, which is the production MFI_14."""
    typical_price = (highs + lows + closes) / 3
    money_flow = typical_price * volumes
    rising = typical_price > typical_price.shift(1)
    inflow = money_flow.where(rising, 0).rolling(window).sum()
    outflow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
    index = PERCENT - PERCENT / (1 + inflow / outflow.replace(0, np.nan))
    index = index.mask((outflow == 0) & (inflow > 0), PERCENT)
    return index.fillna(NEUTRAL_RSI).to_numpy()


def bollinger_band_distance(closes: pd.Series, window: int, edge: str) -> np.ndarray:
    """(Close - band) / Close, the production bb_upper_dist / bb_lower_dist.
    Positive above the upper band, negative below the lower one."""
    mean = closes.rolling(window).mean()
    offset = BOLLINGER_BAND_DEVIATIONS * closes.rolling(window).std()
    band = mean + offset if edge == "upper" else mean - offset
    return ((closes - band) / closes).to_numpy()


def macd_histogram(closes: pd.Series) -> np.ndarray:
    """MACD line minus its signal line, as a percent of price so symbols compare."""
    fast = closes.ewm(span=MACD_FAST_BARS, adjust=False).mean()
    slow = closes.ewm(span=MACD_SLOW_BARS, adjust=False).mean()
    macd = fast - slow
    return ((macd - macd.ewm(span=MACD_SIGNAL_BARS, adjust=False).mean()) / closes * PERCENT).to_numpy()
