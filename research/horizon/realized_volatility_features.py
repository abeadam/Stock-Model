"""Realized volatility over long windows: the largest feature effect measured here.

The base feature set carries `rolling_vol_5` and `rolling_vol_20` and nothing longer --
25 and 100 seconds of lookback on 5-second bars. The volatility head predicts the 18-bar
(90-second) forward high-to-low range, and volatility clusters over horizons far longer
than 100 seconds, so the model could see the current flicker but had no reading of the
prevailing regime.

Adding these two columns lifts the volatility head by +0.0191 R2 (t 6.6) across 13
monthly folds over 315 sessions, gaining in 12 of 13. Direction is untouched (t -0.5,
sign agreement 7/13), which is why they belong to one head and not both. They are also
free: backward-looking, computed from prices already on disk, no subscription and no
latency question -- while every implied-volatility construction tried COSTS about 0.005
R2, three times over.

The within-session rule is not a detail. A rolling window that merely blanks the
overnight return still reaches back into yesterday's tape, so each session's window is
rebuilt from scratch; a test caught that bug once already.

This lived twice, as `variance_risk_premium.realized_per_bar_move` over mid and
`long_history_support.realized_volatility` over close, with identical bodies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deep_ladder_features import session_row_blocks

# Five and fifteen minutes of 5-second bars.
REALIZED_VOLATILITY_WINDOWS = (60, 180)
REALIZED_VOLATILITY_FEATURES = [f"realized_vol_{window}"
                                for window in REALIZED_VOLATILITY_WINDOWS]


def realized_volatility(prices: np.ndarray, session_days: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation of one-bar fractional moves, rebuilt inside each session.

    Works on any price series -- mid or close -- since both are one-bar returns.
    """
    returns = pd.Series(prices).pct_change()
    result = np.full(len(prices), np.nan)
    for rows in session_row_blocks(session_days):
        block = returns.iloc[rows].reset_index(drop=True)
        block.iloc[0] = np.nan                      # the overnight jump is not a bar move
        result[rows] = block.rolling(window, min_periods=window // 2).std().to_numpy()
    return result


def add_realized_volatility(data: pd.DataFrame, prices: np.ndarray,
                            session_days: np.ndarray) -> pd.DataFrame:
    """Return `data` with the two realized-volatility columns added."""
    columns = {name: realized_volatility(prices, session_days, window)
               for name, window in zip(REALIZED_VOLATILITY_FEATURES,
                                       REALIZED_VOLATILITY_WINDOWS)}
    return data.assign(**columns)
