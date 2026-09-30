"""How the model has been doing lately, as an input to what it does next.

The idea is that a model which has just been wrong several times running may be facing a
market its features do not describe, and that knowing so is worth something. The inputs
are its own recent predictions scored against what actually happened.

**The hazard is timing, and it is severe.** A prediction made at bar i is about bars
i+1..i+horizon, so whether it was right is unknown until bar i+horizon. Using the accuracy
of the last few bars' predictions would therefore read the future directly -- at an 18-bar
horizon, the most recent prediction whose outcome exists is the one made **18 bars ago**.
Every column here is lagged by exactly that, and the tests plant an outcome and assert it
cannot be seen before it settles.

The second hazard is inherited: the predictions being scored must themselves be
out-of-fold, or the recorded accuracy is the accuracy of a model grading its own homework.
`out_of_fold_predictions.py` supplies those.

  fb_hit_rate_<n>          share of the last n settled calls that were directionally right
  fb_confidence_error_<n>  mean (probability - outcome): which way the calibration is off
  fb_range_error_<n>       mean (predicted range - realized range) for the volatility head

Windows stay inside the session, as everywhere else here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deep_ladder_features import session_row_blocks

FEEDBACK_WINDOWS = (12, 60)
UP, DOWN = 1.0, 0.0

FEEDBACK_FEATURES = [f"fb_{measure}_{window}" for window in FEEDBACK_WINDOWS
                     for measure in ("hit_rate", "confidence_error", "range_error")]


def settled_rolling_mean(values: np.ndarray, session_days: np.ndarray, window: int,
                         lag: int) -> np.ndarray:
    """Mean of the `window` values ending `lag` bars back, computed inside each session.

    `lag` is the forecast horizon: it is what keeps an unsettled outcome out of the mean."""
    result = np.full(len(values), np.nan)
    for rows in session_row_blocks(session_days):
        block = pd.Series(values[rows])
        result[rows] = block.rolling(window, min_periods=max(1, window // 2)).mean().shift(lag).to_numpy()
    return result


def direction_outcome(target: np.ndarray) -> np.ndarray:
    """1.0 where the move went up, 0.0 where it went down, NaN where there was no move."""
    outcome = np.where(target > 0, UP, DOWN)
    return np.where(np.isfinite(target) & (target != 0), outcome, np.nan)


def add_feedback_features(data: pd.DataFrame, probability: np.ndarray, direction_target: np.ndarray,
                          predicted_range: np.ndarray, realized_range: np.ndarray,
                          horizon: int) -> pd.DataFrame:
    """Adds FEEDBACK_FEATURES, every column lagged by `horizon` so no outcome is read early."""
    session_days = data["session_day"].to_numpy()
    outcome = direction_outcome(direction_target)
    was_right = np.where(np.isfinite(outcome), (probability > 0.5).astype(float) == outcome, np.nan)
    confidence_error = probability - outcome
    range_error = predicted_range - realized_range
    columns = {}
    for window in FEEDBACK_WINDOWS:
        for measure, values in (("hit_rate", was_right.astype(float)),
                                ("confidence_error", confidence_error),
                                ("range_error", range_error)):
            columns[f"fb_{measure}_{window}"] = settled_rolling_mean(values, session_days,
                                                                     window, horizon)
    return data.assign(**columns)
