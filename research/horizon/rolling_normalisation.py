"""Rescale each feature against its own recent past, so a split means the same thing later.

A tree learns absolute thresholds: "bb_lower_dist below -0.0004". That sentence describes
June's distribution, and June's distribution is not September's -- a model can tell the two
months apart from these features perfectly. Rescaling each feature by its own trailing mean
and spread turns the threshold into a relative statement -- "unusually low for the last
hour" -- which is a sentence that can still be true in a month the model has never seen.

The window ends at the current bar and includes it, so nothing from the future enters. It
is allowed to reach back across the overnight halt: what matters is that the comparison is
to recent trading, and the halt is not a reason to forget it.

A feature that has not moved inside the window has no spread to divide by; those bars get
a zero, meaning "exactly typical", rather than an infinity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_WINDOW_FRACTION = 4          # a window needs a quarter of its bars before it says anything
NO_SPREAD = 0.0


def rolling_z_scores(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Each column as standard deviations from its own trailing mean."""
    minimum = max(2, window // MIN_WINDOW_FRACTION)
    rolling = frame.rolling(window, min_periods=minimum)
    spread = rolling.std()
    scores = (frame - rolling.mean()) / spread.where(spread > 0)
    return scores.fillna(NO_SPREAD)


def normalised_names(names, window: int) -> list[str]:
    return [f"{name}_z{window}" for name in names]
