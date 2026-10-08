"""First-to-reach labels matching the take-profit exit: from each bar, does the mid
first rise by `barrier` points (+1) or first fall by it (-1)? Rows where neither
happens within max_bars, or before the next gap in the series, are NaN.
"""
from __future__ import annotations

import numpy as np

from hold_policy import last_bar_before_gap

UP, DOWN = 1.0, -1.0


def run_ends(seconds: np.ndarray) -> np.ndarray:
    """For each bar, the index of the last bar before the next gap."""
    ends = np.flatnonzero(last_bar_before_gap(seconds))
    return ends[np.searchsorted(ends, np.arange(len(seconds)))]


def first_barrier_labels(mid: np.ndarray, seconds: np.ndarray, barrier: np.ndarray, max_bars: int) -> np.ndarray:
    count = len(mid)
    labels = np.full(count, np.nan)
    unresolved = np.ones(count, bool)
    last_allowed = np.minimum(run_ends(seconds), np.arange(count) + max_bars)
    for step in range(1, max_bars + 1):
        later = np.arange(count) + step
        active = unresolved & (later <= last_allowed)
        if not active.any():
            break
        change = mid[np.minimum(later, count - 1)] - mid
        rose = active & (change >= barrier)
        fell = active & (change <= -barrier)
        labels[rose] = UP
        labels[fell] = DOWN
        unresolved &= ~(rose | fell)
    return labels
