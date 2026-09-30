"""Fold splits whose purge is tied to the target horizon.

A forward-looking label for the row at time s is built from bars s+1..s+horizon, so any
training row within `horizon` bars of the test fold carries prices from inside it. The
purge is the gap that removes that overlap, and it has exactly one defensible length: the
horizon itself.

Shorter leaks. Longer does not leak but is not free either -- the validation block is
positioned as a fraction of the history the purge leaves behind, and early stopping is
decided on that block, so changing the purge slides the boundary and changes which round
wins. Measured on this data at 0.0017 mean AUC and 0.0043 at worst, which is larger than
most of the effects this project is trying to measure. Tying the purge to the horizon
removes the leak and removes the free parameter at the same time.
"""
from __future__ import annotations

from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE
from short_horizon_walk_forward import fold_split

DEV_STAGE = "dev"


def purged_fold_splits(times, horizon: int, stage: str = DEV_STAGE, purge_bars: int | None = None,
                       fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC):
    """One split per fold of `stage`, purged by `horizon` bars unless overridden.

    `purge_bars` exists only so that a robustness check can vary the gap deliberately;
    leaving it out is the correct choice everywhere else."""
    gap = horizon if purge_bars is None else purge_bars
    if gap < horizon:
        raise ValueError(f"purge of {gap} bars is shorter than the {horizon}-bar label window")
    return [fold_split(times, fold, purge_bars=gap, fold_starts=fold_starts, data_end=data_end)
            for fold in FOLDS_BY_STAGE[stage]]
