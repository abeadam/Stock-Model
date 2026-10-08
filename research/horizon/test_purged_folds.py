"""Tests for horizon-tied fold purging.

The first test is the one that matters: no training label may reach into the test fold.
Had it existed, the 12-bar purge sitting under a 5-bar horizon would have been visible
as an over-gap rather than discovered through two ablations disagreeing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purged_folds import purged_fold_splits

BAR_SECONDS = 5
SERIES_START = pd.Timestamp("2026-01-01 00:00:00")
SERIES_BARS = 4320                                   # six hours of five-second bars
FOLD_STARTS = [SERIES_START + pd.Timedelta(hours=hours) for hours in (4, 5)]
DATA_END = SERIES_START + pd.Timedelta(hours=6)


def synthetic_times() -> pd.Series:
    return pd.Series([SERIES_START + pd.Timedelta(seconds=BAR_SECONDS * bar)
                      for bar in range(SERIES_BARS)])


def splits_at(horizon: int, purge_bars: int | None = None):
    return purged_fold_splits(synthetic_times(), horizon, purge_bars=purge_bars,
                              fold_starts=FOLD_STARTS, data_end=DATA_END)


def label_end(times: pd.Series, row: int, horizon: int) -> pd.Timestamp:
    """When the forward window of the row's label closes: bars row+1..row+horizon."""
    return times.iloc[row] + pd.Timedelta(seconds=BAR_SECONDS * horizon)


@pytest.mark.parametrize("horizon", [1, 5, 12, 60])
def test_no_training_label_reaches_the_test_fold(horizon):
    times = synthetic_times()
    for split in splits_at(horizon):
        assert label_end(times, split.train.max(), horizon) < times.iloc[split.test.min()]


@pytest.mark.parametrize("horizon", [1, 5, 12, 60])
def test_no_training_label_reaches_the_validation_block(horizon):
    times = synthetic_times()
    for split in splits_at(horizon):
        assert label_end(times, split.train.max(), horizon) < times.iloc[split.validation.min()]


def test_a_purge_shorter_than_the_horizon_is_refused():
    with pytest.raises(ValueError, match="shorter than"):
        splits_at(horizon=12, purge_bars=5)


def test_an_equal_purge_is_allowed():
    assert splits_at(horizon=12, purge_bars=12)


def test_a_longer_purge_is_allowed_for_robustness_checks():
    assert splits_at(horizon=12, purge_bars=24)


def test_a_longer_purge_keeps_fewer_training_rows():
    tight = splits_at(horizon=12)[0]
    loose = splits_at(horizon=12, purge_bars=60)[0]
    assert len(loose.train) < len(tight.train)


def test_test_rows_do_not_depend_on_the_purge():
    tight = splits_at(horizon=12)[0]
    loose = splits_at(horizon=12, purge_bars=60)[0]
    assert np.array_equal(tight.test, loose.test)


def test_the_three_blocks_never_overlap():
    for split in splits_at(horizon=12):
        assert not (set(split.train) & set(split.validation))
        assert not (set(split.validation) & set(split.test))
        assert not (set(split.train) & set(split.test))


def test_dev_stage_yields_one_split_per_fold():
    assert len(splits_at(horizon=5)) == 2
