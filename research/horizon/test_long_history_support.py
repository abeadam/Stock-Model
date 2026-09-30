"""Tests for the long-history harness.

The fold tests matter most: a monthly walk-forward that trains on later months is the
exact mistake `tick_walk_forward.py` shipped with, and it is invisible in the output.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from long_history_support import (BAR_SECONDS, MIN_TRAIN_SESSIONS, REALIZED_FEATURES,
                                  SMOOTHED_FEATURES, add_long_history_features, base_columns,
                                  forward_average_target, forward_range_target, monthly_folds,
                                  realized_volatility)

START_EPOCH = 1_790_000_000
BARS_PER_SESSION = 100


def session_days(count: int, bars: int = BARS_PER_SESSION) -> np.ndarray:
    start = dt.date(2025, 7, 1)
    return np.repeat([start + dt.timedelta(days=offset) for offset in range(count)], bars)


def frame(sessions: int = 80) -> pd.DataFrame:
    days = session_days(sessions)
    rows = len(days)
    generator = np.random.default_rng(0)
    close = 5000 + np.cumsum(generator.normal(0, 0.5, rows))
    return pd.DataFrame({"session_day": days, "Close": close, "High": close + 1.0,
                         "Low": close - 1.0, "high_close_ratio": generator.normal(size=rows),
                         "low_close_ratio": generator.normal(size=rows)})


class ColumnTest:
    pass


def test_brk_is_always_excluded():
    assert not any("brk" in name for name in base_columns(include_spy=True))


def test_spy_is_excluded_only_when_asked():
    assert any("spy" in name for name in base_columns(include_spy=True))
    assert not any("spy" in name for name in base_columns(include_spy=False))


def test_dropping_spy_removes_both_its_families():
    with_spy = set(base_columns(include_spy=True))
    without = set(base_columns(include_spy=False))
    removed = with_spy - without
    assert any("spy_ret" in name for name in removed)
    assert any("spy_minus_es" in name for name in removed)


def test_the_basket_columns_survive_dropping_one_member():
    assert any("basket" in name for name in base_columns(include_spy=False))


def test_realized_volatility_does_not_span_a_session_gap():
    data = frame(sessions=3)
    values = realized_volatility(data["Close"].to_numpy(), data["session_day"].to_numpy(), 10)
    assert np.isnan(values[BARS_PER_SESSION])


def test_realized_volatility_rises_with_a_noisier_tape():
    days = session_days(2)
    quiet = np.cumsum(np.random.default_rng(1).normal(0, 0.1, len(days))) + 5000
    loud = np.cumsum(np.random.default_rng(1).normal(0, 2.0, len(days))) + 5000
    assert np.nanmean(realized_volatility(loud, days, 20)) > np.nanmean(realized_volatility(quiet, days, 20))


def test_every_added_column_is_built():
    built = add_long_history_features(frame())
    assert set(REALIZED_FEATURES + SMOOTHED_FEATURES) <= set(built.columns)


def test_the_forward_target_measures_the_average_ahead():
    close = np.arange(100, dtype=float)
    seconds = START_EPOCH + BAR_SECONDS * np.arange(100)
    target = forward_average_target(close, seconds, bars=4, delay=0)
    # from close[10]=10 to the mean of 11,12,13,14 = 12.5
    assert target[10] == pytest.approx((12.5 - 10) / 10 * 100)


def test_the_delay_moves_both_the_entry_and_the_window():
    """At delay 1 the entry is the next bar, so the signal bar's own bounce is excluded."""
    close = np.arange(100, dtype=float)
    seconds = START_EPOCH + BAR_SECONDS * np.arange(100)
    delayed = forward_average_target(close, seconds, bars=4, delay=1)
    # from close[11]=11 to the mean of 12,13,14,15 = 13.5
    assert delayed[10] == pytest.approx((13.5 - 11) / 11 * 100)


def test_the_target_is_blank_across_a_session_gap():
    seconds = np.concatenate([START_EPOCH + BAR_SECONDS * np.arange(50),
                              START_EPOCH + 86_400 + BAR_SECONDS * np.arange(50)])
    close = np.arange(100, dtype=float)
    target = forward_average_target(close, seconds, bars=4)
    assert np.isnan(target[48])


def test_the_range_target_is_the_high_low_span_ahead():
    close = np.full(50, 100.0)
    high, low = close + 2.0, close - 3.0
    seconds = START_EPOCH + BAR_SECONDS * np.arange(50)
    assert forward_range_target(high, low, close, seconds, 5)[10] == pytest.approx(5.0)


def test_no_fold_trains_on_a_later_month():
    """The leak that shipped in tick_walk_forward: training on sessions after the test."""
    days = session_days(200)
    for fold in monthly_folds(days, horizon=18, min_train_sessions=20):
        assert days[fold.split.train].max() < days[fold.split.test].min()
        assert days[fold.split.validation].max() < days[fold.split.test].min()


def test_validation_always_follows_training():
    days = session_days(200)
    for fold in monthly_folds(days, horizon=18, min_train_sessions=20):
        assert fold.split.train.max() < fold.split.validation.min()


def test_training_stops_a_horizon_before_validation():
    days = session_days(200)
    for fold in monthly_folds(days, horizon=18, min_train_sessions=20):
        assert fold.split.validation.min() - fold.split.train.max() > 18


def test_the_test_block_starts_a_horizon_into_the_month():
    days = session_days(200)
    months = np.array([f"{day.year:04d}-{day.month:02d}" for day in days])
    for fold in monthly_folds(days, horizon=18, min_train_sessions=20):
        first_of_month = np.flatnonzero(months == fold.month)[0]
        assert fold.split.test.min() >= first_of_month + 18


def test_months_without_enough_history_are_skipped():
    days = session_days(200)
    folds = monthly_folds(days, horizon=18, min_train_sessions=MIN_TRAIN_SESSIONS)
    assert folds and folds[0].month > f"{days[0].year:04d}-{days[0].month:02d}"


def test_enough_folds_exist_to_be_worth_the_exercise():
    days = session_days(317)
    assert len(monthly_folds(days, horizon=18)) >= 8
