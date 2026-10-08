"""Tests for prediction-feedback features.

The first test is the whole point: an outcome must be invisible until it has settled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_feedback import (FEEDBACK_FEATURES, FEEDBACK_WINDOWS, add_feedback_features,
                                 direction_outcome, settled_rolling_mean)

HORIZON = 18
BARS = 400


def frame(sessions: int = 2, bars: int = BARS) -> pd.DataFrame:
    return pd.DataFrame({"session_day": np.repeat(np.arange(sessions), bars)})


def test_an_outcome_is_invisible_until_it_settles():
    """A spike at bar k must not reach any feature before bar k + horizon."""
    values = np.zeros(BARS)
    spike = 100
    values[spike] = 1000.0
    days = np.zeros(BARS, dtype=int)
    rolled = settled_rolling_mean(values, days, window=12, lag=HORIZON)
    before = rolled[spike + 1:spike + HORIZON]
    assert np.nanmax(np.abs(before)) == pytest.approx(0.0)


def test_the_outcome_becomes_visible_exactly_when_it_settles():
    values = np.zeros(BARS)
    spike = 100
    values[spike] = 1000.0
    rolled = settled_rolling_mean(values, np.zeros(BARS, dtype=int), window=12, lag=HORIZON)
    assert rolled[spike + HORIZON] > 0


def test_a_longer_horizon_delays_visibility_further():
    values = np.zeros(BARS)
    values[100] = 1000.0
    days = np.zeros(BARS, dtype=int)
    short = settled_rolling_mean(values, days, window=12, lag=5)
    long = settled_rolling_mean(values, days, window=12, lag=40)
    assert short[105] > 0 and long[105] == pytest.approx(0.0)


def test_windows_do_not_span_a_session_gap():
    days = np.repeat(np.arange(2), 100)
    values = np.concatenate([np.ones(100), np.zeros(100)])
    rolled = settled_rolling_mean(values, days, window=12, lag=1)
    assert np.isnan(rolled[100])                       # first bar of the second session


def test_each_session_starts_blank_for_the_lag_and_window():
    days = np.repeat(np.arange(2), 100)
    rolled = settled_rolling_mean(np.ones(200), days, window=12, lag=18)
    assert np.isnan(rolled[100:100 + 18]).all()


def test_direction_outcome_marks_up_and_down():
    outcome = direction_outcome(np.array([0.5, -0.5]))
    assert list(outcome) == [1.0, 0.0]


def test_a_zero_move_has_no_outcome():
    assert np.isnan(direction_outcome(np.array([0.0]))[0])


def test_a_missing_target_has_no_outcome():
    assert np.isnan(direction_outcome(np.array([np.nan]))[0])


def test_a_model_that_is_always_right_scores_a_hit_rate_of_one():
    data = frame(sessions=1)
    target = np.where(np.arange(BARS) % 2 == 0, 1.0, -1.0)
    probability = np.where(target > 0, 0.9, 0.1)
    built = add_feedback_features(data, probability, target, np.zeros(BARS), np.zeros(BARS), HORIZON)
    assert np.nanmax(built["fb_hit_rate_12"].to_numpy()) == pytest.approx(1.0)


def test_a_model_that_is_always_wrong_scores_a_hit_rate_of_zero():
    data = frame(sessions=1)
    target = np.where(np.arange(BARS) % 2 == 0, 1.0, -1.0)
    probability = np.where(target > 0, 0.1, 0.9)
    built = add_feedback_features(data, probability, target, np.zeros(BARS), np.zeros(BARS), HORIZON)
    assert np.nanmin(built["fb_hit_rate_12"].to_numpy()) == pytest.approx(0.0)


def test_range_error_is_predicted_less_realized():
    data = frame(sessions=1)
    predicted, realized = np.full(BARS, 3.0), np.full(BARS, 1.0)
    built = add_feedback_features(data, np.full(BARS, 0.5), np.ones(BARS), predicted, realized, HORIZON)
    assert np.nanmax(built["fb_range_error_12"].to_numpy()) == pytest.approx(2.0)


def test_confidence_error_is_probability_less_outcome():
    data = frame(sessions=1)
    built = add_feedback_features(data, np.full(BARS, 0.8), np.ones(BARS),
                                  np.zeros(BARS), np.zeros(BARS), HORIZON)
    assert np.nanmax(built["fb_confidence_error_12"].to_numpy()) == pytest.approx(-0.2)


def test_every_named_column_is_built():
    data = frame(sessions=1)
    built = add_feedback_features(data, np.full(BARS, 0.5), np.ones(BARS),
                                  np.zeros(BARS), np.zeros(BARS), HORIZON)
    assert set(FEEDBACK_FEATURES) <= set(built.columns)


def test_both_windows_are_present():
    assert all(f"fb_hit_rate_{window}" in FEEDBACK_FEATURES for window in FEEDBACK_WINDOWS)
