"""Tests for the variance-risk-premium features.

The one that matters most is causality: a row's seasonal profile must never include its
own session, or the feature quietly knows the day it is standing in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from variance_risk_premium import (BAR_SECONDS, SEASONAL_BUCKET_BARS, TRADING_SECONDS_PER_YEAR,
                                   VARIANCE_PREMIUM_FEATURES, add_variance_premium_features,
                                   implied_per_bar_move, position_in_session,
                                   realized_per_bar_move, trailing_seasonal_median)

START_EPOCH = 1_790_000_000
SESSION_BARS = 400


def frame(sessions: int = 30, bars: int = SESSION_BARS, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    epochs, days, mids = [], [], []
    price = 5000.0
    for session in range(sessions):
        base = START_EPOCH + session * 86_400
        for bar in range(bars):
            epochs.append(base + bar * BAR_SECONDS)
            days.append(session)
            price *= 1 + generator.normal(0, 1e-4)
            mids.append(price)
    return pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "mid": mids,
                         "session_day": days})


def vix_series(data: pd.DataFrame, level: float = 16.0) -> pd.Series:
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    return pd.Series(np.full(len(epochs), level), index=pd.Index(epochs, dtype="int64"))


def test_implied_move_converts_an_annual_vol_to_one_bar():
    expected = 16.0 / 100 * np.sqrt(BAR_SECONDS / TRADING_SECONDS_PER_YEAR)
    assert implied_per_bar_move(np.array([16.0]))[0] == pytest.approx(expected)


def test_a_higher_vix_implies_a_larger_move():
    assert implied_per_bar_move(np.array([30.0]))[0] > implied_per_bar_move(np.array([12.0]))[0]


def test_the_trading_year_is_252_sessions_of_six_and_a_half_hours():
    assert TRADING_SECONDS_PER_YEAR == 252 * 6.5 * 3600


def test_position_in_session_restarts_each_day():
    days = np.array([0, 0, 0, 1, 1, 2])
    assert list(position_in_session(days)) == [0, 1, 2, 0, 1, 0]


def test_realized_vol_does_not_span_the_overnight_gap():
    data = frame(sessions=3, bars=100)
    realized = realized_per_bar_move(data["mid"].to_numpy(float),
                                     data["session_day"].to_numpy(), window=10)
    first_of_second_session = 100
    assert np.isnan(realized[first_of_second_session])


def test_realized_vol_rises_with_a_noisier_series():
    quiet = frame(sessions=2, bars=200, seed=1)
    loud = quiet.assign(mid=5000 * np.cumprod(
        1 + np.random.default_rng(2).normal(0, 1e-3, len(quiet))))
    window = 60
    quiet_vol = np.nanmean(realized_per_bar_move(quiet["mid"].to_numpy(float),
                                                 quiet["session_day"].to_numpy(), window))
    loud_vol = np.nanmean(realized_per_bar_move(loud["mid"].to_numpy(float),
                                                loud["session_day"].to_numpy(), window))
    assert loud_vol > quiet_vol


def test_the_seasonal_profile_never_sees_its_own_session():
    """A day whose values are wildly different must not move its own profile."""
    days = np.repeat(np.arange(10), 4)
    buckets = np.tile(np.arange(4), 10)
    values = np.ones(40)
    values[days == 9] = 1000.0
    profile = trailing_seasonal_median(values, days, buckets, sessions=5)
    assert np.allclose(profile[days == 9], 1.0)


def test_the_seasonal_profile_is_blank_until_enough_sessions_exist():
    days = np.repeat(np.arange(10), 4)
    buckets = np.tile(np.arange(4), 10)
    profile = trailing_seasonal_median(np.ones(40), days, buckets, sessions=5)
    assert np.isnan(profile[days == 0]).all()


def test_the_seasonal_profile_is_per_bucket_not_per_day():
    days = np.repeat(np.arange(12), 2)
    buckets = np.tile(np.array([0, 1]), 12)
    values = np.where(buckets == 0, 1.0, 5.0)
    profile = trailing_seasonal_median(values, days, buckets, sessions=5)
    late = days >= 8
    assert np.allclose(profile[late & (buckets == 0)], 1.0)
    assert np.allclose(profile[late & (buckets == 1)], 5.0)


def test_buckets_divide_the_session_at_the_stated_width():
    days = np.zeros(SEASONAL_BUCKET_BARS * 3, dtype=int)
    buckets = position_in_session(days) // SEASONAL_BUCKET_BARS
    assert list(np.unique(buckets)) == [0, 1, 2]


def test_every_named_column_is_built():
    data = frame()
    built = add_variance_premium_features(data, vix_series(data))
    assert set(VARIANCE_PREMIUM_FEATURES) <= set(built.columns)


def test_a_constant_vix_gives_a_constant_implied_leg():
    data = frame()
    built = add_variance_premium_features(data, vix_series(data))
    implied = built["vrp_implied_per_bar"].to_numpy()
    assert np.nanstd(implied) == pytest.approx(0.0)


def test_the_ratio_rises_when_implied_rises_against_the_same_tape():
    data = frame()
    low = add_variance_premium_features(data, vix_series(data, level=12.0))
    high = add_variance_premium_features(data, vix_series(data, level=30.0))
    assert np.nanmean(high["vrp_ratio_60"]) > np.nanmean(low["vrp_ratio_60"])


def test_a_missing_vix_bar_leaves_a_blank_rather_than_raising():
    data = frame()
    closes = vix_series(data).iloc[:10]
    built = add_variance_premium_features(data, closes)
    assert built["vrp_implied_per_bar"].isna().any()


def test_original_columns_survive():
    data = frame()
    built = add_variance_premium_features(data, vix_series(data))
    assert np.allclose(built["mid"].to_numpy(), data["mid"].to_numpy())
