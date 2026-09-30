"""Tests for the shared deep lookback ladder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_ladder_features import (DEEP_STEP_BARS, add_deep_ladder_features, deep_feature_names,
                                  deep_ladder, session_row_blocks)

BAR_SECONDS = 5
START_EPOCH = 1_790_000_000


def es_frame(bar_count: int, session_days: list | None = None) -> pd.DataFrame:
    epochs = START_EPOCH + BAR_SECONDS * np.arange(bar_count)
    prices = 100.0 + np.arange(bar_count) * 0.01
    return pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "mid": prices, "Close": prices,
                         "session_day": session_days if session_days is not None else [1] * bar_count})


def symbol_closes(bar_count: int) -> pd.Series:
    epochs = START_EPOCH + BAR_SECONDS * np.arange(bar_count)
    return pd.Series(20.0 + np.arange(bar_count) * 0.001,
                     index=pd.Index(epochs, dtype="int64"), dtype=float)


def test_ladder_steps_two_minutes_at_a_time():
    assert deep_ladder(3) == (24, 48, 72)


def test_ladder_starts_beyond_the_reach_families_already_have():
    assert min(deep_ladder(MAX := 6)) == DEEP_STEP_BARS == 24 and MAX == 6


def test_an_empty_ladder_is_allowed_and_adds_nothing():
    data = es_frame(50)
    assert deep_ladder(0) == ()
    assert list(add_deep_ladder_features(data, {"SPY": symbol_closes(50)}, ()).columns) == list(data.columns)


def test_negative_rungs_are_refused():
    with pytest.raises(ValueError):
        deep_ladder(-1)


def test_names_cover_returns_mid_columns_and_scaled_indicators():
    names = deep_feature_names((24,), ("SPY",))
    assert names == ["deep_spy_ret_24", "deep_mid_change_ticks_24", "deep_mid_reversion_24",
                     "deep_es_rsi_24", "deep_es_bollinger_24", "deep_es_ema_dist_24"]


def test_indicator_windows_scale_with_the_rung():
    names = deep_feature_names((24, 48), ("SPY",))
    assert "deep_es_rsi_24" in names and "deep_es_rsi_48" in names


def test_every_named_column_is_built():
    data = es_frame(300)
    ladder = deep_ladder(2)
    built = add_deep_ladder_features(data, {"SPY": symbol_closes(300)}, ladder)
    assert set(deep_feature_names(ladder, ("SPY",))) <= set(built.columns)


def test_mid_change_is_measured_in_ticks():
    data = es_frame(60)
    built = add_deep_ladder_features(data, {"SPY": symbol_closes(60)}, (24,))
    # prices rise 0.01 a bar, so 24 bars is 0.24 points, which is 0.96 ticks
    assert built["deep_mid_change_ticks_24"].iloc[30] == pytest.approx(0.24 / 0.25)


def test_mid_columns_are_blank_across_a_session_gap():
    epochs = np.concatenate([START_EPOCH + BAR_SECONDS * np.arange(30),
                             START_EPOCH + 86_400 + BAR_SECONDS * np.arange(30)])
    data = pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "mid": np.linspace(100, 101, 60),
                         "Close": np.linspace(100, 101, 60), "session_day": [1] * 30 + [2] * 30})
    built = add_deep_ladder_features(data, {}, (24,))
    assert np.isnan(built["deep_mid_change_ticks_24"].iloc[35])


def test_indicators_never_span_two_sessions():
    blocks = session_row_blocks(np.array([1] * 30 + [2] * 30))
    assert [len(block) for block in blocks] == [30, 30]


def test_sessions_are_split_at_every_day_change():
    assert len(session_row_blocks(np.array([1, 1, 2, 2, 3]))) == 3


def test_column_count_grows_linearly_with_depth():
    one, two = len(deep_feature_names(deep_ladder(1))), len(deep_feature_names(deep_ladder(2)))
    assert two == 2 * one


def test_original_columns_survive():
    data = es_frame(60)
    built = add_deep_ladder_features(data, {"SPY": symbol_closes(60)}, (24,))
    assert list(built["mid"])[:3] == list(data["mid"])[:3]
