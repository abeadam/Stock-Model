"""Tests for the VIX-complex features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_features import (LADDER_STEP_BARS, add_volatility_features, lookback_ladder,
                                 volatility_feature_names)

BAR_SECONDS = 5
START_EPOCH = 1_790_000_000


def es_frame(mid_prices: list[float], start_epoch: int = START_EPOCH) -> pd.DataFrame:
    epochs = start_epoch + BAR_SECONDS * np.arange(len(mid_prices))
    return pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "mid": mid_prices})


def series_closes(prices: list[float], start_epoch: int = START_EPOCH) -> pd.Series:
    epochs = start_epoch + BAR_SECONDS * np.arange(len(prices))
    return pd.Series(prices, index=pd.Index(epochs, dtype="int64"), dtype=float)


def test_ladder_starts_at_one_bar_then_steps_regularly():
    assert lookback_ladder(depth=3, step_bars=6) == (1, 6, 12, 18)


def test_deeper_ladder_extends_the_shallower_one():
    shallow = lookback_ladder(depth=2)
    assert lookback_ladder(depth=5)[:len(shallow)] == shallow


def test_ladder_rejects_an_empty_depth():
    with pytest.raises(ValueError):
        lookback_ladder(depth=0)


def test_default_step_spaces_rungs_half_a_minute_apart():
    assert LADDER_STEP_BARS * BAR_SECONDS == 30


def test_names_cover_both_measures_of_every_series_at_every_rung():
    names = volatility_feature_names((1, 6), ("VXX", "UVXY"))
    assert names == ["vol_vxx_ret_1", "vol_vxx_minus_mid_1", "vol_vxx_ret_6", "vol_vxx_minus_mid_6",
                     "vol_uvxy_ret_1", "vol_uvxy_minus_mid_1", "vol_uvxy_ret_6", "vol_uvxy_minus_mid_6"]


def test_return_is_the_percent_change_over_the_window():
    data = es_frame([100.0] * 4)
    closes = series_closes([20.0, 20.0, 21.0, 21.0])
    built = add_volatility_features(data, {"VXX": closes}, ladder=(2,))
    assert built["vol_vxx_ret_2"].iloc[2] == pytest.approx(5.0)


def test_divergence_subtracts_the_mid_move_over_the_same_window():
    """The mid leg divides by the current mid, matching the ms_*_minus_mid convention."""
    data = es_frame([100.0, 100.0, 101.0])
    closes = series_closes([20.0, 20.0, 21.0])
    built = add_volatility_features(data, {"VXX": closes}, ladder=(2,))
    mid_percent = (101.0 - 100.0) / 101.0 * 100
    assert built["vol_vxx_minus_mid_2"].iloc[2] == pytest.approx(5.0 - mid_percent)


def test_divergence_is_blank_where_the_window_crosses_a_session_gap():
    epochs = np.array([START_EPOCH, START_EPOCH + BAR_SECONDS, START_EPOCH + 86_400])
    data = pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "mid": [100.0, 100.0, 101.0]})
    closes = pd.Series([20.0, 20.0, 21.0], index=pd.Index(epochs, dtype="int64"), dtype=float)
    built = add_volatility_features(data, {"VXX": closes}, ladder=(1,))
    assert np.isnan(built["vol_vxx_minus_mid_1"].iloc[2])


def test_missing_series_bars_leave_blanks_rather_than_raising():
    data = es_frame([100.0] * 3)
    closes = series_closes([20.0, 20.0, 20.0]).iloc[:1]
    built = add_volatility_features(data, {"VXX": closes}, ladder=(1,))
    assert built["vol_vxx_ret_1"].isna().all()


def test_every_named_column_is_built():
    data = es_frame([100.0] * 30)
    closes = {series: series_closes(list(np.linspace(20.0, 21.0, 30))) for series in ("VXX", "UVXY")}
    ladder = lookback_ladder(depth=2)
    built = add_volatility_features(data, closes, ladder)
    assert set(volatility_feature_names(ladder, ("VXX", "UVXY"))) <= set(built.columns)


def test_original_columns_survive():
    data = es_frame([100.0, 100.5])
    built = add_volatility_features(data, {"VXX": series_closes([20.0, 20.1])}, ladder=(1,))
    assert list(built["mid"]) == [100.0, 100.5]
