import numpy as np
import pandas as pd
import pytest

from realized_volatility_features import (REALIZED_VOLATILITY_FEATURES,
                                          REALIZED_VOLATILITY_WINDOWS,
                                          add_realized_volatility, realized_volatility)

WINDOW = 4


def two_sessions(first, second):
    prices = np.array(first + second, dtype=float)
    days = np.array([1] * len(first) + [2] * len(second))
    return prices, days


def test_the_first_bar_of_each_session_has_no_return():
    prices, days = two_sessions([100.0] * 8, [500.0] * 8)
    result = realized_volatility(prices, days, WINDOW)
    assert np.isnan(result[0]) and np.isnan(result[8])


def test_no_window_spans_the_overnight_gap():
    """The jump from 100 to 500 must not register as volatility in the second session."""
    prices, days = two_sessions([100.0] * 8, [500.0] * 8)
    result = realized_volatility(prices, days, WINDOW)
    second = result[8:]
    assert np.nanmax(second) == pytest.approx(0.0), "the overnight jump leaked into the window"


def test_a_flat_session_has_zero_volatility():
    prices, days = two_sessions([100.0] * 8, [100.0] * 8)
    assert np.nanmax(realized_volatility(prices, days, WINDOW)) == pytest.approx(0.0)


def test_a_moving_session_has_positive_volatility():
    prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0], dtype=float)
    days = np.ones(len(prices))
    assert np.nanmax(realized_volatility(prices, days, 3)) > 0


def test_it_matches_a_hand_computed_standard_deviation():
    prices = np.array([100.0, 110.0, 121.0, 133.1], dtype=float)
    days = np.ones(len(prices))
    result = realized_volatility(prices, days, 3)
    assert result[-1] == pytest.approx(pd.Series([0.1, 0.1, 0.1]).std())


def test_a_window_is_blank_until_half_of_it_is_available():
    prices = np.arange(100.0, 110.0)
    days = np.ones(len(prices))
    result = realized_volatility(prices, days, 6)
    assert np.isnan(result[2]) and not np.isnan(result[4])


def test_the_result_is_as_long_as_the_input():
    prices, days = two_sessions([100.0] * 5, [100.0] * 7)
    assert len(realized_volatility(prices, days, WINDOW)) == 12


def test_a_single_session_needs_no_boundary():
    prices = np.linspace(100.0, 110.0, 20)
    result = realized_volatility(prices, np.ones(20), 5)
    assert np.isnan(result[0]) and np.isfinite(result[-1])


def test_the_windows_are_sixty_and_one_eighty_bars():
    assert REALIZED_VOLATILITY_WINDOWS == (60, 180)


def test_the_column_names_carry_their_windows():
    assert REALIZED_VOLATILITY_FEATURES == ["realized_vol_60", "realized_vol_180"]


def test_adding_them_leaves_the_original_columns_alone():
    prices, days = two_sessions([100.0] * 200, [100.0] * 200)
    frame = pd.DataFrame({"Close": prices, "other": 1})
    result = add_realized_volatility(frame, prices, days)
    assert list(result.columns) == ["Close", "other", *REALIZED_VOLATILITY_FEATURES]


def test_adding_them_does_not_mutate_the_input():
    prices, days = two_sessions([100.0] * 200, [100.0] * 200)
    frame = pd.DataFrame({"Close": prices})
    add_realized_volatility(frame, prices, days)
    assert list(frame.columns) == ["Close"]


def test_the_longer_window_is_blank_for_longer():
    prices = np.linspace(100.0, 200.0, 400)
    days = np.ones(400)
    frame = add_realized_volatility(pd.DataFrame({"Close": prices}), prices, days)
    assert frame["realized_vol_180"].isna().sum() > frame["realized_vol_60"].isna().sum()


def test_mid_and_close_give_the_same_answer_for_the_same_series():
    """The unified function replaced one over mid and one over close."""
    prices = np.linspace(100.0, 110.0, 50)
    days = np.ones(50)
    assert np.allclose(realized_volatility(prices, days, 10),
                       realized_volatility(prices.copy(), days, 10), equal_nan=True)
