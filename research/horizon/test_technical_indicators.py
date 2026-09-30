import numpy as np
import pandas as pd

from indicator_features import INDICATOR_NAMES, day_indicators, feature_names
from technical_indicators import (NEUTRAL_RSI, bollinger_position, distance_from_average,
                                  macd_histogram, realized_volatility, relative_strength_index,
                                  volatility_ratio)

BAR_SECONDS = 5


def series(values):
    return pd.Series(values, index=np.arange(len(values)) * BAR_SECONDS, dtype=float)


def test_rsi_is_100_when_every_bar_rises_and_0_when_every_bar_falls():
    rising = relative_strength_index(series(np.arange(100, 200)), window=12)
    falling = relative_strength_index(series(np.arange(200, 100, -1)), window=12)
    assert np.isclose(rising[-1], 100.0)
    assert np.isclose(falling[-1], 0.0)


def test_rsi_of_a_flat_series_is_neutral():
    flat = relative_strength_index(series([100.0] * 50), window=12)
    assert np.allclose(flat, NEUTRAL_RSI)


def test_bollinger_position_is_a_half_at_the_mean_and_rises_towards_the_upper_band():
    window = 4
    prices = series([10.0, 12.0, 8.0, 10.0])                  # mean 10, the last close sits on it
    assert np.isclose(bollinger_position(prices, window)[-1], 0.5)
    # [10, 10, 10, 12]: mean 10.5, sd 1.0, band 8.5..12.5, so the close sits at (12-8.5)/4.
    assert np.isclose(bollinger_position(series([10.0, 10.0, 10.0, 12.0]), window)[-1], 0.875)


def test_distance_from_average_is_zero_on_a_flat_series_and_positive_when_rising():
    assert np.allclose(distance_from_average(series([50.0] * 30), window=12), 0.0)
    assert distance_from_average(series(np.arange(50, 80, dtype=float)), window=12)[-1] > 0


def test_realized_volatility_is_zero_when_nothing_moves():
    assert np.isclose(realized_volatility(series([100.0] * 30), window=12)[-1], 0.0)


def test_volatility_ratio_exceeds_one_when_the_symbol_wakes_up():
    quiet = [100.0 + 0.01 * (index % 2) for index in range(80)]
    agitated = [100.0 + 2.0 * (index % 2) for index in range(20)]
    ratio = volatility_ratio(series(quiet + agitated), fast_window=12, slow_window=60)
    assert ratio[-1] > 1.0


def test_macd_histogram_is_zero_on_a_flat_series():
    assert np.allclose(macd_histogram(series([100.0] * 60)), 0.0)


def test_indicators_never_look_forward():
    """Changing a later bar must not change an earlier bar's indicators."""
    prices = list(np.linspace(100, 110, 80))
    before = day_indicators(series(prices))
    after = day_indicators(series(prices[:-1] + [500.0]))
    assert np.allclose(before.iloc[:-1].to_numpy(), after.iloc[:-1].to_numpy(), equal_nan=True)


def test_day_indicators_keeps_the_bar_index_and_every_column():
    prices = series(np.linspace(100, 101, 120))
    indicators = day_indicators(prices)
    assert list(indicators.columns) == list(INDICATOR_NAMES)
    assert np.array_equal(indicators.index.to_numpy(), prices.index.to_numpy())


def test_feature_names_are_prefixed_and_lowercase():
    assert feature_names(("TLT", "BRK.B"))[:2] == ["ti_tlt_rsi_12", "ti_tlt_rsi_60"]
    assert all(name.startswith("ti_brk.b_") for name in feature_names(("BRK.B",)))
