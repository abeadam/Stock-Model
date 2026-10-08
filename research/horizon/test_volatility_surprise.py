"""Tests for the implied-volatility surprise feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_surprise import (SURPRISE_WINDOWS, VOLATILITY_SURPRISE_FEATURES,
                                 VOLATILITY_SURPRISE_LEGS, add_volatility_surprise,
                                 block_range_pct, delta_implied, delta_realized,
                                 span_is_contiguous)

BAR_SECONDS = 5
START_EPOCH = 1_790_000_000


def es_frame(closes: list[float], half_ranges: list[float] | None = None) -> pd.DataFrame:
    epochs = START_EPOCH + BAR_SECONDS * np.arange(len(closes))
    spread = np.array(half_ranges if half_ranges is not None else [0.5] * len(closes))
    return pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "Close": closes,
                         "High": np.array(closes) + spread, "Low": np.array(closes) - spread})


def series_closes(prices: list[float], start_epoch: int = START_EPOCH) -> pd.Series:
    epochs = start_epoch + BAR_SECONDS * np.arange(len(prices))
    return pd.Series(prices, index=pd.Index(epochs, dtype="int64"), dtype=float)


def test_windows_stay_shallow():
    """Deep lookback damaged the volatility head; this feature must not reach far."""
    assert max(SURPRISE_WINDOWS) <= 12


def test_contiguity_covers_the_whole_double_window():
    """A gap anywhere inside the reach must blank the row, not only one at its edge."""
    epochs = np.array([START_EPOCH + BAR_SECONDS * bar for bar in range(6)])
    epochs[3:] += 3600                                  # a session break partway through
    assert not span_is_contiguous(epochs, bars=4)[5]


def test_contiguity_passes_on_unbroken_bars():
    epochs = START_EPOCH + BAR_SECONDS * np.arange(10)
    assert span_is_contiguous(epochs, bars=4)[9]


def test_block_range_is_the_high_low_span_as_a_percent():
    data = es_frame([100.0] * 6, half_ranges=[1.0] * 6)
    span = block_range_pct(data["High"].to_numpy(), data["Low"].to_numpy(),
                           data["Close"].to_numpy(), window=3, offset=0)
    assert span[5] == pytest.approx(2.0)                # 101 - 99 over 100, as a percent


def test_the_prior_block_does_not_overlap_the_recent_one():
    closes = [100.0] * 3 + [200.0] * 3
    data = es_frame(closes, half_ranges=[1.0] * 6)
    prior = block_range_pct(data["High"].to_numpy(), data["Low"].to_numpy(),
                            data["Close"].to_numpy(), window=3, offset=3)
    assert prior[5] == pytest.approx(2.0)               # reads the first block, not the second


def test_a_widening_tape_gives_a_positive_realized_change():
    closes = [100.0] * 8
    half = [0.25] * 4 + [1.0] * 4
    data = es_frame(closes, half_ranges=half)
    change = delta_realized(data["High"].to_numpy(), data["Low"].to_numpy(),
                            data["Close"].to_numpy(), window=4)
    assert change[7] > 0


def test_a_quietening_tape_gives_a_negative_realized_change():
    closes = [100.0] * 8
    data = es_frame(closes, half_ranges=[1.0] * 4 + [0.25] * 4)
    change = delta_realized(data["High"].to_numpy(), data["Low"].to_numpy(),
                            data["Close"].to_numpy(), window=4)
    assert change[7] < 0


def test_a_flat_tape_gives_no_realized_change():
    data = es_frame([100.0] * 8, half_ranges=[0.5] * 8)
    change = delta_realized(data["High"].to_numpy(), data["Low"].to_numpy(),
                            data["Close"].to_numpy(), window=4)
    assert change[7] == pytest.approx(0.0)


def test_implied_change_is_the_log_ratio_over_the_window():
    epochs = START_EPOCH + BAR_SECONDS * np.arange(6)
    closes = series_closes([20.0, 20.0, 20.0, 22.0, 22.0, 22.0])
    change = delta_implied(closes, epochs, window=3)
    assert change[3] == pytest.approx(np.log(22.0 / 20.0))


def test_surprise_is_implied_change_less_realized_change():
    data = es_frame([100.0] * 30, half_ranges=[0.25] * 15 + [1.0] * 15)
    built = add_volatility_surprise(data, {"VXX": series_closes(list(np.linspace(20.0, 23.0, 30)))})
    row = 29
    realized = built["vs_delta_realized_6"].to_numpy()[row]
    implied = built["vs_vxx_delta_implied_6"].to_numpy()[row]
    surprise = built["vs_vxx_surprise_6"].to_numpy()[row]
    assert np.isfinite(surprise)
    assert surprise == pytest.approx(implied - realized)


def test_implied_rising_against_a_flat_tape_is_a_positive_surprise():
    data = es_frame([100.0] * 26, half_ranges=[0.5] * 26)
    closes = series_closes([20.0] * 20 + [24.0] * 6)    # implied jumps inside the last window
    built = add_volatility_surprise(data, {"VXX": closes})
    assert built["vs_vxx_surprise_6"].to_numpy()[25] > 0


def test_every_named_column_is_built():
    data = es_frame([100.0] * 60)
    closes = {series: series_closes(list(np.linspace(20.0, 21.0, 60)))
              for series in ("VXX", "UVXY")}
    built = add_volatility_surprise(data, closes)
    assert set(VOLATILITY_SURPRISE_LEGS + VOLATILITY_SURPRISE_FEATURES) <= set(built.columns)


def test_columns_are_blank_across_a_session_gap():
    epochs = np.concatenate([START_EPOCH + BAR_SECONDS * np.arange(20),
                             START_EPOCH + 86_400 + BAR_SECONDS * np.arange(20)])
    data = pd.DataFrame({"time": pd.to_datetime(epochs, unit="s"), "Close": [100.0] * 40,
                         "High": [100.5] * 40, "Low": [99.5] * 40})
    closes = pd.Series([20.0] * 40, index=pd.Index(epochs, dtype="int64"), dtype=float)
    built = add_volatility_surprise(data, closes_by_series={"VXX": closes})
    assert np.isnan(built["vs_vxx_surprise_6"].to_numpy()[22])


def test_a_missing_series_bar_leaves_a_blank():
    data = es_frame([100.0] * 40)
    built = add_volatility_surprise(data, {"VXX": series_closes([20.0] * 5)})
    assert built["vs_vxx_delta_implied_6"].isna().all()


def test_original_columns_survive():
    data = es_frame([100.0, 100.5])
    built = add_volatility_surprise(data, {"VXX": series_closes([20.0, 20.1])})
    assert list(built["Close"]) == [100.0, 100.5]
