"""Tests for the TA-Lib candidate sweep."""
from __future__ import annotations

import numpy as np
import pytest
import talib

from talib_candidates import (MONOTONE_OF_ANOTHER, WINDOWS, candidate_name, correlation, evaluate,
                              parameter_settings, windowed_single_series_functions)

ROWS = 20_000


def series(seed: int = 0) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(size=ROWS)) + 1000.0


def test_pointwise_math_transforms_are_excluded():
    """Monotone maps of one column give a tree identical splits, so they cannot help."""
    chosen = set(windowed_single_series_functions())
    assert not chosen & set(talib.get_function_groups()["Math Transform"])


def test_linearreg_angle_is_excluded_as_arctan_of_slope():
    assert "LINEARREG_ANGLE" in MONOTONE_OF_ANOTHER
    assert "LINEARREG_ANGLE" not in windowed_single_series_functions()


def test_ohlcv_functions_are_excluded():
    chosen = set(windowed_single_series_functions())
    for name in ("ATR", "CDLDOJI", "OBV", "MFI", "ADX"):
        assert name not in chosen


def test_familiar_single_series_functions_are_included():
    chosen = set(windowed_single_series_functions())
    for name in ("SMA", "EMA", "RSI", "STDDEV", "MOM", "LINEARREG_SLOPE"):
        assert name in chosen


def test_a_window_function_gets_one_setting_per_window():
    assert len(parameter_settings("SMA")) == len(WINDOWS)


def test_settings_carry_the_requested_windows():
    windows = [setting["timeperiod"] for setting in parameter_settings("RSI")]
    assert windows == list(WINDOWS)


def test_a_function_without_a_window_gets_its_defaults_once():
    assert len(parameter_settings("MACD")) == 1


def test_evaluate_returns_one_column_for_a_single_output_function():
    assert len(evaluate("SMA", series(), {"timeperiod": 30})) == 1


def test_evaluate_returns_every_output_of_a_multi_output_function():
    assert len(evaluate("MINMAX", series(), {"timeperiod": 30})) == 2


def test_evaluate_survives_a_function_that_cannot_run():
    """A window longer than the series must give back a result, not raise mid-sweep."""
    outputs = evaluate("SMA", np.full(5, np.nan), {"timeperiod": 30})
    assert isinstance(outputs, list)
    assert all(np.isnan(column).all() for column in outputs)


def test_correlation_finds_a_planted_relationship():
    values = series()
    assert correlation(values, values * 3.0 + 1.0) == pytest.approx(1.0, abs=1e-6)


def test_correlation_is_zero_for_a_constant_column():
    assert correlation(np.ones(ROWS), series()) == 0.0


def test_correlation_is_zero_when_too_few_rows_overlap():
    values = series()
    target = np.full(ROWS, np.nan)
    target[:100] = values[:100]
    assert correlation(values, target) == 0.0


def test_correlation_ignores_the_sign():
    values = series()
    assert correlation(values, -values) == pytest.approx(1.0, abs=1e-6)


def test_candidate_names_carry_column_function_and_window():
    assert candidate_name("xa_spy_ret_1", "RSI", {"timeperiod": 12}, 0, 1) == "ta_xa_spy_ret_1_rsi_12"


def test_multi_output_names_are_distinct():
    first = candidate_name("mid", "MINMAX", {"timeperiod": 6}, 0, 2)
    second = candidate_name("mid", "MINMAX", {"timeperiod": 6}, 1, 2)
    assert first != second
