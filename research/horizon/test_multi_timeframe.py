import numpy as np
import pandas as pd

from multi_timeframe_features import (BASE_BAR_SECONDS, add_span_features, add_timeframe_features,
                                      indicators_of, widen)
from technical_indicators import money_flow_index

START = 1_700_000_100        # a multiple of 300, so it starts every bar width cleanly


def five_second_bars(count, start=START, step=BASE_BAR_SECONDS):
    seconds = np.arange(count) * step + start
    return pd.DataFrame({"seconds": seconds, "close": np.arange(count, dtype=float) + 100,
                         "high": np.arange(count, dtype=float) + 100.5,
                         "low": np.arange(count, dtype=float) + 99.5,
                         "volume": np.ones(count)})


def test_widening_aggregates_open_high_low_volume():
    wide = widen(five_second_bars(6), 15)                 # two complete 15-second bars
    assert len(wide) == 2
    assert wide["close"].tolist() == [102.0, 105.0]       # last close of each group
    assert wide["high"].tolist() == [102.5, 105.5]
    assert wide["low"].tolist() == [99.5, 102.5]
    assert wide["volume"].tolist() == [3.0, 3.0]


def test_a_wide_bar_closes_at_the_end_of_its_span():
    wide = widen(five_second_bars(6), 15)
    assert (wide["closes_at"] - wide["start"]).tolist() == [15, 15]


def test_incomplete_groups_are_dropped_so_no_bar_spans_a_gap():
    bars = five_second_bars(6)
    with_gap = bars.drop(index=4).reset_index(drop=True)   # one bar missing from the second group
    assert len(widen(with_gap, 15)) == 1


def test_five_second_width_keeps_every_bar():
    assert len(widen(five_second_bars(7), BASE_BAR_SECONDS)) == 7


def test_features_only_use_wide_bars_that_have_already_closed():
    """A 30-second indicator must not change until its bar completes."""
    bars = five_second_bars(600)
    data = pd.DataFrame({"time": pd.to_datetime(bars["seconds"], unit="s")})
    features = add_timeframe_features(data, widths=(30,), bars=bars)
    column = features["mtf_30s_rsi"].to_numpy()
    # Bars 0..5 close at +5..+30; only the last of them may see the first 30-second bar.
    assert np.isnan(column[:5]).all()
    assert not np.isnan(column[5])
    assert column[5] == column[6] == column[7]            # unchanged until the next one closes


def test_indicator_columns_are_named_by_width():
    wide = widen(five_second_bars(120), 20)
    assert list(indicators_of(wide, 20).columns)[:2] == ["mtf_20s_mfi", "mtf_20s_bb_upper_dist"]


def test_money_flow_index_is_100_when_every_bar_rises():
    rising = pd.Series(np.arange(50, dtype=float) + 100)
    mfi = money_flow_index(rising + 0.5, rising - 0.5, rising, pd.Series(np.ones(50)), window=14)
    assert np.isclose(mfi[-1], 100.0)


def test_stretched_windows_reach_the_same_span_at_full_resolution():
    """A 30-second view can be built two ways; the stretched one updates every bar."""
    bars = five_second_bars(600)
    wave = 100 + np.sin(np.arange(600) / 7)                # RSI saturates on a pure ramp
    bars = bars.assign(close=wave, high=wave + 0.5, low=wave - 0.5)
    data = pd.DataFrame({"time": pd.to_datetime(bars["seconds"], unit="s")})
    widened = add_timeframe_features(data, widths=(30,), bars=bars)["mtf_30s_rsi"].to_numpy()
    stretched = add_span_features(data, spans=(30,), bars=bars)["span_30s_rsi"].to_numpy()
    # Over 60 bars the widened view can only change once every six; the stretched one
    # reaches the same 30 seconds of history but moves on every bar.
    assert len(np.unique(widened[100:160])) <= 11
    assert len(np.unique(stretched[100:160])) >= 55
