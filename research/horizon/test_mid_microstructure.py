import numpy as np
import pandas as pd

from mid_microstructure import (BAR_SECONDS, MID_FEATURES, TICK, add_mid_features, mid_volatility,
                                trailing_change)

START = 1_700_000_000


def seconds(n, step=BAR_SECONDS):
    return np.arange(n) * step + START


def es_frame(mid, close=None, high=None, low=None, step=BAR_SECONDS):
    close = mid if close is None else close
    return pd.DataFrame({"time": pd.to_datetime(seconds(len(mid), step), unit="s"), "mid": mid,
                         "Close": close, "High": mid if high is None else high,
                         "Low": mid if low is None else low, "bid": mid - TICK / 2, "ask": mid + TICK / 2})


def returns_for(frame, value=0.0):
    from mid_microstructure import DIVERGENCE_WINDOWS
    return {f"{symbol}_{bars}": np.full(len(frame), value)
            for symbol, windows in DIVERGENCE_WINDOWS.items() for bars in windows}


def test_trailing_change_looks_backwards_only():
    values = np.array([1.0, 2.0, 4.0, 7.0])
    change = trailing_change(values, seconds(4), bars=1)
    assert np.isnan(change[0]) and change[1] == 1.0 and change[3] == 3.0


def test_a_gap_leaves_the_change_blank():
    gapped = np.array([START, START + 5, START + 605, START + 610])
    change = trailing_change(np.array([1.0, 2.0, 3.0, 4.0]), gapped, bars=1)
    assert np.isnan(change[2]) and change[3] == 1.0


def test_mid_change_is_counted_in_ticks():
    mid = np.arange(20) * TICK + 5000
    frame = add_mid_features(es_frame(mid), returns_for(es_frame(mid)))
    assert np.isclose(frame["ms_mid_change_ticks_1"].to_numpy()[5], 1.0)
    assert np.isclose(frame["ms_mid_change_ticks_3"].to_numpy()[5], 3.0)


def test_the_z_score_judges_a_move_against_recent_movement():
    quiet = 5000 + np.zeros(200)
    quiet[1::2] = 5000 + TICK                       # a one-tick shuffle
    mid = np.concatenate([quiet, [5000 + 10 * TICK]])
    frame = add_mid_features(es_frame(mid), returns_for(es_frame(mid)))
    assert frame["ms_mid_change_z_1"].to_numpy()[-1] > 5


def test_reversion_is_distance_from_the_recent_average():
    mid = np.full(80, 5000.0)
    mid[-1] = 5000 + 4 * TICK
    frame = add_mid_features(es_frame(mid), returns_for(es_frame(mid)))
    assert frame["ms_mid_reversion_12"].to_numpy()[-1] > 3.5


def test_wicks_are_measured_in_ticks_from_the_close():
    mid = np.full(10, 5000.0)
    frame = add_mid_features(es_frame(mid, high=mid + 2 * TICK, low=mid - TICK), returns_for(es_frame(mid)))
    assert np.allclose(frame["ms_wick_up_ticks"], 2.0)
    assert np.allclose(frame["ms_wick_down_ticks"], 1.0)
    assert np.allclose(frame["ms_wick_asymmetry_ticks"], 1.0)


def test_divergence_is_the_stock_move_less_the_mid_move():
    mid = np.full(10, 5000.0)                        # the mid does not move
    frame = add_mid_features(es_frame(mid), returns_for(es_frame(mid), value=0.5))
    assert np.allclose(frame["ms_spy_minus_mid_1"].to_numpy()[1:], 0.5)


def test_each_symbol_is_read_at_its_own_windows():
    """Oil and the dollar are given slower looks than the mega-caps."""
    from mid_microstructure import DIVERGENCE_WINDOWS
    assert DIVERGENCE_WINDOWS["SPY"] == (1, 2, 3)
    assert min(DIVERGENCE_WINDOWS["USO"]) >= 3
    mid = np.full(40, 5000.0)
    frame = add_mid_features(es_frame(mid), returns_for(es_frame(mid)))
    assert "ms_uso_minus_mid_6" in frame.columns
    assert "ms_uso_minus_mid_1" not in frame.columns      # oil is not read at one bar
    assert "ms_uup_minus_mid_12" in frame.columns


def test_every_declared_feature_is_produced():
    mid = 5000 + np.cumsum(np.random.default_rng(1).normal(size=300)) * TICK
    frame = add_mid_features(es_frame(mid), returns_for(es_frame(mid)))
    assert all(name in frame.columns for name in MID_FEATURES)
