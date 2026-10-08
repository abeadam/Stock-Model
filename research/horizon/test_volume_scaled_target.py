import numpy as np
import pandas as pd

from volume_scaled_target import (BAR_SECONDS, forward_percent, forward_volume, regular_hours,
                                  scaled_targets, trailing_volume)

START = 1_700_000_000


def seconds(n, step=BAR_SECONDS):
    return np.arange(n) * step + START


def test_forward_percent_is_the_move_over_the_next_bars():
    price = np.array([100.0, 101.0, 102.0, 103.0])
    move = forward_percent(price, seconds(4), bars=2)
    assert np.isclose(move[0], 2.0)                      # 100 -> 102
    assert np.isnan(move[-1]) and np.isnan(move[-2])     # no forward data


def test_a_gap_in_the_bars_leaves_no_target():
    gapped = np.array([START, START + 5, START + 3605, START + 3610])
    move = forward_percent(np.array([100.0, 101.0, 102.0, 103.0]), gapped, bars=2)
    assert np.isnan(move[0]) and np.isnan(move[1])


def test_forward_volume_sums_the_bars_being_predicted():
    volume = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert forward_volume(volume, bars=2)[0] == 50.0     # bars 1 and 2: 20 + 30


def test_trailing_volume_sums_the_bars_already_traded():
    volume = np.array([10.0, 20.0, 30.0, 40.0])
    assert trailing_volume(volume, bars=2)[2] == 50.0    # bars 1 and 2: 20 + 30


def test_a_silent_bar_cannot_divide_by_zero():
    scaled = scaled_targets(np.array([100.0, 101.0, 102.0]), np.zeros(3), seconds(3), bars=1)
    assert np.all(np.isfinite(scaled["percent_per_forward_volume"][:2]))


def test_scaling_never_changes_the_sign():
    price = 100 + np.cumsum(np.random.default_rng(2).normal(size=500))
    volume = np.random.default_rng(3).integers(1, 500, size=500).astype(float)
    scaled = scaled_targets(price, volume, seconds(500), bars=5)
    known = np.isfinite(scaled["percent"])
    for name in ("percent_per_forward_volume", "percent_per_trailing_volume"):
        assert np.array_equal(np.sign(scaled[name][known]), np.sign(scaled["percent"][known]))


def test_regular_hours_covers_the_stock_session_only():
    eastern = pd.to_datetime(["2026-06-01 13:29", "2026-06-01 13:30", "2026-06-01 19:59",
                              "2026-06-01 20:00"])                      # UTC, so 09:29..16:00 ET
    assert regular_hours(pd.Series(eastern)).tolist() == [False, True, True, False]
