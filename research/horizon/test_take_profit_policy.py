import numpy as np
import pytest

from hold_policy import last_bar_before_gap
from take_profit_policy import recent_volatility, simulate_take_profit, take_profit_points


def run(score, buy, sell, target, seconds=None, threshold=0.1, stop=None):
    seconds = np.arange(len(buy)) * 5 if seconds is None else np.asarray(seconds)
    return simulate_take_profit(np.asarray(score, float), np.asarray(buy, float), np.asarray(sell, float),
                                np.asarray(target, float), last_bar_before_gap(seconds), threshold,
                                point_value=50, round_trip_cost=5,
                                stop_points=None if stop is None else np.asarray(stop, float))


def test_long_exits_at_first_close_past_the_target():
    mid = [100, 100.25, 100.5, 101, 99]
    result = run(score=[0.2, 0, 0, 0, 0], buy=mid, sell=mid, target=[0.5] * 5)
    assert result["trades"] == 1 and result["target_rate"] == 1.0
    assert result["gross"] == 0.5 * 50                              # exits at bar 2 (100.5), not later
    assert result["mean_bars_held"] == 2


def test_no_target_exits_at_period_end():
    mid = [100, 99.75, 99.5, 99.0, 50.0]
    seconds = [0, 5, 10, 15, 4000]                                  # gap after bar 3 ends the period
    result = run(score=[0.2, 0, 0, 0, 0], buy=mid, sell=mid, target=[0.5] * 5, seconds=seconds)
    assert result["target_rate"] == 0.0
    assert result["gross"] == -1.0 * 50                             # held to bar 3, not across the gap


def test_short_crossing_the_spread():
    bid = np.array([100.0, 99.75, 99.25, 99.0])
    ask = bid + 0.25
    result = run(score=[-0.2, 0, 0, 0], buy=ask, sell=bid, target=[0.5] * 4)
    # sells the bid at 100, needs the ask at or below 99.5: bar 2 ask is 99.5
    assert result["gross"] == (100.0 - 99.5) * 50


def test_reentry_after_exit():
    mid = [100, 100.5, 100.5, 101]
    result = run(score=[0.2, 0.2, 0.2, 0.2], buy=mid, sell=mid, target=[0.5] * 4)
    assert result["trades"] == 2                                    # 0->1 hits target, then 2->3 hits target


def test_target_has_a_one_tick_floor():
    assert list(take_profit_points(np.array([0.05, 0.5, np.nan]))) == [0.25, 1.0, 0.25]


def test_volatility_measures():
    mid = np.linspace(100, 100 + 61 * 0.25, 62)                     # steady 0.25 steps
    seconds = np.arange(62) * 5
    changes = recent_volatility(mid, seconds, "changes")
    levels = recent_volatility(mid, seconds, "levels")
    assert np.isnan(changes[59]) and changes[60] == pytest.approx(0.0, abs=1e-9)
    assert levels[59] == pytest.approx(np.std(mid[:60], ddof=1))


def test_volatility_is_nan_across_a_gap():
    mid = np.arange(70, dtype=float)
    seconds = np.arange(70) * 5
    seconds[40:] += 3600
    assert np.isnan(recent_volatility(mid, seconds, "levels")[65])


def test_opposite_signal_closes_and_can_reverse():
    mid = [100, 99.75, 99.5, 99.0, 98.75, 98.5]
    score = [0.2, 0, -0.2, -0.2, 0, 0]
    result = run(score=score, buy=mid, sell=mid, target=[1.0] * 6)
    # long at 100 closed at bar 2 (99.5) by the short signal; the signal is still there
    # at bar 3, so a short opens at 99.0 and runs to the period end at bar 5 (98.5)
    assert result["trades"] == 2
    assert result["reversal_rate"] == 0.5
    assert result["gross"] == (-0.5 + 0.5) * 50


def test_one_bar_opposite_signal_closes_without_reversing():
    mid = [100, 99.75, 99.5, 99.25, 99.0]
    result = run(score=[0.2, 0, -0.2, 0, 0], buy=mid, sell=mid, target=[1.0] * 5)
    assert result["trades"] == 1 and result["reversal_rate"] == 1.0
    assert result["gross"] == -0.5 * 50


def test_stop_exits_a_losing_long():
    mid = [100, 99.75, 99.0, 98.0, 101.0]
    result = run(score=[0.2, 0, 0, 0, 0], buy=mid, sell=mid, target=[1.0] * 5, stop=[1.0] * 5)
    assert result["stop_rate"] == 1.0
    assert result["gross"] == -1.0 * 50                              # stopped at bar 2 (99.0), not held to 101


def test_without_a_stop_the_loser_is_held():
    mid = [100, 99.75, 99.0, 98.0, 101.0]
    result = run(score=[0.2, 0, 0, 0, 0], buy=mid, sell=mid, target=[1.0] * 5)
    assert result["target_rate"] == 1.0 and result["gross"] == 1.0 * 50
