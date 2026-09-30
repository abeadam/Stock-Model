import numpy as np

from hold_policy import HoldRule, last_bar_before_gap, simulate_hold

RULE = HoldRule(entry_threshold=0.1, exit_threshold=0.0, min_hold_bars=1)


def test_long_holds_until_signal_fades():
    prob = np.array([0.5, 0.7, 0.6, 0.55, 0.4, 0.5])
    close = np.array([100, 100, 101, 102, 103, 104.0])
    seconds = np.arange(6) * 5
    result = simulate_hold(prob - 0.5, close, close, last_bar_before_gap(seconds), RULE, point_value=50, round_trip_cost=5)
    assert result["trades"] == 1
    assert result["gross"] == 3 * 50            # entered at bar 1 (100), exited at bar 4 (103)
    assert result["net"] == 3 * 50 - 5
    assert result["mean_bars_held"] == 3


def test_position_closes_before_gap():
    prob = np.array([0.7, 0.7, 0.7, 0.7])
    close = np.array([100, 101, 102, 90.0])
    seconds = np.array([0, 5, 10, 3600])          # gap after bar 2
    result = simulate_hold(prob - 0.5, close, close, last_bar_before_gap(seconds), RULE, point_value=50, round_trip_cost=5)
    assert result["trades"] == 1                    # bar 3 is the final bar, never entered
    assert result["gross"] == 2 * 50                # exit forced at bar 2, not across the gap


def test_short_mirrors_long():
    prob = np.array([0.3, 0.45, 0.6, 0.5])
    close = np.array([100, 99, 98, 97.0])
    seconds = np.arange(4) * 5
    result = simulate_hold(prob - 0.5, close, close, last_bar_before_gap(seconds), RULE, point_value=50, round_trip_cost=5)
    assert result["trades"] == 1
    assert result["gross"] == 2 * 50                # short at 100, covered at bar 2 (98) when prob > 0.5


def test_min_hold_skips_early_exit_signal():
    prob = np.array([0.7, 0.4, 0.4, 0.4, 0.5])
    close = np.array([100, 101, 102, 103, 104.0])
    seconds = np.arange(5) * 5
    rule = HoldRule(entry_threshold=0.1, exit_threshold=0.0, min_hold_bars=3)
    result = simulate_hold(prob - 0.5, close, close, last_bar_before_gap(seconds), rule, point_value=50, round_trip_cost=5)
    assert result["gross"] == 3 * 50


def test_fill_delay_uses_the_next_bar_close():
    prob = np.array([0.7, 0.6, 0.4, 0.5, 0.5])
    close = np.array([100, 101, 102, 104, 105.0])
    seconds = np.arange(5) * 5
    rule = HoldRule(entry_threshold=0.1, exit_threshold=0.0, min_hold_bars=1, fill_delay_bars=1)
    result = simulate_hold(prob - 0.5, close, close, last_bar_before_gap(seconds), rule, point_value=50, round_trip_cost=5)
    assert result["trades"] == 1
    assert result["gross"] == (104 - 101) * 50       # decided at bars 0 and 2, filled at bars 1 and 3


def test_fill_delay_never_crosses_a_gap():
    prob = np.array([0.7, 0.7, 0.7, 0.7])
    close = np.array([100, 101, 102, 90.0])
    seconds = np.array([0, 5, 10, 3600])
    rule = HoldRule(entry_threshold=0.1, exit_threshold=0.0, min_hold_bars=1, fill_delay_bars=1)
    result = simulate_hold(prob - 0.5, close, close, last_bar_before_gap(seconds), rule, point_value=50, round_trip_cost=5)
    assert result["gross"] == (102 - 101) * 50       # forced exit fills at bar 2, not after the gap


def test_crossing_the_spread_buys_at_ask_and_sells_at_bid():
    prob = np.array([0.7, 0.6, 0.4, 0.5])
    bid = np.array([100.0, 101.0, 102.0, 103.0])
    ask = bid + 0.25
    seconds = np.arange(4) * 5
    rule = HoldRule(entry_threshold=0.1, exit_threshold=0.0, min_hold_bars=1)
    long_result = simulate_hold(prob - 0.5, ask, bid, last_bar_before_gap(seconds), rule, point_value=50, round_trip_cost=5)
    assert long_result["gross"] == (102.0 - 100.25) * 50          # buy ask at bar 0, sell bid at bar 2
    short_result = simulate_hold(0.5 - prob, ask, bid, last_bar_before_gap(seconds), rule, point_value=50, round_trip_cost=5)
    assert short_result["gross"] == (100.0 - 102.25) * 50         # sell bid at bar 0, buy ask at bar 2
