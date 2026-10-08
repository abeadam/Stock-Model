"""Tests for quote-priced passive entries."""
from __future__ import annotations

import numpy as np
import pytest

from quote_passive_entry import (CROSS_EXIT, LONG, PASSIVE_EXIT, SHORT, QuotePassiveRule, Quotes,
                                 TICK, entry_limit, session_end_index, simulate, simulate_crossing)

POINT_VALUE = 50.0
NO_COST = 0.0


def flat_quotes(bar_count: int, bid: float = 100.00, spread: float = TICK) -> Quotes:
    """A still book: every bar quotes the same bid and ask and trades between them."""
    bids = np.full(bar_count, bid)
    asks = bids + spread
    is_last = np.zeros(bar_count, bool)
    is_last[-1] = True
    return Quotes(bid=bids, ask=asks, high=asks.copy(), low=bids.copy(), is_last=is_last)


def rule(**overrides) -> QuotePassiveRule:
    settings = {"threshold": 0.01, "entry_offset_ticks": 0, "wait_bars": 3, "hold_bars": 2,
                "exit_mode": CROSS_EXIT}
    return QuotePassiveRule(**{**settings, **overrides})


def test_a_long_rests_on_the_bid():
    quotes = flat_quotes(5)
    assert entry_limit(quotes, 0, LONG, offset_ticks=0) == 100.00


def test_a_short_rests_on_the_ask():
    quotes = flat_quotes(5)
    assert entry_limit(quotes, 0, SHORT, offset_ticks=0) == 100.25


def test_an_offset_rests_behind_the_touch_on_both_sides():
    quotes = flat_quotes(5)
    assert entry_limit(quotes, 0, LONG, offset_ticks=1) == 100.00 - TICK
    assert entry_limit(quotes, 0, SHORT, offset_ticks=1) == 100.25 + TICK


def test_a_touch_is_not_a_fill():
    """The book never trades below the bid, so a limit resting on it is never filled."""
    quotes = flat_quotes(10)
    result = simulate(np.full(10, 0.2), quotes, rule(), POINT_VALUE, NO_COST)
    assert result["orders"] > 0 and result["fills"] == 0


def test_trading_through_the_limit_fills_it():
    quotes = flat_quotes(10)
    quotes.low[3] = 99.50                       # a later bar trades below the resting bid
    result = simulate(np.full(10, 0.2), quotes, rule(), POINT_VALUE, NO_COST)
    assert result["fills"] == 1


def test_an_unfilled_signal_still_counts_as_an_order():
    quotes = flat_quotes(10)
    result = simulate(np.full(10, 0.2), quotes, rule(), POINT_VALUE, NO_COST)
    assert result["orders"] >= 1 and result["fill_rate"] == 0.0


def test_a_long_filled_then_crossed_out_pays_the_spread_once():
    """Bought at the bid, sold at the bid: flat in a still book, which is the mid assumption."""
    quotes = flat_quotes(12)
    quotes.low[1] = 99.50
    result = simulate(np.full(12, 0.2), quotes, rule(wait_bars=1, hold_bars=2), POINT_VALUE, NO_COST)
    assert result["fills"] == 1 and result["gross"] == pytest.approx(0.0)


def test_a_passive_exit_that_fills_earns_the_spread():
    quotes = flat_quotes(12)
    quotes.low[1] = 99.50                       # entry fills at the bid, 100.00
    quotes.high[4] = 100.50                     # later trade lifts through the resting ask
    result = simulate(np.full(12, 0.2), quotes,
                      rule(wait_bars=1, hold_bars=2, exit_mode=PASSIVE_EXIT), POINT_VALUE, NO_COST)
    assert result["gross"] == pytest.approx(TICK * POINT_VALUE)


def test_a_passive_exit_that_never_fills_falls_back_to_crossing():
    quotes = flat_quotes(12)
    quotes.low[1] = 99.50
    result = simulate(np.full(12, 0.2), quotes,
                      rule(wait_bars=1, hold_bars=2, exit_mode=PASSIVE_EXIT), POINT_VALUE, NO_COST)
    assert result["fills"] == 1 and result["gross"] == pytest.approx(0.0)


def test_a_short_is_the_mirror_of_a_long():
    quotes = flat_quotes(12)
    quotes.high[1] = 100.75                     # trades up through the resting ask
    result = simulate(np.full(12, -0.2), quotes, rule(wait_bars=1, hold_bars=2), POINT_VALUE, NO_COST)
    assert result["fills"] == 1 and result["gross"] == pytest.approx(0.0)


def test_signals_below_the_threshold_place_no_order():
    quotes = flat_quotes(10)
    assert simulate(np.full(10, 0.005), quotes, rule(threshold=0.01), POINT_VALUE, NO_COST)["orders"] == 0


def test_cost_is_charged_once_per_fill():
    quotes = flat_quotes(12)
    quotes.low[1] = 99.50
    result = simulate(np.full(12, 0.2), quotes, rule(wait_bars=1, hold_bars=2), POINT_VALUE, 5.0)
    assert result["net"] == pytest.approx(result["gross"] - 5.0 * result["fills"])


def test_no_order_rests_across_a_session_gap():
    quotes = flat_quotes(10)
    quotes.is_last[4] = True
    quotes.low[6] = 99.50                       # a fill on the far side of the gap
    result = simulate(np.concatenate([np.full(5, 0.2), np.zeros(5)]), quotes,
                      rule(wait_bars=6), POINT_VALUE, NO_COST)
    assert result["fills"] == 0


def test_session_end_index_points_at_each_block_end():
    is_last = np.array([False, False, True, False, True])
    assert list(session_end_index(is_last)) == [2, 2, 2, 4, 4]


def test_only_one_position_is_open_at_a_time():
    quotes = flat_quotes(20)
    quotes.low[:] = 99.50                       # every bar would fill an order
    result = simulate(np.full(20, 0.2), quotes, rule(wait_bars=1, hold_bars=5), POINT_VALUE, NO_COST)
    assert result["fills"] <= 4


def test_a_wider_spread_makes_a_crossed_exit_cost_more():
    quotes = flat_quotes(12, spread=4 * TICK)
    quotes.low[1] = 99.00
    narrow = flat_quotes(12)
    narrow.low[1] = 99.50
    wide_result = simulate(np.full(12, 0.2), quotes, rule(wait_bars=1, hold_bars=2), POINT_VALUE, NO_COST)
    narrow_result = simulate(np.full(12, 0.2), narrow, rule(wait_bars=1, hold_bars=2), POINT_VALUE, NO_COST)
    assert wide_result["fills"] == narrow_result["fills"] == 1


def test_crossing_pays_the_spread_twice():
    """Bought at the ask, sold at the bid, in a book that never moved: down one spread."""
    quotes = flat_quotes(12)
    result = simulate_crossing(np.full(12, 0.2), quotes, threshold=0.01, hold_bars=2,
                               point_value=POINT_VALUE, round_trip_cost=NO_COST)
    assert result["fills"] >= 1
    assert result["gross"] == pytest.approx(-TICK * POINT_VALUE * result["fills"])


def test_crossing_never_misses_a_signal():
    quotes = flat_quotes(12)
    result = simulate_crossing(np.full(12, 0.2), quotes, threshold=0.01, hold_bars=2,
                               point_value=POINT_VALUE, round_trip_cost=NO_COST)
    assert result["fill_rate"] == 1.0
