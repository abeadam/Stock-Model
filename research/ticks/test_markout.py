"""Tests for the markout arithmetic.

Everything in the adverse-selection study rests on these three quantities, and a sign
error in any of them would produce a plausible-looking curve pointing the wrong way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markout import (BUY, POINT_VALUE, SELL, TICK, adverse_dollars, attempt, capture_dollars,
                     drift_dollars, fresh_seconds, markout_curve, markout_dollars, mid_at,
                     queue_ahead, touch_price)
from passive_fill import Book

START = 1_790_343_000


def book_from(quote_rows, trade_rows=()) -> Book:
    quotes = pd.DataFrame(quote_rows, columns=["Date", "Bid", "Ask", "BidSize", "AskSize"])
    trades = pd.DataFrame(list(trade_rows) or [[START, 100.0, 1]], columns=["Date", "Price", "Size"])
    return Book(trades, quotes)


def steady_book(bid=99.875, ask=100.125, seconds=200, size=50) -> Book:
    rows = [[START + s, bid, ask, size, size] for s in range(seconds)]
    return book_from(rows)


def all_fresh(seconds=200) -> set[int]:
    return {START + s for s in range(seconds)}


def test_a_buy_rests_on_the_bid():
    assert touch_price(steady_book(), START, BUY) == 99.875


def test_a_sell_rests_on_the_ask():
    assert touch_price(steady_book(), START, SELL) == 100.125


def test_capture_is_half_the_spread_for_a_buy():
    """Bought at 99.875 with a mid of 100.0: half a tick, which is $6.25."""
    assert capture_dollars(BUY, 99.875, 100.0) == pytest.approx(TICK / 2 * POINT_VALUE)


def test_capture_is_half_the_spread_for_a_sell():
    assert capture_dollars(SELL, 100.125, 100.0) == pytest.approx(TICK / 2 * POINT_VALUE)


def test_a_buy_gains_when_the_mid_rises():
    assert markout_dollars(BUY, 99.875, 100.5) > markout_dollars(BUY, 99.875, 100.0)


def test_a_sell_gains_when_the_mid_falls():
    assert markout_dollars(SELL, 100.125, 99.5) > markout_dollars(SELL, 100.125, 100.0)


def test_adverse_selection_is_positive_when_the_market_moves_against_a_buy():
    """Bought, then the mid fell: the market took something back."""
    assert adverse_dollars(BUY, 100.0, 99.75) > 0


def test_adverse_selection_is_positive_when_the_market_moves_against_a_sell():
    assert adverse_dollars(SELL, 100.0, 100.25) > 0


def test_markout_is_capture_less_adverse_selection():
    """The identity the whole study rests on, checked rather than assumed."""
    for side in (BUY, SELL):
        for mid_later in (99.5, 100.0, 100.5):
            price, mid_then = 99.875 if side == BUY else 100.125, 100.0
            assert markout_dollars(side, price, mid_later) == pytest.approx(
                capture_dollars(side, price, mid_then) - adverse_dollars(side, mid_then, mid_later))


def test_a_fill_on_a_stale_second_is_dropped():
    book = steady_book()
    assert markout_curve(book, set(), float(START), 99.875, BUY) is None


def test_a_horizon_landing_on_a_stale_quote_is_left_out():
    """Carrying a mid forward would understate how far the market moved."""
    book = steady_book()
    fresh = {START, START + 1}                       # 5s and beyond are stale
    curve = markout_curve(book, fresh, float(START), 99.875, BUY)
    assert "markout_1" in curve and "markout_5" not in curve


def test_the_curve_reports_every_fresh_horizon():
    curve = markout_curve(steady_book(), all_fresh(), float(START), 99.875, BUY)
    assert all(f"markout_{h}" in curve for h in (1, 5, 10, 30, 60))


def test_a_flat_book_gives_capture_and_no_adverse_selection():
    curve = markout_curve(steady_book(), all_fresh(), float(START), 99.875, BUY)
    assert curve["capture"] == pytest.approx(6.25)
    assert curve["adverse_30"] == pytest.approx(0.0)
    assert curve["markout_30"] == pytest.approx(6.25)


def test_a_market_walking_away_shows_adverse_selection():
    rows = [[START + s, 99.875 - 0.25 * (s >= 10), 100.125 - 0.25 * (s >= 10), 50, 50]
            for s in range(200)]
    curve = markout_curve(book_from(rows), all_fresh(), float(START), 99.875, BUY)
    assert curve["adverse_30"] == pytest.approx(0.25 * POINT_VALUE)
    assert curve["markout_30"] == pytest.approx(6.25 - 0.25 * POINT_VALUE)


def test_the_queue_ahead_is_the_resting_size_on_your_own_side():
    assert queue_ahead(steady_book(size=37), START, BUY) == 37


def test_an_order_fills_when_the_queue_ahead_is_consumed():
    trades = [[START + 1, 99.875, 60]]               # 60 clears a queue of 50
    book = book_from([[START + s, 99.875, 100.125, 50, 50] for s in range(50)], trades)
    assert attempt(book, float(START), BUY, entry_limit=30) is not None


def test_an_order_does_not_fill_when_nothing_trades_at_its_price():
    trades = [[START + 1, 100.125, 500]]             # all trading on the far side
    book = book_from([[START + s, 99.875, 100.125, 50, 50] for s in range(50)], trades)
    assert attempt(book, float(START), BUY, entry_limit=30) is None


def test_drift_is_signed_like_a_fill_would_be():
    """The counterfactual must carry the same sign convention as the markout."""
    rows = [[START + s, 99.875 - 0.25 * (s >= 10), 100.125 - 0.25 * (s >= 10), 50, 50]
            for s in range(200)]
    drift = drift_dollars(book_from(rows), all_fresh(), START, BUY)
    assert drift["adverse_30"] == pytest.approx(0.25 * POINT_VALUE)


def test_fresh_seconds_reads_the_grid_flag():
    grid = pd.DataFrame({"quote_is_fresh": [True, False, True]}, index=[START, START + 1, START + 2])
    assert fresh_seconds(grid) == {START, START + 2}


def test_mid_before_the_first_quote_is_unknown():
    assert np.isnan(mid_at(steady_book(), START - 100))
