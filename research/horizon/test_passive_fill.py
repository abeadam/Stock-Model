import numpy as np
import pandas as pd

from passive_fill import Book, passive_round_trip, simulate_limit_order


def book(trades, quotes):
    return Book(pd.DataFrame(trades, columns=["Date", "Price", "Size"]),
                pd.DataFrame(quotes, columns=["Date", "Bid", "Ask", "BidSize", "AskSize"]))


QUOTES = [(0, 100.0, 100.25, 10, 10)]


def test_queue_must_clear_before_filling():
    trades = [(1, 100.0, 4), (2, 100.0, 5)]                     # 9 of the 10 ahead of us
    assert simulate_limit_order(book(trades, QUOTES), 0, 100.0, 1, queue_ahead=10, time_limit=60) is None
    trades = trades + [(3, 100.0, 2)]                            # now the queue is gone
    assert simulate_limit_order(book(trades, QUOTES), 0, 100.0, 1, queue_ahead=10, time_limit=60) == 3


def test_trading_through_the_level_fills_us():
    trades = [(1, 99.75, 1)]                                     # printed below our bid
    assert simulate_limit_order(book(trades, QUOTES), 0, 100.0, 1, queue_ahead=999, time_limit=60) == 1


def test_trades_elsewhere_do_not_fill_us():
    trades = [(1, 100.25, 50), (2, 100.5, 50)]                   # all on the offer side
    assert simulate_limit_order(book(trades, QUOTES), 0, 100.0, 1, queue_ahead=10, time_limit=60) is None


def test_time_limit_cancels_the_order():
    trades = [(30, 100.0, 50)]
    assert simulate_limit_order(book(trades, QUOTES), 0, 100.0, 1, queue_ahead=10, time_limit=10) is None


def test_selling_mirrors_buying():
    trades = [(1, 100.25, 12)]                                   # lifts the offer, clearing 10 ahead
    assert simulate_limit_order(book(trades, QUOTES), 0, 100.25, -1, queue_ahead=10, time_limit=60) == 1


def test_round_trip_that_reaches_its_target():
    trades = [(1, 100.0, 20),                                    # our buy fills at 100.00
              (2, 100.25, 20)]                                   # someone lifts our offer at 100.25
    result = passive_round_trip(book(trades, QUOTES), decision_at=0, side=1, target_ticks=1,
                                entry_time_limit=30, hold_time_limit=30)
    assert result["filled"] and result["exit"] == "target"
    assert np.isclose(result["points"], 0.25)


def test_round_trip_that_times_out_leaves_at_market():
    trades = [(1, 100.0, 20)]                                    # filled long at 100.00, then nothing
    quotes = QUOTES + [(20, 99.5, 99.75, 10, 10)]                # market drops before the deadline
    result = passive_round_trip(book(trades, quotes), decision_at=0, side=1, target_ticks=1,
                                entry_time_limit=30, hold_time_limit=30)
    assert result["exit"] == "market"
    assert np.isclose(result["points"], -0.5)                    # sold the bid at 99.50


def test_unfilled_entry_reports_no_trade():
    result = passive_round_trip(book([(1, 100.25, 5)], QUOTES), decision_at=0, side=1, target_ticks=1,
                                entry_time_limit=30, hold_time_limit=30)
    assert result == {"filled": False}


def test_stop_leaves_at_market_when_price_runs_against_us():
    trades = [(1, 100.0, 20),                                    # filled long at 100.00
              (2, 99.75, 5)]                                     # trades a tick below: stop triggers
    quotes = QUOTES + [(2, 99.5, 99.75, 10, 10)]
    result = passive_round_trip(book(trades, quotes), decision_at=0, side=1, target_ticks=1,
                                entry_time_limit=30, hold_time_limit=60, stop_ticks=1)
    assert result["exit"] == "stop"
    assert np.isclose(result["points"], -0.5)                    # sold the bid at 99.50


def test_target_still_wins_when_it_comes_first():
    trades = [(1, 100.0, 20), (2, 100.25, 20), (3, 99.5, 5)]
    result = passive_round_trip(book(trades, QUOTES), decision_at=0, side=1, target_ticks=1,
                                entry_time_limit=30, hold_time_limit=60, stop_ticks=1)
    assert result["exit"] == "target"
