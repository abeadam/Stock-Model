"""Tests for trading a decision sequence."""
from __future__ import annotations

import numpy as np
import pytest

from decision_trading import CROSS_FILL, MID_FILL, simulate_decisions
from quote_passive_entry import Quotes

POINT_VALUE = 50.0
COST = 5.0
TICK = 0.25


def book(mid_prices: list[float]) -> tuple[Quotes, np.ndarray]:
    mid = np.array(mid_prices, dtype=float)
    is_last = np.zeros(len(mid), bool)
    is_last[-1] = True
    quotes = Quotes(bid=mid - TICK / 2, ask=mid + TICK / 2,
                    high=mid + TICK, low=mid - TICK, is_last=is_last)
    return quotes, mid


def test_a_long_into_a_rise_earns_the_move_less_cost():
    quotes, mid = book([100.0] * 3 + [101.0] * 3)
    direction = np.zeros(6, int)
    direction[0] = 1
    result = simulate_decisions(direction, quotes, mid, hold_bars=3, point_value=POINT_VALUE, cost=COST)
    assert result["per_trade"] == pytest.approx(1.0 * POINT_VALUE - COST)


def test_a_short_into_a_rise_loses_the_move_plus_cost():
    quotes, mid = book([100.0] * 3 + [101.0] * 3)
    direction = np.zeros(6, int)
    direction[0] = -1
    result = simulate_decisions(direction, quotes, mid, hold_bars=3, point_value=POINT_VALUE, cost=COST)
    assert result["per_trade"] == pytest.approx(-1.0 * POINT_VALUE - COST)


def test_doing_nothing_trades_nothing():
    quotes, mid = book([100.0] * 6)
    result = simulate_decisions(np.zeros(6, int), quotes, mid, 3, POINT_VALUE, COST)
    assert result["trades"] == 0


def test_crossing_the_spread_costs_a_full_tick_more_than_the_mid():
    quotes, mid = book([100.0] * 8)
    direction = np.zeros(8, int)
    direction[0] = 1
    at_mid = simulate_decisions(direction, quotes, mid, 3, POINT_VALUE, COST, MID_FILL)
    crossed = simulate_decisions(direction, quotes, mid, 3, POINT_VALUE, COST, CROSS_FILL)
    assert at_mid["per_trade"] - crossed["per_trade"] == pytest.approx(TICK * POINT_VALUE)


def test_only_one_position_is_open_at_a_time():
    quotes, mid = book([100.0] * 20)
    result = simulate_decisions(np.ones(20, int), quotes, mid, hold_bars=5,
                                point_value=POINT_VALUE, cost=COST)
    assert result["trades"] <= 4


def test_a_position_never_reaches_past_a_session_gap():
    quotes, mid = book([100.0] * 5 + [200.0] * 5)
    quotes.is_last[4] = True
    direction = np.zeros(10, int)
    direction[3] = 1
    result = simulate_decisions(direction, quotes, mid, hold_bars=5, point_value=POINT_VALUE, cost=COST)
    assert result["per_trade"] == pytest.approx(-COST)          # exits at the session end, flat


def test_no_trade_is_opened_on_the_last_bar_of_a_session():
    quotes, mid = book([100.0] * 5)
    direction = np.zeros(5, int)
    direction[4] = 1
    assert simulate_decisions(direction, quotes, mid, 3, POINT_VALUE, COST)["trades"] == 0


def test_cost_is_charged_once_per_trade():
    quotes, mid = book([100.0] * 12)
    direction = np.zeros(12, int)
    direction[[0, 5]] = 1
    result = simulate_decisions(direction, quotes, mid, 3, POINT_VALUE, COST)
    assert result["net"] == pytest.approx(-COST * result["trades"])


def test_the_win_rate_counts_profitable_trades():
    quotes, mid = book([100.0, 100.0, 101.0, 101.0, 99.0, 99.0])
    direction = np.zeros(6, int)
    direction[0] = 1
    result = simulate_decisions(direction, quotes, mid, hold_bars=2, point_value=POINT_VALUE, cost=COST)
    assert result["win_rate"] == 1.0
