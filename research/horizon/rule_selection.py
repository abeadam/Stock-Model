"""Choosing a hold rule on a validation period and running it on another.

Entry thresholds are quantiles of the validation |score|, so one grid fits every
score's scale; exits are fractions of the entry threshold; minimum holds scale
with the target horizon.
"""
from __future__ import annotations

import itertools

import numpy as np

from hold_policy import HoldRule, last_bar_before_gap, simulate_hold
from take_profit_policy import simulate_take_profit

POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
ENTRY_QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999)
EXIT_FRACTIONS = (0.5, 0.0, -0.5)


def min_hold_choices(target_bars: int) -> tuple[int, ...]:
    """Minimum holds of one bar, half the target horizon, and the full horizon."""
    return tuple(sorted({1, max(1, target_bars // 2), target_bars}))


def candidate_rules(validation_score: np.ndarray, target_bars: int, fill_delay_bars: int = 0) -> list[HoldRule]:
    thresholds = np.quantile(np.abs(validation_score), ENTRY_QUANTILES)
    return [HoldRule(entry, entry * exit_fraction, hold, fill_delay_bars)
            for entry, exit_fraction, hold in itertools.product(thresholds, EXIT_FRACTIONS,
                                                                min_hold_choices(target_bars))]


def run_rule(score, buy_price, sell_price, seconds, rule: HoldRule) -> dict:
    return simulate_hold(score, buy_price, sell_price, last_bar_before_gap(seconds), rule,
                         POINT_VALUE, ROUND_TRIP_COST)


def choose_rule(score, buy_price, sell_price, seconds, target_bars: int, min_trades: int,
                fill_delay_bars: int = 0) -> HoldRule | None:
    """The candidate with the best net P&L among those with at least min_trades trades."""
    best_rule, best_net = None, -np.inf
    for rule in candidate_rules(score, target_bars, fill_delay_bars):
        result = run_rule(score, buy_price, sell_price, seconds, rule)
        if result["trades"] >= min_trades and result["net"] > best_net:
            best_rule, best_net = rule, result["net"]
    return best_rule


def run_take_profit(score, buy_price, sell_price, target_points, seconds, threshold: float,
                    stop_points=None) -> dict:
    return simulate_take_profit(score, buy_price, sell_price, target_points, last_bar_before_gap(seconds),
                                threshold, POINT_VALUE, ROUND_TRIP_COST, stop_points)


def choose_take_profit_threshold(score, buy_price, sell_price, target_points, seconds,
                                 min_trades: int, stop_points=None) -> float | None:
    """The entry threshold (a quantile of |score|) with the best net P&L among those trading enough."""
    best_threshold, best_net = None, -np.inf
    for threshold in np.quantile(np.abs(score), ENTRY_QUANTILES):
        result = run_take_profit(score, buy_price, sell_price, target_points, seconds, threshold, stop_points)
        if result["trades"] >= min_trades and result["net"] > best_net:
            best_threshold, best_net = threshold, result["net"]
    return best_threshold
