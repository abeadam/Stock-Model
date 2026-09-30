"""Tests for cost-aware labels, weights and the expected-value decision rule."""
from __future__ import annotations

import numpy as np
import pytest

from cost_aware_targets import (DOWN, NEITHER, UP, class_means, decide, expected_dollars,
                                move_in_dollars, profit_weight, three_class_label)

COST = 5.0
POINT_VALUE = 50.0


def test_a_percent_target_converts_to_dollars():
    """0.01% of a 7700 mid is 0.77 points, which is $38.50 on one contract."""
    dollars = move_in_dollars(np.array([0.01]), np.array([7700.0]), POINT_VALUE)
    assert dollars[0] == pytest.approx(38.5)


def test_a_move_clearing_the_cost_upward_is_up():
    assert three_class_label(np.array([COST + 1]), COST)[0] == UP


def test_a_move_clearing_the_cost_downward_is_down():
    assert three_class_label(np.array([-COST - 1]), COST)[0] == DOWN


def test_a_move_inside_the_cost_is_neither():
    assert three_class_label(np.array([COST - 0.01, 0.0, -COST + 0.01]), COST).tolist() == [NEITHER] * 3


def test_the_boundary_itself_is_neither():
    """Exactly covering the cost leaves nothing, so it is not a trade."""
    assert three_class_label(np.array([COST, -COST]), COST).tolist() == [NEITHER, NEITHER]


def test_an_unknown_move_has_no_label():
    assert np.isnan(three_class_label(np.array([np.nan]), COST)[0])


def test_a_bar_that_cannot_pay_has_no_weight():
    assert profit_weight(np.array([COST - 1]), COST)[0] == 0.0


def test_weight_is_the_money_left_after_the_cost():
    assert profit_weight(np.array([30.0, -30.0]), COST).tolist() == [25.0, 25.0]


def test_an_unknown_move_has_no_weight():
    assert profit_weight(np.array([np.nan]), COST)[0] == 0.0


def test_class_means_are_taken_within_each_class():
    moves = np.array([-20.0, -10.0, 0.0, 10.0, 30.0])
    label = three_class_label(moves, COST)
    means = class_means(moves, label)
    assert means[UP] == pytest.approx(20.0)
    assert means[DOWN] == pytest.approx(-15.0)
    assert means[NEITHER] == pytest.approx(0.0)


def test_expected_dollars_weights_the_class_means():
    probabilities = np.array([[0.2, 0.3, 0.5]])           # down, neither, up
    means = {DOWN: -20.0, NEITHER: 0.0, UP: 30.0}
    assert expected_dollars(probabilities, means)[0] == pytest.approx(0.2 * -20 + 0.5 * 30)


def test_the_decision_rule_uses_the_cost_as_its_threshold():
    expected = np.array([COST + 1, COST - 1, -COST + 1, -COST - 1])
    assert decide(expected, COST).tolist() == [1, 0, 0, -1]


def test_a_confident_model_on_a_tiny_move_still_does_not_trade():
    """Certainty about a move smaller than the cost is not a reason to trade."""
    probabilities = np.array([[0.0, 0.0, 1.0]])
    means = {DOWN: -20.0, NEITHER: 0.0, UP: 2.0}          # certain, but only $2 of move
    assert decide(expected_dollars(probabilities, means), COST)[0] == 0


def test_a_larger_cost_makes_the_rule_trade_less():
    expected = np.array([10.0, -10.0, 3.0])
    assert (decide(expected, 5.0) != 0).sum() > (decide(expected, 20.0) != 0).sum()


def test_raising_the_cost_moves_labels_into_neither():
    moves = np.array([10.0, -10.0, 30.0])
    cheap = three_class_label(moves, 5.0)
    dear = three_class_label(moves, 20.0)
    assert (cheap == NEITHER).sum() < (dear == NEITHER).sum()
