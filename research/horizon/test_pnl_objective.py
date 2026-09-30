"""Tests for the P&L custom objective.

The gradient is checked against a numerical derivative, because a hand-derived gradient
that is subtly wrong still trains -- it just optimises something nobody intended.
"""
from __future__ import annotations

import numpy as np
import pytest

from pnl_objective import (MIN_HESSIAN, POSITION_PENALTY, decide, gradient_and_hessian,
                           make_evaluation, make_objective, penalised_loss, positions, profit)

COST = 5.0


class FakeDataset:
    def __init__(self, label):
        self._label = label

    def get_label(self):
        return self._label


def test_a_position_is_bounded_by_one():
    assert np.abs(positions(np.array([-50.0, 0.0, 50.0]))).max() <= 1.0


def test_a_zero_score_takes_no_position():
    assert positions(np.array([0.0]))[0] == pytest.approx(0.0)


def test_profit_pays_the_cost_in_proportion_to_size():
    assert profit(np.array([1.0]), np.array([30.0]), COST)[0] == pytest.approx(25.0)
    assert profit(np.array([0.5]), np.array([30.0]), COST)[0] == pytest.approx(0.5 * 30 - 2.5)


def test_a_flat_position_earns_and_costs_nothing():
    assert profit(np.array([0.0]), np.array([30.0]), COST)[0] == pytest.approx(0.0)


def test_a_move_smaller_than_the_cost_is_unprofitable_at_any_size():
    sizes = np.linspace(0.01, 1.0, 20)
    earned = profit(sizes, np.full(20, COST - 1.0), COST)
    assert (earned < 0).all()


def test_the_gradient_matches_a_numerical_derivative():
    """A wrong-but-plausible gradient still trains; it just optimises the wrong thing."""
    rng = np.random.default_rng(0)
    score = rng.normal(size=200)
    move = rng.normal(scale=40.0, size=200)
    analytic, _ = gradient_and_hessian(score, move, COST)
    step = 1e-6
    higher = penalised_loss(positions(score + step), move, COST, POSITION_PENALTY)
    lower = penalised_loss(positions(score - step), move, COST, POSITION_PENALTY)
    numerical = (higher - lower) / (2 * step)
    assert np.allclose(analytic, numerical, atol=1e-3)


def test_the_hessian_is_always_positive():
    """LightGBM needs it: the true curvature changes sign, so its magnitude is used."""
    rng = np.random.default_rng(1)
    _, hessian = gradient_and_hessian(rng.normal(size=500), rng.normal(scale=40, size=500), COST)
    assert (hessian >= MIN_HESSIAN).all()


def test_a_large_favourable_move_pushes_the_score_up():
    gradient, _ = gradient_and_hessian(np.array([0.1]), np.array([40.0]), COST)
    assert gradient[0] < 0                       # descent raises the score


def test_a_large_adverse_move_pushes_the_score_down():
    gradient, _ = gradient_and_hessian(np.array([0.1]), np.array([-40.0]), COST)
    assert gradient[0] > 0


def test_a_move_inside_the_cost_band_pushes_the_position_toward_flat():
    """With a long position and a move too small to pay, the gradient must shrink it."""
    gradient, _ = gradient_and_hessian(np.array([0.5]), np.array([COST - 2.0]), COST)
    assert gradient[0] > 0                       # descent lowers the score toward zero


def test_unknown_moves_contribute_no_gradient():
    objective = make_objective(COST)
    gradient, hessian = objective(np.array([0.3, 0.3]), FakeDataset(np.array([np.nan, 40.0])))
    assert gradient[0] == 0.0 and hessian[0] == MIN_HESSIAN
    assert gradient[1] != 0.0


def test_the_evaluation_reports_mean_profit_and_prefers_more():
    name, value, higher_is_better = make_evaluation(COST)(
        np.array([5.0]), FakeDataset(np.array([30.0])))
    assert name == "profit_per_row" and higher_is_better
    assert value == pytest.approx(profit(positions(np.array([5.0])), np.array([30.0]), COST)[0])


def test_the_evaluation_ignores_unknown_moves():
    _, value, _ = make_evaluation(COST)(np.array([5.0, 5.0]),
                                        FakeDataset(np.array([30.0, np.nan])))
    assert np.isfinite(value)


def test_a_real_position_becomes_a_trade():
    assert decide(np.array([4.0, -4.0])).tolist() == [1, -1]


def test_numerical_dust_stays_flat():
    assert decide(np.array([1e-4, -1e-4])).tolist() == [0, 0]


def test_the_penalty_gives_an_interior_optimum():
    """Without it the loss is linear in the position and every row saturates at +-1."""
    move = np.full(1, 20.0)
    sizes = np.linspace(-0.999, 0.999, 4001)
    losses = penalised_loss(sizes, np.full(4001, move[0]), COST, POSITION_PENALTY)
    best = sizes[np.argmin(losses)]
    assert abs(best) < 0.95
    assert best == pytest.approx((20.0 - COST) / (2 * POSITION_PENALTY), abs=0.02)


def test_an_edge_inside_the_cost_band_optimises_to_flat():
    sizes = np.linspace(-0.999, 0.999, 4001)
    losses = penalised_loss(sizes, np.full(4001, COST - 1.0), COST, POSITION_PENALTY)
    assert abs(sizes[np.argmin(losses)]) < 0.02


def test_a_bigger_edge_takes_a_bigger_position():
    sizes = np.linspace(-0.999, 0.999, 4001)
    small = sizes[np.argmin(penalised_loss(sizes, np.full(4001, 15.0), COST, POSITION_PENALTY))]
    large = sizes[np.argmin(penalised_loss(sizes, np.full(4001, 45.0), COST, POSITION_PENALTY))]
    assert large > small > 0
