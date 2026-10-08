"""A LightGBM objective that maximises expected P&L, with the cost inside the loss.

Putting the cost in the label or the sample weight tells the model which rows matter. This
tells it what it is being paid for. The raw score is read as a position through a tanh, so
the model chooses a size in (-1, 1) rather than a probability:

    position   a = tanh(s)
    profit     a * m - cost * |a|        where m is the move in dollars
    loss       L = -(a * m - cost * |a|) + penalty * a^2

**The quadratic penalty is what makes this work, and it was not obvious.** Without it the
loss is linear in the position, so the gradient pushes every row to the boundary: a first
version saturated 100% of positions at +-1 and never went flat, which defeats the purpose.
With it the optimum is interior and is exactly the rule we want:

    a* = (E[m] - cost * sign(a)) / (2 * penalty)

which is zero whenever |E[m]| < cost, because the sign term then opposes any position the
edge would justify. The flat region is part of the loss rather than a threshold chosen on
validation, and the fitted size is proportional to the edge instead of bang-bang.

    dL/ds = (-(m - cost * sign(a)) + 2 * penalty * a) * (1 - a^2)

**The loss is not convex**, and that curvature is genuine rather than an artefact: the
profit is linear in the position while tanh is S-shaped, so the second derivative changes
sign around a = 0. LightGBM needs a positive hessian, so its magnitude is used with a
floor. That is the standard treatment and it makes the step sizes approximate -- the
gradient direction stays correct, which is what decides the splits.
"""
from __future__ import annotations

import numpy as np

MIN_HESSIAN = 1e-3
# A position of 1 corresponds to an edge of 2 * penalty dollars, so this sets how large
# an expected move counts as "all in". Median |move| at 18 bars is about $38.
POSITION_PENALTY = 25.0
# Any position at all means the fitted edge cleared the cost; this only excludes numerical dust.
FLAT_POSITION = 0.01


def positions(raw_score: np.ndarray) -> np.ndarray:
    """The score read as a position in (-1, 1)."""
    return np.tanh(raw_score)


def profit(position: np.ndarray, move_dollars: np.ndarray, cost: float) -> np.ndarray:
    """Dollars earned by holding `position` through `move_dollars`, after the toll."""
    return position * move_dollars - cost * np.abs(position)


def penalised_loss(position: np.ndarray, move_dollars: np.ndarray, cost: float,
                   penalty: float) -> np.ndarray:
    """What the objective actually minimises: minus profit, plus the sizing penalty."""
    return -profit(position, move_dollars, cost) + penalty * position ** 2


def gradient_and_hessian(raw_score: np.ndarray, move_dollars: np.ndarray, cost: float,
                         penalty: float = POSITION_PENALTY) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the penalised loss with respect to the raw score."""
    position = positions(raw_score)
    slack = 1.0 - position ** 2
    edge = move_dollars - cost * np.sign(position)
    inner = -edge + 2.0 * penalty * position
    gradient = inner * slack
    hessian = 2.0 * penalty * slack ** 2 - 2.0 * position * slack * inner
    return gradient, np.maximum(np.abs(hessian), MIN_HESSIAN)


def make_objective(cost: float, penalty: float = POSITION_PENALTY):
    """A LightGBM custom objective; the dataset's label must be the move in dollars."""
    def objective(raw_score: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        move_dollars = dataset.get_label()
        finite = np.isfinite(move_dollars)
        gradient, hessian = gradient_and_hessian(raw_score, np.where(finite, move_dollars, 0.0),
                                                 cost, penalty)
        return np.where(finite, gradient, 0.0), np.where(finite, hessian, MIN_HESSIAN)
    return objective


def make_evaluation(cost: float):
    """Mean profit per row, so early stopping watches money rather than a proxy."""
    def evaluation(raw_score: np.ndarray, dataset) -> tuple[str, float, bool]:
        move_dollars = dataset.get_label()
        finite = np.isfinite(move_dollars)
        earned = profit(positions(raw_score[finite]), move_dollars[finite], cost)
        return "profit_per_row", float(earned.mean()), True
    return evaluation


def decide(raw_score: np.ndarray, flat: float = FLAT_POSITION) -> np.ndarray:
    """Trade where the model took a position at all; flat where it did not.

    With the penalty in place a non-zero position already means the fitted edge cleared
    the cost, so this is not a tuned threshold -- it only discards numerical dust."""
    position = positions(raw_score)
    return np.where(position > flat, 1, np.where(position < -flat, -1, 0)).astype(int)
