"""Labels and weights that already know what a trade costs.

Everything tried so far optimises something that is not profit -- AUC, or squared error on
a range -- and then chooses a threshold on validation to turn the score into trades. That
second step is where the results died: the best of 216 rules scored -$205 on validation
and -$36,362 on test. The threshold is fitted on a few hundred trades in one month and
asked to hold in the next, and it does not.

The alternative is to put the cost where it cannot be fitted away:

  three_class_label   UP only where the move clears the cost, DOWN only where it clears it
                      the other way, NEITHER in between. A bar that cannot pay for itself
                      is no longer a training example arguing for a direction.
  profit_weight       what is at stake net of cost, so a bar that cannot pay has weight
                      zero and one that can counts in proportion to the money on it.
  expected_dollars    E[move] from the predicted class probabilities and the class means,
                      which turns a model into a decision with **no fitted threshold**:
                      trade when the expected move beats the cost, and not otherwise.

The cost stops being a post-hoc filter and becomes part of the question.
"""
from __future__ import annotations

import numpy as np

DOWN, NEITHER, UP = 0, 1, 2
CLASS_COUNT = 3
PERCENT = 100.0


def move_in_dollars(target_percent: np.ndarray, mid: np.ndarray, point_value: float) -> np.ndarray:
    """The percent-of-price target as dollars on one contract."""
    return target_percent / PERCENT * mid * point_value


def three_class_label(move_dollars: np.ndarray, cost: float) -> np.ndarray:
    """UP / DOWN only where the move covers `cost`; NEITHER in between, NaN where unknown."""
    label = np.where(move_dollars > cost, float(UP),
                     np.where(move_dollars < -cost, float(DOWN), float(NEITHER)))
    return np.where(np.isfinite(move_dollars), label, np.nan)


def profit_weight(move_dollars: np.ndarray, cost: float) -> np.ndarray:
    """Dollars at stake net of cost; zero where the move cannot pay for the trade."""
    return np.where(np.isfinite(move_dollars), np.maximum(np.abs(move_dollars) - cost, 0.0), 0.0)


def class_means(move_dollars: np.ndarray, label: np.ndarray) -> dict[int, float]:
    """Average dollars within each class, measured on training rows only."""
    means = {}
    for value in (DOWN, NEITHER, UP):
        rows = np.isfinite(move_dollars) & (label == value)
        means[value] = float(move_dollars[rows].mean()) if rows.any() else 0.0
    return means


def expected_dollars(probabilities: np.ndarray, means: dict[int, float]) -> np.ndarray:
    """E[move in dollars] implied by the predicted class probabilities."""
    return sum(probabilities[:, value] * means[value] for value in (DOWN, NEITHER, UP))


def decide(expected: np.ndarray, cost: float) -> np.ndarray:
    """Long where the expected move beats the cost, short where it beats it the other way.

    The cost is the threshold. Nothing here is fitted on validation."""
    return np.where(expected > cost, 1, np.where(expected < -cost, -1, 0)).astype(int)
