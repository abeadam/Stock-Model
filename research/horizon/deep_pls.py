"""Nonlinear extensions of PLS, following Polson, Sokolov and Xu (arXiv 2106.14085).

Their construct is to predict the output from a *deep learner of the input scores*
rather than from a linear map of them. PLS-Trees in that paper is what we have already
run -- PLS scores into LightGBM -- so the untested pieces are:

  pls_relu       scores through a ReLU layer, then a linear read-out. Splitting each
                 score into its positive and negative parts lets the model take a
                 different slope on each side of zero: a piecewise-linear link, which is
                 the cheapest honest nonlinearity.
  deep_pls       that layer stacked: PLS -> ReLU -> PLS again -> ReLU -> read-out. Their
                 Theorem 1 says the weights of such a ReLU stack can be estimated
                 recursively by ordinary least squares, no gradient descent required,
                 which is what makes this affordable on two million rows.
  pls_mlp        the same scores through a small trained network, for comparison with
                 the OLS-estimated stack.
  brillinger     their Section 1.2. For Y | X ~ N(g(a + bX), s^2) with g unknown, the
                 OLS estimate of b is consistent up to scale by Stein's lemma, so the
                 direction can be had linearly and g recovered afterwards by smoothing
                 the outcome against the index.

One caveat governs how to read the Brillinger result here, and it is worth stating
plainly: AUC is invariant to any *monotone* transformation of the score. Fitting g and
scoring by AUC can therefore only change the answer if the fitted g is non-monotone. So
this is not a weak test of nonlinearity -- it is precisely a test of whether the link
bends back on itself, which is the only kind of nonlinearity that could change a ranking.
"""
from __future__ import annotations

import numpy as np

RELU_FLOOR = 0.0
INDEX_BINS = 50
MIN_ROWS_PER_BIN = 200


def relu_expand(scores: np.ndarray) -> np.ndarray:
    """Each score split into its positive and negative parts: max(t,0) and max(-t,0).

    A linear read-out over both parts is a piecewise-linear function of the score with a
    kink at zero, which a linear read-out over the score itself cannot express."""
    return np.hstack([np.maximum(scores, RELU_FLOOR), np.maximum(-scores, RELU_FLOOR)])


def least_squares_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brillinger's estimate: plain OLS, consistent for the true direction up to scale."""
    centred = X - X.mean(axis=0)
    coefficients, *_ = np.linalg.lstsq(centred, y - y.mean(), rcond=None)
    return coefficients


def fit_link(index: np.ndarray, outcome: np.ndarray, bins: int = INDEX_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Non-parametric g: the mean outcome within equal-count bins of the index.

    Returns (bin centres, mean outcome), to be interpolated at new index values."""
    edges = np.quantile(index, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    which = np.clip(np.searchsorted(edges, index, side="right") - 1, 0, len(edges) - 2)
    centres, means = [], []
    for bin_index in range(len(edges) - 1):
        rows = which == bin_index
        if rows.sum() >= MIN_ROWS_PER_BIN:
            centres.append(index[rows].mean())
            means.append(outcome[rows].mean())
    return np.array(centres), np.array(means)


def apply_link(index: np.ndarray, centres: np.ndarray, means: np.ndarray) -> np.ndarray:
    return np.interp(index, centres, means)


def link_monotonicity(means: np.ndarray) -> float:
    """Rank correlation between a bin's position and its mean outcome.

    +1 is a link that only rises, -1 one that only falls, and near 0 a link that bends
    back on itself -- the only shape that can change an AUC, since ranking is blind to
    any monotone transformation. Counting rising steps instead would call a link with a
    long flat stretch non-monotone, which is the wrong reading."""
    order = np.argsort(np.argsort(means)).astype(float)
    position = np.arange(len(means), dtype=float)
    return float(np.corrcoef(position, order)[0, 1])
