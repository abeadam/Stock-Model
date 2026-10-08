"""Which original columns a projection actually leans on.

A rotation hands the model blended columns, so "which feature matters" has to be asked of
the blend. Two statistics, and they answer different questions:

  vip        Variable Importance in Projection. For each feature, its weight in each
             component, squared and averaged across components in proportion to how much
             of the target that component explains. Scaled so the mean squared VIP is 1,
             which is where the customary "above 1 is worth keeping" comes from. This is
             a *supervised* measure: a feature scores well by helping explain the target.

  loading_weight
             the same sum of squared weights taken from a principal-component rotation,
             which never saw the target. A feature scores well here by being involved in
             how the data varies, whether or not that variation predicts anything.

Reading them together separates "this column carries signal" from "this column carries
variance", which is the distinction that decides whether a feature is worth adding.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def explained_by_component(scores: np.ndarray, y_loadings: np.ndarray) -> np.ndarray:
    """How much of the target each component accounts for, component by component."""
    squared_scores = (scores ** 2).sum(axis=0)
    return (y_loadings.ravel() ** 2) * squared_scores


def variable_importance(weights: np.ndarray, scores: np.ndarray, y_loadings: np.ndarray) -> np.ndarray:
    """VIP per feature: weights (features x components), scores (rows x components)."""
    explained = explained_by_component(scores, y_loadings)
    total = explained.sum()
    if total <= EPSILON:
        return np.zeros(weights.shape[0])
    normalised = weights / np.maximum(np.linalg.norm(weights, axis=0), EPSILON)
    share = (normalised ** 2) @ explained / total
    return np.sqrt(weights.shape[0] * share)


def loading_weight(axes: np.ndarray) -> np.ndarray:
    """Sum of squared rotation weights per feature, for an unsupervised rotation."""
    return (axes ** 2).sum(axis=1)


def ranked(names, values, top: int) -> list[tuple[str, float]]:
    order = np.argsort(-np.asarray(values))[:top]
    return [(names[index], float(values[index])) for index in order]
