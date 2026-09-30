"""How much independent information is in a feature matrix, and how to keep only that.

The leave-one-out attribution showed eight indicators where removing any one costs
nothing: they are near-duplicates of each other. A column count is therefore a poor
measure of how much a model has been told. These are better ones:

  variance rank    how many principal components it takes to carry 95% and 99% of the
                   variance -- 255 columns spanning ten directions is ten features in
                   an expensive coat
  effective rank   exp(entropy of the normalised singular values), the usual continuous
                   measure, which counts a direction in proportion to what it carries

The same decomposition then rebuilds the matrix as its leading components, which is what
"refactor to higher-rank columns" means in practice: orthogonal columns, each carrying
something the others do not.

Everything is fitted on training rows only -- the mean, the scale and the rotation all
leak the future otherwise.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def prepare(train: np.ndarray) -> dict[str, np.ndarray]:
    """Median and spread of each column, from training rows only."""
    median = np.nanmedian(train, axis=0)
    filled = np.where(np.isfinite(train), train, median)
    scale = filled.std(axis=0)
    return {"median": median, "scale": np.where(scale > EPSILON, scale, 1.0)}


def standardise(X: np.ndarray, fitted: dict[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isfinite(X), X, fitted["median"])
    return (filled - fitted["median"]) / fitted["scale"]


def singular_values(standardised_train: np.ndarray) -> np.ndarray:
    return np.linalg.svd(standardised_train, compute_uv=False)


def variance_rank(values: np.ndarray, share: float) -> int:
    """Components needed to carry `share` of the total variance."""
    variance = values ** 2
    return int(np.searchsorted(np.cumsum(variance) / variance.sum(), share) + 1)


def effective_rank(values: np.ndarray) -> float:
    """exp of the entropy of the singular-value distribution (Roy & Vetterli)."""
    weights = values / values.sum()
    weights = weights[weights > EPSILON]
    return float(np.exp(-(weights * np.log(weights)).sum()))


def principal_axes(standardised_train: np.ndarray, components: int) -> np.ndarray:
    """The leading right-singular vectors: the directions to project onto."""
    _, _, right = np.linalg.svd(standardised_train, full_matrices=False)
    return right[:components].T


def describe(name: str, standardised_train: np.ndarray) -> dict:
    values = singular_values(standardised_train)
    summary = {"features": standardised_train.shape[1], "rank_95": variance_rank(values, 0.95),
               "rank_99": variance_rank(values, 0.99), "effective_rank": round(effective_rank(values), 1)}
    print(f"{name:<30}{summary['features']:>5} columns   95% in {summary['rank_95']:>3}   "
          f"99% in {summary['rank_99']:>3}   effective rank {summary['effective_rank']:>6.1f}", flush=True)
    return summary
