"""Kernel PCA at the size this data actually is.

A kernel method needs the kernel between every pair of rows, which at 573,480 rows is
3e11 entries. So the map is fitted on a sample of the fold's training rows and every row
is then projected onto it. That is the standard landmark approach and the only one that
runs here.

Two mistakes this project has already made are designed out rather than left to care:

  The sample is drawn at **random**, never by stride. A stride-40 sample once manufactured
  a -0.094 correlation where the true value across every row was -0.0003, and flipped sign
  to +0.043 at stride 41.

  The map is fitted **per fold, on that fold's training rows alone**. One rotation shared
  across folds leaked badly: 53% of fold 1's test rows sit inside fold 2's training block,
  so the shared fit had already seen what it was later scored on.

Gamma follows the median heuristic -- the reciprocal of the median squared distance
between sampled points -- so the kernel width adapts to the data's own scale instead of
assuming one. A linear map is offered alongside on identical sampling, because the only
interesting question about a kernel is what it buys over not having one.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA

from feature_rank import prepare, principal_axes, standardise

LANDMARK_COUNT = 3000
CHUNK_ROWS = 20_000
GAMMA_SAMPLE = 1000
EPSILON = 1e-12


def sample_rows(rows: np.ndarray, count: int, seed: int) -> np.ndarray:
    """A random subset of `rows`, reproducible from `seed`. Never a stride."""
    generator = np.random.default_rng(seed)
    if len(rows) <= count:
        return rows
    return np.sort(generator.choice(rows, size=count, replace=False))


def median_heuristic_gamma(standardised: np.ndarray, seed: int = 0) -> float:
    """1 / median pairwise squared distance, the usual scale-free choice of kernel width."""
    generator = np.random.default_rng(seed)
    points = standardised[generator.choice(len(standardised),
                                           size=min(GAMMA_SAMPLE, len(standardised)),
                                           replace=False)]
    squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    median = np.median(squared[np.triu_indices(len(points), k=1)])
    return float(1.0 / max(median, EPSILON))


def in_chunks(project, X: np.ndarray, fitted: dict, rows: np.ndarray, components: int) -> np.ndarray:
    """Project `rows` a block at a time: the whole kernel against the landmarks will not fit."""
    result = np.empty((len(rows), components), dtype=np.float32)
    for start in range(0, len(rows), CHUNK_ROWS):
        block = rows[start:start + CHUNK_ROWS]
        result[start:start + len(block)] = project(standardise(X[block], fitted)).astype(np.float32)
    return result


def kernel_map(X: np.ndarray, train_rows: np.ndarray, components: int, gamma: float | None = None,
               landmarks: int = LANDMARK_COUNT, seed: int = 0):
    """Fit kernel PCA on a random sample of this fold's training rows; return a projector."""
    sample = sample_rows(train_rows, landmarks, seed)
    fitted = prepare(X[sample])
    standardised = standardise(X[sample], fitted)
    width = median_heuristic_gamma(standardised, seed) if gamma is None else gamma
    model = KernelPCA(n_components=components, kernel="rbf", gamma=width, copy_X=False)
    model.fit(standardised)
    return lambda rows: in_chunks(model.transform, X, fitted, rows, components), width


def linear_map(X: np.ndarray, train_rows: np.ndarray, components: int,
               landmarks: int = LANDMARK_COUNT, seed: int = 0):
    """The same sampling and standardisation, rotated linearly: the control for the kernel."""
    sample = sample_rows(train_rows, landmarks, seed)
    fitted = prepare(X[sample])
    axes = principal_axes(standardise(X[sample], fitted), components)
    return lambda rows: in_chunks(lambda block: block @ axes, X, fitted, rows, components), 0.0
