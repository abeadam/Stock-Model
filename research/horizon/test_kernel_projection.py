"""Tests for the landmark kernel projection.

Two of these guard mistakes this project has actually made: sampling by stride, and
fitting one map across folds.
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_projection import (kernel_map, linear_map, median_heuristic_gamma, sample_rows)

FEATURES = 12
COMPONENTS = 4


def synthetic(rows: int = 600, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(rows, FEATURES)).astype(np.float32)


def test_sampling_is_random_not_a_stride():
    rows = np.arange(1000)
    gaps = np.diff(sample_rows(rows, 50, seed=1))
    assert len(np.unique(gaps)) > 1


def test_sampling_is_reproducible_from_the_seed():
    rows = np.arange(1000)
    assert np.array_equal(sample_rows(rows, 50, seed=7), sample_rows(rows, 50, seed=7))


def test_a_different_seed_draws_a_different_sample():
    rows = np.arange(1000)
    assert not np.array_equal(sample_rows(rows, 50, seed=1), sample_rows(rows, 50, seed=2))


def test_sampling_never_leaves_the_rows_it_was_given():
    rows = np.arange(500, 900)
    assert set(sample_rows(rows, 40, seed=3)).issubset(set(rows))


def test_asking_for_more_rows_than_exist_returns_them_all():
    rows = np.arange(10)
    assert np.array_equal(sample_rows(rows, 50, seed=1), rows)


def test_gamma_is_positive_and_finite():
    gamma = median_heuristic_gamma(synthetic(200))
    assert gamma > 0 and np.isfinite(gamma)


def test_gamma_shrinks_when_the_points_spread_out():
    points = synthetic(200)
    assert median_heuristic_gamma(points * 10) < median_heuristic_gamma(points)


def test_kernel_map_returns_the_requested_components():
    X = synthetic()
    project, _ = kernel_map(X, np.arange(400), COMPONENTS, landmarks=200)
    assert project(np.arange(400, 600)).shape == (200, COMPONENTS)


def test_linear_map_returns_the_requested_components():
    X = synthetic()
    project, _ = linear_map(X, np.arange(400), COMPONENTS, landmarks=200)
    assert project(np.arange(400, 600)).shape == (200, COMPONENTS)


def test_the_map_is_fold_local():
    """Different training rows must produce a different map, or folds share a fit."""
    X = synthetic()
    first, _ = kernel_map(X, np.arange(0, 300), COMPONENTS, landmarks=150)
    second, _ = kernel_map(X, np.arange(300, 600), COMPONENTS, landmarks=150)
    rows = np.arange(100, 200)
    assert not np.allclose(first(rows), second(rows))


def test_chunking_does_not_change_the_projection():
    import kernel_projection
    X = synthetic(1200)
    project, _ = kernel_map(X, np.arange(600), COMPONENTS, landmarks=200)
    rows = np.arange(600, 1200)
    whole = project(rows)
    original = kernel_projection.CHUNK_ROWS
    try:
        kernel_projection.CHUNK_ROWS = 37
        chunked = project(rows)
    finally:
        kernel_projection.CHUNK_ROWS = original
    assert np.allclose(whole, chunked, atol=1e-5)


def test_missing_values_are_filled_rather_than_propagated():
    X = synthetic()
    X[5, 3] = np.nan
    project, _ = kernel_map(X, np.arange(400), COMPONENTS, landmarks=200)
    assert np.isfinite(project(np.arange(0, 20))).all()


def test_an_explicit_gamma_overrides_the_heuristic():
    X = synthetic()
    _, width = kernel_map(X, np.arange(400), COMPONENTS, gamma=0.125, landmarks=200)
    assert width == pytest.approx(0.125)


def test_the_heuristic_reports_the_width_it_chose():
    X = synthetic()
    _, width = kernel_map(X, np.arange(400), COMPONENTS, landmarks=200)
    assert width > 0
