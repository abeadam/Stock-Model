import numpy as np

from feature_rank import (effective_rank, prepare, principal_axes, singular_values, standardise,
                          variance_rank)

RANDOM = np.random.default_rng(11)


def duplicated_matrix(rows=2000, signal_columns=3, copies=4):
    """`signal_columns` real directions, each repeated `copies` times with a little noise."""
    signal = RANDOM.normal(size=(rows, signal_columns))
    columns = [signal[:, index] + RANDOM.normal(scale=0.01, size=rows)
               for index in range(signal_columns) for _ in range(copies)]
    return np.column_stack(columns)


def test_variance_rank_counts_real_directions_not_columns():
    X = duplicated_matrix()                                   # 12 columns, 3 directions
    values = singular_values(standardise(X, prepare(X)))
    assert X.shape[1] == 12
    assert variance_rank(values, 0.95) == 3


def test_effective_rank_is_the_column_count_when_columns_are_independent():
    X = RANDOM.normal(size=(4000, 6))
    assert 5.0 < effective_rank(singular_values(standardise(X, prepare(X)))) <= 6.0


def test_effective_rank_collapses_when_columns_repeat():
    duplicated = effective_rank(singular_values(standardise(duplicated_matrix(), prepare(duplicated_matrix()))))
    assert duplicated < 6.0


def test_standardising_uses_training_statistics_only():
    train = np.array([[0.0], [10.0]])
    fitted = prepare(train)
    later = standardise(np.array([[5.0]]), fitted)
    assert np.isclose(later[0, 0], (5.0 - 5.0) / 5.0)


def test_missing_values_become_the_training_median():
    train = np.array([[1.0], [3.0], [np.nan]])
    fitted = prepare(train)
    assert np.isclose(standardise(np.array([[np.nan]]), fitted)[0, 0], 0.0)


def test_principal_axes_are_orthonormal_and_span_the_signal():
    X = duplicated_matrix()
    axes = principal_axes(standardise(X, prepare(X)), components=3)
    assert axes.shape == (12, 3)
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-8)


def test_projection_keeps_the_information_of_duplicated_columns():
    X = duplicated_matrix()
    fitted = prepare(X)
    standardised = standardise(X, fitted)
    projected = standardised @ principal_axes(standardised, components=3)
    # Three orthogonal columns should predict any original column almost perfectly.
    target = standardised[:, 0]
    coefficients, *_ = np.linalg.lstsq(projected, target, rcond=None)
    residual = target - projected @ coefficients
    assert residual.var() < 0.01 * target.var()
