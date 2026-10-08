import numpy as np

from deep_pls import (apply_link, fit_link, least_squares_direction, link_monotonicity, relu_expand)

RANDOM = np.random.default_rng(5)


def test_relu_expand_keeps_both_sides_of_zero():
    scores = np.array([[-2.0, 3.0]])
    assert relu_expand(scores).tolist() == [[0.0, 3.0, 2.0, 0.0]]


def test_relu_expand_loses_nothing():
    """Positive minus negative part reconstructs the score exactly."""
    scores = RANDOM.normal(size=(100, 3))
    expanded = relu_expand(scores)
    assert np.allclose(expanded[:, :3] - expanded[:, 3:], scores)


def test_least_squares_recovers_the_direction_up_to_scale():
    X = RANDOM.normal(size=(5000, 4))
    truth = np.array([1.0, -2.0, 0.5, 0.0])
    y = np.tanh(X @ truth) + RANDOM.normal(scale=0.1, size=5000)   # unknown monotone g
    estimate = least_squares_direction(X, y)
    cosine = estimate @ truth / (np.linalg.norm(estimate) * np.linalg.norm(truth))
    assert cosine > 0.99


def test_link_recovers_a_bent_shape():
    index = RANDOM.normal(size=40000)
    outcome = (np.abs(index) > 1).astype(float)                    # a U shape, not monotone
    centres, means = fit_link(index, outcome, bins=20)
    assert means[0] > means[len(means) // 2] and means[-1] > means[len(means) // 2]
    assert abs(link_monotonicity(means)) < 0.5                     # bends back, so ranking can change


def test_link_of_a_monotone_relationship_is_monotone():
    index = RANDOM.normal(size=40000)
    centres, means = fit_link(index, (index + RANDOM.normal(scale=0.5, size=40000) > 0).astype(float), bins=20)
    assert link_monotonicity(means) > 0.9


def test_applying_the_link_interpolates_between_bin_centres():
    centres, means = np.array([0.0, 1.0]), np.array([0.2, 0.8])
    assert np.isclose(apply_link(np.array([0.5]), centres, means)[0], 0.5)
