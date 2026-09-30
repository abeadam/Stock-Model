import numpy as np
from sklearn.cross_decomposition import PLSRegression

from projection_importance import loading_weight, ranked, variable_importance

RANDOM = np.random.default_rng(31)


def fitted_pls(X, y, components=3):
    model = PLSRegression(n_components=components, scale=False).fit(X, y)
    return variable_importance(model.x_weights_, model.x_scores_, model.y_loadings_)


def test_vip_finds_the_features_the_target_was_built_from():
    X = RANDOM.normal(size=(4000, 10))
    y = 2 * X[:, 3] - X[:, 7] + RANDOM.normal(scale=0.5, size=4000)
    vip = fitted_pls(X, y)
    assert set(np.argsort(-vip)[:2]) == {3, 7}


def test_a_pure_noise_column_scores_below_one():
    X = RANDOM.normal(size=(4000, 6))
    y = X[:, 0] + RANDOM.normal(scale=0.3, size=4000)
    vip = fitted_pls(X, y)
    assert vip[0] > 1.0 and vip[1:].max() < 1.0


def test_mean_squared_vip_is_one():
    """The scaling that makes 'above 1' the usual keep-or-drop line."""
    X = RANDOM.normal(size=(3000, 8))
    y = X[:, 2] + 0.5 * X[:, 5] + RANDOM.normal(scale=0.4, size=3000)
    assert np.isclose((fitted_pls(X, y, components=4) ** 2).mean(), 1.0, atol=0.02)


def test_vip_is_never_negative():
    X = RANDOM.normal(size=(2000, 5))
    assert np.all(fitted_pls(X, RANDOM.normal(size=2000)) >= 0)


def test_loading_weight_counts_involvement_in_the_rotation():
    axes = np.array([[1.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
    assert np.allclose(loading_weight(axes), [1.0, 0.25, 0.0])


def test_ranked_returns_the_largest_first_with_their_names():
    assert ranked(["a", "b", "c"], [0.1, 0.9, 0.5], top=2) == [("b", 0.9), ("c", 0.5)]
