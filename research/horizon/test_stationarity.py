import numpy as np

from stationarity import CRITICAL_VALUES, augmented_dickey_fuller, has_unit_root

RANDOM = np.random.default_rng(13)


def random_walk(n=4000):
    return np.cumsum(RANDOM.normal(size=n))


def mean_reverting(n=4000, pull=0.05):
    series = np.zeros(n)
    for step in range(1, n):
        series[step] = series[step - 1] * (1 - pull) + RANDOM.normal()
    return series


def test_a_random_walk_is_not_called_stationary():
    statistic = augmented_dickey_fuller(random_walk())
    assert statistic > CRITICAL_VALUES["5%"]
    assert has_unit_root(statistic)


def test_a_mean_reverting_series_is_called_stationary():
    statistic = augmented_dickey_fuller(mean_reverting())
    assert statistic < CRITICAL_VALUES["1%"]
    assert not has_unit_root(statistic)


def test_white_noise_is_very_clearly_stationary():
    assert augmented_dickey_fuller(RANDOM.normal(size=4000)) < -10


def test_the_statistic_is_scale_free():
    series = mean_reverting()
    assert np.isclose(augmented_dickey_fuller(series), augmented_dickey_fuller(series * 1000), rtol=1e-6)


def test_a_constant_or_tiny_series_returns_no_answer():
    assert np.isnan(augmented_dickey_fuller(np.ones(500)))
    assert np.isnan(augmented_dickey_fuller(np.array([1.0, 2.0, 3.0])))


def test_missing_values_are_dropped_not_propagated():
    series = mean_reverting()
    with_gaps = series.copy()
    with_gaps[::97] = np.nan
    assert np.isfinite(augmented_dickey_fuller(with_gaps))
