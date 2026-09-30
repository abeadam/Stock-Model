import numpy as np

from barrier_targets import first_barrier_labels


def labels(mid, barrier=0.5, max_bars=3, seconds=None):
    mid = np.asarray(mid, float)
    seconds = np.arange(len(mid)) * 5 if seconds is None else np.asarray(seconds)
    return first_barrier_labels(mid, seconds, np.full(len(mid), barrier), max_bars)


def test_first_barrier_wins():
    result = labels([100, 100.25, 100.5, 99.0])
    assert result[0] == 1.0                     # +0.5 reached at bar 2 before the fall
    assert result[2] == -1.0                    # 100.5 -> 99.0


def test_unresolved_within_horizon_is_nan():
    assert np.isnan(labels([100, 100.25, 100.0, 100.25, 101.0], max_bars=3)[0])


def test_no_label_across_a_gap():
    result = labels([100, 100.25, 105.0], seconds=[0, 5, 4000])
    assert np.isnan(result[0]) and np.isnan(result[1])
