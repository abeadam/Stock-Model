import numpy as np
import pandas as pd

from rolling_normalisation import normalised_names, rolling_z_scores

RANDOM = np.random.default_rng(21)


def frame(values):
    return pd.DataFrame({"a": values})


def test_a_level_shift_is_removed_once_the_window_has_moved_past_it():
    """The point of the exercise: the same behaviour at a new level reads the same."""
    early = RANDOM.normal(size=500)
    late = RANDOM.normal(size=500) + 100.0                       # identical, 100 higher
    scores = rolling_z_scores(frame(np.concatenate([early, late])), window=100)["a"].to_numpy()
    settled_early, settled_late = scores[400:500], scores[900:1000]
    assert abs(settled_early.mean() - settled_late.mean()) < 0.2
    assert abs(settled_early.std() - settled_late.std()) < 0.3


def test_nothing_from_the_future_is_used():
    values = list(RANDOM.normal(size=300))
    before = rolling_z_scores(frame(values), window=50)["a"].to_numpy()
    after = rolling_z_scores(frame(values[:200] + list(RANDOM.normal(size=100) + 50)), window=50)["a"].to_numpy()
    assert np.allclose(before[:200], after[:200])

def test_a_flat_stretch_reads_as_typical_rather_than_infinite():
    scores = rolling_z_scores(frame([5.0] * 200), window=50)["a"].to_numpy()
    assert np.all(np.isfinite(scores)) and np.allclose(scores, 0.0)


def test_a_value_far_above_its_recent_past_scores_high():
    values = np.concatenate([RANDOM.normal(size=200), [20.0]])
    assert rolling_z_scores(frame(values), window=100)["a"].to_numpy()[-1] > 5


def test_the_start_of_the_series_is_filled_not_dropped():
    scores = rolling_z_scores(frame(RANDOM.normal(size=100)), window=80)
    assert len(scores) == 100 and np.all(np.isfinite(scores.to_numpy()))


def test_names_carry_the_window():
    assert normalised_names(["rsi", "mfi"], 720) == ["rsi_z720", "mfi_z720"]
