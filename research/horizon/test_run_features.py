import numpy as np
import pandas as pd

from run_features import RUN_FEATURES, add_run_features, expanding_percentile, run_state

START = 1_700_000_000
TICK = 0.25


def frame(mid, day="2026-06-01"):
    return pd.DataFrame({"time": pd.to_datetime(np.arange(len(mid)) * 5 + START, unit="s"),
                         "mid": mid, "session_day": [day] * len(mid)})


def test_a_streak_counts_up_and_resets_on_a_turn():
    length, sign, _ = run_state(np.array([0.0, 1.0, 1.0, 1.0, -1.0, -1.0]))
    assert length.tolist() == [0, 1, 2, 3, 1, 2]
    assert sign.tolist() == [0.0, 1.0, 1.0, 1.0, -1.0, -1.0]


def test_a_flat_bar_ends_the_streak():
    length, _, _ = run_state(np.array([0.0, 1.0, 1.0, 0.0, 1.0]))
    assert length.tolist() == [0, 1, 2, 0, 1]


def test_carried_accumulates_over_the_streak_only():
    _, _, carried = run_state(np.array([0.0, 0.25, 0.25, -0.25]))
    assert np.allclose(carried, [0.0, 0.25, 0.50, -0.25])


def test_the_sign_of_the_length_says_which_way():
    rising = add_run_features(frame(5000 + np.arange(6) * TICK))
    falling = add_run_features(frame(5000 - np.arange(6) * TICK))
    assert rising["run_length_signed"].to_numpy()[-1] > 0
    assert falling["run_length_signed"].to_numpy()[-1] < 0


def test_extension_is_counted_in_ticks():
    rising = add_run_features(frame(5000 + np.arange(5) * TICK))
    assert np.isclose(rising["run_extension_ticks"].to_numpy()[-1], 4.0)


def test_a_percentile_looks_only_backwards():
    values = np.arange(400, dtype=float)
    day = np.array(["2026-06-01"] * 400)
    ranks = expanding_percentile(values, day)
    assert ranks[-1] > 0.99                     # the largest yet
    assert np.all(ranks[:200] == 0.5)           # withheld until there is history


def test_percentiles_restart_each_session():
    """Day two's values are all smaller than day one's; if yesterday leaked in, the
    second day would rank near zero instead of against its own morning."""
    values = np.concatenate([np.arange(1000, 1300, dtype=float), np.arange(300, dtype=float)])
    day = np.array(["2026-06-01"] * 300 + ["2026-06-02"] * 300)
    ranks = expanding_percentile(values, day)
    assert ranks[599] > 0.99


def test_every_declared_column_is_produced():
    built = add_run_features(frame(5000 + np.cumsum(np.random.default_rng(4).choice([-TICK, TICK], 300))))
    assert all(name in built.columns for name in RUN_FEATURES)
