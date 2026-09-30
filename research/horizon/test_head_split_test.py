"""The slice is the risk: the direction head reads matrix[:, :len(columns.direction)].

That is only correct while the volatility head's columns are the direction head's
columns in the same order, followed by the added ones. Nothing in the type system says
so, and a silent reordering would train the direction head on the wrong columns while
still producing a plausible AUC.
"""
import numpy as np
import pandas as pd
import pytest

from head_columns import DROPPED_FEATURES, VOLATILITY_ONLY_FEATURES, split_by_head
from head_split_test import arms_for, gain_share

BASE = ["ema_9", "rolling_vol_5", "ms_uso_minus_mid_3", "ms_uup_minus_mid_1", "bb_dist"]


class FakeBooster:
    def __init__(self, gains):
        self._gains = np.array(gains, dtype=float)

    def feature_importance(self, importance_type):
        assert importance_type == "gain"
        return self._gains


class FakeModel:
    def __init__(self, gains):
        self.booster_ = FakeBooster(gains)


def test_the_volatility_columns_start_with_the_direction_columns():
    columns = split_by_head(BASE)
    assert columns.volatility[:len(columns.direction)] == columns.direction


def test_slicing_the_matrix_recovers_the_direction_columns():
    columns = split_by_head(BASE)
    frame = pd.DataFrame({name: [float(index)] for index, name in enumerate(columns.volatility)})
    matrix = frame[columns.volatility].to_numpy(np.float32)
    sliced = matrix[:, :len(columns.direction)]
    expected = frame[columns.direction].to_numpy(np.float32)
    assert np.array_equal(sliced, expected)


def test_the_slice_is_a_view_not_a_copy():
    """A million-row second copy is the thing this design avoids."""
    matrix = np.zeros((4, 6), dtype=np.float32)
    assert matrix[:, :4].base is matrix


def test_the_added_columns_are_the_tail_of_the_matrix():
    columns = split_by_head(BASE)
    assert columns.volatility[len(columns.direction):] == list(VOLATILITY_ONLY_FEATURES)


def test_every_arm_is_offered():
    assert list(arms_for(BASE)) == ["shared", "dropped only", "split"]


def test_the_shared_arm_gives_both_heads_the_same_columns():
    shared = arms_for(BASE)["shared"]
    assert shared.direction == shared.volatility == BASE


def test_the_shared_arm_keeps_the_oil_and_dollar_columns():
    """It is the status quo, so it must not quietly adopt the change under test."""
    assert set(DROPPED_FEATURES) & set(arms_for(BASE)["shared"].direction)


def test_the_dropped_arm_changes_only_the_drop():
    dropped = arms_for(BASE)["dropped only"]
    assert dropped.direction == dropped.volatility
    assert not set(DROPPED_FEATURES) & set(dropped.direction)


def test_the_dropped_arm_adds_nothing():
    assert arms_for(BASE)["dropped only"].volatility_added == 0


def test_the_split_arm_differs_from_dropped_only_by_the_new_columns():
    arms = arms_for(BASE)
    assert (set(arms["split"].volatility) - set(arms["dropped only"].volatility)
            == set(VOLATILITY_ONLY_FEATURES))


def test_the_split_arm_leaves_direction_identical_to_dropped_only():
    """So any direction difference between them would be noise, not a feature change."""
    arms = arms_for(BASE)
    assert arms["split"].direction == arms["dropped only"].direction


def test_gain_share_reads_the_trailing_columns():
    assert gain_share(FakeModel([1.0, 1.0, 6.0, 2.0]), added=2) == pytest.approx(0.8)


def test_gain_share_is_zero_when_nothing_was_added():
    assert gain_share(FakeModel([1.0, 2.0]), added=0) == 0.0


def test_gain_share_survives_a_model_that_used_nothing():
    assert gain_share(FakeModel([0.0, 0.0]), added=1) == 0.0
