"""Tests for session walk-forward folds.

The first two exist because `tick_walk_forward.py` shipped a look-ahead leak worth
+0.0051 AUC: its training set was "every day except test and validation", which trains on
later sessions once the test day is not the last one. These assert the property directly.
"""
from __future__ import annotations

import pytest

from session_folds import MIN_TRAIN_SESSIONS, SessionFold, session_folds

SESSIONS = ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23",
            "2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"]


def test_no_fold_trains_on_a_later_session():
    """The exact leak that shipped: training on sessions after the test day."""
    for fold in session_folds(SESSIONS):
        assert all(day < fold.test for day in fold.train)


def test_no_fold_trains_on_its_own_test_session():
    for fold in session_folds(SESSIONS):
        assert fold.test not in fold.train


def test_the_validation_session_is_never_trained_on():
    for fold in session_folds(SESSIONS):
        assert fold.validation not in fold.train


def test_validation_sits_between_training_and_test():
    for fold in session_folds(SESSIONS):
        assert max(fold.train) < fold.validation < fold.test


def test_validation_is_the_session_immediately_before_the_test():
    for fold in session_folds(SESSIONS):
        assert SESSIONS[SESSIONS.index(fold.test) - 1] == fold.validation


def test_training_grows_as_the_test_day_advances():
    folds = session_folds(SESSIONS)
    assert all(len(earlier.train) < len(later.train)
               for earlier, later in zip(folds, folds[1:]))


def test_sessions_without_enough_history_are_skipped():
    folds = session_folds(SESSIONS, min_train_sessions=2)
    assert folds[0].test == SESSIONS[3]


def test_a_larger_minimum_skips_more_sessions():
    assert len(session_folds(SESSIONS, 4)) < len(session_folds(SESSIONS, 2))


def test_too_few_sessions_gives_no_folds():
    assert session_folds(["2026-09-18", "2026-09-21"]) == []


def test_input_order_does_not_matter():
    assert session_folds(list(reversed(SESSIONS))) == session_folds(SESSIONS)


def test_eight_sessions_give_five_test_days():
    assert len(session_folds(SESSIONS)) == 5


def test_the_default_minimum_is_two_sessions():
    assert MIN_TRAIN_SESSIONS == 2


def test_a_fold_is_immutable():
    fold = session_folds(SESSIONS)[0]
    with pytest.raises(Exception):
        fold.test = "2026-01-01"


def test_every_session_appears_as_a_test_day_once_eligible():
    tested = [fold.test for fold in session_folds(SESSIONS)]
    assert tested == SESSIONS[3:]
