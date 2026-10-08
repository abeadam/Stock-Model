"""Walk-forward by session, with the leak that shipped last time designed out.

`tick_walk_forward.py` built its training set as "every day except the test and the
validation day". That is correct only when the test day is the last one. Once the test day
moves through the middle of the series it silently trains on **later** sessions, which was
worth +0.0051 AUC and went unnoticed because that module was the one written without
tests.

Here a fold is defined by what it may not see: training sessions are strictly earlier than
the validation session, which is strictly earlier than the test session. The tests below
assert that directly rather than trusting the construction.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_TRAIN_SESSIONS = 2


@dataclass(frozen=True)
class SessionFold:
    test: str
    validation: str
    train: tuple[str, ...]


def session_folds(days, min_train_sessions: int = MIN_TRAIN_SESSIONS) -> list[SessionFold]:
    """One fold per session that has enough earlier sessions behind it.

    The session immediately before the test day is held out for early stopping, so the
    model never sees it either."""
    ordered = sorted(days)
    folds = []
    for index in range(len(ordered)):
        train = tuple(ordered[:max(index - 1, 0)])
        if index < 1 or len(train) < min_train_sessions:
            continue
        folds.append(SessionFold(test=ordered[index], validation=ordered[index - 1], train=train))
    return folds
