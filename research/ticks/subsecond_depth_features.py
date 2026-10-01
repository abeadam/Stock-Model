"""How fast the queue ahead is draining, measured below the one-second grid.

Snapshot depth was null: five levels sampled once a second correlate with a toxic fill
at 0.0659 against 0.2088 for the best L1 feature, and adding them cost -0.0037 AUC. The
book is simply stable at that resolution -- the own side averages 340 contracts and the
touch is a 10% skin on it -- so its shape says little about who is about to cross.

That is not the question the update stream was collected to answer. The depth feed is
the one place sub-second time exists at all: IBKR types the historical tick `time` field
as an int, so trades and quotes arrive rounded to the second with ~11 sharing each one,
while depth arrives live and carries the local clock. On 2026-09-30, 95.7% of book
updates landed under 100 ms apart and every one of 768,628 timestamps was distinct.

So the hypothesis here is about rate, not level: an order about to be run over should
see the size ahead of it disappearing in the tens of milliseconds beforehand, and that
motion is invisible once averaged into a one-second row.

Replaying the stream through DepthBook reproduces the snapshot grid exactly, which is
what licenses using it as the book's true state between snapshots.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/abeadam/dev/interactive-broker-python/Updated Stats")
from depth_book import ASK, BID, DepthBook

DEPTH_DIR = "/Users/abeadam/dev/model/Stock-Model/daily_data/depth"
LEVELS = 5
EPSILON = 1e-9
# Below the one-second grid by 20x at the short end; 1000 ms is the grid itself, kept as
# a control -- if only that window carries signal, the sub-second claim is not supported.
SUBSECOND_WINDOWS_MS = (50, 100, 250, 1000)


def updates_path(day: str) -> str:
    return os.path.join(DEPTH_DIR, f"{day}_ES_DEPTH_UPDATES.txt")


def has_updates(day: str) -> bool:
    return os.path.exists(updates_path(day))


def replay(day: str) -> pd.DataFrame:
    """Touch and total size on each side after every update, at microsecond times."""
    stream = pd.read_csv(updates_path(day)).to_numpy()
    book = DepthBook(LEVELS)
    times = np.empty(len(stream))
    state = np.empty((len(stream), 4))               # bid touch, bid total, ask touch, ask total
    for row, (moment, position, operation, side, price, size) in enumerate(stream):
        book.apply(int(position), int(operation), int(side), float(price), float(size))
        bid, ask = book.sides[BID], book.sides[ASK]
        times[row] = moment
        state[row] = (bid[0][1] if bid else 0.0, sum(s for _, s in bid),
                      ask[0][1] if ask else 0.0, sum(s for _, s in ask))
    return pd.DataFrame(state, index=times,
                        columns=["bid_touch", "bid_total", "ask_touch", "ask_total"])


def state_at(replayed: pd.DataFrame, moments: np.ndarray) -> np.ndarray:
    """The book as it stood at each moment: the last update at or before it."""
    rows = np.searchsorted(replayed.index.to_numpy(), moments, side="right") - 1
    rows = np.clip(rows, 0, len(replayed) - 1)
    return replayed.to_numpy()[rows]


def subsecond_frame(replayed: pd.DataFrame, seconds: np.ndarray) -> pd.DataFrame:
    """For each second, how much the touch and the book moved over each window before it.

    Every window ends AT the second, so nothing reads past the placement moment.
    """
    now = state_at(replayed, seconds.astype(float))
    frame = pd.DataFrame(index=seconds)
    for window in SUBSECOND_WINDOWS_MS:
        before = state_at(replayed, seconds - window / 1000.0)
        for index, name in enumerate(("bid_touch", "bid_total", "ask_touch", "ask_total")):
            # Positive means size LEFT over the window.
            frame[f"{name}_drain_{window}ms"] = (
                (before[:, index] - now[:, index]) / np.maximum(before[:, index], EPSILON))
    counts = np.searchsorted(replayed.index.to_numpy(), seconds.astype(float), side="right")
    for window in SUBSECOND_WINDOWS_MS:
        earlier = np.searchsorted(replayed.index.to_numpy(), seconds - window / 1000.0, side="right")
        frame[f"updates_{window}ms"] = counts - earlier
    return frame


def at_placement_subsecond(frame: pd.DataFrame, second: int, side: int) -> dict[str, float] | None:
    """Signed by side: 'own' is the side the order rests on."""
    if second not in frame.index:
        return None
    row = frame.loc[second]
    own, other = ("bid", "ask") if side > 0 else ("ask", "bid")
    values = {}
    for window in SUBSECOND_WINDOWS_MS:
        values[f"ss_own_touch_drain_{window}ms"] = float(row[f"{own}_touch_drain_{window}ms"])
        values[f"ss_other_touch_drain_{window}ms"] = float(row[f"{other}_touch_drain_{window}ms"])
        values[f"ss_own_total_drain_{window}ms"] = float(row[f"{own}_total_drain_{window}ms"])
        values[f"ss_updates_{window}ms"] = float(row[f"updates_{window}ms"])
    return values


SUBSECOND_FEATURE_NAMES = [name for window in SUBSECOND_WINDOWS_MS for name in (
    f"ss_own_touch_drain_{window}ms", f"ss_other_touch_drain_{window}ms",
    f"ss_own_total_drain_{window}ms", f"ss_updates_{window}ms")]
