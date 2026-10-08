"""Book features behind the touch, for deciding whether to rest an order.

The toxicity model reaches AUC 0.6767 on top-of-book data and filtering gets toxicity
to 28.2% against a ~23% break-even. The recorded reason it stalls there is that **queue
position is unknowable at 1-second L1 resolution**: the model infers "will this level get
swept" from the touch size and recent flow, when the decisive variable is how much sits
ahead and how much support stands behind.

Depth answers that directly. Five levels a side give the shape of the book, not just its
edge: whether the touch is a thin skin over size, or all there is.

Only entry is worth this. The eight-session markout puts the loss at -$9.88 before the
fill and -$0.48 after, so exit timing addresses 5% of the problem and resting-or-not
addresses 95%.

Every feature is signed by side, so "own" always means the side the order rests on, and
positive always means the book leans the way the order wants. Nothing here reads past the
placement second -- `test_depth_features` asserts that directly.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEPTH_DIR = "/Users/abeadam/dev/model/Stock-Model/daily_data/depth"
LEVELS = 5
EPSILON = 1e-9
# Seconds over which the whole book's size is compared with its own recent past.
DRAIN_WINDOWS = (1, 5, 30)


def depth_path(day: str) -> str:
    return os.path.join(DEPTH_DIR, f"{day}_ES_DEPTH.txt")


def has_depth(day: str) -> bool:
    return os.path.exists(depth_path(day))


def load_depth(day: str) -> pd.DataFrame:
    """One row a second of the five-deep book, indexed by second.

    Duplicate seconds keep the last row: the collector wrote a few before it was fixed
    to one row per second, and a duplicated index silently fans out a later join.
    """
    frame = pd.read_csv(depth_path(day))
    frame = frame.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    return frame


def side_sizes(frame: pd.DataFrame, side_name: str) -> np.ndarray:
    """The size at each level as a (rows, LEVELS) array, with absent levels zero."""
    columns = [f"{side_name}Size{level}" for level in range(LEVELS)]
    return frame[columns].to_numpy(dtype=float, na_value=0.0)


def book_frame(depth: pd.DataFrame) -> pd.DataFrame:
    """Per-second book shape, unsigned. `at_placement_depth` signs it for a side."""
    bid, ask = side_sizes(depth, "Bid"), side_sizes(depth, "Ask")
    shaped = pd.DataFrame(index=depth.index)
    for name, sizes in (("bid", bid), ("ask", ask)):
        total = sizes.sum(axis=1)
        shaped[f"{name}_touch"] = sizes[:, 0]
        shaped[f"{name}_total"] = total
        shaped[f"{name}_behind"] = total - sizes[:, 0]
        # How much of the side's size sits at the touch. Near 1 is a lone order with
        # nothing behind it; near 0 is a thin edge resting on a deep book.
        shaped[f"{name}_touch_share"] = sizes[:, 0] / np.maximum(total, EPSILON)
    for window in DRAIN_WINDOWS:
        for name in ("bid", "ask"):
            earlier = shaped[f"{name}_total"].shift(window)
            shaped[f"{name}_drain_{window}s"] = (
                (earlier - shaped[f"{name}_total"]) / np.maximum(earlier, EPSILON))
    return shaped


def at_placement_depth(shaped: pd.DataFrame, second: int, side: int) -> dict[str, float] | None:
    """Depth features for an order resting on `side` at `second`, signed by side.

    `side` is +1 for a buy resting on the bid, -1 for a sell resting on the ask.
    """
    if second not in shaped.index:
        return None
    row = shaped.loc[second]
    own, other = ("bid", "ask") if side > 0 else ("ask", "bid")
    own_total, other_total = row[f"{own}_total"], row[f"{other}_total"]
    both = own_total + other_total
    values = {
        "depth_own_total": float(own_total),
        "depth_other_total": float(other_total),
        "depth_own_behind": float(row[f"{own}_behind"]),
        # Near 1 means nothing stands behind this order's level.
        "depth_own_touch_share": float(row[f"{own}_touch_share"]),
        "depth_other_touch_share": float(row[f"{other}_touch_share"]),
        # Positive means the whole book, not just the touch, leans the order's way.
        "depth_imbalance": float((own_total - other_total) / max(both, EPSILON)),
    }
    for window in DRAIN_WINDOWS:
        values[f"depth_own_drain_{window}s"] = float(row[f"{own}_drain_{window}s"])
        values[f"depth_other_drain_{window}s"] = float(row[f"{other}_drain_{window}s"])
    return values


DEPTH_FEATURE_NAMES = (["depth_own_total", "depth_other_total", "depth_own_behind",
                        "depth_own_touch_share", "depth_other_touch_share", "depth_imbalance"]
                       + [f"depth_own_drain_{window}s" for window in DRAIN_WINDOWS]
                       + [f"depth_other_drain_{window}s" for window in DRAIN_WINDOWS])
