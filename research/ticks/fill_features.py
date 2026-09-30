"""What is knowable at the moment an order is placed.

The decomposition showed the loss is 84% pre-fill: an order gives up $9.45 while it waits
and only $1.82 after it fills. So the decision worth predicting is **whether to rest at
all**, which means every feature has to be available at placement -- before the fill, not
at it.

That excludes one item from the original brief, "size of the print that filled you". By
definition that print arrives at the fill, so it can inform a hold-or-flatten decision but
not a place-or-not one. Since flattening immediately pays the spread again, and since the
pre-fill term is five times the post-fill term, the placement decision is the one worth
having and the filling print is left out.

Everything here reads the second-grid frame `tick_probe.load_day` builds, at the placement
second and earlier. Features are signed by side where direction matters, so a buy and a
sell with mirror-image books get the same values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.25
DEPLETION_WINDOWS = (1, 5)
FLOW_WINDOWS = (1, 5, 30)
VOLATILITY_WINDOWS = (5, 30)
QUOTE_ACTIVITY_WINDOW = 5
EPSILON = 1e-9

FILL_FEATURES = (["own_size", "other_size", "book_imbalance", "spread_ticks",
                  "quote_updates_5s", "second_of_session"]
                 + [f"own_depletion_{w}s" for w in DEPLETION_WINDOWS]
                 + [f"signed_flow_{w}s" for w in FLOW_WINDOWS]
                 + [f"volume_{w}s" for w in FLOW_WINDOWS]
                 + [f"mid_volatility_{w}s" for w in VOLATILITY_WINDOWS])


def session_frame(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-second quantities that do not depend on which side an order rests."""
    mid = grid["mid"]
    frame = pd.DataFrame(index=grid.index)
    frame["bid_size"] = grid["BidSize"]
    frame["ask_size"] = grid["AskSize"]
    frame["spread_ticks"] = (grid["Ask"] - grid["Bid"]) / TICK
    frame["quote_updates_5s"] = grid["quote_is_fresh"].rolling(QUOTE_ACTIVITY_WINDOW,
                                                               min_periods=1).sum()
    frame["second_of_session"] = np.arange(len(grid))
    for window in FLOW_WINDOWS:
        frame[f"signed_flow_raw_{window}s"] = grid["signed"].rolling(window, min_periods=1).sum()
        frame[f"volume_{window}s"] = grid["volume"].rolling(window, min_periods=1).sum()
    for window in VOLATILITY_WINDOWS:
        frame[f"mid_volatility_{window}s"] = mid.diff().rolling(window, min_periods=2).std() / TICK
    for window in DEPLETION_WINDOWS:
        frame[f"bid_depletion_{window}s"] = grid["BidSize"].diff(window)
        frame[f"ask_depletion_{window}s"] = grid["AskSize"].diff(window)
    return frame


def at_placement(frame: pd.DataFrame, second: int, side: int) -> dict[str, float] | None:
    """The feature row for an order resting on `side` at `second`, signed by side.

    `side` is +1 for a buy resting on the bid, -1 for a sell resting on the ask."""
    if second not in frame.index:
        return None
    row = frame.loc[second]
    own, other = ("bid_size", "ask_size") if side > 0 else ("ask_size", "bid_size")
    total = row["bid_size"] + row["ask_size"]
    values = {
        "own_size": float(row[own]),
        "other_size": float(row[other]),
        # Positive means the book leans the way this order wants it to.
        "book_imbalance": float(side * (row["bid_size"] - row["ask_size"]) / max(total, EPSILON)),
        "spread_ticks": float(row["spread_ticks"]),
        "quote_updates_5s": float(row["quote_updates_5s"]),
        "second_of_session": float(row["second_of_session"]),
    }
    for window in DEPLETION_WINDOWS:
        own_column = f"{'bid' if side > 0 else 'ask'}_depletion_{window}s"
        values[f"own_depletion_{window}s"] = float(row[own_column])
    for window in FLOW_WINDOWS:
        values[f"signed_flow_{window}s"] = float(side * row[f"signed_flow_raw_{window}s"])
        values[f"volume_{window}s"] = float(row[f"volume_{window}s"])
    for window in VOLATILITY_WINDOWS:
        values[f"mid_volatility_{window}s"] = float(row[f"mid_volatility_{window}s"])
    return values
