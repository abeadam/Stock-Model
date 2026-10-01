"""Order flow imbalance from the depth update stream, after Cont, Kukanov and Stoikov.

The first depth probe was null, but it tested the wrong thing. It built book *shape* --
totals, size behind the touch, what share sits at the touch -- when the construct the
literature settles on is an *event flow*. Same data, different representation.

Cont, Kukanov and Stoikov (arXiv:1011.6402) define the contribution of the n-th book
observation, comparing (P^B, q^B, P^A, q^A) with its predecessor:

    e_n = 1{P^B_n >= P^B_{n-1}} q^B_n  -  1{P^B_n <= P^B_{n-1}} q^B_{n-1}
        - 1{P^A_n <= P^A_{n-1}} q^A_n  +  1{P^A_n >= P^A_{n-1}} q^A_{n-1}

    OFI_k = sum of e_n over the interval

The point they stress is that this counts **trades, limit orders and cancellations**
alike, whereas "trade imbalance" counts only trades. Our `signed_flow_*` features are
trade imbalance, which is precisely the weaker measure they contrast against: they report
R^2 ~ 65% for mid-price change against OFI at 10-second intervals, 35-60% once
price-changing events are excluded to avoid the tautology.

Two further details from the paper that the earlier probe got wrong:

  * price impact per unit of OFI goes as beta ~ 1/AD^lambda, inversely with average
    depth, so OFI must be scaled by the depth prevailing over the window. Raw size
    changes conflate a ten-lot move in a thin book with the same move in a deep one.
  * their fit *increases* with the interval, and they use 10 seconds as the base. The
    earlier probe's windows topped out at one second.

Levels beyond the touch were marginal in the original paper, but a later re-evaluation
reports much larger gains for large-tick instruments, which ES is: a 0.25 tick on ~7750,
with the touch holding roughly a tenth of the visible side. So best-level and multi-level
are kept as separate arms rather than assumed equivalent.
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
# CKS use 10s and report the fit rising with the interval; 1s is kept as the short end
# because it is the grid the rest of the tick work uses.
OFI_WINDOWS_SECONDS = (1, 5, 10, 30)


def updates_path(day: str) -> str:
    return os.path.join(DEPTH_DIR, f"{day}_ES_DEPTH_UPDATES.txt")


def replay_levels(day: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Price and size at each level after every update, plus the update times.

    Absent levels carry NaN price and zero size, so an event touching a level that does
    not exist on one side of the comparison contributes nothing rather than a spurious
    jump from a zero price.
    """
    stream = pd.read_csv(updates_path(day)).to_numpy()
    book = DepthBook(LEVELS)
    times = np.empty(len(stream))
    bid_price = np.full((len(stream), LEVELS), np.nan)
    bid_size = np.zeros((len(stream), LEVELS))
    ask_price = np.full((len(stream), LEVELS), np.nan)
    ask_size = np.zeros((len(stream), LEVELS))
    for row, (moment, position, operation, side, price, size) in enumerate(stream):
        book.apply(int(position), int(operation), int(side), float(price), float(size))
        times[row] = moment
        for level, (level_price, level_size) in enumerate(book.sides[BID][:LEVELS]):
            bid_price[row, level], bid_size[row, level] = level_price, level_size
        for level, (level_price, level_size) in enumerate(book.sides[ASK][:LEVELS]):
            ask_price[row, level], ask_size[row, level] = level_price, level_size
    return times, bid_price, bid_size, ask_price, ask_size


def event_contributions(bid_price, bid_size, ask_price, ask_size) -> np.ndarray:
    """e_n per level, exactly as CKS define it. Shape (events - 1, LEVELS)."""
    now_bp, prev_bp = bid_price[1:], bid_price[:-1]
    now_bq, prev_bq = bid_size[1:], bid_size[:-1]
    now_ap, prev_ap = ask_price[1:], ask_price[:-1]
    now_aq, prev_aq = ask_size[1:], ask_size[:-1]
    contribution = ((now_bp >= prev_bp) * now_bq - (now_bp <= prev_bp) * prev_bq
                    - (now_ap <= prev_ap) * now_aq + (now_ap >= prev_ap) * prev_aq)
    # A level missing from either observation cannot be compared; NaN comparisons are
    # already False, which zeroes the price terms but not the size terms.
    comparable = np.isfinite(now_bp) & np.isfinite(prev_bp) & np.isfinite(now_ap) & np.isfinite(prev_ap)
    return np.where(comparable, contribution, 0.0)


def cumulative(times: np.ndarray, contribution: np.ndarray, bid_size, ask_size):
    """Running sums, so any window is one subtraction. Depth runs alongside for scaling."""
    event_times = times[1:]
    running_ofi = np.vstack([np.zeros(LEVELS), np.cumsum(contribution, axis=0)])
    depth = (bid_size + ask_size).sum(axis=1) / (2 * LEVELS)
    running_depth = np.concatenate([[0.0], np.cumsum(depth[1:])])
    running_count = np.arange(len(running_ofi), dtype=float)
    return np.concatenate([[times[0]], event_times]), running_ofi, running_depth, running_count


def window_ofi(day: str, seconds: np.ndarray,
               windows: tuple[int, ...] = OFI_WINDOWS_SECONDS) -> pd.DataFrame:
    """Depth-normalised OFI per level, over each window ending AT each second."""
    times, bid_price, bid_size, ask_price, ask_size = replay_levels(day)
    stamps, running_ofi, running_depth, running_count = cumulative(
        times, event_contributions(bid_price, bid_size, ask_price, ask_size), bid_size, ask_size)

    def rows_at(moments: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(stamps, moments, side="right") - 1, 0, len(stamps) - 1)

    end = rows_at(seconds.astype(float))
    frame = pd.DataFrame(index=seconds)
    for window in windows:
        start = rows_at(seconds - float(window))
        events = np.maximum(running_count[end] - running_count[start], 1.0)
        # Average depth over the window, which is what beta scales inversely with.
        average_depth = (running_depth[end] - running_depth[start]) / events
        scale = np.maximum(average_depth, EPSILON)
        flows = (running_ofi[end] - running_ofi[start]) / scale[:, None]
        for level in range(LEVELS):
            frame[f"ofi_L{level + 1}_{window}s"] = flows[:, level]
        frame[f"ofi_integrated_{window}s"] = flows.sum(axis=1)
        frame[f"ofi_depth_{window}s"] = average_depth
    return frame


def at_placement_ofi(frame: pd.DataFrame, second: int, side: int,
                     windows: tuple[int, ...] = OFI_WINDOWS_SECONDS) -> dict[str, float] | None:
    """Signed by side: positive means flow is pushing the way the resting order wants.

    OFI is signed buy-positive, so a resting buy multiplies by +1 and a resting sell by
    -1. A buy that is about to be run through sits in front of selling pressure.
    """
    if second not in frame.index:
        return None
    row = frame.loc[second]
    values = {}
    for window in windows:
        for level in range(1, LEVELS + 1):
            values[f"ofi_own_L{level}_{window}s"] = float(side * row[f"ofi_L{level}_{window}s"])
        values[f"ofi_own_integrated_{window}s"] = float(side * row[f"ofi_integrated_{window}s"])
        values[f"ofi_depth_{window}s"] = float(row[f"ofi_depth_{window}s"])
    return values


BEST_LEVEL_OFI_NAMES = [f"ofi_own_L1_{window}s" for window in OFI_WINDOWS_SECONDS]
MULTI_LEVEL_OFI_NAMES = ([f"ofi_own_L{level}_{window}s"
                          for window in OFI_WINDOWS_SECONDS for level in range(1, LEVELS + 1)]
                         + [f"ofi_own_integrated_{window}s" for window in OFI_WINDOWS_SECONDS])
OFI_FEATURE_NAMES = MULTI_LEVEL_OFI_NAMES + [f"ofi_depth_{window}s" for window in OFI_WINDOWS_SECONDS]
