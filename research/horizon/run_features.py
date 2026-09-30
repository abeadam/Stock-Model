"""How far this move has run, measured against how far moves usually run.

The study found that a move which has carried six bars gives back eight times what an
average move gives back. That is a conditioning variable, so here it becomes columns.

The question "is this unusual?" needs a yardstick, and the yardstick has to be recent:
six bars in a quiet hour is a different event from six bars in a violent one. So each
reading is expressed three ways -- as itself, against the session's own distribution so
far, and in units of recent volatility -- because it is not obvious in advance which of
the three the market actually respects.

The session-so-far comparison uses an expanding count within the day, which looks only
backwards and starts each morning knowing nothing, exactly as a live system would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mid_microstructure import TICK, mid_volatility, trailing_change

BAR_SECONDS = 5
MIN_HISTORY = 200                  # bars into the session before a percentile means anything
RUN_FEATURES = ("run_length_signed", "run_extension_ticks", "run_extension_z",
                "run_length_percentile", "run_extension_percentile", "run_bars_since_reversal")


def run_state(change: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(length, direction, carried) of the same-signed streak ending at each bar."""
    sign = np.sign(np.nan_to_num(change))
    length = np.zeros(len(sign), dtype=int)
    carried = np.zeros(len(sign))
    for index in range(1, len(sign)):
        if sign[index] != 0 and sign[index] == sign[index - 1]:
            length[index] = length[index - 1] + 1
            carried[index] = carried[index - 1] + change[index]
        elif sign[index] != 0:
            length[index] = 1
            carried[index] = change[index]
    return length, sign, carried


def expanding_percentile(values: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Where each value sits among those seen earlier the same session; 0.5 until enough."""
    result = np.full(len(values), 0.5)
    frame = pd.DataFrame({"value": values, "day": day})
    for _, rows in frame.groupby("day").groups.items():
        index = np.asarray(rows)
        series = pd.Series(values[index])
        rank = series.expanding().apply(lambda window: (window[:-1] <= window[-1]).mean()
                                        if len(window) > 1 else 0.5, raw=True)
        settled = np.arange(len(index)) >= MIN_HISTORY
        result[index] = np.where(settled, rank.to_numpy(), 0.5)
    return result


def add_run_features(data: pd.DataFrame) -> pd.DataFrame:
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    day = data["session_day"].to_numpy()
    change = trailing_change(mid, seconds, 1)
    length, sign, carried = run_state(change)
    volatility = mid_volatility(mid, seconds) * TICK
    extension_ticks = carried / TICK
    with np.errstate(invalid="ignore", divide="ignore"):
        extension_z = np.where(volatility > 0, carried / volatility, np.nan)
    reversal = np.zeros(len(sign), dtype=int)
    for index in range(1, len(sign)):
        reversal[index] = 0 if length[index] == 1 else reversal[index - 1] + 1
    return data.assign(run_length_signed=length * sign,
                       run_extension_ticks=extension_ticks,
                       run_extension_z=extension_z,
                       run_length_percentile=expanding_percentile(length.astype(float), day),
                       run_extension_percentile=expanding_percentile(np.abs(extension_ticks), day),
                       run_bars_since_reversal=reversal.astype(float))
