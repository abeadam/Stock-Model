"""Forward move per unit of volume, as a target, in regular trading hours.

A quarter-point move on two hundred contracts and the same move on two thousand are not
the same event: the first says the book was thin, the second that real size had to be
absorbed. Dividing the forward move by the volume that produced it is the ratio between
them -- price impact per contract, Kyle's lambda in miniature -- and it is a steadier
quantity than the move itself, which swells and shrinks with the hour of the day.

One consequence decides how this can be read. Volume is positive, so dividing by it
**cannot change the sign of the target**. A classifier on "did the mid rise" sees exactly
the same labels either way. The scaling can only matter to a model that fits the
magnitude -- a regression -- where it evens out a target whose spread otherwise tracks
activity. So the comparison worth running is regression on the scaled target against
regression on the raw one, both ranked afterwards by what they imply about direction.

Two divisors are offered, and they are not interchangeable:
  forward   the volume of the bars being predicted. The natural reading of "move per
            volume", but unknown at the decision, so a prediction of it cannot be turned
            back into an expected move without also forecasting volume.
  trailing  the volume already traded, known when the bet is placed, so a prediction is
            directly usable. The weaker normalisation of the two, and the honest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_SECONDS = 5
PERCENT = 100.0
EASTERN = "America/New_York"
REGULAR_HOURS = (9.5, 16.0)          # 09:30 to 16:00 ET, the stock session
MIN_VOLUME = 1.0                     # a bar that traded nothing cannot scale anything


def regular_hours(times: pd.Series) -> np.ndarray:
    """True for bars inside the regular stock session."""
    eastern = pd.to_datetime(times).dt.tz_localize("UTC").dt.tz_convert(EASTERN)
    hours = eastern.dt.hour + eastern.dt.minute / 60
    start, end = REGULAR_HOURS
    return ((hours >= start) & (hours < end)).to_numpy()


def window_is_contiguous(seconds: np.ndarray, bars: int) -> np.ndarray:
    contiguous = np.zeros(len(seconds), bool)
    contiguous[:-bars] = (seconds[bars:] - seconds[:-bars]) == bars * BAR_SECONDS
    return contiguous


def forward_percent(price: np.ndarray, seconds: np.ndarray, bars: int) -> np.ndarray:
    """Percent change of the price over the next `bars` bars; NaN across any gap."""
    move = np.full(len(price), np.nan)
    move[:-bars] = (price[bars:] / price[:-bars] - 1) * PERCENT
    move[~window_is_contiguous(seconds, bars)] = np.nan
    return move


def forward_volume(volume: np.ndarray, bars: int) -> np.ndarray:
    """Volume traded over the bars being predicted, t+1 .. t+bars."""
    rolled = pd.Series(volume).rolling(bars).sum().shift(-bars).to_numpy()
    return np.maximum(rolled, MIN_VOLUME)


def trailing_volume(volume: np.ndarray, bars: int) -> np.ndarray:
    """Volume already traded over t-bars+1 .. t, known when the bet is placed."""
    rolled = pd.Series(volume).rolling(bars, min_periods=1).sum().to_numpy()
    return np.maximum(rolled, MIN_VOLUME)


def scaled_targets(price: np.ndarray, volume: np.ndarray, seconds: np.ndarray, bars: int) -> dict[str, np.ndarray]:
    """The raw forward move and the two volume-scaled versions of it."""
    move = forward_percent(price, seconds, bars)
    return {"percent": move,
            "percent_per_forward_volume": move / forward_volume(volume, bars),
            "percent_per_trailing_volume": move / trailing_volume(volume, bars)}
