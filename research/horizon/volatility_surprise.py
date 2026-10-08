"""Implied volatility moving by more than the tape justifies.

Every volatility column in the model is built from the price path: `rolling_vol_5/20`,
`vol_ratio` (realized-fast over realized-slow), `ms_mid_change_z_n` (a move over a 60-bar
realized vol). The VIX family enters only as levels, returns and lags, and `vxm_es_ratio`
is `vxm_close / close` -- VIX points over index points, not a volatility comparison.

`vol_<series>_minus_mid_<bars>` is the nearest existing thing and is a different object:
implied change minus *price* change, signed. This is implied change minus *realized
volatility* change, unsigned.

    realized_recent = (max High - min Low over t-W+1..t)    / Close[t]      x 100
    realized_prior  = (max High - min Low over t-2W+1..t-W) / Close[t-W]    x 100
    delta_realized  = log(realized_recent / realized_prior)
    delta_implied   = log(series[t] / series[t-W])
    surprise        = delta_implied - delta_realized

Positive means implied rose by more than the tape justified.

Three decisions worth stating. The realized leg uses the **high-low range**, not a
close-to-close standard deviation: over twelve observations a std is unusable, the range
is several times more efficient, and it matches the volatility target, which is already a
forward range. Only **VXX and UVXY** are offered, because both are ordinary SMART stocks
and stream in real time, while VXM and the VIX index arrive ~15 minutes late on this
account -- anything found here is therefore tradeable as found. And the windows stay
**shallow** (6 and 12 bars): reaching back damaged the volatility head badly at the 18-bar
horizon, t -4.9 with 72.7% of tree gain consumed.

Nothing reaches across a session gap: contiguity is checked over the whole 2W span.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_SECONDS = 5
PERCENT = 100.0
SURPRISE_WINDOWS = (6, 12)
SURPRISE_SERIES = ("VXX", "UVXY")
EPSILON = 1e-12

VOLATILITY_SURPRISE_LEGS = ([f"vs_delta_realized_{window}" for window in SURPRISE_WINDOWS]
                            + [f"vs_{series.lower()}_delta_implied_{window}"
                               for series in SURPRISE_SERIES for window in SURPRISE_WINDOWS])
VOLATILITY_SURPRISE_FEATURES = [f"vs_{series.lower()}_surprise_{window}"
                                for series in SURPRISE_SERIES for window in SURPRISE_WINDOWS]


def span_is_contiguous(seconds: np.ndarray, bars: int) -> np.ndarray:
    """True where the whole `bars`-bar reach behind each row is unbroken 5-second bars.

    Compares the two endpoints, which is sufficient because the frame's epochs ascend on a
    5-second grid: any missing bar inside the reach pushes the endpoints further apart than
    the span allows. This is the same argument `trailing_change` relies on."""
    contiguous = np.zeros(len(seconds), bool)
    contiguous[bars:] = (seconds[bars:] - seconds[:-bars]) == bars * BAR_SECONDS
    return contiguous


def block_range_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    window: int, offset: int) -> np.ndarray:
    """High-to-low range of the `window`-bar block ending `offset` bars back, as % of its close."""
    highest = pd.Series(high).rolling(window, min_periods=window).max().shift(offset).to_numpy()
    lowest = pd.Series(low).rolling(window, min_periods=window).min().shift(offset).to_numpy()
    reference = pd.Series(close).shift(offset).to_numpy()
    span = (highest - lowest) / reference * PERCENT
    return np.where(span > EPSILON, span, np.nan)


def delta_realized(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Log change in realized range between two adjacent non-overlapping blocks."""
    recent = block_range_pct(high, low, close, window, offset=0)
    prior = block_range_pct(high, low, close, window, offset=window)
    return np.log(recent / prior)


def delta_implied(closes: pd.Series, epochs: np.ndarray, window: int) -> np.ndarray:
    """Log change in an implied-volatility proxy over `window` bars."""
    now = closes.reindex(epochs).to_numpy()
    before = closes.reindex(epochs - window * BAR_SECONDS).to_numpy()
    ratio = np.where((now > EPSILON) & (before > EPSILON), now / before, np.nan)
    return np.log(ratio.astype(float))


def add_volatility_surprise(data: pd.DataFrame, closes_by_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Adds the legs and the surprise columns to ES rows holding `time`, `High`, `Low`, `Close`."""
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    high, low, close = (data[name].to_numpy(float) for name in ("High", "Low", "Close"))
    columns = {}
    for window in SURPRISE_WINDOWS:
        unbroken = span_is_contiguous(epochs, 2 * window)
        realized = np.where(unbroken, delta_realized(high, low, close, window), np.nan)
        columns[f"vs_delta_realized_{window}"] = realized
        for series, closes in closes_by_series.items():
            implied = np.where(unbroken, delta_implied(closes, epochs, window), np.nan)
            columns[f"vs_{series.lower()}_delta_implied_{window}"] = implied
            columns[f"vs_{series.lower()}_surprise_{window}"] = implied - realized
    return data.assign(**columns)
