"""The VIX complex as features, at a ladder of lookbacks.

Three volatility series are downloaded every day at five seconds and have never reached a
model: the VIX index itself, and the two traded VIX-futures funds. The funds are the ones
worth building on -- they are ordinary stocks on SMART, so a live feed costs nothing extra,
while the index and the VXM future sit behind a CFE subscription this account does not
have and arrive delayed by a quarter of an hour.

Two measurements per series, at every lookback in the ladder:

  vol_<series>_ret_<bars>        its percent change over the last `bars` bars
  vol_<series>_minus_mid_<bars>  that change less the ES mid's change over the same span,
                                 so a move in the volatility complex counts only when the
                                 index has not already made it

The ladder is a parameter rather than a constant. The question these features exist to
answer is whether reaching further back helps, and that is answered by running the same
code at several depths rather than by picking one and hoping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_features import lookback_return_pct
from mid_microstructure import trailing_change

PERCENT = 100.0
# The two VIX-futures funds trade as ordinary stocks, so they can be streamed live; the
# index is carried alongside them to show what the untradeable version would have added.
TRADED_VOLATILITY_SERIES = ("VXX", "UVXY")
VOLATILITY_SERIES = TRADED_VOLATILITY_SERIES + ("VIX",)
# One bar, then every half minute: the shortest reading the data supports, followed by a
# regular spacing that a deeper ladder simply extends.
LADDER_STEP_BARS = 6
SHALLOW_LADDER_DEPTH = 3


def lookback_ladder(depth: int, step_bars: int = LADDER_STEP_BARS) -> tuple[int, ...]:
    """A one-bar reading followed by `depth` readings spaced `step_bars` apart."""
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    return (1,) + tuple(step_bars * rung for rung in range(1, depth + 1))


def volatility_feature_names(ladder: tuple[int, ...],
                             series_names: tuple[str, ...] = TRADED_VOLATILITY_SERIES) -> list[str]:
    return [f"vol_{series.lower()}_{measure}_{bars}"
            for series in series_names for bars in ladder
            for measure in ("ret", "minus_mid")]


def add_volatility_features(data: pd.DataFrame, closes_by_series: dict[str, pd.Series],
                            ladder: tuple[int, ...]) -> pd.DataFrame:
    """Adds `volatility_feature_names(ladder, closes_by_series)` to ES rows holding `time` and `mid`.

    `closes_by_series` maps a series name to its close indexed by bar-start epoch."""
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    # Divided by the current mid rather than the mid `bars` ago, which is what the
    # ms_<symbol>_minus_mid columns do; the two families have to stay comparable, and at
    # these window lengths the two denominators differ in the third decimal place.
    mid_percent = {bars: trailing_change(mid, epochs, bars) / mid * PERCENT for bars in ladder}
    columns = {}
    for series, closes in closes_by_series.items():
        for bars in ladder:
            series_return = lookback_return_pct(closes, epochs, bars)
            columns[f"vol_{series.lower()}_ret_{bars}"] = series_return
            columns[f"vol_{series.lower()}_minus_mid_{bars}"] = series_return - mid_percent[bars]
    return data.assign(**columns)
