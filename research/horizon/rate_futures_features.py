"""Treasury futures as an intraday read on policy expectations.

The Fed Funds future is the purest statement of where policy is going, and it is useless
here: it changes in 0.6% of five-second bars. Treasury futures say the same thing far more
actively -- ZT moves in 12.8% of bars, ZN in 18.1% -- and they trade in the same CME
ecosystem as ES, which is where a lead-lag would plausibly live if one does.

  <symbol>_ret_n        the future's own move over the last n bars
  <symbol>_minus_mid_n  its move less the ES mid's over the same window: the divergence,
                        which is what the cross-asset attribution kept finding
  curve_change_n        the two-year against the ten-year. A level move and a steepening
                        are different macro statements and need not act alike on stocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_features import load_symbol_closes, lookback_return_pct
from mid_microstructure import PERCENT, trailing_change

RATE_SYMBOLS = ("ZT", "ZN")
RETURN_BARS = (1, 3, 12)
DIVERGENCE_BARS = (1, 2, 3)
CURVE_BARS = (1, 3, 12)

RATE_FEATURES = ([f"rf_{symbol.lower()}_ret_{bars}" for symbol in RATE_SYMBOLS for bars in RETURN_BARS]
                 + [f"rf_{symbol.lower()}_minus_mid_{bars}" for symbol in RATE_SYMBOLS
                    for bars in DIVERGENCE_BARS]
                 + [f"rf_curve_change_{bars}" for bars in CURVE_BARS])


def add_rate_features(data: pd.DataFrame) -> pd.DataFrame:
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    seconds = epochs
    returns = {}
    for symbol in RATE_SYMBOLS:
        closes = load_symbol_closes(symbol)
        for bars in sorted(set(RETURN_BARS) | set(DIVERGENCE_BARS) | set(CURVE_BARS)):
            returns[(symbol, bars)] = lookback_return_pct(closes, epochs, bars)
    mid_percent = {bars: trailing_change(mid, seconds, bars) / mid * PERCENT
                   for bars in sorted(set(DIVERGENCE_BARS))}
    columns = {}
    for symbol in RATE_SYMBOLS:
        for bars in RETURN_BARS:
            columns[f"rf_{symbol.lower()}_ret_{bars}"] = returns[(symbol, bars)]
        for bars in DIVERGENCE_BARS:
            columns[f"rf_{symbol.lower()}_minus_mid_{bars}"] = returns[(symbol, bars)] - mid_percent[bars]
    for bars in CURVE_BARS:
        # The short end moving more than the long end is the curve flattening, which says
        # something about policy that a parallel shift does not.
        columns[f"rf_curve_change_{bars}"] = returns[("ZT", bars)] - returns[("ZN", bars)]
    return data.assign(**columns)
