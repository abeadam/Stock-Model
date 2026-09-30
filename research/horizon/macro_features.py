"""Rates (TLT), crude oil (USO), traded VIX products (VXX, UVXY), gold (GLD),
bitcoin (IBIT) and the dollar (UUP) as lead-lag features for 5-second ES decisions.

The VIX index itself is computed from option prices and does not trade; these two
ETFs do, so they carry order flow that the index cannot. The same holds for the dollar:
the index and the spot currencies trade over the counter, while UUP trades on an
exchange and prints bars that line up with the ES ones.

Same construction as cross_asset_features: the bar starting at t closes with the
ES bar starting at t, so a same-start bar is known at decision time. Outside the
ETFs' 04:00-16:00 ET bars these features are NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_features import load_symbol_closes, lookback_return_pct

MACRO_SYMBOLS = ("TLT", "USO", "VXX", "UVXY", "GLD", "IBIT", "UUP")
MACRO_LOOKBACK_BARS = (1, 3, 12, 60)
MACRO_FEATURES = [f"macro_{symbol.lower()}_ret_{bars}" for symbol in MACRO_SYMBOLS for bars in MACRO_LOOKBACK_BARS]


def add_macro_features(data: pd.DataFrame, series: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Adds MACRO_FEATURES to ES rows (`time` in UTC)."""
    series = {symbol: load_symbol_closes(symbol) for symbol in MACRO_SYMBOLS} if series is None else series
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    return data.assign(**{f"macro_{symbol.lower()}_ret_{bars}": lookback_return_pct(series[symbol], epochs, bars)
                          for symbol in MACRO_SYMBOLS for bars in MACRO_LOOKBACK_BARS})
