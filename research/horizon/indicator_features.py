"""Technical-indicator features for the symbols added alongside ES.

Those symbols -- rates (TLT), oil (USO), the traded VIX products (VXX, UVXY), gold
(GLD), bitcoin (IBIT), SPY, VXM and the ten index heavyweights -- have so far entered
the model only as returns over the last 1, 3, 12 or 60 bars. This adds the state each
one is in: stretched or compressed against its own band, above or below its moving
averages, quiet or agitated, and where its MACD stands.

Indicators are computed within each trading day, so a window never reaches back across
the overnight gap, and are then read at the ES decision bar. Outside a symbol's trading
hours its features are NaN, exactly as its returns already are.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from cross_asset_features import BASKET_SYMBOLS, load_symbol_closes
from macro_features import MACRO_SYMBOLS
from technical_indicators import (bollinger_position, distance_from_average, macd_histogram,
                                  realized_volatility, relative_strength_index, volatility_ratio)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indicator_cache")
EASTERN = "America/New_York"
EQUITY_SYMBOLS = ("SPY", "VXM", *BASKET_SYMBOLS)
FAST_BARS, SLOW_BARS = 12, 60                     # one minute and five minutes of 5-second bars
INDICATOR_NAMES = (f"rsi_{FAST_BARS}", f"rsi_{SLOW_BARS}", f"bollinger_{SLOW_BARS}",
                   f"ema_dist_{FAST_BARS}", f"ema_dist_{SLOW_BARS}", "vol_ratio", "macd")


def feature_names(symbols) -> list[str]:
    return [f"ti_{symbol.lower()}_{name}" for symbol in symbols for name in INDICATOR_NAMES]


MACRO_INDICATOR_FEATURES = feature_names(MACRO_SYMBOLS)
EQUITY_INDICATOR_FEATURES = feature_names(EQUITY_SYMBOLS)


def day_indicators(closes: pd.Series) -> pd.DataFrame:
    """Every indicator for one symbol on one day, indexed by bar-start epoch."""
    return pd.DataFrame({f"rsi_{FAST_BARS}": relative_strength_index(closes, FAST_BARS),
                         f"rsi_{SLOW_BARS}": relative_strength_index(closes, SLOW_BARS),
                         f"bollinger_{SLOW_BARS}": bollinger_position(closes, SLOW_BARS),
                         f"ema_dist_{FAST_BARS}": distance_from_average(closes, FAST_BARS),
                         f"ema_dist_{SLOW_BARS}": distance_from_average(closes, SLOW_BARS),
                         "vol_ratio": volatility_ratio(closes, FAST_BARS, SLOW_BARS),
                         "macd": macd_histogram(closes)}, index=closes.index)


def symbol_indicators(symbol: str) -> pd.DataFrame:
    """Every indicator for one symbol over every downloaded day, cached on disk."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    closes = load_symbol_closes(symbol)
    if closes.empty:
        return pd.DataFrame(columns=list(INDICATOR_NAMES), index=pd.Index([], dtype="int64"))
    trading_days = pd.to_datetime(closes.index, unit="s", utc=True).tz_convert(EASTERN).date
    indicators = pd.concat([day_indicators(day_closes) for _, day_closes in closes.groupby(trading_days)])
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(indicators, handle)
    return indicators


def add_indicator_features(data: pd.DataFrame, symbols) -> pd.DataFrame:
    """Adds one symbol's indicators per column to ES rows (`time` in UTC)."""
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    per_symbol = []
    for symbol in symbols:
        indicators = symbol_indicators(symbol).reindex(epochs).set_axis(data.index)
        per_symbol.append(indicators.add_prefix(f"ti_{symbol.lower()}_"))
    return pd.concat([data, *per_symbol], axis=1)
