"""Cross-asset lead-lag features for 5-second ES decisions.

Short returns of SPY, a mega-cap stock basket and VXM, and how far each has moved
beyond ES over the same window. All series are 5-second IBKR bars keyed by bar
start (epoch seconds, UTC); the bar starting at t closes at t + 5, the same moment
as the ES bar starting at t, so a same-start bar is known at decision time.
Outside a symbol's hours (stocks: 04:00-16:00 ET) its features are NaN.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import pandas as pd

DAILY_DATA_DIR = "/Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_asset_cache")
BAR_SECONDS = 5
LOOKBACK_BARS = (1, 3, 12)
# The S&P 500's ten heaviest names (BRK.B was missing from the downloads until 2026-09-22).
BASKET_SYMBOLS = ("NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOG", "AVGO", "TSLA", "JPM", "BRK.B")
MIN_BASKET_MEMBERS = 5
CROSS_ASSET_FEATURES = ([f"xa_{name}_{bars}" for bars in LOOKBACK_BARS
                         for name in ("spy_ret", "basket_ret", "vxm_ret", "spy_minus_es", "basket_minus_es")]
                        + [f"xa_{symbol.lower()}_ret_{bars}" for symbol in BASKET_SYMBOLS
                           for bars in LOOKBACK_BARS])


def load_symbol_closes(symbol: str, directory: str = DAILY_DATA_DIR) -> pd.Series:
    """Close by bar-start epoch for every downloaded day of `symbol`, cached on disk.
    Delete the cache after downloading more days."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    files = sorted(glob.glob(os.path.join(directory, f"*_{symbol}.txt")))
    if not files:                       # still downloading: its features stay blank
        return pd.Series(dtype=float, index=pd.Index([], dtype="int64"))
    frames = [pd.read_csv(path, usecols=["Date", "Close"]) for path in files]
    closes = (pd.concat(frames).drop_duplicates("Date", keep="last")
              .set_index("Date")["Close"].sort_index().astype(float))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(closes, handle)
    return closes


def lookback_return_pct(closes: pd.Series, epochs: np.ndarray, bars: int) -> np.ndarray:
    """Percent change from the bar `bars` bars before each epoch to the bar at it; NaN if either is missing."""
    now = closes.reindex(epochs).to_numpy()
    before = closes.reindex(epochs - bars * BAR_SECONDS).to_numpy()
    return (now / before - 1) * 100


def basket_return_pct(basket: list[pd.Series], epochs: np.ndarray, bars: int) -> np.ndarray:
    returns = np.vstack([lookback_return_pct(closes, epochs, bars) for closes in basket])
    members = np.sum(~np.isnan(returns), axis=0)
    total = np.nansum(returns, axis=0)
    return np.where(members >= MIN_BASKET_MEMBERS, total / np.maximum(members, 1), np.nan)


def add_cross_asset_features(data: pd.DataFrame, series: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Adds CROSS_ASSET_FEATURES to ES rows (columns `time` in UTC and `Close`).
    `series` maps symbol -> closes by epoch; loaded from the daily files when omitted."""
    if series is None:
        series = {symbol: load_symbol_closes(symbol) for symbol in ("SPY", "VXM", *BASKET_SYMBOLS)}
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    es_closes = pd.Series(data["Close"].to_numpy(float), index=epochs)
    es_closes = es_closes[~es_closes.index.duplicated()]
    basket = [series[symbol] for symbol in BASKET_SYMBOLS]
    features = {}
    # Each index heavyweight on its own: a basket average hides one name leading the rest.
    for symbol in BASKET_SYMBOLS:
        for bars in LOOKBACK_BARS:
            features[f"xa_{symbol.lower()}_ret_{bars}"] = lookback_return_pct(series[symbol], epochs, bars)
    for bars in LOOKBACK_BARS:
        es_return = lookback_return_pct(es_closes, epochs, bars)
        spy_return = lookback_return_pct(series["SPY"], epochs, bars)
        basket_return = basket_return_pct(basket, epochs, bars)
        features[f"xa_spy_ret_{bars}"] = spy_return
        features[f"xa_basket_ret_{bars}"] = basket_return
        features[f"xa_vxm_ret_{bars}"] = lookback_return_pct(series["VXM"], epochs, bars)
        features[f"xa_spy_minus_es_{bars}"] = spy_return - es_return
        features[f"xa_basket_minus_es_{bars}"] = basket_return - es_return
    return data.assign(**features)
