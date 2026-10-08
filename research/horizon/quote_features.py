"""ES bid/ask quote features and mid prices from 5-second BID and ASK bars.

A BID bar's close is the last bid in that 5-second bar (ASK likewise), so at an ES
decision bar's close we know the standing quote. Mid-price moves carry no bid-ask
bounce, and fills can be priced at the actual bid and ask.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cross_asset_features import BAR_SECONDS, DAILY_DATA_DIR, load_symbol_closes

QUOTES_DIR = os.path.join(DAILY_DATA_DIR, "quotes")
TICK = 0.25
MID_LOOKBACK_BARS = (1, 3, 12)
QUOTE_FEATURES = (["quote_spread_ticks", "quote_trade_vs_mid_ticks"]
                  + [f"quote_mid_change_{bars}" for bars in MID_LOOKBACK_BARS])


def load_quotes() -> pd.DataFrame:
    """Last bid and ask by bar-start epoch; rows with a missing or crossed quote dropped."""
    quotes = pd.DataFrame({"bid": load_symbol_closes("ES_BID", QUOTES_DIR),
                           "ask": load_symbol_closes("ES_ASK", QUOTES_DIR)}).dropna()
    return quotes[(quotes["bid"] > 0) & (quotes["ask"] >= quotes["bid"])]


def add_quote_columns(data: pd.DataFrame, quotes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds bid, ask, mid (for targets and fills) and QUOTE_FEATURES to ES rows (`time` UTC, `Close`)."""
    quotes = load_quotes() if quotes is None else quotes
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    bid = quotes["bid"].reindex(epochs).to_numpy()
    ask = quotes["ask"].reindex(epochs).to_numpy()
    mid = (bid + ask) / 2
    mid_by_epoch = pd.Series(mid, index=epochs)
    mid_by_epoch = mid_by_epoch[~mid_by_epoch.index.duplicated()]
    columns = {"bid": bid, "ask": ask, "mid": mid,
               "quote_spread_ticks": (ask - bid) / TICK,
               "quote_trade_vs_mid_ticks": (data["Close"].to_numpy() - mid) / TICK}
    for bars in MID_LOOKBACK_BARS:
        earlier_mid = mid_by_epoch.reindex(epochs - bars * BAR_SECONDS).to_numpy()
        columns[f"quote_mid_change_{bars}"] = (mid - earlier_mid) / TICK
    return data.assign(**columns)
