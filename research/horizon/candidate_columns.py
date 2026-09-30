"""Candidate columns to screen before any of them is allowed into the model.

Three ways of building one, all measured against the ES mid so none can carry the bounce:

  divergence   a symbol's return less the mid's over the same window, for every symbol
               downloaded that is not already used this way -- rates, the VIX funds,
               bitcoin, and the mega-caps beyond Google
  scaled       the divergences already in the model, divided by how much the mid has been
               moving lately, so a gap is judged against the current regime instead of an
               absolute size
  spread       differences between symbols rather than against ES: how far the index fund
               has run from the mega-cap basket, how wide the basket itself is spread, and
               gold against oil

Nothing here is added to the model. Its purpose is to be ranked, so that the handful worth
a paired ablation can be picked on evidence rather than on which ticker came to mind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_features import BASKET_SYMBOLS, load_symbol_closes, lookback_return_pct
from mid_microstructure import DIVERGENCE_WINDOWS, PERCENT, mid_volatility, trailing_change

CANDIDATE_SYMBOLS = ("VXM", "TLT", "VXX", "UVXY", "IBIT", "NVDA", "AAPL", "MSFT", "AMZN",
                     "AVGO", "META", "TSLA", "BRK.B", "JPM")
CANDIDATE_BARS = (1, 3, 12)
SCALED_SYMBOLS = tuple(DIVERGENCE_WINDOWS)
SPREAD_BARS = (1, 3)
MIN_BASKET_MEMBERS = 5


def clean(symbol: str) -> str:
    return symbol.lower().replace(".", "")


def candidate_names() -> list[str]:
    names = [f"cand_{clean(symbol)}_minus_mid_{bars}"
             for symbol in CANDIDATE_SYMBOLS for bars in CANDIDATE_BARS]
    names += [f"cand_{clean(symbol)}_minus_mid_scaled_{bars}"
              for symbol in SCALED_SYMBOLS for bars in DIVERGENCE_WINDOWS[symbol]]
    names += [f"cand_spy_minus_basket_{bars}" for bars in SPREAD_BARS]
    names += [f"cand_basket_dispersion_{bars}" for bars in SPREAD_BARS]
    names += [f"cand_gold_minus_oil_{bars}" for bars in SPREAD_BARS]
    return names


def add_candidates(data: pd.DataFrame) -> pd.DataFrame:
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    volatility = mid_volatility(mid, epochs)
    mid_percent = {bars: trailing_change(mid, epochs, bars) / mid * PERCENT
                   for bars in sorted(set(CANDIDATE_BARS) | set(SPREAD_BARS)
                                      | {b for w in DIVERGENCE_WINDOWS.values() for b in w})}
    returns = {}

    def symbol_return(symbol: str, bars: int) -> np.ndarray:
        if (symbol, bars) not in returns:
            returns[(symbol, bars)] = lookback_return_pct(load_symbol_closes(symbol), epochs, bars)
        return returns[(symbol, bars)]

    columns = {}
    for symbol in CANDIDATE_SYMBOLS:
        for bars in CANDIDATE_BARS:
            columns[f"cand_{clean(symbol)}_minus_mid_{bars}"] = symbol_return(symbol, bars) - mid_percent[bars]
    for symbol in SCALED_SYMBOLS:
        for bars in DIVERGENCE_WINDOWS[symbol]:
            gap = symbol_return(symbol, bars) - mid_percent[bars]
            columns[f"cand_{clean(symbol)}_minus_mid_scaled_{bars}"] = gap / (volatility * np.sqrt(bars))
    for bars in SPREAD_BARS:
        members = np.vstack([symbol_return(symbol, bars) for symbol in BASKET_SYMBOLS])
        present = np.sum(~np.isnan(members), axis=0)
        basket = np.where(present >= MIN_BASKET_MEMBERS, np.nanmean(members, axis=0), np.nan)
        columns[f"cand_spy_minus_basket_{bars}"] = symbol_return("SPY", bars) - basket
        columns[f"cand_basket_dispersion_{bars}"] = np.where(present >= MIN_BASKET_MEMBERS,
                                                             np.nanstd(members, axis=0), np.nan)
        columns[f"cand_gold_minus_oil_{bars}"] = symbol_return("GLD", bars) - symbol_return("USO", bars)
    return data.assign(**columns)
