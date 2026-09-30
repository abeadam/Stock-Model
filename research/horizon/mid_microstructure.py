"""Microstructure features built on the mid, and on how far the stocks have run from it.

The attribution against a bounce-free target left two kinds of contributor standing:
the mid's own recent change, and the very recent returns of SPY, the mega-caps and gold.
Everything that measured where the last *trade* printed collapsed once the target could
no longer bounce. These features are built to keep the first kind and exclude the second.

  mid_change_ticks_n      the mid's move over the last n bars, in ticks rather than
                          percent, so it means the same thing at any index level
  mid_change_z_n          that move divided by how much the mid has been moving lately,
                          so "a big push" is judged against the current regime rather
                          than an absolute threshold
  mid_reversion_n         how far the mid stands from its own recent average, in ticks:
                          the stretched-rubber-band reading
  spread_ticks            the quoted spread, which says how thin the book is
  wick_up/down_ticks      the traded high and low against the close, in ticks. The ratio
                          form of these survived the bounce test only partly, and ticks
                          are the better-posed unit, so both are offered here
  <symbol>_minus_mid_n    how much more a stock or fund has moved than the ES mid over
                          the same n bars: the divergence that the attribution found,
                          measured against the mid instead of the bouncing trade price.
                          Each symbol is read at the windows where its own signal sits

Everything looks backwards only. Windows reaching across a session gap are left blank
rather than quietly spanning the halt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.25
PERCENT = 100.0
BAR_SECONDS = 5
CHANGE_BARS = (1, 2, 3, 6, 12)
REVERSION_BARS = (12, 60)
VOLATILITY_BARS = 60
# Each symbol gets the windows its own signal lives at rather than one shared set: the
# attribution put SPY, Google and gold at one bar and oil at three, and a currency fund
# moves on a slower clock than a mega-cap, so oil and the dollar are given longer looks.
DIVERGENCE_WINDOWS = {"SPY": (1, 2, 3), "GOOG": (1, 2, 3), "GLD": (1, 2, 3),
                      "USO": (3, 6, 12), "UUP": (1, 3, 12)}
DIVERGENCE_SYMBOLS = tuple(DIVERGENCE_WINDOWS)
DIVERGENCE_BARS = tuple(sorted({bars for windows in DIVERGENCE_WINDOWS.values() for bars in windows}))
MIN_SPREAD = 1e-9

MID_FEATURES = ([f"ms_mid_change_ticks_{bars}" for bars in CHANGE_BARS]
                + [f"ms_mid_change_z_{bars}" for bars in CHANGE_BARS]
                + [f"ms_mid_reversion_{bars}" for bars in REVERSION_BARS]
                + ["ms_wick_up_ticks", "ms_wick_down_ticks", "ms_wick_asymmetry_ticks"]
                + [f"ms_{symbol.lower()}_minus_mid_{bars}" for symbol, windows in DIVERGENCE_WINDOWS.items()
                   for bars in windows])


def trailing_change(values: np.ndarray, seconds: np.ndarray, bars: int) -> np.ndarray:
    """values[t] - values[t-bars]; blank where the window crosses a gap."""
    change = np.full(len(values), np.nan)
    change[bars:] = values[bars:] - values[:-bars]
    contiguous = np.zeros(len(values), bool)
    contiguous[bars:] = (seconds[bars:] - seconds[:-bars]) == bars * BAR_SECONDS
    change[~contiguous] = np.nan
    return change


def mid_volatility(mid: np.ndarray, seconds: np.ndarray, window: int = VOLATILITY_BARS) -> np.ndarray:
    """Typical size of a one-bar mid move over the recent past, in ticks."""
    step = trailing_change(mid, seconds, 1) / TICK
    spread = pd.Series(step).rolling(window, min_periods=window // 4).std().to_numpy()
    return np.where(spread > MIN_SPREAD, spread, np.nan)


def add_mid_features(data: pd.DataFrame, symbol_returns: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Adds MID_FEATURES to ES rows holding `time`, `mid`, `High`, `Low`, `Close`, `bid`, `ask`.

    `symbol_returns` maps a symbol to its percent return over each divergence window,
    keyed "SPY_1" and so on; taken from the cross-asset columns when omitted."""
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    close, high, low = (data[name].to_numpy(float) for name in ("Close", "High", "Low"))
    volatility = mid_volatility(mid, seconds)
    columns = {}
    for bars in CHANGE_BARS:
        change_ticks = trailing_change(mid, seconds, bars) / TICK
        columns[f"ms_mid_change_ticks_{bars}"] = change_ticks
        columns[f"ms_mid_change_z_{bars}"] = change_ticks / (volatility * np.sqrt(bars))
    for bars in REVERSION_BARS:
        average = pd.Series(mid).rolling(bars, min_periods=bars).mean().to_numpy()
        reversion = (mid - average) / TICK
        reversion[np.isnan(trailing_change(mid, seconds, bars))] = np.nan
        columns[f"ms_mid_reversion_{bars}"] = reversion
    columns["ms_wick_up_ticks"] = (high - close) / TICK
    columns["ms_wick_down_ticks"] = (close - low) / TICK
    columns["ms_wick_asymmetry_ticks"] = (high + low - 2 * close) / TICK
    mid_percent = {bars: trailing_change(mid, seconds, bars) / mid * PERCENT for bars in DIVERGENCE_BARS}
    for symbol, windows in DIVERGENCE_WINDOWS.items():
        for bars in windows:
            symbol_return = symbol_returns[f"{symbol}_{bars}"]
            columns[f"ms_{symbol.lower()}_minus_mid_{bars}"] = symbol_return - mid_percent[bars]
    return data.assign(**columns)
