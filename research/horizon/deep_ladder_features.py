"""The measurements already in the model, read at more lookback positions.

Every family stops short, and nobody has tested where. Cross-asset returns reach 12 bars,
the macro ETFs 60, the mid's own change 12, and the technical indicators are computed at
windows of 12 and 60 and nowhere else. The volatility grid tested that question for two
VIX funds and found depth worthless there -- but those funds are near-mirrors of ES, the
worst possible case. This asks it of every family instead.

One shared grid extends all of them from a single depth setting:

  deep_<symbol>_ret_<bars>        percent return of a symbol already in the model
  deep_mid_change_ticks_<bars>    the ES mid's own move over the span, in ticks
  deep_mid_reversion_<bars>       how far the mid stands from its average over that span
  deep_es_rsi_<bars>              indicators on ES itself with their windows scaled to the
  deep_es_bollinger_<bars>        rung, rather than left at the 12 and 60 the existing
  deep_es_ema_dist_<bars>         indicator set uses -- the count of indicators grows with
                                  the reach instead of the reach growing under fixed ones

The grid starts at 24 bars because almost every family already covers 12, and a rung that
restates a column already present is the dilution this is meant to measure, not add to.

Symbol returns are read by epoch, so they may reach into pre-market exactly as the existing
cross-asset columns do. The mid and indicator columns are gap-aware and stay inside the
session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_features import BASKET_SYMBOLS, lookback_return_pct
from macro_features import MACRO_SYMBOLS
from mid_microstructure import TICK, trailing_change
from technical_indicators import bollinger_position, distance_from_average, relative_strength_index

# Two minutes a rung: far enough past the 12-bar reach of most families that no rung
# duplicates an existing column.
DEEP_STEP_BARS = 24
MAX_DEEP_RUNGS = 6
LADDER_SYMBOLS = ("SPY", "VXM", *BASKET_SYMBOLS, *MACRO_SYMBOLS)
SCALED_INDICATORS = {"rsi": relative_strength_index, "bollinger": bollinger_position,
                     "ema_dist": distance_from_average}
MIN_INDICATOR_ROWS = 2


def deep_ladder(rungs: int, step_bars: int = DEEP_STEP_BARS) -> tuple[int, ...]:
    """`rungs` lookbacks spaced `step_bars` apart, the shallowest one step out."""
    if rungs < 0:
        raise ValueError(f"rungs must not be negative, got {rungs}")
    return tuple(step_bars * rung for rung in range(1, rungs + 1))


def deep_feature_names(ladder: tuple[int, ...],
                       symbols: tuple[str, ...] = LADDER_SYMBOLS) -> list[str]:
    return ([f"deep_{symbol.lower()}_ret_{bars}" for symbol in symbols for bars in ladder]
            + [f"deep_mid_{measure}_{bars}" for bars in ladder
               for measure in ("change_ticks", "reversion")]
            + [f"deep_es_{name}_{bars}" for bars in ladder for name in SCALED_INDICATORS])


def session_row_blocks(session_days: np.ndarray) -> list[np.ndarray]:
    """Row indices of each session, for a frame already ordered by time."""
    return np.split(np.arange(len(session_days)), np.flatnonzero(np.diff(session_days)) + 1)


def within_session(values: pd.Series, blocks: list[np.ndarray], indicator, window: int) -> np.ndarray:
    """An indicator computed inside each session, so no window spans the overnight gap."""
    result = np.full(len(values), np.nan)
    for rows in blocks:
        if len(rows) >= MIN_INDICATOR_ROWS:
            result[rows] = indicator(values.iloc[rows].reset_index(drop=True), window)
    return result


def add_deep_ladder_features(data: pd.DataFrame, closes_by_symbol: dict[str, pd.Series],
                             ladder: tuple[int, ...]) -> pd.DataFrame:
    """Adds `deep_feature_names(ladder, closes_by_symbol)` to ES rows holding `time`, `mid`,
    `Close` and `session_day`."""
    if not ladder:
        return data
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    blocks = session_row_blocks(data["session_day"].to_numpy())
    columns = {}
    for symbol, closes in closes_by_symbol.items():
        for bars in ladder:
            columns[f"deep_{symbol.lower()}_ret_{bars}"] = lookback_return_pct(closes, epochs, bars)
    for bars in ladder:
        change = trailing_change(mid, epochs, bars)
        columns[f"deep_mid_change_ticks_{bars}"] = change / TICK
        average = pd.Series(mid).rolling(bars, min_periods=bars).mean().to_numpy()
        reversion = (mid - average) / TICK
        reversion[np.isnan(change)] = np.nan
        columns[f"deep_mid_reversion_{bars}"] = reversion
    for bars in ladder:
        for name, indicator in SCALED_INDICATORS.items():
            columns[f"deep_es_{name}_{bars}"] = within_session(data["Close"], blocks, indicator, bars)
    return data.assign(**columns)
