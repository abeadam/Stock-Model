"""The same indicators read on wider ES bars: 5, 10, 15, 20, 25 and 30 seconds.

A 14-bar MFI on 5-second bars looks back 70 seconds. The same indicator on 30-second
bars looks back seven minutes and is far less sensitive to single prints, which is the
span a chart reader actually works at. This builds wider bars from the 5-second series
and reads the production indicator set on each width.

Two rules keep it honest:
  a wide bar is built only from a complete set of 5-second bars, so none straddles the
  overnight halt or a missing bar;
  a decision at the close of a 5-second bar may use a wide bar only once that wide bar
  has itself closed, so a half-finished 30-second bar is never read.

usage: multi_timeframe_features.py    # build the 5-second bar cache
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from forward_extremes import MODEL_INPUT
from technical_indicators import (bollinger_band_distance, distance_from_average, macd_histogram,
                                  money_flow_index, relative_strength_index, volatility_ratio)

BAR_CACHE = "bars_5s.npz"
BAR_COLUMNS = ("DateTime", "Close", "High", "Low", "Volume")
BASE_BAR_SECONDS = 5
TIMEFRAME_SECONDS = (5, 10, 15, 20, 25, 30)
MFI_BARS, RSI_BARS, BOLLINGER_BARS = 14, 14, 20
FAST_EMA_BARS, SLOW_EMA_BARS = 10, 20
INDICATOR_NAMES = ("mfi", "bb_upper_dist", "bb_lower_dist", "rsi", "ema_fast_dist", "ema_slow_dist",
                   "vol_ratio", "macd")


def feature_names(seconds: int, prefix: str = "mtf") -> list[str]:
    return [f"{prefix}_{seconds}s_{name}" for name in INDICATOR_NAMES]


TIMEFRAME_FEATURES = {seconds: feature_names(seconds) for seconds in TIMEFRAME_SECONDS}
ALL_TIMEFRAME_FEATURES = [name for names in TIMEFRAME_FEATURES.values() for name in names]
SPAN_FEATURES = {seconds: feature_names(seconds, "span") for seconds in TIMEFRAME_SECONDS}
ALL_SPAN_FEATURES = [name for names in SPAN_FEATURES.values() for name in names]


def build_bar_cache() -> None:
    frame = pd.read_csv(MODEL_INPUT, usecols=list(BAR_COLUMNS)).drop_duplicates("DateTime", keep="first")
    seconds = pd.to_datetime(frame["DateTime"]).to_numpy().astype("datetime64[s]").astype(np.int64)
    order = np.argsort(seconds, kind="stable")
    np.savez(BAR_CACHE, seconds=seconds[order],
             **{name.lower(): frame[name].to_numpy(np.float64)[order] for name in BAR_COLUMNS[1:]})
    print(f"{len(order):,} five-second bars cached to {BAR_CACHE}", flush=True)


def load_bars() -> pd.DataFrame:
    if not os.path.exists(BAR_CACHE):
        build_bar_cache()
    cached = np.load(BAR_CACHE)
    return pd.DataFrame({name: cached[name] for name in cached.files})


def widen(bars: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Bars of `seconds`, indexed by the epoch at which each one closes.

    Only groups holding every constituent 5-second bar are kept, so a wide bar never
    spans a halt or a gap."""
    if seconds == BASE_BAR_SECONDS:
        wide = bars.rename(columns={"seconds": "start"})
    else:
        grouped = bars.assign(start=bars["seconds"] // seconds * seconds).groupby("start")
        wide = grouped.agg(close=("close", "last"), high=("high", "max"), low=("low", "min"),
                           volume=("volume", "sum"), bars=("close", "size")).reset_index()
        wide = wide[wide["bars"] == seconds // BASE_BAR_SECONDS]
    return wide.assign(closes_at=wide["start"] + seconds).reset_index(drop=True)


def indicators_of(wide: pd.DataFrame, seconds: int, window_scale: int = 1,
                  prefix: str = "mtf") -> pd.DataFrame:
    """Every indicator for one bar width, indexed by the epoch the bar closes at.

    `window_scale` multiplies every lookback, so the same span of time can be reached
    either by widening the bars or by stretching the windows over narrow ones."""
    close, high, low, volume = (wide["close"], wide["high"], wide["low"], wide["volume"])
    mfi_bars, rsi_bars, bollinger_bars = (MFI_BARS * window_scale, RSI_BARS * window_scale,
                                          BOLLINGER_BARS * window_scale)
    fast_bars, slow_bars = FAST_EMA_BARS * window_scale, SLOW_EMA_BARS * window_scale
    values = {"mfi": money_flow_index(high, low, close, volume, mfi_bars),
              "bb_upper_dist": bollinger_band_distance(close, bollinger_bars, "upper"),
              "bb_lower_dist": bollinger_band_distance(close, bollinger_bars, "lower"),
              "rsi": relative_strength_index(close, rsi_bars),
              "ema_fast_dist": distance_from_average(close, fast_bars),
              "ema_slow_dist": distance_from_average(close, slow_bars),
              "vol_ratio": volatility_ratio(close, fast_bars, slow_bars),
              "macd": macd_histogram(close)}
    return pd.DataFrame({f"{prefix}_{seconds}s_{name}": series for name, series in values.items()},
                        index=wide["closes_at"].to_numpy())


def read_at_decisions(indicators: pd.DataFrame, decided_at: np.ndarray, index) -> pd.DataFrame:
    """The most recent indicator row that had already closed at each decision moment."""
    latest = np.searchsorted(indicators.index.to_numpy(), decided_at, side="right") - 1
    aligned = indicators.to_numpy()[np.clip(latest, 0, None)]
    aligned[latest < 0] = np.nan
    return pd.DataFrame(aligned, columns=indicators.columns, index=index)


def add_views(data: pd.DataFrame, views: list[tuple[int, int, int, str]],
              bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds indicator columns to ES rows (`time` in UTC) for each (bar width, label,
    window scale, prefix) view. `bars` is the 5-second series; the cached ES bars when omitted."""
    bars = load_bars() if bars is None else bars
    decided_at = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64) + BASE_BAR_SECONDS
    frames = [read_at_decisions(indicators_of(widen(bars, width), label, scale, prefix),
                                decided_at, data.index)
              for width, label, scale, prefix in views]
    return pd.concat([data, *frames], axis=1)


def add_timeframe_features(data: pd.DataFrame, widths=TIMEFRAME_SECONDS,
                           bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Indicators read on bars of each width, with the production lookbacks."""
    return add_views(data, [(width, width, 1, "mtf") for width in widths], bars)


def add_span_features(data: pd.DataFrame, spans=TIMEFRAME_SECONDS,
                      bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """The same spans of time reached the other way: native 5-second bars with the
    lookbacks stretched instead. Same history, full resolution."""
    return add_views(data, [(BASE_BAR_SECONDS, span, span // BASE_BAR_SECONDS, "span")
                            for span in spans], bars)


if __name__ == "__main__":
    build_bar_cache()
