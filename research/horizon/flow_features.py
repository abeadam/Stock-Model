"""Order-flow and short-lookback features from 5-second ES bars.

Everything uses the current bar and earlier ones only (a decision is made at the
bar's close). A lookback whose window crosses a gap in the 5-second series (the
daily break, a halt, missing data) is NaN rather than silently spanning the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_SECONDS = 5
IMBALANCE_WINDOWS = (6, 12, 60)          # 30 s, 1 min, 5 min
RETURN_WINDOWS = (12, 60, 360)           # 1 min, 5 min, 30 min
RANGE_WINDOW = 60
VOLUME_BASELINE_WINDOW = 360
FLOW_FEATURES = ([f"flow_imbalance_{n}" for n in IMBALANCE_WINDOWS]
                 + [f"flow_return_{n}" for n in RETURN_WINDOWS]
                 + ["flow_volume_ratio", "flow_bar_close_location",
                    f"flow_range_position_{RANGE_WINDOW}", f"flow_range_pct_{RANGE_WINDOW}",
                    "flow_bars_since_price_change"])


def _window_is_contiguous(seconds: np.ndarray, bars: int) -> np.ndarray:
    """True where the window of `bars` bars ending at each row has no gap."""
    span = np.full(len(seconds), -1, np.int64)
    span[bars - 1:] = seconds[bars - 1:] - seconds[:len(seconds) - bars + 1]
    return span == (bars - 1) * BAR_SECONDS


def _bars_since_price_change(close: np.ndarray) -> np.ndarray:
    changed = np.r_[True, np.diff(close) != 0]
    last_change = np.maximum.accumulate(np.where(changed, np.arange(len(close)), 0))
    return (np.arange(len(close)) - last_change).astype(float)


def add_flow_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adds FLOW_FEATURES to a frame of consecutive bars sorted by time (Close, High, Low, Volume, time)."""
    close, high, low = data["Close"], data["High"], data["Low"]
    volume = data["Volume"].astype(float)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    signed_volume = np.sign(close.diff().fillna(0)) * volume

    features = {}
    for bars in IMBALANCE_WINDOWS:
        total = volume.rolling(bars).sum()
        imbalance = signed_volume.rolling(bars).sum() / total.where(total > 0)
        features[f"flow_imbalance_{bars}"] = imbalance.where(_window_is_contiguous(seconds, bars + 1))
    for bars in RETURN_WINDOWS:
        change_pct = (close / close.shift(bars) - 1) * 100
        features[f"flow_return_{bars}"] = change_pct.where(_window_is_contiguous(seconds, bars + 1))
    baseline = volume.rolling(VOLUME_BASELINE_WINDOW).mean()
    features["flow_volume_ratio"] = (volume / baseline.where(baseline > 0)).where(
        _window_is_contiguous(seconds, VOLUME_BASELINE_WINDOW))
    bar_range = (high - low).where(high > low)
    features["flow_bar_close_location"] = (close - low) / bar_range
    window_high = high.rolling(RANGE_WINDOW).max()
    window_low = low.rolling(RANGE_WINDOW).min()
    window_range = (window_high - window_low).where(window_high > window_low)
    range_ok = _window_is_contiguous(seconds, RANGE_WINDOW)
    features[f"flow_range_position_{RANGE_WINDOW}"] = ((close - window_low) / window_range).where(range_ok)
    features[f"flow_range_pct_{RANGE_WINDOW}"] = (window_range / close * 100).where(range_ok)
    features["flow_bars_since_price_change"] = _bars_since_price_change(close.to_numpy())
    return data.assign(**features)
