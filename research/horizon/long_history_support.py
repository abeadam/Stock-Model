"""The 18-bar test over as much history as the data allows.

Everything measured this week rests on two dev folds, and two folds disagreed on nearly
every effect: a rotation helped in June and hurt in July, an implied-vol level cost 0.049
in one month and gained 0.006 in the next, a P&L objective fitted one tree in one fold and
fifty in the other. Eight folds is the point of this.

Reaching back costs columns. Only ES, VXM and nine of the ten index heavyweights run to
2025-06-26; BRK.B starts 2026-03-22, SPY only 2026-01-29, and the quote, macro-ETF and
mid-microstructure families all begin 2026-03-26. So the long-history base is FEATURES,
FLOW_FEATURES and the cross-asset columns that reach back, and the two results worth
retesting are recomputed from **Close** because there is no mid this far back:

  realized volatility at 60 and 180 bars, measured inside each session
  EMA and RMA at 30 and 60 of high_close_ratio and low_close_ratio -- the TA-Lib
  smoothings that won the range screen

**A Close-based target carries bid-ask bounce**, which is what turned the gradient model's
0.546 into roughly 0.52 under a one-bar fill delay. Every result here is therefore
reported twice: at the signal bar, and with the move measured from one bar later. If the
delayed version collapses, the long-history result is bounce and nothing else.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

from cross_asset_features import BASKET_SYMBOLS, CROSS_ASSET_FEATURES
from deep_ladder_features import session_row_blocks
from realized_volatility_features import realized_volatility
from flow_features import FLOW_FEATURES
from short_horizon_walk_forward import FEATURES, FoldSplit, load as load_model_input
from volume_scaled_target import regular_hours

DAILY_DATA_DIR = "/Users/abeadam/dev/model/Stock-Model/daily_data"
BAR_SECONDS = 5
PERCENT = 100.0
# Reach back only as far as every required symbol does.
LATE_SYMBOLS = ("BRK.B",)
REALIZED_WINDOWS = (60, 180)
SMOOTHED_COLUMNS = ("high_close_ratio", "low_close_ratio")
SMOOTHERS = {"ema": talib.EMA, "rma": talib.RMA}
SMOOTHING_WINDOWS = (30, 60)
VALIDATION_FRACTION = 0.15
MIN_TRAIN_SESSIONS = 40
EPSILON = 1e-12

REALIZED_FEATURES = [f"lh_realized_vol_{window}" for window in REALIZED_WINDOWS]
SMOOTHED_FEATURES = [f"lh_{column}_{name}_{window}" for column in SMOOTHED_COLUMNS
                     for name in SMOOTHERS for window in SMOOTHING_WINDOWS]


def sessions_with(symbol: str) -> set[str]:
    return {os.path.basename(path).split('_')[0]
            for path in glob.glob(os.path.join(DAILY_DATA_DIR, f"*_{symbol}.txt"))}


def long_history_symbols(include_spy: bool) -> tuple[str, ...]:
    """Cross-asset symbols that reach back far enough to be worth carrying."""
    basket = tuple(name for name in BASKET_SYMBOLS if name not in LATE_SYMBOLS)
    return ("SPY", "VXM", *basket) if include_spy else ("VXM", *basket)


def base_columns(include_spy: bool) -> list[str]:
    """FEATURES + flow + the cross-asset columns whose symbols reach back.

    Every cross-asset name carries its symbol in lower case, so dropping a symbol is a
    substring test: "spy" removes both xa_spy_ret_* and xa_spy_minus_es_*. The basket
    columns stay -- `basket_return_pct` needs five of the ten names and has nine."""
    excluded = [name.lower() for name in LATE_SYMBOLS] + ([] if include_spy else ["spy"])
    cross = [name for name in CROSS_ASSET_FEATURES
             if not any(symbol in name for symbol in excluded)]
    return list(FEATURES) + FLOW_FEATURES + cross


def usable_sessions(include_spy: bool) -> set[str]:
    available = None
    for symbol in ("ES", *long_history_symbols(include_spy)):
        found = sessions_with(symbol)
        available = found if available is None else available & found
    return available or set()


def add_long_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """The two results worth retesting, recomputed from Close."""
    session_days = data["session_day"].to_numpy()
    close = data["Close"].to_numpy(float)
    columns = {f"lh_realized_vol_{window}": realized_volatility(close, session_days, window)
               for window in REALIZED_WINDOWS}
    for column in SMOOTHED_COLUMNS:
        values = data[column].to_numpy(float)
        for name, smoother in SMOOTHERS.items():
            for window in SMOOTHING_WINDOWS:
                columns[f"lh_{column}_{name}_{window}"] = smoother(values, timeperiod=window)
    return data.assign(**columns)


def forward_average_target(close: np.ndarray, seconds: np.ndarray, bars: int,
                           delay: int = 0) -> np.ndarray:
    """Percent move from the bar `delay` ahead to the average of the `bars` after it.

    `delay` is the fill-delay control: at 1 the entry price is the next bar's close, so the
    signal bar's own bid-ask bounce cannot be part of the measured move."""
    series = pd.Series(close)
    reached = series.rolling(bars, min_periods=bars).mean().shift(-(delay + bars)).to_numpy()
    reference = series.shift(-delay).to_numpy()
    reach = delay + bars
    contiguous = np.zeros(len(close), bool)
    contiguous[:-reach] = (seconds[reach:] - seconds[:-reach]) == reach * BAR_SECONDS
    safe = np.where(np.abs(reference) > EPSILON, reference, np.nan)
    return np.where(contiguous, (reached - safe) / safe * PERCENT, np.nan)


def forward_range_target(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         seconds: np.ndarray, bars: int) -> np.ndarray:
    """High-to-low range over the next `bars` bars, as a percent of the current close."""
    highest = pd.Series(high).rolling(bars, min_periods=bars).max().shift(-bars).to_numpy()
    lowest = pd.Series(low).rolling(bars, min_periods=bars).min().shift(-bars).to_numpy()
    contiguous = np.zeros(len(close), bool)
    contiguous[:-bars] = (seconds[bars:] - seconds[:-bars]) == bars * BAR_SECONDS
    safe = np.where(np.abs(close) > EPSILON, close, np.nan)
    return np.where(contiguous, (highest - lowest) / safe * PERCENT, np.nan)


@dataclass(frozen=True)
class MonthlyFold:
    month: str
    split: FoldSplit


def monthly_folds(session_days: np.ndarray, horizon: int,
                  min_train_sessions: int = MIN_TRAIN_SESSIONS) -> list[MonthlyFold]:
    """Each month tested on everything before it, purged by the horizon at both joins."""
    months = np.array([f"{day.year:04d}-{day.month:02d}" for day in session_days])
    folds = []
    for month in sorted(set(months)):
        test = np.flatnonzero(months == month)
        history = np.flatnonzero(months < month)
        if len(set(session_days[history])) < min_train_sessions:
            continue
        cut = int(len(history) * (1 - VALIDATION_FRACTION))
        validation = history[cut:]
        train = history[:max(cut - horizon, 0)]
        folds.append(MonthlyFold(month, FoldSplit(train=train, validation=validation,
                                                  test=test[horizon:])))
    return folds


def build(include_spy: bool) -> pd.DataFrame:
    """Regular-hours rows over every session the long-history symbols all cover."""
    data = load_model_input()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    eastern = data["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    data = data.assign(session_day=eastern.dt.date.to_numpy())
    keep = {pd.Timestamp(session).date() for session in usable_sessions(include_spy)}
    data = data[data["session_day"].isin(keep)].reset_index(drop=True)
    return add_long_history_features(data)
