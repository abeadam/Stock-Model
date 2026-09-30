"""Session segments of the ES day, and the days that have every data source.

Segments (US/Eastern, by decision time):
  opening    09:30-10:30   the first hour of the stock session
  normal     10:30-15:00
  closing    15:00-16:00   the last hour of the stock session
  overnight  everything else ES trades: 16:00-17:00 and 18:00-09:30
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cross_asset_features import DAILY_DATA_DIR
from quote_features import QUOTES_DIR

EASTERN = "America/New_York"
SEGMENT_BOUNDARIES = (("opening", 9.5, 10.5), ("normal", 10.5, 15.0), ("closing", 15.0, 16.0))
OVERNIGHT = "overnight"
SEGMENTS = (OVERNIGHT, "opening", "normal", "closing")
SESSION_START_HOUR = 18          # an ES session starting 18:00 belongs to the next calendar day
REQUIRED_FILES = (
    (DAILY_DATA_DIR, "{day}_ES.txt"), (QUOTES_DIR, "{day}_ES_BID.txt"), (QUOTES_DIR, "{day}_ES_ASK.txt"),
    (DAILY_DATA_DIR, "{day}_TLT.txt"), (DAILY_DATA_DIR, "{day}_USO.txt"),
    (DAILY_DATA_DIR, "{day}_VXX.txt"), (DAILY_DATA_DIR, "{day}_UVXY.txt"),
)
MIN_FILE_BYTES = 1_000           # a header-only file holds no bars


def eastern_times(utc_times: pd.Series) -> pd.Series:
    return pd.to_datetime(utc_times).dt.tz_localize("UTC").dt.tz_convert(EASTERN)


def segment_of(et_times: pd.Series) -> np.ndarray:
    hours = (et_times.dt.hour + et_times.dt.minute / 60).to_numpy()
    segment = np.full(len(hours), OVERNIGHT, dtype=object)
    for name, start, end in SEGMENT_BOUNDARIES:
        segment[(hours >= start) & (hours < end)] = name
    return segment


def session_date(et_times: pd.Series) -> pd.Series:
    """The trading day each bar belongs to (ES sessions open at 18:00 the evening before)."""
    next_day = et_times.dt.hour >= SESSION_START_HOUR
    return (et_times.dt.normalize() + pd.to_timedelta(next_day.astype(int), unit="D")).dt.date


def days_with_all_data(candidate_days) -> set:
    """Days for which every REQUIRED_FILES entry exists and holds bars."""
    def has_bars(directory: str, pattern: str, day) -> bool:
        path = os.path.join(directory, pattern.format(day=day.isoformat()))
        return os.path.exists(path) and os.path.getsize(path) >= MIN_FILE_BYTES

    return {day for day in candidate_days
            if all(has_bars(directory, pattern, day) for directory, pattern in REQUIRED_FILES)}
