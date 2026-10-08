"""The clock every model feature is computed on: US/Eastern wall-clock time.

futures_price.py builds the training data and converts each bar from UTC to
US/Eastern before deriving anything time-based, so live code must read bar
times the same way. Unix seconds are UTC; date strings carry no zone and are
taken to be Eastern already.
"""

from __future__ import annotations

import pandas as pd

EASTERN_TIMEZONE = "America/New_York"

# Anchors used by futures_price.calculate_trading_hours.
FORMAL_TRADING_START_HOUR = 9.0
# The CME Globex trading day for ES opens at 18:00 ET the evening before.
OVERNIGHT_SESSION_START_HOUR = 18.0
HOURS_PER_DAY = 24.0


def epoch_to_eastern(epoch_seconds: pd.Series) -> pd.Series:
    """Unix seconds (always UTC) as naive US/Eastern wall-clock datetimes."""
    utc = pd.to_datetime(epoch_seconds, unit="s", utc=True, errors="coerce")
    return utc.dt.tz_convert(EASTERN_TIMEZONE).dt.tz_localize(None)


def parse_to_eastern(date_col: pd.Series) -> pd.Series:
    """Parse a date column to naive US/Eastern datetimes, without the dateutil
    per-element fallback.

    Handles the formats this code sees: Unix seconds as a number (offline ES
    files) or as a string (the live IB feed under formatDate=2), and string
    timestamps from either the IB feed ("YYYYMMDD HH:MM:SS", sometimes
    double-spaced) or pandas/ISO ("YYYY-MM-DD HH:MM:SS"). Passing an explicit
    format / "mixed" avoids pandas' "Could not infer format" warning, which is
    both noisy and slow in the live per-bar loop.
    """
    if pd.api.types.is_numeric_dtype(date_col):
        return epoch_to_eastern(date_col)
    cleaned = date_col.astype(str).str.strip().str.replace("  ", " ", regex=False)
    # The TWS API hands epoch seconds back as a *string* under formatDate=2, so a
    # column of all-digit strings is still Unix seconds and must not go down the
    # date-string path — neither format below matches it, which would silently
    # yield NaT and turn every time-of-day feature into NaN.
    as_epoch = pd.to_numeric(cleaned, errors="coerce")
    if as_epoch.notna().all():
        return epoch_to_eastern(as_epoch)
    parsed = pd.to_datetime(cleaned, format="%Y%m%d %H:%M:%S", errors="coerce")
    if parsed.isna().all():  # not the IB compact format — fall back to general parsing
        parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed")
    return parsed
