"""Load and align ES, SPX, and SPY 5-second bars onto a common RTH grid.

ES is a 24-hour future; SPX and SPY only trade during regular hours (RTH).
All three share an identical 5-second timestamp grid, so an inner join on the
timestamp naturally restricts every output row to RTH (09:30:00-15:59:55 ET),
where all three instruments are simultaneously live.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterable

import pandas as pd

INSTRUMENTS = ("ES", "SPX", "SPY")
TIMESTAMP_COLUMN = "Date"
TRADING_DATE_COLUMN = "TradingDate"

# Per-instrument columns kept after alignment. Only ES carries the volume used
# for normalization; SPX volume is always zero and SPY volume is unused here.
_CLOSE_TEMPLATE = "{instrument}_Close"
ES_VOLUME_COLUMN = "ES_Volume"

_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "daily_data")
)
_FILENAME_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})_(\w+)\.txt$")


def close_column(instrument: str) -> str:
    """Column name holding the close price for an instrument after alignment."""
    return _CLOSE_TEMPLATE.format(instrument=instrument)


def find_common_trading_dates(data_dir: str = _DEFAULT_DATA_DIR) -> list[str]:
    """Return sorted dates for which every instrument has a data file."""
    dates_per_instrument = [
        _available_dates(instrument, data_dir) for instrument in INSTRUMENTS
    ]
    common_dates = set.intersection(*dates_per_instrument)
    return sorted(common_dates)


def load_aligned_bars(
    data_dir: str = _DEFAULT_DATA_DIR,
    trading_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and inner-join ES/SPX/SPY closes (plus ES volume) for each day.

    The result is sorted by timestamp and tagged with ``TradingDate`` so that
    downstream per-day computations never cross a session boundary. Passing
    ``trading_dates`` limits the load to a subset, which is useful for tests.
    """
    dates_to_load = (
        sorted(trading_dates)
        if trading_dates is not None
        else find_common_trading_dates(data_dir)
    )

    aligned_days = [
        _load_single_day(trading_date, data_dir) for trading_date in dates_to_load
    ]
    combined = pd.concat(aligned_days, ignore_index=True)
    return combined.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def _available_dates(instrument: str, data_dir: str) -> set[str]:
    paths = glob.glob(os.path.join(data_dir, f"*_{instrument}.txt"))
    dates = set()
    for path in paths:
        match = _FILENAME_DATE_PATTERN.search(os.path.basename(path))
        if match:
            dates.add(match.group(1))
    return dates


def _load_single_day(trading_date: str, data_dir: str) -> pd.DataFrame:
    """Inner-join one day's three instruments on the 5-second timestamp."""
    aligned: pd.DataFrame | None = None
    for instrument in INSTRUMENTS:
        bars = _read_instrument_day(instrument, trading_date, data_dir)
        aligned = bars if aligned is None else aligned.merge(bars, on=TIMESTAMP_COLUMN)

    assert aligned is not None  # INSTRUMENTS is non-empty
    aligned[TRADING_DATE_COLUMN] = trading_date
    return aligned


def _read_instrument_day(
    instrument: str, trading_date: str, data_dir: str
) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{trading_date}_{instrument}.txt")
    raw = pd.read_csv(path, usecols=[TIMESTAMP_COLUMN, "Close", "Volume"])

    columns = {"Close": _CLOSE_TEMPLATE.format(instrument=instrument)}
    if instrument == "ES":
        columns["Volume"] = ES_VOLUME_COLUMN
    else:
        raw = raw.drop(columns="Volume")
    return raw.rename(columns=columns)


if __name__ == "__main__":
    sample_dates = find_common_trading_dates()[:3]
    frame = load_aligned_bars(trading_dates=sample_dates)
    print(f"Loaded {len(frame):,} aligned RTH bars across {len(sample_dates)} days")
    print(f"Columns: {list(frame.columns)}")
    print(frame.head())
    print("\nRows per trading day:")
    print(frame[TRADING_DATE_COLUMN].value_counts().sort_index())
