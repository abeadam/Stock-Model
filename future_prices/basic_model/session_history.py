"""Prior sessions' ranges for the live session features, read from daily_data.

Training measures every session's range from the 5-second daily_data files
(futures_price.py reads all of them). Live code reads the same files, rather than
its own bar buffer or IBKR's daily bars, whose ranges differ, so a live lookback
matches the one the models were trained on.

The table is cached. Every REFRESH_INTERVAL_SECONDS the file list is checked and
re-read only if a file was added or changed: the weekday download writes the
session that just closed shortly after 17:30 ET, and a running trader picks it
up within minutes.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from market_clock import epoch_to_eastern
from session_features import PRIOR_SESSIONS_FOR_RANGE, session_ranges

# basic_model/ -> future_prices/ -> Stock-Model/daily_data. resolve() follows the
# symlinks the live traders import basic_model through.
DAILY_DATA_DIR = Path(__file__).resolve().parents[2] / "daily_data"
ES_FILE_GLOB = "*_ES.txt"
# ES day files read per pass while collecting enough complete sessions.
FILES_PER_PASS = 30
REFRESH_INTERVAL_SECONDS = 600
BAR_COLUMNS = ["Date", "High", "Low"]

_cache: dict = {"checked_at": None, "files_key": None, "ranges": None}


def load_prior_session_ranges(data_dir: Path = DAILY_DATA_DIR) -> pd.Series:
    """session_ranges() over the most recent ES day files, reading further back
    until there are more complete sessions than one lookback needs, or no files remain."""
    files = sorted(data_dir.glob(ES_FILE_GLOB))
    files_to_read = FILES_PER_PASS
    while True:
        bars = (pd.concat([pd.read_csv(path, usecols=BAR_COLUMNS) for path in files[-files_to_read:]],
                          ignore_index=True)
                .drop_duplicates(subset="Date", keep="last")
                .sort_values("Date")
                .reset_index(drop=True))
        ranges = session_ranges(epoch_to_eastern(bars["Date"]), bars["High"], bars["Low"])
        if len(ranges) > PRIOR_SESSIONS_FOR_RANGE or files_to_read >= len(files):
            return ranges
        files_to_read += FILES_PER_PASS


def prior_session_ranges(data_dir: Path = DAILY_DATA_DIR) -> pd.Series:
    """The cached history table; an empty one if the files cannot be read, which
    leaves session_move_vs_prior_range NaN rather than stopping the trader."""
    now = time.monotonic()
    if _cache["checked_at"] is not None and now - _cache["checked_at"] < REFRESH_INTERVAL_SECONDS:
        return _cache["ranges"]
    _cache["checked_at"] = now
    try:
        files_key = tuple((path.name, path.stat().st_mtime) for path in sorted(data_dir.glob(ES_FILE_GLOB)))
        if files_key != _cache["files_key"]:
            _cache["ranges"] = load_prior_session_ranges(data_dir)
            _cache["files_key"] = files_key
    except (OSError, ValueError) as error:
        print(f"Warning: cannot read prior session ranges from {data_dir}: {error}")
        _cache["ranges"] = pd.Series(dtype=float, name="range")
    return _cache["ranges"]
