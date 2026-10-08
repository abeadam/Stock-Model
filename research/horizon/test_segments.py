import datetime as dt
from unittest import mock

import numpy as np
import pandas as pd

import segments
from segments import days_with_all_data, eastern_times, segment_of, session_date


def et(*clock_times):
    return pd.Series(pd.to_datetime(list(clock_times))).dt.tz_localize("America/New_York")


def test_segment_boundaries():
    times = et("2026-09-15 09:29:55", "2026-09-15 09:30:00", "2026-09-15 10:30:00", "2026-09-15 14:59:55",
               "2026-09-15 15:00:00", "2026-09-15 16:00:00", "2026-09-15 02:00:00")
    assert list(segment_of(times)) == ["overnight", "opening", "normal", "normal", "closing", "overnight", "overnight"]


def test_evening_bars_belong_to_the_next_session():
    days = session_date(et("2026-09-14 18:00:00", "2026-09-15 16:59:55"))
    assert list(days) == [dt.date(2026, 9, 15), dt.date(2026, 9, 15)]


def test_eastern_times_from_utc():
    assert eastern_times(pd.Series(pd.to_datetime(["2026-09-15 13:30:00"]))).dt.hour.iloc[0] == 9


def test_day_needs_every_file(tmp_path):
    trades, quotes = tmp_path / "daily", tmp_path / "quotes"
    trades.mkdir(), quotes.mkdir()
    required = ((str(trades), "{day}_ES.txt"), (str(quotes), "{day}_ES_BID.txt"), (str(trades), "{day}_TLT.txt"))
    for name in ("2026-09-15_ES.txt", "2026-09-15_TLT.txt", "2026-09-16_ES.txt", "2026-09-16_TLT.txt"):
        (trades / name).write_text("x" * 2000)
    (quotes / "2026-09-15_ES_BID.txt").write_text("x" * 2000)
    (quotes / "2026-09-16_ES_BID.txt").write_text("header only")
    with mock.patch.object(segments, "REQUIRED_FILES", required):
        assert days_with_all_data({dt.date(2026, 9, 15), dt.date(2026, 9, 16)}) == {dt.date(2026, 9, 15)}
