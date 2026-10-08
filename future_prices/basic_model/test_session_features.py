"""Tests for session_features.py and session_history.py.

Run from this directory:  python -m unittest test_session_features
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")  # futures_price imports pyplot

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import es_features  # noqa: E402
import futures_price  # noqa: E402
import session_history  # noqa: E402
from market_clock import EASTERN_TIMEZONE  # noqa: E402
from session_features import (  # noqa: E402
    PRIOR_SESSIONS_FOR_RANGE,
    SESSION_FEATURE_NAMES,
    average_prior_range,
    session_features_for_bars,
    session_ranges,
    session_start_times,
)

EASTERN = ZoneInfo(EASTERN_TIMEZONE)
BAR_SECONDS = 5


def epoch_at_eastern(year, month, day, hour, minute=0, second=0) -> int:
    return int(datetime(year, month, day, hour, minute, second, tzinfo=EASTERN).timestamp())


def bars(start_epoch: int, prices: list[tuple], spacing_seconds: int = BAR_SECONDS) -> pd.DataFrame:
    """Consecutive bars from (open, high, low, close, volume) tuples."""
    return pd.DataFrame({
        "Date": [start_epoch + spacing_seconds * i for i in range(len(prices))],
        "Open": [p[0] for p in prices], "High": [p[1] for p in prices],
        "Low": [p[2] for p in prices], "Close": [p[3] for p in prices],
        "Volume": [p[4] for p in prices],
    })


def eastern_times(frame: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(frame["Date"], unit="s", utc=True)
            .dt.tz_convert(EASTERN_TIMEZONE).dt.tz_localize(None))


def features_of(frame: pd.DataFrame, history: pd.Series | None = None) -> dict[str, np.ndarray]:
    times = eastern_times(frame)
    if history is None:
        history = session_ranges(times, frame["High"], frame["Low"])
    return session_features_for_bars(times, frame, history)


def flat_history(before: datetime, count: int, session_range: float) -> pd.Series:
    """`count` complete sessions, each with the same range, ending the day before `before`."""
    starts = [pd.Timestamp(before) - pd.Timedelta(days=day) for day in range(count, 0, -1)]
    return pd.Series(session_range, index=pd.DatetimeIndex(starts), name="range")


# One session opening Tuesday 3 June 2025, 18:00 ET: values worked out by hand.
TUESDAY_OPEN = epoch_at_eastern(2025, 6, 3, 18)
TUESDAY_OPEN_NAIVE = datetime(2025, 6, 3, 18)
THREE_BARS = [(100, 101, 99, 100, 10), (100, 103, 100, 102, 30), (102, 102, 98, 98, 40)]


class HandComputedValuesTest(unittest.TestCase):
    def setUp(self):
        history = flat_history(TUESDAY_OPEN_NAIVE, PRIOR_SESSIONS_FOR_RANGE, session_range=10.0)
        self.features = features_of(bars(TUESDAY_OPEN, THREE_BARS), history)

    def assert_values(self, name, expected):
        np.testing.assert_allclose(self.features[name], expected, rtol=1e-9, equal_nan=True)

    def test_vwap_distance(self):
        # VWAP: 100, then (1000 + 101.667*30)/40 = 101.25, then 8023.33/80 = 100.2917
        self.assert_values("session_vwap_dist",
                           [0.0, (102 - 101.25) / 102, (98 - 8023.333333333333 / 80) / 98])

    def test_share_of_bars_above_vwap(self):
        self.assert_values("session_vwap_above_frac", [0.0, 0.5, 1 / 3])

    def test_return_since_the_session_open(self):
        self.assert_values("session_return_pct", [0.0, 2.0, -2.0])

    def test_move_in_units_of_the_prior_average_range(self):
        # Moves of 0, +2 and -2 points against a 10-point average range.
        self.assert_values("session_move_vs_prior_range", [0.0, 0.2, -0.2])

    def test_position_in_the_range_so_far(self):
        self.assert_values("session_range_position", [0.5, 0.75, 0.0])


class EfficiencyRatioTest(unittest.TestCase):
    """Closes 30 seconds apart, so the path runs through 1-minute closes."""

    def test_path_uses_completed_minutes_then_the_current_close(self):
        closes = [100, 102, 101, 105, 104]           # 18:00:00, :30, 18:01:00, :30, 18:02:00
        frame = bars(TUESDAY_OPEN, [(c, c, c, c, 10) for c in closes], spacing_seconds=30)
        ratio = features_of(frame, flat_history(TUESDAY_OPEN_NAIVE, 20, 10.0))["session_efficiency_ratio"]
        # Minute closes 102 (18:00) and 105 (18:01). Paths: 0, 2, 2+1, 2+3, 5+1.
        np.testing.assert_allclose(ratio, [np.nan, 1.0, 1 / 3, 1.0, 4 / 6], equal_nan=True)

    def test_bounce_within_a_minute_does_not_lengthen_the_path_of_later_minutes(self):
        bounce = [100, 101, 100, 101, 100, 101, 102]  # 5-second bars in the first minute
        then = [103] * 12                             # the next minute, steady
        frame = bars(TUESDAY_OPEN, [(c, c, c, c, 10) for c in bounce + then])
        ratio = features_of(frame, flat_history(TUESDAY_OPEN_NAIVE, 20, 10.0))["session_efficiency_ratio"]
        self.assertAlmostEqual(ratio[-1], 1.0)        # 100 -> 102 -> 103, a straight line


class PriorRangeTest(unittest.TestCase):
    def test_fewer_than_twenty_prior_sessions_leaves_it_blank(self):
        history = flat_history(TUESDAY_OPEN_NAIVE, PRIOR_SESSIONS_FOR_RANGE - 1, session_range=10.0)
        moves = features_of(bars(TUESDAY_OPEN, THREE_BARS), history)["session_move_vs_prior_range"]
        self.assertTrue(np.isnan(moves).all())

    def test_average_uses_only_the_twenty_sessions_before_each_session(self):
        starts = pd.date_range("2025-05-01 18:00", periods=25, freq="D")
        history = pd.Series(np.arange(25, dtype=float), index=starts, name="range")
        averages = average_prior_range(pd.DatetimeIndex([starts[22]]), history)
        self.assertAlmostEqual(averages.iloc[0], np.mean(np.arange(2, 22)))   # sessions 2..21

    def test_sessions_not_observed_from_their_open_are_not_history(self):
        complete = bars(epoch_at_eastern(2025, 6, 2, 18), [(100, 110, 90, 100, 1)] * 3)
        late = bars(epoch_at_eastern(2025, 6, 3, 18, 10), [(100, 150, 50, 100, 1)] * 3)
        frame = pd.concat([complete, late], ignore_index=True)
        ranges = session_ranges(eastern_times(frame), frame["High"], frame["Low"])
        self.assertEqual(list(ranges.index), [pd.Timestamp("2025-06-02 18:00")])
        self.assertEqual(ranges.iloc[0], 20.0)

    def test_ranges_include_the_hour_after_the_stock_close(self):
        session = bars(epoch_at_eastern(2025, 6, 2, 18), [(100, 101, 99, 100, 1)] * 3)
        last_hour = bars(epoch_at_eastern(2025, 6, 3, 16, 30), [(100, 130, 70, 100, 1)])
        frame = pd.concat([session, last_hour], ignore_index=True)
        ranges = session_ranges(eastern_times(frame), frame["High"], frame["Low"])
        self.assertEqual(ranges.iloc[0], 60.0)


class NoLookaheadTest(unittest.TestCase):
    def test_later_bars_never_change_earlier_values(self):
        history = flat_history(TUESDAY_OPEN_NAIVE, 20, 10.0)
        prices = [(100 + i % 7, 101 + i % 7, 99 + i % 7, 100 + (i * 3) % 11, 5 + i) for i in range(40)]
        short = features_of(bars(TUESDAY_OPEN, prices), history)
        longer = features_of(bars(TUESDAY_OPEN, prices + [(98, 120, 90, 110, 500)] * 30), history)
        for name in SESSION_FEATURE_NAMES:
            np.testing.assert_array_equal(longer[name][:40], short[name], err_msg=name)


class SessionBoundaryTest(unittest.TestCase):
    def test_sessions_open_at_18_00_eastern(self):
        times = pd.Series(pd.to_datetime([
            "2025-06-02 10:00:00", "2025-06-03 17:59:55", "2025-06-03 18:00:00", "2025-06-06 16:59:55",
        ]))
        self.assertEqual(list(session_start_times(times)), list(pd.to_datetime([
            "2025-06-01 18:00:00", "2025-06-02 18:00:00", "2025-06-03 18:00:00", "2025-06-05 18:00:00",
        ])))

    def test_everything_resets_at_the_open(self):
        history = flat_history(TUESDAY_OPEN_NAIVE, 20, 10.0)
        before = bars(TUESDAY_OPEN - 2 * BAR_SECONDS, [(90, 95, 85, 94, 99)] * 2)
        both = pd.concat([before, bars(TUESDAY_OPEN, THREE_BARS)], ignore_index=True)
        features = features_of(both, history)
        alone = features_of(bars(TUESDAY_OPEN, THREE_BARS), history)
        for name in SESSION_FEATURE_NAMES:
            self.assertTrue(np.isnan(features[name][:2]).all(), name)   # not observed from its open
            np.testing.assert_array_equal(features[name][2:], alone[name], err_msg=name)

    def test_session_first_seen_after_the_tolerance_is_all_nan(self):
        late = bars(TUESDAY_OPEN + 10 * 60, THREE_BARS)
        for name, values in features_of(late, flat_history(TUESDAY_OPEN_NAIVE, 20, 10.0)).items():
            self.assertTrue(np.isnan(values).all(), name)


def many_sessions(first_open: datetime, count: int, bars_per_session: int, seed: int = 7) -> pd.DataFrame:
    """`count` weekday sessions of random-walk bars, each starting at its 18:00 ET open."""
    rng = np.random.default_rng(seed)
    sessions, day = [], first_open
    while len(sessions) < count:
        if day.weekday() not in (4, 5):   # no session opens Friday or Saturday evening
            closes = 6000 + np.cumsum(rng.normal(0, 1, bars_per_session))
            highs, lows = closes + rng.uniform(0, 1, bars_per_session), closes - rng.uniform(0, 1, bars_per_session)
            open_epoch = epoch_at_eastern(day.year, day.month, day.day, 18)
            sessions.append(bars(open_epoch, list(zip(closes, highs, lows, closes,
                                                      rng.integers(1, 50, bars_per_session))), 60))
        day += timedelta(days=1)
    return pd.concat(sessions, ignore_index=True)


class TrainingMatchesLiveTest(unittest.TestCase):
    """futures_price.py (training) and es_features.py (live) must agree exactly."""

    def test_same_values_through_both_paths_across_a_dst_change(self):
        frame = many_sessions(datetime(2025, 10, 16), count=PRIOR_SESSIONS_FOR_RANGE + 5, bars_per_session=120)
        training = futures_price.add_session_features(futures_price.calculate_trading_hours(frame))
        history = session_ranges(eastern_times(frame), frame["High"], frame["Low"])
        live_input = frame.drop(columns="Date").assign(date=frame["Date"].astype(str))
        with mock.patch.object(es_features, "prior_session_ranges", return_value=history):
            live = es_features._build_all_feature_arrays(live_input)

        for name in SESSION_FEATURE_NAMES:
            np.testing.assert_array_equal(live[name], training[name].to_numpy(), err_msg=name)
        moves = live["session_move_vs_prior_range"]
        self.assertTrue(np.isfinite(moves[-120:]).all())   # the last sessions have 20 prior ones


class SessionHistoryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        session_history._cache.update(checked_at=None, files_key=None, ranges=None)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_day_files(self, frame: pd.DataFrame) -> None:
        """One file per session, named by its trading day, like download_daily.py."""
        starts = session_start_times(eastern_times(frame))
        for start, day in frame.groupby(starts.values):
            (day.to_csv(self.data_dir / f"{(start + pd.Timedelta(hours=6)).date()}_ES.txt", index=False))

    def test_files_give_the_same_ranges_as_the_bars(self):
        frame = many_sessions(datetime(2025, 10, 16), count=24, bars_per_session=30)
        self.write_day_files(frame)
        expected = session_ranges(eastern_times(frame), frame["High"], frame["Low"])
        pd.testing.assert_series_equal(session_history.load_prior_session_ranges(self.data_dir), expected)

    def test_reads_further_back_past_incomplete_sessions(self):
        frame = many_sessions(datetime(2025, 10, 16), count=45, bars_per_session=30)
        self.write_day_files(frame)
        # Make the newest 30 files start late, so the first pass finds no complete sessions.
        for path in sorted(self.data_dir.glob("*_ES.txt"))[-30:]:
            day = pd.read_csv(path)
            day.iloc[len(day) // 2:].to_csv(path, index=False)
        ranges = session_history.load_prior_session_ranges(self.data_dir)
        self.assertEqual(len(ranges), 15)                   # every complete session, from two passes

    def test_unreadable_directory_gives_an_empty_history(self):
        with mock.patch.object(session_history, "load_prior_session_ranges", side_effect=OSError("gone")):
            self.write_day_files(many_sessions(datetime(2025, 10, 16), count=2, bars_per_session=5))
            self.assertTrue(session_history.prior_session_ranges(self.data_dir).empty)


if __name__ == "__main__":
    unittest.main()
