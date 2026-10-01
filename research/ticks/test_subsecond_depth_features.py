"""Sub-second windows make the look-ahead risk sharper, not milder.

A window that ends one microsecond late reads the very event it is meant to anticipate,
and at 50 ms that mistake is invisible in a plot. Both directions are tested: nothing
after the placement second may be read, and a window must actually see what happened
inside it.
"""
import numpy as np
import pandas as pd
import pytest

from subsecond_depth_features import (SUBSECOND_FEATURE_NAMES, SUBSECOND_WINDOWS_MS,
                                      at_placement_subsecond, state_at, subsecond_frame)

BUY, SELL = 1, -1
COLUMNS = ["bid_touch", "bid_total", "ask_touch", "ask_total"]


def replayed(rows):
    """rows: (time, bid_touch, bid_total, ask_touch, ask_total)."""
    times = [row[0] for row in rows]
    return pd.DataFrame([row[1:] for row in rows], index=times, columns=COLUMNS, dtype=float)


class TestStateAt:
    def test_it_takes_the_last_update_at_or_before(self):
        book = replayed([(100.0, 10, 100, 12, 120), (100.5, 20, 200, 22, 220)])
        assert state_at(book, np.array([100.4]))[0, 0] == 10

    def test_an_update_exactly_on_the_moment_counts(self):
        book = replayed([(100.0, 10, 100, 12, 120), (100.5, 20, 200, 22, 220)])
        assert state_at(book, np.array([100.5]))[0, 0] == 20

    def test_a_moment_before_any_update_clamps_to_the_first(self):
        book = replayed([(100.0, 10, 100, 12, 120)])
        assert state_at(book, np.array([50.0]))[0, 0] == 10


class TestDrain:
    def test_size_leaving_the_touch_drains_positive(self):
        book = replayed([(99.0, 100, 500, 12, 120), (99.99, 25, 400, 12, 120)])
        frame = subsecond_frame(book, np.array([100]))
        assert frame["bid_touch_drain_1000ms"].iloc[0] == pytest.approx(0.75)

    def test_size_arriving_drains_negative(self):
        book = replayed([(99.0, 50, 500, 12, 120), (99.99, 100, 500, 12, 120)])
        assert subsecond_frame(book, np.array([100]))["bid_touch_drain_1000ms"].iloc[0] < 0

    def test_a_still_book_does_not_drain(self):
        book = replayed([(99.0, 50, 500, 12, 120), (99.99, 50, 500, 12, 120)])
        assert subsecond_frame(book, np.array([100]))["bid_touch_drain_50ms"].iloc[0] == 0.0

    def test_a_short_window_sees_only_what_happened_inside_it(self):
        """Size that left 500 ms ago is in the 1000 ms window and not the 50 ms one."""
        book = replayed([(99.0, 100, 500, 12, 120), (99.5, 20, 500, 12, 120),
                         (99.99, 20, 500, 12, 120)])
        frame = subsecond_frame(book, np.array([100]))
        assert frame["bid_touch_drain_50ms"].iloc[0] == pytest.approx(0.0)
        assert frame["bid_touch_drain_1000ms"].iloc[0] == pytest.approx(0.8)

    def test_a_fifty_millisecond_move_is_visible(self):
        book = replayed([(99.9, 100, 500, 12, 120), (99.98, 10, 500, 12, 120)])
        assert subsecond_frame(book, np.array([100]))["bid_touch_drain_50ms"].iloc[0] > 0

    def test_an_empty_level_does_not_divide_by_zero(self):
        book = replayed([(99.0, 0, 0, 12, 120), (99.99, 0, 0, 12, 120)])
        assert np.isfinite(subsecond_frame(book, np.array([100]))["bid_touch_drain_50ms"].iloc[0])


class TestUpdateCounts:
    def test_updates_inside_the_window_are_counted(self):
        book = replayed([(99.96, 1, 1, 1, 1), (99.97, 1, 1, 1, 1), (99.98, 1, 1, 1, 1)])
        assert subsecond_frame(book, np.array([100]))["updates_50ms"].iloc[0] == 3

    def test_updates_outside_the_window_are_not(self):
        book = replayed([(99.0, 1, 1, 1, 1), (99.98, 1, 1, 1, 1)])
        assert subsecond_frame(book, np.array([100]))["updates_50ms"].iloc[0] == 1


class TestSigning:
    def frame(self):
        book = replayed([(99.0, 100, 500, 50, 300), (99.99, 25, 400, 50, 300)])
        return subsecond_frame(book, np.array([100]))

    def test_a_buy_reads_the_bid_as_its_own(self):
        values = at_placement_subsecond(self.frame(), 100, BUY)
        assert values["ss_own_touch_drain_1000ms"] == pytest.approx(0.75)

    def test_a_sell_reads_the_ask_as_its_own(self):
        values = at_placement_subsecond(self.frame(), 100, SELL)
        assert values["ss_own_touch_drain_1000ms"] == pytest.approx(0.0)

    def test_every_declared_feature_is_produced(self):
        assert sorted(at_placement_subsecond(self.frame(), 100, BUY)) == sorted(SUBSECOND_FEATURE_NAMES)

    def test_a_second_with_no_row_gives_nothing(self):
        assert at_placement_subsecond(self.frame(), 999, BUY) is None


class TestNoLookAhead:
    def test_nothing_after_the_placement_second_is_read(self):
        rows = [(99.0, 100, 500, 12, 120), (99.5, 90, 490, 12, 120), (99.99, 80, 480, 12, 120)]
        quiet = subsecond_frame(replayed(rows), np.array([100]))
        shocked = subsecond_frame(replayed(rows + [(100.01, 1, 1, 999, 999)]), np.array([100]))
        assert at_placement_subsecond(quiet, 100, BUY) == at_placement_subsecond(shocked, 100, BUY)

    def test_an_update_one_microsecond_later_is_excluded(self):
        rows = [(99.0, 100, 500, 12, 120), (99.99, 80, 480, 12, 120)]
        base = subsecond_frame(replayed(rows), np.array([100]))
        late = subsecond_frame(replayed(rows + [(100.000001, 1, 1, 1, 1)]), np.array([100]))
        assert base["bid_touch_drain_50ms"].iloc[0] == late["bid_touch_drain_50ms"].iloc[0]
