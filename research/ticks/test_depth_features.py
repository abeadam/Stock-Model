"""The leak test is the important one: nothing may read past the placement second.

A fill feature that peeks forward produces a model that looks excellent and is useless,
and that mistake has already been made once in this project -- tick_walk_forward trained
on sessions later than its test day and gained +0.0051 AUC from it.
"""
import numpy as np
import pandas as pd
import pytest

from depth_features import (DEPTH_FEATURE_NAMES, DRAIN_WINDOWS, LEVELS, at_placement_depth,
                            book_frame, side_sizes)

BUY, SELL = 1, -1


def depth_rows(rows):
    """rows: list of (second, bid_sizes, ask_sizes)."""
    records = []
    for second, bids, asks in rows:
        record = {"Date": second}
        for level in range(LEVELS):
            record[f"BidPrice{level}"] = 7749.0 - 0.25 * level
            record[f"BidSize{level}"] = bids[level] if level < len(bids) else 0
            record[f"AskPrice{level}"] = 7749.25 + 0.25 * level
            record[f"AskSize{level}"] = asks[level] if level < len(asks) else 0
        records.append(record)
    return pd.DataFrame(records).set_index("Date")


def flat_book(seconds, bid=(10, 20, 30, 40, 50), ask=(12, 22, 32, 42, 52)):
    return depth_rows([(second, bid, ask) for second in seconds])


class TestShape:
    def test_total_sums_every_level(self):
        shaped = book_frame(flat_book([1]))
        assert shaped["bid_total"].iloc[0] == 150

    def test_behind_excludes_the_touch(self):
        shaped = book_frame(flat_book([1]))
        assert shaped["bid_behind"].iloc[0] == 140

    def test_touch_share_is_the_touch_over_the_total(self):
        shaped = book_frame(flat_book([1]))
        assert shaped["bid_touch_share"].iloc[0] == pytest.approx(10 / 150)

    def test_a_lone_order_has_a_touch_share_of_one(self):
        shaped = book_frame(depth_rows([(1, (10, 0, 0, 0, 0), (12, 0, 0, 0, 0))]))
        assert shaped["bid_touch_share"].iloc[0] == pytest.approx(1.0)

    def test_an_empty_side_does_not_divide_by_zero(self):
        shaped = book_frame(depth_rows([(1, (0, 0, 0, 0, 0), (12, 22, 32, 42, 52))]))
        assert np.isfinite(shaped["bid_touch_share"].iloc[0])

    def test_absent_levels_count_as_zero(self):
        frame = flat_book([1])
        frame.loc[1, "BidSize4"] = np.nan
        assert side_sizes(frame, "Bid")[0, 4] == 0.0


class TestDrain:
    def test_a_shrinking_book_drains_positive(self):
        frame = depth_rows([(1, (50, 50, 50, 50, 50), (12, 22, 32, 42, 52)),
                            (2, (25, 25, 25, 25, 25), (12, 22, 32, 42, 52))])
        assert book_frame(frame)["bid_drain_1s"].iloc[1] == pytest.approx(0.5)

    def test_a_growing_book_drains_negative(self):
        frame = depth_rows([(1, (10, 10, 10, 10, 10), (12, 22, 32, 42, 52)),
                            (2, (20, 20, 20, 20, 20), (12, 22, 32, 42, 52))])
        assert book_frame(frame)["bid_drain_1s"].iloc[1] < 0

    def test_a_steady_book_does_not_drain(self):
        assert book_frame(flat_book([1, 2, 3]))["bid_drain_1s"].iloc[2] == pytest.approx(0.0)

    def test_drain_is_blank_before_its_window_exists(self):
        assert np.isnan(book_frame(flat_book([1, 2]))["bid_drain_30s"].iloc[1])


class TestSigning:
    def test_a_buy_reads_the_bid_as_its_own(self):
        shaped = book_frame(flat_book([1]))
        assert at_placement_depth(shaped, 1, BUY)["depth_own_total"] == 150

    def test_a_sell_reads_the_ask_as_its_own(self):
        shaped = book_frame(flat_book([1]))
        assert at_placement_depth(shaped, 1, SELL)["depth_own_total"] == 160

    def test_imbalance_flips_sign_with_the_side(self):
        shaped = book_frame(flat_book([1]))
        buy = at_placement_depth(shaped, 1, BUY)["depth_imbalance"]
        sell = at_placement_depth(shaped, 1, SELL)["depth_imbalance"]
        assert buy == pytest.approx(-sell)

    def test_imbalance_is_positive_when_the_book_leans_the_order_s_way(self):
        frame = depth_rows([(1, (90, 90, 90, 90, 90), (1, 1, 1, 1, 1))])
        assert at_placement_depth(book_frame(frame), 1, BUY)["depth_imbalance"] > 0

    def test_a_second_with_no_book_gives_nothing(self):
        assert at_placement_depth(book_frame(flat_book([1])), 99, BUY) is None

    def test_every_declared_feature_is_produced(self):
        values = at_placement_depth(book_frame(flat_book(range(1, 40))), 39, BUY)
        assert sorted(values) == sorted(DEPTH_FEATURE_NAMES)


class TestNoLookAhead:
    def test_no_feature_reads_the_future(self):
        """Truncating everything after the placement second must change nothing."""
        seconds = list(range(1, 60))
        full = book_frame(flat_book(seconds))
        truncated = book_frame(flat_book([s for s in seconds if s <= 40]))
        assert (at_placement_depth(full, 40, BUY)
                == at_placement_depth(truncated, 40, BUY))

    def test_a_later_shock_does_not_reach_backwards(self):
        rows = [(second, (10, 20, 30, 40, 50), (12, 22, 32, 42, 52)) for second in range(1, 50)]
        quiet = book_frame(depth_rows(rows))
        rows[45] = (rows[45][0], (9999, 1, 1, 1, 1), (1, 1, 1, 1, 1))
        shocked = book_frame(depth_rows(rows))
        assert at_placement_depth(quiet, 40, BUY) == at_placement_depth(shocked, 40, BUY)
