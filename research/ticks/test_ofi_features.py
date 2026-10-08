"""e_n is checked against the cases Cont, Kukanov and Stoikov state in words.

From the paper: "if q^B increases but P^B remains the same, we assign e_n = q^B_n -
q^B_{n-1}, representing the size that was added at the bid. If q^B decreases, we also
assign e_n = q^B_n - q^B_{n-1}... If P^B increases, we let e_n = q^B_n, representing the
size of a price-improving limit order. If P^B decreases, we let e_n = q^B_{n-1},
representing the size that was removed... The same classification is done for events on
the ask side, with signs reversed."

Getting a sign wrong here would produce a feature that is anti-predictive in a way no
AUC inspection would reveal, so each case is pinned separately.
"""
import numpy as np
import pandas as pd
import pytest

from ofi_features import (BEST_LEVEL_OFI_NAMES, LEVELS, MULTI_LEVEL_OFI_NAMES,
                          OFI_WINDOWS_SECONDS, at_placement_ofi, cumulative,
                          event_contributions)

BUY, SELL = 1, -1


def two_observations(bid, ask):
    """bid/ask: ((price_before, size_before), (price_after, size_after)) for level 1."""
    def column(values, filler):
        out = np.full((2, LEVELS), filler)
        out[0, 0], out[1, 0] = values[0], values[1]
        return out
    bp = column((bid[0][0], bid[1][0]), np.nan)
    bq = column((bid[0][1], bid[1][1]), 0.0)
    ap = column((ask[0][0], ask[1][0]), np.nan)
    aq = column((ask[0][1], ask[1][1]), 0.0)
    return event_contributions(bp, bq, ap, aq)[0, 0]


STEADY_ASK = ((100.25, 50), (100.25, 50))
STEADY_BID = ((100.00, 50), (100.00, 50))


class TestBidSide:
    def test_size_added_at_the_bid_is_positive(self):
        assert two_observations(((100.0, 100), (100.0, 107)), STEADY_ASK) == 7

    def test_size_removed_from_the_bid_is_negative(self):
        assert two_observations(((100.0, 100), (100.0, 85)), STEADY_ASK) == -15

    def test_a_price_improving_bid_contributes_its_whole_size(self):
        assert two_observations(((100.0, 100), (100.25, 30)), STEADY_ASK) == 30

    def test_a_falling_bid_removes_the_whole_previous_size(self):
        assert two_observations(((100.0, 100), (99.75, 30)), STEADY_ASK) == -100


class TestAskSide:
    def test_size_added_at_the_ask_is_negative(self):
        assert two_observations(STEADY_BID, ((100.25, 100), (100.25, 107))) == -7

    def test_size_removed_from_the_ask_is_positive(self):
        assert two_observations(STEADY_BID, ((100.25, 100), (100.25, 85))) == 15

    def test_a_price_improving_ask_contributes_its_whole_size_negatively(self):
        assert two_observations(STEADY_BID, ((100.5, 100), (100.25, 30))) == -30

    def test_a_rising_ask_releases_the_whole_previous_size(self):
        assert two_observations(STEADY_BID, ((100.25, 100), (100.5, 30))) == 100


class TestSymmetry:
    def test_a_still_book_contributes_nothing(self):
        assert two_observations(STEADY_BID, STEADY_ASK) == 0

    def test_buying_and_selling_pressure_cancel(self):
        both = two_observations(((100.0, 100), (100.0, 110)), ((100.25, 100), (100.25, 110)))
        assert both == 0

    def test_an_absent_level_contributes_nothing(self):
        bp = np.array([[np.nan] * LEVELS, [np.nan] * LEVELS])
        bq = np.zeros((2, LEVELS))
        ap = np.array([[100.25] * LEVELS, [100.25] * LEVELS])
        aq = np.full((2, LEVELS), 50.0)
        assert event_contributions(bp, bq, ap, aq)[0, 0] == 0.0


def synthetic_book(events):
    """events: list of (time, bid_price, bid_size, ask_price, ask_size) for level 1."""
    rows = len(events)
    times = np.array([event[0] for event in events], dtype=float)
    bp = np.full((rows, LEVELS), np.nan); bq = np.zeros((rows, LEVELS))
    ap = np.full((rows, LEVELS), np.nan); aq = np.zeros((rows, LEVELS))
    for row, (_, bid_price, bid_size, ask_price, ask_size) in enumerate(events):
        bp[row, 0], bq[row, 0] = bid_price, bid_size
        ap[row, 0], aq[row, 0] = ask_price, ask_size
    return times, bp, bq, ap, aq


class TestAggregation:
    def test_running_sums_recover_a_window(self):
        times, bp, bq, ap, aq = synthetic_book(
            [(0.0, 100.0, 100, 100.25, 100), (1.0, 100.0, 90, 100.25, 100),
             (2.0, 100.0, 80, 100.25, 100)])
        stamps, ofi, _depth, _count = cumulative(times, event_contributions(bp, bq, ap, aq), bq, aq)
        assert ofi[-1, 0] == -20                      # ten left, then ten more

    def test_depth_runs_alongside_the_flow(self):
        times, bp, bq, ap, aq = synthetic_book(
            [(0.0, 100.0, 100, 100.25, 100), (1.0, 100.0, 100, 100.25, 100)])
        _stamps, _ofi, depth, count = cumulative(times, event_contributions(bp, bq, ap, aq), bq, aq)
        assert depth[-1] / max(count[-1], 1) == pytest.approx(200 / (2 * LEVELS))


class TestSigning:
    def frame(self):
        return pd.DataFrame({f"ofi_L{level}_{window}s": [4.0]
                             for window in OFI_WINDOWS_SECONDS for level in range(1, LEVELS + 1)}
                            | {f"ofi_integrated_{window}s": [20.0] for window in OFI_WINDOWS_SECONDS}
                            | {f"ofi_depth_{window}s": [300.0] for window in OFI_WINDOWS_SECONDS},
                            index=[100])

    def test_a_buy_sees_positive_flow_as_favourable(self):
        assert at_placement_ofi(self.frame(), 100, BUY)["ofi_own_L1_1s"] == 4.0

    def test_a_sell_sees_the_same_flow_as_adverse(self):
        assert at_placement_ofi(self.frame(), 100, SELL)["ofi_own_L1_1s"] == -4.0

    def test_depth_is_not_signed(self):
        buy = at_placement_ofi(self.frame(), 100, BUY)["ofi_depth_1s"]
        sell = at_placement_ofi(self.frame(), 100, SELL)["ofi_depth_1s"]
        assert buy == sell == 300.0

    def test_a_second_with_no_row_gives_nothing(self):
        assert at_placement_ofi(self.frame(), 999, BUY) is None

    def test_the_declared_names_are_all_produced(self):
        values = at_placement_ofi(self.frame(), 100, BUY)
        assert set(BEST_LEVEL_OFI_NAMES) <= set(values)
        assert set(MULTI_LEVEL_OFI_NAMES) <= set(values)


class TestNoLookAhead:
    """A window ends AT the placement second. One microsecond late reads the event the
    feature is supposed to anticipate, and nothing downstream would reveal it."""

    def book(self, extra=()):
        events = [(99.0, 100.0, 100, 100.25, 100), (99.5, 100.0, 90, 100.25, 100),
                  (99.99, 100.0, 80, 100.25, 100), *extra]
        return synthetic_book(events)

    def flows(self, monkeypatch, extra=()):
        import ofi_features
        monkeypatch.setattr(ofi_features, "replay_levels", lambda day: self.book(extra))
        return ofi_features.window_ofi("ignored", np.array([100]))

    def test_an_event_after_the_second_is_excluded(self, monkeypatch):
        quiet = self.flows(monkeypatch)
        loud = self.flows(monkeypatch, extra=((100.000001, 100.0, 1, 100.25, 999),))
        assert quiet["ofi_L1_1s"].iloc[0] == loud["ofi_L1_1s"].iloc[0]

    def test_events_inside_the_window_are_included(self, monkeypatch):
        frame = self.flows(monkeypatch)
        assert frame["ofi_L1_1s"].iloc[0] != 0.0

    def test_a_longer_window_reaches_further_back(self, monkeypatch):
        frame = self.flows(monkeypatch)
        assert abs(frame["ofi_L1_30s"].iloc[0]) >= abs(frame["ofi_L1_1s"].iloc[0])

    def test_depth_normalisation_divides_the_flow(self, monkeypatch):
        frame = self.flows(monkeypatch)
        # 20 contracts left the bid over the window; depth averages (bid+ask)/(2*LEVELS).
        assert frame["ofi_L1_1s"].iloc[0] == pytest.approx(-20.0 / frame["ofi_depth_1s"].iloc[0])

    def test_depth_is_positive(self, monkeypatch):
        assert self.flows(monkeypatch)["ofi_depth_1s"].iloc[0] > 0
