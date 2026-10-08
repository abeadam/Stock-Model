"""Tests for placement-time features.

The important property is that nothing reads the future: every value at second t must be
computable from seconds <= t. A feature that peeks would make the toxicity classifier
look predictive for the wrong reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fill_features import FILL_FEATURES, at_placement, session_frame

START = 1_790_343_000


def grid(seconds=60, bid=99.875, ask=100.125, bid_size=40, ask_size=60) -> pd.DataFrame:
    index = pd.Index([START + s for s in range(seconds)], name="second")
    return pd.DataFrame({
        "Bid": bid, "Ask": ask, "BidSize": bid_size, "AskSize": ask_size,
        "mid": (bid + ask) / 2, "signed": 0.0, "volume": 10.0, "quote_is_fresh": True,
    }, index=index)


def test_a_buy_sees_the_bid_as_its_own_side():
    values = at_placement(session_frame(grid()), START + 30, side=1)
    assert values["own_size"] == 40 and values["other_size"] == 60


def test_a_sell_sees_the_ask_as_its_own_side():
    values = at_placement(session_frame(grid()), START + 30, side=-1)
    assert values["own_size"] == 60 and values["other_size"] == 40


def test_book_imbalance_is_positive_when_the_book_favours_the_order():
    """More resting on the far side than ours means the book leans our way."""
    frame = session_frame(grid(bid_size=20, ask_size=80))
    assert at_placement(frame, START + 30, side=-1)["book_imbalance"] > 0
    assert at_placement(frame, START + 30, side=1)["book_imbalance"] < 0


def test_book_imbalance_is_zero_on_a_balanced_book():
    frame = session_frame(grid(bid_size=50, ask_size=50))
    assert at_placement(frame, START + 30, side=1)["book_imbalance"] == pytest.approx(0.0)


def test_signed_flow_is_mirrored_for_the_two_sides():
    data = grid()
    data["signed"] = 5.0
    frame = session_frame(data)
    buy = at_placement(frame, START + 30, side=1)["signed_flow_5s"]
    sell = at_placement(frame, START + 30, side=-1)["signed_flow_5s"]
    assert buy == pytest.approx(-sell) and buy > 0


def test_spread_is_measured_in_ticks():
    frame = session_frame(grid(bid=99.75, ask=100.25))
    assert at_placement(frame, START + 30, side=1)["spread_ticks"] == pytest.approx(2.0)


def test_own_depletion_is_negative_when_our_side_is_being_eaten():
    data = grid()
    data.loc[data.index >= START + 20, "BidSize"] = 10
    frame = session_frame(data)
    # The drop lands ON second 20, so the one-second difference is negative there.
    assert at_placement(frame, START + 20, side=1)["own_depletion_1s"] < 0


def test_depletion_reads_our_own_side_only():
    data = grid()
    data.loc[data.index >= START + 20, "AskSize"] = 5
    frame = session_frame(data)
    assert at_placement(frame, START + 20, side=1)["own_depletion_1s"] == 0
    assert at_placement(frame, START + 20, side=-1)["own_depletion_1s"] < 0


def test_volatility_rises_on_a_moving_mid():
    quiet = session_frame(grid())
    noisy_grid = grid()
    noisy_grid["mid"] = 100 + np.random.default_rng(0).normal(0, 0.5, len(noisy_grid))
    noisy = session_frame(noisy_grid)
    assert (at_placement(noisy, START + 50, 1)["mid_volatility_30s"]
            > at_placement(quiet, START + 50, 1)["mid_volatility_30s"])


def test_quote_activity_counts_fresh_seconds_in_the_window():
    data = grid()
    data["quote_is_fresh"] = False
    data.loc[data.index[:3], "quote_is_fresh"] = True
    assert session_frame(data)["quote_updates_5s"].iloc[2] == 3


def test_no_feature_reads_the_future():
    """Change a later second and nothing at an earlier second may move."""
    base = grid()
    changed = base.copy()
    changed.loc[changed.index >= START + 40, ["BidSize", "AskSize", "mid", "signed", "volume"]] = 999
    before = at_placement(session_frame(base), START + 30, side=1)
    after = at_placement(session_frame(changed), START + 30, side=1)
    assert before == after


def test_every_named_feature_is_produced():
    values = at_placement(session_frame(grid()), START + 40, side=1)
    assert set(FILL_FEATURES) == set(values)


def test_a_second_outside_the_session_has_no_row():
    assert at_placement(session_frame(grid()), START + 10_000, side=1) is None
