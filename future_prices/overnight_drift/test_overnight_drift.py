"""Tests for session_prices.py, contract_rolls.py and overnight_backtest.py.

Run from this directory:  python -m unittest test_overnight_drift
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract_rolls import roll_gaps, third_friday  # noqa: E402
from overnight_backtest import POINT_VALUE, ROUND_TRIP_COST, overnight_trades, summarize  # noqa: E402
from session_prices import BAR_SECONDS, closes_by_bar_end, price_at, trading_weekdays  # noqa: E402

EASTERN = ZoneInfo("America/New_York")


def bars(prices_by_bar_end: dict[str, float]) -> pd.DataFrame:
    """5-second bars (keyed by start epoch, as in the daily files) ending at the given Eastern times."""
    rows = [{"Date": int(datetime.fromisoformat(end).replace(tzinfo=EASTERN).timestamp()) - BAR_SECONDS,
             "Close": price} for end, price in prices_by_bar_end.items()]
    return pd.DataFrame(rows)


def closes(prices_by_bar_end: dict[str, float]) -> pd.Series:
    return closes_by_bar_end(bars(prices_by_bar_end))


def session(day: str, open_price: float, close_price: float, halt_price: float | None = None) -> dict:
    """Bars at 09:30, 16:00 and 17:00 on `day`."""
    halt = close_price if halt_price is None else halt_price
    return {f"{day} 09:30:00": open_price, f"{day} 16:00:00": close_price, f"{day} 17:00:00": halt}


class PriceAtTest(unittest.TestCase):
    def test_uses_the_bar_ending_at_the_moment(self):
        series = closes({"2026-03-02 15:59:55": 99.0, "2026-03-02 16:00:00": 100.0})
        moment = pd.DatetimeIndex([pd.Timestamp("2026-03-02 16:00:00")])
        self.assertEqual(price_at(series, moment)[0], 100.0)

    def test_stale_price_is_missing(self):
        series = closes({"2026-03-02 13:15:00": 100.0})            # early close
        moment = pd.DatetimeIndex([pd.Timestamp("2026-03-02 16:00:00")])
        self.assertTrue(np.isnan(price_at(series, moment)[0]))


class ContractRollTest(unittest.TestCase):
    def test_third_friday(self):
        self.assertEqual(third_friday(pd.Timestamp("2025-09-18")), pd.Timestamp("2025-09-19"))
        self.assertEqual(third_friday(pd.Timestamp("2026-06-01")), pd.Timestamp("2026-06-19"))

    def test_jump_in_roll_week_is_a_roll(self):
        series = closes({"2025-09-18 17:00:00": 6600.0, "2025-09-18 18:00:05": 6660.25})
        gaps = roll_gaps(series, trading_weekdays(series))
        self.assertAlmostEqual(gaps.loc[pd.Timestamp("2025-09-18")], 60.25)

    def test_only_the_largest_jump_in_a_roll_week_is_a_roll(self):
        # 2026-06-12 weekend: a real +55 point move; 2026-06-17: the switch to the September contract.
        series = closes({"2026-06-12 17:00:00": 7433.0, "2026-06-14 18:00:05": 7488.25,
                         "2026-06-17 17:00:00": 7423.0, "2026-06-17 18:00:05": 7496.5})
        gaps = roll_gaps(series, trading_weekdays(series))
        self.assertEqual(gaps.loc[pd.Timestamp("2026-06-12")], 0.0)
        self.assertAlmostEqual(gaps.loc[pd.Timestamp("2026-06-17")], 73.5)

    def test_jump_outside_roll_week_is_a_real_move(self):
        series = closes({"2026-04-07 17:00:00": 6600.0, "2026-04-07 18:00:05": 6652.75})
        gaps = roll_gaps(series, trading_weekdays(series))
        self.assertEqual(gaps.loc[pd.Timestamp("2026-04-07")], 0.0)


class OvernightTradesTest(unittest.TestCase):
    def test_weeknight_trade(self):
        trades = overnight_trades(closes({**session("2026-03-02", 99, 100), **session("2026-03-03", 103, 104)}))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades["points"].iloc[0], 3.0)
        self.assertEqual(trades["net"].iloc[0], 3.0 * POINT_VALUE - ROUND_TRIP_COST)

    def test_friday_exits_monday(self):
        trades = overnight_trades(closes({**session("2026-03-06", 99, 100), **session("2026-03-09", 98, 97)}))
        self.assertEqual(trades["exit_day"].iloc[0], pd.Timestamp("2026-03-09"))
        self.assertEqual(trades["points"].iloc[0], -2.0)

    def test_holiday_is_skipped_to_next_open(self):
        prices = {**session("2025-11-26", 99, 100), **session("2025-11-28", 101, 101)}   # Thanksgiving: no bars
        trades = overnight_trades(closes(prices))
        self.assertEqual(trades["exit_day"].iloc[0], pd.Timestamp("2025-11-28"))

    def test_roll_jump_is_removed(self):
        prices = {**session("2025-09-18", 6599, 6600, halt_price=6600), "2025-09-18 18:00:05": 6660.0,
                  **session("2025-09-19", 6662, 6662)}
        trades = overnight_trades(closes(prices))
        self.assertAlmostEqual(trades.loc[pd.Timestamp("2025-09-18"), "points"], 2.0)

    def test_no_trade_without_a_16_00_price(self):
        prices = {"2025-11-28 09:30:00": 100.0, "2025-11-28 13:15:00": 101.0, **session("2025-12-01", 102, 103)}
        trades = overnight_trades(closes(prices))
        self.assertNotIn(pd.Timestamp("2025-11-28"), trades.index)


class SummarizeTest(unittest.TestCase):
    def test_drawdown_and_hit_rate(self):
        summary = summarize(pd.Series([100.0, -50.0, -75.0, 200.0]))
        self.assertEqual(summary.trades, 4)
        self.assertEqual(summary.net, 175.0)
        self.assertEqual(summary.hit_rate, 0.5)
        self.assertEqual(summary.max_drawdown, -125.0)


if __name__ == "__main__":
    unittest.main()
