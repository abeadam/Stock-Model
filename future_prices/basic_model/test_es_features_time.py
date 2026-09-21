"""Live time features must equal the ones the models were trained on.

futures_price.py builds the training data and converts every bar from UTC to
US/Eastern before deriving time features. es_features.py builds the live
features. These tests run both on the same timestamps and require identical
output, so the two cannot drift apart again unnoticed.

Run from this directory:  python -m unittest test_es_features_time
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # futures_price imports pyplot

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import es_features  # noqa: E402
import futures_price  # noqa: E402

TIME_FEATURES = (
    "Hours_From_Formal_Trading", "Hours_From_Overnight_Trading",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
)


def epoch(year, month, day, hour, minute, second) -> int:
    return int(datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp())


# UTC instants chosen at the edges where a wrong clock shows up.
EDGE_CASES_UTC = [
    epoch(2025, 5, 29, 9, 13, 5),     # 05:13:05 EDT, UTC-4
    epoch(2025, 1, 15, 14, 30, 59),   # 09:30:59 EST, UTC-5
    epoch(2025, 6, 3, 21, 59, 59),    # 17:59:59 EDT, just before the Globex open
    epoch(2025, 6, 3, 22, 0, 0),      # 18:00:00 EDT, the Globex open
    epoch(2025, 6, 4, 2, 30, 0),      # 22:30 EDT Tue, already Wed in UTC
    epoch(2025, 2, 1, 2, 0, 0),       # 21:00 EST Jan 31, already Feb in UTC
    epoch(2025, 11, 3, 4, 59, 0),     # 23:59 EST, day after DST ends
    epoch(2025, 3, 10, 13, 0, 1),     # 09:00:01 EDT, day after DST starts
]


def training_time_features(epochs: list[int]) -> pd.DataFrame:
    """The time features exactly as futures_price.py writes them for training."""
    frame = pd.DataFrame({"Date": epochs})
    frame = futures_price.calculate_trading_hours(frame)
    return futures_price.extract_cyclical_time_features(frame)


def live_time_features(date_column: pd.Series) -> dict[str, np.ndarray]:
    return es_features._time_features(date_column, len(date_column))


class LiveMatchesTrainingTest(unittest.TestCase):
    def assert_match(self, live: dict[str, np.ndarray], training: pd.DataFrame) -> None:
        for feature in TIME_FEATURES:
            with self.subTest(feature=feature):
                np.testing.assert_allclose(live[feature], training[feature].to_numpy(),
                                           atol=1e-9, err_msg=feature)

    def test_numeric_epoch_seconds(self):
        """Offline ES files store the date as a number of Unix seconds."""
        self.assert_match(live_time_features(pd.Series(EDGE_CASES_UTC)),
                          training_time_features(EDGE_CASES_UTC))

    def test_epoch_seconds_as_strings(self):
        """The live TWS feed (formatDate=2) sends Unix seconds as strings."""
        as_strings = pd.Series([str(value) for value in EDGE_CASES_UTC])
        self.assert_match(live_time_features(as_strings),
                          training_time_features(EDGE_CASES_UTC))


class KnownValuesTest(unittest.TestCase):
    """Pinned values, so the intent survives even if both sides change together."""

    def test_utc_morning_bar_is_read_in_eastern(self):
        features = live_time_features(pd.Series([epoch(2025, 5, 29, 9, 13, 5)]))
        # 09:13:05 UTC is 05:13:05 EDT: nearly 3.8h before 09:00 ET, not 0.2h after.
        self.assertAlmostEqual(features["Hours_From_Formal_Trading"][0], 5 + 13 / 60 + 5 / 3600 - 9)
        self.assertAlmostEqual(features["hour_sin"][0], np.sin(2 * np.pi * 5 / 24))

    def test_hour_features_use_the_whole_hour(self):
        on_the_hour = live_time_features(pd.Series([epoch(2025, 5, 29, 13, 0, 0)]))
        late_in_hour = live_time_features(pd.Series([epoch(2025, 5, 29, 13, 59, 59)]))
        self.assertEqual(on_the_hour["hour_sin"][0], late_in_hour["hour_sin"][0])


if __name__ == "__main__":
    unittest.main()
