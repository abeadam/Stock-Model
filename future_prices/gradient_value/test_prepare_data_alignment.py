"""run_inference must put each basic-model prediction on the bar it was made from.

It used to rebuild row positions by counting in from the ends, assuming five
NaN-target rows at the tail. futures_price.py already trims those, so every
prediction landed five bars early: bar i carried the prediction made from bar
i+5's features, the very window the gradient target measures.

Run from this directory:  python -m unittest test_prepare_data_alignment
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data  # noqa: E402

ROWS = 40
WARMUP_ROWS = 5            # dropped at the start, like indicator warmup
INTERIOR_NAN_ROWS = (20, 21)  # dropped mid-dataset, e.g. an unobserved session


class MarkerModel:
    """Predicts each row's own marker, so a misplaced prediction is visible."""
    n_features_in_ = 1

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]


def fake_prepare_features_and_target(df, target_column, return_valid_mask=False):
    kept = np.ones(len(df), bool)
    kept[:WARMUP_ROWS] = False
    kept[list(INTERIOR_NAN_ROWS)] = False
    marker = df["marker"].to_numpy(float)[kept].reshape(-1, 1)
    result = (marker, None, ["marker"], None, None, None)
    return (*result, kept) if return_valid_mask else result


class RunInferenceAlignmentTest(unittest.TestCase):
    def test_each_prediction_lands_on_its_own_bar(self):
        frame = pd.DataFrame({"marker": np.arange(ROWS, dtype=float) / 100})
        with mock.patch.object(prepare_data, "prepare_features_and_target",
                               fake_prepare_features_and_target), \
             mock.patch.object(prepare_data, "apply_top_n_features_from_csv_or_correlation",
                               lambda X, cols, _y, _csv, _n: (X, cols, None)), \
             mock.patch.object(prepare_data.joblib, "load", lambda _path: MarkerModel()):
            result = prepare_data.run_inference(frame, Path("model.pkl"), "target")

        expected = np.sinh(frame["marker"].to_numpy()) / prepare_data._ASINH_SCALE
        expected[:WARMUP_ROWS] = np.nan
        expected[list(INTERIOR_NAN_ROWS)] = np.nan
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
