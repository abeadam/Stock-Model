"""Tests for out-of-fold prediction generation.

The first two are the ones that matter: a model must never be asked to predict a row it
was fitted on, and the purge must hold inside the inner blocks as well as between folds.
"""
from __future__ import annotations

import numpy as np
import pytest

from out_of_fold_predictions import MIN_FIT_ROWS, block_edges, out_of_fold, scatter

TRAIN_ROWS = np.arange(1000, 51000)


def recording_fit_predict(seen: list):
    def fit_predict(history_rows, wanted_rows):
        seen.append((history_rows, wanted_rows))
        return np.full(len(wanted_rows), float(history_rows[-1]))
    return fit_predict


def test_no_block_is_predicted_by_a_model_that_saw_it():
    seen = []
    out_of_fold(TRAIN_ROWS, recording_fit_predict(seen), blocks=5, min_fit_rows=1)
    for history, wanted in seen:
        assert not set(history) & set(wanted)


def test_every_fit_uses_only_rows_earlier_than_what_it_predicts():
    seen = []
    out_of_fold(TRAIN_ROWS, recording_fit_predict(seen), blocks=5, min_fit_rows=1)
    for history, wanted in seen:
        assert history.max() < wanted.min()


def test_the_purge_removes_rows_immediately_before_each_block():
    seen = []
    out_of_fold(TRAIN_ROWS, recording_fit_predict(seen), blocks=5, purge_bars=18, min_fit_rows=1)
    for history, wanted in seen:
        assert wanted.min() - history.max() > 18


def test_the_earliest_block_is_left_blank():
    predictions = out_of_fold(TRAIN_ROWS, recording_fit_predict([]), blocks=5, min_fit_rows=1)
    first_block = block_edges(len(TRAIN_ROWS), 5)[0]
    assert np.isnan(predictions[first_block]).all()


def test_later_blocks_are_filled():
    predictions = out_of_fold(TRAIN_ROWS, recording_fit_predict([]), blocks=5, min_fit_rows=1)
    last_block = block_edges(len(TRAIN_ROWS), 5)[-1]
    assert np.isfinite(predictions[last_block]).all()


def test_blocks_with_too_little_history_stay_blank():
    """Better a gap than a feature built by a model fitted on a handful of rows."""
    predictions = out_of_fold(np.arange(100), recording_fit_predict([]), blocks=5,
                              min_fit_rows=MIN_FIT_ROWS)
    assert np.isnan(predictions).all()


def test_blocks_partition_the_training_rows():
    blocks = block_edges(len(TRAIN_ROWS), 5)
    assert np.array_equal(np.concatenate(blocks), np.arange(len(TRAIN_ROWS)))


def test_blocks_stay_in_time_order():
    blocks = block_edges(997, 5)
    assert all(earlier[-1] < later[0] for earlier, later in zip(blocks, blocks[1:]))


def test_the_returned_array_lines_up_with_the_training_rows():
    predictions = out_of_fold(TRAIN_ROWS, recording_fit_predict([]), blocks=5, min_fit_rows=1)
    assert len(predictions) == len(TRAIN_ROWS)


def test_predictions_carry_the_value_the_model_returned():
    predictions = out_of_fold(TRAIN_ROWS, recording_fit_predict([]), blocks=2, min_fit_rows=1)
    second_block = block_edges(len(TRAIN_ROWS), 2)[1]
    assert predictions[second_block][0] == pytest.approx(float(TRAIN_ROWS[second_block[0] - 1]))


def test_scatter_places_values_and_blanks_the_rest():
    column = scatter(np.array([1.0, 2.0]), np.array([3, 7]), total_rows=10)
    assert column[3] == 1.0 and column[7] == 2.0
    assert np.isnan(column[[0, 1, 2, 4, 5, 6, 8, 9]]).all()
