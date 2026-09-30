"""Predictions that never saw the row they describe.

Feeding one model's output to another is the fastest way to manufacture a result here, and
this project has already paid for it twice: `PRED_FEATURES` exists as a leak control
because the highest/lowest sub-models were fit on every row, and the gradient five-bar
prediction leak was the same mistake in another place. A model fitted on all the training
rows predicts those rows far better than it will ever predict a new one, so a downstream
model learns to trust a signal that will not be that good live.

The fix is the standard one. Inside a fold's training block, rows are cut into
time-ordered blocks and each block is predicted by a model fitted only on the blocks
**before** it, with a purge so no training label reaches into the block being predicted.
The earliest block has nothing before it and stays blank. Validation and test rows are
predicted by a model fitted on the whole training block, which they are already outside.

The result is a feature whose accuracy on training rows is as honest as its accuracy live.
"""
from __future__ import annotations

import numpy as np

INNER_BLOCKS = 5
MIN_FIT_ROWS = 5_000


def block_edges(row_count: int, blocks: int) -> list[np.ndarray]:
    """Positions 0..row_count-1 cut into `blocks` contiguous, time-ordered pieces."""
    return np.array_split(np.arange(row_count), blocks)


def out_of_fold(train_rows: np.ndarray, fit_predict, blocks: int = INNER_BLOCKS,
                purge_bars: int = 0, min_fit_rows: int = MIN_FIT_ROWS) -> np.ndarray:
    """Predictions for every row of `train_rows`, each from a model that never saw it.

    `fit_predict(history_rows, wanted_rows)` fits on the first and returns predictions for
    the second. Blocks with too little history behind them stay blank rather than being
    filled by a model fitted on a handful of rows."""
    predictions = np.full(len(train_rows), np.nan)
    for block in block_edges(len(train_rows), blocks):
        if len(block) == 0 or block[0] == 0:
            continue                                    # nothing earlier to learn from
        history = train_rows[:block[0]]
        if purge_bars:
            history = history[:-purge_bars] if len(history) > purge_bars else history[:0]
        if len(history) < min_fit_rows:
            continue
        predictions[block] = fit_predict(history, train_rows[block])
    return predictions


def scatter(values: np.ndarray, rows: np.ndarray, total_rows: int) -> np.ndarray:
    """A full-length column holding `values` at `rows` and NaN everywhere else."""
    column = np.full(total_rows, np.nan)
    column[rows] = values
    return column
