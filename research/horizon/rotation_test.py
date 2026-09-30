"""Do rotated, orthogonal columns beat the raw feature set -- measured without leakage?

Two rotations of the same 44 production features:
  svd  principal components: the directions the features vary along most, target unseen
  pls  partial least squares: the directions that covary with the move, target in the loop

The rotation is part of the model, so it is fitted on a fold's training rows and nothing
else. An earlier version of this experiment built one rotated matrix for every fold in
turn, and since 53% of fold 1's test rows sit inside fold 2's training rows, the later
fold's rotation overwrote them -- fold 1 was then scored on columns fitted with knowledge
of its own test period. Here each fold builds its own matrix and throws it away.

usage: rotation_test.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from feature_rank import prepare, principal_axes, standardise
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, load_cut, mid_move
from short_horizon_walk_forward import FEATURES, direction_auc, fold_split, train_direction_model

TARGET_BARS = 12
STAGE = "dev"
COMPONENT_COUNTS = (5, 10, 25)
SVD_ROW_STRIDE = 20
PLS_ROW_STRIDE = 40


def fold_splits(data, stage: str = STAGE):
    return [fold_split(data["time"], fold, purge_bars=TARGET_BARS, fold_starts=FOLD_STARTS_UTC,
                       data_end=DATA_END_UTC) for fold in FOLDS_BY_STAGE[stage]]


def svd_transform(X, move, split, components):
    """Project onto the widest directions of this fold's training rows."""
    sample = X[split.train[::SVD_ROW_STRIDE]]
    fitted = prepare(sample)
    axes = principal_axes(standardise(sample, fitted), components)
    return lambda rows: standardise(X[rows], fitted) @ axes


def pls_transform(X, move, split, components):
    """Project onto the directions of this fold's training rows that covary with the move."""
    rows = split.train[::PLS_ROW_STRIDE]
    rows = rows[np.isfinite(move[rows])]
    fitted = prepare(X[rows])
    model = PLSRegression(n_components=components, scale=False)
    model.fit(standardise(X[rows], fitted), np.sign(move[rows]))
    return lambda wanted: model.transform(standardise(X[wanted], fitted))


def score_rotated(X, move, data, make_transform, components, stage=STAGE) -> tuple[float, list[float]]:
    """Each fold rotates with its own axes, trains, and scores its own test rows."""
    aucs = []
    for split in fold_splits(data, stage):
        transform = make_transform(X, move, split, components)
        matrix = np.zeros((len(X), components), dtype=np.float32)
        for rows in (split.train, split.validation, split.test):
            matrix[rows] = transform(rows).astype(np.float32)
        model = train_direction_model(matrix, move, split)
        aucs.append(direction_auc(model.predict_proba(matrix[split.test])[:, 1], move[split.test]))
    return float(np.mean(aucs)), [round(auc, 4) for auc in aucs]


def score_raw(X, move, data, stage=STAGE) -> tuple[float, list[float]]:
    aucs = []
    for split in fold_splits(data, stage):
        model = train_direction_model(X, move, split)
        aucs.append(direction_auc(model.predict_proba(X[split.test])[:, 1], move[split.test]))
    return float(np.mean(aucs)), [round(auc, 4) for auc in aucs]


def evaluate(stage: str = STAGE, counts=COMPONENT_COUNTS) -> None:
    started = time.time()
    print(f"stage {stage}: folds {FOLDS_BY_STAGE[stage]}", flush=True)
    data = load_cut()
    move = mid_move(data, TARGET_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    mean, folds = score_raw(X, move, data, stage)
    rows = [{"axes": "raw", "components": len(FEATURES), "mean_auc": mean, "fold_auc": folds}]
    print(f"raw  {len(FEATURES):>4} columns  AUC {mean:.4f}  folds {folds}  ({time.time() - started:.0f}s)",
          flush=True)
    for components in counts:
        for name, transform in (("svd", svd_transform), ("pls", pls_transform)):
            mean, folds = score_rotated(X, move, data, transform, components, stage)
            rows.append({"axes": name, "components": components, "mean_auc": mean, "fold_auc": folds})
            print(f"{name:<5}{components:>4} columns  AUC {mean:.4f}  folds {folds}  "
                  f"({time.time() - started:.0f}s)", flush=True)
    with open(f"rotation_test_{stage}.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else STAGE
    counts = tuple(int(n) for n in sys.argv[2:]) or COMPONENT_COUNTS
    evaluate(stage, counts)
