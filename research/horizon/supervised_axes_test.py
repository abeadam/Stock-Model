"""SVD keeps the widest directions. Should it keep the most predictive ones instead?

Principal components are chosen without ever looking at the target: they are the
directions the features vary along most, which need not be the directions that move
price. A feature set can carry its signal in a low-variance direction that a 95% cut
throws away first.

Partial least squares picks its directions by covariance with the target instead, so it
answers the same "give the model orthogonal columns" idea with the target in the loop.
Scored against raw features and against the SVD rotation, same window, folds and target.

Both rotations are fitted on training rows only.

usage: supervised_axes_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from feature_rank import prepare, standardise
from rank_reduction_test import SVD_ROW_STRIDE, STAGE, TARGET_BARS, projected
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, load_cut, mid_move
from short_horizon_walk_forward import FEATURES, direction_auc, fold_split, train_direction_model

COMPONENT_COUNTS = (5, 10, 25)
PLS_ROW_STRIDE = 40                    # partial least squares is the slower fit of the two


def fold_splits(data):
    return {fold: fold_split(data["time"], fold, purge_bars=TARGET_BARS,
                             fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
            for fold in FOLDS_BY_STAGE[STAGE]}


def supervised_axes(X: np.ndarray, move: np.ndarray, data, components: int) -> np.ndarray:
    """X rebuilt as `components` columns chosen for covariance with the move."""
    rebuilt = np.zeros((len(X), components), dtype=np.float32)
    for split in fold_splits(data).values():
        rows = split.train[::PLS_ROW_STRIDE]
        rows = rows[np.isfinite(move[rows])]
        fitted = prepare(X[rows])
        model = PLSRegression(n_components=components, scale=False)
        model.fit(standardise(X[rows], fitted), np.sign(move[rows]))
        for part in (split.train, split.validation, split.test):
            rebuilt[part] = model.transform(standardise(X[part], fitted)).astype(np.float32)
    return rebuilt


def score(matrix: np.ndarray, move: np.ndarray, data) -> tuple[float, list[float]]:
    aucs = []
    for split in fold_splits(data).values():
        model = train_direction_model(matrix, move, split)
        aucs.append(direction_auc(model.predict_proba(matrix[split.test])[:, 1], move[split.test]))
    return float(np.mean(aucs)), [round(auc, 4) for auc in aucs]


def evaluate() -> None:
    started = time.time()
    data = load_cut()
    move = mid_move(data, TARGET_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    rows = [{"axes": "raw", "components": len(FEATURES), "auc": score(X, move, data)}]
    for components in COMPONENT_COUNTS:
        rows.append({"axes": "svd", "components": components,
                     "auc": score(projected(X, data, components), move, data)})
        rows.append({"axes": "pls", "components": components,
                     "auc": score(supervised_axes(X, move, data, components), move, data)})
        for row in rows[-2:]:
            mean, folds = row["auc"]
            print(f"{row['axes']:<5}{row['components']:>4} columns  AUC {mean:.4f}  folds {folds}  "
                  f"({time.time() - started:.0f}s)", flush=True)
    mean, folds = rows[0]["auc"]
    print(f"raw  {len(FEATURES):>4} columns  AUC {mean:.4f}  folds {folds}")
    with open("supervised_axes_test.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    evaluate()
