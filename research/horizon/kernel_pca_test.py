"""Does a non-linear projection find anything a linear one misses?

Every rotation tried here so far has been linear -- SVD, PCA, PLS, deep PLS -- and all of
them landed within noise of the raw columns. A kernel map is the obvious next question:
if the relationship between these features and the move is curved, a linear rotation
cannot represent it however many components it keeps.

The target is the 18-bar average mid move, which the horizon sweep found to be the most
stable place to stand: accuracy against break-even is least bad there and the per-trade
loss is smallest anywhere in this project.

Seven arms, all paired on the same seeds and folds:

  base            the 163 columns as they are
  pca 25          a linear rotation to 25 components, sampled and standardised exactly as
                  the kernel arms are -- the control that isolates the kernel itself
  kpca 25/50      the kernel map at two widths of output
  kpca gamma x10, /10
                  the same map with a deliberately wrong kernel width. A result that does
                  not move when gamma moves by two orders of magnitude is not a kernel
                  result, it is noise wearing one
  base + kpca 25  the kernel components added to the raw columns rather than replacing
                  them, which is how they would actually be used if they helped

usage: kernel_pca_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from kernel_projection import kernel_map, linear_map
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model

HORIZON_BARS = 18
COMPONENTS = 25
WIDE_COMPONENTS = 50
GAMMA_MULTIPLIERS = (10.0, 0.1)
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
PROJECTION_SEED = 11
RESULTS_PATH = "kernel_pca_test.json"


def paired_t(differences: np.ndarray) -> tuple[float, float]:
    error = differences.std(ddof=1) / np.sqrt(len(differences))
    return differences.mean(), differences.mean() / error if error > 0 else 0.0


def project_all(project, rows_by_part: dict, total_rows: int, components: int) -> np.ndarray:
    """One matrix holding every row this fold will touch, projected by this fold's own map."""
    matrix = np.zeros((total_rows, components), dtype=np.float32)
    for rows in rows_by_part.values():
        matrix[rows] = project(rows)
    return matrix


def fold_matrices(X: np.ndarray, split, base_columns: int) -> dict[str, np.ndarray]:
    """Every arm's feature matrix for one fold, fitted on that fold's training rows alone."""
    parts = {"train": split.train, "validation": split.validation, "test": split.test}
    rows = len(X)
    linear, _ = linear_map(X, split.train, COMPONENTS, seed=PROJECTION_SEED)
    kernel, gamma = kernel_map(X, split.train, COMPONENTS, seed=PROJECTION_SEED)
    wide, _ = kernel_map(X, split.train, WIDE_COMPONENTS, seed=PROJECTION_SEED)
    matrices = {"base": X,
                "pca 25": project_all(linear, parts, rows, COMPONENTS),
                "kpca 25": project_all(kernel, parts, rows, COMPONENTS),
                "kpca 50": project_all(wide, parts, rows, WIDE_COMPONENTS)}
    for multiplier in GAMMA_MULTIPLIERS:
        skewed, _ = kernel_map(X, split.train, COMPONENTS, gamma=gamma * multiplier,
                               seed=PROJECTION_SEED)
        matrices[f"kpca gamma x{multiplier:g}"] = project_all(skewed, parts, rows, COMPONENTS)
    matrices["base + kpca 25"] = np.hstack([X, matrices["kpca 25"]])
    return matrices, gamma


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), epochs, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    splits = purged_fold_splits(data["time"], HORIZON_BARS)
    print(f"{X.shape[1]} columns, horizon {HORIZON_BARS}, purge {HORIZON_BARS}, "
          f"{int(decided.sum()):,} decided rows, {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    scores: dict[str, list[float]] = {}
    for fold_number, split in enumerate(splits, start=1):
        matrices, gamma = fold_matrices(X, split, X.shape[1])
        print(f"fold {fold_number}: gamma {gamma:.5f}, {len(split.train):,} training rows "
              f"({time.time() - started:.0f}s)", flush=True)
        for name, matrix in matrices.items():
            for seed in SEEDS:
                model = train_direction_model(matrix, target, split,
                                              {**LGBM_PARAMS, "random_state": seed})
                test = split.test[decided[split.test]]
                scores.setdefault(name, []).append(
                    roc_auc_score((target[test] > 0).astype(int),
                                  model.predict_proba(matrix[test])[:, 1]))
            print(f"   {name:<20}{np.mean(scores[name][-len(SEEDS):]):.4f}"
                  f"   ({time.time() - started:.0f}s)", flush=True)
        del matrices
        with open(RESULTS_PATH, "w") as handle:
            json.dump(scores, handle, indent=1)

    print(f"\n{'arm':<22}{'columns':>9}{'mean AUC':>11}{'vs base':>11}{'t':>7}")
    print(f"{'base':<22}{X.shape[1]:>9}{np.mean(scores['base']):>11.4f}")
    widths = {"pca 25": COMPONENTS, "kpca 25": COMPONENTS, "kpca 50": WIDE_COMPONENTS,
              "base + kpca 25": X.shape[1] + COMPONENTS}
    for name in scores:
        if name == "base":
            continue
        mean, t = paired_t(np.array(scores[name]) - np.array(scores["base"]))
        print(f"{name:<22}{widths.get(name, COMPONENTS):>9}{np.mean(scores[name]):>11.4f}"
              f"{mean:>+11.5f}{t:>7.1f}")


if __name__ == "__main__":
    main()
