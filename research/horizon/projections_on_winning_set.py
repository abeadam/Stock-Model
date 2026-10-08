"""Deep PLS and kernel PCA on the best feature set this project has.

The projections tried so far ran on the 163 columns. That set has since grown: two
realized-volatility windows and a hundred TA-Lib range picks together take the volatility
head from 0.4674 to 0.5083. Whether a projection helps is a question about a particular
feature space, so it is worth re-asking on this one rather than assuming the old answer.

Five arms beyond the full set, every transform fitted on **that fold's training rows
alone** and on a **random** sample of them -- the two mistakes this project has already
paid for, a rotation shared across folds and a stride-sampled one:

  pca 25        linear rotation, the control that isolates everything else
  pls 25        supervised rotation: directions that covary with the target
  pls-relu      each PLS score split into its positive and negative parts, so a linear
                read-out becomes piecewise linear -- the cheapest honest nonlinearity
                (Polson, Sokolov & Xu, arXiv 2106.14085)
  deep pls      that layer stacked: PLS -> ReLU -> PLS -> ReLU, which their Theorem 1
                says can be estimated recursively by least squares
  kpca 25       an RBF kernel map, with gamma from the median heuristic

Both heads, paired over the same seeds and folds, purged by the horizon.

usage: projections_on_winning_set.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from cross_asset_features import load_symbol_closes
from deep_ladder_both_heads import head_scores, paired_t
from deep_pls import relu_expand
from feature_rank import prepare, principal_axes, standardise
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from kernel_projection import kernel_map, sample_rows
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS
from talib_ablation import build_selected
from variance_risk_premium import REALIZED_WINDOWS, add_variance_premium_features

HORIZON_BARS = 18
COMPONENTS = 25
LANDMARKS = 3000
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
PROJECTION_SEED = 11
RANKING_PATH = "talib_screen_ranking.json"
RECIPES_PATH = "talib_screen_recipes.json"
RESULTS_PATH_TEMPLATE = "projections_on_winning_set_h{horizon}.json"


def fitted_pls(X, target, rows, components):
    """PLS fitted on a random sample of the given rows, with its standardisation."""
    usable = rows[np.isfinite(target[rows])]
    sample = sample_rows(usable, LANDMARKS, PROJECTION_SEED)
    fitted = prepare(X[sample])
    model = PLSRegression(n_components=components, scale=False)
    model.fit(standardise(X[sample], fitted), np.sign(target[sample]))
    return lambda block: model.transform(standardise(block, fitted))


def project_everything(transform, X, parts, width) -> np.ndarray:
    matrix = np.zeros((len(X), width), dtype=np.float32)
    for rows in parts:
        matrix[rows] = transform(X[rows]).astype(np.float32)
    return matrix


def by_row_index(project, X, parts, width) -> np.ndarray:
    """For a projector that takes row indices rather than a block of values."""
    matrix = np.zeros((len(X), width), dtype=np.float32)
    for rows in parts:
        matrix[rows] = project(rows).astype(np.float32)
    return matrix


def fold_matrices(X, target, split) -> dict[str, np.ndarray]:
    """Every projection for one fold, each fitted on that fold's training rows alone."""
    parts = (split.train, split.validation, split.test)
    axes_sample = sample_rows(split.train, LANDMARKS, PROJECTION_SEED)
    fitted = prepare(X[axes_sample])
    axes = principal_axes(standardise(X[axes_sample], fitted), COMPONENTS)
    pls = fitted_pls(X, target, split.train, COMPONENTS)

    labelled = split.train[np.isfinite(target[split.train])]
    inner_sample = sample_rows(labelled, LANDMARKS, PROJECTION_SEED)
    inner = PLSRegression(n_components=COMPONENTS, scale=False)
    inner.fit(relu_expand(pls(X[inner_sample])), np.sign(target[inner_sample]))

    def stacked(block):
        """PLS -> ReLU -> PLS -> ReLU, the recursive least-squares stack."""
        return relu_expand(inner.transform(relu_expand(pls(block))))

    kernel, _ = kernel_map(X, split.train, COMPONENTS, landmarks=LANDMARKS, seed=PROJECTION_SEED)
    return {"pca 25": project_everything(lambda b: standardise(b, fitted) @ axes, X, parts, COMPONENTS),
            "pls 25": project_everything(pls, X, parts, COMPONENTS),
            "pls-relu": project_everything(lambda b: relu_expand(pls(b)), X, parts, 2 * COMPONENTS),
            "deep pls": project_everything(stacked, X, parts, 2 * COMPONENTS),
            "kpca 25": by_row_index(kernel, X, parts, COMPONENTS)}


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    with open(RECIPES_PATH) as handle:
        recipes = json.load(handle)
    with open(RANKING_PATH) as handle:
        range_picks = json.load(handle)["range"]
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = add_variance_premium_features(data, load_symbol_closes("VIX"))
    data = build_selected(data, {name: recipes[name] for name in range_picks if name in recipes})
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    winning = (list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
               + [f"vrp_realized_per_bar_{window}" for window in REALIZED_WINDOWS]
               + [name for name in range_picks if name in data.columns])
    X = data[winning].to_numpy(np.float32)
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; winning set {len(winning)} columns; "
          f"{COMPONENTS} components; {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    auc, r2 = {}, {}
    for fold_number, split in enumerate(splits, start=1):
        matrices = {"full set": X, **fold_matrices(X, direction_target, split)}
        print(f"fold {fold_number}: projections fitted ({time.time() - started:.0f}s)", flush=True)
        for name, matrix in matrices.items():
            for seed in SEEDS:
                aucs, r2s, _ = head_scores(matrix, direction_target, decided, range_target,
                                           [split], {**LGBM_PARAMS, "random_state": seed}, added=0)
                auc.setdefault(name, []).extend(aucs)
                r2.setdefault(name, []).extend(r2s)
            print(f"   {name:<12}({matrix.shape[1]} columns): direction "
                  f"{np.mean(auc[name][-len(SEEDS):]):.4f}  volatility R2 "
                  f"{np.mean(r2[name][-len(SEEDS):]):.4f}   ({time.time() - started:.0f}s)",
                  flush=True)
        del matrices
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": auc, "r2": r2}, handle, indent=1)

    print(f"\n{'arm':<14}{'direction':>12}{'t':>7}{'volatility':>13}{'t':>7}")
    print(f"{'full set':<14}{np.mean(auc['full set']):>12.4f}{'':>7}{np.mean(r2['full set']):>13.4f}")
    for name in auc:
        if name == "full set":
            continue
        auc_gain, auc_t = paired_t(np.array(auc[name]) - np.array(auc["full set"]))
        r2_gain, r2_t = paired_t(np.array(r2[name]) - np.array(r2["full set"]))
        print(f"{name:<14}{auc_gain:>+12.5f}{auc_t:>7.1f}{r2_gain:>+13.5f}{r2_t:>7.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
