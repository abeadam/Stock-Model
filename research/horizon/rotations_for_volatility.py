"""Do rotations help the target that actually has signal?

Direction has been rotated every way and never gained. Volatility never has been, and it
is the different case: range is predictable (R2 ~0.47), so the question is whether
compressing redundant columns helps a model that is already finding a lot, or whether it
throws away detail the model was using.

The existing PLS helper fits on the sign of its target, which is meaningless for a range
that is always positive, so the supervised rotation here is fitted on the continuous
target instead. SVD never looks at the target and is unchanged.

Both targets, the same rotations, dev folds, regular hours, 153 features.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_auc_score

from feature_rank import prepare, principal_axes, standardise
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import PLS_ROW_STRIDE, SVD_ROW_STRIDE, fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_target_test import fit_regressor

HORIZON_BARS = 5
STAGE = "dev"
COMPONENT_COUNTS = (5, 10, 25)


def pls_regression_transform(X, target, split, components):
    """Directions of greatest covariance with the target itself, not with its sign."""
    rows = split.train[::PLS_ROW_STRIDE]
    rows = rows[np.isfinite(target[rows])]
    fitted = prepare(X[rows])
    model = PLSRegression(n_components=components, scale=False)
    model.fit(standardise(X[rows], fitted), target[rows])
    return lambda wanted: model.transform(standardise(X[wanted], fitted))


def svd_only_transform(X, target, split, components):
    sample = X[split.train[::SVD_ROW_STRIDE]]
    fitted = prepare(sample)
    axes = principal_axes(standardise(sample, fitted), components)
    return lambda wanted: standardise(X[wanted], fitted) @ axes


def rotated(X, target, split, transform, components):
    rotate = transform(X, target, split, components)
    matrix = np.zeros((len(X), components), dtype=np.float32)
    for rows in (split.train, split.validation, split.test):
        matrix[rows] = rotate(rows).astype(np.float32)
    return matrix


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    direction_target = average_target(mid, seconds, HORIZON_BARS)
    range_target = forward_target(prices, HORIZON_BARS, "range")
    decided = np.isfinite(direction_target) & (direction_target != 0)
    splits = fold_splits(data, STAGE)
    results = {}

    print("VOLATILITY (forward range), out-of-sample R2")
    for label, transform, counts in (("raw", None, (0,)), ("pls", pls_regression_transform, COMPONENT_COUNTS),
                                     ("svd", svd_only_transform, COMPONENT_COUNTS)):
        for components in counts:
            scores = []
            for split in splits:
                matrix = X if transform is None else rotated(X, range_target, split, transform, components)
                model = fit_regressor(matrix, range_target, split)
                known = split.test[np.isfinite(range_target[split.test])]
                scores.append(r2_score(range_target[known], model.predict(matrix[known])))
            name = label if transform is None else f"{label} {components}"
            results[f"range|{name}"] = float(np.mean(scores))
            print(f"   {name:<10}R2 {np.mean(scores):.4f}  folds {[round(s, 4) for s in scores]}",
                  flush=True)

    print("\nDIRECTION (average mid), AUC")
    for label, transform, counts in (("raw", None, (0,)), ("pls", pls_regression_transform, COMPONENT_COUNTS),
                                     ("svd", svd_only_transform, COMPONENT_COUNTS)):
        for components in counts:
            scores = []
            for split in splits:
                matrix = X if transform is None else rotated(X, direction_target, split, transform, components)
                model = train_direction_model(matrix, direction_target, split)
                test = split.test[decided[split.test]]
                scores.append(roc_auc_score((direction_target[test] > 0).astype(int),
                                            model.predict_proba(matrix[test])[:, 1]))
            name = label if transform is None else f"{label} {components}"
            results[f"direction|{name}"] = float(np.mean(scores))
            print(f"   {name:<10}AUC {np.mean(scores):.4f}  folds {[round(s, 4) for s in scores]}",
                  flush=True)
    with open("rotations_for_volatility.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
