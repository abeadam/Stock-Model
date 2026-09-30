"""PLS and SVD as transforms on the new target, not just as a way of reading it.

PLS found which values matter, but the models that passed the holdout took the columns
raw. Rotation failed its own holdout earlier -- on the old target: endpoint, all day,
trade closes. This asks the question again on the target that worked: the average mid
move over the next five bars, regular hours, with the microstructure features.

Dev folds only. August and September are spent on this question, so a win here is a
reason to test on October, not a result.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits, pls_transform, svd_transform
from short_horizon_walk_forward import FEATURES, train_direction_model

HORIZON_BARS = 5
STAGE = "dev"
COMPONENT_COUNTS = (5, 10, 25)


def score(X, target, decided, data, transform=None, components=0) -> tuple[float, list[float]]:
    aucs = []
    for split in fold_splits(data, STAGE):
        if transform is None:
            matrix = X
        else:
            rotate = transform(X, target, split, components)
            matrix = np.zeros((len(X), components), dtype=np.float32)
            for rows in (split.train, split.validation, split.test):
                matrix[rows] = rotate(rows).astype(np.float32)
        model = train_direction_model(matrix, target, split)
        test = split.test[decided[split.test]]
        aucs.append(roc_auc_score((target[test] > 0).astype(int),
                                  model.predict_proba(matrix[test])[:, 1]))
    return float(np.mean(aucs)), [round(a, 4) for a in aucs]


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    sets = {"core + mid microstructure": list(FEATURES) + MID_FEATURES,
            "all three": list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES}
    results = {}
    for name, columns in sets.items():
        X = data[columns].to_numpy(np.float32)
        print(f"\n{name} ({len(columns)} features)")
        mean, folds = score(X, target, decided, data)
        results[f"{name}|raw"] = mean
        print(f"   {'raw':<16}AUC {mean:.4f}  folds {folds}", flush=True)
        for components in COMPONENT_COUNTS:
            for label, transform in (("pls", pls_transform), ("svd", svd_transform)):
                mean, folds = score(X, target, decided, data, transform, components)
                results[f"{name}|{label}_{components}"] = mean
                print(f"   {f'{label} {components}':<16}AUC {mean:.4f}  folds {folds}", flush=True)
    with open("mid_target_rotations.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
