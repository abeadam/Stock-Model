"""Does rotating the features rescue direction at the long horizon?

The 21-bar rule failed its holdout because the direction model had no out-of-sample skill
there (AUC 0.4985 and 0.5014 against 0.5110 on dev). Break-even at that horizon is only
about 52.5%, so even a small, durable lift in direction would matter -- which is the case
for asking whether compressing the features helps where the signal is weakest. Rotation
was only ever tested at five bars, where it lost to the raw columns.

Dev folds. The holdout months have been used three times on this line of work, so nothing
here is tested against them; a dev win would be an argument for October, not a result.
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

HORIZONS = (5, 21)
COMPONENT_COUNTS = (5, 10, 25)
STAGE = "dev"


def score(X, target, decided, data, transform=None, components=0):
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
    mid = data["mid"].to_numpy(float)
    columns = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    X = data[columns].to_numpy(np.float32)
    results = {}
    for horizon in HORIZONS:
        target = average_target(mid, seconds, horizon)
        decided = np.isfinite(target) & (target != 0)
        print(f"\n{horizon} bars, {len(columns)} features")
        mean, folds = score(X, target, decided, data)
        results[f"{horizon}|raw"] = mean
        print(f"   {'raw':<12}AUC {mean:.4f}  folds {folds}", flush=True)
        for components in COMPONENT_COUNTS:
            for label, transform in (("pls", pls_transform), ("svd", svd_transform)):
                mean, folds = score(X, target, decided, data, transform, components)
                results[f"{horizon}|{label}_{components}"] = mean
                print(f"   {f'{label} {components}':<12}AUC {mean:.4f}  folds {folds}", flush=True)
    with open("long_horizon_rotations.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
