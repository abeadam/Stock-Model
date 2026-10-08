"""Is the forward move per unit of volume a better thing to learn than the move itself?

Regular trading hours only (09:30-16:00 ET), five bars ahead (25 seconds), mid prices so
no bid-ask bounce enters the label, dev folds, every model fitted inside its own fold.

Three targets, each fitted by the same regressor and then judged two ways -- how much of
its own variance it explains, and how well its prediction ranks the direction the mid
actually went, which is the only thing a trade can use. The sign-classifier baseline is
printed alongside, since scaling by a positive number leaves the sign untouched and a
classifier therefore cannot tell the targets apart.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from rotation_test import TARGET_BARS, fold_splits
from segment_walk_forward import load_cut
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_scaled_target import regular_hours, scaled_targets

HORIZON_BARS = 5
STAGE = "dev"
REGRESSOR_PARAMS = dict(objective="regression", num_leaves=127, learning_rate=0.05,
                        min_child_samples=500, feature_fraction=0.8, bagging_fraction=0.7,
                        bagging_freq=1, lambda_l2=1.0, n_estimators=2000, n_jobs=-1,
                        verbose=-1, random_state=11)
EARLY_STOPPING = 200


def fit_regressor(X, target, split):
    known = np.isfinite(target)
    train, validation = split.train[known[split.train]], split.validation[known[split.validation]]
    model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    model.fit(X[train], target[train], eval_set=[(X[validation], target[validation])],
              eval_metric="l2", callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                                           lgb.log_evaluation(0)])
    return model


def main() -> None:
    data = load_cut()
    session = regular_hours(data["time"])
    data = data[session].reset_index(drop=True)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    targets = scaled_targets(data["mid"].to_numpy(), data["Volume"].to_numpy(float), seconds, HORIZON_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    move = targets["percent"]
    decided = np.isfinite(move) & (move != 0)
    print(f"regular hours only: {len(data):,} bars over {data['session_day'].nunique()} days; "
          f"{decided.mean():.1%} of bars have a non-zero {HORIZON_BARS}-bar mid move\n")

    results = {}
    for name, target in targets.items():
        r2s, aucs = [], []
        for split in fold_splits(data, STAGE):
            model = fit_regressor(X, target, split)
            test = split.test[np.isfinite(target[split.test])]
            prediction = model.predict(X[test])
            r2s.append(r2_score(target[test], prediction))
            ranked = split.test[decided[split.test]]
            aucs.append(roc_auc_score((move[ranked] > 0).astype(int), model.predict(X[ranked])))
        results[name] = {"r2": float(np.mean(r2s)), "auc": float(np.mean(aucs)),
                         "fold_auc": [round(a, 4) for a in aucs], "fold_r2": [round(r, 4) for r in r2s]}
        print(f"{name:<30}R2 {np.mean(r2s):>8.4f}   direction AUC {np.mean(aucs):.4f}  "
              f"folds {results[name]['fold_auc']}", flush=True)

    aucs = []
    for split in fold_splits(data, STAGE):
        model = train_direction_model(X, move, split)
        ranked = split.test[decided[split.test]]
        aucs.append(roc_auc_score((move[ranked] > 0).astype(int), model.predict_proba(X[ranked])[:, 1]))
    results["sign_classifier"] = {"auc": float(np.mean(aucs)), "fold_auc": [round(a, 4) for a in aucs]}
    print(f"{'sign classifier (baseline)':<30}{'':>8}      direction AUC {np.mean(aucs):.4f}  "
          f"folds {results['sign_classifier']['fold_auc']}")
    with open("volume_target_test.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
