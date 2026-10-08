"""Are the 44 production features stationary, and has their distribution moved?

ADF on each feature, sampled to a one-minute grid so the test sees a sensible time
structure. Then the question that actually bears on a failing walk-forward: train a model
to tell a June bar from a September bar using the features alone. An AUC near 0.5 means
the two months look alike and a threshold learned in one still describes the other; an
AUC near 1 means they are different worlds.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from rotation_test import TARGET_BARS, fold_splits
from segment_walk_forward import load_cut
from short_horizon_walk_forward import FEATURES
from stationarity import CRITICAL_VALUES, augmented_dickey_fuller, has_unit_root

MINUTE_STRIDE = 12
DRIFT_PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=200,
                    min_child_samples=200, n_jobs=-1, verbose=-1, random_state=7)


def main() -> None:
    data = load_cut()
    X = data[FEATURES].to_numpy(np.float32)

    print("=== Augmented Dickey-Fuller, one-minute grid ===")
    statistics = {name: augmented_dickey_fuller(X[::MINUTE_STRIDE, index])
                  for index, name in enumerate(FEATURES)}
    wandering = {name: value for name, value in statistics.items() if has_unit_root(value)}
    ordered = sorted(statistics.items(), key=lambda item: item[1], reverse=True)
    print(f"5% critical value {CRITICAL_VALUES['5%']}; {len(wandering)} of {len(FEATURES)} "
          f"features fail to reject a unit root")
    print("least stationary (largest statistic):")
    for name, value in ordered[:8]:
        print(f"   {name:<30}{value:>12.1f}{'   UNIT ROOT' if has_unit_root(value) else ''}")
    print(f"most stationary: {ordered[-1][0]} at {ordered[-1][1]:.0f}")

    print("\n=== Can a model tell June from September using only the features? ===")
    june = fold_splits(data, "dev")[0].test
    september = fold_splits(data, "holdout")[1].test
    rows = np.concatenate([june, september])
    period = np.concatenate([np.zeros(len(june)), np.ones(len(september))])
    shuffled = np.random.default_rng(3).permutation(len(rows))
    rows, period = rows[shuffled], period[shuffled]
    half = len(rows) // 2
    model = lgb.LGBMClassifier(**DRIFT_PARAMS)
    model.fit(X[rows[:half]], period[:half])
    auc = roc_auc_score(period[half:], model.predict_proba(X[rows[half:]])[:, 1])
    print(f"{len(june):,} June bars vs {len(september):,} September bars: AUC {auc:.4f}")
    gains = model.booster_.feature_importance("gain")
    print("features that give the month away:")
    for index in np.argsort(-gains)[:8]:
        print(f"   {FEATURES[index]:<30}{gains[index] / gains.sum():>7.1%} of split gain")
    with open("stationarity_report.json", "w") as handle:
        json.dump({"adf": statistics, "drift_auc": auc}, handle, indent=1)


if __name__ == "__main__":
    main()
