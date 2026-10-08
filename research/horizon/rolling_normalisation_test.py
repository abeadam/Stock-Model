"""Does rescaling each feature against its own recent past survive where absolutes did not?

Two readings, on the same window, folds and target as every other feature study:
  how well a model can still tell June from September (the covariate shift itself);
  what the direction model scores.

Windows: one hour and one session of 5-second bars.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rolling_normalisation import normalised_names, rolling_z_scores
from rotation_test import TARGET_BARS, fold_splits, score_raw
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES

HOUR_BARS, SESSION_BARS = 720, 16_560
DRIFT_PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=200,
                    min_child_samples=200, n_jobs=-1, verbose=-1, random_state=7)


def month_identification_auc(data, columns) -> float:
    X = data[columns].to_numpy(np.float32)
    june = fold_splits(data, "dev")[0].test
    september = fold_splits(data, "holdout")[1].test
    rows = np.concatenate([june, september])
    period = np.concatenate([np.zeros(len(june)), np.ones(len(september))])
    order = np.random.default_rng(3).permutation(len(rows))
    rows, period = rows[order], period[order]
    half = len(rows) // 2
    model = lgb.LGBMClassifier(**DRIFT_PARAMS).fit(X[rows[:half]], period[:half])
    return roc_auc_score(period[half:], model.predict_proba(X[rows[half:]])[:, 1])


def main() -> None:
    started = time.time()
    data = load_cut()
    raw = data[FEATURES]
    sets = {"raw 44": FEATURES}
    for window in (HOUR_BARS, SESSION_BARS):
        scores = rolling_z_scores(raw, window)
        scores.columns = normalised_names(FEATURES, window)
        data = pd.concat([data, scores], axis=1)
        sets[f"z-scored, {window} bars"] = list(scores.columns)
        print(f"normalised over {window} bars ({time.time() - started:.0f}s)", flush=True)
    sets["raw + both windows"] = (FEATURES + normalised_names(FEATURES, HOUR_BARS)
                                  + normalised_names(FEATURES, SESSION_BARS))

    print("\nHow well can a model still tell June from September?")
    for label, columns in sets.items():
        print(f"   {label:<26}AUC {month_identification_auc(data, columns):.4f}", flush=True)

    print("\nDirection model, dev folds:")
    move = mid_move(data, TARGET_BARS)
    results = {}
    for label, columns in sets.items():
        mean, folds = score_raw(data[columns].to_numpy(np.float32), move, data, "dev")
        results[label] = {"mean": mean, "folds": folds}
        print(f"   {label:<26}AUC {mean:.4f}  folds {folds}  ({time.time() - started:.0f}s)", flush=True)
    with open("rolling_normalisation_test.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
