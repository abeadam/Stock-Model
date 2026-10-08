"""The holdout check for the regular-hours mid-average work: August and September.

Pre-registered before running. Three configurations only, so the holdout is not shopped:
  core 44                      the production features
  core + mid microstructure    the 24 new bounce-free columns
  all three                    plus the returns of the added symbols

Pass condition, fixed in advance: a candidate passes only if it beats core 44 in BOTH
holdout months. Dev gave core 0.5019, +mid 0.5090, all three 0.5128.

Economics are printed alongside, since an AUC that passes and a trade that loses are
different facts and the second one decides whether any of this matters.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_economics import BANDS, MIN_VALIDATION_TRADES, outcome
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model

HORIZON_BARS = 5
PERCENT = 100.0
POINT_VALUE = 50.0
HOLDOUT = "holdout"
MONTHS = ("August", "September")
DEV = {"core 44": 0.5019, "core + mid microstructure": 0.5090, "all three": 0.5128}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    points = target / PERCENT * data["mid"].to_numpy(float)
    decided = np.isfinite(target) & (target != 0)
    sets = {"core 44": list(FEATURES),
            "core + mid microstructure": list(FEATURES) + MID_FEATURES,
            "all three": list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES}
    splits = fold_splits(data, HOLDOUT)
    results, trades = {}, {}
    for name, columns in sets.items():
        X = data[columns].to_numpy(np.float32)
        aucs, per_fold = [], []
        for split in splits:
            model = train_direction_model(X, target, split)
            test = split.test[decided[split.test]]
            score = model.predict_proba(X[test])[:, 1] - 0.5
            aucs.append(roc_auc_score((target[test] > 0).astype(int), score))
            validation = split.validation[decided[split.validation]]
            validation_score = model.predict_proba(X[validation])[:, 1] - 0.5
            best, best_net = None, -np.inf
            for quantile in BANDS:
                threshold = float(np.quantile(np.abs(validation_score), quantile))
                chosen = outcome(validation_score, threshold, points[validation])
                if chosen["trades"] >= MIN_VALIDATION_TRADES and chosen["net"] > best_net:
                    best, best_net = threshold, chosen["net"]
            per_fold.append(outcome(score, best, points[test]) if best is not None else {"trades": 0})
        results[name] = aucs
        trades[name] = per_fold
    baseline = results["core 44"]
    print(f"{'model':<28}{'dev':>8}{'holdout':>9}{'August':>9}{'September':>11}   verdict")
    for name, aucs in results.items():
        verdict = ""
        if name != "core 44":
            beats = [aucs[i] > baseline[i] for i in range(len(aucs))]
            verdict = "PASS" if all(beats) else f"FAIL ({sum(beats)} of {len(beats)} months)"
        print(f"{name:<28}{DEV[name]:>8.4f}{np.mean(aucs):>9.4f}{aucs[0]:>9.4f}{aucs[1]:>11.4f}   {verdict}")
    print(f"\n{'model':<28}{'month':>11}{'trades':>9}{'accuracy':>10}{'per trade':>11}")
    for name, per_fold in trades.items():
        for month, result in zip(MONTHS, per_fold):
            print(f"{name:<28}{month:>11}{result.get('trades', 0):>9,}"
                  f"{result.get('accuracy', float('nan')):>9.2%}"
                  f"{result.get('per_trade', float('nan')):>+11.2f}")
    with open("holdout_mid_features.json", "w") as handle:
        json.dump({"auc": {k: v for k, v in results.items()},
                   "trades": {k: v for k, v in trades.items()}}, handle, indent=1)


if __name__ == "__main__":
    main()
