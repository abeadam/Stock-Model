"""Stacking size onto direction at the horizons where direction actually survives.

The 21-bar attempt failed because direction had no out-of-sample skill there. The five-bar
configuration is the one that passed a holdout, so this asks the same question where the
signal lives: 5, 8 and 12 bars, with the full 153-feature set rather than the 68 used in
the sweeps.

Two ways of joining the two models:
  rule     |direction| * predicted range, or a gate on predicted range, then the strongest
           direction scores -- arithmetic, nothing fitted
  stack    a meta-model over [direction score, predicted range, predicted high, predicted
           low]. The base models are fitted on the training rows, their predictions are
           taken on the *validation* rows, the meta-model is fitted there, and only then is
           anything scored on test -- so the stacker never sees its inputs' training data.

Per-fold figures throughout: a fold with few trades must not be averaged against one with
many.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from direction_times_volatility import RANGE_GATE, settle, take
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_target_test import fit_regressor

HORIZONS = (5, 8, 12)
SHARES = (0.05, 0.01)
PERCENT = 100.0
STAGE = "dev"
STACKER_PARAMS = dict(objective="binary", num_leaves=15, max_depth=3, learning_rate=0.05,
                      n_estimators=200, min_child_samples=200, n_jobs=-1, verbose=-1, random_state=9)


def base_predictions(X, targets, split, rows):
    return {name: model.predict(X[rows]) if name != "direction"
            else model.predict_proba(X[rows])[:, 1] - 0.5
            for name, model in targets.items()}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    results = {}
    for horizon in HORIZONS:
        direction_target = average_target(mid, seconds, horizon)
        points = direction_target / PERCENT * mid
        decided = np.isfinite(direction_target) & (direction_target != 0)
        sizes = {kind: forward_target(prices, horizon, kind) for kind in ("range", "highest", "lowest")}
        print(f"\n=== {horizon} bars ===")
        print(f"   {'rule':<16}{'share':>6}{'fold':>5}{'trades':>8}{'accuracy':>10}{'move':>7}"
              f"{'need':>7}{'$/trade':>10}")
        for fold, split in enumerate(fold_splits(data, STAGE), 1):
            models = {"direction": train_direction_model(X, direction_target, split)}
            for kind, target in sizes.items():
                models[kind] = fit_regressor(X, target, split)
            test = split.test[decided[split.test]]
            validation = split.validation[decided[split.validation]]
            on_test = base_predictions(X, models, split, test)
            on_validation = base_predictions(X, models, split, validation)
            score, predicted_range = on_test["direction"], on_test["range"]
            auc = roc_auc_score((direction_target[test] > 0).astype(int), score)

            stacker = lgb.LGBMClassifier(**STACKER_PARAMS)
            stacker.fit(np.column_stack([on_validation[name] for name in models]),
                        (direction_target[validation] > 0).astype(int))
            stacked = stacker.predict_proba(np.column_stack([on_test[name] for name in models]))[:, 1] - 0.5
            stacked_auc = roc_auc_score((direction_target[test] > 0).astype(int), stacked)

            wide = predicted_range >= np.quantile(predicted_range, RANGE_GATE)
            for share in SHARES:
                for name, taken, side in (
                        ("confidence", take(score, share), score),
                        ("expected", take(score * np.maximum(predicted_range, 0), share), score),
                        ("gated", take(np.where(wide, score, 0.0), share) & wide, score),
                        ("stacked", take(stacked, share), stacked),
                        ("stacked+size", take(stacked * np.maximum(predicted_range, 0), share), stacked)):
                    result = settle(side, taken, points[test])
                    result["per_trade"] = result["total"] / max(result["trades"], 1)
                    results[f"{horizon}|{name}|{share}|{fold}"] = result
                    print(f"   {name:<16}{share:>5.0%}{fold:>5}{result['trades']:>8,}"
                          f"{result['accuracy']:>9.2%}{result['median_move']:>7.3f}"
                          f"{result['break_even']:>7.1%}{result['per_trade']:>+10.2f}", flush=True)
            print(f"   {'':<16}{'':>6}{fold:>5}  direction AUC {auc:.4f}   stacked AUC {stacked_auc:.4f}")
    with open("stacked_direction_volatility.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
