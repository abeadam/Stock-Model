"""What the best regular-hours feature set is worth on the average-mid target.

Accuracy by confidence band, and a trade at $5 the round trip whose entry threshold is
chosen on each fold's validation rows and then applied unchanged to its test rows.
Collecting an average means leaving in equal parts across the five bars, so the single
round trip charged here is the optimistic reading.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model

HORIZON_BARS = 5
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
PERCENT = 100.0
STAGE = "dev"
BANDS = (0.5, 0.8, 0.9, 0.95, 0.99)
MIN_VALIDATION_TRADES = 200


def outcome(score, threshold, points) -> dict:
    side = np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, 0.0))
    taken = side != 0
    if taken.sum() == 0:
        return {"trades": 0, "accuracy": np.nan, "per_trade": np.nan, "net": 0.0}
    gained = side[taken] * points[taken] * POINT_VALUE - ROUND_TRIP_COST
    return {"trades": int(taken.sum()), "accuracy": float((side[taken] * points[taken] > 0).mean()),
            "per_trade": float(gained.mean()), "net": float(gained.sum())}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    points = target / PERCENT * data["mid"].to_numpy(float)
    decided = np.isfinite(target) & (target != 0)
    columns = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    X = data[columns].to_numpy(np.float32)
    median_move = float(np.nanmedian(np.abs(points)))
    break_even = 0.5 * (1 + ROUND_TRIP_COST / (median_move * POINT_VALUE))
    print(f"median |average mid move| {median_move:.3f} pt = ${median_move * POINT_VALUE:.2f}; "
          f"break-even accuracy about {break_even:.1%}\n")
    report = {}
    for fold, split in enumerate(fold_splits(data, STAGE), 1):
        model = train_direction_model(X, target, split)
        test = split.test[decided[split.test]]
        score = model.predict_proba(X[test])[:, 1] - 0.5
        print(f"fold {fold}: AUC {roc_auc_score((target[test] > 0).astype(int), score):.4f}")
        for quantile in BANDS:
            threshold = float(np.quantile(np.abs(score), quantile))
            result = outcome(score, threshold, points[test])
            print(f"   top {1 - quantile:>5.0%} by confidence: {result['trades']:>7,} trades  "
                  f"accuracy {result['accuracy']:6.2%}  ${result['per_trade']:+6.2f} a trade")
        validation = split.validation[decided[split.validation]]
        validation_score = model.predict_proba(X[validation])[:, 1] - 0.5
        best, best_net = None, -np.inf
        for quantile in BANDS:
            threshold = float(np.quantile(np.abs(validation_score), quantile))
            chosen = outcome(validation_score, threshold, points[validation])
            if chosen["trades"] >= MIN_VALIDATION_TRADES and chosen["net"] > best_net:
                best, best_net = threshold, chosen["net"]
        applied = outcome(score, best, points[test]) if best is not None else {"trades": 0}
        report[f"fold_{fold}"] = applied
        print(f"   threshold chosen on validation: {applied['trades']:,} trades  "
              f"accuracy {applied.get('accuracy', float('nan')):.2%}  "
              f"${applied.get('per_trade', float('nan')):+.2f} a trade  "
              f"net ${applied.get('net', 0):,.0f}\n")
    with open("mid_feature_economics.json", "w") as handle:
        json.dump({"median_points": median_move, "break_even": break_even, "folds": report}, handle, indent=1)


if __name__ == "__main__":
    main()
