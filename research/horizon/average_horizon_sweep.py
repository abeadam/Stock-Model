"""Averaging over 1 to 20 bars, plain and divided by volume.

Two things move in opposite directions as the averaging window lengthens. The move being
predicted grows, so the $5 round trip is a smaller share of it and the accuracy needed to
break even falls. But the horizon also reaches further into the future, where less is
knowable. Somewhere between those is the best place to stand, and this finds it.

Three models per horizon, all on the holdout-validated feature set:
  classifier    on the sign of the average move -- what passed the holdout at five bars.
                Dividing by volume cannot change a sign, so this arm is the same for both
                target versions and is printed once.
  regressor     on the average move itself
  regressor     on the average move divided by the volume of the bars being predicted

Break-even is computed from each horizon's own median move, since that is what a trade
collects. Regular hours, clean window, dev folds; the holdout has been spent.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from mid_feature_economics import BANDS, MIN_VALIDATION_TRADES, outcome
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_scaled_target import forward_volume
from volume_target_test import fit_regressor

FIRST_BAR, LAST_BAR = 1, 20          # overridden from the command line
PERCENT = 100.0
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
STAGE = "dev"


def best_threshold(score, points) -> float | None:
    best, best_net = None, -np.inf
    for quantile in BANDS:
        threshold = float(np.quantile(np.abs(score), quantile))
        result = outcome(score, threshold, points, )
        if result["trades"] >= MIN_VALIDATION_TRADES and result["net"] > best_net:
            best, best_net = threshold, result["net"]
    return best


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    volume = data["Volume"].to_numpy(float)
    columns = list(FEATURES) + MID_FEATURES
    X = data[columns].to_numpy(np.float32)
    splits = fold_splits(data, STAGE)
    print(f"{len(data):,} regular-hours bars, {len(columns)} features\n")
    print(f"{'bars':>5}{'move pt':>9}{'need':>7}{'AUC sign':>10}{'AUC reg':>9}{'AUC /vol':>10}"
          f"{'best $/trade':>14}{'accuracy':>10}")
    rows = []
    for horizon in (range(FIRST_BAR, LAST_BAR + 1) if LAST_BAR - FIRST_BAR < 30
                    else (40, 50, 60, 90, 120)):
        raw = average_target(mid, seconds, horizon)
        scaled = raw / forward_volume(volume, horizon)
        points = raw / PERCENT * mid
        decided = np.isfinite(raw) & (raw != 0)
        median_move = float(np.nanmedian(np.abs(points)))
        break_even = 0.5 * (1 + ROUND_TRIP_COST / (median_move * POINT_VALUE))
        scores = {"sign": [], "regressor": [], "scaled": []}
        traded = []
        for split in splits:
            test = split.test[decided[split.test]]
            up = (raw[test] > 0).astype(int)
            classifier = train_direction_model(X, raw, split)
            sign_score = classifier.predict_proba(X[test])[:, 1] - 0.5
            scores["sign"].append(roc_auc_score(up, sign_score))
            for name, target in (("regressor", raw), ("scaled", scaled)):
                model = fit_regressor(X, target, split)
                scores[name].append(roc_auc_score(up, model.predict(X[test])))
            validation = split.validation[decided[split.validation]]
            threshold = best_threshold(classifier.predict_proba(X[validation])[:, 1] - 0.5,
                                       points[validation])
            traded.append(outcome(sign_score, threshold, points[test]) if threshold else {"trades": 0})
        per_trade = float(np.mean([t.get("per_trade", np.nan) for t in traded]))
        accuracy = float(np.mean([t.get("accuracy", np.nan) for t in traded]))
        rows.append({"bars": horizon, "median_points": median_move, "break_even": break_even,
                     "auc": {k: float(np.mean(v)) for k, v in scores.items()},
                     "per_trade": per_trade, "accuracy": accuracy})
        print(f"{horizon:>5}{median_move:>9.3f}{break_even:>7.1%}{np.mean(scores['sign']):>10.4f}"
              f"{np.mean(scores['regressor']):>9.4f}{np.mean(scores['scaled']):>10.4f}"
              f"{per_trade:>+14.2f}{accuracy:>10.2%}", flush=True)
    with open(f"average_horizon_sweep_{FIRST_BAR}_{LAST_BAR}.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        FIRST_BAR, LAST_BAR = int(sys.argv[1]), int(sys.argv[2])
    main()
