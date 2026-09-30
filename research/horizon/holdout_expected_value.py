"""The holdout test for the expected-value rule: August and September.

Frozen before running, no re-tuning of any part:
  horizon    21 bars
  direction  classifier on the sign of the average mid move
  size       regressor on the forward high-to-low range
  score      |direction| * max(predicted range, 0)
  selection  the strongest 5%, side taken from the direction model
  control    the same 5% chosen by direction confidence alone, to show whether the
             range model is what is doing the work

Pass condition: positive per-trade P&L in BOTH months. Dev gave +$2.68 and +$3.80.
Costs are one round trip of $5: entering once and leaving at a single arbitrary moment
inside the window collects the average move in expectation, which is the target. Exiting
in tranches would smooth the result and cost several round trips instead.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from direction_times_volatility import POINT_VALUE, settle, take
from forward_extremes import forward_target
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_target_test import fit_regressor

HORIZON = 21
SHARE = 0.05
PERCENT = 100.0
MONTHS = ("August", "September")
DEV = {"expected value": (2.68, 3.80), "confidence only": (-6.24, -6.24)}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    X = data[list(FEATURES) + MID_FEATURES].to_numpy(np.float32)
    direction_target = average_target(mid, seconds, HORIZON)
    range_target = forward_target(prices, HORIZON, "range")
    points = direction_target / PERCENT * mid
    decided = np.isfinite(direction_target) & (direction_target != 0)

    outcomes = {"expected value": [], "confidence only": []}
    print(f"{'rule':<18}{'month':>11}{'trades':>9}{'accuracy':>10}{'move':>8}{'need':>7}"
          f"{'$/trade':>10}{'net':>12}")
    for month, split in zip(MONTHS, fold_splits(data, "holdout")):
        direction_model = train_direction_model(X, direction_target, split)
        range_model = fit_regressor(X, range_target, split)
        test = split.test[decided[split.test]]
        score = direction_model.predict_proba(X[test])[:, 1] - 0.5
        predicted_range = range_model.predict(X[test])
        auc = roc_auc_score((direction_target[test] > 0).astype(int), score)
        chosen = {"expected value": take(score * np.maximum(predicted_range, 0), SHARE),
                  "confidence only": take(score, SHARE)}
        for name, taken in chosen.items():
            result = settle(score, taken, points[test])
            result["per_trade"] = result["total"] / result["trades"]
            outcomes[name].append(result)
            print(f"{name:<18}{month:>11}{result['trades']:>9,}{result['accuracy']:>9.2%}"
                  f"{result['median_move']:>8.3f}{result['break_even']:>7.1%}"
                  f"{result['per_trade']:>+10.2f}{result['total']:>+12,.0f}")
        print(f"{'':<18}{'(AUC ' + f'{auc:.4f})':>11}")
    print()
    for name, parts in outcomes.items():
        passed = all(part["per_trade"] > 0 for part in parts)
        dev = DEV[name]
        print(f"{name:<18}dev {dev[0]:+.2f} / {dev[1]:+.2f}   holdout "
              f"{parts[0]['per_trade']:+.2f} / {parts[1]['per_trade']:+.2f}   "
              f"{'PASS' if passed else 'FAIL'}")
    with open("holdout_expected_value.json", "w") as handle:
        json.dump({name: parts for name, parts in outcomes.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
