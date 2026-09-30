"""The combination rules on the holdout months, both feature sets, both live horizons.

Nothing here won both dev folds, so this is a check rather than a test of a candidate.
Reported per month and per configuration, never averaged across folds.

  feature sets  68  = core 44 + mid microstructure
                153 = plus the returns of the added symbols
  horizons      5 and 12 bars -- the two where direction still had dev skill
  rules         confidence (strongest direction scores) and expected (|direction| *
                predicted range), each taking the strongest 1%
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from direction_times_volatility import settle, take
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_target_test import fit_regressor

HORIZONS = (5, 12)
SHARE = 0.01
PERCENT = 100.0
MONTHS = ("August", "September")


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    sets = {68: list(FEATURES) + MID_FEATURES,
            153: list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES}
    splits = fold_splits(data, "holdout")
    results = {}
    print(f"{'features':>9}{'bars':>6}{'rule':>12}{'month':>11}{'trades':>8}{'accuracy':>10}"
          f"{'move':>7}{'need':>7}{'$/trade':>10}{'AUC':>9}")
    for count, columns in sets.items():
        X = data[columns].to_numpy(np.float32)
        for horizon in HORIZONS:
            direction_target = average_target(mid, seconds, horizon)
            range_target = forward_target(prices, horizon, "range")
            points = direction_target / PERCENT * mid
            decided = np.isfinite(direction_target) & (direction_target != 0)
            for month, split in zip(MONTHS, splits):
                direction_model = train_direction_model(X, direction_target, split)
                range_model = fit_regressor(X, range_target, split)
                test = split.test[decided[split.test]]
                score = direction_model.predict_proba(X[test])[:, 1] - 0.5
                predicted_range = range_model.predict(X[test])
                auc = roc_auc_score((direction_target[test] > 0).astype(int), score)
                for name, taken in (("confidence", take(score, SHARE)),
                                    ("expected", take(score * np.maximum(predicted_range, 0), SHARE))):
                    result = settle(score, taken, points[test])
                    result["per_trade"] = result["total"] / max(result["trades"], 1)
                    result["auc"] = auc
                    results[f"{count}|{horizon}|{name}|{month}"] = result
                    print(f"{count:>9}{horizon:>6}{name:>12}{month:>11}{result['trades']:>8,}"
                          f"{result['accuracy']:>9.2%}{result['median_move']:>7.3f}"
                          f"{result['break_even']:>7.1%}{result['per_trade']:>+10.2f}{auc:>9.4f}",
                          flush=True)
    with open("holdout_combination_grid.json", "w") as handle:
        json.dump(results, handle, indent=1)
    wins = {key: value for key, value in results.items() if value["per_trade"] > 0}
    print(f"\n{len(wins)} of {len(results)} configurations positive")
    for key in wins:
        print(f"   {key}: {wins[key]['per_trade']:+.2f} on {wins[key]['trades']:,} trades")


if __name__ == "__main__":
    main()
