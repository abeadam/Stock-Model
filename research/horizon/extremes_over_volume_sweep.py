"""Predicting the highest and lowest percent change, each divided by volume.

A single extreme carries no direction -- the high is usually above the close and the low
below it -- so the pair is read the way production reads its two sub-models: predict how
far price can run up and how far down, and let the **sum** say which side has more room.
A positive sum means the expected upside exceeds the expected downside.

Four regressions per horizon: highest and lowest, each plain and each divided by the
volume of the bars being predicted. Judged by how well each target is explained (R2) and
by how well the resulting sum ranks the direction the mid actually took, which is what a
trade needs. Regular hours, dev folds, fitted inside each fold.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from forward_extremes import forward_target
from mid_feature_economics import BANDS, MIN_VALIDATION_TRADES, outcome
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES
from volume_scaled_target import forward_volume
from volume_target_test import fit_regressor

HORIZONS = (1, 2, 3, 5, 8, 12, 15, 20, 26, 30)
PERCENT = 100.0
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
STAGE = "dev"


def best_threshold(score, points):
    best, best_net = None, -np.inf
    for quantile in BANDS:
        threshold = float(np.quantile(np.abs(score), quantile))
        result = outcome(score, threshold, points)
        if result["trades"] >= MIN_VALIDATION_TRADES and result["net"] > best_net:
            best, best_net = threshold, result["net"]
    return best


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    volume_raw = data["Volume"].to_numpy(float)
    columns = list(FEATURES) + MID_FEATURES
    X = data[columns].to_numpy(np.float32)
    splits = fold_splits(data, STAGE)
    print(f"{len(data):,} regular-hours bars, {len(columns)} features; the sum of the two "
          f"predictions is the directional signal\n")
    print(f"{'bars':>5}{'R2 high':>9}{'R2 high/v':>11}{'R2 low':>9}{'R2 low/v':>10}"
          f"{'AUC pair':>10}{'AUC pair/v':>12}{'$/trade':>10}{'$/trade v':>11}")
    rows = []
    for horizon in HORIZONS:
        volume = forward_volume(volume_raw, horizon)
        targets = {}
        for kind in ("highest", "lowest"):
            raw = forward_target(prices, horizon, kind)
            targets[kind] = raw
            targets[f"{kind}_v"] = raw / volume
        direction = average_target(mid, seconds, horizon)
        points = direction / PERCENT * mid
        decided = np.isfinite(direction) & (direction != 0)
        scores = {name: [] for name in ("r2_highest", "r2_highest_v", "r2_lowest", "r2_lowest_v",
                                        "auc_pair", "auc_pair_v")}
        money = {"plain": [], "scaled": []}
        for split in splits:
            test = split.test[decided[split.test]]
            validation = split.validation[decided[split.validation]]
            up = (direction[test] > 0).astype(int)
            predictions = {}
            for name, target in targets.items():
                model = fit_regressor(X, target, split)
                known = split.test[np.isfinite(target[split.test])]
                scores[f"r2_{name.replace('_v', '') if name.endswith('_v') else name}"
                       + ("_v" if name.endswith("_v") else "")].append(
                    r2_score(target[known], model.predict(X[known])))
                predictions[name] = {"test": model.predict(X[test]),
                                     "validation": model.predict(X[validation])}
            for label, suffix, key in (("plain", "", "auc_pair"), ("scaled", "_v", "auc_pair_v")):
                pair_test = predictions[f"highest{suffix}"]["test"] + predictions[f"lowest{suffix}"]["test"]
                pair_validation = (predictions[f"highest{suffix}"]["validation"]
                                   + predictions[f"lowest{suffix}"]["validation"])
                scores[key].append(roc_auc_score(up, pair_test))
                threshold = best_threshold(pair_validation, points[validation])
                money[label].append(outcome(pair_test, threshold, points[test]) if threshold
                                    else {"trades": 0, "per_trade": np.nan})
        row = {"bars": horizon, **{k: float(np.mean(v)) for k, v in scores.items()},
               "per_trade": float(np.mean([m.get("per_trade", np.nan) for m in money["plain"]])),
               "per_trade_scaled": float(np.mean([m.get("per_trade", np.nan) for m in money["scaled"]]))}
        rows.append(row)
        print(f"{horizon:>5}{row['r2_highest']:>9.4f}{row['r2_highest_v']:>11.4f}"
              f"{row['r2_lowest']:>9.4f}{row['r2_lowest_v']:>10.4f}{row['auc_pair']:>10.4f}"
              f"{row['auc_pair_v']:>12.4f}{row['per_trade']:>+10.2f}{row['per_trade_scaled']:>+11.2f}",
              flush=True)
    with open("extremes_over_volume_sweep.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()
