"""The average-of-next-five-bars target: does it hold up, and is it worth anything?

An average is not a price you can trade at by accident, but it is one you can trade at on
purpose: enter now and leave in equal parts over the next five bars, and what you collect
is the average move, which is exactly this target. So the economics can be measured.

Fold detail, the sign-classifier comparison, and a trade at $5 the round trip with the
entry threshold chosen on each fold's validation rows -- never on the rows being scored.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from forward_extremes import forward_target
from rotation_test import fold_splits
from segment_walk_forward import load_cut
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_scaled_target import forward_volume, regular_hours
from volume_target_test import HORIZON_BARS, STAGE, fit_regressor

POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
PERCENT = 100.0
THRESHOLD_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)


def trade(score, average_points, threshold) -> dict:
    """Take every bar whose score passes the threshold; collect the average move."""
    side = np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, 0.0))
    taken = side != 0
    if taken.sum() == 0:
        return {"trades": 0, "per_trade": np.nan, "net": 0.0, "accuracy": np.nan}
    gained = side[taken] * average_points[taken] * POINT_VALUE - ROUND_TRIP_COST
    return {"trades": int(taken.sum()), "per_trade": float(gained.mean()), "net": float(gained.sum()),
            "accuracy": float((side[taken] * average_points[taken] > 0).mean())}


def main() -> None:
    data = load_cut()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    raw = forward_target(prices, HORIZON_BARS, "average")
    scaled = raw / forward_volume(data["Volume"].to_numpy(float), HORIZON_BARS)
    average_points = raw / PERCENT * prices["close"]
    X = data[FEATURES].to_numpy(np.float32)
    decided = np.isfinite(raw) & (raw != 0)
    print(f"median |average move| {np.nanmedian(np.abs(average_points)):.3f} points "
          f"(= ${np.nanmedian(np.abs(average_points)) * POINT_VALUE:.2f} a contract)\n")

    outcome = {}
    for name, target in (("average", raw), ("average / volume", scaled)):
        aucs, trades = [], []
        for split in fold_splits(data, STAGE):
            model = fit_regressor(X, target, split)
            ranked = split.test[decided[split.test]]
            aucs.append(roc_auc_score((raw[ranked] > 0).astype(int), model.predict(X[ranked])))
            validation = split.validation[decided[split.validation]]
            validation_score = model.predict(X[validation])
            best, best_net = None, -np.inf
            for quantile in THRESHOLD_QUANTILES:
                threshold = float(np.quantile(np.abs(validation_score), quantile))
                result = trade(validation_score, average_points[validation], threshold)
                if result["trades"] > 100 and result["net"] > best_net:
                    best, best_net = threshold, result["net"]
            trades.append(trade(model.predict(X[ranked]), average_points[ranked], best))
        outcome[name] = {"auc": float(np.mean(aucs)), "fold_auc": [round(a, 4) for a in aucs],
                         "trades": trades}
        print(f"{name:<20}AUC {np.mean(aucs):.4f}  folds {[round(a, 4) for a in aucs]}")
        for fold, result in enumerate(trades, 1):
            print(f"    fold {fold}: {result['trades']:>6,} trades  accuracy {result['accuracy']:6.1%}  "
                  f"${result['per_trade']:+.2f} a trade  net ${result['net']:>10,.0f}")

    aucs = []
    for split in fold_splits(data, STAGE):
        model = train_direction_model(X, raw, split)
        ranked = split.test[decided[split.test]]
        aucs.append(roc_auc_score((raw[ranked] > 0).astype(int), model.predict_proba(X[ranked])[:, 1]))
    print(f"\n{'sign classifier':<20}AUC {np.mean(aucs):.4f}  folds {[round(a, 4) for a in aucs]}")
    with open("average_target_economics.json", "w") as handle:
        json.dump(outcome, handle, indent=1)


if __name__ == "__main__":
    main()
