"""Combining what predicts direction with what predicts size.

Direction is barely predictable (AUC ~0.52) and range is very predictable (R2 ~0.47), and
the two are different models of different targets, so bagging or boosting -- which combine
many fits of the *same* target -- is not the tool. What joins them is the arithmetic of the
cost: a trade needs

    accuracy > 0.5 + cost / (2 * move * point value)

so the accuracy required falls as the move grows. At the median 0.45-point move that is
61%; on a 1.5-point move it is 53.3%, which is inside what the direction model achieves.
The range model can say in advance which bars are the large ones.

Three ways of choosing when to trade, each taking the same number of trades so the
comparison is fair:
  confidence   the strongest direction scores -- what every earlier test did
  expected     |direction score| * predicted range: rank by expected gain, not certainty
  gated        only bars whose predicted range is in the top band, then the strongest
               direction scores among those

P&L is pooled across folds and weighted by trade, never averaged per fold: a fold with
twenty trades must not weigh the same as one with twenty thousand.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from forward_extremes import forward_target
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model
from volume_target_test import fit_regressor

HORIZONS = (5, 15, 21)
PERCENT = 100.0
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
STAGE = "dev"
TRADE_SHARES = (0.20, 0.05, 0.01)
RANGE_GATE = 0.70              # keep the largest 30% of predicted ranges


def take(score: np.ndarray, share: float) -> np.ndarray:
    """The `share` strongest scores, as a mask."""
    if share >= 1.0:
        return np.ones(len(score), bool)
    return np.abs(score) >= np.quantile(np.abs(score), 1 - share)


def settle(direction_score, taken, points) -> dict:
    side = np.sign(direction_score[taken])
    gained = side * points[taken] * POINT_VALUE - ROUND_TRIP_COST
    move = float(np.median(np.abs(points[taken])))
    return {"trades": int(taken.sum()), "accuracy": float((side * points[taken] > 0).mean()),
            "median_move": move, "break_even": 0.5 * (1 + ROUND_TRIP_COST / (move * POINT_VALUE)),
            "total": float(gained.sum())}


def pooled(parts: list[dict]) -> dict:
    trades = sum(part["trades"] for part in parts)
    if trades == 0:
        return {"trades": 0}
    weighted = lambda key: sum(part[key] * part["trades"] for part in parts) / trades
    return {"trades": trades, "accuracy": weighted("accuracy"), "median_move": weighted("median_move"),
            "break_even": weighted("break_even"),
            "per_trade": sum(part["total"] for part in parts) / trades}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    X = data[list(FEATURES) + MID_FEATURES].to_numpy(np.float32)
    splits = fold_splits(data, STAGE)
    results = {}
    for horizon in HORIZONS:
        direction_target = average_target(mid, seconds, horizon)
        range_target = forward_target(prices, horizon, "range")
        points = direction_target / PERCENT * mid
        decided = np.isfinite(direction_target) & (direction_target != 0)
        per_rule = {name: {share: [] for share in TRADE_SHARES}
                    for name in ("confidence", "expected", "gated")}
        aucs, range_r2 = [], []
        for split in splits:
            direction_model = train_direction_model(X, direction_target, split)
            range_model = fit_regressor(X, range_target, split)
            test = split.test[decided[split.test]]
            score = direction_model.predict_proba(X[test])[:, 1] - 0.5
            predicted_range = range_model.predict(X[test])
            aucs.append(roc_auc_score((direction_target[test] > 0).astype(int), score))
            known = split.test[np.isfinite(range_target[split.test])]
            range_r2.append(r2_score(range_target[known], range_model.predict(X[known])))
            wide = predicted_range >= np.quantile(predicted_range, RANGE_GATE)
            for share in TRADE_SHARES:
                per_rule["confidence"][share].append(settle(score, take(score, share), points[test]))
                per_rule["expected"][share].append(
                    settle(score, take(score * np.maximum(predicted_range, 0), share), points[test]))
                inside = np.where(wide, score, 0.0)
                per_rule["gated"][share].append(
                    settle(score, take(inside, share) & wide, points[test]))
        print(f"\n{horizon} bars: direction AUC {np.mean(aucs):.4f}, range R2 {np.mean(range_r2):.4f}")
        print(f"   {'rule':<12}{'share':>7}{'trades':>9}{'accuracy':>10}{'move pt':>9}"
              f"{'need':>7}{'$/trade':>10}")
        for name, by_share in per_rule.items():
            for share, parts in by_share.items():
                summary = pooled(parts)
                results[f"{horizon}|{name}|{share}"] = summary
                print(f"   {name:<12}{share:>6.0%}{summary['trades']:>9,}{summary['accuracy']:>9.2%}"
                      f"{summary['median_move']:>9.3f}{summary['break_even']:>7.1%}"
                      f"{summary['per_trade']:>+10.2f}", flush=True)
    with open("direction_times_volatility.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
