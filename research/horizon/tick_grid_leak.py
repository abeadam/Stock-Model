"""Do the wick ratios leak the price level, and does fixing them help?

high_close_ratio is (High - Close) / Close and low_close_ratio is (Close - Low) / Close.
High, Low and Close all sit on the 0.25 tick grid, so each ratio is a whole number of
ticks divided by the price: its set of possible values is a grid whose spacing is
0.25/Close and therefore drifts as the market moves. A tree splitting on such a feature
is partly splitting on the calendar.

The fix keeps the intended quantity and drops the price dependence: measure the wick in
ticks. Measured here three ways -- how well the month can be identified, and what the
direction model scores with each version.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from rotation_test import TARGET_BARS, fold_splits, score_raw
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES

TICK = 0.25
WICK_RATIOS = ["high_close_ratio", "low_close_ratio"]
DRIFT_PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05, n_estimators=200,
                    min_child_samples=200, n_jobs=-1, verbose=-1, random_state=7)


def month_identification_auc(data, columns) -> float:
    """How well a model can tell a June bar from a September bar using these columns."""
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
    data = load_cut()
    close = data["Close"].to_numpy()
    # The same wicks in ticks: a whole number that means the same thing at any price.
    data = data.assign(high_close_ticks=data["high_close_ratio"].to_numpy() * close / TICK,
                       low_close_ticks=data["low_close_ratio"].to_numpy() * close / TICK)
    in_ticks = [name for name in FEATURES if name not in WICK_RATIOS] + ["high_close_ticks", "low_close_ticks"]
    without = [name for name in FEATURES if name not in WICK_RATIOS]

    print("How well can a model tell June from September?")
    for label, columns in (("all 44 as they are", FEATURES), ("wick ratios removed", without),
                           ("wicks measured in ticks", in_ticks)):
        print(f"   {label:<26}AUC {month_identification_auc(data, columns):.4f}", flush=True)

    print("\nDirection model, dev folds:")
    move = mid_move(data, TARGET_BARS)
    results = {}
    for label, columns in (("all 44 as they are", FEATURES), ("wick ratios removed", without),
                           ("wicks measured in ticks", in_ticks)):
        mean, folds = score_raw(data[columns].to_numpy(np.float32), move, data, "dev")
        results[label] = {"mean": mean, "folds": folds}
        print(f"   {label:<26}AUC {mean:.4f}  folds {folds}", flush=True)
    with open("tick_grid_leak.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
