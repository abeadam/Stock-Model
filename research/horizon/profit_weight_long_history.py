"""Was the profit weight a better model, or a luckier threshold?

Weighting each training row by `max(|move| * 50 - cost, 0)` improved per-trade P&L from
-$5.56 to -$4.24 on the short history -- the only change all week that moved money the
right way. But that arm still chose its trading threshold on validation, and the threshold
grid was separately shown to be fragile: best-of-216 scored -$205 on validation and lost
$36,362 on test. So the gain has two possible sources, and P&L alone cannot tell them
apart:

  a better model      the weight teaches the classifier to rank bars that pay above bars
                      that do not, which would show up as higher AUC
  a luckier cut       the weight merely reshapes the score distribution so the grid
                      happens to pick a better threshold, which would not

**AUC is threshold-free, so 13 folds of it separates them.** If the weight improves
ranking month after month it was a real gain. If AUC is flat, the -$4.24 was selection
luck and nothing this week moved money.

Direction head only: the weight prices a directional call, and means nothing to a range
regressor. Delay 1 only: at delay 0 the Close-based target carries bid-ask bounce, which
took the same baseline from 0.5239 to 0.5058.

usage: profit_weight_long_history.py [HORIZON_BARS]
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from cost_aware_targets import move_in_dollars, profit_weight
from deep_ladder_both_heads import paired_t
from long_history_support import (REALIZED_FEATURES, SMOOTHED_FEATURES, base_columns, build,
                                  forward_average_target, monthly_folds)
from short_horizon_walk_forward import LGBM_PARAMS, POINT_VALUE, ROUND_TRIP_COST

HORIZON_BARS = 18
DELAY_BARS = 1
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
EARLY_STOPPING_ROUNDS = 200
RESULTS_PATH = "/Users/abeadam/dev/model/Stock-Model/scratchpad_backup/profit_weight_long_history.json"


def fit_direction(X, label, split, params, weight=None):
    """Classifier for the direction, optionally weighted by what each bar has at stake."""
    usable = np.isfinite(label)
    train = split.train[usable[split.train]]
    validation = split.validation[usable[split.validation]]
    model = lgb.LGBMClassifier(**params)
    model.fit(X[train], label[train],
              sample_weight=None if weight is None else weight[train],
              eval_set=[(X[validation], label[validation])], eval_metric="auc",
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)])
    return model


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build(include_spy=False)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    close = data["Close"].to_numpy(float)
    direction = forward_average_target(close, seconds, horizon, DELAY_BARS)
    decided = np.isfinite(direction) & (direction != 0)
    label = np.where(decided, (direction > 0).astype(float), np.nan)
    weight = profit_weight(move_in_dollars(direction, close, POINT_VALUE), ROUND_TRIP_COST)
    columns = base_columns(include_spy=False) + REALIZED_FEATURES + SMOOTHED_FEATURES
    X = data[columns].to_numpy(np.float32)
    folds = monthly_folds(data["session_day"].to_numpy(), horizon)
    carrying = float((weight > 0).mean())
    print(f"{len(columns)} columns, delay {DELAY_BARS}, {len(folds)} monthly folds; "
          f"{carrying:.1%} of bars carry any weight, mean weight ${weight[weight > 0].mean():.2f} "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    scores: dict[str, dict[str, list[float]]] = {"unweighted": {}, "profit weighted": {}}
    for fold in folds:
        for name, sample_weight in (("unweighted", None), ("profit weighted", weight)):
            for seed in SEEDS:
                model = fit_direction(X, label, fold.split, {**LGBM_PARAMS, "random_state": seed},
                                      sample_weight)
                test = fold.split.test[decided[fold.split.test]]
                scores[name].setdefault(fold.month, []).append(
                    roc_auc_score((direction[test] > 0).astype(int),
                                  model.predict_proba(X[test])[:, 1]))
        print(f"   {fold.month}: unweighted {np.mean(scores['unweighted'][fold.month]):.4f}  "
              f"weighted {np.mean(scores['profit weighted'][fold.month]):.4f}   "
              f"({time.time() - started:.0f}s)", flush=True)
        with open(RESULTS_PATH, "w") as handle:
            json.dump(scores, handle, indent=1)

    months = [fold.month for fold in folds]
    plain = np.array([np.mean(scores["unweighted"][month]) for month in months])
    weighted = np.array([np.mean(scores["profit weighted"][month]) for month in months])
    gain, t = paired_t(weighted - plain)
    print(f"\nunweighted   {plain.mean():.4f}")
    print(f"weighted     {weighted.mean():.4f}   {gain:+.5f}   t {t:.1f}   "
          f"{int((weighted > plain).sum())}/{len(months)} folds up")
    verdict = ("the weight ranks better, so the P&L gain was a real modelling improvement"
               if t > 2 else
               "the weight does not improve ranking, so the -$4.24 was threshold selection")
    print(f"\n{verdict}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
