"""Two ways of telling the direction model about itself.

  predicted volatility   the volatility head's forecast as an input column, so the trees
                         can interact "how much is about to happen" with every other
                         feature rather than only multiplying it in afterwards
  recent accuracy        how the model has lately been doing: hit rate, calibration drift
                         and range error over its own settled predictions

Neither has been tried. `stacked_direction_volatility.py` combined the two heads at the
*output* level -- a meta-model over finished scores -- which cannot let volatility shape
the direction model's splits. And nothing has ever fed the model its own track record.

**Both inputs are predictions, so both are leak-shaped.** Every prediction used here is
out-of-fold: inside the training block, rows are predicted by models fitted only on
earlier blocks, purged. Otherwise the direction model learns to trust a volatility
forecast that is far better on training rows than it will ever be live -- which is exactly
why `PRED_FEATURES` exists as a leak control. The feedback columns carry a second hazard:
an 18-bar call is not settled for 18 bars, so every one is lagged by the horizon.

A caution held in advance. Gating trades on predicted volatility already failed at
roughly 1:1, because the model knows direction only when little is about to happen. This
is a different construction -- volatility as an input rather than a filter -- but if the
predicted-volatility arm helps, that closeness is the first thing to check.

usage: model_self_knowledge_test.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from out_of_fold_predictions import out_of_fold, scatter
from prediction_feedback import FEEDBACK_FEATURES, add_feedback_features
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
GENERATOR_SEED = 11
PREDICTED_VOLATILITY = "sk_predicted_range"
EARLY_STOPPING_ROUNDS = 200
VALIDATION_TAIL = 0.15
RESULTS_PATH_TEMPLATE = "model_self_knowledge_h{horizon}.json"


def fit_range_model(X, target, rows, params):
    """Regressor for the forward range, early-stopped on the tail of its own training rows."""
    usable = rows[np.isfinite(target[rows])]
    cut = int(len(usable) * (1 - VALIDATION_TAIL))
    train, validation = usable[:cut], usable[cut:]
    model = lgb.LGBMRegressor(**{**params, "objective": "regression"})
    model.fit(X[train], target[train], eval_set=[(X[validation], target[validation])],
              eval_metric="l2", callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                           lgb.log_evaluation(0)])
    return model


def fit_probability_model(X, target, rows, params):
    """Classifier for the direction, early-stopped on the tail of its own training rows."""
    usable = rows[np.isfinite(target[rows]) & (target[rows] != 0)]
    cut = int(len(usable) * (1 - VALIDATION_TAIL))
    train, validation = usable[:cut], usable[cut:]
    labels = (target > 0).astype(int)
    model = lgb.LGBMClassifier(**params)
    model.fit(X[train], labels[train], eval_set=[(X[validation], labels[validation])],
              eval_metric="auc", callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                            lgb.log_evaluation(0)])
    return model


def honest_predictions(X, target, split, horizon, fit, predict):
    """One column of predictions: out-of-fold on training rows, full-fit elsewhere."""
    params = {**LGBM_PARAMS, "random_state": GENERATOR_SEED}
    inner = out_of_fold(split.train,
                        lambda history, wanted: predict(fit(X, target, history, params), X[wanted]),
                        purge_bars=horizon)
    column = scatter(inner, split.train, len(X))
    trained = fit(X, target, split.train, params)
    for rows in (split.validation, split.test):
        column[rows] = predict(trained, X[rows])
    return column


def paired_t(differences: np.ndarray) -> tuple[float, float]:
    error = differences.std(ddof=1) / np.sqrt(len(differences))
    return differences.mean(), differences.mean() / error if error > 0 else 0.0


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    X = data[base].to_numpy(np.float32)
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; {len(base)} base columns; "
          f"{len(SEEDS)} seeds x {len(splits)} folds ({time.time() - started:.0f}s to build)\n",
          flush=True)

    scores: dict[str, list[float]] = {}
    for fold_number, split in enumerate(splits, start=1):
        predicted_range = honest_predictions(X, range_target, split, horizon, fit_range_model,
                                             lambda model, rows: model.predict(rows))
        probability = honest_predictions(X, direction_target, split, horizon, fit_probability_model,
                                         lambda model, rows: model.predict_proba(rows)[:, 1])
        covered = np.isfinite(predicted_range[split.train]).mean()
        with_feedback = add_feedback_features(data, probability, direction_target,
                                              predicted_range, range_target, horizon)
        with_feedback[PREDICTED_VOLATILITY] = predicted_range
        arms = {"base": base,
                "plus predicted vol": base + [PREDICTED_VOLATILITY],
                "plus feedback": base + FEEDBACK_FEATURES,
                "plus both": base + [PREDICTED_VOLATILITY] + FEEDBACK_FEATURES}
        print(f"fold {fold_number}: {covered:.0%} of training rows have an out-of-fold "
              f"prediction ({time.time() - started:.0f}s)", flush=True)
        for name, columns in arms.items():
            matrix = with_feedback[columns].to_numpy(np.float32)
            for seed in SEEDS:
                model = train_direction_model(matrix, direction_target, split,
                                              {**LGBM_PARAMS, "random_state": seed})
                test = split.test[decided[split.test]]
                scores.setdefault(name, []).append(
                    roc_auc_score((direction_target[test] > 0).astype(int),
                                  model.predict_proba(matrix[test])[:, 1]))
            print(f"   {name:<20}{np.mean(scores[name][-len(SEEDS):]):.4f}"
                  f"   ({time.time() - started:.0f}s)", flush=True)
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump(scores, handle, indent=1)

    print(f"\n{'arm':<22}{'columns':>9}{'direction AUC':>15}{'vs base':>11}{'t':>7}")
    print(f"{'base':<22}{len(base):>9}{np.mean(scores['base']):>15.4f}")
    widths = {"plus predicted vol": len(base) + 1, "plus feedback": len(base) + len(FEEDBACK_FEATURES),
              "plus both": len(base) + 1 + len(FEEDBACK_FEATURES)}
    for name in widths:
        mean, t = paired_t(np.array(scores[name]) - np.array(scores["base"]))
        print(f"{name:<22}{widths[name]:>9}{np.mean(scores[name]):>15.4f}{mean:>+11.5f}{t:>7.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
