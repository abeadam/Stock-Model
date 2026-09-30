"""Train on the objective instead of optimising a proxy and selecting afterwards.

Every configuration this project has tried optimises AUC or squared error and then picks a
threshold on validation to turn the score into trades. The threshold is the weak joint: the
best of 216 rules scored -$205 on validation and lost $36,362 on test. It is fitted on a
few hundred trades in one month and asked to hold in the next.

Four arms, two of which have no fitted threshold at all:

  binary + threshold    the incumbent. Classify move > 0, choose the cut on validation.
  weighted + threshold  the same, but each row weighted by the dollars at stake net of
                        cost, so bars that cannot pay for a trade carry no weight. Isolates
                        what the weighting alone buys inside the old pipeline.
  three class + EV      UP only where the move clears the cost, DOWN only where it clears
                        it the other way, NEITHER between. E[move] comes from the predicted
                        probabilities and the class means, and the rule is simply "trade
                        when the expected move beats the cost". **Nothing fitted on
                        validation.**
  regression + EV       predict the move in dollars directly and apply the same rule. The
                        shortest path from "train on the objective" to a decision.

Class means are taken on training rows only. Reported at mid fills and crossing, because
the mid is a resting order that filled and yesterday's measurement put that at 53-64% fill
with adverse selection worth more than the spread saved.

usage: cost_aware_training.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np

from cost_aware_targets import (CLASS_COUNT, class_means, decide, expected_dollars,
                                move_in_dollars, profit_weight, three_class_label)
from decision_trading import CROSS_FILL, MID_FILL, simulate_decisions
from hold_policy import last_bar_before_gap
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from quote_passive_entry import Quotes
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, POINT_VALUE, ROUND_TRIP_COST

HORIZON_BARS = 18
SEEDS = (11, 23, 37)
THRESHOLD_QUANTILES = (0.50, 0.80, 0.90, 0.95, 0.98, 0.99)
MIN_VALIDATION_TRADES = 100
EARLY_STOPPING_ROUNDS = 200
RESULTS_PATH_TEMPLATE = "cost_aware_training_h{horizon}.json"


def fit(model, X, labels, split, weight=None):
    usable = np.isfinite(labels)
    train = split.train[usable[split.train]]
    validation = split.validation[usable[split.validation]]
    model.fit(X[train], labels[train],
              sample_weight=None if weight is None else weight[train],
              eval_set=[(X[validation], labels[validation])],
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)])
    return model


def threshold_decisions(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(score > threshold, 1, np.where(score < -threshold, -1, 0)).astype(int)


def best_threshold(score, quotes, mid, rows, horizon) -> float:
    """The cut that made the most money on the validation block -- the step under test."""
    best, best_net = None, -np.inf
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(np.quantile(np.abs(score[rows]), quantile))
        decisions = np.zeros(len(score), int)
        decisions[rows] = threshold_decisions(score[rows], threshold)
        result = simulate_decisions(decisions, quotes, mid, horizon, POINT_VALUE, ROUND_TRIP_COST)
        if result["trades"] >= MIN_VALIDATION_TRADES and result["net"] > best_net:
            best, best_net = threshold, result["net"]
    return best if best is not None else float(np.quantile(np.abs(score[rows]), 0.99))


def report(results: dict[str, list[dict]]) -> None:
    print(f"\n{'arm':<24}{'trades':>9}{'win':>8}{'mid $/trade':>14}{'cross $/trade':>16}")
    for name, outcomes in results.items():
        trades = sum(outcome[MID_FILL]["trades"] for outcome in outcomes)
        if not trades:
            print(f"{name:<24}{0:>9}{'':>8}{'no trades':>14}")
            continue
        weighted = {mode: sum(outcome[mode]["per_trade"] * outcome[mode]["trades"]
                              for outcome in outcomes if outcome[mode]["trades"])
                    / max(sum(outcome[mode]["trades"] for outcome in outcomes), 1)
                    for mode in (MID_FILL, CROSS_FILL)}
        wins = sum(outcome[MID_FILL]["win_rate"] * outcome[MID_FILL]["trades"]
                   for outcome in outcomes if outcome[MID_FILL]["trades"]) / trades
        print(f"{name:<24}{trades:>9,}{wins:>8.1%}{weighted[MID_FILL]:>14.2f}"
              f"{weighted[CROSS_FILL]:>16.2f}")


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    target = average_target(mid, epochs, horizon)
    dollars = move_in_dollars(target, mid, POINT_VALUE)
    label_binary = np.where(np.isfinite(target) & (target != 0), (target > 0).astype(float), np.nan)
    label_three = three_class_label(dollars, ROUND_TRIP_COST)
    weight = profit_weight(dollars, ROUND_TRIP_COST)
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    quotes = Quotes(bid=data["bid"].to_numpy(float), ask=data["ask"].to_numpy(float),
                    high=data["High"].to_numpy(float), low=data["Low"].to_numpy(float),
                    is_last=last_bar_before_gap(epochs))
    splits = purged_fold_splits(data["time"], horizon)
    tradeable = np.isfinite(label_three) & (label_three != 1)
    print(f"horizon {horizon}, purge {horizon}, ${ROUND_TRIP_COST:.2f} a round trip; "
          f"{tradeable.mean():.1%} of bars move more than the cost; "
          f"{len(SEEDS)} seeds x {len(splits)} folds ({time.time() - started:.0f}s to build)\n",
          flush=True)

    results: dict[str, list[dict]] = {name: [] for name in
                                      ("binary + threshold", "weighted + threshold",
                                       "three class + EV", "regression + EV")}
    for fold_number, split in enumerate(splits, start=1):
        means = class_means(dollars[split.train], label_three[split.train])
        for seed in SEEDS:
            params = {**LGBM_PARAMS, "random_state": seed}
            scores = {}
            binary = fit(lgb.LGBMClassifier(**params), X, label_binary, split)
            scores["binary + threshold"] = binary.predict_proba(X)[:, 1] - 0.5
            weighted = fit(lgb.LGBMClassifier(**params), X, label_binary, split, weight)
            scores["weighted + threshold"] = weighted.predict_proba(X)[:, 1] - 0.5

            decisions = {}
            for name, score in scores.items():
                cut = best_threshold(score, quotes, mid, split.validation, horizon)
                chosen = np.zeros(len(X), int)
                chosen[split.test] = threshold_decisions(score[split.test], cut)
                decisions[name] = chosen

            multiclass = fit(lgb.LGBMClassifier(**{**params, "objective": "multiclass",
                                                   "num_class": CLASS_COUNT}),
                             X, label_three, split)
            expected = expected_dollars(multiclass.predict_proba(X), means)
            chosen = np.zeros(len(X), int)
            chosen[split.test] = decide(expected[split.test], ROUND_TRIP_COST)
            decisions["three class + EV"] = chosen

            regressor = fit(lgb.LGBMRegressor(**{**params, "objective": "regression"}),
                            X, dollars, split)
            predicted = regressor.predict(X)
            chosen = np.zeros(len(X), int)
            chosen[split.test] = decide(predicted[split.test], ROUND_TRIP_COST)
            decisions["regression + EV"] = chosen

            for name, chosen in decisions.items():
                results[name].append({mode: simulate_decisions(chosen, quotes, mid, horizon,
                                                               POINT_VALUE, ROUND_TRIP_COST, mode)
                                      for mode in (MID_FILL, CROSS_FILL)})
        print(f"fold {fold_number} done ({time.time() - started:.0f}s): " + "  ".join(
            f"{name.split()[0]} {results[name][-1][MID_FILL]['trades']}" for name in results),
              flush=True)

    report(results)
    with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
        json.dump({name: outcomes for name, outcomes in results.items()}, handle, indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
