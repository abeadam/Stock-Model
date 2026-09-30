"""Does a LightGBM objective that maximises P&L beat one that maximises AUC?

The earlier cost-aware run put the cost in the label, the sample weight and the decision
rule. It never put it in the **loss**, which was the first of the two suggestions. This
does: a custom objective whose gradient is the derivative of minus expected profit, with
the toll inside, and an early-stopping metric that watches dollars per row rather than a
proxy.

Three arms, the first two being the best of the previous run:

  binary + threshold    the incumbent: classify move > 0, cut chosen on validation
  weighted + threshold  the same with rows weighted by dollars at stake net of cost, which
                        improved the incumbent by $1.32 a trade
  pnl objective         position = tanh(raw score), loss = -(a*m - cost*|a|) + penalty*a^2,
                        trade wherever the fitted position is non-zero. No cut is chosen
                        on validation: the flat region comes out of the loss.

Reported at mid fills and crossing, trade-weighted across folds and seeds.

usage: pnl_objective_test.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np

from cost_aware_targets import move_in_dollars, profit_weight
from cost_aware_training import best_threshold, fit, report, threshold_decisions
from decision_trading import CROSS_FILL, MID_FILL, simulate_decisions
from hold_policy import last_bar_before_gap
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from pnl_objective import POSITION_PENALTY, decide, make_evaluation, make_objective, positions
from purged_folds import purged_fold_splits
from quote_passive_entry import Quotes
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, POINT_VALUE, ROUND_TRIP_COST

HORIZON_BARS = 18
SEEDS = (11, 23, 37)
BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 200
RESULTS_PATH_TEMPLATE = "pnl_objective_test_h{horizon}.json"
BOOSTER_KEYS = ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction",
                "bagging_fraction", "bagging_freq", "lambda_l2")


def train_pnl_booster(X, dollars, split, seed):
    """A booster whose loss is minus expected profit, early-stopped on profit per row."""
    params = {key: LGBM_PARAMS[key] for key in BOOSTER_KEYS}
    params.update({"objective": make_objective(ROUND_TRIP_COST, POSITION_PENALTY),
                   "verbose": -1, "num_threads": -1, "seed": seed})
    train = lgb.Dataset(X[split.train], label=dollars[split.train], free_raw_data=False)
    validation = lgb.Dataset(X[split.validation], label=dollars[split.validation],
                             reference=train, free_raw_data=False)
    return lgb.train(params, train, num_boost_round=BOOST_ROUNDS, valid_sets=[validation],
                     feval=make_evaluation(ROUND_TRIP_COST),
                     callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                lgb.log_evaluation(0)])


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    target = average_target(mid, epochs, horizon)
    dollars = move_in_dollars(target, mid, POINT_VALUE)
    label_binary = np.where(np.isfinite(target) & (target != 0), (target > 0).astype(float), np.nan)
    weight = profit_weight(dollars, ROUND_TRIP_COST)
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    quotes = Quotes(bid=data["bid"].to_numpy(float), ask=data["ask"].to_numpy(float),
                    high=data["High"].to_numpy(float), low=data["Low"].to_numpy(float),
                    is_last=last_bar_before_gap(epochs))
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}, ${ROUND_TRIP_COST:.2f} a round trip, "
          f"position penalty {POSITION_PENALTY}; {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    results = {name: [] for name in ("binary + threshold", "weighted + threshold", "pnl objective")}
    sizing = []
    for fold_number, split in enumerate(splits, start=1):
        for seed in SEEDS:
            params = {**LGBM_PARAMS, "random_state": seed}
            decisions = {}
            for name, sample_weight in (("binary + threshold", None),
                                        ("weighted + threshold", weight)):
                model = fit(lgb.LGBMClassifier(**params), X, label_binary, split, sample_weight)
                score = model.predict_proba(X)[:, 1] - 0.5
                cut = best_threshold(score, quotes, mid, split.validation, horizon)
                chosen = np.zeros(len(X), int)
                chosen[split.test] = threshold_decisions(score[split.test], cut)
                decisions[name] = chosen

            booster = train_pnl_booster(X, dollars, split, seed)
            raw = booster.predict(X[split.test], raw_score=True)
            position = positions(raw)
            chosen = np.zeros(len(X), int)
            chosen[split.test] = decide(raw)
            decisions["pnl objective"] = chosen
            sizing.append({"fold": fold_number, "seed": seed,
                           "mean_abs_position": float(np.abs(position).mean()),
                           "flat_share": float((np.abs(position) < 0.01).mean()),
                           "trees": int(booster.best_iteration or booster.num_trees())})

            for name, chosen in decisions.items():
                results[name].append({mode: simulate_decisions(chosen, quotes, mid, horizon,
                                                               POINT_VALUE, ROUND_TRIP_COST, mode)
                                      for mode in (MID_FILL, CROSS_FILL)})
        latest = sizing[-1]
        print(f"fold {fold_number} done ({time.time() - started:.0f}s): pnl booster "
              f"{latest['trees']} trees, |position| {latest['mean_abs_position']:.3f}, "
              f"flat {latest['flat_share']:.1%}", flush=True)

    report(results)
    with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
        json.dump({"results": results, "sizing": sizing}, handle, indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
