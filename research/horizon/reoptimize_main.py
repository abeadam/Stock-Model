"""Re-optimize the main (gradient) model's LightGBM settings with honest walk-forward.

Two targets:
  production  sign of target_avg5_close_pct (mean of the next 5 close changes), what
              the live model is trained on today
  tradeable   sign of the 5-bar close move starting one bar later, which keeps the
              decision bar's own bid-ask bounce out of the label

Search: random settings scored by mean test AUC over development folds 1-4, each
fold trained on everything before it (early stopping on a validation tail). To keep
the search affordable it trains on every third bar; the production settings and the
best one found are then retrained on every bar and scored once on holdout folds 5-6
(development folds were already scored in the search).

usage: reoptimize_main.py search TARGET FEATURE_SET | reoptimize_main.py confirm TARGET FEATURE_SET
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from short_horizon_walk_forward import (DEV_FOLDS, FEATURE_SETS, HOLDOUT_FOLDS, MODEL_INPUT, direction_auc,
                                        fold_split, forward_move, load)

PRODUCTION_TARGET = "target_avg5_close_pct"
TARGET_BARS = 5
SEARCH_ROW_STRIDE = 3
SEARCH_CONFIGS = 24
EARLY_STOPPING_ROUNDS = 100
MAX_TREES = 3000
RANDOM_SEED = 2026
# The live model's settings (train_predictor.LGBM_PARAMS): 500 trees, no early stopping.
PRODUCTION_PARAMS = dict(objective="binary", n_estimators=500, learning_rate=0.01, max_depth=6, num_leaves=63,
                         feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.05,
                         lambda_l2=0.05, min_data_in_leaf=50)
SEARCH_SPACE = dict(num_leaves=[7, 15, 31, 63, 127], max_depth=[-1, 4, 6, 8], learning_rate=[0.01, 0.03, 0.1],
                    min_data_in_leaf=[200, 1000, 5000, 20000], feature_fraction=[0.4, 0.6, 0.8, 1.0],
                    bagging_fraction=[0.5, 0.7, 0.9], lambda_l2=[0.0, 1.0, 10.0, 100.0])
COMMON_PARAMS = dict(n_jobs=-1, verbose=-1, random_state=11)


def with_production_target(data: pd.DataFrame) -> pd.DataFrame:
    target = pd.read_csv(MODEL_INPUT, usecols=["DateTime", PRODUCTION_TARGET]).drop_duplicates("DateTime")
    target["time"] = pd.to_datetime(target["DateTime"])
    return data.merge(target[["time", PRODUCTION_TARGET]], on="time", how="left")


def labels(data: pd.DataFrame, target: str) -> np.ndarray:
    """A signed value per row whose sign is the class; NaN where unknown."""
    if target == "production":
        return data[PRODUCTION_TARGET].to_numpy()
    return forward_move(data, TARGET_BARS, delay=1)


def random_configs(count: int, seed: int) -> list[dict]:
    generator = np.random.default_rng(seed)
    return [{name: values[generator.integers(len(values))] for name, values in SEARCH_SPACE.items()}
            | {"bagging_freq": 1, "objective": "binary", "n_estimators": MAX_TREES} for _ in range(count)]


def fit_and_score(X, move, times, fold: int, params: dict, stride: int, early_stopping: bool) -> tuple[float, int]:
    split = fold_split(times, fold, purge_bars=TARGET_BARS + 1)
    usable = ~np.isnan(move) & (move != 0)
    train = split.train[usable[split.train]][::stride]
    validation = split.validation[usable[split.validation]]
    model = lgb.LGBMClassifier(**params, **COMMON_PARAMS)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if early_stopping else []
    model.fit(X[train], (move[train] > 0).astype(int), eval_set=[(X[validation], (move[validation] > 0).astype(int))],
              eval_metric="auc", callbacks=callbacks + [lgb.log_evaluation(0)])
    probability = model.predict_proba(X[split.test])[:, 1]
    trees = model.best_iteration_ if early_stopping and model.best_iteration_ else params["n_estimators"]
    return direction_auc(probability, move[split.test]), int(trees)


def score_config(X, move, times, params, folds, stride, early_stopping) -> dict:
    results = [fit_and_score(X, move, times, fold, params, stride, early_stopping) for fold in folds]
    aucs = [auc for auc, _ in results]
    return {"mean_auc": float(np.mean(aucs)), "fold_aucs": [round(a, 4) for a in aucs],
            "trees": [trees for _, trees in results]}


def prepare(target: str, feature_set: str):
    data = load()
    if target == "production":
        data = with_production_target(data)
    return data[FEATURE_SETS[feature_set]].to_numpy(np.float32), labels(data, target), data["time"]


def search(target: str, feature_set: str) -> None:
    started = time.time()
    X, move, times = prepare(target, feature_set)
    candidates = [("production settings", PRODUCTION_PARAMS, False),
                  ("production + early stopping", PRODUCTION_PARAMS | {"n_estimators": MAX_TREES}, True)]
    candidates += [(f"random {index}", params, True) for index, params in enumerate(random_configs(SEARCH_CONFIGS, RANDOM_SEED))]
    rows = []
    for name, params, early_stopping in candidates:
        result = score_config(X, move, times, params, DEV_FOLDS, SEARCH_ROW_STRIDE, early_stopping)
        rows.append({"name": name, "params": params, "early_stopping": early_stopping, **result})
        print(f"{name:<30} mean AUC {result['mean_auc']:.4f}  folds {result['fold_aucs']}  trees {result['trees']}  "
              f"({time.time() - started:.0f}s)", flush=True)
    with open(f"reoptimize_main_{target}_{feature_set}.json", "w") as handle:
        json.dump(rows, handle, indent=1, default=float)


def confirm(target: str, feature_set: str) -> None:
    with open(f"reoptimize_main_{target}_{feature_set}.json") as handle:
        rows = json.load(handle)
    best = max(rows[2:], key=lambda row: row["mean_auc"])
    X, move, times = prepare(target, feature_set)
    for row in (rows[0], best):
        result = score_config(X, move, times, row["params"], HOLDOUT_FOLDS, 1, row["early_stopping"])
        print(f"{row['name']:<22} full data holdout mean AUC {result['mean_auc']:.4f}  folds {result['fold_aucs']}  "
              f"trees {result['trees']}", flush=True)


if __name__ == "__main__":
    {"search": search, "confirm": confirm}[sys.argv[1]](sys.argv[2], sys.argv[3])
