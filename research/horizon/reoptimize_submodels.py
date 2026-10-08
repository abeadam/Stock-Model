"""Re-optimize the two sub-models (highest and lowest: the next 5 bars' max high and
min low relative to the close) with honest walk-forward.

Data and features are the production pipeline's: es_with_indicators.csv through
lightgbm_utils.prepare_features_and_target (asinh-scaled target), narrowed to the
top-N features the saved model uses. Score: out-of-sample R-squared on each test
fold, each fold trained only on earlier bars (early stopping on a validation tail).
The search trains on every third bar and uses development folds 1-4; the production
settings and the best one found are then retrained on every bar and scored once on
holdout folds 5-6.

The "features" command keeps only the top-N features ranked by their own R-squared
with the target (squared correlation, measured on each fold's training bars only),
for several N, with the production settings plus early stopping.

"confirm_features" retrains the search's best settings on every bar with only the top-N
features (same ranking) and scores the holdout folds once.

usage: reoptimize_submodels.py search|confirm|features highest|lowest
       reoptimize_submodels.py confirm_features highest|lowest N
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from short_horizon_walk_forward import DEV_FOLDS, HOLDOUT_FOLDS, fold_split

BASIC_MODEL_DIR = Path("/Users/abeadam/dev/model/Stock-Model/future_prices/basic_model")
sys.path.insert(0, str(BASIC_MODEL_DIR))
from lightgbm_utils import load_top_n_features_from_importance_csv, prepare_features_and_target  # noqa: E402

ES_WITH_INDICATORS = BASIC_MODEL_DIR.parent / "es_with_indicators.csv"
TARGET_COLUMNS = {"highest": "PctChange_ToMaxHigh_5", "lowest": "PctChange_ToMinLow_5"}
PURGE_BARS = 5
SEARCH_ROW_STRIDE = 3
SEARCH_CONFIGS = 24
EARLY_STOPPING_ROUNDS = 100
MAX_TREES = 5000
FEATURE_COUNTS = (10, 20, 30, 50, 80, None)          # None keeps every feature
RANDOM_SEED = 2027
# The saved models' settings (lightgbm_model_highest.py / _lowest.py), no early stopping.
PRODUCTION_PARAMS = dict(objective="regression", n_estimators=7885, max_depth=6, num_leaves=4095,
                         learning_rate=0.07232447872229951, min_data_in_leaf=8, bagging_freq=5,
                         feature_fraction=0.9661638227728979, bagging_fraction=0.7956488938538635,
                         lambda_l1=0.04822149251619493, lambda_l2=0.06079712296915085, min_gain_to_split=0.0)
SEARCH_SPACE = dict(num_leaves=[15, 31, 63, 127, 255], max_depth=[-1, 6, 8, 12], learning_rate=[0.02, 0.05, 0.1],
                    min_data_in_leaf=[50, 200, 1000, 5000], feature_fraction=[0.5, 0.7, 0.9, 1.0],
                    bagging_fraction=[0.5, 0.7, 0.9], lambda_l2=[0.0, 1.0, 10.0, 100.0])
COMMON_PARAMS = dict(n_jobs=-1, verbose=-1, random_state=11)


def cache_path(model_name: str) -> str:
    return f"submodel_data_{model_name}.npz"


def columns_needed(model_name: str) -> list[str]:
    """Stored columns prepare_features_and_target can read (every CSV column its source
    names), plus the timestamp and the target: the 46 GB file never loads whole."""
    header = pd.read_csv(ES_WITH_INDICATORS, nrows=0).columns
    source = inspect.getsource(prepare_features_and_target)
    quoted = set(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", source))
    return sorted((quoted & set(header)) | {"DateTime", TARGET_COLUMNS[model_name]})


def prepare(model_name: str):
    """(X float32, scaled target, bar times) for one sub-model, cached after the first build."""
    if os.path.exists(cache_path(model_name)):
        cached = np.load(cache_path(model_name), allow_pickle=True)
        return cached["X"], cached["y"], pd.Series(pd.to_datetime(cached["times"]))
    frame = pd.read_csv(ES_WITH_INDICATORS, usecols=columns_needed(model_name))
    X, y, feature_cols, _scale, _method, _original, kept = prepare_features_and_target(
        frame, target_column=TARGET_COLUMNS[model_name], return_valid_mask=True)
    model_path = BASIC_MODEL_DIR / f"lightgbm_model_{model_name}.pkl"
    top_n = joblib.load(model_path).n_features_in_
    names = load_top_n_features_from_importance_csv(
        BASIC_MODEL_DIR / f"lightgbm_model_{model_name}_feature_importances_splits_and_gain.csv", top_n)
    columns = [feature_cols.index(name) for name in names]
    X = np.asarray(X)[:, columns].astype(np.float32)
    times = pd.to_datetime(frame.loc[kept, "DateTime"]).to_numpy()
    order = np.argsort(times, kind="stable")
    X, y, times = X[order], np.asarray(y, np.float64)[order], times[order]
    np.savez(cache_path(model_name), X=X, y=y, times=times)
    print(f"{model_name}: {len(y):,} rows, {X.shape[1]} features ({', '.join(names)})", flush=True)
    return X, y, pd.Series(pd.to_datetime(times))


def random_configs(count: int, seed: int) -> list[dict]:
    generator = np.random.default_rng(seed)
    return [{name: values[generator.integers(len(values))] for name, values in SEARCH_SPACE.items()}
            | {"bagging_freq": 1, "objective": "regression", "n_estimators": MAX_TREES} for _ in range(count)]


def fit_and_score(X, y, times, fold: int, params: dict, stride: int, early_stopping: bool) -> tuple[float, int]:
    split = fold_split(times, fold, purge_bars=PURGE_BARS)
    train = split.train[::stride]
    model = lgb.LGBMRegressor(**params, **COMMON_PARAMS)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if early_stopping else []
    model.fit(X[train], y[train], eval_set=[(X[split.validation], y[split.validation])], eval_metric="l2",
              callbacks=callbacks + [lgb.log_evaluation(0)])
    trees = model.best_iteration_ if early_stopping and model.best_iteration_ else params["n_estimators"]
    return float(r2_score(y[split.test], model.predict(X[split.test]))), int(trees)


def score_config(X, y, times, params, folds, stride, early_stopping) -> dict:
    results = [fit_and_score(X, y, times, fold, params, stride, early_stopping) for fold in folds]
    r2s = [r2 for r2, _ in results]
    return {"mean_r2": float(np.mean(r2s)), "fold_r2": [round(r, 4) for r in r2s], "trees": [t for _, t in results]}


def univariate_r2(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared correlation of each column with y, ignoring rows where that column is NaN."""
    scores = np.zeros(X.shape[1])
    for column in range(X.shape[1]):
        values = X[:, column].astype(np.float64)
        known = np.isfinite(values)
        if known.sum() > 2 and np.std(values[known]) > 0:
            scores[column] = np.corrcoef(values[known], y[known])[0, 1] ** 2
    return scores


def fit_and_score_top_features(X, y, times, fold: int, count: int | None, params: dict, stride: int) -> tuple[float, list[int]]:
    split = fold_split(times, fold, purge_bars=PURGE_BARS)
    train = split.train[::stride]
    ranking = np.argsort(-univariate_r2(X[train], y[train]))
    columns = ranking if count is None else ranking[:count]
    model = lgb.LGBMRegressor(**params, **COMMON_PARAMS)
    model.fit(X[train][:, columns], y[train], eval_set=[(X[split.validation][:, columns], y[split.validation])],
              eval_metric="l2", callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
    return float(r2_score(y[split.test], model.predict(X[split.test][:, columns]))), list(columns[:10])


def features(model_name: str) -> None:
    """Out-of-sample R-squared when keeping only the top-N features by their own R-squared."""
    started = time.time()
    X, y, times = prepare(model_name)
    params = PRODUCTION_PARAMS | {"n_estimators": MAX_TREES}
    rows = []
    for count in FEATURE_COUNTS:
        results = [fit_and_score_top_features(X, y, times, fold, count, params, SEARCH_ROW_STRIDE) for fold in DEV_FOLDS]
        r2s = [r2 for r2, _ in results]
        label = "all" if count is None else str(count)
        rows.append({"features": label, "mean_r2": float(np.mean(r2s)), "fold_r2": [round(r, 4) for r in r2s]})
        print(f"top {label:>3} features by own R2   mean R2 {np.mean(r2s):.4f}  folds {rows[-1]['fold_r2']}  "
              f"({time.time() - started:.0f}s)", flush=True)
    with open(f"reoptimize_submodel_{model_name}_features.json", "w") as handle:
        json.dump(rows, handle, indent=1)


def confirm_features(model_name: str, count: str) -> None:
    with open(f"reoptimize_submodel_{model_name}.json") as handle:
        best = max(json.load(handle)[2:], key=lambda row: row["mean_r2"])
    X, y, times = prepare(model_name)
    r2s = [fit_and_score_top_features(X, y, times, fold, int(count), best["params"], 1)[0] for fold in HOLDOUT_FOLDS]
    print(f"best settings, top {count} features, full data holdout mean R2 {np.mean(r2s):.4f}  "
          f"folds {[round(r, 4) for r in r2s]}", flush=True)


def search(model_name: str) -> None:
    started = time.time()
    X, y, times = prepare(model_name)
    candidates = [("production settings", PRODUCTION_PARAMS, False),
                  ("production + early stopping", PRODUCTION_PARAMS | {"n_estimators": MAX_TREES}, True)]
    candidates += [(f"random {index}", params, True) for index, params in enumerate(random_configs(SEARCH_CONFIGS, RANDOM_SEED))]
    rows = []
    for name, params, early_stopping in candidates:
        result = score_config(X, y, times, params, DEV_FOLDS, SEARCH_ROW_STRIDE, early_stopping)
        rows.append({"name": name, "params": params, "early_stopping": early_stopping, **result})
        print(f"{name:<30} mean R2 {result['mean_r2']:.4f}  folds {result['fold_r2']}  trees {result['trees']}  "
              f"({time.time() - started:.0f}s)", flush=True)
    with open(f"reoptimize_submodel_{model_name}.json", "w") as handle:
        json.dump(rows, handle, indent=1, default=float)


def confirm(model_name: str) -> None:
    """Production settings against the search's best, retrained on every bar, scored once
    on the holdout folds (development folds were already scored in the search)."""
    with open(f"reoptimize_submodel_{model_name}.json") as handle:
        rows = json.load(handle)
    best = max(rows[2:], key=lambda row: row["mean_r2"])
    X, y, times = prepare(model_name)
    # rows[1] is production settings with early stopping: what the weekly run actually
    # produces (the saved models hold 420 and 115 trees, not the 7885 they ask for).
    for row in (rows[0], rows[1], best):
        result = score_config(X, y, times, row["params"], HOLDOUT_FOLDS, 1, row["early_stopping"])
        print(f"{row['name']:<22} full data holdout mean R2 {result['mean_r2']:.4f}  folds {result['fold_r2']}  "
              f"trees {result['trees']}", flush=True)


if __name__ == "__main__":
    {"search": search, "confirm": confirm, "features": features,
     "confirm_features": confirm_features}[sys.argv[1]](*sys.argv[2:])
