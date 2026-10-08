"""What should a sub-model predict: which horizon, and the extreme or the average?

Production asks each sub-model for the highest high (or lowest low) of the next five
5-second bars. This varies the target while holding everything else fixed: the same
features, the same settings the weekly run actually trains (production parameters with
early stopping), the same walk-forward folds, each fold purged by its own horizon.

Read the two numbers together. R-squared is measured against each target's own
variance, so it says how much of that target is explainable, not how much money it is
worth; the median target in points says how big the thing being predicted is. A long
horizon can score well merely because its extreme tracks the volatility level, and an
average scores well partly because averaging cancels noise an extreme collects --
neither is by itself worth more money.

The feature set belongs to a sub-model (20 columns for highest, 134 for lowest), so a
target is only ever compared against another target scored on the same features.

usage: submodel_horizons.py FEATURES_OF TARGET [horizon ...]
       FEATURES_OF  highest|lowest   whose feature set and settings to use
       TARGET       highest|lowest|average   what to predict (defaults to FEATURES_OF)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from forward_extremes import forward_target, load_prices, scale_for_training
from reoptimize_submodels import MAX_TREES, PRODUCTION_PARAMS, fit_and_score, prepare
from short_horizon_walk_forward import DEV_FOLDS

HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 60)
PRODUCTION_HORIZON = 5
PRODUCTION_TARGETS = ("highest", "lowest")
ROW_STRIDE = 3                      # every third bar, as in the settings search
PERCENT = 100.0


def aligned_prices(times: pd.Series) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The price series restricted, in order, to the sub-model's rows, and which
    of those rows it covers (a couple of bars are missing from the price file)."""
    prices = load_prices()
    wanted = times.to_numpy().astype("datetime64[s]").astype(np.int64)
    position = np.clip(np.searchsorted(prices["seconds"], wanted), 0, len(prices["seconds"]) - 1)
    matched = prices["seconds"][position] == wanted
    return {name: values[position[matched]] for name, values in prices.items()}, matched


def report_replication(prices: dict[str, np.ndarray], production_target: np.ndarray, target: str) -> None:
    """Prove the rebuilt five-bar target is the one production trains on."""
    rebuilt = scale_for_training(forward_target(prices, PRODUCTION_HORIZON, target))
    both = np.isfinite(rebuilt) & np.isfinite(production_target)
    difference = np.abs(rebuilt[both] - production_target[both])
    print(f"five-bar target rebuilt on {both.sum():,} rows: max difference {difference.max():.2e}, "
          f"correlation {np.corrcoef(rebuilt[both], production_target[both])[0, 1]:.6f}", flush=True)


def score_horizon(X, times, prices, target_kind: str, horizon: int, params: dict) -> dict:
    raw = forward_target(prices, horizon, target_kind)
    target = scale_for_training(raw)
    usable = np.isfinite(target)
    # Rows without a target cannot train or score; LightGBM sees the rest unchanged.
    scored = [fit_and_score(X[usable], target[usable], times[usable].reset_index(drop=True),
                            fold, params, ROW_STRIDE, early_stopping=True) for fold in DEV_FOLDS]
    median_points = float(np.median(np.abs(raw[usable]) / PERCENT * prices["close"][usable]))
    return {"horizon": horizon, "mean_r2": float(np.mean([r2 for r2, _ in scored])),
            "fold_r2": [round(r2, 4) for r2, _ in scored], "trees": [trees for _, trees in scored],
            "median_target_points": round(median_points, 3), "rows": int(usable.sum())}


def evaluate(features_of: str, target_kind: str, horizons: tuple[int, ...]) -> None:
    started = time.time()
    X, production_target, times = prepare(features_of)
    prices, matched = aligned_prices(times)
    X, production_target = X[matched], production_target[matched]
    times = times[matched].reset_index(drop=True)
    if target_kind in PRODUCTION_TARGETS:
        report_replication(prices, production_target, target_kind)
    params = PRODUCTION_PARAMS | {"n_estimators": MAX_TREES}
    rows = []
    for horizon in horizons:
        rows.append(score_horizon(X, times, prices, target_kind, horizon, params))
        print(f"{horizon:>3} bars ({horizon * 5:>3}s)  mean R2 {rows[-1]['mean_r2']:.4f}  "
              f"folds {rows[-1]['fold_r2']}  median target {rows[-1]['median_target_points']:5.2f} pt  "
              f"trees {rows[-1]['trees']}  ({time.time() - started:.0f}s)", flush=True)
    name = features_of if target_kind == features_of else f"{features_of}_predicting_{target_kind}"
    with open(f"submodel_horizons_{name}.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    features_of, target_kind = sys.argv[1], (sys.argv[2:3] or [sys.argv[1]])[0]
    evaluate(features_of, target_kind, tuple(int(bars) for bars in sys.argv[3:]) or HORIZONS)
