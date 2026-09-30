"""Sweep every applicable TA-Lib indicator over all 163 columns and keep the best.

30,644 candidates cannot be held at once, so each is computed, scored against both heads,
and thrown away, with only a running top list kept. Scoring uses the **earliest fold's
training rows alone**, a block that precedes both test months, so nothing chosen here has
seen either.

What the screen can and cannot claim: taking the best of 30,644 correlations inflates them
enormously by multiple comparison, and the ranking is linear where the model is not. The
screen proposes a shortlist; only the paired ablation on held-out months decides.

usage: talib_screen.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES
from talib_candidates import (candidate_name, correlation, evaluate, parameter_settings,
                              windowed_single_series_functions)

HORIZON_BARS = 18
KEEP_PER_HEAD = 100
PROGRESS_EVERY = 2_000
RECIPES_PATH = "talib_screen_recipes.json"
RANKING_PATH = "talib_screen_ranking.json"


def best_of(scored: list[tuple[float, str, dict]], keep: int) -> list[tuple[float, str, dict]]:
    return sorted(scored, key=lambda row: -row[0])[:keep]


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    direction = average_target(data["mid"].to_numpy(float), epochs, horizon)
    direction_label = np.where(np.isfinite(direction) & (direction != 0),
                               (direction > 0).astype(float), np.nan)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    train_rows = purged_fold_splits(data["time"], horizon)[0].train
    direction_on_train = direction_label[train_rows]
    range_on_train = range_target[train_rows]

    functions = windowed_single_series_functions()
    settings = {name: parameter_settings(name) for name in functions}
    planned = len(base) * sum(len(values) for values in settings.values())
    print(f"{len(base)} columns x {len(functions)} functions = {planned:,} candidate settings; "
          f"scored on {len(train_rows):,} training rows of the earliest fold "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    direction_best, range_best, examined = [], [], 0
    for column in base:
        series = data[column].to_numpy(float)
        if not np.isfinite(series).any():
            continue
        for function in functions:
            for params in settings[function]:
                outputs = evaluate(function, series, params)
                for index, values in enumerate(outputs):
                    examined += 1
                    name = candidate_name(column, function, params, index, len(outputs))
                    on_train = values[train_rows]
                    direction_best.append((correlation(on_train, direction_on_train), name,
                                           {"column": column, "function": function,
                                            "params": params, "output": index,
                                            "outputs": len(outputs)}))
                    range_best.append((correlation(on_train, range_on_train), name,
                                       {"column": column, "function": function,
                                        "params": params, "output": index,
                                        "outputs": len(outputs)}))
                if examined and examined % PROGRESS_EVERY < len(outputs):
                    direction_best = best_of(direction_best, KEEP_PER_HEAD)
                    range_best = best_of(range_best, KEEP_PER_HEAD)
                    print(f"   {examined:>7,} examined   best direction "
                          f"{direction_best[0][0]:.4f}   best range {range_best[0][0]:.4f}   "
                          f"({time.time() - started:.0f}s)", flush=True)

    direction_best = best_of(direction_best, KEEP_PER_HEAD)
    range_best = best_of(range_best, KEEP_PER_HEAD)
    chosen = {}
    for score, name, recipe in direction_best + range_best:
        chosen.setdefault(name, recipe)
    print(f"\nexamined {examined:,} candidates; kept {len(chosen)} unique "
          f"({time.time() - started:.0f}s)")
    print(f"\n{'top by direction':<44}{'|corr|':>9}      {'top by range':<44}{'|corr|':>9}")
    for (left_score, left_name, _), (right_score, right_name, _) in zip(direction_best[:15],
                                                                         range_best[:15]):
        print(f"{left_name:<44}{left_score:>9.4f}      {right_name:<44}{right_score:>9.4f}")
    with open(RECIPES_PATH, "w") as handle:
        json.dump(chosen, handle, indent=1)
    with open(RANKING_PATH, "w") as handle:
        json.dump({"direction": [name for _, name, _ in direction_best],
                   "range": [name for _, name, _ in range_best],
                   "direction_scores": [score for score, _, _ in direction_best],
                   "range_scores": [score for score, _, _ in range_best]}, handle, indent=1)
    print(f"\nrecipes written to {RECIPES_PATH}, per-head ranking to {RANKING_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
