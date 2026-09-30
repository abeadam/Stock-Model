"""Search the entry threshold, take-profit distance and stop distance on each
model's validation rows, then trade the chosen settings, unchanged, on its test rows.

This is the low-dimensional version of what a reinforcement-learning agent would
learn for entries, exits and stops. Distances are multiples of the standard
deviation of 5-second mid changes over the last 5 minutes, never below 0.25
points. Every trade also closes on a confident opposite signal and at the end
of its period. Uses the predictions cached by segment_walk_forward.py.

usage: optimize_exits.py dev|holdout
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
from dataclasses import dataclass
import multiprocessing

import numpy as np
import pandas as pd

from rule_selection import run_take_profit
from segment_walk_forward import (FILL_MODES, MIN_TRADES_PER_VALIDATION_DAY, TARGET_GROUPS, bar_seconds,
                                  fill_prices, load_cut, predictions_path)
from take_profit_policy import recent_volatility, take_profit_points

THRESHOLD_QUANTILES = (0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
TAKE_PROFIT_MULTIPLES = (1.0, 2.0, 3.0, 4.0)
STOP_MULTIPLES = (None, 1.0, 2.0, 4.0)
TARGETS_SEARCHED = ("mid_12", "mid_60", "barrier_changes")
WORKERS = max(1, (os.cpu_count() or 2) - 2)
NO_TRADES = {"trades": 0, "net": 0.0, "hit_rate": np.nan, "target_rate": np.nan, "stop_rate": np.nan}

# Filled in before the worker pool forks, so workers share the arrays (hence "fork",
# not macOS's default "spawn").
_SHARED: dict = {}


@dataclass(frozen=True)
class ExitSettings:
    threshold: float
    take_profit_multiple: float
    stop_multiple: float | None


def distances(settings: ExitSettings) -> tuple[np.ndarray, np.ndarray | None]:
    volatility = _SHARED["volatility"]
    stop = None if settings.stop_multiple is None else take_profit_points(volatility, settings.stop_multiple)
    return take_profit_points(volatility, settings.take_profit_multiple), stop


def trade(score, rows, mode: str, settings: ExitSettings) -> dict:
    buy, sell = _SHARED["prices"][mode]
    target, stop = distances(settings)
    return run_take_profit(score, buy[rows], sell[rows], target[rows], _SHARED["seconds"][rows],
                           settings.threshold, None if stop is None else stop[rows])


def choose_settings(prediction: dict, mode: str) -> ExitSettings | None:
    rows, score = prediction["validation_rows"], prediction["validation_score"]
    min_trades = MIN_TRADES_PER_VALIDATION_DAY * len(np.unique(_SHARED["session_day"][rows]))
    best, best_net = None, -np.inf
    for quantile, take_profit, stop in itertools.product(THRESHOLD_QUANTILES, TAKE_PROFIT_MULTIPLES, STOP_MULTIPLES):
        settings = ExitSettings(float(np.quantile(np.abs(score), quantile)), take_profit, stop)
        result = trade(score, rows, mode, settings)
        if result["trades"] >= min_trades and result["net"] > best_net:
            best, best_net = settings, result["net"]
    return best


def evaluate_prediction(prediction: dict) -> dict:
    row = {key: prediction[key] for key in ("fold", "target", "features", "model", "segment", "auc")}
    for mode in FILL_MODES:
        settings = choose_settings(prediction, mode)
        result = NO_TRADES if settings is None else trade(prediction["test_score"], prediction["test_rows"], mode, settings)
        row |= {f"{mode}_{key}": result[key] for key in NO_TRADES}
        row |= {f"{mode}_take_profit_multiple": None if settings is None else settings.take_profit_multiple,
                f"{mode}_stop_multiple": None if settings is None else settings.stop_multiple}
    return row


def load_predictions(stage: str) -> list[dict]:
    predictions = []
    for group in TARGET_GROUPS:
        if os.path.exists(predictions_path(stage, group)):
            with open(predictions_path(stage, group), "rb") as handle:
                predictions += pickle.load(handle)
    return [p for p in predictions if p.get("target", f"mid_{p['bars']}") in TARGETS_SEARCHED]


def main(stage: str) -> None:
    data = load_cut()
    seconds = bar_seconds(data)
    _SHARED.update(seconds=seconds, session_day=data["session_day"].to_numpy(),
                   volatility=recent_volatility(data["mid"].to_numpy(), seconds, "changes"),
                   prices={mode: fill_prices(data, mode) for mode in FILL_MODES})
    del data
    predictions = [p | {"target": p.get("target", f"mid_{p['bars']}")} for p in load_predictions(stage)]
    print(f"{len(predictions)} model predictions, {WORKERS} workers", flush=True)
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        rows = pool.map(evaluate_prediction, predictions, chunksize=1)
    results = pd.DataFrame(rows)
    results.to_pickle(f"optimized_exits_{stage}.pkl")
    print(results.groupby(["target", "segment", "model"])[["mid_net", "cross_net", "mid_trades", "cross_trades"]]
          .sum().round(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
