"""Which of the eight 5-second indicators carry the signal, and which are passengers?

Feature importance answers the wrong question: a boosted tree spends splits on any column
it is handed. These are the two answers that mean something, on the same window, folds
and target as every other feature study:

  alone           one indicator as the model's only input -- what it knows by itself
  without it      the other seven -- what is lost when it goes, which is what a feature
                  has to justify when its neighbours already say much of the same thing

A column can score well alone and cost nothing when removed; that means it is duplicated,
not that it is useless.

usage: indicator_attribution.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from multi_timeframe_features import BASE_BAR_SECONDS, TIMEFRAME_FEATURES, add_timeframe_features
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, load_cut, mid_move
from short_horizon_walk_forward import direction_auc, fold_split, train_direction_model

TARGET_BARS = 12                      # the 1-minute mid target the eight were measured on
STAGE = "dev"
INDICATORS = TIMEFRAME_FEATURES[BASE_BAR_SECONDS]


def mean_auc(data, columns: list[str], move: np.ndarray) -> tuple[float, list[float]]:
    X = data[columns].to_numpy(np.float32)
    aucs = []
    for fold in FOLDS_BY_STAGE[STAGE]:
        split = fold_split(data["time"], fold, purge_bars=TARGET_BARS,
                           fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
        model = train_direction_model(X, move, split)
        aucs.append(direction_auc(model.predict_proba(X[split.test])[:, 1], move[split.test]))
    return float(np.mean(aucs)), [round(auc, 4) for auc in aucs]


def evaluate() -> None:
    started = time.time()
    data = add_timeframe_features(load_cut(), widths=(BASE_BAR_SECONDS,))
    move = mid_move(data, TARGET_BARS)
    together, folds = mean_auc(data, INDICATORS, move)
    print(f"all eight together: AUC {together:.4f}  folds {folds}  ({time.time() - started:.0f}s)\n", flush=True)
    print(f"{'indicator':<22}{'alone':>9}{'without it':>13}{'cost of removing':>19}")
    rows = []
    for name in INDICATORS:
        alone, _ = mean_auc(data, [name], move)
        without, _ = mean_auc(data, [other for other in INDICATORS if other != name], move)
        rows.append({"indicator": name, "alone": alone, "without": without, "cost": together - without})
        print(f"{name.replace('mtf_5s_', ''):<22}{alone:>9.4f}{without:>13.4f}{together - without:>+19.4f}",
              flush=True)
    with open("indicator_attribution.json", "w") as handle:
        json.dump({"together": together, "indicators": rows}, handle, indent=1)


if __name__ == "__main__":
    evaluate()
