"""Does giving the model orthogonal columns beat giving it redundant ones?

The feature sets have grown to 255 columns that largely repeat each other. This measures
how many independent directions each set really spans, then rebuilds it as that many
orthogonal columns and scores the model both ways, on the same window, folds and target
as every other feature study.

A boosted tree splits one column at a time, so it is not rotation-invariant: rotating to
the principal axes can help, by putting a combination the tree would need many splits to
carve out onto a single axis, or hurt, by mixing a sharp feature into a blend. That is
the question here, and it is why the raw set is scored alongside.

Fitted on training rows only -- median, scale and rotation alike.

usage: rank_reduction_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from feature_rank import describe, prepare, principal_axes, singular_values, standardise, variance_rank
from indicator_features import EQUITY_INDICATOR_FEATURES, EQUITY_SYMBOLS, MACRO_INDICATOR_FEATURES
from indicator_features import MACRO_SYMBOLS, add_indicator_features
from indicator_test import RETURNS_OF_NEW_DATA
from multi_timeframe_features import ALL_TIMEFRAME_FEATURES, BASE_BAR_SECONDS, TIMEFRAME_FEATURES
from multi_timeframe_features import add_timeframe_features
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, load_cut, mid_move
from short_horizon_walk_forward import FEATURES, direction_auc, fold_split, train_direction_model

TARGET_BARS = 12
STAGE = "dev"
SVD_ROW_STRIDE = 20                    # every 20th training row is plenty to find the axes
VARIANCE_SHARES = (0.95, 0.99)


def feature_groups() -> dict[str, list[str]]:
    everything = (FEATURES + RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES
                  + EQUITY_INDICATOR_FEATURES + ALL_TIMEFRAME_FEATURES)
    return {"indicators_8": TIMEFRAME_FEATURES[BASE_BAR_SECONDS], "current_44": FEATURES,
            "everything": everything}


def score(X: np.ndarray, move: np.ndarray, data) -> float:
    aucs = []
    for fold in FOLDS_BY_STAGE[STAGE]:
        split = fold_split(data["time"], fold, purge_bars=TARGET_BARS,
                           fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
        model = train_direction_model(X, move, split)
        aucs.append(direction_auc(model.predict_proba(X[split.test])[:, 1], move[split.test]))
    return float(np.mean(aucs))


def projected(X: np.ndarray, data, components: int) -> np.ndarray:
    """X rebuilt as `components` orthogonal columns, the rotation fitted per fold.

    Each fold's test rows are projected on axes found from that fold's training rows, so
    the columns differ between folds; the model is refitted per fold anyway."""
    rebuilt = np.zeros((len(X), components), dtype=np.float32)
    for fold in FOLDS_BY_STAGE[STAGE]:
        split = fold_split(data["time"], fold, purge_bars=TARGET_BARS,
                           fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
        fitted = prepare(X[split.train[::SVD_ROW_STRIDE]])
        axes = principal_axes(standardise(X[split.train[::SVD_ROW_STRIDE]], fitted), components)
        for rows in (split.train, split.validation, split.test):
            rebuilt[rows] = (standardise(X[rows], fitted) @ axes).astype(np.float32)
    return rebuilt


def evaluate() -> None:
    started = time.time()
    data = add_timeframe_features(add_indicator_features(load_cut(), (*MACRO_SYMBOLS, *EQUITY_SYMBOLS)))
    move = mid_move(data, TARGET_BARS)
    first = fold_split(data["time"], FOLDS_BY_STAGE[STAGE][0], purge_bars=TARGET_BARS,
                       fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
    print(f"{len(data):,} bars ({time.time() - started:.0f}s)\n")
    results = []
    for name, columns in feature_groups().items():
        X = data[columns].to_numpy(np.float32)
        sample = X[first.train[::SVD_ROW_STRIDE]]
        standardised = standardise(sample, prepare(sample))
        summary = describe(name, standardised)
        values = singular_values(standardised)
        raw_auc = score(X, move, data)
        row = {"group": name, **summary, "raw_auc": raw_auc, "projected": {}}
        print(f"{'':<30}raw AUC {raw_auc:.4f}", flush=True)
        for share in VARIANCE_SHARES:
            components = variance_rank(values, share)
            auc = score(projected(X, data, components), move, data)
            row["projected"][f"{share:.0%}"] = {"components": components, "auc": auc}
            print(f"{'':<30}{components:>3} orthogonal columns ({share:.0%} of variance)  "
                  f"AUC {auc:.4f}  ({time.time() - started:.0f}s)", flush=True)
        results.append(row)
    with open("rank_reduction_test.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    evaluate()
