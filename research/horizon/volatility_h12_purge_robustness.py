"""Does the horizon-12 volatility column survive the purge being moved?

The depth x horizon grid found one coherent positive column: at a 12-bar horizon the
volatility ladder gained +0.0016, +0.0024, +0.0031 and +0.0013 AUC as it reached further
back, three of those past t 2, rising to a peak six minutes back and falling away after.
A monotone shape is much harder to get by chance than one lucky cell.

But the purge moves base AUC by 0.0017 on this data, which is half the size of the peak,
so the shape has to be shown to survive it. The same five depths are run here at three
gaps: 12 bars (the correct one, exactly the label window), 18 and 24. The larger two
discard usable rows and are not what anyone should train on -- they are here only because
a result that appears at one arbitrary gap and vanishes at the next was never real.

Anything below 12 would leak, so it is not offered.

usage: volatility_h12_purge_robustness.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from cross_asset_features import load_symbol_closes
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model
from volatility_features import (TRADED_VOLATILITY_SERIES, add_volatility_features, lookback_ladder,
                                 volatility_feature_names)

HORIZON_BARS = 12
PURGE_BARS = (12, 18, 24)
LADDER_DEPTHS = (1, 3, 6, 12, 24)
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RESULTS_PATH = "volatility_h12_purge_robustness.json"


def fold_scores(X, target, decided, splits, params) -> list[float]:
    scores = []
    for split in splits:
        model = train_direction_model(X, target, split, params)
        test = split.test[decided[split.test]]
        scores.append(roc_auc_score((target[test] > 0).astype(int), model.predict_proba(X[test])[:, 1]))
    return scores


def paired_t(differences: np.ndarray) -> tuple[float, float]:
    error = differences.std(ddof=1) / np.sqrt(len(differences))
    return differences.mean(), differences.mean() / error if error > 0 else 0.0


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = add_volatility_features(data, {series: load_symbol_closes(series)
                                          for series in TRADED_VOLATILITY_SERIES},
                                   lookback_ladder(max(LADDER_DEPTHS)))
    target = average_target(data["mid"].to_numpy(float), epochs, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    base_matrix = data[base].to_numpy(np.float32)
    print(f"horizon {HORIZON_BARS}, {int(decided.sum()):,} decided rows, depths {LADDER_DEPTHS} "
          f"at purges {PURGE_BARS}, {len(SEEDS)} seeds x 2 folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    scores: dict[str, list[float]] = {}
    for purge in PURGE_BARS:
        splits = purged_fold_splits(data["time"], HORIZON_BARS, purge_bars=purge)
        for seed in SEEDS:
            scores.setdefault(f"base_p{purge}", []).extend(
                fold_scores(base_matrix, target, decided, splits, {**LGBM_PARAMS, "random_state": seed}))
        for depth in LADDER_DEPTHS:
            columns = base + volatility_feature_names(lookback_ladder(depth), TRADED_VOLATILITY_SERIES)
            X = data[columns].to_numpy(np.float32)
            for seed in SEEDS:
                scores.setdefault(f"depth{depth}_p{purge}", []).extend(
                    fold_scores(X, target, decided, splits, {**LGBM_PARAMS, "random_state": seed}))
            del X
        base_mean = np.mean(scores[f"base_p{purge}"])
        summary = "  ".join(f"d{depth} {np.mean(scores[f'depth{depth}_p{purge}']) - base_mean:+.4f}"
                            for depth in LADDER_DEPTHS)
        print(f"   purge {purge}: base {base_mean:.4f}   {summary}   "
              f"({time.time() - started:.0f}s)", flush=True)
        with open(RESULTS_PATH, "w") as handle:
            json.dump({name: list(map(float, values)) for name, values in scores.items()}, handle, indent=1)

    print(f"\npaired AUC gain over base, and its t, at each purge\n{'reaches back':<14}"
          + "".join(f"{'purge ' + str(purge):>20}" for purge in PURGE_BARS))
    print(f"{'base AUC':<14}" + "".join(f"{np.mean(scores[f'base_p{purge}']):>20.4f}" for purge in PURGE_BARS))
    for depth in LADDER_DEPTHS:
        cells = []
        for purge in PURGE_BARS:
            mean, t = paired_t(np.array(scores[f"depth{depth}_p{purge}"])
                               - np.array(scores[f"base_p{purge}"]))
            cells.append(f"{mean:>+11.5f} t{t:>5.1f}")
        print(f"{str(max(lookback_ladder(depth))) + ' bars':<14}" + "".join(f"{cell:>20}" for cell in cells))
    print("\nThe shape has to hold across all three columns to count as real.")


if __name__ == "__main__":
    main()
