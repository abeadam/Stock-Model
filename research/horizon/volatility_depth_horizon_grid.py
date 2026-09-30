"""How far back the volatility reading should reach, against how far ahead it is asked to see.

Two questions that only make sense together. Reaching further back gives the model more of
the same measurement and more ways to overfit it; predicting further ahead makes the target
less noisy but pushes the move past the point where a decision is still actionable. The
honest answer to either needs the other held at several values, so this is a grid.

At every cell the comparison is paired in the usual way -- same seeds, same folds, the
difference taken pair by pair -- between the 163 columns already there and those columns
plus the volatility ladder at that depth. The baseline does not depend on the depth, so it
is fitted once per (horizon, seed, fold) and reused down the column.

Each horizon gets its own purge: training labels for a 60-bar target have to stop 60 bars
before the test fold, or the model is scored on rows it has already half seen.

usage: volatility_depth_horizon_grid.py
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
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, fold_split, train_direction_model
from volatility_features import (TRADED_VOLATILITY_SERIES, add_volatility_features, lookback_ladder,
                                 volatility_feature_names)

# Reaching back half a minute, a minute and a half, three minutes, six and twelve.
LADDER_DEPTHS = (1, 3, 6, 12, 24)
# 25 seconds out to five minutes.
HORIZON_BARS = (5, 12, 30, 60)
STAGE = "dev"
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RESULTS_PATH = "volatility_depth_horizon_grid.json"


def purged_fold_splits(data, horizon: int):
    """Folds whose training and validation labels stop `horizon` bars short of what follows."""
    return [fold_split(data["time"], fold, purge_bars=horizon, fold_starts=FOLD_STARTS_UTC,
                       data_end=DATA_END_UTC) for fold in FOLDS_BY_STAGE[STAGE]]


def fold_scores(X, target, decided, splits, params, added_columns: int = 0) -> tuple[list[float], list[float]]:
    """AUC per fold, and the share of the trees' total gain that the added columns took.

    A ladder can fail two ways that look alike in the AUC alone: the model may ignore the
    columns, or it may spend splits on them and lose. The gain share separates them."""
    scores, gain_shares = [], []
    for split in splits:
        model = train_direction_model(X, target, split, params)
        test = split.test[decided[split.test]]
        scores.append(roc_auc_score((target[test] > 0).astype(int), model.predict_proba(X[test])[:, 1]))
        if added_columns:
            gain = model.booster_.feature_importance(importance_type="gain")
            gain_shares.append(float(gain[-added_columns:].sum() / max(gain.sum(), 1e-12)))
    return scores, gain_shares


def paired_t(differences: np.ndarray) -> tuple[float, float]:
    error = differences.std(ddof=1) / np.sqrt(len(differences))
    return differences.mean(), differences.mean() / error if error > 0 else 0.0


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    deepest = lookback_ladder(max(LADDER_DEPTHS))
    data = add_volatility_features(data, {series: load_symbol_closes(series)
                                          for series in TRADED_VOLATILITY_SERIES}, deepest)
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    columns_by_depth = {depth: base + volatility_feature_names(lookback_ladder(depth),
                                                               TRADED_VOLATILITY_SERIES)
                        for depth in LADDER_DEPTHS}
    # One depth's matrix at a time: all five at once would be 2.7 GB of float32.
    base_matrix = data[base].to_numpy(np.float32)
    print(f"{len(base)} base columns; depths {LADDER_DEPTHS} reaching back to "
          f"{max(deepest)} bars; horizons {HORIZON_BARS} bars; {len(SEEDS)} seeds x 2 folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    scores: dict[str, list[float]] = {}
    gain_shares: dict[str, list[float]] = {}
    for horizon in HORIZON_BARS:
        target = average_target(data["mid"].to_numpy(float), epochs, horizon)
        decided = np.isfinite(target) & (target != 0)
        splits = purged_fold_splits(data, horizon)
        print(f"horizon {horizon}: {int(decided.sum()):,} decided rows, purge {horizon} bars", flush=True)
        for seed in SEEDS:
            fold_aucs, _ = fold_scores(base_matrix, target, decided, splits,
                                       {**LGBM_PARAMS, "random_state": seed})
            scores.setdefault(f"base_h{horizon}", []).extend(fold_aucs)
        for depth in LADDER_DEPTHS:
            columns = columns_by_depth[depth]
            X = data[columns].to_numpy(np.float32)
            for seed in SEEDS:
                fold_aucs, shares = fold_scores(X, target, decided, splits,
                                                {**LGBM_PARAMS, "random_state": seed},
                                                added_columns=len(columns) - len(base))
                scores.setdefault(f"depth{depth}_h{horizon}", []).extend(fold_aucs)
                gain_shares.setdefault(f"depth{depth}_h{horizon}", []).extend(shares)
            del X
        base_mean = np.mean(scores[f"base_h{horizon}"])
        summary = "  ".join(f"d{depth} {np.mean(scores[f'depth{depth}_h{horizon}']) - base_mean:+.4f}"
                            for depth in LADDER_DEPTHS)
        print(f"   base {base_mean:.4f}   {summary}   ({time.time() - started:.0f}s)", flush=True)
        with open(RESULTS_PATH, "w") as handle:
            json.dump({"auc": {name: list(map(float, values)) for name, values in scores.items()},
                       "gain_share": {name: list(map(float, values)) for name, values in gain_shares.items()}},
                      handle, indent=1)

    header = "".join(f"{'h=' + str(horizon):>18}" for horizon in HORIZON_BARS)
    print(f"\npaired AUC gain from adding the ladder, and its t-statistic\n{'reaches back':<14}{header}")
    print(f"{'base AUC':<14}" + "".join(f"{np.mean(scores[f'base_h{h}']):>18.4f}" for h in HORIZON_BARS))
    for depth in LADDER_DEPTHS:
        cells = []
        for horizon in HORIZON_BARS:
            mean, t = paired_t(np.array(scores[f"depth{depth}_h{horizon}"])
                               - np.array(scores[f"base_h{horizon}"]))
            cells.append(f"{mean:>+11.5f} t{t:>5.1f}")
        print(f"{str(max(lookback_ladder(depth))) + ' bars':<14}" + "".join(f"{cell:>18}" for cell in cells))

    print(f"\nshare of tree gain the added columns took\n{'reaches back':<14}{header}")
    for depth in LADDER_DEPTHS:
        cells = [f"{np.mean(gain_shares[f'depth{depth}_h{horizon}']):.1%}" for horizon in HORIZON_BARS]
        print(f"{str(max(lookback_ladder(depth))) + ' bars':<14}" + "".join(f"{cell:>18}" for cell in cells))


if __name__ == "__main__":
    main()
