"""How large a difference this harness can manufacture on its own.

Two ablations of the same 163 columns, the same target and the same folds disagreed by
0.001 AUC across two processes -- which is the size of every effect this project has been
chasing. Nothing measured here means anything until that number is known, so three
questions are asked inside a single process, where nothing else can drift:

  repeat        the same seed fitted twice on identical data. Anything but exactly zero is
                LightGBM's own nondeterminism: at n_jobs=-1 with deterministic left false,
                histogram construction depends on how threads happen to be scheduled.
  pinned        the same, with deterministic=True, force_row_wise=True and a fixed thread
                count -- the documented cure. If this repeats exactly and the default does
                not, the cause is settled and every future ablation should pin.
  purge         purge 5 against purge 12, the one deliberate difference between the two
                runs that disagreed.

The comparison that matters is `repeat` against the effect sizes being reported. If the
harness moves 0.001 on its own, a 0.001 finding is not a finding.

usage: harness_reproducibility.py
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from segment_walk_forward import DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, fold_split, train_direction_model

HORIZON_BARS = 5
COMPARISON_PURGE_BARS = 12
PINNED_THREADS = 4
SEEDS = (11, 23, 37, 51)
STAGE = "dev"


def splits_for(data, purge_bars: int):
    return [fold_split(data["time"], fold, purge_bars=purge_bars, fold_starts=FOLD_STARTS_UTC,
                       data_end=DATA_END_UTC) for fold in FOLDS_BY_STAGE[STAGE]]


def auc_per_fold(X, target, decided, splits, params) -> list[float]:
    scores = []
    for split in splits:
        model = train_direction_model(X, target, split, params)
        test = split.test[decided[split.test]]
        scores.append(roc_auc_score((target[test] > 0).astype(int), model.predict_proba(X[test])[:, 1]))
    return scores


def report(name: str, first: list[float], second: list[float]) -> None:
    gap = np.abs(np.array(first) - np.array(second))
    print(f"{name:<28}{np.mean(first):>10.5f}{np.mean(second):>10.5f}"
          f"{gap.mean():>12.6f}{gap.max():>12.6f}")


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), epochs, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    splits = splits_for(data, HORIZON_BARS)
    print(f"{X.shape[1]} columns, horizon {HORIZON_BARS}, {len(SEEDS)} seeds x 2 folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)
    print(f"{'':<28}{'run A':>10}{'run B':>10}{'mean |A-B|':>12}{'worst':>12}")

    default_a, default_b, pinned_a, pinned_b, purged = [], [], [], [], []
    pinned_params = {**LGBM_PARAMS, "deterministic": True, "force_row_wise": True,
                     "n_jobs": PINNED_THREADS}
    comparison_splits = splits_for(data, COMPARISON_PURGE_BARS)
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        default_a += auc_per_fold(X, target, decided, splits, params)
        default_b += auc_per_fold(X, target, decided, splits, params)
        pinned = {**pinned_params, "random_state": seed}
        pinned_a += auc_per_fold(X, target, decided, splits, pinned)
        pinned_b += auc_per_fold(X, target, decided, splits, pinned)
        purged += auc_per_fold(X, target, decided, comparison_splits, params)
        print(f"   seed {seed} done ({time.time() - started:.0f}s)", flush=True)

    print()
    report("default, same seed twice", default_a, default_b)
    report("pinned, same seed twice", pinned_a, pinned_b)
    report(f"purge {HORIZON_BARS} vs purge {COMPARISON_PURGE_BARS}", default_a, purged)
    spread = np.abs(np.array(default_a) - np.array(default_b)).mean()
    print(f"\nEffects smaller than {spread:.5f} AUC cannot be told from this harness's own noise.")


if __name__ == "__main__":
    main()
