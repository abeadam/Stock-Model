"""Do over-extension and the release clock earn their place?

Two ideas, tested the same way everything else has been: eight seeds, same folds, same
randomness on both sides, differences taken pair by pair.

  run      how far this move has already carried, against the session's own distribution.
           The study behind it: a move that has run six bars gives back $3.41 over the
           next 25 seconds, against $0.43 for a move that has run one.
  clock    minutes from the 10:00 and 14:00 release slots. The study behind it: 10:00 is
           the fourth busiest minute of the session, at twice the average bar.
  both     together, since a stretched move at 10:01 is a different thing from a
           stretched move at 11:47.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from release_clock_features import RELEASE_FEATURES, add_release_features
from rotation_test import fold_splits
from run_features import RUN_FEATURES, add_run_features
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model

HORIZON_BARS = 5
STAGE = "dev"
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)


def main() -> None:
    data = add_release_features(add_run_features(build_data()))
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    sets = {"base": base, "plus_run": base + list(RUN_FEATURES),
            "plus_clock": base + list(RELEASE_FEATURES),
            "plus_both": base + list(RUN_FEATURES) + list(RELEASE_FEATURES)}
    print("  ".join(f"{name} {len(columns)}" for name, columns in sets.items()) + "\n")
    scores = {name: [] for name in sets}
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        for name, columns in sets.items():
            X = data[columns].to_numpy(np.float32)
            for split in fold_splits(data, STAGE):
                model = train_direction_model(X, target, split, params)
                test = split.test[decided[split.test]]
                scores[name].append(roc_auc_score((target[test] > 0).astype(int),
                                                  model.predict_proba(X[test])[:, 1]))
        print(f"   seed {seed}: " + "  ".join(f"{n} {np.mean(scores[n][-2:]):.4f}" for n in sets),
              flush=True)
    baseline = np.array(scores["base"])
    print(f"\n{'features':<14}{'columns':>9}{'mean AUC':>10}{'difference':>12}{'std err':>10}{'t':>7}")
    print(f"{'base':<14}{len(base):>9}{baseline.mean():>10.4f}")
    for name in ("plus_run", "plus_clock", "plus_both"):
        values = np.array(scores[name])
        gap = values - baseline
        error = gap.std(ddof=1) / np.sqrt(len(gap))
        print(f"{name:<14}{len(sets[name]):>9}{values.mean():>10.4f}{gap.mean():>+12.5f}"
              f"{error:>10.5f}{gap.mean() / error:>7.2f}")
    with open("run_clock_ablation.json", "w") as handle:
        json.dump({k: list(map(float, v)) for k, v in scores.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
