"""Paired ablation of the families the screen turned up.

The screen ranks; only this decides. Each family is added to the current model and both
sides are trained with the same seeds on the same folds, so the difference is not read
through the model's own randomness.

  dispersion  how far apart the ten mega-cap returns are in a bar (2 columns)
  scaled      the divergences already used, divided by recent mid movement (15 columns)
  screened    every candidate that cleared the keep line on both folds (15 columns)
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from candidate_columns import add_candidates
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model

HORIZON_BARS = 5
STAGE = "dev"
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)


def main() -> None:
    data = add_candidates(build_data())
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    screened = [row["name"] for row in json.load(open("screen_candidates.json")) if row["weaker"] > 1.0]
    dispersion = [name for name in screened if "dispersion" in name]
    scaled = [name for name in data.columns if "_scaled_" in name]
    sets = {"base": base, "plus_dispersion": base + dispersion,
            "plus_scaled": base + scaled, "plus_screened": base + screened}
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
    print(f"\n{'family':<20}{'columns':>9}{'mean AUC':>10}{'difference':>12}{'std err':>10}{'t':>7}")
    print(f"{'base':<20}{len(base):>9}{baseline.mean():>10.4f}")
    for name in ("plus_dispersion", "plus_scaled", "plus_screened"):
        values = np.array(scores[name])
        difference = values - baseline
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        print(f"{name:<20}{len(sets[name]):>9}{values.mean():>10.4f}{difference.mean():>+12.5f}"
              f"{error:>10.5f}{difference.mean() / error if error > 0 else 0:>7.2f}")
    with open("ablation_candidates.json", "w") as handle:
        json.dump({name: list(map(float, values)) for name, values in scores.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
