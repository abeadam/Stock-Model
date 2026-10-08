"""Is today's 163-column set better than the 153 that stood before it?

The earlier ablation removed the six divergence columns but left the dollar's four plain
return columns in place, so it never compared against the set that actually stood before
today. Same paired design: one seed and one fold at a time, both sides given the same
randomness, difference taken.

  full        163: everything, including the dollar's returns and the six new divergences
  previous    153: the set that passed the holdout, before the dollar and the oil windows
  oil_only    157: the previous set plus the three oil divergence columns
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

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
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    full = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    oil = [name for name in full if "uso_minus_mid" in name]
    added_today = [name for name in full if "uup" in name or "uso_minus_mid" in name]
    sets = {"full": full,
            "previous": [name for name in full if name not in added_today],
            "oil_only": [name for name in full if name not in added_today or name in oil]}
    print(f"sizes: " + ", ".join(f"{name} {len(columns)}" for name, columns in sets.items())
          + f"; {len(SEEDS)} seeds x 2 folds\n")
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
        print(f"   seed {seed}: " + "  ".join(f"{name} {np.mean(scores[name][-2:]):.4f}"
                                              for name in sets), flush=True)
    print(f"\n{'comparison':<34}{'mean AUC':>10}{'difference':>12}{'std err':>10}{'t':>7}")
    previous = np.array(scores["previous"])
    print(f"{'previous (153 columns)':<34}{previous.mean():>10.4f}")
    for name in ("oil_only", "full"):
        values = np.array(scores[name])
        difference = values - previous
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        print(f"{name + ' vs previous':<34}{values.mean():>10.4f}{difference.mean():>+12.5f}"
              f"{error:>10.5f}{difference.mean() / error if error > 0 else 0:>7.2f}")
    with open("ablation_versus_previous.json", "w") as handle:
        json.dump({name: list(map(float, values)) for name, values in scores.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
