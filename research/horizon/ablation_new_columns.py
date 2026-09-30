"""Do the oil and dollar columns add anything, given everything else already there?

Comparing two feature sets once tells you very little when the effect might be 0.001 and
the spread between two months is 0.003. The model itself is random -- it subsamples rows
and columns -- so the same configuration scores differently from one seed to the next, and
that randomness is the ruler.

So each configuration is trained many times with different seeds on the same folds, and
the comparison is **paired**: for every (seed, fold) the same randomness is given to both
sides and the difference is taken. The average of those differences, against their own
spread, says whether removing the columns costs anything real.

  full        163 columns, everything
  no_new      the six oil and dollar divergence columns removed
  no_uso      only the three oil columns removed
  no_uup      only the three dollar columns removed

A negative difference for "full minus no_new" means the columns are worth keeping.
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
    dollar = [name for name in full if "uup_minus_mid" in name]
    sets = {"full": full,
            "no_new": [name for name in full if name not in oil + dollar],
            "no_uso": [name for name in full if name not in oil],
            "no_uup": [name for name in full if name not in dollar]}
    print(f"removing {len(oil)} oil and {len(dollar)} dollar columns from {len(full)}; "
          f"{len(SEEDS)} seeds x {len(fold_splits(data, STAGE))} folds\n")
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
    print(f"\n{'comparison':<26}{'mean AUC':>10}{'difference':>12}{'std err':>10}{'t':>7}{'verdict':>26}")
    baseline = np.array(scores["full"])
    print(f"{'full (163 columns)':<26}{baseline.mean():>10.4f}")
    for name in ("no_new", "no_uso", "no_uup"):
        other = np.array(scores[name])
        difference = baseline - other            # positive means the columns help
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        t = difference.mean() / error if error > 0 else 0.0
        verdict = ("columns earn their place" if t > 2 else
                   "columns cost something" if t < -2 else "no measurable difference")
        print(f"{'vs ' + name:<26}{other.mean():>10.4f}{difference.mean():>+12.5f}"
              f"{error:>10.5f}{t:>7.2f}{verdict:>26}")
    with open("ablation_new_columns.json", "w") as handle:
        json.dump({name: list(map(float, values)) for name, values in scores.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
