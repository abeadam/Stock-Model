"""Does the VIX complex add anything to the 163 columns already there?

Paired against the same seeds and folds, as everything else here is: comparing two feature
sets once reads the model's own randomness rather than the feature. Three arms:

  base          the 163 columns the mid-microstructure holdout was won with
  plus_traded   base + VXX and UVXY, which are ordinary stocks and can be streamed live
  plus_index    plus_traded + the VIX index, which this account gets delayed and so could
                not trade on -- carried only to show what the untradeable version is worth

A positive "plus_traded minus base" means the funds earn their place. The second
comparison is the one that says whether the subscription would be worth buying.

usage: ablation_volatility_series.py
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
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model
from volatility_features import (SHALLOW_LADDER_DEPTH, TRADED_VOLATILITY_SERIES, VOLATILITY_SERIES,
                                 add_volatility_features, lookback_ladder, volatility_feature_names)

HORIZON_BARS = 5
STAGE = "dev"
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RESULTS_PATH = "ablation_volatility_series.json"


def fold_scores(data, columns, target, decided, params) -> list[float]:
    X = data[columns].to_numpy(np.float32)
    scores = []
    for split in fold_splits(data, STAGE):
        model = train_direction_model(X, target, split, params)
        test = split.test[decided[split.test]]
        scores.append(roc_auc_score((target[test] > 0).astype(int), model.predict_proba(X[test])[:, 1]))
    return scores


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    ladder = lookback_ladder(SHALLOW_LADDER_DEPTH)
    data = add_volatility_features(data, {series: load_symbol_closes(series)
                                          for series in VOLATILITY_SERIES}, ladder)
    target = average_target(data["mid"].to_numpy(float), epochs, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)

    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    traded = volatility_feature_names(ladder, TRADED_VOLATILITY_SERIES)
    index_only = volatility_feature_names(ladder, ("VIX",))
    sets = {"base": base, "plus_traded": base + traded, "plus_index": base + traded + index_only}
    print(f"ladder {ladder}, {len(traded)} traded columns and {len(index_only)} index columns "
          f"on top of {len(base)}; {len(SEEDS)} seeds x {len(fold_splits(data, STAGE))} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    scores = {name: [] for name in sets}
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        for name, columns in sets.items():
            scores[name] += fold_scores(data, columns, target, decided, params)
        print(f"   seed {seed}: " + "  ".join(f"{name} {np.mean(scores[name][-2:]):.4f}"
                                              for name in sets)
              + f"   ({time.time() - started:.0f}s)", flush=True)

    print(f"\n{'comparison':<34}{'mean AUC':>10}{'difference':>12}{'std err':>10}{'t':>7}  verdict")
    for name, against in (("base", None), ("plus_traded", "base"), ("plus_index", "plus_traded")):
        mean = np.mean(scores[name])
        if against is None:
            print(f"{name + f' ({len(sets[name])} columns)':<34}{mean:>10.4f}")
            continue
        difference = np.array(scores[name]) - np.array(scores[against])
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        t = difference.mean() / error if error > 0 else 0.0
        verdict = ("earns its place" if t > 2 else "costs something" if t < -2 else "no measurable difference")
        print(f"{name + ' vs ' + against:<34}{mean:>10.4f}{difference.mean():>+12.5f}"
              f"{error:>10.5f}{t:>7.2f}  {verdict}")
    with open(RESULTS_PATH, "w") as handle:
        json.dump({name: list(map(float, values)) for name, values in scores.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
