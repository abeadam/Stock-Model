"""Do the indicator families rejected on the old target earn a place on the new one?

Three families -- technical indicators on the macro ETFs (42), on the equities (84), and
on widened bars (48) -- were dropped because each lost its test. Every one of those tests
used the old target: all day, endpoint, trade-based. The added-symbol returns failed under
those same conditions and then gained +0.011 once the target became the regular-hours
average mid, so a rejection made there does not carry over without checking.

Dev folds, the average mid move over five bars, regular hours.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from indicator_features import (EQUITY_INDICATOR_FEATURES, EQUITY_SYMBOLS, MACRO_INDICATOR_FEATURES,
                                MACRO_SYMBOLS, add_indicator_features)
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from multi_timeframe_features import ALL_TIMEFRAME_FEATURES, add_timeframe_features
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model

HORIZON_BARS = 5
STAGE = "dev"


def main() -> None:
    data = add_timeframe_features(add_indicator_features(build_data(), (*MACRO_SYMBOLS, *EQUITY_SYMBOLS)))
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    sets = {"153 (current best)": base,
            "+ macro indicators": base + MACRO_INDICATOR_FEATURES,
            "+ equity indicators": base + EQUITY_INDICATOR_FEATURES,
            "+ timeframe indicators": base + ALL_TIMEFRAME_FEATURES,
            "327 (everything)": base + MACRO_INDICATOR_FEATURES + EQUITY_INDICATOR_FEATURES
                                + ALL_TIMEFRAME_FEATURES,
            "303 (no mid features)": (list(FEATURES) + RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES
                                      + EQUITY_INDICATOR_FEATURES + ALL_TIMEFRAME_FEATURES)}
    print(f"{len(data):,} regular-hours bars; target = average mid over {HORIZON_BARS} bars\n")
    results = {}
    for name, columns in sets.items():
        X = data[columns].to_numpy(np.float32)
        aucs = []
        for split in fold_splits(data, STAGE):
            model = train_direction_model(X, target, split)
            test = split.test[decided[split.test]]
            aucs.append(roc_auc_score((target[test] > 0).astype(int),
                                      model.predict_proba(X[test])[:, 1]))
        results[name] = {"features": len(columns), "auc": float(np.mean(aucs)),
                         "folds": [round(a, 4) for a in aucs]}
        print(f"{name:<26}{len(columns):>5} features  AUC {np.mean(aucs):.4f}  "
              f"folds {results[name]['folds']}", flush=True)
    with open("full_feature_set_test.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
