"""The same rotations, applied to every feature rather than just the production 44.

The wide set -- 44 production features, the returns of the added symbols, their technical
indicators, and the multi-timeframe indicators -- is 303 columns spanning about 196
independent directions, so it is where compression has the most to remove. It is also
where an unsupervised rotation did worst (0.5071 raw -> 0.5029 at 196 components), which
is the argument for trying a supervised one: PLS drops a direction for carrying no
signal, not for being narrow.

Dev folds only, same window and target as every other feature study, every transform
fitted inside its own fold.

usage: wide_rotation_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from indicator_features import (EQUITY_INDICATOR_FEATURES, EQUITY_SYMBOLS, MACRO_INDICATOR_FEATURES,
                                MACRO_SYMBOLS, add_indicator_features)
from indicator_test import RETURNS_OF_NEW_DATA
from multi_timeframe_features import ALL_TIMEFRAME_FEATURES, add_timeframe_features
from rotation_test import TARGET_BARS, pls_transform, score_raw, score_rotated, svd_transform
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES
from svd_then_pls_test import chained_transform

STAGE = "dev"
PLS_COMPONENTS = (2, 5, 10, 25)
SVD_COMPONENTS = (25, 136)
CHAINS = ((136, 5), (136, 25), (25, 5))


def main() -> None:
    started = time.time()
    data = add_timeframe_features(add_indicator_features(load_cut(), (*MACRO_SYMBOLS, *EQUITY_SYMBOLS)))
    columns = (FEATURES + RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES
               + EQUITY_INDICATOR_FEATURES + ALL_TIMEFRAME_FEATURES)
    move = mid_move(data, TARGET_BARS)
    X = data[columns].to_numpy(np.float32)
    print(f"{len(columns)} features, {len(data):,} bars ({time.time() - started:.0f}s)\n", flush=True)
    rows = []

    def record(label, scored):
        mean, folds = scored
        rows.append({"model": label, "mean_auc": mean, "fold_auc": folds})
        print(f"{label:<26}AUC {mean:.4f}  folds {folds}  ({time.time() - started:.0f}s)", flush=True)

    record(f"raw {len(columns)}", score_raw(X, move, data, STAGE))
    for components in SVD_COMPONENTS:
        record(f"svd {components}", score_rotated(X, move, data, svd_transform, components, STAGE))
    for components in PLS_COMPONENTS:
        record(f"pls {components}", score_rotated(X, move, data, pls_transform, components, STAGE))
    for svd_components, pls_components in CHAINS:
        record(f"svd {svd_components} -> pls {pls_components}",
               score_rotated(X, move, data, chained_transform(svd_components), pls_components, STAGE))
    with open("wide_rotation_test.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()
