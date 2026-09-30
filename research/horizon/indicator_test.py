"""Do technical indicators on the added symbols predict ES better than their returns do?

Every symbol added alongside ES -- TLT, USO, VXX, UVXY, GLD, IBIT, SPY, VXM and the ten
index heavyweights -- has entered the model only as short returns, which say how far it
moved but not what state it is in. This measures whether the indicator view of the same
symbols (RSI, Bollinger position, distance from moving averages, volatility ratio, MACD)
adds anything the returns did not.

Same window and target as the earlier feature studies, so the numbers line up with them:
the days that have every data source, bars with a standing quote, the mid-price move over
1 and 5 minutes, one global all-day model, development folds only.

usage: indicator_test.py [feature_set ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from cross_asset_features import CROSS_ASSET_FEATURES
from flow_features import FLOW_FEATURES
from indicator_features import (EQUITY_INDICATOR_FEATURES, EQUITY_SYMBOLS, MACRO_INDICATOR_FEATURES,
                                add_indicator_features)
from macro_features import MACRO_FEATURES, MACRO_SYMBOLS
from quote_features import QUOTE_FEATURES
from segment_walk_forward import (DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, MID_TARGET_BARS,
                                  load_cut, mid_move)
from short_horizon_walk_forward import FEATURES, direction_auc, fold_split, train_direction_model

RETURNS_OF_NEW_DATA = FLOW_FEATURES + CROSS_ASSET_FEATURES + QUOTE_FEATURES + MACRO_FEATURES
FEATURE_SETS = {
    "current_44": FEATURES,
    "returns_only": FEATURES + RETURNS_OF_NEW_DATA,
    "plus_macro_indicators": FEATURES + RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES,
    "plus_all_indicators": FEATURES + RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES + EQUITY_INDICATOR_FEATURES,
    "indicators_without_returns": FEATURES + MACRO_INDICATOR_FEATURES + EQUITY_INDICATOR_FEATURES,
}
STAGE = "dev"


def evaluate(feature_sets: list[str]) -> None:
    started = time.time()
    data = add_indicator_features(load_cut(), (*MACRO_SYMBOLS, *EQUITY_SYMBOLS))
    print(f"{len(data):,} bars over {data['session_day'].nunique()} days "
          f"({data['session_day'].min()}..{data['session_day'].max()}), "
          f"{len(FEATURE_SETS['plus_all_indicators'])} features at most ({time.time() - started:.0f}s)", flush=True)
    rows = []
    for bars in MID_TARGET_BARS:
        move = mid_move(data, bars)
        for name in feature_sets:
            X = data[FEATURE_SETS[name]].to_numpy(np.float32)
            aucs = []
            for fold in FOLDS_BY_STAGE[STAGE]:
                split = fold_split(data["time"], fold, purge_bars=bars,
                                   fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
                model = train_direction_model(X, move, split)
                score = model.predict_proba(X[split.test])[:, 1]
                aucs.append(direction_auc(score, move[split.test]))
            rows.append({"bars": bars, "features": name, "count": len(FEATURE_SETS[name]),
                         "mean_auc": float(np.mean(aucs)), "fold_auc": [round(auc, 4) for auc in aucs]})
            print(f"  {bars:>2} bars  {name:<27} {len(FEATURE_SETS[name]):>3} features  "
                  f"mean AUC {rows[-1]['mean_auc']:.4f}  folds {rows[-1]['fold_auc']}  "
                  f"({time.time() - started:.0f}s)", flush=True)
    with open("indicator_test_dev.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    evaluate(sys.argv[1:] or list(FEATURE_SETS))
