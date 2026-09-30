"""Do indicators read on wider ES bars predict better than the 5-second ones?

A 14-bar MFI on 5-second bars spans 70 seconds and reacts to single prints; on 30-second
bars it spans seven minutes, which is closer to how a chart is read. This scores each
bar width on its own, every width together, and the production features with and without
them, on the clean window and the bounce-free mid target.

usage: multi_timeframe_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from multi_timeframe_features import (ALL_SPAN_FEATURES, ALL_TIMEFRAME_FEATURES, SPAN_FEATURES,
                                      TIMEFRAME_FEATURES, TIMEFRAME_SECONDS, add_span_features,
                                      add_timeframe_features)
from segment_walk_forward import (DATA_END_UTC, FOLD_STARTS_UTC, FOLDS_BY_STAGE, MID_TARGET_BARS,
                                  load_cut, mid_move)
from short_horizon_walk_forward import FEATURES, direction_auc, fold_split, train_direction_model

STAGE = "dev"
FEATURE_SETS = ({"current_44": FEATURES}
                | {f"{seconds}s_wide_bars": TIMEFRAME_FEATURES[seconds] for seconds in TIMEFRAME_SECONDS}
                | {f"{seconds}s_span_5s_bars": SPAN_FEATURES[seconds] for seconds in TIMEFRAME_SECONDS}
                | {"all_widths": ALL_TIMEFRAME_FEATURES, "all_spans": ALL_SPAN_FEATURES,
                   "current_44_plus_all_widths": FEATURES + ALL_TIMEFRAME_FEATURES,
                   "current_44_plus_all_spans": FEATURES + ALL_SPAN_FEATURES})


def evaluate() -> None:
    started = time.time()
    data = add_span_features(add_timeframe_features(load_cut()))
    print(f"{len(data):,} bars over {data['session_day'].nunique()} days, "
          f"{len(ALL_TIMEFRAME_FEATURES)} timeframe features ({time.time() - started:.0f}s)", flush=True)
    rows = []
    for bars in MID_TARGET_BARS:
        move = mid_move(data, bars)
        for name, columns in FEATURE_SETS.items():
            X = data[columns].to_numpy(np.float32)
            aucs = []
            for fold in FOLDS_BY_STAGE[STAGE]:
                split = fold_split(data["time"], fold, purge_bars=bars,
                                   fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
                model = train_direction_model(X, move, split)
                aucs.append(direction_auc(model.predict_proba(X[split.test])[:, 1], move[split.test]))
            rows.append({"bars": bars, "features": name, "count": len(columns),
                         "mean_auc": float(np.mean(aucs)), "fold_auc": [round(auc, 4) for auc in aucs]})
            print(f"  {bars:>2} bars  {name:<28}{len(columns):>4} features  "
                  f"mean AUC {rows[-1]['mean_auc']:.4f}  folds {rows[-1]['fold_auc']}  "
                  f"({time.time() - started:.0f}s)", flush=True)
    with open("multi_timeframe_test_dev2.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    evaluate()
