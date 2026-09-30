"""How much of the regular-hours frame the volatility series actually cover, and how
long one model fit takes -- the two numbers that decide whether the depth x horizon grid
is affordable before any of it is run.

usage: volatility_coverage.py
"""
from __future__ import annotations

import time

import numpy as np

from cross_asset_features import load_symbol_closes
from mid_feature_support import build_data
from mid_versus_trade_target import average_target
from rotation_test import fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model
from volatility_features import VOLATILITY_SERIES, add_volatility_features, lookback_ladder

HORIZON_BARS = 5
STAGE = "dev"


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    print(f"{len(data):,} regular-hours bars over {data['session_day'].nunique()} days "
          f"({data['session_day'].min()}..{data['session_day'].max()}), {time.time() - started:.0f}s\n")

    closes = {series: load_symbol_closes(series) for series in VOLATILITY_SERIES}
    print(f"{'series':<8}{'bars downloaded':>18}{'matching an ES bar':>22}{'share':>9}")
    for series, series_closes in closes.items():
        matched = series_closes.reindex(epochs).notna().sum()
        print(f"{series:<8}{len(series_closes):>18,}{matched:>22,}{matched / len(data):>8.1%}")

    ladder = lookback_ladder(depth=1)
    built = add_volatility_features(data, closes, ladder)
    target = average_target(built["mid"].to_numpy(float), epochs, HORIZON_BARS)
    columns = list(FEATURES)
    X = built[columns].to_numpy(np.float32)
    split = fold_splits(built, STAGE)[0]
    print(f"\nfold 1: {len(split.train):,} train, {len(split.validation):,} validation, "
          f"{len(split.test):,} test rows")
    started = time.time()
    train_direction_model(X, target, split, LGBM_PARAMS)
    print(f"one fit on {len(columns)} columns: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
