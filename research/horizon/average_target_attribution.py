"""Which values drive the average percent change over the next five bars?

Regular trading hours, the clean window, the target the last tests settled on. Each
dev fold fits its own standardisation and its own PLS rotation, and the importances are
averaged across folds -- a feature that only matters in one month is not a driver.

Three readings per feature:
  VIP        its weight in the supervised rotation, scaled so 1.0 is the usual keep line
  loading    its weight in an unsupervised rotation of the same columns (variance only)
  |corr|     its own correlation with the target, ignoring every other column

VIP high with loading low is the interesting case: a column that carries little of the
data's variation but much of what relates to the target. The reverse -- loading high,
VIP low -- is a column that moves a lot and predicts nothing.

usage: average_target_attribution.py [wide]
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from feature_rank import prepare, principal_axes, standardise
from forward_extremes import forward_target
from indicator_features import (EQUITY_INDICATOR_FEATURES, EQUITY_SYMBOLS, MACRO_INDICATOR_FEATURES,
                                MACRO_SYMBOLS, add_indicator_features)
from indicator_test import RETURNS_OF_NEW_DATA
from multi_timeframe_features import ALL_TIMEFRAME_FEATURES, add_timeframe_features
from projection_importance import loading_weight, ranked, variable_importance
from rotation_test import PLS_ROW_STRIDE, SVD_ROW_STRIDE, fold_splits
from segment_walk_forward import load_cut
from short_horizon_walk_forward import FEATURES
from volume_scaled_target import regular_hours

HORIZON_BARS = 5
COMPONENTS = 5
STAGE = "dev"
TOP = 18
KEEP_LINE = 1.0


def build(wide: bool):
    data = load_cut()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    columns = list(FEATURES)
    if wide:
        data = add_timeframe_features(add_indicator_features(data, (*MACRO_SYMBOLS, *EQUITY_SYMBOLS)))
        columns += RETURNS_OF_NEW_DATA + MACRO_INDICATOR_FEATURES + EQUITY_INDICATOR_FEATURES
        columns += ALL_TIMEFRAME_FEATURES
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    return data, columns, forward_target(prices, HORIZON_BARS, "average")


def main() -> None:
    wide = "wide" in sys.argv[1:]
    data, columns, target = build(wide)
    X = data[columns].to_numpy(np.float32)
    print(f"{len(data):,} regular-hours bars, {len(columns)} features, "
          f"target = average percent change over the next {HORIZON_BARS} bars\n")
    vips, loadings, correlations = [], [], []
    for split in fold_splits(data, STAGE):
        rows = split.train[::PLS_ROW_STRIDE]
        rows = rows[np.isfinite(target[rows])]
        fitted = prepare(X[rows])
        standardised = standardise(X[rows], fitted)
        pls = PLSRegression(n_components=COMPONENTS, scale=False).fit(standardised, target[rows])
        vips.append(variable_importance(pls.x_weights_, pls.x_scores_, pls.y_loadings_))
        svd_rows = split.train[::SVD_ROW_STRIDE]
        svd_fitted = prepare(X[svd_rows])
        loadings.append(loading_weight(principal_axes(standardise(X[svd_rows], svd_fitted), COMPONENTS)))
        centred = target[rows] - target[rows].mean()
        spread = standardised.std(axis=0)
        correlations.append(np.abs((standardised * centred[:, None]).mean(axis=0)
                                   / np.maximum(spread * centred.std(), 1e-12)))
    vip, loading, correlation = (np.mean(vips, axis=0), np.mean(loadings, axis=0),
                                 np.mean(correlations, axis=0))
    agreement = np.corrcoef(vips[0], vips[1])[0, 1]
    print(f"VIP agreement between the two folds: {agreement:.3f}\n")
    print(f"{'feature':<32}{'VIP':>8}{'loading':>10}{'|corr|':>9}")
    for name, value in ranked(columns, vip, TOP):
        index = columns.index(name)
        print(f"{name:<32}{value:>8.2f}{loading[index]:>10.3f}{correlation[index]:>9.4f}")
    above = [name for name, value in zip(columns, vip) if value > KEEP_LINE]
    print(f"\n{len(above)} of {len(columns)} features score above {KEEP_LINE}")
    if wide:
        added = [name for name in above if name not in FEATURES]
        print(f"of those, {len(added)} are newly added columns:")
        for name, value in ranked(added, [vip[columns.index(n)] for n in added], TOP):
            print(f"   {name:<32}{value:>8.2f}")
    suffix = "wide" if wide else "core"
    with open(f"average_target_attribution_{suffix}.json", "w") as handle:
        json.dump({"vip": dict(zip(columns, vip.tolist())), "above_keep_line": above,
                   "fold_agreement": agreement}, handle, indent=1)


if __name__ == "__main__":
    main()
