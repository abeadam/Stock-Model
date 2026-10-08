"""Rank the candidate columns, so only the few worth an ablation get one.

VIP against the average mid move, fitted per fold and averaged, alongside each column's
own correlation with the target. A column has to earn attention on both folds, so the
weaker fold's VIP is reported too: a candidate that only matters in June is a candidate
that only matters in June.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from candidate_columns import add_candidates, candidate_names
from feature_rank import prepare, standardise
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from projection_importance import variable_importance
from rotation_test import PLS_ROW_STRIDE, fold_splits
from short_horizon_walk_forward import FEATURES

HORIZON_BARS = 5
COMPONENTS = 5
STAGE = "dev"
KEEP_LINE = 1.0
TOP = 14


def main() -> None:
    data = add_candidates(build_data())
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    columns = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES + candidate_names()
    X = data[columns].to_numpy(np.float32)
    print(f"{len(columns)} columns, of which {len(candidate_names())} are candidates\n")
    vips, correlations = [], []
    for split in fold_splits(data, STAGE):
        rows = split.train[::PLS_ROW_STRIDE]
        rows = rows[np.isfinite(target[rows])]
        fitted = prepare(X[rows])
        standardised = standardise(X[rows], fitted)
        pls = PLSRegression(n_components=COMPONENTS, scale=False).fit(standardised, target[rows])
        vips.append(variable_importance(pls.x_weights_, pls.x_scores_, pls.y_loadings_))
        centred = target[rows] - target[rows].mean()
        correlations.append(np.abs((standardised * centred[:, None]).mean(axis=0)
                                   / np.maximum(standardised.std(axis=0) * centred.std(), 1e-12)))
    mean_vip, weakest = np.mean(vips, axis=0), np.min(vips, axis=0)
    correlation = np.mean(correlations, axis=0)
    candidates = [(name, mean_vip[columns.index(name)], weakest[columns.index(name)],
                   correlation[columns.index(name)]) for name in candidate_names()]
    candidates.sort(key=lambda row: -row[2])              # rank on the weaker fold
    print(f"{'candidate':<38}{'VIP':>7}{'weaker fold':>13}{'|corr|':>9}")
    for name, vip, low, corr in candidates[:TOP]:
        print(f"{name:<38}{vip:>7.2f}{low:>13.2f}{corr:>9.4f}")
    above = [row for row in candidates if row[2] > KEEP_LINE]
    print(f"\n{len(above)} of {len(candidates)} candidates clear {KEEP_LINE} on BOTH folds")
    with open("screen_candidates.json", "w") as handle:
        json.dump([{"name": n, "vip": float(v), "weaker": float(l), "corr": float(c)}
                   for n, v, l, c in candidates], handle, indent=1)


if __name__ == "__main__":
    main()
