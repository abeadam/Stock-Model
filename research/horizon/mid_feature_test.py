"""Do the mid-based microstructure features earn their place on the honest target?

Target: the average mid percent change over the next five bars, regular hours, clean
window, dev folds. The mid cannot bounce, so nothing here can be the old artifact.

Feature sets are cumulative, to see what each layer adds, and the new columns' VIP is
printed so a gain can be traced to the columns that caused it.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score

from feature_rank import prepare, standardise
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from projection_importance import ranked, variable_importance
from rotation_test import PLS_ROW_STRIDE, fold_splits
from short_horizon_walk_forward import FEATURES, train_direction_model

HORIZON_BARS = 5
COMPONENTS = 5
STAGE = "dev"


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0)
    sets = {"core 44": list(FEATURES),
            "core + returns of added data": list(FEATURES) + RETURNS_OF_NEW_DATA,
            "core + mid microstructure": list(FEATURES) + MID_FEATURES,
            "all three": list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES}
    print(f"{len(data):,} regular-hours bars; target = average MID percent change over "
          f"the next {HORIZON_BARS} bars\n")
    results = {}
    for name, columns in sets.items():
        X = data[columns].to_numpy(np.float32)
        aucs = []
        for split in fold_splits(data, STAGE):
            model = train_direction_model(X, target, split)
            ranked_rows = split.test[decided[split.test]]
            aucs.append(roc_auc_score((target[ranked_rows] > 0).astype(int),
                                      model.predict_proba(X[ranked_rows])[:, 1]))
        results[name] = {"auc": float(np.mean(aucs)), "folds": [round(a, 4) for a in aucs],
                         "features": len(columns)}
        print(f"{name:<32}{len(columns):>4} features  AUC {np.mean(aucs):.4f}  "
              f"folds {results[name]['folds']}", flush=True)

    columns = sets["all three"]
    X = data[columns].to_numpy(np.float32)
    vips = []
    for split in fold_splits(data, STAGE):
        rows = split.train[::PLS_ROW_STRIDE]
        rows = rows[np.isfinite(target[rows])]
        fitted = prepare(X[rows])
        pls = PLSRegression(n_components=COMPONENTS, scale=False).fit(standardise(X[rows], fitted),
                                                                     target[rows])
        vips.append(variable_importance(pls.x_weights_, pls.x_scores_, pls.y_loadings_))
    vip = np.mean(vips, axis=0)
    print("\nTop twelve by VIP across everything:")
    for name, value in ranked(columns, vip, 12):
        marker = "  <- new" if name in MID_FEATURES else ""
        print(f"   {name:<34}{value:>7.2f}{marker}")
    new_above = [(name, float(vip[columns.index(name)])) for name in MID_FEATURES
                 if vip[columns.index(name)] > 1.0]
    print(f"\n{len(new_above)} of {len(MID_FEATURES)} new features score above 1.0")
    with open("mid_feature_test.json", "w") as handle:
        json.dump({"auc": results, "new_above_keep_line": new_above}, handle, indent=1)


if __name__ == "__main__":
    main()
