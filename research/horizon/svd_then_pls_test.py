"""Denoise with SVD first, then find supervised directions inside what survives.

PLS searches all 44 dimensions for covariance with the move, including directions the
features barely vary along. Such a direction is poorly determined -- a handful of rows
fix where it points -- so it is the easiest place for PLS to find a correlation that
holds in the training months and nowhere else. That is the shape of the failure the
holdout just showed: dev +0.0057, holdout +0.0002.

Truncating to the leading principal components first removes those directions before PLS
can reach them. Note this only does anything *because* it truncates: PLS is unchanged by
an orthogonal rotation of its inputs, so with every component kept, SVD then PLS is just
PLS. The regularisation is entirely in what the first step throws away.

Dev folds only. A dev win here is not evidence -- PLS-5 already won dev and lost the
holdout -- it only shows whether truncation changes the behaviour at all.

usage: svd_then_pls_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from feature_rank import prepare, principal_axes, standardise
from rotation_test import (PLS_ROW_STRIDE, SVD_ROW_STRIDE, TARGET_BARS, pls_transform,
                           score_raw, score_rotated)
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES

STAGE = "dev"
# (principal components kept, supervised columns taken from them)
CHAINS = ((25, 5), (19, 5), (10, 5), (25, 2), (10, 2))


def chained_transform(svd_components: int):
    """A transform that projects onto `svd_components` axes, then runs PLS inside them."""
    def transform(X, move, split, components):
        sample_rows = split.train[::SVD_ROW_STRIDE]
        svd_fitted = prepare(X[sample_rows])
        axes = principal_axes(standardise(X[sample_rows], svd_fitted), svd_components)
        pls_rows = split.train[::PLS_ROW_STRIDE]
        pls_rows = pls_rows[np.isfinite(move[pls_rows])]
        denoised = standardise(X[pls_rows], svd_fitted) @ axes
        pls = PLSRegression(n_components=components, scale=False)
        pls.fit(denoised, np.sign(move[pls_rows]))
        return lambda rows: pls.transform(standardise(X[rows], svd_fitted) @ axes)
    return transform


def main() -> None:
    started = time.time()
    data = load_cut()
    move = mid_move(data, TARGET_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    rows = []
    mean, folds = score_raw(X, move, data, STAGE)
    print(f"{'raw 44':<22}AUC {mean:.4f}  folds {folds}", flush=True)
    for components in (2, 5):
        mean, folds = score_rotated(X, move, data, pls_transform, components, STAGE)
        print(f"{f'pls {components} direct':<22}AUC {mean:.4f}  folds {folds}", flush=True)
        rows.append({"chain": f"pls_{components}", "mean_auc": mean, "fold_auc": folds})
    for svd_components, pls_components in CHAINS:
        mean, folds = score_rotated(X, move, data, chained_transform(svd_components),
                                    pls_components, STAGE)
        rows.append({"chain": f"svd_{svd_components}_pls_{pls_components}",
                     "mean_auc": mean, "fold_auc": folds})
        print(f"{f'svd {svd_components} -> pls {pls_components}':<22}AUC {mean:.4f}  folds {folds}  "
              f"({time.time() - started:.0f}s)", flush=True)
    with open("svd_then_pls_test.json", "w") as handle:
        json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main()
