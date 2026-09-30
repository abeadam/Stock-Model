"""Is the signal linear? Then the trees, not the features, were the limitation.

Partial least squares is a linear construction, and handing its five columns to a
boosted tree beat handing the tree all 44 raw features. That points at a plain
question: how much of the signal is simply linear? A logistic regression on the same
44 features answers it, and the PLS regression's own linear prediction answers it
without any tree at all.

Same window, folds, target and leak-free fold-local fitting as rotation_test.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score

from feature_rank import prepare, standardise
from rotation_test import PLS_ROW_STRIDE, TARGET_BARS, fold_splits
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES

FIT_ROW_STRIDE = 20


def evaluate() -> None:
    data = load_cut()
    move = mid_move(data, TARGET_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    logistic, pls_linear = [], []
    for split in fold_splits(data):
        # Exactly the convention of train_direction_model and direction_auc:
        # a bar whose mid did not move has no direction to learn or to score.
        decided = np.isfinite(move) & (move != 0)
        rows = split.train[::FIT_ROW_STRIDE]
        rows = rows[decided[rows]]
        fitted = prepare(X[rows])
        train_X, train_y = standardise(X[rows], fitted), (move[rows] > 0).astype(int)
        test = split.test[decided[split.test]]
        test_X, test_up = standardise(X[test], fitted), (move[test] > 0).astype(int)

        model = LogisticRegression(max_iter=2000, C=0.1).fit(train_X, train_y)
        logistic.append(roc_auc_score(test_up, model.predict_proba(test_X)[:, 1]))

        pls = PLSRegression(n_components=5, scale=False).fit(train_X, np.sign(move[rows]))
        pls_linear.append(roc_auc_score(test_up, pls.predict(test_X).ravel()))
    for name, aucs in (("logistic regression, 44 features", logistic),
                       ("PLS-5 linear prediction, no tree", pls_linear)):
        print(f"{name:<36}AUC {np.mean(aucs):.4f}  folds {[round(a, 4) for a in aucs]}", flush=True)


if __name__ == "__main__":
    evaluate()
