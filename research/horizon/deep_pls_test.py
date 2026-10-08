"""The Polson-Sokolov-Xu nonlinear PLS variants, on ES, dev folds only.

Every model is fitted inside its own fold -- standardisation, rotation, link and all --
and scored on that fold's test rows with the production convention (zero-move bars
dropped). Baselines from earlier work are reprinted for comparison.

usage: deep_pls_test.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from deep_pls import apply_link, fit_link, least_squares_direction, link_monotonicity, relu_expand
from feature_rank import prepare, standardise
from rotation_test import PLS_ROW_STRIDE, TARGET_BARS, fold_splits
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES

STAGE = "holdout" if False else "dev"
COMPONENTS = 5
MLP_HIDDEN = (16, 8)
MLP_ROW_STRIDE = 200


def scores_of(X, move, split):
    """PLS scores, with the rotation fitted on this fold's training rows only."""
    rows = split.train[::PLS_ROW_STRIDE]
    rows = rows[np.isfinite(move[rows]) & (move[rows] != 0)]
    fitted = prepare(X[rows])
    pls = PLSRegression(n_components=COMPONENTS, scale=False)
    pls.fit(standardise(X[rows], fitted), np.sign(move[rows]))
    return rows, (lambda wanted: pls.transform(standardise(X[wanted], fitted)))


def evaluate() -> None:
    started = time.time()
    data = load_cut()
    move = mid_move(data, TARGET_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    decided = np.isfinite(move) & (move != 0)
    results = {name: [] for name in ("pls_linear", "pls_relu", "deep_pls_2", "pls_mlp",
                                     "brillinger_linear", "brillinger_link")}
    monotonicity = []
    for split in fold_splits(data, STAGE):
        train_rows, transform = scores_of(X, move, split)
        test = split.test[decided[split.test]]
        train_scores, test_scores = transform(train_rows), transform(test)
        up_train = (move[train_rows] > 0).astype(int)
        up_test = (move[test] > 0).astype(int)

        linear = LogisticRegression(max_iter=1000).fit(train_scores, up_train)
        results["pls_linear"].append(roc_auc_score(up_test, linear.predict_proba(test_scores)[:, 1]))

        relu = LogisticRegression(max_iter=1000).fit(relu_expand(train_scores), up_train)
        results["pls_relu"].append(roc_auc_score(up_test, relu.predict_proba(relu_expand(test_scores))[:, 1]))

        # Second layer: PLS again on the ReLU features, then another ReLU and a read-out.
        second = PLSRegression(n_components=COMPONENTS, scale=False)
        second.fit(relu_expand(train_scores), np.sign(move[train_rows]))
        deep_train = relu_expand(second.transform(relu_expand(train_scores)))
        deep_test = relu_expand(second.transform(relu_expand(test_scores)))
        deep = LogisticRegression(max_iter=1000).fit(deep_train, up_train)
        results["deep_pls_2"].append(roc_auc_score(up_test, deep.predict_proba(deep_test)[:, 1]))

        sample = slice(None, None, max(1, len(train_scores) // (len(train_scores) // MLP_ROW_STRIDE or 1)))
        network = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=60, random_state=3)
        network.fit(train_scores[sample], up_train[sample])
        results["pls_mlp"].append(roc_auc_score(up_test, network.predict_proba(test_scores)[:, 1]))

        # Brillinger: the direction straight from OLS on the raw features, then a
        # non-parametric link fitted on the training index.
        fitted = prepare(X[train_rows])
        direction = least_squares_direction(standardise(X[train_rows], fitted), np.sign(move[train_rows]))
        train_index = standardise(X[train_rows], fitted) @ direction
        test_index = standardise(X[test], fitted) @ direction
        results["brillinger_linear"].append(roc_auc_score(up_test, test_index))
        centres, means = fit_link(train_index, up_train)
        monotonicity.append(link_monotonicity(means))
        results["brillinger_link"].append(roc_auc_score(up_test, apply_link(test_index, centres, means)))

    print(f"stage {STAGE}, {COMPONENTS} components ({time.time() - started:.0f}s)\n")
    for name, aucs in results.items():
        print(f"{name:<20}AUC {np.mean(aucs):.4f}  folds {[round(a, 4) for a in aucs]}")
    print(f"\nfitted link monotonicity per fold: {[round(m, 3) for m in monotonicity]}  "
          f"(+1 = only rises, so ranking unchanged)")
    with open("deep_pls_test.json", "w") as handle:
        json.dump({"results": {k: v for k, v in results.items()}, "monotonicity": monotonicity}, handle, indent=1)


if __name__ == "__main__":
    evaluate()
