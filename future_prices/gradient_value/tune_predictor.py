"""Hyperparameter search for the gradient direction classifier.

The deployed model used fixed, untuned LightGBM parameters. This runs a random
search on a chronological train/validation split (selecting by validation AUC,
which governs how reliable the tail probabilities the strategy trades on are),
then retrains the best configuration on the full 80% train block and reports it
on the untouched 20% test block. The winning model overwrites predictor_model.pkl.

Outputs: predictor_model.pkl (retuned), tune_results.txt
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from train_predictor import TARGET_COL, _BEST_PLUS_ASI as FEATURES

_HERE = Path(__file__).parent
INPUT_CSV = _HERE / "model_input.csv"
MODEL_PKL = _HERE / "predictor_model.pkl"
RESULTS_TXT = _HERE / "tune_results.txt"

TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_TRAIN = 0.20  # of the 80% train block
SEARCH_TRAIN_ROWS = 1_200_000        # most-recent train rows used per search fit (speed)
N_CANDIDATES = 24
EARLY_STOPPING_ROUNDS = 50
MAX_TREES = 4000
RANDOM_SEED = 42

# Fixed across all candidates.
_FIXED_PARAMS = dict(objective="binary", n_jobs=-1, verbose=-1, random_state=RANDOM_SEED)

_SEARCH_SPACE = {
    "num_leaves": [31, 63, 127, 255, 511],
    "max_depth": [-1, 4, 6, 8, 12],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
    "min_data_in_leaf": [50, 100, 200, 500, 1000],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_l1": [0.0, 0.01, 0.05, 0.1, 0.5],
    "lambda_l2": [0.0, 0.01, 0.05, 0.1, 0.5],
}


def load_splits() -> dict:
    """Chronological train / validation / test feature matrices and labels."""
    columns = ["DateTime", *FEATURES, TARGET_COL]
    df = pd.read_csv(INPUT_CSV, usecols=columns)
    df = df.drop_duplicates(subset="DateTime", keep="first").reset_index(drop=True)
    df = df.dropna(subset=[*FEATURES, TARGET_COL]).reset_index(drop=True)

    y = (df[TARGET_COL].to_numpy() > 0).astype(np.int32)
    X = df[FEATURES]

    test_start = int(len(df) * (1 - TEST_FRACTION))
    val_start = int(test_start * (1 - VALIDATION_FRACTION_OF_TRAIN))
    return {
        "X_train": X.iloc[:val_start], "y_train": y[:val_start],
        "X_val": X.iloc[val_start:test_start], "y_val": y[val_start:test_start],
        "X_test": X.iloc[test_start:], "y_test": y[test_start:],
        "X_train_full": X.iloc[:test_start], "y_train_full": y[:test_start],
    }


def sample_params(rng: random.Random) -> dict:
    return {name: rng.choice(choices) for name, choices in _SEARCH_SPACE.items()}


def evaluate_on(model: lgb.LGBMClassifier, X, y) -> dict:
    prob = model.predict_proba(X)[:, 1]
    return {
        "auc": float(roc_auc_score(y, prob)),
        "logloss": float(log_loss(y, prob)),
        "dir_acc": float(np.mean((prob >= 0.5).astype(int) == y) * 100),
    }


def search(splits: dict) -> list[dict]:
    """Random search; each fit early-stops on validation AUC."""
    rng = random.Random(RANDOM_SEED)
    recent = splits["X_train"].tail(SEARCH_TRAIN_ROWS)
    y_recent = splits["y_train"][-len(recent):]
    X_val, y_val = splits["X_val"], splits["y_val"]

    results = []
    for candidate in range(1, N_CANDIDATES + 1):
        params = sample_params(rng)
        model = lgb.LGBMClassifier(n_estimators=MAX_TREES, **params, **_FIXED_PARAMS)
        started = time.time()
        model.fit(
            recent, y_recent,
            eval_set=[(X_val, y_val)], eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        val_auc = float(model.best_score_["valid_0"]["auc"])
        results.append({"params": params, "val_auc": val_auc,
                        "best_iteration": int(model.best_iteration_)})
        print(f"  [{candidate:2d}/{N_CANDIDATES}] val_auc={val_auc:.4f} "
              f"trees={model.best_iteration_:4d} ({time.time()-started:.0f}s) {params}")
    return sorted(results, key=lambda r: r["val_auc"], reverse=True)


def main() -> None:
    print("Loading splits ...")
    splits = load_splits()
    print(f"  train={len(splits['X_train']):,}  val={len(splits['X_val']):,}  "
          f"test={len(splits['X_test']):,}  ({len(FEATURES)} features)")

    print("\nRandom search (selecting by validation AUC) ...")
    ranked = search(splits)
    best = ranked[0]
    print(f"\nBest validation AUC={best['val_auc']:.4f} at {best['best_iteration']} trees: {best['params']}")

    # Retrain the winner on the full 80% train block (train + val), then evaluate
    # on the untouched test block. Scale trees up slightly for the larger data.
    n_trees = max(100, int(best["best_iteration"] * len(splits["X_train_full"]) / max(1, len(splits["X_train"].tail(SEARCH_TRAIN_ROWS)))))
    n_trees = min(n_trees, MAX_TREES)
    print(f"\nRetraining best on full train block with {n_trees} trees ...")
    final_model = lgb.LGBMClassifier(n_estimators=n_trees, **best["params"], **_FIXED_PARAMS)
    final_model.fit(splits["X_train_full"], splits["y_train_full"])

    test_metrics = evaluate_on(final_model, splits["X_test"], splits["y_test"])
    print(f"Test: AUC={test_metrics['auc']:.4f}  LogLoss={test_metrics['logloss']:.4f}  "
          f"Dir%={test_metrics['dir_acc']:.2f}")

    joblib.dump({"model": final_model, "feature_cols": FEATURES}, MODEL_PKL)
    print(f"Saved retuned model to {MODEL_PKL}")

    lines = [
        "Gradient classifier hyperparameter tuning",
        "=" * 60,
        f"Random search: {N_CANDIDATES} candidates, select by validation AUC.",
        f"Features: {len(FEATURES)} (best_plus_asi)",
        "",
        "Untuned baseline (test): AUC=0.6379  LogLoss=0.6542  Dir%=61.04",
        "",
        f"Best params: {best['params']}",
        f"Trees (final): {n_trees}",
        "",
        "Retuned model — TEST block (last 20%, unseen):",
        f"  AUC:      {test_metrics['auc']:.4f}",
        f"  LogLoss:  {test_metrics['logloss']:.4f}",
        f"  Dir%:     {test_metrics['dir_acc']:.2f}",
        "",
        "Top 5 candidates by validation AUC:",
    ]
    for row in ranked[:5]:
        lines.append(f"  AUC={row['val_auc']:.4f} trees={row['best_iteration']:4d}  {row['params']}")
    RESULTS_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {RESULTS_TXT}")


if __name__ == "__main__":
    main()
