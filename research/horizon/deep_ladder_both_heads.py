"""Does reaching further back help either half of the model?

The model has two heads. The direction head is a classifier on whether the average mid over
the next `horizon` bars is above the current one; the volatility head is a regressor on the
high-to-low range over those same bars. They fail differently and there is no reason a
feature that helps one should help the other, so both are measured here.

Every family is extended at once by `deep_ladder_features`: the returns of all twenty
symbols already in the model, the mid's own change and reversion, and RSI, Bollinger and
EMA-distance on ES with their windows scaled to each rung rather than fixed at 12 and 60.
Depth 0 is the 163 columns as they stand; each further rung adds 25 columns reaching two
minutes further back.

Paired throughout -- same seeds, same folds, differences taken pair by pair -- and purged
by the horizon, which is the only defensible gap ([[purged_folds]]).

usage: deep_ladder_both_heads.py [HORIZON_BARS]   (default 5)
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from cross_asset_features import load_symbol_closes
from deep_ladder_features import (LADDER_SYMBOLS, add_deep_ladder_features, deep_feature_names,
                                  deep_ladder)
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model

HORIZON_BARS = 5
LADDER_RUNGS = (1, 2, 4, 6)
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
EARLY_STOPPING_ROUNDS = 200
RESULTS_PATH_TEMPLATE = "deep_ladder_both_heads_h{horizon}.json"


def fit_volatility_model(X, target, split, params):
    """Regressor for the forward high-to-low range, early-stopped on the validation block."""
    known = np.isfinite(target)
    train, validation = split.train[known[split.train]], split.validation[known[split.validation]]
    model = lgb.LGBMRegressor(**{**params, "objective": "regression"})
    model.fit(X[train], target[train], eval_set=[(X[validation], target[validation])],
              eval_metric="l2", callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                           lgb.log_evaluation(0)])
    return model


def head_scores(X, direction_target, decided, range_target, splits, params, added: int):
    """One AUC and one R2 per fold, plus the share of direction gain the added columns took."""
    aucs, r2s, gain_shares = [], [], []
    for split in splits:
        classifier = train_direction_model(X, direction_target, split, params)
        test = split.test[decided[split.test]]
        aucs.append(roc_auc_score((direction_target[test] > 0).astype(int),
                                  classifier.predict_proba(X[test])[:, 1]))
        regressor = fit_volatility_model(X, range_target, split, params)
        known = split.test[np.isfinite(range_target[split.test])]
        r2s.append(r2_score(range_target[known], regressor.predict(X[known])))
        if added:
            gain = classifier.booster_.feature_importance(importance_type="gain")
            gain_shares.append(float(gain[-added:].sum() / max(gain.sum(), 1e-12)))
    return aucs, r2s, gain_shares


def paired_t(differences: np.ndarray) -> tuple[float, float]:
    error = differences.std(ddof=1) / np.sqrt(len(differences))
    return differences.mean(), differences.mean() / error if error > 0 else 0.0


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    closes = {symbol: load_symbol_closes(symbol) for symbol in LADDER_SYMBOLS}
    data = add_deep_ladder_features(data, closes, deep_ladder(max(LADDER_RUNGS)))
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; {len(base)} base columns over "
          f"{len(LADDER_SYMBOLS)} symbols; rungs {LADDER_RUNGS} "
          f"reaching back to {max(deep_ladder(max(LADDER_RUNGS)))} bars; {len(SEEDS)} seeds x "
          f"{len(splits)} folds ({time.time() - started:.0f}s to build)\n", flush=True)

    auc, r2, gain = {}, {}, {}
    for rungs in (0, *LADDER_RUNGS):
        columns = base + deep_feature_names(deep_ladder(rungs))
        X = data[columns].to_numpy(np.float32)
        for seed in SEEDS:
            aucs, r2s, shares = head_scores(X, direction_target, decided, range_target, splits,
                                            {**LGBM_PARAMS, "random_state": seed},
                                            added=len(columns) - len(base))
            auc.setdefault(rungs, []).extend(aucs)
            r2.setdefault(rungs, []).extend(r2s)
            gain.setdefault(rungs, []).extend(shares)
        print(f"   {rungs} rungs ({len(columns)} columns): direction {np.mean(auc[rungs]):.4f}  "
              f"volatility R2 {np.mean(r2[rungs]):.4f}   ({time.time() - started:.0f}s)", flush=True)
        del X
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": {str(k): v for k, v in auc.items()},
                       "r2": {str(k): v for k, v in r2.items()},
                       "gain_share": {str(k): v for k, v in gain.items()}}, handle, indent=1)

    print(f"\n{'reaches back':<16}{'columns':>9}{'direction':>12}{'t':>7}"
          f"{'volatility':>13}{'t':>7}{'gain share':>12}")
    print(f"{'base':<16}{len(base):>9}{np.mean(auc[0]):>12.4f}{'':>7}{np.mean(r2[0]):>13.4f}")
    for rungs in LADDER_RUNGS:
        auc_gain, auc_t = paired_t(np.array(auc[rungs]) - np.array(auc[0]))
        r2_gain, r2_t = paired_t(np.array(r2[rungs]) - np.array(r2[0]))
        reach = f"{max(deep_ladder(rungs))} bars"
        print(f"{reach:<16}{len(base) + len(deep_feature_names(deep_ladder(rungs))):>9}"
              f"{auc_gain:>+12.5f}{auc_t:>7.1f}{r2_gain:>+13.5f}{r2_t:>7.1f}"
              f"{np.mean(gain[rungs]):>11.1%}")
    print("\nGain share is the direction head's; a large share with no gain is dilution.")


if __name__ == "__main__":
    # usage: deep_ladder_both_heads.py [HORIZON_BARS]
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
