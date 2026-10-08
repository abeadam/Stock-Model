"""Do this week's two surviving results replicate across many folds, or only two?

Everything measured this week rests on two dev folds, and those two folds disagreed on
nearly every effect. This runs the same 18-bar question over as much history as the data
allows, month by month, and reports **per-fold numbers and how many folds agree in sign**
rather than a pooled mean that can hide a split.

Arms, both heads, paired over 8 seeds:

  base               FEATURES + flow + the cross-asset columns that reach back
  + realized vol     60 and 180 bars, from Close, inside each session
  + smoothings       EMA/RMA at 30 and 60 of high_close_ratio and low_close_ratio
  + both
  + profit weight    max(|move| * 50 - cost, 0) on whichever of the above wins

The target is Close-based, which carries bid-ask bounce -- the thing that took the
gradient model from 0.546 to roughly 0.52 under a one-bar fill delay. Everything is
therefore run twice, at delay 0 and delay 1. **If the delayed version collapses, the
long-history result is bounce.**

usage: long_history_test.py [--spy] [HORIZON_BARS]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from cost_aware_targets import move_in_dollars, profit_weight
from deep_ladder_both_heads import fit_volatility_model, paired_t
from long_history_support import (REALIZED_FEATURES, SMOOTHED_FEATURES, base_columns, build,
                                  forward_average_target, forward_range_target, monthly_folds)
from short_horizon_walk_forward import (LGBM_PARAMS, POINT_VALUE, ROUND_TRIP_COST,
                                        train_direction_model)

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
DELAYS = (0, 1)
# /private/tmp has been wiped twice this week; a five-hour run must not write only there.
RESULTS_DIR = "/Users/abeadam/dev/model/Stock-Model/scratchpad_backup"
RESULTS_PATH_TEMPLATE = RESULTS_DIR + "/long_history_test_{tag}.json"


def score_fold(X, direction, decided, ranges, split, params, added):
    """One AUC and one R2 for a single fold, plus the added columns' share of tree gain."""
    classifier = train_direction_model(X, direction, split, params)
    test = split.test[decided[split.test]]
    auc = roc_auc_score((direction[test] > 0).astype(int),
                        classifier.predict_proba(X[test])[:, 1])
    regressor = fit_volatility_model(X, ranges, split, params)
    known = split.test[np.isfinite(ranges[split.test])]
    r2 = r2_score(ranges[known], regressor.predict(X[known]))
    share = 0.0
    if added:
        gain = classifier.booster_.feature_importance(importance_type="gain")
        share = float(gain[-added:].sum() / max(gain.sum(), 1e-12))
    return auc, r2, share


def agreement(differences: np.ndarray) -> str:
    positive = int((differences > 0).sum())
    return f"{positive}/{len(differences)} folds up"


def main(include_spy: bool, horizon: int) -> None:
    started = time.time()
    tag = "with_spy" if include_spy else "no_spy"
    data = build(include_spy)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    close = data["Close"].to_numpy(float)
    session_days = data["session_day"].to_numpy()
    folds = monthly_folds(session_days, horizon)
    base = base_columns(include_spy)
    arms = {"base": (base, 0),
            "plus realized": (base + REALIZED_FEATURES, len(REALIZED_FEATURES)),
            "plus smoothings": (base + SMOOTHED_FEATURES, len(SMOOTHED_FEATURES)),
            "plus both": (base + REALIZED_FEATURES + SMOOTHED_FEATURES,
                          len(REALIZED_FEATURES) + len(SMOOTHED_FEATURES))}
    ranges = forward_range_target(data["High"].to_numpy(float), data["Low"].to_numpy(float),
                                  close, seconds, horizon)
    print(f"{tag}: {len(data):,} regular-hours bars over "
          f"{len(set(session_days))} sessions ({min(session_days)} .. {max(session_days)}); "
          f"{len(base)} base columns; {len(folds)} monthly folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    saved = {}
    for delay in DELAYS:
        direction = forward_average_target(close, seconds, horizon, delay)
        decided = np.isfinite(direction) & (direction != 0)
        print(f"=== delay {delay} bar{'s' if delay != 1 else ''}: "
              f"{int(decided.sum()):,} decided rows ===", flush=True)
        auc, r2, gain = {}, {}, {}
        for name, (columns, added) in arms.items():
            X = data[columns].to_numpy(np.float32)
            for fold in folds:
                for seed in SEEDS:
                    values = score_fold(X, direction, decided, ranges, fold.split,
                                        {**LGBM_PARAMS, "random_state": seed}, added)
                    auc.setdefault(name, {}).setdefault(fold.month, []).append(values[0])
                    r2.setdefault(name, {}).setdefault(fold.month, []).append(values[1])
                    gain.setdefault(name, []).append(values[2])
            print(f"   {name:<16}({len(columns)} cols)  direction "
                  f"{np.mean([np.mean(v) for v in auc[name].values()]):.4f}  volatility R2 "
                  f"{np.mean([np.mean(v) for v in r2[name].values()]):.4f}   "
                  f"({time.time() - started:.0f}s)", flush=True)
            del X
        report(auc, r2, gain, arms, folds, delay)
        saved[f"delay_{delay}"] = {"auc": {k: {m: v for m, v in d.items()} for k, d in auc.items()},
                                   "r2": {k: {m: v for m, v in d.items()} for k, d in r2.items()},
                                   "gain": gain}
        with open(RESULTS_PATH_TEMPLATE.format(tag=tag), "w") as handle:
            json.dump(saved, handle, indent=1)


def report(auc, r2, gain, arms, folds, delay) -> None:
    months = [fold.month for fold in folds]
    print(f"\n  per-fold direction AUC, delay {delay}")
    print("  " + f"{'month':<10}" + "".join(f"{name.replace('plus ', '+'):>17}" for name in arms))
    for month in months:
        print("  " + f"{month:<10}"
              + "".join(f"{np.mean(auc[name][month]):>17.4f}" for name in arms))
    print(f"\n  {'arm':<16}{'direction':>11}{'t':>7}{'agree':>14}"
          f"{'volatility':>12}{'t':>7}{'agree':>14}{'gain':>8}")
    base_auc = np.array([np.mean(auc["base"][month]) for month in months])
    base_r2 = np.array([np.mean(r2["base"][month]) for month in months])
    print("  " + f"{'base':<16}{base_auc.mean():>11.4f}{'':>7}{'':>14}{base_r2.mean():>12.4f}")
    for name in arms:
        if name == "base":
            continue
        arm_auc = np.array([np.mean(auc[name][month]) for month in months])
        arm_r2 = np.array([np.mean(r2[name][month]) for month in months])
        auc_gain, auc_t = paired_t(arm_auc - base_auc)
        r2_gain, r2_t = paired_t(arm_r2 - base_r2)
        print("  " + f"{name:<16}{auc_gain:>+11.5f}{auc_t:>7.1f}{agreement(arm_auc - base_auc):>14}"
              f"{r2_gain:>+12.5f}{r2_t:>7.1f}{agreement(arm_r2 - base_r2):>14}"
              f"{np.mean(gain[name]):>7.1%}")
    print(flush=True)


if __name__ == "__main__":
    arguments = [value for value in sys.argv[1:] if value != "--spy"]
    main("--spy" in sys.argv, int(arguments[0]) if arguments else HORIZON_BARS)
