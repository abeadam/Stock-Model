"""Does splitting the columns per head beat giving both heads the same matrix?

Two findings say the heads should not share a feature set, and until now they had to:

  * `realized_vol_60/180` lift the volatility head +0.0191 R2 over 13 folds (12/13) and
    do nothing for direction (t -0.5, 7/13).
  * the six oil and dollar divergence columns were justified by a direction ablation
    run at a 12-bar purge on a 5-bar horizon; at purge = horizon the verdict flips to
    t -1.40 and -1.38, both negative, neither significant.

`split_by_head` applies both: realized volatility to the volatility head alone, the oil
and dollar columns to neither. This measures that applied configuration against the
status quo on the same folds and seeds, so the pair is paired and the fold purge cancels.

Three arms separate the two changes, because a single applied arm cannot say which half
did the work:

  shared          every column to both heads, as the harness has always run
  dropped only    the six columns removed from both heads, nothing added
  split           dropped, plus realized volatility to the volatility head only

usage: head_split_test.py [--with-spy] [--horizon N]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from deep_ladder_both_heads import fit_volatility_model, paired_t
from head_columns import DROPPED_FEATURES, HeadColumns, split_by_head
from long_history_support import (base_columns, build, forward_average_target,
                                  forward_range_target, monthly_folds)
from realized_volatility_features import (REALIZED_VOLATILITY_FEATURES,
                                          REALIZED_VOLATILITY_WINDOWS, realized_volatility)
from short_horizon_walk_forward import LGBM_PARAMS, train_direction_model

SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
DELAY_BARS = 1                      # bounce-free: a Close target at delay 0 is bid-ask bounce
RESULTS_PATH = "/Users/abeadam/dev/model/Stock-Model/scratchpad_backup/head_split_test_{tag}.json"


def gain_share(model, added: int) -> float:
    """The share of total tree gain taken by the last `added` columns."""
    if not added:
        return 0.0
    gain = model.booster_.feature_importance(importance_type="gain")
    return float(gain[-added:].sum() / max(gain.sum(), 1e-12))


def score_fold(matrix, columns: HeadColumns, direction, decided, ranges, split, params):
    """AUC for the direction head and R2 for the volatility head, on their own columns.

    `matrix` holds the volatility head's columns with the direction head's first, so the
    direction head is a slice rather than a second copy of a million-row array.
    """
    direction_matrix = matrix[:, :len(columns.direction)]
    classifier = train_direction_model(direction_matrix, direction, split, params)
    test = split.test[decided[split.test]]
    auc = roc_auc_score((direction[test] > 0).astype(int),
                        classifier.predict_proba(direction_matrix[test])[:, 1])

    regressor = fit_volatility_model(matrix, ranges, split, params)
    known = split.test[np.isfinite(ranges[split.test])]
    r2 = r2_score(ranges[known], regressor.predict(matrix[known]))
    return auc, r2, gain_share(regressor, columns.volatility_added)


def arms_for(base: list[str]) -> dict[str, HeadColumns]:
    """Shared, then each change in turn, so a difference can be attributed."""
    dropped = [name for name in base if name not in DROPPED_FEATURES]
    return {
        "shared": HeadColumns(direction=list(base), volatility=list(base)),
        "dropped only": HeadColumns(direction=dropped, volatility=list(dropped)),
        "split": split_by_head(base),
    }


def main(include_spy: bool, horizon: int) -> None:
    started = time.time()
    tag = "with_spy" if include_spy else "no_spy"
    data = build(include_spy)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    close = data["Close"].to_numpy(float)
    session_days = data["session_day"].to_numpy()
    for name, window in zip(REALIZED_VOLATILITY_FEATURES, REALIZED_VOLATILITY_WINDOWS):
        data[name] = realized_volatility(close, session_days, window)

    folds = monthly_folds(session_days, horizon)
    arms = arms_for(base_columns(include_spy))
    direction = forward_average_target(close, seconds, horizon, DELAY_BARS)
    decided = np.isfinite(direction) & (direction != 0)
    ranges = forward_range_target(data["High"].to_numpy(float), data["Low"].to_numpy(float),
                                  close, seconds, horizon)
    print(f"{tag}: {len(data):,} bars, {len(set(session_days))} sessions, {len(folds)} folds, "
          f"delay {DELAY_BARS}, horizon {horizon} ({time.time() - started:.0f}s to build)\n",
          flush=True)

    auc, r2, gain = {}, {}, {}
    for name, columns in arms.items():
        matrix = data[columns.volatility].to_numpy(np.float32)
        for fold in folds:
            for seed in SEEDS:
                values = score_fold(matrix, columns, direction, decided, ranges,
                                    fold.split, {**LGBM_PARAMS, "random_state": seed})
                auc.setdefault(name, {}).setdefault(fold.month, []).append(values[0])
                r2.setdefault(name, {}).setdefault(fold.month, []).append(values[1])
                gain.setdefault(name, []).append(values[2])
        print(f"   {name:<14}(dir {len(columns.direction)}, vol {len(columns.volatility)})  "
              f"direction {np.mean([np.mean(v) for v in auc[name].values()]):.4f}  "
              f"volatility R2 {np.mean([np.mean(v) for v in r2[name].values()]):.4f}   "
              f"({time.time() - started:.0f}s)", flush=True)
        del matrix
    report(auc, r2, gain, arms, folds)
    with open(RESULTS_PATH.format(tag=tag), "w") as handle:
        json.dump({"auc": {k: {m: list(v) for m, v in d.items()} for k, d in auc.items()},
                   "r2": {k: {m: list(v) for m, v in d.items()} for k, d in r2.items()},
                   "gain": gain, "horizon": horizon, "delay": DELAY_BARS}, handle, indent=1)


def report(auc, r2, gain, arms, folds) -> None:
    months = [fold.month for fold in folds]
    print(f"\n  {'arm':<14}{'direction':>11}{'t':>7}{'agree':>10}"
          f"{'volatility':>12}{'t':>7}{'agree':>10}{'gain':>8}")
    for name in arms:
        if name == "shared":
            print(f"  {name:<14}{np.mean([np.mean(auc[name][m]) for m in months]):>11.4f}"
                  f"{'':>7}{'':>10}{np.mean([np.mean(r2[name][m]) for m in months]):>12.4f}")
            continue
        direction_diff = np.array([np.mean(auc[name][m]) - np.mean(auc["shared"][m]) for m in months])
        volatility_diff = np.array([np.mean(r2[name][m]) - np.mean(r2["shared"][m]) for m in months])
        print(f"  {name:<14}{direction_diff.mean():>+11.5f}{paired_t(direction_diff):>7.1f}"
              f"{f'{int((direction_diff > 0).sum())}/{len(months)}':>10}"
              f"{volatility_diff.mean():>+12.5f}{paired_t(volatility_diff):>7.1f}"
              f"{f'{int((volatility_diff > 0).sum())}/{len(months)}':>10}"
              f"{np.mean(gain[name]):>8.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-spy", action="store_true")
    parser.add_argument("--horizon", type=int, default=18)
    arguments = parser.parse_args()
    main(arguments.with_spy, arguments.horizon)
