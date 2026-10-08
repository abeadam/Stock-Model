"""Is the volatility surprise entangled with the direction model's own confidence?

The direction-times-volatility gate failed for a specific reason: the model knows direction
only when little is about to happen, so its confidence is a volatility forecast in disguise.
Selecting for large moves bought a cheaper break-even and sold an equal amount of skill,
roughly one for one. Any new volatility feature that is simply another read on the same
quantity will fail the same way, and there is no point building the ablation to find out.

So this measures the correlation between each candidate column and the out-of-sample
confidence |prob - 0.5| of the direction model on the dev test blocks.

Reference columns are included deliberately. A correlation means nothing without knowing
what an entangled feature looks like on this data, so `rolling_vol_20` and
`flow_range_pct_60` -- both already in the model, both plainly volatility reads -- are
measured alongside. The question is not whether the surprise correlates at all, but whether
it correlates like those do.

Spearman is the one to read: a tree needs only monotone information.

usage: volatility_surprise_pretest.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from cross_asset_features import load_symbol_closes
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model
from volatility_surprise import (SURPRISE_SERIES, VOLATILITY_SURPRISE_FEATURES,
                                 VOLATILITY_SURPRISE_LEGS, add_volatility_surprise)

HORIZON_BARS = 18
SEED = 11
REFERENCE_COLUMNS = ("rolling_vol_20", "flow_range_pct_60")
CORRELATION_SAMPLE = 200_000
ENTANGLED_THRESHOLD = 0.20


def correlations(values: np.ndarray, confidence: np.ndarray) -> tuple[float, float, int]:
    usable = np.isfinite(values) & np.isfinite(confidence)
    if usable.sum() < 2:
        return float("nan"), float("nan"), int(usable.sum())
    sample = np.flatnonzero(usable)
    if len(sample) > CORRELATION_SAMPLE:                # Spearman ranks the whole vector
        sample = np.random.default_rng(SEED).choice(sample, CORRELATION_SAMPLE, replace=False)
    return (float(pearsonr(values[sample], confidence[sample])[0]),
            float(spearmanr(values[sample], confidence[sample])[0]), len(sample))


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = add_volatility_surprise(data, {series: load_symbol_closes(series)
                                          for series in SURPRISE_SERIES})
    target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    X = data[base].to_numpy(np.float32)
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; {len(base)} base columns; "
          f"{len(splits)} dev folds ({time.time() - started:.0f}s to build)\n", flush=True)

    test_rows, confidence = [], []
    for fold_number, split in enumerate(splits, start=1):
        model = train_direction_model(X, target, split, {**LGBM_PARAMS, "random_state": SEED})
        probability = model.predict_proba(X[split.test])[:, 1]
        test_rows.append(split.test)
        confidence.append(np.abs(probability - 0.5))
        print(f"   fold {fold_number}: {len(split.test):,} test rows, mean confidence "
              f"{np.mean(confidence[-1]):.4f}   ({time.time() - started:.0f}s)", flush=True)
    rows = np.concatenate(test_rows)
    confidence = np.concatenate(confidence)

    print(f"\n{'column':<34}{'pearson':>10}{'spearman':>11}{'rows':>12}")
    for name in REFERENCE_COLUMNS:
        pearson, spearman, count = correlations(data[name].to_numpy(float)[rows], confidence)
        print(f"{'(reference) ' + name:<34}{pearson:>10.4f}{spearman:>11.4f}{count:>12,}")
    strongest = 0.0
    for name in VOLATILITY_SURPRISE_LEGS + VOLATILITY_SURPRISE_FEATURES:
        pearson, spearman, count = correlations(data[name].to_numpy(float)[rows], confidence)
        marker = "  <-- surprise" if name in VOLATILITY_SURPRISE_FEATURES else ""
        print(f"{name:<34}{pearson:>10.4f}{spearman:>11.4f}{count:>12,}{marker}")
        if name in VOLATILITY_SURPRISE_FEATURES and np.isfinite(spearman):
            strongest = max(strongest, abs(spearman))
    verdict = ("entangled with confidence like the existing volatility reads -- the gate has "
               "no room and the feature test is not worth running"
               if strongest >= ENTANGLED_THRESHOLD else
               "not entangled with confidence -- the feature test is worth running")
    print(f"\nStrongest |spearman| among the surprise columns: {strongest:.4f}. {verdict}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
