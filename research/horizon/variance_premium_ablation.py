"""Does the variance risk premium help either head at the 18-bar horizon?

The VIX-family *returns* were tested and rejected earlier today: added to the 163 columns
they changed nothing, and the index added nothing over the two traded funds (t -0.70). The
premium is a different claim. It compares implied and realized volatility as **levels**,
which the funds structurally cannot express -- VXX and UVXY decay, so their level is not a
volatility -- and which no column in the model currently measures.

Four arms, chosen so a positive result cannot be claimed for the wrong reason:

  base                 the 163 columns as they stand
  plus legs            base + the implied and realized levels alone. Trees can combine two
                       columns themselves, so if this matches the arms below, the premium
                       construction is decoration and the levels were the whole story
  plus deseasonal      base + only the trailing-profile-adjusted premium columns, which are
                       the ones that cannot be encoding time of day
  plus all             base + every premium column, seasonal ones included

Both heads, paired on the same seeds and folds, purged by the horizon.

usage: variance_premium_ablation.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from cross_asset_features import load_symbol_closes
from deep_ladder_both_heads import head_scores, paired_t
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS
from variance_risk_premium import (REALIZED_WINDOWS, VARIANCE_PREMIUM_FEATURES,
                                   add_variance_premium_features)

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
VIX_SYMBOL = "VIX"
RESULTS_PATH_TEMPLATE = "variance_premium_ablation_h{horizon}.json"


def arms(base: list[str]) -> dict[str, list[str]]:
    legs = ["vrp_implied_per_bar"] + [f"vrp_realized_per_bar_{w}" for w in REALIZED_WINDOWS]
    deseasonal = [name for name in VARIANCE_PREMIUM_FEATURES if "deseasonal" in name]
    return {"base": base, "plus legs": base + legs, "plus deseasonal": base + deseasonal,
            "plus all": base + list(VARIANCE_PREMIUM_FEATURES)}


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    vix_closes = load_symbol_closes(VIX_SYMBOL)
    data = add_variance_premium_features(data, vix_closes)
    covered = data["vrp_implied_per_bar"].notna().mean()
    usable = data[f"vrp_ratio_deseasonal_{REALIZED_WINDOWS[0]}"].notna().mean()
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; VIX covers {covered:.1%} of regular-hours bars, "
          f"{usable:.1%} have a trailing profile; {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    auc, r2 = {}, {}
    for name, columns in arms(base).items():
        X = data[columns].to_numpy(np.float32)
        for seed in SEEDS:
            aucs, r2s, _ = head_scores(X, direction_target, decided, range_target, splits,
                                       {**LGBM_PARAMS, "random_state": seed}, added=0)
            auc.setdefault(name, []).extend(aucs)
            r2.setdefault(name, []).extend(r2s)
        print(f"   {name:<18}({len(columns)} columns): direction {np.mean(auc[name]):.4f}  "
              f"volatility R2 {np.mean(r2[name]):.4f}   ({time.time() - started:.0f}s)", flush=True)
        del X
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": auc, "r2": r2}, handle, indent=1)

    print(f"\n{'arm':<20}{'columns':>9}{'direction':>12}{'t':>7}{'volatility':>13}{'t':>7}")
    built = arms(base)
    print(f"{'base':<20}{len(base):>9}{np.mean(auc['base']):>12.4f}{'':>7}{np.mean(r2['base']):>13.4f}")
    for name in built:
        if name == "base":
            continue
        auc_gain, auc_t = paired_t(np.array(auc[name]) - np.array(auc["base"]))
        r2_gain, r2_t = paired_t(np.array(r2[name]) - np.array(r2["base"]))
        print(f"{name:<20}{len(built[name]):>9}{auc_gain:>+12.5f}{auc_t:>7.1f}"
              f"{r2_gain:>+13.5f}{r2_t:>7.1f}")
    print("\nIf 'plus legs' matches the others, the premium construction added nothing the "
          "trees could not build themselves.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
