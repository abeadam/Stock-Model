"""Was it the implied leg, or just longer realized-volatility windows?

Adding three columns -- one implied level and two realized levels -- lifted the volatility
head from 0.4674 to 0.4872 at the 18-bar horizon. That is an order of magnitude larger than
anything else measured in this line of work, and it is ambiguous, because the base set
carries realized volatility only at `rolling_vol_5` and `rolling_vol_20`. The new columns
brought **60- and 180-bar** realized volatility with them. Volatility clusters, so a longer
realized window is an excellent and entirely ordinary predictor of an 18-bar forward range.

Every arm in the first ablation that held the implied leg also held those long realized
legs, so nothing there could separate the two. This does:

  plus realized   the 60- and 180-bar realized volatilities alone, no VIX anywhere
  plus implied    the VIX-derived expected move alone, no new realized windows
  plus legs       both, reproducing the ambiguous arm

If "plus realized" reaches the same place as "plus legs", the gain was a missing backward
looking feature and VIX contributed nothing -- which also settles whether a real-time VIX
subscription is worth buying.

usage: variance_premium_disambiguation.py [HORIZON_BARS]   (default 18)
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
from variance_risk_premium import REALIZED_WINDOWS, add_variance_premium_features

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
IMPLIED_COLUMN = "vrp_implied_per_bar"
RESULTS_PATH_TEMPLATE = "variance_premium_disambiguation_h{horizon}.json"


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = add_variance_premium_features(data, load_symbol_closes("VIX"))
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    realized = [f"vrp_realized_per_bar_{window}" for window in REALIZED_WINDOWS]
    arms = {"base": base, "plus realized": base + realized,
            "plus implied": base + [IMPLIED_COLUMN], "plus legs": base + [IMPLIED_COLUMN] + realized}
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; realized windows {REALIZED_WINDOWS} bars; "
          f"{len(SEEDS)} seeds x {len(splits)} folds ({time.time() - started:.0f}s to build)\n",
          flush=True)

    auc, r2 = {}, {}
    for name, columns in arms.items():
        X = data[columns].to_numpy(np.float32)
        for seed in SEEDS:
            aucs, r2s, _ = head_scores(X, direction_target, decided, range_target, splits,
                                       {**LGBM_PARAMS, "random_state": seed}, added=0)
            auc.setdefault(name, []).extend(aucs)
            r2.setdefault(name, []).extend(r2s)
        print(f"   {name:<16}({len(columns)} columns): direction {np.mean(auc[name]):.4f}  "
              f"volatility R2 {np.mean(r2[name]):.4f}   ({time.time() - started:.0f}s)", flush=True)
        del X
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": auc, "r2": r2}, handle, indent=1)

    print(f"\n{'arm':<18}{'columns':>9}{'direction':>12}{'t':>7}{'volatility':>13}{'t':>7}")
    print(f"{'base':<18}{len(base):>9}{np.mean(auc['base']):>12.4f}{'':>7}"
          f"{np.mean(r2['base']):>13.4f}")
    for name, columns in arms.items():
        if name == "base":
            continue
        auc_gain, auc_t = paired_t(np.array(auc[name]) - np.array(auc["base"]))
        r2_gain, r2_t = paired_t(np.array(r2[name]) - np.array(r2["base"]))
        print(f"{name:<18}{len(columns):>9}{auc_gain:>+12.5f}{auc_t:>7.1f}"
              f"{r2_gain:>+13.5f}{r2_t:>7.1f}")
    implied_gain = np.mean(r2["plus legs"]) - np.mean(r2["plus realized"])
    print(f"\nWhat the implied leg adds on top of the realized windows: {implied_gain:+.5f} R2.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
