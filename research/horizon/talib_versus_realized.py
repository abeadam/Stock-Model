"""Did the TA-Lib sweep find anything the two realized-volatility columns did not?

The sweep's range shortlist gained +0.0385 R2 on the volatility head. That is larger than
anything else measured here -- and every column in it is a long smoothing of a volatility
column the model already has: `low_close_ratio_ema_60`, `high_close_ratio_rma_30`,
`ms_wick_up_ticks_ema_60`. Yesterday's realized volatility at 60 and 180 bars gained
+0.0248 by exactly that mechanism, with two columns instead of a hundred.

So the question is not whether the shortlist helps. It is whether it helps for a reason
already known, which decides whether a hundred indicator columns are worth carrying.

  plus realized     the two columns from yesterday
  plus talib        the hundred range picks
  plus both         if this matches the better of the two alone, they are the same finding

The same trap as the variance-premium work, where `plus legs` looked like a VIX result and
was really two realized windows: a composite has to beat its own plain ingredients.

usage: talib_versus_realized.py [HORIZON_BARS]   (default 18)
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
from talib_ablation import build_selected
from variance_risk_premium import REALIZED_WINDOWS, add_variance_premium_features

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RANKING_PATH = "talib_screen_ranking.json"
RECIPES_PATH = "talib_screen_recipes.json"
RESULTS_PATH_TEMPLATE = "talib_versus_realized_h{horizon}.json"


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    with open(RECIPES_PATH) as handle:
        recipes = json.load(handle)
    with open(RANKING_PATH) as handle:
        range_picks = json.load(handle)["range"]

    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = add_variance_premium_features(data, load_symbol_closes("VIX"))
    data = build_selected(data, {name: recipes[name] for name in range_picks if name in recipes})
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    realized = [f"vrp_realized_per_bar_{window}" for window in REALIZED_WINDOWS]
    talib_columns = [name for name in range_picks if name in data.columns]
    arms = {"base": base,
            "plus realized": base + realized,
            "plus talib": base + talib_columns,
            "plus both": base + realized + talib_columns}
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; {len(realized)} realized columns vs "
          f"{len(talib_columns)} TA-Lib range picks; {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

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

    print(f"\n{'arm':<18}{'columns':>9}{'volatility R2':>15}{'vs base':>11}{'t':>7}")
    print(f"{'base':<18}{len(base):>9}{np.mean(r2['base']):>15.4f}")
    for name, columns in arms.items():
        if name == "base":
            continue
        gain, t = paired_t(np.array(r2[name]) - np.array(r2["base"]))
        print(f"{name:<18}{len(columns):>9}{np.mean(r2[name]):>15.4f}{gain:>+11.5f}{t:>7.1f}")
    extra, extra_t = paired_t(np.array(r2["plus both"]) - np.array(r2["plus realized"]))
    print(f"\nWhat 100 TA-Lib columns add on top of the two realized ones: "
          f"{extra:+.5f} R2, t {extra_t:.1f}.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
