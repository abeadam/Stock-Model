"""Do the indicators the screen shortlisted survive a held-out month?

The screen ranked 30,644 candidates by correlation on the earliest fold's training rows.
Picking the best of that many inflates them by multiple comparison however careful the
scoring, so the shortlist is a hypothesis and this is its test: paired over the same seeds
and folds, both heads, purged by the horizon.

Three arms, because the two heads were screened separately and a column chosen for one may
dilute the other:

  plus direction picks   the shortlist ranked against the direction label
  plus range picks       the shortlist ranked against the forward range
  plus all picks         both, which is what "add the indicators" would mean in practice

Gain share is reported alongside: with this many added columns, a large share and no gain
is the dilution already measured at 72.7% for deep ladders.

usage: talib_ablation.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from deep_ladder_both_heads import head_scores, paired_t
from forward_extremes import forward_target
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS
from talib_candidates import evaluate

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RECIPES_PATH = "talib_screen_recipes.json"
RANKING_PATH = "talib_screen_ranking.json"
RESULTS_PATH_TEMPLATE = "talib_ablation_h{horizon}.json"


def build_selected(data: pd.DataFrame, recipes: dict[str, dict]) -> pd.DataFrame:
    """Recompute the shortlisted indicators from their saved recipes."""
    columns = {}
    for name, recipe in recipes.items():
        series = data[recipe["column"]].to_numpy(float)
        outputs = evaluate(recipe["function"], series, recipe["params"])
        if len(outputs) > recipe["output"]:
            columns[name] = outputs[recipe["output"]]
    return data.assign(**columns)


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    with open(RECIPES_PATH) as handle:
        recipes = json.load(handle)
    try:
        with open(RANKING_PATH) as handle:
            ranking = json.load(handle)
    except FileNotFoundError:
        ranking = {"direction": list(recipes), "range": list(recipes)}

    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = build_selected(data, recipes)
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    present = [name for name in recipes if name in data.columns]
    direction_picks = [name for name in ranking["direction"] if name in data.columns]
    range_picks = [name for name in ranking["range"] if name in data.columns]
    arms = {"base": (base, 0),
            "plus direction picks": (base + direction_picks, len(direction_picks)),
            "plus range picks": (base + range_picks, len(range_picks)),
            "plus all picks": (base + present, len(present))}
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}; {len(present)} shortlisted indicators "
          f"({len(direction_picks)} direction, {len(range_picks)} range); "
          f"{len(SEEDS)} seeds x {len(splits)} folds ({time.time() - started:.0f}s to build)\n",
          flush=True)

    auc, r2, gain = {}, {}, {}
    for name, (columns, added) in arms.items():
        X = data[columns].to_numpy(np.float32)
        for seed in SEEDS:
            aucs, r2s, shares = head_scores(X, direction_target, decided, range_target, splits,
                                            {**LGBM_PARAMS, "random_state": seed}, added=added)
            auc.setdefault(name, []).extend(aucs)
            r2.setdefault(name, []).extend(r2s)
            gain.setdefault(name, []).extend(shares)
        print(f"   {name:<22}({len(columns)} columns): direction {np.mean(auc[name]):.4f}  "
              f"volatility R2 {np.mean(r2[name]):.4f}   ({time.time() - started:.0f}s)", flush=True)
        del X
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": auc, "r2": r2, "gain_share": gain}, handle, indent=1)

    print(f"\n{'arm':<24}{'columns':>9}{'direction':>12}{'t':>7}{'volatility':>13}{'t':>7}"
          f"{'gain share':>12}")
    print(f"{'base':<24}{len(base):>9}{np.mean(auc['base']):>12.4f}{'':>7}"
          f"{np.mean(r2['base']):>13.4f}")
    for name, (columns, _) in arms.items():
        if name == "base":
            continue
        auc_gain, auc_t = paired_t(np.array(auc[name]) - np.array(auc["base"]))
        r2_gain, r2_t = paired_t(np.array(r2[name]) - np.array(r2["base"]))
        print(f"{name:<24}{len(columns):>9}{auc_gain:>+12.5f}{auc_t:>7.1f}"
              f"{r2_gain:>+13.5f}{r2_t:>7.1f}{np.mean(gain[name]):>11.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
