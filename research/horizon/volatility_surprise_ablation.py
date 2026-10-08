"""Does an implied-volatility surprise help either head at the 18-bar horizon?

The pre-test cleared the way: the surprise columns correlate with the direction model's
own confidence at |spearman| 0.018, against 0.130 for `rolling_vol_20` and 0.114 for
`flow_range_pct_60`. The volatility reads already in the model are entangled with direction
confidence -- which is why the direction-times-volatility gate sold as much skill as it
bought -- and the surprise is not. Confidence tracks the volatility *level*; the surprise
is a change of a change.

Three arms, so the divergence is tested **given** its legs rather than instead of them.
Testing only the bundle is the trap the oil-and-dollar ablation fell into, where the
verdict reversed once columns were judged against the others rather than removed wholesale:

  base            the 163 columns
  plus legs       + the realized change and both implied changes, at both windows
  plus surprise   + the four divergence columns on top of those legs

Gain share is reported because a large share with no gain is dilution rather than neglect,
and because the surprise columns are currently dominated by their realized leg -- VXX and
UVXY produce near-identical values, and each is close to minus the realized change.

usage: volatility_surprise_ablation.py [HORIZON_BARS]   (default 18)
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
from volatility_surprise import (SURPRISE_SERIES, VOLATILITY_SURPRISE_FEATURES,
                                 VOLATILITY_SURPRISE_LEGS, add_volatility_surprise)

HORIZON_BARS = 18
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
RESULTS_PATH_TEMPLATE = "volatility_surprise_ablation_h{horizon}.json"


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    data = add_volatility_surprise(data, {series: load_symbol_closes(series)
                                          for series in SURPRISE_SERIES})
    direction_target = average_target(data["mid"].to_numpy(float), epochs, horizon)
    decided = np.isfinite(direction_target) & (direction_target != 0)
    range_target = forward_target({"seconds": epochs, "close": data["Close"].to_numpy(float),
                                   "high": data["High"].to_numpy(float),
                                   "low": data["Low"].to_numpy(float)}, horizon, "range")
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    arms = {"base": (base, 0),
            "plus legs": (base + VOLATILITY_SURPRISE_LEGS, len(VOLATILITY_SURPRISE_LEGS)),
            "plus surprise": (base + VOLATILITY_SURPRISE_LEGS + VOLATILITY_SURPRISE_FEATURES,
                              len(VOLATILITY_SURPRISE_FEATURES))}
    splits = purged_fold_splits(data["time"], horizon)
    coverage = data[VOLATILITY_SURPRISE_FEATURES[0]].notna().mean()
    print(f"horizon {horizon}, purge {horizon}; surprise columns cover {coverage:.1%} of "
          f"regular-hours bars; {len(SEEDS)} seeds x {len(splits)} folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    auc, r2, gain = {}, {}, {}
    for name, (columns, added) in arms.items():
        X = data[columns].to_numpy(np.float32)
        for seed in SEEDS:
            aucs, r2s, shares = head_scores(X, direction_target, decided, range_target, splits,
                                            {**LGBM_PARAMS, "random_state": seed}, added=added)
            auc.setdefault(name, []).extend(aucs)
            r2.setdefault(name, []).extend(r2s)
            gain.setdefault(name, []).extend(shares)
        per_fold = "  ".join(f"fold {fold + 1} R2 {np.mean(r2[name][fold::len(splits)]):.4f}"
                             for fold in range(len(splits)))
        print(f"   {name:<15}({len(columns)} columns): direction {np.mean(auc[name]):.4f}  "
              f"volatility R2 {np.mean(r2[name]):.4f}   {per_fold}   "
              f"({time.time() - started:.0f}s)", flush=True)
        del X
        with open(RESULTS_PATH_TEMPLATE.format(horizon=horizon), "w") as handle:
            json.dump({"auc": auc, "r2": r2, "gain_share": gain}, handle, indent=1)

    print(f"\n{'arm':<16}{'columns':>9}{'direction':>12}{'t':>7}{'volatility':>13}{'t':>7}"
          f"{'gain share':>12}")
    print(f"{'base':<16}{len(base):>9}{np.mean(auc['base']):>12.4f}{'':>7}"
          f"{np.mean(r2['base']):>13.4f}")
    for name, (columns, _) in arms.items():
        if name == "base":
            continue
        auc_gain, auc_t = paired_t(np.array(auc[name]) - np.array(auc["base"]))
        r2_gain, r2_t = paired_t(np.array(r2[name]) - np.array(r2["base"]))
        print(f"{name:<16}{len(columns):>9}{auc_gain:>+12.5f}{auc_t:>7.1f}"
              f"{r2_gain:>+13.5f}{r2_t:>7.1f}{np.mean(gain[name]):>11.1%}")
    divergence, divergence_t = paired_t(np.array(r2["plus surprise"]) - np.array(r2["plus legs"]))
    print(f"\nWhat the divergence adds given its own legs: {divergence:+.5f} R2, t {divergence_t:.1f}.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
