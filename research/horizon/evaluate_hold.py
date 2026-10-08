"""Hold-while-the-signal-agrees policy on cached fold predictions.

Each fold picks its rule on its validation tail (best net P&L among rules with
enough trades) and trades the test fold with that rule unchanged; a fold whose
validation has no rule with enough trades does not trade. Entry thresholds are
quantiles of the validation |score|, so one grid fits every score's scale.

usage: evaluate_hold.py CACHE [SCORE] [FILL_DELAY_BARS]
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from rule_selection import choose_rule as choose_rule_for_prices, run_rule

MIN_VALIDATION_TRADES = 500
SCORES = {
    "prob": lambda part: part["prob"] - 0.5,
    "edge_points": lambda part: (2 * part["prob"] - 1) * part["abs_move_pred"],
    "move_pred": lambda part: part["move_pred"],
}


def run_policy(score: np.ndarray, part: dict, rule) -> dict:
    return run_rule(score, part["close"], part["close"], part["seconds"], rule)


def choose_rule(score: np.ndarray, validation: dict, fill_delay_bars: int, target_bars: int):
    return choose_rule_for_prices(score, validation["close"], validation["close"], validation["seconds"],
                                  target_bars, MIN_VALIDATION_TRADES, fill_delay_bars)


def evaluate(cache_path: str, score_name: str, fill_delay_bars: int) -> dict:
    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    score_of = SCORES[score_name]
    per_fold = {}
    for fold, parts in cache["folds"].items():
        rule = choose_rule(score_of(parts["validation"]), parts["validation"], fill_delay_bars,
                           cache["target_bars"])
        if rule is None:
            per_fold[fold] = {"trades": 0, "net": 0.0, "gross": 0.0, "hit_rate": np.nan, "mean_bars_held": np.nan}
            print(f"  fold {fold}: no rule reached {MIN_VALIDATION_TRADES} validation trades; no trading")
            continue
        result = per_fold[fold] = run_policy(score_of(parts["test"]), parts["test"], rule)
        print(f"  fold {fold}: entry {rule.entry_threshold:.4f} exit {rule.exit_threshold:+.4f} "
              f"min hold {rule.min_hold_bars:>2}  trades {result['trades']:>6,}  win {result['hit_rate']:6.1%}  "
              f"held {result['mean_bars_held']:6.1f} bars  gross ${result['gross']:>9,.0f}  net ${result['net']:>9,.0f}",
              flush=True)
    trades = sum(r["trades"] for r in per_fold.values())
    net = sum(r["net"] for r in per_fold.values())
    wins = sum(r["hit_rate"] * r["trades"] for r in per_fold.values() if r["trades"])
    print(f"{cache_path} score {score_name} fill delay {fill_delay_bars}: trades {trades:,}  "
          f"win {wins / max(trades, 1):.1%}  net ${net:,.0f}  per trade ${net / max(trades, 1):.2f}  "
          f"folds+ {sum(r['net'] > 0 for r in per_fold.values())}/{len(per_fold)}", flush=True)
    return per_fold


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "prob",
             int(sys.argv[3]) if len(sys.argv) > 3 else 1)
