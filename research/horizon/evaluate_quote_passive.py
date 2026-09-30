"""What the current model earns once the fills are real.

Every P&L number in this project assumes a fill at the mid with $5 charged per round trip.
That is not a market order -- crossing costs half a tick a side on top -- it is a resting
order that filled. The assumption is therefore a claim about passive execution that has
never been checked against the quotes, and it hides two costs it cannot see: the signals
whose price never comes back, and the wait between the signal and the fill.

This measures both. Each fold trains on its own history, picks a rule on its validation
block, and trades its test month with that rule unchanged. Three execution styles are
compared on identical signals:

  crossing   take every signal immediately, buying the ask and selling the bid. Misses
             nothing, pays the spread twice.
  passive    rest on the near touch, cross to get out. In a still book this nets to the
             mid-to-mid assumption, so the gap between it and the mid number is exactly
             the cost of missed and delayed fills.
  passive exit
             rest on the way out too, crossing only when it times out.

usage: evaluate_quote_passive.py
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from hold_policy import last_bar_before_gap
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from quote_passive_entry import (CROSS_EXIT, EXIT_MODES, QuotePassiveRule, Quotes, simulate,
                                 simulate_crossing)
from short_horizon_walk_forward import (FEATURES, LGBM_PARAMS, POINT_VALUE, ROUND_TRIP_COST,
                                        train_direction_model)

HORIZON_BARS = 5
THRESHOLD_QUANTILES = (0.80, 0.90, 0.95, 0.98, 0.99, 0.995)
ENTRY_OFFSET_TICKS = (0, 1)
WAIT_BARS = (1, 3, 6)
HOLD_BARS = (5, 12, 60)
MIN_VALIDATION_FILLS = 200
SEED = 11
RESULTS_PATH = "evaluate_quote_passive.json"


def quotes_for(data, rows: np.ndarray, seconds: np.ndarray) -> Quotes:
    return Quotes(bid=data["bid"].to_numpy(float)[rows], ask=data["ask"].to_numpy(float)[rows],
                  high=data["High"].to_numpy(float)[rows], low=data["Low"].to_numpy(float)[rows],
                  is_last=last_bar_before_gap(seconds[rows]))


def candidate_rules(score: np.ndarray) -> list[QuotePassiveRule]:
    thresholds = np.quantile(np.abs(score), THRESHOLD_QUANTILES)
    return [QuotePassiveRule(*combination) for combination in
            itertools.product(thresholds, ENTRY_OFFSET_TICKS, WAIT_BARS, HOLD_BARS, EXIT_MODES)]


def best_rule(score: np.ndarray, quotes: Quotes) -> tuple[QuotePassiveRule | None, dict]:
    """The rule with the highest net on this block, among those that fill often enough."""
    chosen, best = None, {"net": -np.inf}
    for rule in candidate_rules(score):
        result = simulate(score, quotes, rule, POINT_VALUE, ROUND_TRIP_COST)
        if result["fills"] >= MIN_VALIDATION_FILLS and result["net"] > best["net"]:
            chosen, best = rule, result
    return chosen, best


def describe(name: str, result: dict) -> str:
    per_trade = result["net"] / result["fills"] if result["fills"] else float("nan")
    return (f"{name:<16}orders {result['orders']:>7,}  fills {result['fills']:>7,}  "
            f"filled {result['fill_rate']:>6.1%}  win {result['win_rate']:>6.1%}  "
            f"gross ${result['gross']:>11,.0f}  net ${result['net']:>11,.0f}  "
            f"per trade ${per_trade:>8,.2f}")


def main() -> None:
    started = time.time()
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    columns = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    X = data[columns].to_numpy(np.float32)
    splits = purged_fold_splits(data["time"], HORIZON_BARS)
    print(f"{len(columns)} columns, horizon {HORIZON_BARS}, purge {HORIZON_BARS}, "
          f"${ROUND_TRIP_COST:.2f} a round trip, {len(splits)} dev folds "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    saved = {}
    for fold_number, split in enumerate(splits, start=1):
        model = train_direction_model(X, target, split, {**LGBM_PARAMS, "random_state": SEED})
        validation_score = model.predict_proba(X[split.validation])[:, 1] - 0.5
        test_score = model.predict_proba(X[split.test])[:, 1] - 0.5
        validation_quotes = quotes_for(data, split.validation, seconds)
        test_quotes = quotes_for(data, split.test, seconds)

        rule, validation_result = best_rule(validation_score, validation_quotes)
        print(f"fold {fold_number}: {len(split.test):,} test bars ({time.time() - started:.0f}s)")
        if rule is None:
            print(f"   no rule reached {MIN_VALIDATION_FILLS} validation fills; nothing traded\n")
            continue
        print(f"   chose threshold {rule.threshold:.4f}, offset {rule.entry_offset_ticks} tick, "
              f"wait {rule.wait_bars}, hold {rule.hold_bars}, exit {rule.exit_mode} "
              f"(validation net ${validation_result['net']:,.0f})")
        crossing = simulate_crossing(test_score, test_quotes, rule.threshold, rule.hold_bars,
                                     POINT_VALUE, ROUND_TRIP_COST)
        passive = simulate(test_score, test_quotes, rule, POINT_VALUE, ROUND_TRIP_COST)
        passive_cross = simulate(test_score, test_quotes,
                                 QuotePassiveRule(rule.threshold, rule.entry_offset_ticks,
                                                  rule.wait_bars, rule.hold_bars, CROSS_EXIT),
                                 POINT_VALUE, ROUND_TRIP_COST)
        print("   " + describe("crossing", crossing))
        print("   " + describe("passive/cross", passive_cross))
        print("   " + describe("chosen rule", passive) + "\n", flush=True)
        saved[fold_number] = {"rule": rule.__dict__, "crossing": crossing,
                              "passive_cross": passive_cross, "chosen": passive}

    with open(RESULTS_PATH, "w") as handle:
        json.dump(saved, handle, indent=1, default=float)
    for name in ("crossing", "passive_cross", "chosen"):
        fills = sum(fold[name]["fills"] for fold in saved.values())
        net = sum(fold[name]["net"] for fold in saved.values())
        print(f"{name:<16}{fills:>8,} fills  net ${net:>12,.0f}  "
              f"per trade ${net / fills if fills else float('nan'):>8,.2f}")


if __name__ == "__main__":
    main()
