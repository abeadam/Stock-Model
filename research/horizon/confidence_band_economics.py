"""Does trading only the confident bars pay, at the 18-bar horizon?

The intuitive fix for an unprofitable model is to trade less and pick better. It has been
measured before at five bars and failed for a specific reason: accuracy does rise with
confidence, but the moves the model is confident about are *smaller*, so the accuracy
needed to break even rises just as fast. Its confidence is a volatility forecast wearing a
direction's clothes.

This re-measures it at 18 bars, where the economics are least bad, and shows the whole
curve rather than one chosen cut. For each confidence band:

  accuracy      share of decided bars called correctly
  median move   the typical |move| in dollars for bars in that band
  break-even    0.5 + cost / (2 * move), the accuracy that band would need
  margin        accuracy - break-even, which is the only column that matters

If the margin is negative everywhere and falls as confidence rises, no threshold can
rescue the model and the question is settled for this feature set.

usage: confidence_band_economics.py [HORIZON_BARS]   (default 18)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from cost_aware_targets import move_in_dollars
from indicator_test import RETURNS_OF_NEW_DATA
from mid_feature_support import build_data
from mid_microstructure import MID_FEATURES
from mid_versus_trade_target import average_target
from purged_folds import purged_fold_splits
from short_horizon_walk_forward import (FEATURES, LGBM_PARAMS, POINT_VALUE, ROUND_TRIP_COST,
                                        train_direction_model)

HORIZON_BARS = 18
SEED = 11
BAND_EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)


def break_even_accuracy(move_dollars: float, cost: float) -> float:
    """The accuracy a trade of this size needs to cover the round trip."""
    return 0.5 + cost / (2.0 * move_dollars) if move_dollars > 0 else float("nan")


def main(horizon: int = HORIZON_BARS) -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    target = average_target(mid, epochs, horizon)
    dollars = move_in_dollars(target, mid, POINT_VALUE)
    decided = np.isfinite(target) & (target != 0)
    X = data[list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES].to_numpy(np.float32)
    splits = purged_fold_splits(data["time"], horizon)
    print(f"horizon {horizon}, purge {horizon}, ${ROUND_TRIP_COST:.2f} a round trip "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    confidence, correct, size = [], [], []
    for split in splits:
        model = train_direction_model(X, target, split, {**LGBM_PARAMS, "random_state": SEED})
        rows = split.test[decided[split.test]]
        probability = model.predict_proba(X[rows])[:, 1]
        confidence.append(np.abs(probability - 0.5))
        correct.append(((probability > 0.5) == (target[rows] > 0)).astype(float))
        size.append(np.abs(dollars[rows]))
    confidence = np.concatenate(confidence)
    correct = np.concatenate(correct)
    size = np.concatenate(size)

    print(f"{'confidence band':<20}{'bars':>9}{'accuracy':>10}{'median move':>13}"
          f"{'break-even':>12}{'margin':>9}{'$/trade':>10}")
    for lower, upper in zip(BAND_EDGES, BAND_EDGES[1:]):
        low, high = np.quantile(confidence, lower), np.quantile(confidence, upper)
        band = (confidence >= low) & (confidence <= high) if upper == 1.0 else \
               (confidence >= low) & (confidence < high)
        if band.sum() < 100:
            continue
        accuracy = correct[band].mean()
        median_move = float(np.median(size[band]))
        needed = break_even_accuracy(median_move, ROUND_TRIP_COST)
        per_trade = (2 * accuracy - 1) * size[band].mean() - ROUND_TRIP_COST
        print(f"{f'{lower:.0%}-{upper:.0%}':<20}{band.sum():>9,}{accuracy:>10.2%}"
              f"{median_move:>13,.2f}{needed:>12.2%}{accuracy - needed:>+9.2%}{per_trade:>10.2f}")
    print("\nMargin is accuracy minus what that band needs. Only a positive margin can pay.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else HORIZON_BARS)
