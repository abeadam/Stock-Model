"""Does the book behind the touch predict a toxic fill better than the touch alone?

Not a verdict -- one session of depth exists (2026-09-30, 11:45 to 16:00, the launch
agent was loaded mid-session) and the toxicity model walks forward across sessions. This
asks the prior question: does depth carry information about toxicity at all, which
decides whether a week of collection is worth waiting for.

Arms share fills, folds and seeds, so only the feature set differs:

  L1        what the 0.6767 model already sees: touch size, imbalance, depletion, flow
  L1+depth  plus the five-level book -- total size, size behind the touch, what share
            sits at the touch, and how fast each side's whole book is draining
  depth     the book alone, to show whether it is adding or merely restating

A single session has no session folds, so it is split in time with a purge: the test
block starts HOLD_SECONDS + ENTRY_LIMIT_SECONDS after the last training placement, which
is the longest a training label can reach forward.

usage: depth_toxicity_probe.py [DECISIONS_PER_SIDE]
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from depth_features import DEPTH_FEATURE_NAMES, at_placement_depth, book_frame, load_depth
from subsecond_depth_features import (SUBSECOND_FEATURE_NAMES, at_placement_subsecond, replay,
                                      subsecond_frame)
from toxicity_model import (ENTRY_LIMIT_SECONDS, HOLD_SECONDS, LGBM_PARAMS, fill_with_reason)
from fill_features import at_placement, session_frame
from markout import BUY, POINT_VALUE, SELL, fresh_seconds, mid_at, queue_ahead, touch_price
from markout_study import load_book
from tick_probe import load_day

SESSION = "2026-09-30"
DECISIONS_PER_SIDE = 4000
SEED = 11
TRAIN_SHARE = 0.6
# The longest a training label reaches forward: time to fill, then the hold.
PURGE_SECONDS = int(ENTRY_LIMIT_SECONDS) + HOLD_SECONDS
SEEDS = (11, 23, 37, 51, 67)


def fills_with_depth(day: str, decisions_per_side: int, seed: int) -> pd.DataFrame:
    """Fills restricted to the seconds depth covers, carrying both feature sets."""
    book, grid = load_book(day), load_day(day)
    fresh, frame = fresh_seconds(grid), session_frame(grid)
    shaped = book_frame(load_depth(day))
    covered = set(shaped.index.astype(int))
    pool = np.array(sorted(fresh & covered))
    fine = subsecond_frame(replay(day), pool)
    generator = np.random.default_rng(seed)
    rows = []
    for side in (BUY, SELL):
        for place in generator.choice(pool, size=min(decisions_per_side, len(pool)), replace=False):
            price = touch_price(book, float(place), side)
            ahead = queue_ahead(book, float(place), side)
            if not np.isfinite(price) or not np.isfinite(ahead):
                continue
            features = at_placement(frame, int(place), side)
            depth = at_placement_depth(shaped, int(place), side)
            fast = at_placement_subsecond(fine, int(place), side)
            if features is None or depth is None or fast is None:
                continue
            if not all(np.isfinite(list(features.values()) + list(depth.values())
                                   + list(fast.values()))):
                continue
            got = fill_with_reason(book, float(place), price, side, ahead, ENTRY_LIMIT_SECONDS)
            if got is None:
                continue
            fill_time, reason = got
            later = int(fill_time) + HOLD_SECONDS
            if int(fill_time) not in fresh or later not in fresh:
                continue
            mid_place, mid_later = mid_at(book, float(place)), mid_at(book, float(later))
            if not all(np.isfinite([mid_place, mid_later])):
                continue
            rows.append({"placed": int(place), "side": side,
                         "toxic": int(reason == "through"),
                         "total": side * (mid_later - price) * POINT_VALUE,
                         **features, **depth, **fast})
    return pd.DataFrame(rows).sort_values("placed", kind="stable").reset_index(drop=True)


def time_split(fills: pd.DataFrame, train_share: float = TRAIN_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Earlier placements train, later ones test, with a gap no label can cross."""
    placed = fills["placed"].to_numpy()
    cut = placed[int(len(placed) * train_share)]
    train = np.flatnonzero(placed <= cut)
    test = np.flatnonzero(placed > cut + PURGE_SECONDS)
    return train, test


def arm_auc(fills: pd.DataFrame, columns: list[str], train, test, seed: int) -> float:
    model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "random_state": seed})
    model.fit(fills.loc[train, columns], fills.loc[train, "toxic"])
    return roc_auc_score(fills.loc[test, "toxic"],
                         model.predict_proba(fills.loc[test, columns])[:, 1])


def main(decisions_per_side: int = DECISIONS_PER_SIDE) -> None:
    fills = fills_with_depth(SESSION, decisions_per_side, SEED)
    reserved = ("placed", "side", "toxic", "total")
    l1 = [name for name in fills.columns if name not in reserved
          and not name.startswith("depth_") and not name.startswith("ss_")]
    arms = {"L1": l1,
            "L1+snapshot": l1 + DEPTH_FEATURE_NAMES,
            "L1+subsecond": l1 + SUBSECOND_FEATURE_NAMES,
            "L1+both": l1 + DEPTH_FEATURE_NAMES + SUBSECOND_FEATURE_NAMES,
            "snapshot only": DEPTH_FEATURE_NAMES,
            "subsecond only": SUBSECOND_FEATURE_NAMES}
    train, test = time_split(fills)
    print(f"{SESSION}: {len(fills):,} fills, {fills['toxic'].mean():.1%} toxic")
    print(f"  train {len(train):,} fills, test {len(test):,}, purge {PURGE_SECONDS}s\n")
    print(f"  {'arm':<12}{'columns':>9}{'AUC':>9}{'sd':>8}")
    scores = {}
    for name, columns in arms.items():
        values = np.array([arm_auc(fills, columns, train, test, seed) for seed in SEEDS])
        scores[name] = values
        print(f"  {name:<12}{len(columns):>9}{values.mean():>9.4f}{values.std(ddof=1):>8.4f}")
    print()
    for name in ("L1+snapshot", "L1+subsecond", "L1+both"):
        lift = scores[name] - scores["L1"]
        print(f"  {name:<14} adds {lift.mean():+.4f} AUC over L1 "
              f"({int((lift > 0).sum())}/{len(lift)} seeds up)")
    print(f"  the recorded 8-session L1 figure is 0.6767; this is one session, so read the\n"
          f"  DIFFERENCE between arms, not the level.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DECISIONS_PER_SIDE)
