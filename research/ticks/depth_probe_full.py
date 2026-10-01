"""The depth probe at full scale, after the first run used 13% of the session.

The first pass drew 2,000 decisions a side from a pool of 15,264 and trained on 2,052
fills with up to 44 features. At that size an interaction-driven gain is fragile whether
or not it is real, and the L1 baseline came out at 0.59 against the 0.6767 the
multi-session walk-forward records -- a gap that itself suggests starvation rather than a
property of the session.

This draws the whole pool. Same arms, same splits, same seeds, so the only change is the
amount of data.
"""
from __future__ import annotations

import json
import time

import numpy as np

from depth_features import DEPTH_FEATURE_NAMES
from depth_toxicity_probe import PURGE_SECONDS, SESSION, arm_auc, fills_with_depth
from ofi_features import BEST_LEVEL_OFI_NAMES, MULTI_LEVEL_OFI_NAMES, OFI_FEATURE_NAMES
from subsecond_depth_features import SUBSECOND_FEATURE_NAMES

ALL_DECISIONS_PER_SIDE = 15_264
SPLITS = (0.40, 0.50, 0.60, 0.70, 0.80)
SEEDS = (11, 23, 37)
MIN_TEST_FILLS = 1000
RESULTS_PATH = "/Users/abeadam/dev/model/Stock-Model/scratchpad_backup/depth_probe_full.json"


def main() -> None:
    started = time.time()
    fills = fills_with_depth(SESSION, ALL_DECISIONS_PER_SIDE, 11)
    reserved = ("placed", "side", "toxic", "total")
    l1 = [name for name in fills.columns if name not in reserved
          and not name.startswith(("depth_", "ss_", "ofi_"))]
    arms = {"snapshot": DEPTH_FEATURE_NAMES, "subsecond": SUBSECOND_FEATURE_NAMES,
            "OFI best": BEST_LEVEL_OFI_NAMES, "OFI multi": MULTI_LEVEL_OFI_NAMES,
            "OFI all": OFI_FEATURE_NAMES}
    print(f"{SESSION}: {len(fills):,} fills, {fills['toxic'].mean():.1%} toxic "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    placed = fills["placed"].to_numpy()
    lifts = {name: [] for name in arms}
    bases = []
    print(f"{'split':>6}{'train':>8}{'test':>8}{'L1':>8}"
          + "".join(f"{name:>11}" for name in arms), flush=True)
    for share in SPLITS:
        cut = placed[int(len(placed) * share)]
        train = np.flatnonzero(placed <= cut)
        test = np.flatnonzero(placed > cut + PURGE_SECONDS)
        if len(test) < MIN_TEST_FILLS:
            continue
        base = float(np.mean([arm_auc(fills, l1, train, test, seed) for seed in SEEDS]))
        bases.append(base)
        row = f"{share:>6.0%}{len(train):>8,}{len(test):>8,}{base:>8.4f}"
        for name, extra in arms.items():
            score = float(np.mean([arm_auc(fills, l1 + extra, train, test, seed)
                                   for seed in SEEDS]))
            lifts[name].append(score - base)
            row += f"{score:>11.4f}"
        print(row + f"   ({time.time() - started:.0f}s)", flush=True)

    print(f"\n  baseline L1 across splits: {np.mean(bases):.4f} "
          f"(the 13%-sample run gave 0.5942; walk-forward records 0.6767)")
    print(f"\n  {'family':<12}{'lift':>10}{'splits up':>12}")
    for name, values in lifts.items():
        found = np.array(values)
        print(f"  {name:<12}{found.mean():>+10.4f}"
              f"{f'{int((found > 0).sum())}/{len(found)}':>12}")
    with open(RESULTS_PATH, "w") as handle:
        json.dump({"fills": len(fills), "bases": bases,
                   "lifts": {name: list(map(float, values)) for name, values in lifts.items()}},
                  handle, indent=1)
    print(f"\n  saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
