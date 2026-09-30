"""The one holdout check: does the PLS rotation survive August and September?

Pre-registered before running, and only these three configurations are scored, so the
holdout is not shopped:
  raw 44 production features
  PLS-2  two supervised columns
  PLS-5  five supervised columns

Pass condition, fixed in advance: a variant passes only if it beats raw 44 in BOTH
holdout months. Better on average while losing a month is a fail. Dev said
raw 0.5024, PLS-2 0.5065, PLS-5 0.5081.
"""
from __future__ import annotations

import json

import numpy as np

from rotation_test import pls_transform, score_raw, score_rotated
from segment_walk_forward import load_cut, mid_move
from short_horizon_walk_forward import FEATURES
from rotation_test import TARGET_BARS

HOLDOUT = "holdout"
DEV_RESULT = {"raw": 0.5024, "pls_2": 0.5065, "pls_5": 0.5081}


def main() -> None:
    data = load_cut()
    move = mid_move(data, TARGET_BARS)
    X = data[FEATURES].to_numpy(np.float32)
    results = {"raw": score_raw(X, move, data, HOLDOUT)}
    for components in (2, 5):
        results[f"pls_{components}"] = score_rotated(X, move, data, pls_transform, components, HOLDOUT)
    raw_mean, raw_folds = results["raw"]
    print(f"\n{'model':<8}{'dev':>9}{'holdout':>10}{'August':>10}{'September':>11}   verdict")
    for name, (mean, folds) in results.items():
        verdict = ""
        if name != "raw":
            beats = [folds[i] > raw_folds[i] for i in range(len(folds))]
            verdict = "PASS" if all(beats) else f"FAIL ({sum(beats)} of {len(beats)} months)"
        print(f"{name:<8}{DEV_RESULT[name]:>9.4f}{mean:>10.4f}{folds[0]:>10.4f}{folds[1]:>11.4f}   {verdict}")
    with open("holdout_pls_test.json", "w") as handle:
        json.dump({name: {"mean": mean, "folds": folds} for name, (mean, folds) in results.items()},
                  handle, indent=1)


if __name__ == "__main__":
    main()
