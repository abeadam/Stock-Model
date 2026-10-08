"""Do FinBERT headline scores add anything to the ES model?

The data_loader study found that these scores do not correlate with SPY's next thirty
seconds. That is a univariate finding; a column can be useless alone and still carry
something once a model has a hundred and sixty others to combine it with. This is the
paired ablation that decides, the same design that overruled the VIP screen.

The news data ends 2026-07-23, so only sessions it actually covers are scored -- an
uncovered day would otherwise read as a very quiet one and dilute the comparison. The
August-September holdout has no coverage at all, so this is a development-fold answer.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/news")
from finbert_news_features import (FINBERT_FEATURES, add_finbert_features,  # noqa: E402
                                   covered_days, load_scored_headlines)

from indicator_test import RETURNS_OF_NEW_DATA  # noqa: E402
from mid_feature_support import build_data  # noqa: E402
from mid_microstructure import MID_FEATURES  # noqa: E402
from mid_versus_trade_target import average_target  # noqa: E402
from rotation_test import fold_splits  # noqa: E402
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model  # noqa: E402

HORIZON_BARS = 5
STAGE = "dev"
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)


def main() -> None:
    data = build_data()
    headlines = load_scored_headlines()
    data = add_finbert_features(data, headlines)
    covered = data["session_day"].isin(covered_days(headlines)).to_numpy()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    target = average_target(data["mid"].to_numpy(float), seconds, HORIZON_BARS)
    decided = np.isfinite(target) & (target != 0) & covered
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    sets = {"base": base, "plus_finbert": base + FINBERT_FEATURES}
    fresh = data["fb_seconds_since_last"].to_numpy() <= 900
    print(f"{len(headlines):,} scored headlines; {covered.mean():.0%} of regular-hours bars are on "
          f"covered sessions, {np.mean(fresh[covered]):.0%} of those within 15 minutes of a headline\n")
    scores = {name: [] for name in sets}
    local = {name: [] for name in sets}
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        for name, columns in sets.items():
            X = data[columns].to_numpy(np.float32)
            for split in fold_splits(data, STAGE):
                model = train_direction_model(X, target, split, params)
                test = split.test[decided[split.test]]
                if len(test) < 1000:
                    continue
                probability = model.predict_proba(X[test])[:, 1]
                up = (target[test] > 0).astype(int)
                scores[name].append(roc_auc_score(up, probability))
                near = fresh[test]
                local[name].append(roc_auc_score(up[near], probability[near])
                                   if 0 < up[near].sum() < near.sum() else np.nan)
        print(f"   seed {seed}: " + "  ".join(f"{n} {np.mean(scores[n][-2:]):.4f}" for n in sets),
              flush=True)
    print(f"\n{'measure':<26}{'base':>9}{'plus FinBERT':>14}{'difference':>12}{'std err':>10}{'t':>7}")
    for label, table in (("direction, covered bars", scores), ("direction, near a headline", local)):
        first = np.array(table["base"], float)
        second = np.array(table["plus_finbert"], float)
        usable = np.isfinite(first) & np.isfinite(second)
        difference = second[usable] - first[usable]
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        print(f"{label:<26}{first[usable].mean():>9.4f}{second[usable].mean():>14.4f}"
              f"{difference.mean():>+12.5f}{error:>10.5f}{difference.mean() / error:>7.2f}")
    with open("finbert_ablation.json", "w") as handle:
        json.dump({k: list(map(float, v)) for k, v in scores.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
