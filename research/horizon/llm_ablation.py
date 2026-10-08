"""Does the model's judgement of headlines improve the ES model?

FinBERT's tone scores did not (t -0.93). The claim being tested now is different: that
what matters is not how a headline sounds but whether the *index* should care, which is a
judgement a language model can make and a tone classifier cannot.

Same paired design as every feature decision today -- eight seeds, same folds, same
randomness on both sides, differences taken pair by pair. Only sessions the news data
covers are scored. Three readings, because a news signal could help in three ways:
direction everywhere, direction where news is fresh, or the size of the coming move.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/news")
from llm_news_features import LLM_FEATURES, add_llm_features, load_scored  # noqa: E402

from forward_extremes import forward_target  # noqa: E402
from indicator_test import RETURNS_OF_NEW_DATA  # noqa: E402
from mid_feature_support import build_data  # noqa: E402
from mid_microstructure import MID_FEATURES  # noqa: E402
from mid_versus_trade_target import average_target  # noqa: E402
from rotation_test import fold_splits  # noqa: E402
from short_horizon_walk_forward import FEATURES, LGBM_PARAMS, train_direction_model  # noqa: E402
from volume_target_test import fit_regressor  # noqa: E402

HORIZON_BARS = 5
STAGE = "dev"
SEEDS = (11, 23, 37, 51, 67, 83, 97, 113)
FRESH_SECONDS = 900


def main() -> None:
    data = build_data()
    scored = load_scored()
    data = add_llm_features(data, scored)
    import pandas as pd
    covered_days = set(pd.to_datetime(scored["available_at"], unit="s", utc=True)
                       .dt.tz_convert("America/New_York").dt.date)
    covered = data["session_day"].isin(covered_days).to_numpy()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    target = average_target(mid, seconds, HORIZON_BARS)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    range_target = forward_target(prices, HORIZON_BARS, "range")
    decided = np.isfinite(target) & (target != 0) & covered
    fresh = data["llm_seconds_since_relevant"].to_numpy() <= FRESH_SECONDS
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    sets = {"base": base, "plus_llm": base + LLM_FEATURES}
    print(f"{len(scored):,} scored headlines; {covered.mean():.0%} of bars on covered sessions; "
          f"{np.mean(fresh[covered]):.0%} of those within 15 min of index-relevant news\n")
    direction, local, volatility = ({name: [] for name in sets} for _ in range(3))
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        for name, columns in sets.items():
            X = data[columns].to_numpy(np.float32)
            for split in fold_splits(data, STAGE):
                model = train_direction_model(X, target, split, params)
                test = split.test[decided[split.test]]
                probability = model.predict_proba(X[test])[:, 1]
                up = (target[test] > 0).astype(int)
                direction[name].append(roc_auc_score(up, probability))
                near = fresh[test]
                local[name].append(roc_auc_score(up[near], probability[near])
                                   if 0 < up[near].sum() < near.sum() else np.nan)
                if seed == SEEDS[0]:
                    sized = fit_regressor(X, range_target, split)
                    known = split.test[np.isfinite(range_target[split.test]) & covered[split.test]]
                    volatility[name].append(r2_score(range_target[known], sized.predict(X[known])))
        print(f"   seed {seed}: " + "  ".join(f"{n} {np.mean(direction[n][-2:]):.4f}" for n in sets),
              flush=True)
    print(f"\n{'measure':<30}{'base':>9}{'plus LLM':>11}{'difference':>12}{'std err':>10}{'t':>7}")
    for label, table in (("direction, covered bars", direction),
                         ("direction, near relevant news", local), ("volatility R2", volatility)):
        first = np.array(table["base"], float)
        second = np.array(table["plus_llm"], float)
        usable = np.isfinite(first) & np.isfinite(second)
        gap = second[usable] - first[usable]
        error = gap.std(ddof=1) / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
        print(f"{label:<30}{first[usable].mean():>9.4f}{second[usable].mean():>11.4f}"
              f"{gap.mean():>+12.5f}{error:>10.5f}{gap.mean() / error if error else np.nan:>7.2f}")
    with open("llm_ablation.json", "w") as handle:
        json.dump({k: list(map(float, v)) for k, v in direction.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
