"""Do headline features improve the model enough to keep them?

The same paired design that overruled the VIP screen: every configuration trained with the
same seeds on the same folds, differences taken pair by pair, so an effect of a thousandth
is not read through the model's own randomness.

Three questions, because a news feature can help in more than one way:
  all bars      does it lift the model everywhere, which it can only do if news is frequent
  news bars     does it lift the model on the bars near a headline, where it has something
                to say -- a small share of bars, so a large local effect can hide in a
                global average
  volatility    do headlines predict the size of the coming move even if not its direction,
                which would make them useful for sizing rather than for side
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/news")
from news_features import NEWS_FEATURES, add_news_features, load_headlines  # noqa: E402

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
RECENT_NEWS_SECONDS = 900


def main() -> None:
    data = build_data()
    first, last = str(data["session_day"].min()), str(data["session_day"].max())
    headlines = load_headlines(first, last)
    data = add_news_features(data, headlines)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    target = average_target(mid, seconds, HORIZON_BARS)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    range_target = forward_target(prices, HORIZON_BARS, "range")
    decided = np.isfinite(target) & (target != 0)
    near_news = data["news_seconds_since_last"].to_numpy() <= RECENT_NEWS_SECONDS
    base = list(FEATURES) + RETURNS_OF_NEW_DATA + MID_FEATURES
    sets = {"base": base, "plus_news": base + NEWS_FEATURES}
    print(f"{len(headlines):,} headlines over {first} to {last}; "
          f"{near_news.mean():.1%} of regular-hours bars are within "
          f"{RECENT_NEWS_SECONDS // 60} minutes of one\n")

    direction, local, volatility = ({name: [] for name in sets} for _ in range(3))
    for seed in SEEDS:
        params = {**LGBM_PARAMS, "random_state": seed}
        for name, columns in sets.items():
            X = data[columns].to_numpy(np.float32)
            for split in fold_splits(data, STAGE):
                model = train_direction_model(X, target, split, params)
                test = split.test[decided[split.test]]
                score = model.predict_proba(X[test])[:, 1]
                up = (target[test] > 0).astype(int)
                direction[name].append(roc_auc_score(up, score))
                on_news = near_news[test]
                local[name].append(roc_auc_score(up[on_news], score[on_news])
                                   if 0 < up[on_news].sum() < on_news.sum() else np.nan)
                if seed == SEEDS[0]:
                    sized = fit_regressor(X, range_target, split)
                    known = split.test[np.isfinite(range_target[split.test])]
                    volatility[name].append(r2_score(range_target[known], sized.predict(X[known])))
        print(f"   seed {seed}: " + "  ".join(f"{n} {np.mean(direction[n][-2:]):.4f}" for n in sets),
              flush=True)

    print(f"\n{'measure':<26}{'base':>9}{'plus news':>11}{'difference':>12}{'std err':>10}{'t':>7}")
    for label, table in (("direction, all bars", direction), ("direction, news bars", local),
                         ("volatility R2", volatility)):
        first_values = np.array(table["base"], float)
        second = np.array(table["plus_news"], float)
        usable = np.isfinite(first_values) & np.isfinite(second)
        difference = second[usable] - first_values[usable]
        error = difference.std(ddof=1) / np.sqrt(len(difference)) if len(difference) > 1 else np.nan
        t = difference.mean() / error if error and error > 0 else np.nan
        print(f"{label:<26}{first_values[usable].mean():>9.4f}{second[usable].mean():>11.4f}"
              f"{difference.mean():>+12.5f}{error:>10.5f}{t:>7.2f}")
    with open("news_ablation.json", "w") as handle:
        json.dump({"direction": {k: list(map(float, v)) for k, v in direction.items()},
                   "news_bars": {k: [float(x) for x in v] for k, v in local.items()},
                   "volatility": {k: list(map(float, v)) for k, v in volatility.items()}}, handle, indent=1)


if __name__ == "__main__":
    main()
