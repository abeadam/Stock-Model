"""What does the ES mid actually do after a headline?

The feature test asked whether a model could use headlines. This asks the prior question:
whether there is anything to use. For every headline, the mid's move over the following
25 seconds, minute, five minutes and half hour, grouped by what the headline said and
whether the wire flagged it urgent, against the unconditional move over the same horizons.

If a positive headline is followed by no more upside than an average moment, then no
scoring scheme or feature window would have helped, and the lexicon is not what failed.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/news")
from news_features import add_news_features, load_headlines  # noqa: E402

from mid_feature_support import build_data  # noqa: E402
from mid_versus_trade_target import average_target  # noqa: E402

HORIZONS = (5, 12, 60, 360)
PERCENT = 100.0
POINT_VALUE = 50.0
FRESH_SECONDS = 60          # a bar counts as "just after" a headline for one minute


def describe(points: np.ndarray) -> str:
    if len(points) < 30:
        return f"{len(points):>7,}      too few"
    mean = points.mean()
    t = mean / (points.std(ddof=1) / np.sqrt(len(points)))
    return (f"{len(points):>7,}{mean * POINT_VALUE:>+11.2f}{(points > 0).mean():>9.1%}{t:>8.1f}")


def main() -> None:
    data = build_data()
    first, last = str(data["session_day"].min()), str(data["session_day"].max())
    data = add_news_features(data, load_headlines(first, last))
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    since = data["news_seconds_since_last"].to_numpy()
    fresh = since <= FRESH_SECONDS
    sentiment_now = np.where(fresh, data["news_sentiment_60"].to_numpy(), 0.0)
    results = {}
    for horizon in HORIZONS:
        move = average_target(mid, seconds, horizon) / PERCENT * mid
        usable = np.isfinite(move)
        print(f"\n=== {horizon} bars ({horizon * 5}s) ===")
        print(f"{'group':<34}{'bars':>7}{'mean $':>11}{'up':>9}{'t':>8}")
        groups = {"every bar": usable,
                  "within a minute of a headline": usable & fresh,
                  "   after a positive headline": usable & fresh & (sentiment_now > 0),
                  "   after a negative headline": usable & fresh & (sentiment_now < 0),
                  "   after a neutral headline": usable & fresh & (sentiment_now == 0),
                  "   after a wire alert": usable & fresh & (data["news_alerts_60"].to_numpy() > 0),
                  "quiet: no headline for an hour": usable & (since >= 3600)}
        for label, mask in groups.items():
            print(f"{label:<34}{describe(move[mask])}")
            results[f"{horizon}|{label.strip()}"] = {"bars": int(mask.sum()),
                                                     "mean_dollars": float(move[mask].mean() * POINT_VALUE)
                                                     if mask.sum() else None}
    with open("news_event_study.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
