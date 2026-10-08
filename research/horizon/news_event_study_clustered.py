"""The same event study, with the two things that make the raw version misleading removed.

**Overlap.** A 30-minute forward window computed on every 5-second bar reuses the same
half hour 360 times. Treating those as independent observations inflates a t-statistic by
roughly the square root of the overlap -- about nineteen at that horizon -- which turns
noise into certainty. Here each day contributes one number per group, and the test is
across days, so the sample is the 123 sessions it really is.

**Time of day.** An hour without a headline is not a random hour; it is the afternoon.
Since the forward move has its own shape across the session, comparing quiet bars to all
bars compares two different times of day. Each group is therefore measured against the
average move *in the same hour*, so only the news difference is left.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/news")
from news_features import add_news_features, load_headlines  # noqa: E402

from mid_feature_support import build_data  # noqa: E402
from mid_versus_trade_target import average_target  # noqa: E402

HORIZONS = (5, 12, 60, 360)
PERCENT = 100.0
POINT_VALUE = 50.0
FRESH_SECONDS = 60
MIN_DAYS = 30


def by_day(excess: np.ndarray, mask: np.ndarray, day: np.ndarray) -> tuple[float, float, int]:
    """One number per session, then a t-test across sessions."""
    frame = pd.DataFrame({"excess": excess[mask], "day": day[mask]}).dropna()
    daily = frame.groupby("day")["excess"].mean()
    if len(daily) < MIN_DAYS:
        return np.nan, np.nan, len(daily)
    return daily.mean(), daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))), len(daily)


def main() -> None:
    data = build_data()
    first, last = str(data["session_day"].min()), str(data["session_day"].max())
    data = add_news_features(data, load_headlines(first, last))
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    day = data["session_day"].to_numpy()
    hour = pd.to_datetime(data["time"]).dt.hour.to_numpy()
    since = data["news_seconds_since_last"].to_numpy()
    fresh = since <= FRESH_SECONDS
    sentiment_now = np.where(fresh, data["news_sentiment_60"].to_numpy(), 0.0)
    results = {}
    for horizon in HORIZONS:
        move = average_target(mid, seconds, horizon) / PERCENT * mid * POINT_VALUE
        # Each bar against the average move in its own hour, so time of day cannot speak.
        hourly = pd.Series(move).groupby(hour).transform("mean").to_numpy()
        excess = move - hourly
        groups = {"within a minute of a headline": fresh,
                  "after a positive headline": fresh & (sentiment_now > 0),
                  "after a negative headline": fresh & (sentiment_now < 0),
                  "after a wire alert": fresh & (data["news_alerts_60"].to_numpy() > 0),
                  "quiet for an hour": since >= 3600}
        print(f"\n=== {horizon} bars ({horizon * 5}s), excess over the same hour, by session ===")
        print(f"{'group':<32}{'sessions':>10}{'mean $':>10}{'t':>8}   verdict")
        for label, mask in groups.items():
            mean, t, days = by_day(excess, mask, day)
            verdict = "" if not np.isfinite(t) else ("real" if abs(t) > 2 else "noise")
            results[f"{horizon}|{label}"] = {"mean_dollars": None if not np.isfinite(mean) else float(mean),
                                             "t": None if not np.isfinite(t) else float(t), "days": days}
            print(f"{label:<32}{days:>10}"
                  + (f"{mean:>+10.2f}{t:>8.2f}   {verdict}" if np.isfinite(mean) else f"{'too few':>18}"))
    with open("news_event_study_clustered.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
