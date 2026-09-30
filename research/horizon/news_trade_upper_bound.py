"""If the news arrived instantly, would trading it pay?

The feature tests asked whether headlines help a model. This asks the simpler and more
generous question: assume the headline reaches us the second it prints, assume we act at
once at the mid, and trade the direction the model called. What comes back?

It is deliberately an upper bound, and every choice favours the strategy:
  only Benzinga, whose timestamps are second-precise, so no minute of uncertainty is added;
  entry at the first bar at or after the headline, filled at the **mid** rather than
  crossing the spread, which no market order achieves;
  no queue, no partial fills, no slippage, and one $5 round trip.

Both readings of the signal are reported -- trading with the model's call and against it --
because the event study hinted the market fades good news at this horizon. Reporting only
the better one would be choosing a side after seeing the answer, so both are shown and
neither is claimed on its own.

Significance is clustered by session, since headlines cluster in time and overlapping
windows would otherwise manufacture certainty.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/news")
from llm_headline_score import CACHE_PATH, headline_key  # noqa: E402
from score_headlines_with_llm import STUDY_DIR  # noqa: E402

from mid_feature_support import build_data  # noqa: E402
from mid_versus_trade_target import average_target  # noqa: E402

HORIZONS = (5, 12, 60)
RELEVANCE_LEVELS = (0.5, 0.8)
PERCENT = 100.0
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
MIN_TRADES = 40


def benzinga_scored() -> pd.DataFrame:
    """Second-precise headlines with the model's call attached."""
    frame = pd.read_csv(os.path.join(STUDY_DIR, "dataset_benzinga.csv"),
                        usecols=["published_utc", "headline"])
    stamped = pd.to_datetime(frame["published_utc"], format="mixed", utc=True)
    frame = pd.DataFrame({"at": stamped.dt.tz_localize(None).astype("datetime64[s]").astype(np.int64),
                          "key": frame["headline"].map(headline_key)})
    return frame.merge(pd.read_csv(CACHE_PATH), on="key").sort_values("at").reset_index(drop=True)


def settle(points: np.ndarray, side: np.ndarray, day: np.ndarray) -> dict:
    """Per-trade money, and a t-statistic across sessions rather than across trades."""
    gained = side * points * POINT_VALUE - ROUND_TRIP_COST
    daily = pd.DataFrame({"gained": gained, "day": day}).groupby("day")["gained"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 2 else np.nan
    return {"trades": len(gained), "sessions": len(daily), "accuracy": float((side * points > 0).mean()),
            "per_trade": float(gained.mean()), "total": float(gained.sum()), "t": float(t)}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    day = data["session_day"].to_numpy()
    headlines = benzinga_scored()
    # The first regular-hours bar at or after the headline: where an instant reaction lands.
    landing = np.searchsorted(seconds, headlines["at"].to_numpy(), side="left")
    inside = landing < len(seconds)
    headlines, landing = headlines[inside].reset_index(drop=True), landing[inside]
    same_minute = seconds[landing] - headlines["at"].to_numpy() <= 60
    print(f"{len(headlines):,} scored Benzinga headlines; {same_minute.sum():,} land on a "
          f"regular-hours bar within a minute\n")
    results = {}
    print(f"{'horizon':>8}{'relevance':>11}{'reading':>9}{'trades':>8}{'sessions':>10}"
          f"{'accuracy':>10}{'$/trade':>10}{'total $':>11}{'t':>7}")
    for horizon in HORIZONS:
        move = average_target(mid, seconds, horizon) / PERCENT * mid
        for level in RELEVANCE_LEVELS:
            chosen = same_minute & (headlines["relevance"].to_numpy() >= level) \
                     & (headlines["direction"].to_numpy() != 0) & np.isfinite(move[landing])
            if chosen.sum() < MIN_TRADES:
                continue
            rows = landing[chosen]
            call = np.sign(headlines["direction"].to_numpy()[chosen])
            for reading, side in (("follow", call), ("fade", -call)):
                outcome = settle(move[rows], side, day[rows])
                results[f"{horizon}|{level}|{reading}"] = outcome
                print(f"{horizon:>8}{level:>11.1f}{reading:>9}{outcome['trades']:>8,}"
                      f"{outcome['sessions']:>10}{outcome['accuracy']:>9.1%}"
                      f"{outcome['per_trade']:>+10.2f}{outcome['total']:>+11,.0f}{outcome['t']:>7.2f}")
    with open("news_trade_upper_bound.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
