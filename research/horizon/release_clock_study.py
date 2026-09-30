"""Do the scheduled release slots actually show up in the data?

The big American numbers land on a clock: 08:30 Eastern for the labour, inflation and
growth releases, 10:00 for the survey data, 14:00 for the Federal Reserve. Only the last
two fall inside the stock session -- an 08:30 release has been in the price for an hour
before the opening bell.

Before building any feature around that clock, it is worth checking the clock is real.
This measures the typical size of a mid move minute by minute through the session, and
asks whether 10:00 and 14:00 stand out from their neighbours. If they do not, a calendar
feature is decoration.

Days are grouped by the rules that govern the main releases, all of which are known in
advance: the first business day of the month (ISM manufacturing), the third (ISM
services), the second Friday (Michigan sentiment), the last Tuesday (consumer
confidence) and the first Friday (the jobs report, at 08:30).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mid_feature_support import build_data
from mid_microstructure import TICK, trailing_change

EASTERN = "America/New_York"
SLOTS = {"10:00": (10, 0), "14:00": (14, 0)}
AROUND_MINUTES = 3


def business_day_of_month(days: pd.Series) -> pd.Series:
    """1 for the month's first trading day, 2 for the second, and so on."""
    frame = pd.DataFrame({"day": sorted(set(days))})
    frame["month"] = pd.to_datetime(frame["day"]).dt.to_period("M")
    frame["position"] = frame.groupby("month").cumcount() + 1
    return frame.set_index("day")["position"]


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    eastern = pd.to_datetime(data["time"]).dt.tz_localize("UTC").dt.tz_convert(EASTERN)
    minute_of_day = eastern.dt.hour * 60 + eastern.dt.minute
    size = np.abs(trailing_change(mid, seconds, 1)) / TICK
    usable = np.isfinite(size)
    typical = np.nanmedian(size[usable])
    print(f"a typical bar moves {typical:.2f} ticks; showing each minute against that\n")

    profile = pd.DataFrame({"minute": minute_of_day[usable], "size": size[usable]}) \
        .groupby("minute")["size"].mean()
    print("the busiest minutes of the session:")
    for minute, value in profile.nlargest(8).items():
        print(f"   {minute // 60:02d}:{minute % 60:02d}   {value:.3f} ticks   "
              f"{value / profile.mean():.2f}x the session average")

    print("\nthe release slots against the minutes either side:")
    for label, (hour, minute) in SLOTS.items():
        centre = hour * 60 + minute
        at = profile.reindex(range(centre, centre + AROUND_MINUTES)).mean()
        before = profile.reindex(range(centre - 10, centre)).mean()
        print(f"   {label}: {at:.3f} ticks in the {AROUND_MINUTES} minutes from, "
              f"{before:.3f} in the 10 before -> {at / before:.2f}x")

    position = business_day_of_month(pd.Series(data["session_day"].unique()))
    day = data["session_day"].to_numpy()
    at_ten = usable & (minute_of_day >= 600) & (minute_of_day < 603)
    groups = {"first business day": np.isin(day, position[position == 1].index),
              "third business day": np.isin(day, position[position == 3].index),
              "second Friday": np.array([d.weekday() == 4 and 8 <= d.day <= 14 for d in day]),
              "last Tuesday": np.array([d.weekday() == 1 and d.day > 21 for d in day]),
              "first Friday": np.array([d.weekday() == 4 and d.day <= 7 for d in day])}
    print(f"\nthe 10:00 slot by day type ({at_ten.sum():,} bars at 10:00 overall):")
    baseline = size[at_ten].mean()
    print(f"   {'every day':<22}{baseline:.3f} ticks")
    results = {}
    for label, mask in groups.items():
        chosen = at_ten & mask
        sessions = len(set(day[chosen]))
        if sessions < 3:
            continue
        value = size[chosen].mean()
        results[label] = {"sessions": sessions, "ticks": float(value), "ratio": float(value / baseline)}
        print(f"   {label:<22}{value:.3f} ticks   {value / baseline:.2f}x   ({sessions} sessions)")
    with open("release_clock_study.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
