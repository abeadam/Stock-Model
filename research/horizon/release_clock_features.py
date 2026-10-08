"""Where the bar sits relative to the release clock.

American data lands on a fixed clock, and two of those slots fall inside the stock
session: 10:00 Eastern for the survey releases, 14:00 for the Federal Reserve. The dated
calendar -- which month's tenth is a CPI day -- sits behind IBKR's Wall Street Horizon
subscription, which this account does not have, so these columns encode the clock rather
than the calendar. That is the half that needs no subscription, and the data says it is
the half that shows: 10:00 is the fourth busiest minute of the session.

The production features already carry the hour as a sine and cosine, which is far too
smooth to express "the next three minutes are different from the last three". These are
minute-resolution and signed, so before and after a slot are distinguishable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EASTERN = "America/New_York"
SLOT_MINUTES = {"ten": 10 * 60, "two": 14 * 60}
FAR_AWAY = 240.0                    # minutes, beyond which "how long until" stops meaning anything
REACTION_MINUTES = 3

RELEASE_FEATURES = ([f"clock_minutes_from_{name}" for name in SLOT_MINUTES]
                    + [f"clock_in_reaction_{name}" for name in SLOT_MINUTES]
                    + ["clock_minute_of_session", "clock_business_day_of_month"])


def add_release_features(data: pd.DataFrame) -> pd.DataFrame:
    eastern = pd.to_datetime(data["time"]).dt.tz_localize("UTC").dt.tz_convert(EASTERN)
    minute_of_day = (eastern.dt.hour * 60 + eastern.dt.minute).to_numpy(float)
    columns = {}
    for name, slot in SLOT_MINUTES.items():
        offset = minute_of_day - slot
        columns[f"clock_minutes_from_{name}"] = np.clip(offset, -FAR_AWAY, FAR_AWAY)
        columns[f"clock_in_reaction_{name}"] = ((offset >= 0) & (offset < REACTION_MINUTES)).astype(float)
    columns["clock_minute_of_session"] = minute_of_day - 9 * 60 - 30
    day = pd.Series(data["session_day"].to_numpy())
    order = pd.DataFrame({"day": sorted(set(day))})
    order["position"] = order.groupby(pd.to_datetime(order["day"]).dt.to_period("M")).cumcount() + 1
    columns["clock_business_day_of_month"] = day.map(
        order.set_index("day")["position"]).to_numpy(float)
    return data.assign(**columns)
