"""Does holding overnight through a macro release differ from holding through a quiet night?

The overnight drift is the one thing in this project that has survived out-of-sample
testing: long at the 16:00 close, out at the 09:30 open. The macro releases print at 08:30,
which is inside that window -- a position held from the close sits through CPI, payrolls,
producer prices and GDP whether or not anyone chose to.

So the calendar's value is here rather than in the five-second model, where an 08:30
release is an hour old before the first bar forms. If release nights pay differently, the
rule has a free improvement available: skip them, or size them differently.

SPY's own open and close, so the result is about the market rather than about a futures
contract's roll. Dates are 2026 only, which is what the calendar covers.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/overnight_history")
from macro_calendar import RELEASES  # noqa: E402

DAILY = "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/overnight_history/SPY_daily.csv"
ES_PRICE = 7800.0
POINT_VALUE = 50.0
FIRST_COVERED, LAST_COVERED = "2026-01-01", "2026-09-30"
MIN_NIGHTS = 4


def describe(returns: pd.Series) -> dict:
    if len(returns) < MIN_NIGHTS:
        return {}
    dollars = returns * ES_PRICE * POINT_VALUE
    t = dollars.mean() / (dollars.std(ddof=1) / np.sqrt(len(dollars))) if len(dollars) > 2 else np.nan
    return {"nights": len(dollars), "up": float((returns > 0).mean()),
            "mean_dollars": float(dollars.mean()), "t": float(t)}


def main() -> None:
    spy = pd.read_csv(DAILY, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    spy["overnight"] = spy["open"].shift(-1) / spy["close"] - 1
    spy["next_day"] = spy["date"].shift(-1).dt.date
    covered = (spy["date"] >= FIRST_COVERED) & (spy["date"] <= LAST_COVERED) & spy["overnight"].notna()
    nights = spy[covered].copy()
    print(f"{len(nights)} nights from {nights['date'].min():%Y-%m-%d} to {nights['date'].max():%Y-%m-%d}; "
          f"the position is held from the close through the next 08:30\n")
    everything = describe(nights["overnight"])
    print(f"{'nights ending in':<26}{'nights':>8}{'up':>8}{'mean $':>10}{'t':>8}")
    print(f"{'any night':<26}{everything['nights']:>8}{everything['up']:>8.1%}"
          f"{everything['mean_dollars']:>+10.2f}{everything['t']:>8.2f}")
    results = {"all": everything}
    release_nights = set()
    for kind, days in RELEASES.items():
        if kind == "fomc":
            continue                                   # prints at 14:00, not inside the night
        chosen = nights[nights["next_day"].isin(set(days))]
        release_nights |= set(chosen["next_day"])
        outcome = describe(chosen["overnight"])
        results[kind] = outcome
        print(f"{'  a ' + kind.upper() + ' release':<26}"
              + (f"{outcome['nights']:>8}{outcome['up']:>8.1%}{outcome['mean_dollars']:>+10.2f}"
                 f"{outcome['t']:>8.2f}" if outcome else f"{'too few':>8}"))
    on_release = nights[nights["next_day"].isin(release_nights)]
    quiet = nights[~nights["next_day"].isin(release_nights)]
    for label, frame in (("any release", on_release), ("no release", quiet)):
        outcome = describe(frame["overnight"])
        results[label] = outcome
        print(f"{label:<26}{outcome['nights']:>8}{outcome['up']:>8.1%}"
              f"{outcome['mean_dollars']:>+10.2f}{outcome['t']:>8.2f}")
    gap = (on_release["overnight"].mean() - quiet["overnight"].mean()) * ES_PRICE * POINT_VALUE
    print(f"\nrelease nights pay {gap:+.2f} a night more than quiet ones")
    with open("overnight_release_study.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
