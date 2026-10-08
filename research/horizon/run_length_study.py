"""How long has this move been running, and does that say anything about what comes next?

A move that has extended further than such moves usually do is a candidate for an
over-reaction. The way to find out is not to assume it but to count: group every bar by
how many consecutive bars the mid has been moving the same way, and by how far it has
carried relative to what the market has been doing lately, then look at what happened
afterwards.

Mid prices throughout, so a run cannot be manufactured by the last trade alternating
between bid and ask - which is exactly what would happen on trade prices and would make
this whole exercise measure the spread instead of the market.

Two groupings:
  run length     consecutive same-direction bars ending at this one
  extension      how far the run has carried, in units of the last five minutes' typical
                 bar move, so "a long way" means a long way for current conditions

Read against the unconditional row. Significance is clustered by session, because runs
cluster and overlapping forward windows otherwise invent certainty.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mid_feature_support import build_data
from mid_microstructure import TICK, mid_volatility, trailing_change
from mid_versus_trade_target import average_target

FORWARD_BARS = 5
PERCENT = 100.0
POINT_VALUE = 50.0
RUN_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 12)
EXTENSION_BUCKETS = (0.5, 1.0, 1.5, 2.0, 3.0)
MIN_BARS = 500


def run_lengths(change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(length, direction) of the same-signed streak ending at each bar; flat bars end a run."""
    sign = np.sign(change)
    length = np.zeros(len(sign), dtype=int)
    for index in range(1, len(sign)):
        if sign[index] != 0 and sign[index] == sign[index - 1]:
            length[index] = length[index - 1] + 1
        elif sign[index] != 0:
            length[index] = 1
    return length, sign


def summarise(forward: np.ndarray, direction: np.ndarray, day: np.ndarray, mask: np.ndarray) -> dict:
    """Continuation seen from the run's own side: positive means the move kept going."""
    if mask.sum() < MIN_BARS:
        return {}
    aligned = forward[mask] * direction[mask]
    daily = pd.DataFrame({"value": aligned, "day": day[mask]}).groupby("day")["value"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 2 else np.nan
    return {"bars": int(mask.sum()), "continues": float((aligned > 0).mean()),
            "mean_dollars": float(aligned.mean() * POINT_VALUE), "t": float(t)}


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    day = data["session_day"].to_numpy()
    change = trailing_change(mid, seconds, 1)
    length, sign = run_lengths(np.nan_to_num(change))
    volatility = mid_volatility(mid, seconds) * TICK
    forward = average_target(mid, seconds, FORWARD_BARS) / PERCENT * mid
    usable = np.isfinite(forward) & np.isfinite(change) & (sign != 0)

    results = {}
    print(f"{'group':<34}{'bars':>9}{'continues':>11}{'mean $':>10}{'t':>8}")
    everything = summarise(forward, sign, day, usable)
    print(f"{'any bar that moved':<34}{everything['bars']:>9,}{everything['continues']:>10.1%}"
          f"{everything['mean_dollars']:>+10.2f}{everything['t']:>8.2f}")
    results["unconditional"] = everything

    print("\nby how many bars the move has already run:")
    for bucket in RUN_BUCKETS:
        mask = usable & (length == bucket) if bucket < max(RUN_BUCKETS) else usable & (length >= bucket)
        outcome = summarise(forward, sign, day, mask)
        label = f"   {bucket} bars" + (" or more" if bucket == max(RUN_BUCKETS) else "")
        results[f"run_{bucket}"] = outcome
        print(f"{label:<34}" + (f"{outcome['bars']:>9,}{outcome['continues']:>10.1%}"
                                f"{outcome['mean_dollars']:>+10.2f}{outcome['t']:>8.2f}"
                                if outcome else f"{'too few':>9}"))

    # How far the run has carried, against what a bar has typically been worth lately.
    carried = np.abs(trailing_change(mid, seconds, 1))
    for bars in (3, 6, 12):
        carried = np.maximum(carried, np.abs(trailing_change(mid, seconds, bars)))
    extension = np.divide(carried, volatility, out=np.full(len(mid), np.nan),
                          where=np.isfinite(volatility) & (volatility > 0))
    print("\nby how far it has carried, in units of a typical recent bar:")
    edges = list(EXTENSION_BUCKETS) + [np.inf]
    for low, high in zip([0.0] + list(EXTENSION_BUCKETS), edges):
        mask = usable & np.isfinite(extension) & (extension >= low) & (extension < high)
        outcome = summarise(forward, sign, day, mask)
        label = f"   {low:.1f} to {high:.1f}" if np.isfinite(high) else f"   over {low:.1f}"
        results[f"extension_{low}"] = outcome
        print(f"{label:<34}" + (f"{outcome['bars']:>9,}{outcome['continues']:>10.1%}"
                                f"{outcome['mean_dollars']:>+10.2f}{outcome['t']:>8.2f}"
                                if outcome else f"{'too few':>9}"))
    with open("run_length_study.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
