"""Is the average-of-next-N-bars target stationary, in regular trading hours?

The target at bar t is the percent change from the current close to the mean close of
the next N bars. Consecutive values share N-1 of their bars, so the series carries a
moving-average structure by construction: the ADF regression needs enough of its own
lagged changes to absorb that, which is why each horizon is also run with lags scaled to
it rather than the default five.

Regular hours leave a gap every evening. Pooling across it would let the test read a
jump between 16:00 and the next 09:30 as part of the dynamics, so the honest reading is
per session: each day tested on its own, and the distribution of statistics reported.
The pooled figure is printed too, for what it is worth.

usage: adf_average_targets.py
"""
from __future__ import annotations

import json

import numpy as np

from forward_extremes import forward_target
from segment_walk_forward import load_cut
from stationarity import CRITICAL_VALUES, augmented_dickey_fuller, has_unit_root
from volume_scaled_target import regular_hours

HORIZONS = (5, 10, 15)
DEFAULT_LAGS = 5
MIN_SESSION_ROWS = 500


def session_statistics(target, session_day, lags) -> np.ndarray:
    statistics = []
    for day in np.unique(session_day):
        rows = session_day == day
        values = target[rows]
        values = values[np.isfinite(values)]
        if len(values) >= MIN_SESSION_ROWS:
            statistic = augmented_dickey_fuller(values, lags)
            if np.isfinite(statistic):
                statistics.append(statistic)
    return np.array(statistics)


def main() -> None:
    data = load_cut()
    data = data[regular_hours(data["time"])].reset_index(drop=True)
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    prices = {"seconds": seconds, "close": data["Close"].to_numpy(float),
              "high": data["High"].to_numpy(float), "low": data["Low"].to_numpy(float)}
    session_day = data["session_day"].to_numpy()
    print(f"regular hours only: {len(data):,} bars over {len(np.unique(session_day))} sessions")
    print(f"5% critical value {CRITICAL_VALUES['5%']}; more negative means more clearly stationary\n")
    print(f"{'target':<22}{'lags':>6}{'pooled':>12}{'per-session median':>21}{'worst':>10}"
          f"{'sessions with a unit root':>28}")
    results = {}
    for horizon in HORIZONS:
        target = forward_target(prices, horizon, "average")
        for lags in (DEFAULT_LAGS, 2 * horizon):
            pooled = augmented_dickey_fuller(target[np.isfinite(target)], lags)
            per_session = session_statistics(target, session_day, lags)
            failures = int(sum(has_unit_root(value) for value in per_session))
            results[f"average_{horizon}_lags{lags}"] = {
                "pooled": pooled, "median": float(np.median(per_session)),
                "worst": float(per_session.max()), "sessions": len(per_session), "unit_roots": failures}
            print(f"{f'average of next {horizon}':<22}{lags:>6}{pooled:>12.1f}{np.median(per_session):>21.1f}"
                  f"{per_session.max():>10.1f}{f'{failures} of {len(per_session)}':>28}", flush=True)
    # The endpoint target, for comparison: no overlap beyond the horizon itself.
    for horizon in HORIZONS:
        endpoint = np.full(len(prices["close"]), np.nan)
        moves = forward_target(prices, horizon, "average") * 0  # shape only
        close = prices["close"]
        endpoint[:-horizon] = (close[horizon:] / close[:-horizon] - 1) * 100
        gap = (seconds[horizon:] - seconds[:-horizon]) != horizon * 5
        endpoint[:-horizon][gap] = np.nan
        per_session = session_statistics(endpoint, session_day, 2 * horizon)
        print(f"{f'endpoint at {horizon}':<22}{2 * horizon:>6}{'':>12}{np.median(per_session):>21.1f}"
              f"{per_session.max():>10.1f}"
              f"{f'{int(sum(has_unit_root(v) for v in per_session))} of {len(per_session)}':>28}")
    with open("adf_average_targets.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
