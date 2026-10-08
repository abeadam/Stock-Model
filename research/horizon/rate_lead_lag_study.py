"""Do Treasury futures lead ES at five seconds, or follow it?

The daily study said expectations move with the market rather than ahead of it. That
cannot rule out a lead-lag living in seconds, which is the question this answers, and it
answers it symmetrically: whether the rate future's recent move says anything about ES's
next move, **and** whether ES's recent move says anything about the rate future's next
one. Only the asymmetry between those two is evidence of anything leading anything.

For reference the same measurement is run on SPY, where the honest correlation against the
ES mid is -0.088 -- the strongest cross-asset number in this project. A rate future worth
adding should be somewhere in that neighbourhood.

Correlations are clustered by session: overlapping five-bar windows on five-second bars
would otherwise turn any number at all into certainty.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cross_asset_features import load_symbol_closes, lookback_return_pct
from mid_feature_support import build_data
from mid_microstructure import PERCENT, trailing_change
from mid_versus_trade_target import average_target

FORWARD_BARS = 5
LAGS = (1, 2, 3, 6, 12)
# ZN's September contract went off the board before this ran, so the two-year
# carries the test -- which is the better instrument for policy expectations anyway.
SYMBOLS = ("ZT", "SPY")
MIN_DAYS = 10


def clustered_correlation(first: np.ndarray, second: np.ndarray, day: np.ndarray) -> tuple[float, float, int]:
    """Correlation overall, with a t-statistic built from one value per session."""
    frame = pd.DataFrame({"a": first, "b": second, "day": day}).dropna()
    if frame["day"].nunique() < MIN_DAYS:
        return np.nan, np.nan, frame["day"].nunique()
    overall = frame["a"].corr(frame["b"])
    daily = frame.groupby("day").apply(lambda rows: rows["a"].corr(rows["b"]), include_groups=False).dropna()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 2 else np.nan
    return float(overall), float(t), len(daily)


def main() -> None:
    data = build_data()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    mid = data["mid"].to_numpy(float)
    day = data["session_day"].to_numpy()
    es_forward = average_target(mid, seconds, FORWARD_BARS)
    results = {}
    print(f"{'symbol':<7}{'lag':>5}{'its move -> ES next':>22}{'t':>8}"
          f"{'ES move -> its next':>22}{'t':>8}{'days':>6}")
    for symbol in SYMBOLS:
        closes = load_symbol_closes(symbol)
        if closes.empty:
            print(f"{symbol:<7}no files on disk")
            continue
        # A join that matches nothing produces all-NaN columns and would otherwise be
        # reported as "0 days" -- indistinguishable from a real absence of overlap. The
        # first time this happened the cause was a timestamp written as a formatted
        # string where the loader keys on epoch seconds, so it is worth saying out loud.
        matched = closes.reindex(seconds).notna().sum()
        if matched == 0:
            overlap = set(closes.index) & set(seconds.tolist())
            raise SystemExit(
                f"{symbol}: {len(closes):,} rows loaded but none align with the ES bars "
                f"(index dtype {closes.index.dtype}, {len(overlap)} shared timestamps). "
                f"Check the Date column is epoch seconds.")
        print(f"{symbol:<7}{matched:,} of {len(seconds):,} ES bars have a quote", flush=True)
        its_forward = np.full(len(mid), np.nan)
        now = closes.reindex(seconds).to_numpy()
        ahead = closes.reindex(seconds + FORWARD_BARS * 5).to_numpy()
        its_forward = (ahead / now - 1) * PERCENT
        for lag in LAGS:
            its_past = lookback_return_pct(closes, seconds, lag)
            es_past = trailing_change(mid, seconds, lag) / mid * PERCENT
            leads, leads_t, days = clustered_correlation(its_past, es_forward, day)
            follows, follows_t, _ = clustered_correlation(es_past, its_forward, day)
            results[f"{symbol}|{lag}"] = {"leads_es": leads, "leads_t": leads_t,
                                          "follows_es": follows, "follows_t": follows_t, "days": days}
            print(f"{symbol:<7}{lag:>5}{leads:>22.4f}{leads_t:>8.1f}{follows:>22.4f}"
                  f"{follows_t:>8.1f}{days:>6}")
    with open("rate_lead_lag_study.json", "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
