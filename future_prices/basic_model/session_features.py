"""Where the current bar sits within today's ES session.

Trend days play out over hours, but every other model input looks back at most
50 five-second bars (about four minutes), so nothing else can tell a day that
has trended since the open from one chopping around it. These six features
describe the session so far:

  session_vwap_dist            (close - session VWAP) / close
  session_vwap_above_frac      share of the session's bars that closed above VWAP
  session_return_pct           % change from the session's first open
  session_move_vs_prior_range  (close - session open) / average high-low range
                               of the prior PRIOR_SESSIONS_FOR_RANGE complete sessions
  session_range_position       where close sits in the session's high-low range, 0..1
  session_efficiency_ratio     |net move| / path length on 1-minute closes, 0..1

A session is one CME Globex trading day: 18:00 ET to 17:00 ET the next day.
Every value uses only bars from its session's start through the current bar,
never later ones, so there is no lookahead.

Bars must be sorted by time. A session whose first bar arrives more than
SESSION_START_TOLERANCE after 18:00 ET was not observed from its open: the first
session in a dataset, a data gap, or a live buffer seeded mid-session. All its
features are NaN, and later sessions skip it when they look back at prior
ranges.

Prior ranges come from a history table built by session_ranges(). Training
builds it from the same bars it computes features on; live code reads the
5-second daily_data files through session_history.py, so both see identical
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_clock import OVERNIGHT_SESSION_START_HOUR

SESSION_FEATURE_NAMES: tuple[str, ...] = (
    "session_vwap_dist",
    "session_vwap_above_frac",
    "session_return_pct",
    "session_move_vs_prior_range",
    "session_range_position",
    "session_efficiency_ratio",
)

# How late a session's first bar may be and still count as observed from its open.
SESSION_START_TOLERANCE = pd.Timedelta(minutes=5)
PRIOR_SESSIONS_FOR_RANGE = 20
# The efficiency ratio walks 1-minute closes: 5-second bars add so much bid/ask
# bounce to the path that the ratio sits near zero.
EFFICIENCY_SAMPLING = "1min"
PERCENT = 100.0

# ES trades 16:00-16:59 ET, after the stock close, but daily_data files older
# than IBKR's 5-second retention lack that hour. Prior-session ranges leave it
# out everywhere, so old and new sessions are measured alike. The hour set a new
# session high or low in 18% of sessions and moved a 20-session average range by
# a median 0.74% (measured on 74 sessions, June-September 2026).
PRIOR_RANGE_EXCLUDES_LAST_FUTURES_HOUR = True
LAST_FUTURES_HOUR = 16

_SESSION_START_OFFSET = pd.Timedelta(hours=OVERNIGHT_SESSION_START_HOUR)


def session_start_times(eastern_times: pd.Series) -> pd.Series:
    """18:00 ET on the evening each bar's session opened (e.g. Mon 10:00 -> Sun 18:00)."""
    return (eastern_times - _SESSION_START_OFFSET).dt.normalize() + _SESSION_START_OFFSET


def session_ranges(eastern_times: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """The high-low range of every session observed from its open, indexed by its start.

    This is the history other sessions look back on. Inputs must share one index.
    """
    session = session_start_times(eastern_times)
    first_bar = eastern_times.groupby(session).min()
    observed = first_bar.index[(first_bar - first_bar.index) <= SESSION_START_TOLERANCE]

    counted = pd.Series(True, index=eastern_times.index)
    if PRIOR_RANGE_EXCLUDES_LAST_FUTURES_HOUR:
        counted = eastern_times.dt.hour != LAST_FUTURES_HOUR
    ranges = (high[counted].groupby(session[counted]).max()
              - low[counted].groupby(session[counted]).min())
    return ranges.reindex(observed).dropna().sort_index().rename("range")


def average_prior_range(session_starts: pd.Index, history: pd.Series) -> pd.Series:
    """For each session, the mean of the PRIOR_SESSIONS_FOR_RANGE most recent history
    ranges from sessions that started before it; NaN when there are fewer."""
    history = history.sort_index()
    rolling_mean = history.rolling(PRIOR_SESSIONS_FOR_RANGE,
                                   min_periods=PRIOR_SESSIONS_FOR_RANGE).mean().to_numpy()
    # Position of the latest history session strictly before each session.
    latest_before = history.index.searchsorted(session_starts, side="left") - 1
    averages = np.full(len(session_starts), np.nan)
    has_history = latest_before >= 0
    averages[has_history] = rolling_mean[latest_before[has_history]]
    return pd.Series(averages, index=session_starts)


def compute_session_features(
    eastern_times: pd.Series,
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    history: pd.Series,
) -> dict[str, np.ndarray]:
    """The six session features for every bar. Inputs must share one index;
    `history` is a session_ranges() table."""
    session = session_start_times(eastern_times)
    session_open = _within(open_price, session).transform("first")
    prior_range = average_prior_range(pd.Index(session.unique()), history)

    vwap = _session_vwap(high, low, close, volume, session)
    features = {
        "session_vwap_dist": (close - vwap) / close,
        "session_vwap_above_frac": _share_of_bars_above(close, vwap, session),
        "session_return_pct": (close - session_open) / session_open * PERCENT,
        "session_move_vs_prior_range": (close - session_open) / session.map(prior_range),
        "session_range_position": _range_position(high, low, close, session),
        "session_efficiency_ratio": _efficiency_ratio(eastern_times, close, session),
    }

    observed = _observed_from_open(eastern_times, session)
    return {name: values.where(observed).to_numpy(dtype=np.float64)
            for name, values in features.items()}


def session_features_for_bars(eastern_times: pd.Series, bars: pd.DataFrame,
                              history: pd.Series) -> dict[str, np.ndarray]:
    """compute_session_features over a frame of Open/High/Low/Close/Volume bars.

    The single entry point for training (futures_price.py) and both live
    consumers (es_features.py, gradient_features.py), so none can diverge.
    """
    def column(name: str) -> pd.Series:
        return bars[name].astype(np.float64).reset_index(drop=True)
    return compute_session_features(eastern_times.reset_index(drop=True), column("Open"),
                                    column("High"), column("Low"), column("Close"),
                                    column("Volume"), history)


def _within(series: pd.Series, session: pd.Series):
    """Group by session in time order; sessions are contiguous in sorted bars."""
    return series.groupby(session, sort=False)


def _session_vwap(high, low, close, volume, session) -> pd.Series:
    typical_price = (high + low + close) / 3
    traded_value = _within(typical_price * volume, session).cumsum()
    traded_volume = _within(volume, session).cumsum()
    return traded_value / traded_volume.where(traded_volume > 0)


def _share_of_bars_above(close, vwap, session) -> pd.Series:
    above = (close > vwap).astype(np.float64).where(vwap.notna())
    bars_counted = _within(above.notna().astype(np.float64), session).cumsum()
    return _within(above.fillna(0.0), session).cumsum() / bars_counted.where(bars_counted > 0)


def _range_position(high, low, close, session) -> pd.Series:
    high_so_far = _within(high, session).cummax()
    low_so_far = _within(low, session).cummin()
    session_range = high_so_far - low_so_far
    return (close - low_so_far) / session_range.where(session_range > 0)


def _efficiency_ratio(eastern_times, close, session) -> pd.Series:
    """|close - the session's first close| over the path length, where the path
    runs through the last close of each completed minute and then to the current
    close. A bar never uses its own minute's last close, which is in its future."""
    minute = eastern_times.dt.floor(EFFICIENCY_SAMPLING)
    first_close = _within(close, session).transform("first")

    # One row per (session, minute): its last close and the close it moved from.
    minutes = (pd.DataFrame({"session": session, "minute": minute, "close": close,
                             "first_close": first_close})
               .groupby(["session", "minute"], sort=False)
               .agg(minute_close=("close", "last"), first_close=("first_close", "first")))
    minutes["previous_close"] = (minutes.groupby(level="session")["minute_close"].shift(1)
                                 .fillna(minutes["first_close"]))
    step = (minutes["minute_close"] - minutes["previous_close"]).abs()
    minutes["path_before"] = step.groupby(level="session").cumsum() - step

    per_bar = minutes.reindex(pd.MultiIndex.from_arrays([session, minute]))
    path = pd.Series(per_bar["path_before"].to_numpy()
                     + np.abs(close.to_numpy() - per_bar["previous_close"].to_numpy()),
                     index=close.index)
    return (close - first_close).abs() / path.where(path > 0)


def _observed_from_open(eastern_times, session) -> pd.Series:
    first_bar = _within(eastern_times, session).transform("first")
    return (first_bar - session) <= SESSION_START_TOLERANCE
