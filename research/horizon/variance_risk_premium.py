"""What options are charging for volatility, against what the index is actually doing.

Every volatility column in the model looks one way. `vol_ratio` is realized-fast over
realized-slow, both backward. `ms_mid_change_z_n` is the right shape -- a move divided by
an expected move -- but its denominator is a 60-bar realized vol. `vxm_es_ratio` is
`vxm_close / close`, VIX points over index points, which is not a volatility comparison
whatever its name suggests. The rest of the VIX family are levels, returns and lags, and
the returns have already been tested and rejected.

So the one genuinely untried thing is the **variance risk premium**: implied volatility
against realized volatility, as a level. The VIX index is an annualized implied vol, so it
converts to an expected move per bar; the mid's own recent moves give the realized one.
Rich implied against quiet realized is a different statement from "VIX went up", which is
what the rejected columns measured.

Two constructions, because they answer slightly different questions and the horizon
cancels in one of them:

  vrp_ratio_<window>        implied expected move / realized expected move. The sqrt(h)
                            scaling cancels, so this is horizon-free.
  vrp_difference_<window>   implied minus realized, per bar, which keeps the scale.

**The seasonality is the trap.** Realized volatility has a hard intraday U-shape, so a raw
ratio would largely encode time of day -- something `hour_sin`, `hour_cos` and the session
features already carry, and a good way to manufacture a result that is really a clock. Each
construction is therefore also offered divided by its own **trailing** profile: the median
at that point of the session over the previous sessions, which looks only backwards, is
what could actually be computed live, and needs no per-fold fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deep_ladder_features import session_row_blocks
from realized_volatility_features import realized_volatility

BAR_SECONDS = 5
PERCENT = 100.0
# 252 sessions of six and a half hours: the clock implied vol is quoted against.
TRADING_SECONDS_PER_YEAR = 252 * 6.5 * 3600
REALIZED_WINDOWS = (60, 180)              # five and fifteen minutes of 5-second bars
SEASONAL_SESSIONS = 20
SEASONAL_BUCKET_BARS = 60                 # five-minute buckets of the session
MIN_SEASONAL_SESSIONS = 5
EPSILON = 1e-12

VARIANCE_PREMIUM_FEATURES = (["vrp_implied_per_bar"]
                             + [f"vrp_realized_per_bar_{window}" for window in REALIZED_WINDOWS]
                             + [f"vrp_{measure}_{window}" for window in REALIZED_WINDOWS
                                for measure in ("ratio", "difference",
                                                "ratio_deseasonal", "difference_deseasonal")])


def implied_per_bar_move(vix_level: np.ndarray) -> np.ndarray:
    """A VIX reading, as the fractional move expected over one 5-second bar."""
    return vix_level / PERCENT * np.sqrt(BAR_SECONDS / TRADING_SECONDS_PER_YEAR)


def realized_per_bar_move(mid: np.ndarray, session_days: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation of one-bar fractional mid moves, measured inside each session."""
    return realized_volatility(mid, session_days, window)


def position_in_session(session_days: np.ndarray) -> np.ndarray:
    """Each row's index within its own session, for a frame ordered by time."""
    starts = np.flatnonzero(np.diff(session_days)) + 1
    position = np.arange(len(session_days))
    return position - np.repeat(np.concatenate([[0], starts]), np.diff(
        np.concatenate([[0], starts, [len(session_days)]])))


def trailing_seasonal_median(values: np.ndarray, session_days: np.ndarray, buckets: np.ndarray,
                             sessions: int = SEASONAL_SESSIONS) -> np.ndarray:
    """The median at each point of the session over the PREVIOUS `sessions` sessions.

    Causal: a row's profile never includes its own day. Blank until enough days exist."""
    frame = pd.DataFrame({"day": session_days, "bucket": buckets, "value": values})
    per_day = frame.groupby(["day", "bucket"], sort=True)["value"].mean().unstack("bucket")
    profile = per_day.rolling(sessions, min_periods=MIN_SEASONAL_SESSIONS).median().shift(1)
    stacked = profile.stack(future_stack=True).rename("profile").reset_index()
    return frame.merge(stacked, on=["day", "bucket"], how="left")["profile"].to_numpy()


def add_variance_premium_features(data: pd.DataFrame, vix_closes: pd.Series) -> pd.DataFrame:
    """Adds VARIANCE_PREMIUM_FEATURES to ES rows holding `time`, `mid` and `session_day`."""
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    session_days = data["session_day"].to_numpy()
    buckets = position_in_session(session_days) // SEASONAL_BUCKET_BARS
    implied = implied_per_bar_move(vix_closes.reindex(epochs).to_numpy())
    columns = {"vrp_implied_per_bar": implied}
    for window in REALIZED_WINDOWS:
        realized = realized_per_bar_move(data["mid"].to_numpy(float), session_days, window)
        ratio = implied / np.where(realized > EPSILON, realized, np.nan)
        difference = implied - realized
        columns[f"vrp_realized_per_bar_{window}"] = realized
        columns[f"vrp_ratio_{window}"] = ratio
        columns[f"vrp_difference_{window}"] = difference
        # A ratio is detrended by dividing out its profile; a difference by subtracting it.
        ratio_profile = trailing_seasonal_median(ratio, session_days, buckets)
        columns[f"vrp_ratio_deseasonal_{window}"] = ratio / np.where(
            np.abs(ratio_profile) > EPSILON, ratio_profile, np.nan)
        columns[f"vrp_difference_deseasonal_{window}"] = (
            difference - trailing_seasonal_median(difference, session_days, buckets))
    return data.assign(**columns)
