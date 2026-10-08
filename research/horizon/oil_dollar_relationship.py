"""Is there an oil or dollar relationship with ES at all, separate from whether six
particular columns improve a 163-column model?

The ablation that produced t -1.40 asks a narrow question: do `ms_uso_minus_mid_{3,6,12}`
and `ms_uup_minus_mid_{1,3,12}` add anything **given everything else already there**. A
real relationship can score nothing on that test if SPY's returns or the mid's own change
already carry the same macro factor. And those six columns encode one hypothesis --
contemporaneous divergence from the ES mid over 15 to 60 seconds -- while oil and currency
move on a far slower clock than five-second equity microstructure.

This measures the relationship itself, with nothing held constant:

  forward return     does the ETF's recent move predict where ES goes next (direction)
  forward range      does it predict how far ES travels (volatility) -- never once tested
                     for these symbols, since both ablations scored only AUC
  the reverse        does ES predict the ETF's next move, which says which one leads

Windows reach from 15 seconds to 30 minutes, because the existing columns stop at 60
seconds and that is a guess, not a finding.

**Significance is clustered by session.** A forward window measured on every 5-second bar
reuses the same period hundreds of times, which inflates an ordinary t-statistic by roughly
the square root of the window length -- this project once read a 30-minute event study at
t 8.3 that was really noise. Correlations are computed per session and the t-statistic is
taken across sessions, which treats a day as one observation.

SPY and GLD ride along as calibration: SPY is the strongest cross-asset relationship
measured here (~0.005), so it says what a real but useless effect looks like.

usage: oil_dollar_relationship.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from cross_asset_features import load_symbol_closes, lookback_return_pct
from mid_feature_support import build_data

BAR_SECONDS = 5
PERCENT = 100.0
SYMBOLS = ("USO", "UUP", "SPY", "GLD")
LOOKBACK_BARS = (3, 12, 60, 360)          # 15 s, 1 min, 5 min, 30 min
FORWARD_BARS = (12, 60, 360)
MIN_SESSION_ROWS = 500
MIN_SESSIONS = 20


def forward_return_pct(mid: np.ndarray, seconds: np.ndarray, bars: int) -> np.ndarray:
    """Percent change from each bar to `bars` ahead; blank across a session gap."""
    ahead = np.full(len(mid), np.nan)
    ahead[:-bars] = mid[bars:]
    contiguous = np.zeros(len(mid), bool)
    contiguous[:-bars] = (seconds[bars:] - seconds[:-bars]) == bars * BAR_SECONDS
    ahead[~contiguous] = np.nan
    return (ahead / mid - 1) * PERCENT


def session_clustered(left: np.ndarray, right: np.ndarray,
                      session_days: np.ndarray) -> tuple[float, float, int]:
    """Mean per-session correlation and its t across sessions.

    One day is one observation: overlapping forward windows make every bar within a day
    near-duplicates of each other, so a pooled t-statistic is meaningless here."""
    per_session = []
    for day in np.unique(session_days):
        rows = session_days == day
        pair = np.isfinite(left) & np.isfinite(right) & rows
        if pair.sum() < MIN_SESSION_ROWS:
            continue
        first, second = left[pair], right[pair]
        if first.std() == 0 or second.std() == 0:
            continue
        per_session.append(np.corrcoef(first, second)[0, 1])
    if len(per_session) < MIN_SESSIONS:
        return float("nan"), float("nan"), len(per_session)
    values = np.array(per_session)
    error = values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(values.mean() / error) if error > 0 else 0.0, len(values)


def main() -> None:
    started = time.time()
    data = build_data()
    epochs = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    session_days = data["session_day"].to_numpy()
    mid = data["mid"].to_numpy(float)
    forward = {bars: forward_return_pct(mid, epochs, bars) for bars in FORWARD_BARS}
    forward_size = {bars: np.abs(values) for bars, values in forward.items()}
    closes = {symbol: load_symbol_closes(symbol) for symbol in SYMBOLS}
    print(f"{len(data):,} regular-hours bars over {len(np.unique(session_days))} sessions "
          f"({time.time() - started:.0f}s to build)\n", flush=True)

    for symbol in SYMBOLS:
        print(f"{symbol}" + " " * 8 + f"{'ES forward return':>34}{'ES forward range':>28}")
        print(f"{'lookback':<12}{'ahead':>7}" + f"{'corr':>10}{'t':>8}{'days':>7}"
              + f"{'corr':>12}{'t':>8}{'days':>7}")
        for lookback in LOOKBACK_BARS:
            history = lookback_return_pct(closes[symbol], epochs, lookback)
            for ahead in FORWARD_BARS:
                direction, direction_t, days = session_clustered(history, forward[ahead], session_days)
                size, size_t, _ = session_clustered(history, forward_size[ahead], session_days)
                print(f"{f'{lookback * BAR_SECONDS}s':<12}{f'{ahead * BAR_SECONDS}s':>7}"
                      f"{direction:>10.4f}{direction_t:>8.1f}{days:>7}"
                      f"{size:>12.4f}{size_t:>8.1f}{days:>7}", flush=True)
        # Which one leads: the ETF's next move against the ES move just made.
        es_history = lookback_return_pct(pd.Series(mid, index=epochs), epochs, 12)
        etf_forward = forward_return_pct(closes[symbol].reindex(epochs).to_numpy(), epochs, 12)
        reverse, reverse_t, days = session_clustered(es_history, etf_forward, session_days)
        print(f"{'reverse: ES 60s -> ' + symbol + ' next 60s':<40}{reverse:>10.4f}{reverse_t:>8.1f}"
              f"{days:>7}\n", flush=True)


if __name__ == "__main__":
    main()
