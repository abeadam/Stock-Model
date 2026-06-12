"""Forward ES targets to correlate features against.

For each horizon ``H`` (in 5-second bars) two targets are built from the ES
close, within a single trading day so the forward window never spills past the
session close:

* signed forward return: ``(ES_Close[t+H] - ES_Close[t]) / ES_Close[t] * 100``,
* binary direction: 1 if that return is positive, 0 if negative, NaN if exactly
  flat (a flat bar carries no up/down information).

The last ``H`` rows of every day are NaN because no forward bar exists for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_alignment import TIMESTAMP_COLUMN, TRADING_DATE_COLUMN, close_column

# Forward horizons in 5-second bars: 1 min, then 5 to 30 minutes.
DEFAULT_HORIZONS = (12, 60, 120, 180, 240, 360)


@dataclass(frozen=True)
class TargetSpec:
    """Describes one target column (forward direction and value form)."""

    name: str
    horizon: int
    kind: str  # "signed" or "binary"


def build_targets(
    bars: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> tuple[pd.DataFrame, list[TargetSpec]]:
    """Return forward ES targets (signed + binary per horizon) and their specs."""
    if any(horizon < 1 for horizon in horizons):
        raise ValueError(f"horizons must all be >= 1, got {horizons}")

    es_close_by_day = bars.groupby(TRADING_DATE_COLUMN, sort=False)[close_column("ES")]

    target_columns: dict[str, pd.Series] = {}
    specs: list[TargetSpec] = []

    for horizon in horizons:
        forward_close = es_close_by_day.shift(-horizon)
        signed_return = (forward_close - bars[close_column("ES")]) / bars[
            close_column("ES")
        ] * 100.0

        signed_name = f"ES_fwd_ret_{horizon}"
        target_columns[signed_name] = signed_return
        specs.append(TargetSpec(signed_name, horizon, kind="signed"))

        binary_name = f"ES_fwd_up_{horizon}"
        target_columns[binary_name] = _direction_from_return(signed_return)
        specs.append(TargetSpec(binary_name, horizon, kind="binary"))

    targets = pd.DataFrame(target_columns)
    targets.insert(0, TRADING_DATE_COLUMN, bars[TRADING_DATE_COLUMN].to_numpy())
    targets.insert(0, TIMESTAMP_COLUMN, bars[TIMESTAMP_COLUMN].to_numpy())
    return targets, specs


def _direction_from_return(signed_return: pd.Series) -> pd.Series:
    """1 for a positive return, 0 for negative, NaN for flat or missing."""
    direction = pd.Series(np.nan, index=signed_return.index)
    direction[signed_return > 0] = 1.0
    direction[signed_return < 0] = 0.0
    return direction


if __name__ == "__main__":
    from data_alignment import find_common_trading_dates, load_aligned_bars

    sample_dates = find_common_trading_dates()[:2]
    bars = load_aligned_bars(trading_dates=sample_dates)
    targets, specs = build_targets(bars)

    print(f"Built {len(specs)} targets over {len(targets):,} rows")
    for spec in specs:
        column = targets[spec.name]
        if spec.kind == "binary":
            up_rate = column.mean()
            print(f"  {spec.name:16s} up-rate={up_rate:.3f}  non-NaN={column.notna().sum():,}")
        else:
            print(f"  {spec.name:16s} mean={column.mean():+.5f}%  non-NaN={column.notna().sum():,}")

    # Leakage check: the last `horizon` rows of each day must be NaN.
    first_day = targets[targets[TRADING_DATE_COLUMN] == sample_dates[0]]
    tail_nans = first_day["ES_fwd_ret_60"].iloc[-60:].isna().sum()
    print(f"\nES_fwd_ret_60 NaNs in last 60 rows of day 1: {tail_nans} (expect 60)")
