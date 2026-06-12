"""Cross-instrument percent-change features for ES/SPX/SPY.

Every feature is anchored at the **previous** bar's close (``Close[t-1]``): the
n-bar percent change spans ``Close[t-1]`` back to ``Close[t-1-n]``. No input ever
uses the current bar ``t``, which is the bar the forward target is measured from.
That one-bar embargo removes the mechanical correlation that arises when a feature
and a target both depend on the noisy print ``Close[t]`` (bid-ask bounce).

From the per-instrument changes we derive:

* per-instrument changes (ES, SPX, SPY),
* pairwise differences of those changes (e.g. ES change minus SPX change), which
  capture short-horizon lead/lag divergence between the three instruments,
* an ES-volume-normalized variant of every feature, dividing by the *previous*
  bar's ES volume so a move on heavy volume is scaled differently from the same
  move on light volume (the divisor is also lagged, so it too excludes bar t).

All changes are computed within a single trading day (grouped by ``TradingDate``)
so a lag never reaches back across an overnight gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_alignment import (
    ES_VOLUME_COLUMN,
    INSTRUMENTS,
    TIMESTAMP_COLUMN,
    TRADING_DATE_COLUMN,
    close_column,
)

# Lookback lags in 5-second bars, from 5s up to 30 minutes, finer at the short end.
DEFAULT_LAGS = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360)

# The previous bar is the most recent input; bar t (the target anchor) is excluded.
_ANCHOR_OFFSET = 1
_VOLUME_NORMALIZED_SUFFIX = "_per_ESvol"

# Pairwise differences computed as (left change - right change).
_INSTRUMENT_PAIRS = (("ES", "SPX"), ("ES", "SPY"), ("SPY", "SPX"))


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one feature column so results can be grouped after correlating."""

    name: str
    family: str  # e.g. "ES" or "ES_minus_SPX"
    lag: int
    volume_normalized: bool


def build_features(
    bars: pd.DataFrame, lags: tuple[int, ...] = DEFAULT_LAGS
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Return aligned feature columns and the spec describing each of them.

    The returned frame carries ``Date`` and ``TradingDate`` for joining and
    grouping, plus one column per feature. Specs are returned separately so the
    correlation step can break results down by family, lag, and normalization.
    """
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError(f"lags must be non-empty and all >= 1, got {lags}")

    percent_changes = _per_instrument_percent_changes(bars, lags)
    raw_features = _assemble_raw_features(percent_changes, lags)

    inverse_previous_es_volume = _inverse_previous_es_volume(bars)
    specs: list[FeatureSpec] = []
    feature_columns: dict[str, pd.Series] = {}

    for raw_name, family, lag, series in raw_features:
        feature_columns[raw_name] = series
        specs.append(FeatureSpec(raw_name, family, lag, volume_normalized=False))

        normalized_name = f"{raw_name}{_VOLUME_NORMALIZED_SUFFIX}"
        feature_columns[normalized_name] = series * inverse_previous_es_volume
        specs.append(FeatureSpec(normalized_name, family, lag, volume_normalized=True))

    features = pd.DataFrame(feature_columns)
    features.insert(0, TRADING_DATE_COLUMN, bars[TRADING_DATE_COLUMN].to_numpy())
    features.insert(0, TIMESTAMP_COLUMN, bars[TIMESTAMP_COLUMN].to_numpy())
    return features, specs


def _per_instrument_percent_changes(
    bars: pd.DataFrame, lags: tuple[int, ...]
) -> dict[str, dict[int, pd.Series]]:
    """n-bar percent change per instrument, anchored at Close[t-1], reset each day.

    The change spans ``Close[t-1]`` back to ``Close[t-1-lag]``. All shifts are
    taken within a trading day, so no window reaches across an overnight gap.
    """
    percent_changes: dict[str, dict[int, pd.Series]] = {}

    for instrument in INSTRUMENTS:
        grouped_close = bars.groupby(TRADING_DATE_COLUMN, sort=False)[
            close_column(instrument)
        ]
        anchor_close = grouped_close.shift(_ANCHOR_OFFSET)
        percent_changes[instrument] = {
            lag: _percent_change(anchor_close, grouped_close.shift(_ANCHOR_OFFSET + lag))
            for lag in lags
        }
    return percent_changes


def _percent_change(recent_close: pd.Series, past_close: pd.Series) -> pd.Series:
    return (recent_close - past_close) / past_close * 100.0


def _assemble_raw_features(
    percent_changes: dict[str, dict[int, pd.Series]], lags: tuple[int, ...]
) -> list[tuple[str, str, int, pd.Series]]:
    """Flatten per-instrument changes and pairwise differences into (name, family, lag, series)."""
    raw_features: list[tuple[str, str, int, pd.Series]] = []

    for instrument in INSTRUMENTS:
        for lag in lags:
            raw_features.append(
                (f"{instrument}_pct_{lag}", instrument, lag, percent_changes[instrument][lag])
            )

    for left, right in _INSTRUMENT_PAIRS:
        family = f"{left}_minus_{right}"
        for lag in lags:
            difference = percent_changes[left][lag] - percent_changes[right][lag]
            raw_features.append((f"{family}_pct_{lag}", family, lag, difference))

    return raw_features


def _inverse_previous_es_volume(bars: pd.DataFrame) -> pd.Series:
    """1 / ES volume of the previous bar, with non-positive volume mapped to NaN.

    Using the previous bar keeps the divisor out of the current bar t, consistent
    with every other input.
    """
    previous_es_volume = bars.groupby(TRADING_DATE_COLUMN, sort=False)[
        ES_VOLUME_COLUMN
    ].shift(_ANCHOR_OFFSET)
    return 1.0 / previous_es_volume.where(previous_es_volume > 0)


if __name__ == "__main__":
    from data_alignment import find_common_trading_dates, load_aligned_bars

    sample_dates = find_common_trading_dates()[:2]
    bars = load_aligned_bars(trading_dates=sample_dates)
    features, specs = build_features(bars)

    print(f"Built {len(specs)} features over {len(features):,} rows")
    families = sorted({spec.family for spec in specs})
    print(f"Families: {families}")
    print(f"Lags: {sorted({spec.lag for spec in specs})}")
    print(f"Volume-normalized columns: {sum(spec.volume_normalized for spec in specs)}")

    # Embargo check: ES_pct_3 spans Close[t-1]..Close[t-4], so the first 4 rows of
    # each day are NaN and the rest are populated.
    first_day = features[features[TRADING_DATE_COLUMN] == sample_dates[0]]
    es_lag_3 = first_day["ES_pct_3"]
    print(f"\nES_pct_3 NaNs in first 4 rows of day 1: {es_lag_3.iloc[:4].isna().sum()} (expect 4)")
    print(f"ES_pct_3 NaNs after row 4: {es_lag_3.iloc[4:].isna().sum()} (expect 0)")
