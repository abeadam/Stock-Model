"""Run the cross-instrument percent-change correlation study end to end.

Loads aligned ES/SPX/SPY RTH bars in day-batches, builds the percent-change
features (all anchored at the previous bar, so the current bar is never an input)
and forward ES targets, accumulates pairwise correlations, and writes a tidy CSV
plus signed-correlation-vs-lag plots.

Usage:
    python run.py            # all common trading days
    python run.py 20         # first 20 trading days (quick check)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from correlation import PairwiseCorrelationAccumulator, build_correlation_table
from data_alignment import find_common_trading_dates, load_aligned_bars
from features import DEFAULT_LAGS, build_features
from reporting import (
    plot_correlation_vs_lag,
    print_top_correlations,
    save_correlation_table,
)
from targets import DEFAULT_HORIZONS, build_targets

_OUTPUT_DIR = Path(__file__).resolve().parent
_CORRELATION_CSV = _OUTPUT_DIR / "feature_target_correlations.csv"
_DAYS_PER_BATCH = 20


def run_correlation_study(
    trading_dates: list[str],
    lags: tuple[int, ...] = DEFAULT_LAGS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Accumulate correlations over all day-batches and return the tidy table."""
    accumulator: PairwiseCorrelationAccumulator | None = None
    feature_specs = target_specs = None

    for batch_start in range(0, len(trading_dates), _DAYS_PER_BATCH):
        batch_dates = trading_dates[batch_start : batch_start + _DAYS_PER_BATCH]
        bars = load_aligned_bars(trading_dates=batch_dates)
        features, feature_specs = build_features(bars, lags)
        targets, target_specs = build_targets(bars, horizons)

        if accumulator is None:
            accumulator = PairwiseCorrelationAccumulator(
                len(feature_specs), len(target_specs)
            )

        feature_matrix = features[[spec.name for spec in feature_specs]].to_numpy(float)
        target_matrix = targets[[spec.name for spec in target_specs]].to_numpy(float)
        accumulator.update(feature_matrix, target_matrix)
        print(
            f"Processed days {batch_start + 1}-{batch_start + len(batch_dates)} "
            f"of {len(trading_dates)} ({len(bars):,} bars)"
        )

    if accumulator is None:
        raise ValueError("No trading dates to process")

    correlation, counts = accumulator.correlation_and_counts()
    return build_correlation_table(correlation, counts, feature_specs, target_specs)


def main() -> None:
    trading_dates = find_common_trading_dates()
    if len(sys.argv) > 1:
        trading_dates = trading_dates[: int(sys.argv[1])]
    print(f"Running correlation study over {len(trading_dates)} trading days")

    table = run_correlation_study(trading_dates)
    save_correlation_table(table, _CORRELATION_CSV)
    print_top_correlations(table)
    plot_correlation_vs_lag(table, _OUTPUT_DIR)


if __name__ == "__main__":
    main()
