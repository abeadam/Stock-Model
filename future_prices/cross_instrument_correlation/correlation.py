"""Streaming pairwise correlation between feature and target matrices.

Correlations are accumulated from sufficient statistics (counts, sums, sums of
squares, and cross-products) so the full feature matrix never has to live in
memory at once: each day-batch updates a handful of small ``n_features x
n_targets`` matrices that combine additively. Every pair uses only rows where
both the feature and the target are present (pairwise-complete observations),
which matters because different lags and horizons leave NaNs in different places.

For a binary 0/1 target the Pearson correlation computed here is exactly the
point-biserial correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import FeatureSpec
from targets import TargetSpec


class PairwiseCorrelationAccumulator:
    """Accumulates sufficient statistics for every (feature, target) pair."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        shape = (n_features, n_targets)
        self._count = np.zeros(shape)
        self._sum_feature = np.zeros(shape)
        self._sum_target = np.zeros(shape)
        self._sum_feature_sq = np.zeros(shape)
        self._sum_target_sq = np.zeros(shape)
        self._sum_product = np.zeros(shape)

    def update(self, feature_values: np.ndarray, target_values: np.ndarray) -> None:
        """Fold one batch of aligned rows into the running statistics."""
        feature_present = np.isfinite(feature_values)
        target_present = np.isfinite(target_values)

        features_zeroed = np.where(feature_present, feature_values, 0.0)
        targets_zeroed = np.where(target_present, target_values, 0.0)
        feature_present = feature_present.astype(np.float64)
        target_present = target_present.astype(np.float64)

        self._count += feature_present.T @ target_present
        self._sum_feature += features_zeroed.T @ target_present
        self._sum_target += feature_present.T @ targets_zeroed
        self._sum_feature_sq += (features_zeroed**2).T @ target_present
        self._sum_target_sq += feature_present.T @ (targets_zeroed**2)
        self._sum_product += features_zeroed.T @ targets_zeroed

    def correlation_and_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (correlation matrix, sample-count matrix), both n_features x n_targets."""
        count = self._count
        with np.errstate(invalid="ignore", divide="ignore"):
            covariance = count * self._sum_product - self._sum_feature * self._sum_target
            feature_variance = count * self._sum_feature_sq - self._sum_feature**2
            target_variance = count * self._sum_target_sq - self._sum_target**2
            denominator = np.sqrt(feature_variance * target_variance)
            correlation = np.where(denominator > 0, covariance / denominator, np.nan)
        return correlation, count


def build_correlation_table(
    correlation: np.ndarray,
    counts: np.ndarray,
    feature_specs: list[FeatureSpec],
    target_specs: list[TargetSpec],
) -> pd.DataFrame:
    """Flatten the correlation/count matrices into one tidy row per pair."""
    records = []
    for feature_index, feature_spec in enumerate(feature_specs):
        for target_index, target_spec in enumerate(target_specs):
            records.append(
                {
                    "feature": feature_spec.name,
                    "family": feature_spec.family,
                    "lag": feature_spec.lag,
                    "volume_normalized": feature_spec.volume_normalized,
                    "target": target_spec.name,
                    "horizon": target_spec.horizon,
                    "target_kind": target_spec.kind,
                    "correlation": correlation[feature_index, target_index],
                    "abs_correlation": abs(correlation[feature_index, target_index]),
                    "n_samples": int(counts[feature_index, target_index]),
                }
            )
    return pd.DataFrame.from_records(records)
