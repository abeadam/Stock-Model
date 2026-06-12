"""Persist and visualize the feature/target correlation table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import pandas as pd

_NORMALIZATION_ROWS = (
    (False, "Raw"),
    (True, "Divided by ES volume"),
)


def save_correlation_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write the full tidy correlation table, strongest signals first."""
    ordered = table.sort_values("abs_correlation", ascending=False)
    ordered.to_csv(output_path, index=False)
    print(f"Saved {len(table):,} correlations to {output_path}")


def print_top_correlations(table: pd.DataFrame, top_n: int = 15) -> None:
    """Print the strongest correlations overall and per target."""
    columns = ["feature", "target", "correlation", "n_samples"]
    print(f"\n=== Top {top_n} correlations overall (by |corr|) ===")
    strongest = table.sort_values("abs_correlation", ascending=False).head(top_n)
    print(strongest[columns].to_string(index=False))

    print("\n=== Strongest feature for each target ===")
    best_per_target = table.loc[table.groupby("target")["abs_correlation"].idxmax()]
    print(best_per_target.sort_values("target")[columns].to_string(index=False))


def plot_correlation_vs_lag(table: pd.DataFrame, output_dir: Path) -> list[Path]:
    """One figure per target kind: signed correlation vs lag, per family.

    Rows split raw vs volume-normalized features; columns split forward horizons.
    """
    saved_paths = []
    for target_kind in sorted(table["target_kind"].unique()):
        kind_table = table[table["target_kind"] == target_kind]
        path = output_dir / f"correlation_vs_lag_{target_kind}.png"
        _plot_single_target_kind(kind_table, target_kind, path)
        saved_paths.append(path)
    return saved_paths


def _plot_single_target_kind(
    kind_table: pd.DataFrame, target_kind: str, output_path: Path
) -> None:
    horizons = sorted(kind_table["horizon"].unique())
    families = sorted(kind_table["family"].unique())

    n_rows, n_cols = len(_NORMALIZATION_ROWS), len(horizons)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), sharex=True, squeeze=False
    )

    for row_index, (is_normalized, row_label) in enumerate(_NORMALIZATION_ROWS):
        for col_index, horizon in enumerate(horizons):
            axis = axes[row_index][col_index]
            panel = kind_table[
                (kind_table["volume_normalized"] == is_normalized)
                & (kind_table["horizon"] == horizon)
            ]
            for family in families:
                family_curve = panel[panel["family"] == family].sort_values("lag")
                axis.plot(
                    family_curve["lag"],
                    family_curve["correlation"],
                    marker=".",
                    label=family,
                )
            axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            axis.set_xscale("log")
            axis.set_title(f"{row_label} — horizon {horizon} bars", fontsize=10)
            axis.grid(True, alpha=0.3, which="both")
            if row_index == n_rows - 1:
                axis.set_xlabel("Lookback lag n (bars, log scale)")
            if col_index == 0:
                axis.set_ylabel("Correlation with ES target")

    axes[0][-1].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Cross-instrument % change vs ES forward move ({target_kind} target)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
