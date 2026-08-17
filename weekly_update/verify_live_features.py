#!/usr/bin/env python3
"""
Verify the live feature generator can produce every feature the freshly trained
LightGBM models ask for.

Each weekly retrain rewrites lightgbm_model_<kind>_feature_importances_splits_and_gain.csv,
and that CSV is the contract futures_trader.py reads at prediction time: es_features.py
returns one column per row of it, in that order. Nothing enforces that es_features.py
can actually compute those columns. When it can't, compute_es_features() fills the
column with NaN, prints a warning, and LightGBM quietly routes it through its
missing-value branch — so the trader keeps predicting, on degraded input, with
no failure anywhere. This script turns that silent degradation into a failed step.

Two ways a feature can be unusable live, both checked here:

  1. Not implemented — es_features.py has no computation for that name at all
     (e.g. BB_50_MA and EMA_10 were selected by the model but never ported).
  2. Not available — the name is implemented, but its source data does not exist
     in the live feed, so it can only ever be NaN. The training set carries
     hundreds of per-stock columns (TSLA_*, NVDA_*, ...) that the live ES+VXM
     feed has no source for; if a retrain ever selects one, no amount of feature
     code makes it computable live.

Exit codes:
    0  every required feature is computable from a live-shaped bar window
    1  at least one required feature is missing or unavoidably NaN
    2  could not run the check (missing model, unreadable CSV, import failure)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Bars of synthetic history to probe with. The longest warmup among the shared
# indicators is 50 (BB_50 / Volatility_50 / rolling-50), so this leaves ample
# room for the final row to be fully populated.
PROBE_BARS = 150
MODEL_KINDS = ("highest", "lowest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basic-model-dir", type=Path, required=True,
                   help="directory holding es_features.py and the trained model files")
    return p.parse_args()


def fail(message: str, code: int = 2) -> None:
    """Print in the format run_weekly_update.sh scans for, then exit."""
    print(f"Error: {message}")
    sys.exit(code)


def build_probe_bars(n: int = PROBE_BARS) -> pd.DataFrame:
    """A live-shaped bar window: exactly the columns futures_trader.py supplies.

    Values are synthetic but well-formed (strictly positive prices, non-zero
    volume, monotonic 5-second timestamps) so that a NaN in the final row means
    the feature genuinely cannot be computed, not that the probe was degenerate.
    """
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    vxm_close = 20.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    return pd.DataFrame({
        "Open": close,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": rng.random(n) * 1000.0 + 10.0,
        "VXM_Open": vxm_close,
        "VXM_High": vxm_close + 0.2,
        "VXM_Low": vxm_close - 0.2,
        "VXM_Close": vxm_close,
        "VXM_Volume": rng.random(n) * 100.0 + 1.0,
        "date": pd.date_range("2026-01-05 09:30:00", periods=n, freq="5s"),
    })


def check_model(es_features, model_kind: str, probe: pd.DataFrame) -> list[str]:
    """Return human-readable problems for one model kind; empty list means clean."""
    required = es_features.get_feature_names(model_kind)
    computable = es_features._build_all_feature_arrays(probe)

    not_implemented = [name for name in required if name not in computable]

    # Assemble exactly as the live predictor does, then inspect the row the
    # trader actually predicts from (the most recent bar).
    matrix, names = es_features.compute_es_features(
        probe, model_kind=model_kind, print_time=False
    )
    last_row = matrix[-1]
    nan_live = [name for name, value in zip(names, last_row) if np.isnan(value)]
    # A name that is simply absent is already reported as not_implemented; the
    # interesting case here is a name that exists but can never hold a value.
    always_nan = [name for name in nan_live if name not in not_implemented]

    print(f"  {model_kind}: {len(required)} features required, "
          f"{len(required) - len(not_implemented) - len(always_nan)} usable live")

    problems: list[str] = []
    if not_implemented:
        problems.append(
            f"{model_kind}: {len(not_implemented)} feature(s) selected by the model but "
            f"not implemented in es_features.py: {sorted(not_implemented)}"
        )
    if always_nan:
        problems.append(
            f"{model_kind}: {len(always_nan)} feature(s) implemented but NaN on a "
            f"fully-populated live bar window (no live data source): {sorted(always_nan)}"
        )
    return problems


def main() -> None:
    args = parse_args()
    basic_model_dir = args.basic_model_dir.resolve()
    if not (basic_model_dir / "es_features.py").exists():
        fail(f"es_features.py not found in {basic_model_dir}")

    # es_features.py imports lightgbm_utils as a bare module name, so its own
    # directory has to be importable.
    sys.path.insert(0, str(basic_model_dir))
    try:
        import es_features
    except Exception as exc:
        fail(f"could not import es_features from {basic_model_dir}: {exc}")

    print("Verifying live feature coverage against the freshly trained models ...")
    probe = build_probe_bars()

    problems: list[str] = []
    for model_kind in MODEL_KINDS:
        try:
            problems.extend(check_model(es_features, model_kind, probe))
        except FileNotFoundError as exc:
            fail(f"{model_kind}: {exc}")
        except Exception as exc:
            fail(f"{model_kind}: feature check raised {type(exc).__name__}: {exc}")

    if problems:
        print()
        for problem in problems:
            print(f"Error: {problem}")
        print()
        print("The trader would predict with these features silently NaN-filled.")
        print("Port the missing calculations into es_features.py (matching the "
              "formulas in futures_price.py / lightgbm_utils.py), or exclude the "
              "feature from training, then rerun.")
        sys.exit(1)

    print("All required features are computable from a live bar window.")


if __name__ == "__main__":
    main()
