"""Every TA-Lib indicator that can be applied to an arbitrary feature column.

TA-Lib ships 201 functions. Three facts decide which of them can be swept over the 163
columns, and they cut the search down by more than they look:

  125 need OHLCV. Only ES has open, high, low and volume; the other 162 columns are
      single series, so all 61 candlestick patterns and every volatility and volume
      indicator apply to ES alone and are handled separately.

  22 are pointwise. SQRT, LN, EXP, ATAN, CEIL, SIN and the rest map each value
      independently and monotonically. **Gradient-boosted trees split on rank order**, so
      a monotone transform of one column produces byte-identical splits -- these cannot
      change a tree model and are excluded on that proof, not on taste. LINEARREG_ANGLE is
      arctan of LINEARREG_SLOPE and goes with them.

  54 remain: windowed single-series functions that genuinely aggregate over time.

At four window settings across 163 columns that is 35,208 candidate columns, which is 80 GB
held as float32 and cannot be materialised. They are therefore streamed: each is computed,
scored, and discarded, with only the running best kept.

**Scoring is done on the earliest fold's TRAINING rows alone.** That block precedes both
test months, so nothing selected here has seen either. Even so, picking the best of 35,208
correlations inflates them badly by multiple comparison -- the screen proposes, and only
the paired ablation on held-out months decides.
"""
from __future__ import annotations

import numpy as np
import talib
from talib import abstract

WINDOWS = (6, 12, 30, 60)
POINTWISE_GROUP = "Math Transform"
MONOTONE_OF_ANOTHER = {"LINEARREG_ANGLE"}
MIN_FINITE_ROWS = 10_000
MIN_SPREAD = 1e-9


def windowed_single_series_functions() -> list[str]:
    """Functions taking one series and a window, excluding provable no-ops for trees."""
    excluded = set(talib.get_function_groups()[POINTWISE_GROUP]) | MONOTONE_OF_ANOTHER
    chosen = []
    for name in talib.get_functions():
        if name in excluded:
            continue
        info = abstract.Function(name).info
        inputs = []
        for value in info["input_names"].values():
            inputs.extend(value if isinstance(value, list) else [value])
        if inputs == ["close"] and info["parameters"]:
            chosen.append(name)
    return chosen


def parameter_settings(name: str) -> list[dict]:
    """One setting per window where the function has a window, else its defaults once."""
    defaults = dict(abstract.Function(name).parameters)
    if "timeperiod" not in defaults:
        return [defaults]
    return [{**defaults, "timeperiod": window} for window in WINDOWS]


def evaluate(name: str, series: np.ndarray, params: dict) -> list[np.ndarray]:
    """Every output of one indicator, as a list of columns; empty if it cannot be computed."""
    try:
        result = abstract.Function(name)({"close": series}, **params)
    except Exception:
        return []
    return [np.asarray(part, dtype=float) for part in result] if isinstance(result, list) \
        else [np.asarray(result, dtype=float)]


def correlation(values: np.ndarray, target: np.ndarray) -> float:
    """|Pearson| over rows where both are finite; 0 where there is too little to judge."""
    usable = np.isfinite(values) & np.isfinite(target)
    if usable.sum() < MIN_FINITE_ROWS:
        return 0.0
    left, right = values[usable], target[usable]
    if left.std() < MIN_SPREAD or right.std() < MIN_SPREAD:
        return 0.0
    return float(abs(np.corrcoef(left, right)[0, 1]))


def candidate_name(column: str, function: str, params: dict, output: int, outputs: int) -> str:
    window = params.get("timeperiod")
    suffix = f"_{window}" if window is not None else ""
    part = f"_{output}" if outputs > 1 else ""
    return f"ta_{column}_{function.lower()}{suffix}{part}"
