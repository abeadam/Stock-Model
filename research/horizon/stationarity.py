"""Two questions about whether the data is fit to learn from, which are not the same question.

**Is each feature stationary?** The Augmented Dickey-Fuller test regresses the change in a
series on its own level and a few of its own lagged changes,

    dx_t = a + rho * x_{t-1} + sum_i c_i * dx_{t-i} + e_t,

and asks whether rho is meaningfully below zero. If it is, the series is pulled back
towards a level and is stationary; if rho is indistinguishable from zero the series
wanders without one. The statistic is rho / se(rho), read against Dickey-Fuller critical
values rather than a normal table, because under the null the regressor is a random walk
and the usual asymptotics do not hold.

Two cautions matter more here than the test itself. At hundreds of thousands of rows the
test rejects a unit root for almost any series with the faintest pull, so "stationary"
becomes near-automatic and carries little information. And stationarity of a *feature*
says nothing about stability of the *relationship* between that feature and the target,
which is what a model actually depends on.

**Has the input distribution moved between the months we train on and the months we test
on?** That is the question a failing walk-forward points at, and it has a direct test:
try to tell a June row from a September row using the features alone. If a model can, the
thresholds a tree learned in June describe a distribution that no longer exists in
September. This is the diagnostic worth having, so it is here alongside the ADF.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LAGS = 5
NO_PULL = 0.0
# MacKinnon critical values for the constant-only ADF regression, large samples.
CRITICAL_VALUES = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def augmented_dickey_fuller(series: np.ndarray, lags: int = DEFAULT_LAGS) -> float:
    """The ADF t-statistic for a unit root. More negative means more clearly stationary."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) <= lags + 3 or np.std(values) == 0:
        return np.nan
    changes = np.diff(values)
    rows = len(changes) - lags
    if rows <= lags + 2:
        return np.nan
    target = changes[lags:]
    design = [np.ones(rows), values[lags:-1]]
    for lag in range(1, lags + 1):
        design.append(changes[lags - lag:-lag])
    design = np.column_stack(design)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    degrees_of_freedom = rows - design.shape[1]
    variance = residuals @ residuals / degrees_of_freedom
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(covariance[1, 1])
    return float(coefficients[1] / standard_error) if standard_error > 0 else np.nan


def has_unit_root(statistic: float, level: str = "5%") -> bool:
    """True when the test cannot reject a unit root: the series wanders."""
    return not (statistic < CRITICAL_VALUES[level])
