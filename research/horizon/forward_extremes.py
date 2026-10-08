"""Sub-model targets at any forward horizon.

Production trains the two sub-models on the next five bars (futures_price.py):

    highest: (max High over t+1..t+N - Close_t) / Close_t * 100
    lowest:  (min Low  over t+1..t+N - Close_t) / Close_t * 100

and scales them for training with sign(y) * asinh(|y| * 50) (lightgbm_utils.py).
This module produces the same quantity for any N, plus a third target:

    average: (mean Close over t+1..t+N - Close_t) / Close_t * 100

An extreme is the best moment in the window and an average is the typical one, so
the average is the smoother target: averaging cancels the tick-by-tick noise an
extreme collects. Expect it to score a higher R-squared for that reason alone.

One difference from production, which shifts rows without looking at time: a window
that crosses a gap in the 5-second series (the 17:00-18:00 halt, weekends, missing
bars) has no target here. At five bars that removes a handful of rows per session;
at sixty bars it would otherwise let the overnight gap masquerade as an intraday move.

usage: forward_extremes.py    # build the price cache
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

MODEL_INPUT = "/Users/abeadam/dev/model/Stock-Model/future_prices/gradient_value/model_input.csv"
PRICE_CACHE = "forward_extreme_prices.npz"
PRICE_COLUMNS = ("DateTime", "Close", "High", "Low")
BAR_SECONDS = 5
ASINH_SCALE = 50.0
PERCENT = 100.0


def build_price_cache() -> None:
    """Read the four price columns out of the multi-gigabyte model input, once."""
    prices = pd.read_csv(MODEL_INPUT, usecols=list(PRICE_COLUMNS))
    prices = prices.drop_duplicates(subset="DateTime", keep="first")
    seconds = pd.to_datetime(prices["DateTime"]).to_numpy().astype("datetime64[s]").astype(np.int64)
    order = np.argsort(seconds, kind="stable")
    np.savez(PRICE_CACHE, seconds=seconds[order],
             close=prices["Close"].to_numpy(np.float64)[order],
             high=prices["High"].to_numpy(np.float64)[order],
             low=prices["Low"].to_numpy(np.float64)[order])
    print(f"{len(order):,} bars cached to {PRICE_CACHE}", flush=True)


def load_prices() -> dict[str, np.ndarray]:
    if not os.path.exists(PRICE_CACHE):
        build_price_cache()
    cached = np.load(PRICE_CACHE)
    return {name: cached[name] for name in ("seconds", "close", "high", "low")}


def window_is_contiguous(seconds: np.ndarray, horizon: int) -> np.ndarray:
    """True where bars t+1..t+horizon all follow t at the regular bar spacing."""
    contiguous = np.zeros(len(seconds), bool)
    contiguous[:-horizon] = (seconds[horizon:] - seconds[:-horizon]) == horizon * BAR_SECONDS
    return contiguous


TARGET_KINDS = {"highest": ("high", "max"), "lowest": ("low", "min"), "average": ("close", "mean")}
RANGE = "range"


def forward_window(values: np.ndarray, horizon: int, summary: str) -> np.ndarray:
    """max, min or mean of bars t+1..t+horizon for every row."""
    return getattr(pd.Series(values).rolling(horizon), summary)().shift(-horizon).to_numpy()


def forward_target(prices: dict[str, np.ndarray], horizon: int, kind: str) -> np.ndarray:
    """Percent of price: the next `horizon` bars' high, low, mean close, or high-to-low
    range. NaN where the window is unknown or crosses a gap.

    The range is the part of an extreme that carries no direction, which is what makes
    it the control for reading the other three."""
    if kind == RANGE:
        reached = (forward_window(prices["high"], horizon, "max")
                   - forward_window(prices["low"], horizon, "min"))
        target = reached / prices["close"] * PERCENT
    else:
        source, summary = TARGET_KINDS[kind]
        reached = forward_window(prices[source], horizon, summary)
        target = (reached - prices["close"]) / prices["close"] * PERCENT
    target[~window_is_contiguous(prices["seconds"], horizon)] = np.nan
    return target


def scale_for_training(target: np.ndarray) -> np.ndarray:
    return np.sign(target) * np.arcsinh(np.abs(target) * ASINH_SCALE)


if __name__ == "__main__":
    build_price_cache()
