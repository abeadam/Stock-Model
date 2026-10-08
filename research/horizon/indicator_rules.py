"""What do MFI and Bollinger Bands actually predict on ES, horizon by horizon?

Reading a chart, a Bollinger touch with MFI at an extreme looks like a reliable turn.
This measures that reading instead of eyeballing it, using the very features the
production model is given (MFI_14, bb_upper_dist, bb_lower_dist on 20-bar, 2-sd bands),
so there is no argument about definitions.

Rules, both directions of the usual interpretation:
  bollinger_reversion  long below the lower band, short above the upper band
  bollinger_breakout   the same touches read as continuation instead
  mfi_reversion        long under MFI 20, short over MFI 80
  mfi_breakout         the same readings as continuation
  both_reversion       a band touch and an MFI extreme agreeing

Entry fills one bar after the signal, never in a position twice at once, exit after the
horizon, $5 a round trip. Prices are trade closes, which flatters every rule by roughly
half the spread; no rule here comes close enough for that to matter.

usage: indicator_rules.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from forward_extremes import MODEL_INPUT, window_is_contiguous

CACHE = "indicator_rule_inputs.npz"
COLUMNS = ("DateTime", "Close", "MFI_14", "bb_upper_dist", "bb_lower_dist")
HORIZON_BARS = (5, 12, 60, 360, 720, 4320)          # 25 s, 1 min, 5 min, 30 min, 1 h, 6 h
ENTRY_DELAY_BARS = 1
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
MFI_OVERSOLD, MFI_OVERBOUGHT = 20.0, 80.0
RECENT_WINDOW_START = "2026-03-26"                  # the window every data source covers
LONG, SHORT = 1, -1


def load_inputs() -> dict[str, np.ndarray]:
    if not os.path.exists(CACHE):
        frame = pd.read_csv(MODEL_INPUT, usecols=list(COLUMNS)).drop_duplicates("DateTime", keep="first")
        seconds = pd.to_datetime(frame["DateTime"]).to_numpy().astype("datetime64[s]").astype(np.int64)
        order = np.argsort(seconds, kind="stable")
        np.savez(CACHE, seconds=seconds[order],
                 **{name: frame[name].to_numpy(np.float64)[order] for name in COLUMNS[1:]})
    cached = np.load(CACHE)
    return {name: cached[name] for name in cached.files}


def rule_signals(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A direction per bar for each rule: +1 long, -1 short, 0 no opinion."""
    below_band = inputs["bb_lower_dist"] < 0
    above_band = inputs["bb_upper_dist"] > 0
    oversold = inputs["MFI_14"] < MFI_OVERSOLD
    overbought = inputs["MFI_14"] > MFI_OVERBOUGHT
    bollinger = np.where(below_band, LONG, np.where(above_band, SHORT, 0))
    mfi = np.where(oversold, LONG, np.where(overbought, SHORT, 0))
    both = np.where(below_band & oversold, LONG, np.where(above_band & overbought, SHORT, 0))
    return {"bollinger_reversion": bollinger, "bollinger_breakout": -bollinger,
            "mfi_reversion": mfi, "mfi_breakout": -mfi,
            "both_reversion": both, "both_breakout": -both}


def trade_rule(direction: np.ndarray, close: np.ndarray, seconds: np.ndarray, horizon: int,
               eligible: np.ndarray) -> dict:
    """One position at a time: enter a bar after the signal, exit `horizon` bars later."""
    reach = ENTRY_DELAY_BARS + horizon
    tradeable = eligible & window_is_contiguous(seconds, reach) & (direction != 0)
    points, next_free_bar = [], -1
    for bar in np.flatnonzero(tradeable):
        if bar < next_free_bar:
            continue
        entry, exit_bar = bar + ENTRY_DELAY_BARS, bar + reach
        points.append(direction[bar] * (close[exit_bar] - close[entry]))
        next_free_bar = exit_bar
    points = np.array(points)
    if len(points) == 0:
        return {"trades": 0}
    net = points * POINT_VALUE - ROUND_TRIP_COST
    return {"trades": len(points), "hit_rate": float((points > 0).mean()),
            "mean_points": float(points.mean()), "per_trade": float(net.mean()),
            "t_stat": float(points.mean() / (points.std(ddof=1) / np.sqrt(len(points))))}


def report(title: str, inputs: dict[str, np.ndarray], eligible: np.ndarray) -> None:
    signals = rule_signals(inputs)
    print(f"\n=== {title}: {eligible.sum():,} bars ===")
    print(f"{'rule':<21}{'horizon':>9}{'trades':>9}{'hit rate':>10}{'mean pt':>9}{'t':>7}{'$/trade':>10}")
    for name, direction in signals.items():
        for horizon in HORIZON_BARS:
            result = trade_rule(direction, inputs["Close"], inputs["seconds"], horizon, eligible)
            if result["trades"] < 2:
                continue
            print(f"{name:<21}{horizon:>7} bars{result['trades']:>9,}{result['hit_rate']:>9.1%}"
                  f"{result['mean_points']:>9.3f}{result['t_stat']:>7.1f}{result['per_trade']:>+10.2f}")


if __name__ == "__main__":
    inputs = load_inputs()
    known = np.isfinite(inputs["MFI_14"]) & np.isfinite(inputs["bb_lower_dist"])
    recent_start = pd.Timestamp(RECENT_WINDOW_START).value // 10**9
    report("full history", inputs, known)
    report("clean window only", inputs, known & (inputs["seconds"] >= recent_start))
