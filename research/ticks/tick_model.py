"""A short-term ES model built on tick data, evaluated honestly.

Features, all known at the decision second:
  tick      aggressor-side trade imbalance, top-of-book size imbalance, spread,
            recent mid changes (from daily_data/ticks)
  cross     returns of SPY, VXX, UVXY, TLT and USO over the last 1 and 5 minutes,
            from their 5-second bars (the VIX ETFs trade, unlike the VIX index)

Target: the mid move over the next N seconds, starting one second after the signal,
so the decision second's own catching-up is excluded. Decisions only on seconds whose
quote updated.

Trained on the earlier sessions, tested on the last one. Trading: enter when the score
passes a threshold chosen on the final training session, hold N seconds, one position
at a time. Fills at the mid and crossing the spread, $5 a round trip.

usage: tick_model.py YYYY-MM-DD [YYYY-MM-DD ...]
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/private/tmp/claude-501/-Users-abeadam-dev-spx-trader/a7828893-f333-47e2-8dc9-61906b6c4599/scratchpad/horizon")
from cross_asset_features import load_symbol_closes  # noqa: E402

from tick_probe import add_features, forward_move, load_day  # noqa: E402

CROSS_SYMBOLS = ("SPY", "VXX", "UVXY", "TLT", "USO")
CROSS_WINDOWS_SECONDS = (60, 300)
HORIZONS_SECONDS = (5, 10, 30)
DELAY_SECONDS = 1
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
TICK = 0.25
THRESHOLD_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)
MIN_VALIDATION_TRADES = 50
LGBM_PARAMS = dict(objective="binary", num_leaves=31, max_depth=5, learning_rate=0.03, n_estimators=400,
                   min_child_samples=500, colsample_bytree=0.7, subsample=0.7, subsample_freq=1,
                   reg_lambda=20.0, n_jobs=-1, verbose=-1, random_state=17)


def price_at(closes: pd.Series, epochs: np.ndarray) -> np.ndarray:
    """The symbol's last 5-second bar close at or before each epoch."""
    bar_starts, values = closes.index.to_numpy(), closes.to_numpy()
    position = np.searchsorted(bar_starts, epochs, side="right") - 1
    prices = np.full(len(epochs), np.nan)
    known = position >= 0
    prices[known] = values[position[known]]
    return prices


def cross_features(seconds: np.ndarray) -> pd.DataFrame:
    """Percent returns of each traded proxy over the last minute and five minutes."""
    features = pd.DataFrame(index=range(len(seconds)))
    for symbol in CROSS_SYMBOLS:
        closes = load_symbol_closes(symbol)
        now = price_at(closes, seconds)
        for window in CROSS_WINDOWS_SECONDS:
            features[f"{symbol.lower()}_ret_{window}s"] = (now / price_at(closes, seconds - window) - 1) * 100
    return features


def build(days: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grids, features = [], []
    for day in days:
        grid = load_day(day)
        tick_part = add_features(grid).reset_index(drop=True)
        cross_part = cross_features(grid.index.to_numpy())
        grids.append(grid.reset_index(drop=True))
        features.append(pd.concat([tick_part, cross_part], axis=1))
    return pd.concat(grids, ignore_index=True), pd.concat(features, ignore_index=True)


def trade(score: np.ndarray, grid: pd.DataFrame, eligible: np.ndarray, threshold: float, horizon: int,
          mode: str) -> dict:
    """Enter when |score| passes the threshold on an eligible second, hold `horizon`
    seconds, one position at a time. Every array covers one session's consecutive seconds."""
    mid, bid, ask = grid["mid"].to_numpy(), grid["Bid"].to_numpy(), grid["Ask"].to_numpy()
    buy, sell = (mid, mid) if mode == "mid" else (ask, bid)
    direction = np.where(score >= threshold, 1, np.where(score <= -threshold, -1, 0)) * eligible
    seconds = grid.index.to_numpy()
    pnl, next_free = [], -np.inf
    for row in np.flatnonzero(direction != 0):
        exit_row = row + DELAY_SECONDS + horizon
        if seconds[row] < next_free or exit_row >= len(mid):
            continue
        entry_row = row + DELAY_SECONDS
        if seconds[exit_row] - seconds[entry_row] != horizon:          # never across a session break
            continue
        gain = (sell[exit_row] - buy[entry_row]) if direction[row] > 0 else (sell[entry_row] - buy[exit_row])
        pnl.append(gain * POINT_VALUE - ROUND_TRIP_COST)
        next_free = seconds[exit_row]
    pnl = np.array(pnl)
    if len(pnl) == 0:
        return {"trades": 0, "net": 0.0, "per_trade": np.nan, "win": np.nan}
    return {"trades": len(pnl), "net": float(pnl.sum()), "per_trade": float(pnl.mean()),
            "win": float((pnl > 0).mean())}


def evaluate(days: list[str]) -> None:
    grid, features = build(days)
    day_column = grid["day"].to_numpy()
    test_day, validation_day = days[-1], days[-2] if len(days) > 2 else days[-1]
    print(f"{len(grid):,} seconds over {len(days)} sessions; train "
          f"{', '.join(days[:-1])}; validate {validation_day}; test {test_day}")
    for horizon in HORIZONS_SECONDS:
        move = forward_move(grid, horizon, DELAY_SECONDS)
        usable = (move.notna() & (move != 0) & features.notna().all(axis=1) & grid["quote_is_fresh"]).to_numpy()
        train = np.flatnonzero(usable & (day_column != test_day) & (day_column != validation_day))
        validation_rows = np.flatnonzero(day_column == validation_day)     # whole session, consecutive
        test_rows = np.flatnonzero(day_column == test_day)
        test = test_rows[usable[test_rows]]
        if len(train) == 0 or len(test) == 0:
            print(f"  {horizon}s: not enough sessions yet")
            continue
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(features.iloc[train], (move.to_numpy()[train] > 0).astype(int))
        score = model.predict_proba(features)[:, 1] - 0.5
        auc = roc_auc_score((move.to_numpy()[test] > 0).astype(int), score[test])
        for mode in ("mid", "cross"):
            best, best_net = None, -np.inf
            for quantile in THRESHOLD_QUANTILES:
                threshold = float(np.quantile(np.abs(score[validation_rows][usable[validation_rows]]), quantile))
                result = trade(score[validation_rows], grid.iloc[validation_rows], usable[validation_rows],
                               threshold, horizon, mode)
                if result["trades"] >= MIN_VALIDATION_TRADES and result["net"] > best_net:
                    best, best_net = threshold, result["net"]
            outcome = (trade(score[test_rows], grid.iloc[test_rows], usable[test_rows], best, horizon, mode)
                       if best else {"trades": 0})
            print(f"  {horizon:>2}s {mode:<5} AUC {auc:.4f}  trades {outcome['trades']:>4}  "
                  + (f"win {outcome['win']:5.1%}  net ${outcome['net']:>8,.0f}  "
                     f"per trade ${outcome['per_trade']:+6.2f}" if outcome["trades"] else "no eligible threshold"))


if __name__ == "__main__":
    evaluate(sorted(sys.argv[1:]))
