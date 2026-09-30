"""Does tick-level information predict the next few seconds better than 5-second bars?

Reads the collected tick files (daily_data/ticks/<date>_ES_TRADES.txt and _BIDASK.txt).
From the quotes: resting bid and ask sizes, so top-of-book imbalance. From the trades:
each print's aggressor side by the quote rule, so signed volume. Neither is visible in
the bars the models use today.

Decisions sit on a 1-second grid and only on seconds whose quote actually updated:
IBKR's tick history is thinned, and a carried-over mid would let trade flow "predict"
nothing more than the stale mid catching up. Each horizon is scored twice, from the
signal second and from one second later, the delay removing any catching-up inside
the signal second.

With several days it trains on the earlier ones and tests on the last; with one day it
splits that day in time order.

usage: tick_probe.py YYYY-MM-DD [YYYY-MM-DD ...]
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TICKS_DIR = "/Users/abeadam/dev/model/Stock-Model/daily_data/ticks"
HORIZONS_SECONDS = (5, 10, 30)
DELAYS_SECONDS = (0, 1)
IMBALANCE_WINDOWS_SECONDS = (1, 5, 30)
RETURN_WINDOWS_SECONDS = (5, 30)
SINGLE_DAY_TRAIN_FRACTION = 0.6
TICK = 0.25
LGBM_PARAMS = dict(objective="binary", num_leaves=15, max_depth=4, learning_rate=0.05, n_estimators=300,
                   min_child_samples=200, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                   reg_lambda=10.0, n_jobs=-1, verbose=-1, random_state=3)


def load_day(day: str) -> pd.DataFrame:
    """One second-grid frame for a session: last quote, trade flow, and whether the quote moved."""
    trades = pd.read_csv(f"{TICKS_DIR}/{day}_ES_TRADES.txt")
    quotes = pd.read_csv(f"{TICKS_DIR}/{day}_ES_BIDASK.txt")
    quotes = quotes[(quotes["Bid"] > 0) & (quotes["Ask"] >= quotes["Bid"])].sort_values("Date", kind="stable")
    quotes = quotes.assign(second=quotes["Date"].astype(int), mid=(quotes["Bid"] + quotes["Ask"]) / 2)
    trades = trades.sort_values("Date", kind="stable").assign(second=lambda frame: frame["Date"].astype(int))

    at_trade = pd.merge_asof(trades, quotes[["Date", "mid"]], on="Date", direction="backward")
    side = np.sign(at_trade["Price"] - at_trade["mid"])               # quote rule: +1 buyer, -1 seller
    flow = at_trade.assign(signed=side * at_trade["Size"], volume=at_trade["Size"])
    per_second = flow.groupby("second")[["signed", "volume"]].sum()

    grid = quotes.groupby("second").last().join(per_second, how="left")
    grid = grid.reindex(range(grid.index.min(), grid.index.max() + 1))
    grid["quote_is_fresh"] = grid["mid"].notna()
    grid = grid.ffill().fillna({"signed": 0.0, "volume": 0.0})
    return grid.assign(day=day)


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=grid.index)
    resting = grid["BidSize"] + grid["AskSize"]
    features["book_imbalance"] = (grid["BidSize"] - grid["AskSize"]) / resting.where(resting > 0)
    for window in IMBALANCE_WINDOWS_SECONDS:
        volume = grid["volume"].rolling(window).sum()
        features[f"trade_imbalance_{window}s"] = grid["signed"].rolling(window).sum() / volume.where(volume > 0)
        features[f"book_imbalance_{window}s"] = features["book_imbalance"].rolling(window).mean()
    for window in RETURN_WINDOWS_SECONDS:
        features[f"mid_change_{window}s"] = (grid["mid"] - grid["mid"].shift(window)) / TICK
    features["spread_ticks"] = (grid["Ask"] - grid["Bid"]) / TICK
    return features


def bar_style_columns(features: pd.DataFrame) -> list[str]:
    """What a 5-second bar model can already see: recent price moves and unsigned flow."""
    return [name for name in features.columns if name.startswith("mid_change") or name == "trade_imbalance_5s"]


def split_rows(grid: pd.DataFrame, usable: np.ndarray, days: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(usable)
    if len(days) == 1:
        cut = int(len(rows) * SINGLE_DAY_TRAIN_FRACTION)
        return rows[:cut], rows[cut:]
    last_day = grid["day"].to_numpy() == days[-1]
    return rows[~last_day[rows]], rows[last_day[rows]]


def forward_move(grid: pd.DataFrame, horizon: int, delay: int) -> pd.Series:
    """Mid move over `horizon` seconds starting `delay` seconds ahead, never across sessions."""
    by_day = grid.groupby("day")["mid"]
    return by_day.shift(-horizon - delay) - by_day.shift(-delay)


def evaluate(grid: pd.DataFrame, features: pd.DataFrame, days: list[str]) -> None:
    bar_style = bar_style_columns(features)
    print(f"{len(grid):,} seconds over {len(days)} session(s), "
          f"{int(grid['quote_is_fresh'].sum()):,} with a fresh quote")
    for horizon in HORIZONS_SECONDS:
        for delay in DELAYS_SECONDS:
            move = forward_move(grid, horizon, delay)
            usable = (move.notna() & (move != 0) & features.notna().all(axis=1)
                      & grid["quote_is_fresh"]).to_numpy()
            train, test = split_rows(grid, usable, days)
            outcome = (move.to_numpy()[test] > 0).astype(int)
            single = {name: roc_auc_score(outcome, features[name].to_numpy()[test]) for name in features.columns}
            best = max(single, key=lambda name: abs(single[name] - 0.5))
            scores = {}
            for label, columns in (("bars", bar_style), ("ticks", list(features.columns))):
                model = lgb.LGBMClassifier(**LGBM_PARAMS)
                model.fit(features.iloc[train][columns], (move.to_numpy()[train] > 0).astype(int))
                scores[label] = roc_auc_score(outcome, model.predict_proba(features.iloc[test][columns])[:, 1])
            print(f"  {horizon:>2}s ahead from +{delay}s: best single {best} {single[best]:.4f}  |  "
                  f"model bar-style {scores['bars']:.4f}  model with ticks {scores['ticks']:.4f}  "
                  f"({len(train):,} train / {len(test):,} test seconds)")


def main() -> None:
    days = sorted(sys.argv[1:])
    sessions = [load_day(day) for day in days]
    # Features per session: a rolling window must not reach into the day before.
    features = pd.concat([add_features(session) for session in sessions], ignore_index=True)
    evaluate(pd.concat(sessions, ignore_index=True), features, days)


if __name__ == "__main__":
    main()
