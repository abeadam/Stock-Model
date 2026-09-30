"""Does tick-level information predict the next few seconds better than the 5-second bars?

From the quote ticks: the resting bid and ask sizes, so top-of-book imbalance
  (bid_size - ask_size) / (bid_size + ask_size), which the 5-second bars cannot show.
From the trade ticks: each print's aggressor side by the quote rule (above the mid
  means a buyer lifted the offer, below means a seller hit the bid), so signed volume.

Decisions are taken on a 1-second grid, and only on seconds whose quote actually
updated: IBKR's tick history is thinned, so a carried-over mid would let trade flow
"predict" nothing more than the stale mid catching up. For each horizon the target is
the mid-price move over the next N seconds, measured both from the signal second and
from one second later (a delay that removes any catching-up inside the signal second). Predictability is measured as AUC from a single feature
and from a small model over all of them, against the baseline of the bar-style
features available today (recent mid returns and trade imbalance over 5-second windows).

usage: tick_signal_probe.py YYYY-MM-DD
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HORIZONS_SECONDS = (5, 10, 30)
IMBALANCE_WINDOWS_SECONDS = (1, 5, 30)
RETURN_WINDOWS_SECONDS = (5, 30)
TRAIN_FRACTION = 0.6
TICK = 0.25
LGBM_PARAMS = dict(objective="binary", num_leaves=15, max_depth=4, learning_rate=0.05, n_estimators=300,
                   min_child_samples=200, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                   reg_lambda=10.0, n_jobs=-1, verbose=-1, random_state=3)


def load(day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # No de-duplication: repeat prints at the same second, price and size are real trades.
    trades = pd.read_csv(f"es_trades_{day}.csv")
    quotes = pd.read_csv(f"es_quotes_{day}.csv")
    quotes = quotes[(quotes["bid"] > 0) & (quotes["ask"] >= quotes["bid"])]
    return trades.sort_values("time"), quotes.sort_values("time")


def second_grid(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """One row per second: the last quote, and the trade flow within that second."""
    quotes = quotes.assign(second=quotes["time"].astype(int), mid=(quotes["bid"] + quotes["ask"]) / 2)
    last_quote = quotes.groupby("second").last()
    trades = trades.assign(second=trades["time"].astype(int))
    quote_at_trade = pd.merge_asof(trades, quotes[["time", "bid", "ask", "mid"]], on="time", direction="backward")
    side = np.sign(quote_at_trade["price"] - quote_at_trade["mid"])          # quote rule: +1 buyer, -1 seller
    flow = quote_at_trade.assign(signed=side * quote_at_trade["size"], volume=quote_at_trade["size"])
    per_second = flow.groupby("second")[["signed", "volume"]].sum()
    grid = last_quote.join(per_second, how="left").fillna({"signed": 0.0, "volume": 0.0})
    grid = grid.reindex(range(grid.index.min(), grid.index.max() + 1))
    grid["quote_is_fresh"] = grid["mid"].notna()
    return grid.ffill().fillna({"signed": 0.0, "volume": 0.0})


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=grid.index)
    sizes = grid["bid_size"] + grid["ask_size"]
    features["book_imbalance"] = (grid["bid_size"] - grid["ask_size"]) / sizes.where(sizes > 0)
    for window in IMBALANCE_WINDOWS_SECONDS:
        volume = grid["volume"].rolling(window).sum()
        features[f"trade_imbalance_{window}s"] = grid["signed"].rolling(window).sum() / volume.where(volume > 0)
        features[f"book_imbalance_{window}s"] = features["book_imbalance"].rolling(window).mean()
    for window in RETURN_WINDOWS_SECONDS:
        features[f"mid_change_{window}s"] = (grid["mid"] - grid["mid"].shift(window)) / TICK
    features["spread_ticks"] = (grid["ask"] - grid["bid"]) / TICK
    return features


def evaluate(grid: pd.DataFrame, features: pd.DataFrame) -> None:
    bar_style = [c for c in features.columns if c.startswith("mid_change") or c == "trade_imbalance_5s"]
    print(f"{len(grid):,} seconds, {features.shape[1]} features "
          f"(bar-style: {', '.join(bar_style)})")
    for horizon, delay in [(h, d) for h in HORIZONS_SECONDS for d in (0, 1)]:
        move = grid["mid"].shift(-horizon - delay) - grid["mid"].shift(-delay)
        usable = move.notna() & (move != 0) & features.notna().all(axis=1) & grid["quote_is_fresh"]
        rows = np.flatnonzero(usable)
        split = int(len(rows) * TRAIN_FRACTION)
        train, test = rows[:split], rows[split:]
        outcome = (move.to_numpy()[test] > 0).astype(int)
        single = {name: roc_auc_score(outcome, features[name].to_numpy()[test]) for name in features.columns}
        best_single = max(single, key=lambda name: abs(single[name] - 0.5))
        scores = {}
        for label, columns in (("bar-style only", bar_style), ("all tick features", list(features.columns))):
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(features.iloc[train][columns], (move.to_numpy()[train] > 0).astype(int))
            scores[label] = roc_auc_score(outcome, model.predict_proba(features.iloc[test][columns])[:, 1])
        print(f"  {horizon:>2}s ahead, from +{delay}s: best single {best_single} AUC {single[best_single]:.4f}  |  "
              f"model bar-style {scores['bar-style only']:.4f}  model with tick features "
              f"{scores['all tick features']:.4f}  ({len(test):,} test seconds)")


def main() -> None:
    trades, quotes = load(sys.argv[1])
    print(f"{len(trades):,} trade ticks, {len(quotes):,} quote ticks")
    grid = second_grid(trades, quotes)
    print(f"{len(grid):,} seconds, {int(grid['quote_is_fresh'].sum()):,} with a fresh quote")
    evaluate(grid, add_features(grid.drop(columns="quote_is_fresh")).join(grid["quote_is_fresh"]).drop(
        columns="quote_is_fresh"))


if __name__ == "__main__":
    main()
