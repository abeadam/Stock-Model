"""The tick model tested on every session it can be tested on, not just the last one.

`tick_model.py` trains on the earlier sessions and tests on the final one. With seven
sessions that is a single test day, and a single day is a sample rather than a result --
the 30-second book-imbalance finding from the one-hour probe already failed to replicate
once a full session was measured.

So each session in turn becomes the test day, trained on everything before it and with its
threshold chosen on the session immediately preceding. That yields several independent
test days, a spread to read alongside the mean, and an answer to whether the edge is
stable or lives in one session.

Two comparisons carry the weight:

  single feature   the one-second trade imbalance on its own. The earlier probe found the
                   fitted models scored *below* it, which would mean the model is adding
                   nothing over one column
  bar-style        what a five-second bar model can already see -- recent mid moves and
                   unsigned flow -- so the tick-specific columns are judged by what they
                   add rather than in isolation

Fills are reported at the mid and crossing the spread. The mid is not a market order: it
is a resting order that filled, and resting was measured yesterday to buy adverse
selection worth more than the spread it saves. Read the crossing column as the taker's
number and the mid as an upper bound.

usage: tick_walk_forward.py YYYY-MM-DD [YYYY-MM-DD ...]
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from tick_model import (LGBM_PARAMS, MIN_VALIDATION_TRADES, THRESHOLD_QUANTILES, build, trade)
from tick_probe import bar_style_columns, forward_move

HORIZONS_SECONDS = (5, 10, 30)
DELAY_SECONDS = 1
SINGLE_FEATURE = "trade_imbalance_1s"
MIN_TRAIN_DAYS = 2


def trade_weighted(outcomes: list[tuple[int, float]]) -> float:
    """Dollars per trade across days, weighted by trades.

    A plain mean lets a two-trade day count as much as a 750-trade one, which has already
    produced a wrong answer in this project once."""
    usable = [(count, value) for count, value in outcomes if count and np.isfinite(value)]
    total = sum(count for count, _ in usable)
    return sum(count * value for count, value in usable) / total if total else float("nan")


def fitted_score(features, labels, train_rows, all_rows) -> np.ndarray:
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(features.iloc[train_rows], labels[train_rows])
    return model.predict_proba(features.iloc[all_rows])[:, 1] - 0.5


def choose_threshold(score, grid, usable, rows, horizon, mode):
    best, best_net = None, -np.inf
    values = np.abs(score[rows][usable[rows]])
    if len(values) == 0:
        return None
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(np.quantile(values, quantile))
        result = trade(score[rows], grid.iloc[rows], usable[rows], threshold, horizon, mode)
        if result["trades"] >= MIN_VALIDATION_TRADES and result["net"] > best_net:
            best, best_net = threshold, result["net"]
    return best


def evaluate(days: list[str]) -> None:
    grid, features = build(days)
    day_column = grid["day"].to_numpy()
    bar_columns = bar_style_columns(features)
    print(f"{len(grid):,} seconds over {len(days)} sessions; "
          f"{len(features.columns)} features ({len(bar_columns)} of them bar-style)\n", flush=True)

    for horizon in HORIZONS_SECONDS:
        move = forward_move(grid, horizon, DELAY_SECONDS).to_numpy()
        labels = (move > 0).astype(int)
        usable = (np.isfinite(move) & (move != 0)
                  & features.notna().all(axis=1).to_numpy() & grid["quote_is_fresh"].to_numpy())
        print(f"{horizon}s ahead" + " " * 6 + f"{'model':>8}{'1s imbalance':>14}{'bar-style':>11}"
              f"{'trades':>8}{'mid $/trade':>13}{'cross $/trade':>15}")
        model_aucs, mid_trades, cross_trades = [], [], []
        for index in range(MIN_TRAIN_DAYS + 1, len(days)):
            test_day, validation_day = days[index], days[index - 1]
            train = np.flatnonzero(usable & (day_column != test_day) & (day_column != validation_day))
            test_rows = np.flatnonzero(day_column == test_day)
            validation_rows = np.flatnonzero(day_column == validation_day)
            test = test_rows[usable[test_rows]]
            if len(train) == 0 or len(test) == 0:
                continue
            everything = np.arange(len(grid))
            score = fitted_score(features, labels, train, everything)
            bar_score = fitted_score(features[bar_columns], labels, train, everything)
            model_auc = roc_auc_score(labels[test], score[test])
            single_auc = roc_auc_score(labels[test], features[SINGLE_FEATURE].to_numpy()[test])
            bar_auc = roc_auc_score(labels[test], bar_score[test])
            outcomes = {}
            for mode in ("mid", "cross"):
                threshold = choose_threshold(score, grid, usable, validation_rows, horizon, mode)
                outcomes[mode] = (trade(score[test_rows], grid.iloc[test_rows], usable[test_rows],
                                        threshold, horizon, mode)
                                  if threshold is not None else {"trades": 0, "per_trade": np.nan})
            model_aucs.append(model_auc)
            mid_trades.append((outcomes["mid"]["trades"], outcomes["mid"]["per_trade"]))
            cross_trades.append((outcomes["cross"]["trades"], outcomes["cross"]["per_trade"]))
            print(f"  test {test_day}{model_auc:>8.4f}{single_auc:>14.4f}{bar_auc:>11.4f}"
                  f"{outcomes['mid']['trades']:>8,}{outcomes['mid']['per_trade']:>13.2f}"
                  f"{outcomes['cross']['per_trade']:>15.2f}", flush=True)
        if model_aucs:
            print(f"  {'mean':<15}{np.mean(model_aucs):>8.4f}"
                  f"{'':>14}{'':>11}{sum(n for n, _ in mid_trades):>8,}"
                  f"{trade_weighted(mid_trades):>13.2f}{trade_weighted(cross_trades):>15.2f}")
            print(f"  {'spread':<15}{np.std(model_aucs, ddof=1):>8.4f}   "
                  f"(worst {min(model_aucs):.4f}, best {max(model_aucs):.4f})\n", flush=True)


if __name__ == "__main__":
    evaluate(sorted(sys.argv[1:]))
