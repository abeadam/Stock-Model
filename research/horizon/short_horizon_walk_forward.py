"""Walk-forward evaluation of short fixed-horizon ES strategies on 5-second decisions.

The gradient model's features predict the next ~25 seconds (AUC ~0.55) but not
5-60 minutes (AUC ~0.50), so this tests trading that short horizon directly:
enter when the model is confident, exit after N bars, one position at a time.

Each fold trains only on earlier bars (the N bars before each boundary purged),
chooses its threshold on a validation tail with a minimum trade count, then
trades the fold with that threshold unchanged. Fold dates match the earlier
walk-forward; folds 1-4 develop, 5-6 stay untouched until the final check.

Cost is the user's figure: $2.50 per fill, $5 per round trip, no slippage.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/Users/abeadam/dev/model/Stock-Model/future_prices/gradient_value")
sys.path.insert(0, "/Users/abeadam/dev/model/Stock-Model/future_prices/basic_model")
from train_predictor import _BEST_PLUS_ASI, SESSION_FEATURES  # noqa: E402

from cross_asset_features import CROSS_ASSET_FEATURES, add_cross_asset_features  # noqa: E402
from flow_features import FLOW_FEATURES, add_flow_features  # noqa: E402

MODEL_INPUT = "/Users/abeadam/dev/model/Stock-Model/future_prices/gradient_value/model_input.csv"
FEATURES = list(_BEST_PLUS_ASI) + list(SESSION_FEATURES)
# The basic highest/lowest models are fit on every row, so their predictions are
# in-sample for the test folds too. This set drops them as a leak control.
PRED_FEATURES = ["pred_CloseToHigh", "pred_CloseToLow", "pred_high_lag1", "pred_low_lag1"]
FEATURE_SETS = {"current_44": FEATURES,
                "no_pred": [f for f in FEATURES if f not in PRED_FEATURES],
                "plus_flow": FEATURES + FLOW_FEATURES,
                "plus_cross": FEATURES + CROSS_ASSET_FEATURES}
BAR_SECONDS = 5
POINT_VALUE = 50.0
ROUND_TRIP_COST = 5.00
FOLD_STARTS = ["2025-12-23", "2026-02-09", "2026-03-25", "2026-05-10", "2026-06-24", "2026-08-06"]
DATA_END = "2026-09-22"
DEV_FOLDS = (1, 2, 3, 4)
HOLDOUT_FOLDS = (5, 6)
VALIDATION_FRACTION = 0.15
THRESHOLD_GRID = np.round(np.arange(0.50, 0.801, 0.01), 2)
MIN_VALIDATION_TRADES = 500
LGBM_PARAMS = dict(objective="binary", num_leaves=127, learning_rate=0.05, min_data_in_leaf=500,
                   feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                   n_estimators=2000, n_jobs=-1, verbose=-1, random_state=11)
# Smaller, slower trees for a noisy target: the default set early-stops after 9-57 trees.
SLOW_LGBM_PARAMS = {**LGBM_PARAMS, "num_leaves": 31, "learning_rate": 0.01, "min_data_in_leaf": 5000,
                    "feature_fraction": 0.6, "n_estimators": 5000}
PARAM_SETS = {"default": LGBM_PARAMS, "slow": SLOW_LGBM_PARAMS}


def load() -> pd.DataFrame:
    data = pd.read_csv(MODEL_INPUT, usecols=["DateTime", "Close", "High", "Low", "Volume", *FEATURES])
    data = data.drop_duplicates(subset="DateTime", keep="first")
    data["time"] = pd.to_datetime(data["DateTime"])                 # UTC
    data = data.sort_values("time").dropna(subset=FEATURES).reset_index(drop=True)
    return add_cross_asset_features(add_flow_features(data.drop(columns="DateTime")))


def forward_move(data: pd.DataFrame, bars: int, delay: int = 0) -> np.ndarray:
    """close[i+delay+bars] - close[i+delay]: the move over `bars` bars starting `delay`
    bars after row i. NaN unless every bar from i to the end is a contiguous 5-second bar."""
    close = data["Close"].to_numpy()
    seconds = data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    reach = delay + bars
    move = np.full(len(close), np.nan)
    move[:-reach] = close[reach:] - close[delay:len(close) - bars]
    contiguous = np.zeros(len(close), bool)
    contiguous[:-reach] = (seconds[reach:] - seconds[:-reach]) == reach * BAR_SECONDS
    move[~contiguous] = np.nan
    return move


def simulate(prob, move, threshold, bars) -> dict:
    """One position at a time: long if prob >= threshold, short if <= 1 - threshold,
    exit after `bars` bars. Rows without a valid exit are never entered."""
    direction = np.where(prob >= threshold, 1, np.where(prob <= 1 - threshold, -1, 0))
    candidates = np.flatnonzero((direction != 0) & ~np.isnan(move))
    taken = []
    next_free = -1
    for i in candidates:
        if i >= next_free:
            taken.append(i)
            next_free = i + bars
    taken = np.array(taken, dtype=int)
    if len(taken) == 0:
        return {"trades": 0, "net": 0.0, "gross": 0.0, "hit_rate": np.nan}
    gross = direction[taken] * move[taken] * POINT_VALUE
    net = gross - ROUND_TRIP_COST
    return {"trades": len(taken), "net": float(net.sum()), "gross": float(gross.sum()),
            "hit_rate": float(np.mean(np.sign(move[taken]) == direction[taken]))}


@dataclass
class FoldResult:
    fold: int
    bars: int
    features: str
    auc: float
    threshold: float
    trading: dict        # simulate() result on the test fold


@dataclass
class FoldSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def fold_split(times: pd.Series, fold: int, purge_bars: int, fold_starts=FOLD_STARTS,
               data_end=DATA_END) -> FoldSplit:
    """Row indices for one fold. Training labels end purge_bars before validation,
    and validation labels end purge_bars before the test fold."""
    start = pd.Timestamp(fold_starts[fold - 1])
    end = pd.Timestamp(fold_starts[fold] if fold < len(fold_starts) else data_end)
    purge = pd.Timedelta(seconds=purge_bars * BAR_SECONDS)
    history = np.flatnonzero((times < start - purge).to_numpy())
    validation_start = history[int(len(history) * (1 - VALIDATION_FRACTION))]
    train = history[times.iloc[history].to_numpy() < times.iloc[validation_start] - purge]
    return FoldSplit(train=train, validation=history[history >= validation_start],
                     test=np.flatnonzero(((times >= start) & (times < end)).to_numpy()))


def train_direction_model(X: np.ndarray, move: np.ndarray, split: FoldSplit,
                          params: dict = LGBM_PARAMS) -> lgb.LGBMClassifier:
    """Classifier for move > 0, fit on the training rows, early-stopped on validation."""
    usable = ~np.isnan(move) & (move != 0)
    fit_rows, val_rows = split.train[usable[split.train]], split.validation[usable[split.validation]]
    model = lgb.LGBMClassifier(**params)
    model.fit(X[fit_rows], (move[fit_rows] > 0).astype(int),
              eval_set=[(X[val_rows], (move[val_rows] > 0).astype(int))], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return model


def train_magnitude_model(X: np.ndarray, move: np.ndarray, split: FoldSplit,
                          params: dict = LGBM_PARAMS) -> lgb.LGBMRegressor:
    """Regressor for |move| in points, fit on the training rows, early-stopped on validation."""
    known = ~np.isnan(move)
    fit_rows, val_rows = split.train[known[split.train]], split.validation[known[split.validation]]
    regression_params = {**params, "objective": "regression"}
    model = lgb.LGBMRegressor(**regression_params)
    model.fit(X[fit_rows], np.abs(move[fit_rows]), eval_set=[(X[val_rows], np.abs(move[val_rows]))],
              eval_metric="l2", callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return model


def train_move_regressor(X: np.ndarray, move: np.ndarray, split: FoldSplit,
                         params: dict = LGBM_PARAMS) -> lgb.LGBMRegressor:
    """Huber regressor for the signed move in points, fit on training rows, early-stopped on validation."""
    known = ~np.isnan(move)
    fit_rows, val_rows = split.train[known[split.train]], split.validation[known[split.validation]]
    regression_params = {**params, "objective": "huber", "alpha": float(np.nanstd(move[fit_rows]))}
    model = lgb.LGBMRegressor(**regression_params)
    model.fit(X[fit_rows], move[fit_rows], eval_set=[(X[val_rows], move[val_rows])],
              eval_metric="l2", callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return model


def direction_auc(prob: np.ndarray, move: np.ndarray) -> float:
    known = ~np.isnan(move) & (move != 0)
    return float(roc_auc_score((move[known] > 0).astype(int), prob[known]))


def run_fold(data: pd.DataFrame, move: np.ndarray, fold: int, bars: int, feature_set: str) -> FoldResult:
    split = fold_split(data["time"], fold, purge_bars=bars)
    X = data[FEATURE_SETS[feature_set]].to_numpy(np.float32)
    model = train_direction_model(X, move, split)
    validation, test = split.validation, split.test
    val_prob = model.predict_proba(X[validation])[:, 1]
    test_prob = model.predict_proba(X[test])[:, 1]
    auc = direction_auc(test_prob, move[test])

    threshold = choose_threshold(val_prob, move[validation], bars)
    return FoldResult(fold, bars, feature_set, auc, threshold, simulate(test_prob, move[test], threshold, bars))


def choose_threshold(prob, move, bars) -> float:
    """The grid threshold with the best validation net P&L among those trading enough."""
    best_threshold, best_net = THRESHOLD_GRID[-1], -np.inf
    for threshold in THRESHOLD_GRID:
        result = simulate(prob, move, threshold, bars)
        if result["trades"] >= MIN_VALIDATION_TRADES and result["net"] > best_net:
            best_threshold, best_net = threshold, result["net"]
    return best_threshold


def summarize(results: list[FoldResult], label: str) -> None:
    rows = [r.trading for r in results]
    trades = sum(x["trades"] for x in rows)
    net = sum(x["net"] for x in rows)
    hits = sum(x["hit_rate"] * x["trades"] for x in rows if x["trades"])
    print(f"{label:<24} trades {trades:>7,}  hit {hits / max(trades, 1):6.1%}  "
          f"net ${net:>10,.0f}  per trade ${net / max(trades, 1):>6.2f}  "
          f"folds+ {sum(x['net'] > 0 for x in rows)}/{len(rows)}  "
          f"per fold {[round(x['net']) for x in rows]}  thresholds {[r.threshold for r in results]}  "
          f"AUC {np.mean([r.auc for r in results]):.4f}", flush=True)
