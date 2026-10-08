"""Walk-forward comparison of one global ES model against specialized models for
the overnight, opening, normal and closing segments of the day.

Only days that have every data source (ES trades, ES bid and ask, TLT, USO) are
used, for training and testing alike, and only bars with a standing quote. The
target is the mid-price move over H bars, so bid-ask bounce is not in the label.

Two steps:
  predict  trains every model on each fold and caches its validation and test scores,
           for one group of targets:
             mid      the mid-price move over 12 and 60 bars
             barrier  which comes first from each bar: a rise or a fall of the
                      take-profit distance (5-s changes or 5-min levels), within
                      30 minutes and before the period ends
  evaluate picks each model's entry threshold (and rule) on its validation rows and
           trades it, unchanged, on its test rows under an exit policy:
             signal_hold          exit when the signal fades (earlier study)
             take_profit_changes  exit at a gain of max(2 sd of 5-s mid changes over
                                  the last 5 min, 0.25) or at the period's end
             take_profit_levels   the same with 2 sd of the mid price over 5 min
             take_profit_*_stop   either one plus a stop loss at the same distance
           Segment models end each trade by the segment's end; the global model
           traded all day ("all_day") ends it by the 17:00 halt.
Two fill modes, both at $5 per round trip: at the mid, and crossing the spread.

Folds are calendar months of the cut window; June-July develop, August-September
are the holdout.

usage: segment_walk_forward.py predict dev|holdout mid|barrier
       segment_walk_forward.py evaluate dev|holdout EXIT_POLICY
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from cross_asset_features import CROSS_ASSET_FEATURES
from flow_features import FLOW_FEATURES
from macro_features import MACRO_FEATURES, add_macro_features
from quote_features import QUOTE_FEATURES, add_quote_columns
from rule_selection import choose_rule, choose_take_profit_threshold, run_rule, run_take_profit
from segments import EASTERN, SEGMENTS, days_with_all_data, eastern_times, segment_of, session_date
from short_horizon_walk_forward import (BAR_SECONDS, FEATURES, FoldSplit, direction_auc, fold_split,
                                        load as load_base, train_direction_model)
from barrier_targets import first_barrier_labels
from take_profit_policy import VOLATILITY_MEASURES, recent_volatility, take_profit_points

FEATURE_SETS = {"current_44": FEATURES,
                "all_new": FEATURES + FLOW_FEATURES + CROSS_ASSET_FEATURES + QUOTE_FEATURES + MACRO_FEATURES}
MID_TARGET_BARS = (12, 60)                     # 1 minute, 5 minutes
BARRIER_MAX_BARS = 360                         # 30 minutes
TARGET_GROUPS = ("mid", "barrier")
FOLD_BOUNDARIES_ET = ("2026-05-31 18:00", "2026-06-30 18:00", "2026-07-31 18:00", "2026-08-31 18:00")
DATA_END_ET = "2026-09-21 18:00"
FOLDS_BY_STAGE = {"dev": (1, 2), "holdout": (3, 4)}
ALL_DAY = "all_day"
# A rule must trade at least this often on validation to be eligible, whatever the segment's length.
MIN_TRADES_PER_VALIDATION_DAY = 3
FILL_MODES = ("mid", "cross")
EXIT_POLICIES = ("signal_hold", "take_profit_changes", "take_profit_levels",
                 "take_profit_changes_stop", "take_profit_levels_stop")
STOP_SUFFIX = "_stop"
NO_TRADES = {"trades": 0, "net": 0.0, "gross": 0.0, "hit_rate": np.nan, "target_rate": np.nan,
             "reversal_rate": np.nan, "mean_bars_held": np.nan}


def to_utc(eastern: str) -> pd.Timestamp:
    return pd.Timestamp(eastern, tz=EASTERN).tz_convert("UTC").tz_localize(None)


FOLD_STARTS_UTC = [to_utc(boundary) for boundary in FOLD_BOUNDARIES_ET]
DATA_END_UTC = to_utc(DATA_END_ET)


def predictions_path(stage: str, target_group: str) -> str:
    suffix = "" if target_group == "mid" else f"_{target_group}"
    return f"segment_predictions_{stage}{suffix}.pkl"


def load_cut() -> pd.DataFrame:
    """Every feature, cut to days with all data sources and to bars with a quote."""
    data = add_macro_features(add_quote_columns(load_base()))
    et = eastern_times(data["time"])
    days = session_date(et)
    keep_days = days_with_all_data(set(days))
    keep = days.isin(keep_days).to_numpy() & ~np.isnan(data["mid"].to_numpy())
    data = data.assign(segment=segment_of(et), session_day=days.to_numpy())
    return data[keep].reset_index(drop=True)


def bar_seconds(data: pd.DataFrame) -> np.ndarray:
    return data["time"].to_numpy().astype("datetime64[s]").astype(np.int64)


def mid_move(data: pd.DataFrame, bars: int) -> np.ndarray:
    """mid[i+bars] - mid[i], NaN unless the bars in between are contiguous."""
    mid = data["mid"].to_numpy()
    seconds = bar_seconds(data)
    move = np.full(len(mid), np.nan)
    move[:-bars] = mid[bars:] - mid[:-bars]
    contiguous = np.zeros(len(mid), bool)
    contiguous[:-bars] = (seconds[bars:] - seconds[:-bars]) == bars * BAR_SECONDS
    move[~contiguous] = np.nan
    return move


def targets(data: pd.DataFrame, target_group: str):
    """(name, label array whose sign is the class, purge bars) for each target in a group."""
    if target_group == "mid":
        return [(f"mid_{bars}", mid_move(data, bars), bars) for bars in MID_TARGET_BARS]
    mid, seconds = data["mid"].to_numpy(), bar_seconds(data)
    return [(f"barrier_{measure}",
             first_barrier_labels(mid, seconds, take_profit_points(recent_volatility(mid, seconds, measure)),
                                  BARRIER_MAX_BARS), BARRIER_MAX_BARS)
            for measure in VOLATILITY_MEASURES]


def fill_prices(data: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(buy price, sell price) per bar for a fill mode."""
    if mode == "mid":
        return data["mid"].to_numpy(), data["mid"].to_numpy()
    return data["ask"].to_numpy(), data["bid"].to_numpy()


def segment_split(split: FoldSplit, in_segment: np.ndarray) -> FoldSplit:
    return FoldSplit(train=split.train[in_segment[split.train]],
                     validation=split.validation[in_segment[split.validation]],
                     test=split.test[in_segment[split.test]])


def scored(classifier, X, part: FoldSplit, move, **labels) -> dict:
    validation_score = classifier.predict_proba(X[part.validation])[:, 1] - 0.5
    test_score = classifier.predict_proba(X[part.test])[:, 1] - 0.5
    return {**labels, "auc": direction_auc(test_score + 0.5, move[part.test]),
            "validation_rows": part.validation, "test_rows": part.test,
            "validation_score": validation_score, "test_score": test_score}


def predict_fold(data, X, move, fold: int, target: str, bars: int, feature_set: str) -> list[dict]:
    split = fold_split(data["time"], fold, purge_bars=bars, fold_starts=FOLD_STARTS_UTC, data_end=DATA_END_UTC)
    labels = {"fold": fold, "target": target, "bars": bars, "features": feature_set}
    global_model = train_direction_model(X, move, split)
    predictions = [scored(global_model, X, split, move, model="global", segment=ALL_DAY, **labels)]
    segment = data["segment"].to_numpy()
    for name in SEGMENTS:
        part = segment_split(split, segment == name)
        predictions.append(scored(global_model, X, part, move, model="global", segment=name, **labels))
        predictions.append(scored(train_direction_model(X, move, part), X, part, move,
                                  model="specialized", segment=name, **labels))
    return predictions


def predict(stage: str, target_group: str) -> None:
    started = time.time()
    data = load_cut()
    days = sorted(set(data["session_day"]))
    print(f"{len(data):,} bars on {len(days)} days with all data ({days[0]} .. {days[-1]}) "
          f"({time.time() - started:.0f}s)", flush=True)
    predictions = []
    for target, move, bars in targets(data, target_group):
        print(f"  {target}: {np.mean(~np.isnan(move)):.1%} of bars labelled ({time.time() - started:.0f}s)", flush=True)
        for feature_set, columns in FEATURE_SETS.items():
            X = data[columns].to_numpy(np.float32)
            for fold in FOLDS_BY_STAGE[stage]:
                predictions += predict_fold(data, X, move, fold, target, bars, feature_set)
                print(f"  {target} {feature_set} fold {fold} ({time.time() - started:.0f}s)", flush=True)
    with open(predictions_path(stage, target_group), "wb") as handle:
        pickle.dump(predictions, handle)


def trade(prediction: dict, prices: dict, exit_policy: str) -> dict:
    """Pick the rule on validation rows, trade it on test rows, per fill mode."""
    validation, test = prediction["validation_rows"], prediction["test_rows"]
    min_trades = MIN_TRADES_PER_VALIDATION_DAY * len(np.unique(prices["session_day"][validation]))
    results = {}
    for mode in FILL_MODES:
        buy, sell = prices[mode]
        if exit_policy == "signal_hold":
            rule = choose_rule(prediction["validation_score"], buy[validation], sell[validation],
                               prices["seconds"][validation], prediction["bars"], min_trades)
            results[mode] = NO_TRADES if rule is None else run_rule(
                prediction["test_score"], buy[test], sell[test], prices["seconds"][test], rule)
            continue
        target = prices[exit_policy.removesuffix(STOP_SUFFIX)]
        stop = target if exit_policy.endswith(STOP_SUFFIX) else None
        threshold = choose_take_profit_threshold(prediction["validation_score"], buy[validation],
                                                 sell[validation], target[validation],
                                                 prices["seconds"][validation], min_trades,
                                                 None if stop is None else stop[validation])
        results[mode] = NO_TRADES if threshold is None else run_take_profit(
            prediction["test_score"], buy[test], sell[test], target[test], prices["seconds"][test], threshold,
            None if stop is None else stop[test])
    return results


def evaluate(stage: str, exit_policy: str) -> pd.DataFrame:
    data = load_cut()
    seconds = bar_seconds(data)
    mid = data["mid"].to_numpy()
    prices = {"seconds": seconds, "session_day": data["session_day"].to_numpy(),
              **{mode: fill_prices(data, mode) for mode in FILL_MODES},
              "take_profit_changes": take_profit_points(recent_volatility(mid, seconds, "changes")),
              "take_profit_levels": take_profit_points(recent_volatility(mid, seconds, "levels"))}
    if exit_policy.startswith("take_profit"):
        target = prices[exit_policy.removesuffix(STOP_SUFFIX)]
        print(f"take-profit target (points): median {np.median(target):.2f}, "
              f"10th pct {np.percentile(target, 10):.2f}, 90th pct {np.percentile(target, 90):.2f}")
    predictions = []
    for target_group in TARGET_GROUPS:
        if os.path.exists(predictions_path(stage, target_group)):
            with open(predictions_path(stage, target_group), "rb") as handle:
                predictions += pickle.load(handle)
    rows = []
    for prediction in predictions:
        trading = trade(prediction, prices, exit_policy)
        rows.append({"target": prediction.get("target", f"mid_{prediction['bars']}")}
                    | {key: prediction[key] for key in ("fold", "features", "model", "segment", "auc")}
                    | {f"{mode}_{key}": trading[mode].get(key, np.nan) for mode in FILL_MODES
                       for key in ("trades", "net", "hit_rate", "target_rate", "reversal_rate", "stop_rate",
                                   "mean_bars_held")})
    results = pd.DataFrame(rows)
    results.to_pickle(f"segment_results_{stage}_{exit_policy}.pkl")
    summarize(results)
    return results


def count_positive(net: pd.Series) -> int:
    return int((net > 0).sum())


def summarize(results: pd.DataFrame) -> None:
    """Per horizon, feature set, segment and model: mean AUC, and per fill mode the
    trades, win rate, net P&L and number of folds that made money."""
    wins = {f"{mode}_wins": results[f"{mode}_hit_rate"].fillna(0) * results[f"{mode}_trades"] for mode in FILL_MODES}
    aggregations = {"auc": ("auc", "mean")}
    for mode in FILL_MODES:
        aggregations |= {f"{mode}_trades": (f"{mode}_trades", "sum"), f"{mode}_wins": (f"{mode}_wins", "sum"),
                         f"{mode}_net": (f"{mode}_net", "sum"), f"{mode}_folds_up": (f"{mode}_net", count_positive)}
    table = results.assign(**wins).groupby(["target", "features", "segment", "model"], sort=False).agg(**aggregations)
    for mode in FILL_MODES:
        table.insert(table.columns.get_loc(f"{mode}_wins"), f"{mode}_win",
                     table[f"{mode}_wins"] / table[f"{mode}_trades"].where(table[f"{mode}_trades"] > 0))
        table = table.drop(columns=f"{mode}_wins")
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(table.round({"auc": 4, "mid_win": 3, "cross_win": 3, "mid_net": 0, "cross_net": 0}).to_string())


if __name__ == "__main__":
    command, stage = sys.argv[1], sys.argv[2]
    if command == "predict":
        predict(stage, sys.argv[3])
    else:
        evaluate(stage, sys.argv[3])
